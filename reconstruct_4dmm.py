"""Case-driven inference for independently trained endo and epi 4DMM models."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import skimage.measure
import torch
import trimesh

from case_config import CaseConfig
from deep_sdf.dataset import get_sdf_samples_test
from project_config import RANDOM_SEED


SURFACES = ("endo", "epi")
MAX_GRID_BATCH = 2**17


def _load_json(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return str(device)


def load_decoder(surface: str, checkpoint_dir: Path, device: str):
    checkpoint_dir = Path(checkpoint_dir)
    specs_path = checkpoint_dir / "specs.json"
    checkpoint_path = checkpoint_dir / "ModelParameters" / "latest.pth"
    if not specs_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Incomplete checkpoint directory for {surface}: {checkpoint_dir}"
        )
    specs = _load_json(specs_path)
    arch = importlib.import_module("networks." + specs["NetworkArch"])
    decoder = arch.Decoder(**specs["NetworkSpecs"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"model_state_dict is absent from {checkpoint_path}")
    decoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    metadata = {
        "path": str(checkpoint_path.resolve()),
        "sha256": _sha256(checkpoint_path),
        "parameters": int(sum(parameter.numel() for parameter in decoder.parameters())),
    }
    return decoder, specs, metadata


def _load_case_frames(manifest_path: Path, mapping: list[dict], case_id: str, surface: str):
    manifest = _load_json(Path(manifest_path))
    sequences = manifest.get("test", {}).get("case", {})
    name = f"{case_id}-{surface}"
    if set(sequences) != {name}:
        raise ValueError(
            f"Manifest {manifest_path} must contain exactly test.case.{name}"
        )
    instances = sequences[name].get("instance_list", [])
    if len(instances) != len(mapping):
        raise ValueError(
            f"Manifest/mapping length mismatch: {len(instances)} vs {len(mapping)}"
        )
    frames = []
    for npz_path, item in zip(instances, mapping):
        path = Path(npz_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as archive:
            stored_t = float(np.asarray(archive["t"]).reshape(-1)[0])
        if not np.isclose(stored_t, item["t"], rtol=0.0, atol=1e-6):
            raise ValueError(f"Time mismatch in {path}: {stored_t} vs {item['t']}")
        frames.append({
            "npz": str(path),
            "internal_phase": int(item["internal_phase"]),
            "original_phase": int(item["original_phase"]),
            "t": float(item["t"]),
        })
    internal = [frame["internal_phase"] for frame in frames]
    if internal != list(range(len(frames))):
        raise ValueError(f"Internal phase order is invalid: {internal}")
    return frames


def optimize_shape(
    decoder, frame, c_s, c_m, specs, num_iterations, device,
):
    c_s.requires_grad_(True)
    optimizer = torch.optim.Adam([c_s], lr=5e-3)
    final_error = math.inf
    adjust_every = max(num_iterations // 2, 1)
    for iteration in range(num_iterations):
        optimizer.param_groups[0]["lr"] = 5e-3 * (0.1 ** (iteration // adjust_every))
        samples, time_value = get_sdf_samples_test(frame["npz"], specs["SamplesPerScene"])
        xyz = samples[None, :, :3].to(device)
        sdf_gt = samples[:, 3:4].to(device)
        time = torch.tensor([time_value], dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        _, sdf_pred = decoder(xyz, time, c_m[0:1], c_s)
        clamp = specs["ClampingDistance"]
        sdf_loss = torch.nn.functional.l1_loss(
            torch.clamp(sdf_pred, -clamp, clamp),
            torch.clamp(sdf_gt, -clamp, clamp),
        )
        loss = sdf_loss + 1e-4 * torch.mean(c_s.square())
        loss.backward()
        optimizer.step()
        final_error = float(sdf_loss.detach())
    return final_error


def optimize_motion_batch(
    decoder, frames, c_s, c_m, specs, num_iterations, device,
):
    if not frames:
        return {}
    indices = [frame["internal_phase"] for frame in frames]
    if any(index == 0 for index in indices):
        raise ValueError("Reference phase must be optimized separately")
    c_s.requires_grad_(False)
    cm_batch = c_m[indices].detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([cm_batch], lr=5e-4)
    adjust_every = max(num_iterations // 2, 1)
    final_errors = None
    for iteration in range(num_iterations):
        optimizer.param_groups[0]["lr"] = 5e-4 * (0.1 ** (iteration // adjust_every))
        sampled = [
            get_sdf_samples_test(frame["npz"], specs["SamplesPerScene"])
            for frame in frames
        ]
        samples = torch.stack([item[0] for item in sampled]).to(device)
        times = torch.tensor(
            [item[1] for item in sampled], dtype=torch.float32, device=device
        )
        optimizer.zero_grad(set_to_none=True)
        _, sdf_pred = decoder(samples[:, :, :3], times, cm_batch, c_s)
        sdf_pred = sdf_pred.reshape(len(frames), samples.shape[1], 1)
        clamp = specs["ClampingDistance"]
        errors = torch.abs(
            torch.clamp(sdf_pred, -clamp, clamp)
            - torch.clamp(samples[:, :, 3:4], -clamp, clamp)
        )
        final_errors = errors.mean(dim=(1, 2))
        loss = torch.sum(final_errors + 1e-4 * cm_batch.square().mean(dim=1))
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        c_m[indices] = cm_batch.detach()
    return {
        frame["original_phase"]: float(error)
        for frame, error in zip(frames, final_errors.detach())
    }


def _restore_coordinates(vertices: np.ndarray, archive) -> np.ndarray:
    scale = float(np.asarray(archive["scale"]).reshape(-1)[0])
    offset = np.asarray(archive["offset"], dtype=np.float64).reshape(3)
    inverse = np.asarray(archive["Pi"], dtype=np.float64)
    restored = vertices.astype(np.float64) / scale - offset
    homogeneous = np.c_[restored, np.ones(len(restored), dtype=np.float64)]
    return (homogeneous @ inverse.T)[:, :3]


def _valid_obj(path: Path) -> bool:
    try:
        mesh = trimesh.load(path, process=False)
        return (
            isinstance(mesh, trimesh.Trimesh)
            and len(mesh.vertices) > 0
            and len(mesh.faces) > 0
            and np.isfinite(mesh.vertices).all()
            and np.asarray(mesh.faces).min() >= 0
            and np.asarray(mesh.faces).max() < len(mesh.vertices)
        )
    except Exception:
        return False


def _atomic_torch_save(value, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def create_mesh(
    decoder, c_s, c_m, time_value, npz_filename, output_path,
    resolution, device, max_batch=MAX_GRID_BATCH,
):
    coordinates = torch.linspace(-1.0, 1.0, resolution)
    grid = torch.stack(
        torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    chunks = []
    time = torch.tensor([time_value], dtype=torch.float32, device=device)
    for start in range(0, len(grid), max_batch):
        _, sdf = decoder(grid[start:start + max_batch].to(device)[None], time, c_m, c_s)
        chunks.append(sdf[:, 0].detach().cpu())
    volume = torch.cat(chunks).reshape(resolution, resolution, resolution).numpy()
    minimum, maximum = float(volume.min()), float(volume.max())
    if not minimum <= 0.0 <= maximum:
        raise RuntimeError(f"SDF has no zero crossing: range=({minimum}, {maximum})")
    voxel_size = 2.0 / (resolution - 1)
    vertices, faces, _, _ = skimage.measure.marching_cubes(
        volume, level=0.0, spacing=(voxel_size, voxel_size, voxel_size)
    )
    vertices -= 1.0
    with np.load(npz_filename, allow_pickle=False) as archive:
        vertices = _restore_coordinates(vertices, archive)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp.obj")
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(temporary)
    if not _valid_obj(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Generated invalid mesh: {output_path}")
    temporary.replace(output_path)
    return int(len(vertices)), int(len(faces))


def _motion_groups_with_oom_retry(
    decoder, motion_frames, c_s, c_m, specs, iterations, requested_batch, device,
):
    errors, used_batches = {}, []
    cursor = 0
    batch_size = min(requested_batch, len(motion_frames)) if motion_frames else 0
    while cursor < len(motion_frames):
        attempt = min(batch_size, len(motion_frames) - cursor)
        group = motion_frames[cursor:cursor + attempt]
        try:
            errors.update(
                optimize_motion_batch(
                    decoder, group, c_s, c_m, specs, iterations, device
                )
            )
            used_batches.append(attempt)
            cursor += attempt
        except torch.cuda.OutOfMemoryError:
            if not str(device).startswith("cuda") or attempt == 1:
                raise
            torch.cuda.empty_cache()
            batch_size = max(1, attempt // 2)
            logging.warning("CUDA OOM; retrying motion optimization with batch=%d", batch_size)
    return errors, used_batches


def run_case_inference(config: CaseConfig, preprocess_report: dict) -> dict:
    device = resolve_device(config.device)
    seed = RANDOM_SEED + int.from_bytes(
        hashlib.sha256(config.case_id.encode("utf-8")).digest()[:4], "little"
    )
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    mapping = preprocess_report["phase_mapping"]
    output_root = config.output_dir
    results, model_metadata, effective_batches = [], {}, {}
    for surface in SURFACES:
        decoder, specs, checkpoint_metadata = load_decoder(
            surface, config.checkpoints[surface], device
        )
        model_metadata[surface] = checkpoint_metadata
        frames = _load_case_frames(
            preprocess_report["manifests"][surface], mapping, config.case_id, surface
        )
        c_s = torch.empty(1, specs["CsLength"], device=device).normal_(
            0, 1 / math.sqrt(specs["CsLength"])
        )
        c_m = torch.empty(len(frames), specs["CmLength"], device=device).normal_(
            0, 1 / math.sqrt(specs["CmLength"])
        )
        shape_error = optimize_shape(
            decoder, frames[0], c_s, c_m, specs, config.num_iterations, device
        )
        motion_errors, batches = _motion_groups_with_oom_retry(
            decoder,
            frames[1:],
            c_s,
            c_m,
            specs,
            config.num_iterations,
            config.phase_batch_size,
            device,
        )
        effective_batches[surface] = batches
        errors = {frames[0]["original_phase"]: shape_error, **motion_errors}
        for frame in frames:
            original_phase = frame["original_phase"]
            output_path = (
                output_root / "artifacts" / "raw" / surface
                / f"{config.case_id}-{surface}-{original_phase:02d}.obj"
            )
            status = "generated"
            if config.skip_existing and output_path.is_file() and _valid_obj(output_path):
                mesh = trimesh.load(output_path, process=False)
                vertices, faces, status = len(mesh.vertices), len(mesh.faces), "skipped_valid"
            else:
                internal = frame["internal_phase"]
                vertices, faces = create_mesh(
                    decoder,
                    c_s.detach(),
                    c_m[internal:internal + 1].detach(),
                    frame["t"],
                    frame["npz"],
                    output_path,
                    config.resolution,
                    device,
                )
            results.append({
                "surface": surface,
                "case_id": config.case_id,
                "original_phase": original_phase,
                "internal_phase": frame["internal_phase"],
                "t": frame["t"],
                "error": float(errors[original_phase]),
                "iterations": config.num_iterations,
                "resolution": config.resolution,
                "vertices": int(vertices),
                "faces": int(faces),
                "mesh": str(output_path),
                "status": status,
            })
        codes_dir = output_root / "artifacts" / "codes" / surface
        _atomic_torch_save(
            c_s.detach().cpu(), codes_dir / f"{config.case_id}_{surface}_cs.pth"
        )
        _atomic_torch_save(
            c_m.detach().cpu(), codes_dir / f"{config.case_id}_{surface}_cm.pth"
        )
        del decoder
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    report = {
        "case_id": config.case_id,
        "device": device,
        "models": model_metadata,
        "effective_motion_batches": effective_batches,
        "results": results,
        "all_passed": all(
            np.isfinite(item["error"]) and item["vertices"] > 0 and item["faces"] > 0
            for item in results
        ),
    }
    return report
