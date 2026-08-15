"""Train the independent endocardial and epicardial 4DMM models.

``train_4dmm.py`` is the public unified entry point; ``run_training`` remains
available for programmatic single-surface experiments.
"""

import argparse
import datetime
import csv
import gc
import importlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from deep_sdf.dataset import SDFSamples, resolve_manifest_paths
from deep_sdf.loss import HuberFunc, LipschitzLoss, apply_pointpair_reg, apply_pointwise_reg
from deep_sdf.lr_schedule import get_learning_rate_schedules
from project_config import (
    EXPERIMENT_ROOT,
    OUTPUT_ROOT,
    RANDOM_SEED,
    REPO_ROOT,
    TRAIN_CONFIG_PATH,
    TRAIN_MANIFEST_ROOT,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SURFACES = ("endo", "epi")
INITIALIZATION_MODES = {"scratch", "resume"}
EXECUTION_MODES = {"auto", "serial"}
CONTINUE_ON_SURFACE_ERROR = True


def _load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_config_path(value):
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise TypeError("Training paths must be non-empty strings or null")
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_training_config(path=TRAIN_CONFIG_PATH):
    """Load and validate the dataset-level training configuration."""
    expected = {
        "execution", "target_epoch", "batch_size", "samples_per_scene", "surfaces",
    }
    payload = _load_json(path)
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown or missing:
        raise ValueError(
            f"Training config fields mismatch; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    execution = payload["execution"]
    if not isinstance(execution, dict) or set(execution) != {"mode", "devices"}:
        raise ValueError("training.execution must contain exactly mode and devices")
    if execution["mode"] not in EXECUTION_MODES:
        raise ValueError(f"execution.mode must be one of {sorted(EXECUTION_MODES)}")
    devices = execution["devices"]
    if devices != "auto":
        if not isinstance(devices, list) or not devices or not all(
            isinstance(value, str) and value for value in devices
        ):
            raise TypeError("execution.devices must be auto or a non-empty string list")
        execution["devices"] = list(devices)
    for key in ("target_epoch", "batch_size", "samples_per_scene"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise TypeError(f"training.{key} must be a positive integer")
    surfaces = payload["surfaces"]
    if not isinstance(surfaces, dict) or set(surfaces) != set(SURFACES):
        raise ValueError("training.surfaces must contain exactly endo and epi")
    surface_expected = {"enabled", "initialization_mode", "resume_from"}
    for surface in SURFACES:
        config = surfaces[surface]
        if not isinstance(config, dict) or set(config) != surface_expected:
            raise ValueError(
                f"training.surfaces.{surface} must contain exactly "
                f"{sorted(surface_expected)}"
            )
        if not isinstance(config["enabled"], bool):
            raise TypeError(f"training.surfaces.{surface}.enabled must be boolean")
        if config["initialization_mode"] not in INITIALIZATION_MODES:
            raise ValueError(
                f"training.surfaces.{surface}.initialization_mode must be one of "
                f"{sorted(INITIALIZATION_MODES)}"
            )
        config["resume_from"] = _resolve_config_path(config["resume_from"])
        if config["initialization_mode"] == "resume":
            checkpoint_dir = config["resume_from"] or EXPERIMENT_ROOT / surface
            if not Path(checkpoint_dir).is_dir():
                raise FileNotFoundError(
                    f"training.surfaces.{surface}.resume_from does not exist: "
                    f"{checkpoint_dir}"
                )
            required = (
                Path(checkpoint_dir) / "ModelParameters" / "latest.pth",
                Path(checkpoint_dir) / "OptimizerParameters" / "latest.pth",
                Path(checkpoint_dir) / "LatentCodes" / "latest_cs.pth",
                Path(checkpoint_dir) / "LatentCodes" / "latest_cm.pth",
            )
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(
                    f"training.surfaces.{surface}.resume_from is not a complete "
                    "user training run; missing: " + ", ".join(missing)
                )
    if not any(config["enabled"] for config in surfaces.values()):
        raise ValueError("At least one training surface must be enabled")
    return payload


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_history_csv(path, epoch_history):
    """Write the aggregated epoch history in a stable, spreadsheet-friendly form."""
    fields = (
        "epoch", "batches", "seconds", "total", "sdf", "regularization",
        "pointwise", "pointpair", "eta_seconds",
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in epoch_history:
            row = {field: item.get(field, "") for field in fields}
            writer.writerow(row)
        temporary = Path(handle.name)
    temporary.replace(path)


def _write_loss_curve(path, epoch_history, surface):
    """Render a small loss plot without making matplotlib a module import dependency."""
    if not epoch_history:
        return
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    epochs = [item["epoch"] for item in epoch_history]
    totals = [item.get("total", float("nan")) for item in epoch_history]
    sdfs = [item.get("sdf", float("nan")) for item in epoch_history]
    figure, axis = plt.subplots(figsize=(6.4, 4.0), dpi=120)
    axis.plot(epochs, totals, marker="o", label="total")
    axis.plot(epochs, sdfs, marker="o", label="sdf")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.set_title(f"{surface} training loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(temporary, format="png")
    plt.close(figure)
    temporary.replace(path)


def _checkpoint_epoch(experiment_dir, device="cpu"):
    """Read and cross-check the epoch of a complete latest checkpoint set."""
    experiment_dir = Path(experiment_dir)
    paths = (
        experiment_dir / "ModelParameters" / "latest.pth",
        experiment_dir / "OptimizerParameters" / "latest.pth",
        experiment_dir / "LatentCodes" / "latest_cs.pth",
        experiment_dir / "LatentCodes" / "latest_cm.pth",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Incomplete user training checkpoint under "
            f"{experiment_dir}; missing: {', '.join(missing)}"
        )
    epochs = set()
    for path in paths:
        state = torch.load(path, map_location=device)
        if "epoch" not in state:
            raise KeyError(f"Checkpoint has no epoch field: {path}")
        epochs.add(int(state["epoch"]))
    if len(epochs) != 1:
        raise RuntimeError(f"Checkpoint epoch mismatch under {experiment_dir}: {sorted(epochs)}")
    return epochs.pop()


def _resolve_training_devices(config):
    """Resolve visible devices and whether independent surface jobs can run in parallel."""
    execution = config["execution"]
    requested = execution["devices"]
    if requested == "auto":
        if torch.cuda.is_available():
            devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]
    else:
        devices = list(requested)
        for device in devices:
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"Configured device {device} but CUDA is unavailable")
            if device.startswith("cuda:"):
                try:
                    index = int(device.split(":", 1)[1])
                except ValueError as error:
                    raise ValueError(f"Invalid CUDA device name: {device}") from error
                if index < 0 or index >= torch.cuda.device_count():
                    raise ValueError(
                        f"Configured device {device} is not visible; "
                        f"visible device count is {torch.cuda.device_count()}"
                    )
    enabled = [surface for surface in SURFACES if config["surfaces"][surface]["enabled"]]
    can_parallel = (
        execution["mode"] == "auto"
        and len(enabled) >= 2
        and len(devices) >= 2
        and all(device.startswith("cuda:") for device in devices[:2])
    )
    if can_parallel:
        assignments = {surface: devices[index] for index, surface in enumerate(enabled)}
        return True, assignments
    if not devices:
        raise RuntimeError("No training device is available")
    return False, {surface: devices[0] for surface in enabled}


def _surface_output_dir(run_id, surface, surface_config):
    return OUTPUT_ROOT / "training" / run_id / surface


def _run_surface_from_config(config, surface, device, run_id):
    """Run one independent surface, converting exceptions into a report."""
    surface_config = config["surfaces"][surface]
    output_dir = _surface_output_dir(run_id, surface, surface_config)
    result = {
        "surface": surface,
        "device": device,
        "output_dir": str(output_dir),
        "target_epoch": config["target_epoch"],
    }
    try:
        initialization_mode = surface_config["initialization_mode"]
        resume_from = surface_config.get("resume_from")
        if initialization_mode == "resume":
            if resume_from is None:
                raise ValueError(
                    f"{surface} resume requires resume_from to point to one of your "
                    "own training runs; bundled ACDC weights are inference-only"
                )
            checkpoint_dir = Path(resume_from)
            completed_epoch = _checkpoint_epoch(checkpoint_dir, device="cpu")
            result["checkpoint_epoch"] = completed_epoch
            if completed_epoch >= config["target_epoch"]:
                result.update({
                    "status": "already_complete",
                    "start_epoch": completed_epoch + 1,
                })
                return result
        manifest_path = TRAIN_MANIFEST_ROOT / f"data_load_{surface}.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Training manifest is missing: {manifest_path}. Run process_data.py first."
            )
        run_result = run_training(
            surface=surface,
            device=device,
            output_dir=output_dir,
            manifest_path=manifest_path,
            samples_per_scene=config["samples_per_scene"],
            batch_size=config["batch_size"],
            initialization_mode=initialization_mode,
            continue_from=resume_from,
            target_epoch=config["target_epoch"],
            max_batches_per_epoch=None,
            checkpoint_every=None,
            shuffle=True,
        )
        result.update({
            "status": "completed",
            "start_epoch": run_result["start_epoch"],
            "final_epoch": run_result["epoch"],
            "epoch_count": len(run_result["epoch_history"]),
            "last_metrics": run_result["metrics"],
        })
    except Exception as error:  # each surface gets a durable failure report
        result.update({
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        })
    return result


def _training_worker(queue, config, surface, device, run_id):
    try:
        result = _run_surface_from_config(config, surface, device, run_id)
    except Exception as error:
        result = {
            "surface": surface,
            "device": device,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
    queue.put(result)


def _serializable_config(config):
    return json.loads(json.dumps(config, default=str))


def run_all_training(config=None, run_id=None):
    """Run enabled surfaces with automatic independent-device scheduling."""
    config = load_training_config() if config is None else config
    run_id = run_id or datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    summary_dir = OUTPUT_ROOT / "training" / run_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    enabled = [surface for surface in SURFACES if config["surfaces"][surface]["enabled"]]
    parallel, assignments = _resolve_training_devices(config)
    results = {}
    if parallel:
        context = mp.get_context("spawn")
        queue = context.Queue()
        processes = {
            surface: context.Process(
                target=_training_worker,
                args=(queue, config, surface, assignments[surface], run_id),
                name=f"train-{surface}",
            )
            for surface in enabled
        }
        for process in processes.values():
            process.start()
        for _ in processes:
            item = queue.get()
            results[item["surface"]] = item
        for process in processes.values():
            process.join()
    else:
        for surface in enabled:
            item = _run_surface_from_config(config, surface, assignments[surface], run_id)
            results[surface] = item
            # Release the completed surface before constructing the next model on
            # a one-GPU/CPU run.  run_training's locals are already out of scope,
            # but explicit collection prevents cached CUDA blocks accumulating.
            gc.collect()
            if assignments[surface].startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if item["status"] == "failed" and not CONTINUE_ON_SURFACE_ERROR:
                break
    failed = [item for item in results.values() if item["status"] == "failed"]
    summary = {
        "run_id": run_id,
        "status": "failed" if failed else "success",
        "mode": "parallel" if parallel else "serial",
        "devices": assignments,
        "config": _serializable_config(config),
        "surfaces": results,
    }
    _atomic_json(summary_dir / "report.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)
    return summary


def _load_checkpoint(experiment_dir, decoder, optimizer, c_s, c_m, device):
    paths = {
        "model": experiment_dir / "ModelParameters" / "latest.pth",
        "optimizer": experiment_dir / "OptimizerParameters" / "latest.pth",
        "cs": experiment_dir / "LatentCodes" / "latest_cs.pth",
        "cm": experiment_dir / "LatentCodes" / "latest_cm.pth",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    model_state = torch.load(paths["model"], map_location=device)
    optimizer_state = torch.load(paths["optimizer"], map_location=device)
    cs_state = torch.load(paths["cs"], map_location=device)
    cm_state = torch.load(paths["cm"], map_location=device)
    decoder.load_state_dict(model_state["model_state_dict"], strict=True)
    c_s.load_state_dict(cs_state["latent_codes"])
    c_m.load_state_dict(cm_state["latent_codes"])
    optimizer.load_state_dict(optimizer_state["optimizer_state_dict"])
    epochs = {model_state["epoch"], optimizer_state["epoch"], cs_state["epoch"], cm_state["epoch"]}
    if len(epochs) != 1:
        raise RuntimeError(f"Checkpoint epoch mismatch: {sorted(epochs)}")
    return epochs.pop()


def _checkpoint_payloads(epoch, decoder, optimizer, c_s, c_m):
    return {
        "model": {"epoch": epoch, "model_state_dict": decoder.state_dict()},
        "optimizer": {"epoch": epoch, "optimizer_state_dict": optimizer.state_dict()},
        "cs": {"epoch": epoch, "latent_codes": c_s.state_dict()},
        "cm": {"epoch": epoch, "latent_codes": c_m.state_dict()},
    }


def _write_checkpoint_files(destinations, payloads):
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    for name, path in destinations.items():
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        )
        temporary = Path(handle.name)
        handle.close()
        try:
            torch.save(payloads[name], temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _save_checkpoint(output_dir, epoch, decoder, optimizer, c_s, c_m, snapshot=False):
    suffix = f"{epoch:04d}.pth" if snapshot else "latest.pth"
    destinations = {
        "model": output_dir / "ModelParameters" / suffix,
        "optimizer": output_dir / "OptimizerParameters" / suffix,
        "cs": output_dir / "LatentCodes" / (f"{epoch:04d}_cs.pth" if snapshot else "latest_cs.pth"),
        "cm": output_dir / "LatentCodes" / (f"{epoch:04d}_cm.pth" if snapshot else "latest_cm.pth"),
    }
    _write_checkpoint_files(
        destinations, _checkpoint_payloads(epoch, decoder, optimizer, c_s, c_m)
    )


def _make_embeddings(sequence_count, frame_count, specs, device):
    code_bound = specs.get("CodeBound")
    c_s = torch.nn.Embedding(sequence_count, specs["CsLength"], max_norm=code_bound).to(device)
    c_m = torch.nn.Embedding(frame_count, specs["CmLength"], max_norm=code_bound).to(device)
    std = specs.get("CodeInitStdDev", 1.0)
    torch.nn.init.normal_(c_s.weight, 0.0, std / math.sqrt(specs["CsLength"]))
    torch.nn.init.normal_(c_m.weight, 0.0, std / math.sqrt(specs["CmLength"]))
    return c_s, c_m


def _make_optimizer(decoder, c_s, c_m, schedules):
    return torch.optim.Adam([
        {"params": decoder.motion_net.parameters(), "lr": schedules[0].get_learning_rate(0)},
        {"params": decoder.shape_net.parameters(), "lr": schedules[1].get_learning_rate(0)},
        {"params": c_s.parameters(), "lr": schedules[2].get_learning_rate(0)},
        {"params": c_m.parameters(), "lr": schedules[3].get_learning_rate(0)},
    ])


def _train_batch(batch, decoder, c_s, c_m, optimizer, specs, epoch, device):
    sdf_data, indices = batch
    xyz = sdf_data["p_sdf"][:, :, :3].to(device).requires_grad_(True)
    sdf_gt = sdf_data["p_sdf"][:, :, 3].to(device).reshape(-1, 1)
    t = sdf_data["t"].to(device).float().reshape(-1)
    seq_idx = sdf_data["seq_idx"].to(device).long().reshape(-1)
    indices = indices.to(device).long().reshape(-1)
    clamp = specs["ClampingDistance"]
    sdf_gt = torch.clamp(sdf_gt, -clamp, clamp)
    cs_vecs = c_s(seq_idx)
    cm_vecs = c_m(indices)

    optimizer.zero_grad(set_to_none=True)
    new_xyz, sdf_pred = decoder(xyz, t, cm_vecs, cs_vecs)
    new_xyz = new_xyz.reshape(-1, 3)
    xyz_flat = xyz.reshape(-1, 3)
    sdf_pred = torch.clamp(sdf_pred, -clamp, clamp)
    point_count = sdf_pred.shape[0]
    sdf_loss = torch.nn.functional.l1_loss(sdf_pred, sdf_gt, reduction="sum") / point_count
    total = sdf_loss
    metrics = {"sdf": float(sdf_loss.detach()), "regularization": 0.0, "pointwise": 0.0, "pointpair": 0.0}

    if specs.get("CodeRegularization", True):
        non_ed = torch.nonzero(t != 0, as_tuple=True)[0]
        cm_reg = torch.zeros((), device=device)
        if non_ed.numel() > 0:
            cm_reg = torch.norm(cm_vecs.index_select(0, non_ed), dim=1).mean()
        # Preserve the original cardiac_4d scaling for checkpoint/training parity.
        cs_reg = torch.norm(cs_vecs, dim=1).sum() / point_count
        regularization = cm_reg + cs_reg
        total = total + specs.get("CodeRegularizationLambda", 1e-4) * min(1.0, epoch / 100) * regularization
        metrics["regularization"] = float(regularization.detach())

    if specs.get("UsePointwiseLoss", False):
        pointwise = apply_pointwise_reg(new_xyz, xyz_flat, HuberFunc(reduction="sum"), point_count)
        total = total + pointwise * specs.get("PointwiseLossWeight", 0.0) * max(1.0, 10.0 * (1 - epoch / 100))
        metrics["pointwise"] = float(pointwise.detach())

    if specs.get("UsePointpairLoss", False):
        pointpair = apply_pointpair_reg(
            new_xyz,
            xyz_flat,
            LipschitzLoss(k=0.5, reduction="sum"),
            xyz.shape[0],
            point_count,
        )
        total = total + pointpair * specs.get("PointpairLossWeight", 0.0) * min(1.0, epoch / 100)
        metrics["pointpair"] = float(pointpair.detach())

    total.backward()
    if specs.get("GradientClipNorm") is not None:
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), specs["GradientClipNorm"])
    optimizer.step()
    metrics["total"] = float(total.detach())
    return metrics


def run_training(
    surface="endo",
    device=DEVICE,
    smoke_test=False,
    output_dir=None,
    manifest_path=None,
    samples_per_scene=None,
    batch_size=None,
    continue_from=None,
    initialization_mode=None,
    final_epoch=None,
    max_batches_per_epoch=None,
    target_epoch=None,
    checkpoint_every=None,
    shuffle=True,
):
    if surface not in {"endo", "epi"}:
        raise ValueError(f"Unknown surface: {surface}")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    experiment_dir = EXPERIMENT_ROOT / surface
    specs = _load_json(experiment_dir / "specs.json")
    if manifest_path is None:
        manifest_path = TRAIN_MANIFEST_ROOT / f"data_load_{surface}.json"
    else:
        manifest_path = Path(manifest_path)
    manifest = resolve_manifest_paths(_load_json(manifest_path), REPO_ROOT)
    sequence_count = sum(
        len(sequences) for sequences in manifest.get("train", {}).values()
    )
    if sequence_count == 0:
        raise ValueError(f"Training manifest has no sequences: {manifest_path}")

    if samples_per_scene is None:
        samples_per_scene = 256 if smoke_test else specs["SamplesPerScene"]
    if batch_size is None:
        batch_size = 2 if smoke_test else specs["BatchSize"]
    dataset = SDFSamples(manifest, samples_per_scene)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=True)
    arch = importlib.import_module("networks." + specs["NetworkArch"])
    decoder = arch.Decoder(**specs["NetworkSpecs"]).to(device)
    c_s, c_m = _make_embeddings(sequence_count, len(dataset), specs, device)
    schedules = get_learning_rate_schedules(specs)
    optimizer = _make_optimizer(decoder, c_s, c_m, schedules)
    start_epoch = 1
    if initialization_mode is None:
        if continue_from is not None:
            initialization_mode = "resume"
        else:
            initialization_mode = "scratch"
    if initialization_mode not in {"scratch", "resume"}:
        raise ValueError(f"Unknown initialization_mode: {initialization_mode}")

    if initialization_mode == "resume":
        resume_dir = Path(continue_from or experiment_dir)
        start_epoch = _load_checkpoint(
            resume_dir, decoder, optimizer, c_s, c_m, device
        ) + 1

    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = experiment_dir / "runs" / timestamp
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_specs = dict(specs)
    output_specs["DataLoad"] = (
        Path("training_data") / f"data_load_{surface}.json"
    ).as_posix()
    (output_dir / "specs.json").write_text(json.dumps(output_specs, indent=2) + "\n")

    if final_epoch is None:
        final_epoch = start_epoch if smoke_test else (target_epoch or specs["NumEpochs"])
    if final_epoch < start_epoch:
        raise ValueError(f"final_epoch={final_epoch} is before start_epoch={start_epoch}")
    if checkpoint_every is not None and (
        isinstance(checkpoint_every, bool) or checkpoint_every < 1
    ):
        raise ValueError("checkpoint_every must be positive or None")
    last_metrics = None
    history = []
    epoch_history = []
    run_started = time.perf_counter()
    for epoch in range(start_epoch, final_epoch + 1):
        epoch_started = time.perf_counter()
        decoder.train()
        for group, schedule in zip(optimizer.param_groups, schedules):
            group["lr"] = schedule.get_learning_rate(epoch)
        for batch_index, batch in enumerate(loader):
            last_metrics = _train_batch(batch, decoder, c_s, c_m, optimizer, specs, epoch, device)
            logging.info("%s epoch=%d batch=%d metrics=%s", surface, epoch, batch_index, last_metrics)
            history.append({"epoch": epoch, "batch": batch_index, **last_metrics})
            batch_limit = max_batches_per_epoch
            if smoke_test or (batch_limit is not None and batch_index + 1 >= batch_limit):
                break
        epoch_metrics = {
            "epoch": epoch,
            "batches": len([item for item in history if item["epoch"] == epoch]),
            "seconds": time.perf_counter() - epoch_started,
        }
        for key in ("total", "sdf", "regularization", "pointwise", "pointpair"):
            epoch_metrics[key] = 0.0
        if epoch_metrics["batches"]:
            for key in ("total", "sdf", "regularization", "pointwise", "pointpair"):
                values = [item[key] for item in history if item["epoch"] == epoch]
                epoch_metrics[key] = float(np.mean(values))
        elapsed = time.perf_counter() - run_started
        average = elapsed / max(epoch - start_epoch + 1, 1)
        epoch_metrics["eta_seconds"] = average * max(final_epoch - epoch, 0)
        epoch_history.append(epoch_metrics)
        _save_checkpoint(output_dir, epoch, decoder, optimizer, c_s, c_m)
        if checkpoint_every is not None and (
            epoch % checkpoint_every == 0 or epoch == final_epoch
        ):
            _save_checkpoint(output_dir, epoch, decoder, optimizer, c_s, c_m, snapshot=True)
        _write_history_csv(output_dir / "history.csv", epoch_history)
        _write_loss_curve(output_dir / "loss_curve.png", epoch_history, surface)
        print(
            f"[{surface}] epoch {epoch}/{final_epoch} "
            f"loss={epoch_metrics.get('total', float('nan')):.6f} "
            f"epoch_seconds={epoch_metrics['seconds']:.1f} "
            f"eta_seconds={epoch_metrics['eta_seconds']:.1f}",
            flush=True,
        )
    return {
        "surface": surface,
        "epoch": final_epoch,
        "metrics": last_metrics,
        "history": history,
        "epoch_history": epoch_history,
        "start_epoch": start_epoch,
        "target_epoch": final_epoch,
        "output_dir": str(output_dir),
    }


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Train independent endo and epi 4DMM models."
    )
    parser.add_argument(
        "--config", type=Path, default=TRAIN_CONFIG_PATH,
        help="training configuration JSON (default: configs/training.json)",
    )
    parser.add_argument("--run-id", help="optional output run name")
    parser.add_argument(
        "--surface", choices=("both", "endo", "epi"),
        help="override the enabled surface models from the config",
    )
    parser.add_argument("--target-epoch", type=int, help="override target epoch")
    parser.add_argument("--batch-size", type=int, help="override batch size")
    parser.add_argument(
        "--samples-per-scene", type=int,
        help="override sampled SDF points per frame",
    )
    parser.add_argument(
        "--mode", choices=tuple(sorted(EXECUTION_MODES)),
        help="override serial/automatic scheduling",
    )
    parser.add_argument(
        "--devices", nargs="+",
        help="visible training devices, for example cuda:0 cuda:1 or cpu",
    )
    parser.add_argument(
        "--resume-endo", type=Path,
        help="resume endo from one of your own training runs",
    )
    parser.add_argument(
        "--resume-epi", type=Path,
        help="resume epi from one of your own training runs",
    )
    return parser


def _apply_cli_overrides(config, args):
    for key in ("target_epoch", "batch_size", "samples_per_scene"):
        value = getattr(args, key)
        if value is not None:
            if value < 1:
                raise ValueError(f"--{key.replace('_', '-')} must be positive")
            config[key] = value
    if args.mode is not None:
        config["execution"]["mode"] = args.mode
    if args.devices is not None:
        config["execution"]["devices"] = (
            "auto" if args.devices == ["auto"] else args.devices
        )
    for surface in SURFACES:
        if args.surface is not None:
            config["surfaces"][surface]["enabled"] = (
                args.surface == "both" or args.surface == surface
            )
        resume_path = getattr(args, f"resume_{surface}")
        if resume_path is not None:
            resolved = resume_path if resume_path.is_absolute() else REPO_ROOT / resume_path
            config["surfaces"][surface]["enabled"] = True
            config["surfaces"][surface]["initialization_mode"] = "resume"
            config["surfaces"][surface]["resume_from"] = resolved
    return config


def main(argv=None):
    args = _build_parser().parse_args(argv)
    config = _apply_cli_overrides(load_training_config(args.config), args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_all_training(config=config, run_id=args.run_id)
    raise SystemExit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
