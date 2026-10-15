"""Generate the SDF archives used by the public training entry point.

Training accepts either already separated normalized OBJ files or the public
dataset's labeled VTK files. Inference starts from masks and does not use this
script.
"""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
import trimesh

from mesh_to_sdf.mesh_to_sdf import sample_sdf_near_surface
from dataset_layout import discover_training_cases
from preprocess_mask import write_mask_transform
from project_config import (
    MIN_FREE_SPACE_GB,
    PREPROCESS_WORKERS,
    PROCESS_ROOT,
    RANDOM_SEED,
    REPO_ROOT,
    SDF_SAMPLE_COUNT,
    SURFACE_SAMPLE_COUNT,
    SURFACES,
    TRAIN_DATA_ROOT,
    TRAIN_MESH_ROOT,
    TRANSFORM_ROOT,
)


SPLITS = ("train",)
TRAIN_CASE_IDS = None
TRAIN_PHASES = None
OVERWRITE = False
VALIDATE_ONLY = False
INPUT_FORMAT = "obj"
DATASET_ROOT = None
DATASET_CASES = {}
TRANSFORM_CACHE = {}
REFERENCE_PHASE = 0
MIRROR = "auto"
MIRROR_FALLBACK = "error"
SLICE_ORDER = "auto"

TRAIN_KEYS = {"pos", "neg", "t", "P", "Pi", "offset", "scale"}


def load_transform(path):
    blocks = {}
    current = None
    rows = []
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.endswith(":"):
            if current is not None:
                blocks[current] = np.asarray(rows, dtype=np.float64)
            current = line[:-1]
            rows = []
        else:
            rows.append([float(value) for value in line.split()])
    if current is not None:
        blocks[current] = np.asarray(rows, dtype=np.float64)
    missing = {"P", "Pi", "offset", "scale"} - blocks.keys()
    if missing:
        raise ValueError(f"{path} is missing transform blocks: {sorted(missing)}")
    return {
        "P": blocks["P"],
        "Pi": blocks["Pi"],
        "offset": blocks["offset"].reshape(-1),
        "scale": float(blocks["scale"].reshape(-1)[0]),
    }


def ensure_dataset_transforms() -> None:
    """Load one transform per selected case, generating it from Masks if absent."""
    selected = set(TRAIN_CASE_IDS or DATASET_CASES)
    for case_id, case in sorted(DATASET_CASES.items()):
        if case_id not in selected:
            continue
        transform_path = Path(case["transform"])
        if not transform_path.is_file():
            if VALIDATE_ONLY:
                raise FileNotFoundError(
                    f"Missing transform during validation: {transform_path}"
                )
            mask_dir = case.get("mask_dir")
            if mask_dir is None:
                raise FileNotFoundError(
                    f"Missing {transform_path} and no Mask is available for "
                    f"case {case_id}"
                )
            print(
                f"[transform] {case_id}: generating {transform_path} from "
                f"reference phase {REFERENCE_PHASE}",
                flush=True,
            )
            generated = write_mask_transform(
                mask_dir=mask_dir,
                output_path=transform_path,
                reference_phase=REFERENCE_PHASE,
                mirror=MIRROR,
                mirror_fallback=MIRROR_FALLBACK,
                slice_order=SLICE_ORDER,
            )
            orientation = generated["orientation"]
            print(
                f"[transform] {case_id}: mirror={orientation['selected']}"
                + (" (low confidence)" if orientation.get("low_confidence") else ""),
                flush=True,
            )
        # Reload the persisted text so all frames use exactly what users keep.
        TRANSFORM_CACHE[case_id] = load_transform(transform_path)


def _phase_id(path: Path, prefix: str, suffix: str) -> int:
    stem = path.name
    if not stem.startswith(prefix) or not stem.endswith(suffix):
        raise ValueError(f"Unexpected frame filename: {path.name}")
    value = stem[len(prefix) : -len(suffix)] if suffix else stem[len(prefix) :]
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"Frame filename must contain a numeric phase: {path}") from error


def discover_frames(source_root, surface, patients=None, phases=None):
    """Return (case, phase id, sequence position, count, source path) tuples.

    Both OBJ and labeled-VTK inputs use numeric phase IDs.  IDs may be
    non-zero and non-contiguous; the sequence position is used only for the
    normalized temporal coordinate stored in the SDF archive.
    """
    frames = []
    if INPUT_FORMAT == "vtk":
        case_ids = tuple(DATASET_CASES)
    elif patients is None:
        case_ids = tuple(
            path.name
            for path in source_root.iterdir()
            if path.is_dir() and path.name not in {"P", "processed"}
        )
    else:
        case_ids = tuple(str(value) for value in patients)
    allowed_patients = set(str(value) for value in (patients or case_ids))
    for patient in sorted(case_ids):
        if patient not in allowed_patients:
            continue
        patient_dir = DATASET_CASES[patient]["mesh_dir"] if INPUT_FORMAT == "vtk" else source_root / patient
        if INPUT_FORMAT == "vtk":
            mesh_dir = patient_dir
            phase_files = sorted(
                (path for path in mesh_dir.glob("frame_*.vtk")),
                key=lambda path: _phase_id(path, "frame_", ".vtk"),
            )
            prefix, suffix = "frame_", ".vtk"
        else:
            suffix = f"-{surface}.obj"
            phase_files = sorted(
                (path for path in patient_dir.iterdir() if path.name.endswith(suffix)),
                key=lambda path: _phase_id(path, "", suffix),
            )
            prefix = ""
        if not phase_files:
            raise ValueError(f"No {INPUT_FORMAT} frames found for {patient}")
        phase_ids = [_phase_id(path, prefix, suffix) for path in phase_files]
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError(f"Duplicate phase IDs for {patient}")
        frame_count = len(phase_files)
        for position, (phase_index, mesh_path) in enumerate(zip(phase_ids, phase_files)):
            if phases is None or phase_index in phases:
                frames.append((patient, phase_index, position, frame_count, mesh_path))
    return frames


def _triangles_from_polydata(mesh: pv.PolyData, path: Path) -> np.ndarray:
    raw = np.asarray(mesh.faces, dtype=np.int64)
    triangles = []
    cursor = 0
    while cursor < len(raw):
        count = int(raw[cursor])
        end = cursor + count + 1
        if count != 3 or end > len(raw):
            raise ValueError(f"VTK mesh must contain only triangular cells: {path}")
        triangles.append(raw[cursor + 1 : end])
        cursor = end
    if cursor != len(raw) or not triangles:
        raise ValueError(f"VTK mesh contains no triangular cells: {path}")
    if len(triangles) != mesh.n_cells:
        raise ValueError(f"VTK mesh contains non-polygon cells: {path}")
    faces = np.asarray(triangles, dtype=np.int64)
    if faces.min() < 0 or faces.max() >= mesh.n_points:
        raise ValueError(f"VTK mesh contains an invalid face index: {path}")
    return faces


def _world_to_normalized(points: np.ndarray, transform: dict) -> np.ndarray:
    homogeneous = np.column_stack(
        (np.asarray(points, dtype=np.float64), np.ones(len(points), dtype=np.float64))
    )
    transformed = homogeneous @ np.asarray(transform["P"], dtype=np.float64).T
    return (transformed[:, :3] + np.asarray(transform["offset"], dtype=np.float64)) * float(transform["scale"])


def _load_labeled_vtk(path: Path, surface: str, transform: dict) -> trimesh.Trimesh:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Empty VTK mesh: {path}")
    if "surface_id" not in mesh.cell_data or "label" not in mesh.cell_data:
        raise ValueError(
            f"{path} must contain cell data 'surface_id' and 'label' "
            "(0/1 and 1/2)"
        )
    surface_id = np.asarray(mesh.cell_data["surface_id"]).reshape(-1)
    label = np.asarray(mesh.cell_data["label"]).reshape(-1)
    if len(surface_id) != mesh.n_cells or len(label) != mesh.n_cells:
        raise ValueError(f"VTK label arrays do not match cell count: {path}")
    if not np.isin(surface_id, (0, 1)).all() or not np.isin(label, (1, 2)).all():
        raise ValueError(f"VTK labels must be surface_id=0/1 and label=1/2: {path}")
    expected = np.where(surface_id == 0, 1, 2)
    if not np.array_equal(label.astype(np.int64), expected.astype(np.int64)):
        raise ValueError(f"Inconsistent surface_id/label values: {path}")
    faces = _triangles_from_polydata(mesh, path)
    selected_value = 0 if surface == "epi" else 1
    selected = surface_id.astype(np.int64) == selected_value
    if not np.any(selected):
        raise ValueError(f"VTK contains no {surface} faces: {path}")
    selected_faces = faces[selected]
    used = np.unique(selected_faces)
    remap = np.full(mesh.n_points, -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    points = _world_to_normalized(np.asarray(mesh.points, dtype=np.float64)[used], transform)
    selected_faces = remap[selected_faces]
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite normalized VTK coordinates: {path}")
    return trimesh.Trimesh(vertices=points, faces=selected_faces, process=False)


def validate_npz(path, split):
    if split != "train":
        return False, "only train NPZ archives are supported"
    expected = TRAIN_KEYS
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                return False, f"keys={sorted(archive.files)}"
            point_keys = ("pos", "neg")
            for key in point_keys:
                values = archive[key]
                if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] == 0:
                    return False, f"{key}.shape={values.shape}"
                if not np.isfinite(values).all():
                    return False, f"{key} contains non-finite values"
            if archive["P"].shape != (4, 4) or archive["Pi"].shape != (4, 4):
                return False, "P/Pi must be 4x4"
            for key in expected - set(point_keys):
                if not np.isfinite(archive[key]).all():
                    return False, f"{key} contains non-finite values"
    except Exception as error:
        return False, repr(error)
    return True, "ok"


def atomic_savez(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=path.stem + ".", suffix=".npz", dir=path.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_seed(path):
    digest = hashlib.sha256(f"{path.parent.name}/{path.name}".encode("utf8")).digest()
    return RANDOM_SEED + int.from_bytes(digest[:4], "little")


def process_frame(split, surface, patient, phase_index, frame_position, frame_count, input_path):
    if split != "train":
        raise ValueError("Only the train split is generated by the public processor")
    output_path = PROCESS_ROOT / f"{patient}_{phase_index:02d}-{surface}.npz"
    if output_path.exists() and not OVERWRITE:
        valid, _ = validate_npz(output_path, split)
        if valid:
            return output_path, "skipped"
    if VALIDATE_ONLY:
        valid, reason = validate_npz(output_path, split)
        if not valid:
            raise ValueError(f"Invalid {output_path}: {reason}")
        return output_path, "validated"

    transform = TRANSFORM_CACHE.get(patient)
    if transform is None:
        transform_path = (
            DATASET_CASES[patient]["transform"]
            if INPUT_FORMAT == "vtk"
            else TRANSFORM_ROOT / f"{patient}.txt"
        )
        transform = load_transform(transform_path)
        TRANSFORM_CACHE[patient] = transform
    time_value = np.float32(frame_position / max(frame_count - 1, 1))
    if INPUT_FORMAT == "vtk":
        mesh = _load_labeled_vtk(input_path, surface, transform)
    else:
        mesh = trimesh.load(input_path, process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

    if split == "train":
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected a triangular training OBJ mesh: {input_path}")
        if len(mesh.faces) == 0:
            raise ValueError(f"Training mesh has no faces: {input_path}")
        # The local sampler uses NumPy's global generator internally.
        np.random.seed(_stable_seed(input_path) % (2**32 - 1))
        points, sdf = sample_sdf_near_surface(
            mesh,
            number_of_points=SDF_SAMPLE_COUNT,
            sampling_type="sphere",
            surface_point_method="sample",
            sign_method="normal",
            scan_count=0,
            scan_resolution=0,
            sample_point_count=SURFACE_SAMPLE_COUNT,
            normal_sample_count=5,
            min_size=0,
            return_gradients=False,
            test_sampling=False,
        )
        positive = np.column_stack((points[sdf > 0], sdf[sdf > 0])).astype(np.float32)
        negative = np.column_stack((points[sdf < 0], sdf[sdf < 0])).astype(np.float32)
        arrays = {"pos": positive, "neg": negative, "t": time_value, **transform}
    atomic_savez(output_path, **arrays)
    valid, reason = validate_npz(output_path, split)
    if not valid:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"Generated invalid archive {output_path}: {reason}")
    return output_path, "written"


def build_manifest(surface):
    manifest = {"train": {"training": {}}, "test": {}}
    for patient, phase_index, _, _, _ in discover_frames(
        TRAIN_MESH_ROOT, surface, TRAIN_CASE_IDS, TRAIN_PHASES
    ):
        sequence_name = f"{patient}-{surface}"
        sequence = manifest["train"]["training"].setdefault(
            sequence_name, {"instance_list": []}
        )
        output_path = PROCESS_ROOT / f"{patient}_{phase_index:02d}-{surface}.npz"
        try:
            manifest_path = output_path.resolve().relative_to(REPO_ROOT.resolve())
        except ValueError:
            manifest_path = output_path.resolve()
        sequence["instance_list"].append(manifest_path.as_posix())
    return manifest


def write_manifests(surfaces=SURFACES):
    TRAIN_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for surface in surfaces:
        destination = TRAIN_DATA_ROOT / f"data_load_{surface}.json"
        destination.write_text(json.dumps(build_manifest(surface), indent=2) + "\n")


def verify_generated_subset(surfaces=SURFACES):
    errors = []
    total = 0
    for surface in surfaces:
        manifest = build_manifest(surface)
        expected_paths = [
            REPO_ROOT / item
            for sequence in manifest["train"]["training"].values()
            for item in sequence["instance_list"]
        ]
        total += len(expected_paths)
        for path in expected_paths:
            valid, reason = validate_npz(path, "train")
            if not valid:
                errors.append(f"{path}: {reason}")
    if errors:
        raise RuntimeError("NPZ validation failed:\n" + "\n".join(errors[:50]))
    return total


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Convert separated OBJ or labeled VTK meshes into 4DMM SDF archives."
    )
    parser.add_argument(
        "--training-data-root", type=Path, default=TRAIN_DATA_ROOT,
        help="training data directory (default: training_data)",
    )
    parser.add_argument(
        "--dataset-root", type=Path,
        help=(
            "labeled VTK dataset root; P.txt is reused when present or "
            "generated from the case Mask when absent"
        ),
    )
    parser.add_argument(
        "--case", dest="case_ids", action="append",
        help="process only this case; repeat for multiple cases",
    )
    parser.add_argument(
        "--phase", dest="phases", type=int, action="append",
        help="process only this zero-based phase; repeat for multiple phases",
    )
    parser.add_argument(
        "--surface", choices=SURFACES, action="append",
        help="process only this surface; repeat to select both",
    )
    parser.add_argument(
        "--reference-phase", type=int, default=0,
        help="reference/ED phase used only when generating a missing P.txt",
    )
    parser.add_argument(
        "--mirror", choices=("auto", "normal", "mirrored"), default="auto",
        help="orientation used only when generating a missing P.txt",
    )
    parser.add_argument(
        "--mirror-fallback", choices=("error", "normal", "mirrored"),
        default="error",
        help="explicit fallback when automatic RV top-view scoring is inconclusive",
    )
    parser.add_argument(
        "--slice-order", choices=("auto", "atob", "btoa"), default="auto",
        help="slice order used only when generating a missing P.txt",
    )
    parser.add_argument(
        "--sdf-samples", type=int, default=SDF_SAMPLE_COUNT,
        help="SDF samples generated per frame",
    )
    parser.add_argument(
        "--surface-samples", type=int, default=SURFACE_SAMPLE_COUNT,
        help="surface samples used by the SDF sampler",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _apply_cli_args(args):
    global TRAIN_DATA_ROOT, TRAIN_MESH_ROOT, TRANSFORM_ROOT, PROCESS_ROOT
    global INPUT_FORMAT, DATASET_ROOT, DATASET_CASES, TRANSFORM_CACHE
    global TRAIN_CASE_IDS, TRAIN_PHASES, OVERWRITE, VALIDATE_ONLY
    global SDF_SAMPLE_COUNT, SURFACE_SAMPLE_COUNT
    global REFERENCE_PHASE, MIRROR, MIRROR_FALLBACK, SLICE_ORDER

    root = args.training_data_root
    if not root.is_absolute():
        root = REPO_ROOT / root
    TRAIN_DATA_ROOT = root.resolve()
    TRAIN_MESH_ROOT = TRAIN_DATA_ROOT / "normalized_meshes"
    TRANSFORM_ROOT = TRAIN_DATA_ROOT / "P"
    PROCESS_ROOT = TRAIN_DATA_ROOT / "processed"
    DATASET_ROOT = args.dataset_root.resolve() if args.dataset_root else None
    INPUT_FORMAT = "vtk" if DATASET_ROOT is not None else "obj"
    DATASET_CASES = {}
    TRANSFORM_CACHE = {}
    if DATASET_ROOT is not None:
        DATASET_CASES = discover_training_cases(DATASET_ROOT)
        if not DATASET_CASES:
            raise ValueError(f"No training cases found under {DATASET_ROOT}")
        TRAIN_MESH_ROOT = DATASET_ROOT
        TRANSFORM_ROOT = DATASET_ROOT
    TRAIN_CASE_IDS = tuple(args.case_ids) if args.case_ids else None
    TRAIN_PHASES = tuple(args.phases) if args.phases else None
    OVERWRITE = args.overwrite
    VALIDATE_ONLY = args.validate_only
    REFERENCE_PHASE = args.reference_phase
    MIRROR = args.mirror
    MIRROR_FALLBACK = args.mirror_fallback
    SLICE_ORDER = args.slice_order
    if args.sdf_samples < 1 or args.surface_samples < 1:
        raise ValueError("sample counts must be positive")
    SDF_SAMPLE_COUNT = args.sdf_samples
    SURFACE_SAMPLE_COUNT = args.surface_samples
    return tuple(dict.fromkeys(args.surface or SURFACES))


def main(argv=None):
    args = _build_parser().parse_args(argv)
    surfaces = _apply_cli_args(args)
    if INPUT_FORMAT == "vtk":
        if not DATASET_ROOT.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {DATASET_ROOT}")
        ensure_dataset_transforms()
    elif not TRAIN_MESH_ROOT.is_dir():
        raise FileNotFoundError(f"Training mesh directory does not exist: {TRAIN_MESH_ROOT}")
    free_gb = shutil.disk_usage(REPO_ROOT).free / 1024**3
    if not VALIDATE_ONLY and free_gb < MIN_FREE_SPACE_GB:
        raise RuntimeError(f"Only {free_gb:.1f} GB free; at least {MIN_FREE_SPACE_GB} GB is required")
    print(f"Repository: {REPO_ROOT}")
    print(f"Free space: {free_gb:.1f} GB; workers: {PREPROCESS_WORKERS}")

    for split in SPLITS:
        source_root = TRAIN_MESH_ROOT
        patients = TRAIN_CASE_IDS
        phases = TRAIN_PHASES
        for surface in surfaces:
            frames = discover_frames(source_root, surface, patients, phases)
            print(f"{split}/{surface}: {len(frames)} frames")
            for index, frame in enumerate(frames, 1):
                output, status = process_frame(split, surface, *frame)
                if index == 1 or index % 25 == 0 or index == len(frames):
                    print(f"[{split}/{surface}] {index}/{len(frames)} {status}: {output.name}", flush=True)

    write_manifests(surfaces)
    total = verify_generated_subset(surfaces)
    print(f"Validation complete: {total} selected NPZ files")


if __name__ == "__main__":
    main()
