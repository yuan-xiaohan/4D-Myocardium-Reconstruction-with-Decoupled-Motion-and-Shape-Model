"""Minimal mask-to-P/point-cloud preprocessing for the 4DMM pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from case_config import CaseConfig
from project_config import REGISTRATION_ASSET


SCALE = 0.014467
OFFSET = np.zeros(3, dtype=np.float64)
ALLOWED_LABELS = {0, 1, 2, 3}
LOW_CONFIDENCE_THRESHOLD = 0.05


@dataclass(frozen=True)
class FrameInfo:
    phase: int
    path: Path
    size: tuple[int, ...]
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]


@dataclass
class SliceContours:
    epi: list[np.ndarray]
    endo: list[np.ndarray]
    rv: list[np.ndarray]
    lv: list[np.ndarray]
    slice_order: str
    slice_order_info: dict


@dataclass(frozen=True)
class RegistrationAssets:
    centerline: np.ndarray
    rv_raw: np.ndarray
    lv_norm: np.ndarray
    rv_norm: np.ndarray
    insertion_curves: tuple[np.ndarray, np.ndarray]
    sha256: str


@dataclass(frozen=True)
class RegistrationResult:
    rigid: np.ndarray
    inverse_rigid: np.ndarray
    scale: float
    offset: np.ndarray
    mode: str
    score: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registration_assets(path: Path = REGISTRATION_ASSET) -> RegistrationAssets:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Registration asset is missing or is a symlink: {path}")
    manifest_path = path.parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Registration asset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get(path.name, {}).get("sha256")
    actual_hash = _sha256(path)
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError(
            f"Registration asset hash mismatch: expected={expected_hash}, actual={actual_hash}"
        )
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "centerline", "rv_raw", "lv_norm", "rv_norm",
            "insertion_curve_0", "insertion_curve_1",
        }
        if set(archive.files) != expected:
            raise ValueError(
                f"Registration asset keys mismatch: {sorted(archive.files)}"
            )
        arrays = {key: np.asarray(archive[key], dtype=np.float64) for key in expected}
    for key, value in arrays.items():
        if value.ndim != 2 or value.shape[1] != 3 or not np.isfinite(value).all():
            raise ValueError(f"Invalid registration array {key}: {value.shape}")
    return RegistrationAssets(
        centerline=arrays["centerline"],
        rv_raw=arrays["rv_raw"],
        lv_norm=arrays["lv_norm"],
        rv_norm=arrays["rv_norm"],
        insertion_curves=(arrays["insertion_curve_0"], arrays["insertion_curve_1"]),
        sha256=actual_hash,
    )


def discover_frames(mask_dir: Path) -> list[FrameInfo]:
    mask_dir = Path(mask_dir)
    indexed = {}
    invalid = []
    for path in sorted(mask_dir.glob("*.nii.gz")):
        stem = path.name[:-7]
        if not stem.isdigit():
            invalid.append(path.name)
            continue
        phase = int(stem)
        if phase in indexed:
            raise ValueError(f"Duplicate phase {phase}: {indexed[phase]}, {path}")
        indexed[phase] = path
    if invalid:
        raise ValueError(f"Non-numeric NIfTI names in {mask_dir}: {invalid}")
    if not indexed:
        raise FileNotFoundError(f"No numeric *.nii.gz frames found in {mask_dir}")
    phases = sorted(indexed)
    if phases != list(range(phases[0], phases[-1] + 1)):
        raise ValueError(f"Non-contiguous phases in {mask_dir}: {phases}")

    frames = []
    for phase in phases:
        path = indexed[phase]
        image = sitk.ReadImage(str(path))
        if image.GetDimension() != 3:
            raise ValueError(f"Expected a 3D mask: {path}")
        frames.append(FrameInfo(
            phase=phase,
            path=path,
            size=image.GetSize(),
            spacing=image.GetSpacing(),
            origin=image.GetOrigin(),
            direction=image.GetDirection(),
        ))
    reference = frames[0]
    for frame in frames[1:]:
        if frame.size != reference.size:
            raise ValueError(f"Frame size mismatch: {reference.path} vs {frame.path}")
        for field in ("spacing", "origin", "direction"):
            if not np.allclose(
                getattr(reference, field), getattr(frame, field), rtol=0.0, atol=1e-6
            ):
                raise ValueError(
                    f"Frame {field} mismatch: {reference.path} vs {frame.path}"
                )
    return frames


def validate_labels(path: Path, required: tuple[int, ...]) -> list[int]:
    array = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    if not np.issubdtype(array.dtype, np.integer):
        if not np.allclose(array, np.round(array), rtol=0.0, atol=0.0):
            raise ValueError(f"Mask contains non-integer labels: {path}")
        array = np.round(array).astype(np.int32)
    labels = set(int(value) for value in np.unique(array))
    unexpected = labels - ALLOWED_LABELS
    if unexpected:
        raise ValueError(f"Unexpected labels in {path}: {sorted(unexpected)}")
    missing = set(required) - labels
    if missing:
        raise ValueError(f"Required labels absent in {path}: {sorted(missing)}")
    return sorted(labels)


def _contour_points(binary_yx: np.ndarray, slice_index: int, image: sitk.Image) -> np.ndarray:
    contours, _ = cv2.findContours(
        binary_yx.T.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    pixel_parts = []
    last_degenerate = False
    for contour in contours:
        points = np.squeeze(contour, axis=1) if contour.ndim == 3 else np.squeeze(contour)
        last_degenerate = points.ndim != 2 or points.shape[0] < 2
        if last_degenerate:
            continue
        pixel_parts.append(points[:, [1, 0]])
    # Preserve the historical contour-cache behavior used by the canonicalizer.
    if contours and last_degenerate:
        return np.empty((0, 3), dtype=np.float64)
    if not pixel_parts:
        return np.empty((0, 3), dtype=np.float64)
    xy = np.vstack(pixel_parts).astype(np.float64)
    indices = np.column_stack((xy, np.full(len(xy), slice_index)))
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    return origin + (indices * spacing) @ direction.T


def _infer_slice_order(mask_zyx: np.ndarray) -> tuple[str, dict]:
    areas, centers, valid_z = [], [], []
    for z_index, plane in enumerate(mask_zyx):
        yy, xx = np.where(plane == 3)
        if len(xx) >= 30:
            areas.append(int(len(xx)))
            centers.append([float(z_index), float(np.mean(yy)), float(np.mean(xx))])
            valid_z.append(z_index)
    if len(areas) < 2:
        raise ValueError("Fewer than two valid LV-cavity slices for slice order")
    areas_array = np.asarray(areas, dtype=np.float64)
    centers_array = np.asarray(centers, dtype=np.float64)
    centered = centers_array - centers_array.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    if np.dot(axis, centers_array[-1] - centers_array[0]) < 0:
        axis *= -1.0
    projection = centered @ axis
    correlation = 0.0
    if np.std(areas_array) > 1e-12 and np.std(projection) > 1e-12:
        correlation = float(np.corrcoef(areas_array, projection)[0, 1])
        correlation = correlation if np.isfinite(correlation) else 0.0
    selected = "btoa" if correlation <= 1e-6 else "atob"
    return selected, {
        "areas": areas,
        "valid_slices": valid_z,
        "correlation": correlation,
        "detected": selected,
    }


def extract_contours(path: Path, slice_order: str) -> SliceContours:
    image = sitk.ReadImage(str(path))
    mask = sitk.GetArrayFromImage(image).astype(np.int32)
    epi, endo, rv, lv = [], [], [], []
    for z_index, plane in enumerate(mask):
        epi_points = _contour_points((plane == 2) * 255, z_index, image)
        endo_points = _contour_points((plane == 3) * 255, z_index, image)
        rv_points = _contour_points((plane == 1) * 255, z_index, image)
        epi.append(epi_points)
        endo.append(endo_points)
        rv.append(rv_points)
        parts = [points for points in (epi_points, endo_points) if len(points)]
        lv.append(np.vstack(parts) if parts else np.empty((0, 3), dtype=np.float64))
    detected, info = _infer_slice_order(mask)
    selected = detected if slice_order == "auto" else slice_order
    info.update({"selected": selected, "source": "auto" if slice_order == "auto" else "override"})
    return SliceContours(epi, endo, rv, lv, selected, info)


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return (homogeneous @ np.asarray(matrix, dtype=np.float64).T)[:, :3]


def world_to_norm(points: np.ndarray, rigid: np.ndarray) -> np.ndarray:
    return (_transform_points(points, rigid) + OFFSET) * SCALE


def _stack(parts: list[np.ndarray]) -> np.ndarray:
    valid = [np.asarray(points) for points in parts if len(points)]
    return np.vstack(valid) if valid else np.empty((0, 3), dtype=np.float64)


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        raise ValueError("At least two points are required to fit a long axis")
    centroid = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - centroid, full_matrices=False)
    axis = vh[0]
    if axis[2] < 0:
        axis *= -1.0
    return axis, centroid


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _constrained_icp_angle(source: np.ndarray, target: np.ndarray, axis: np.ndarray) -> float:
    arbitrary = np.asarray([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.asarray([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, arbitrary)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    projection = np.column_stack((basis_u, basis_v))
    target_2d = target @ projection
    tree = cKDTree(target_2d)
    current = source.copy()
    total = 0.0
    for iteration in range(30):
        source_2d = current @ projection
        distances, indices = tree.query(source_2d)
        threshold = float(np.max(distances) + 0.1) if iteration == 0 else float(np.mean(distances) * 3.0)
        valid = distances < threshold
        if np.count_nonzero(valid) < 10:
            break
        covariance = source_2d[valid].T @ target_2d[indices[valid]]
        u, _, vt = np.linalg.svd(covariance)
        rotation_2d = vt.T @ u.T
        delta = float(np.arctan2(rotation_2d[1, 0], rotation_2d[0, 0]))
        total += delta
        current = current @ _axis_rotation(axis, delta).T
    return total


class RigidRegistrar:
    def __init__(self, assets: RegistrationAssets):
        self.assets = assets
        self.temp_axis, self.temp_centroid = _fit_line(assets.centerline * SCALE)
        self.temp_rv_vec = np.asarray([0.993, 0.120, 0.0], dtype=np.float64)

    def _coarse(self, contours: SliceContours) -> np.ndarray:
        lv_centers, rv_vectors, weights = [], [], []
        for lv_points, rv_points in zip(contours.lv, contours.rv):
            if len(lv_points) <= 10:
                continue
            lv_center = lv_points.mean(axis=0)
            lv_centers.append(lv_center)
            if len(rv_points) > 10:
                rv_vectors.append(rv_points.mean(axis=0) - lv_center)
                weights.append(len(lv_points))
        if len(lv_centers) < 2 or not rv_vectors:
            raise ValueError("Reference frame has insufficient LV/RV contours for registration")
        lv_centers = np.asarray(lv_centers)
        sub_axis, sub_centroid = _fit_line(lv_centers)
        if np.dot(sub_axis, lv_centers[-1] - lv_centers[0]) < 0:
            sub_axis *= -1.0
        if contours.slice_order == "atob":
            sub_axis *= -1.0
        cross = np.cross(sub_axis, self.temp_axis)
        sin_angle = np.linalg.norm(cross)
        cos_angle = float(np.dot(sub_axis, self.temp_axis))
        if sin_angle < 1e-6:
            first = np.eye(3) if cos_angle >= 0 else _axis_rotation(np.asarray([1.0, 0.0, 0.0]), np.pi)
        else:
            x, y, z = cross
            skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
            first = np.eye(3) + skew + skew @ skew * ((1.0 - cos_angle) / sin_angle**2)
        straight = np.asarray(rv_vectors) @ first.T
        projected = straight - np.outer(straight @ self.temp_axis, self.temp_axis)
        mean_rv = np.average(projected, axis=0, weights=np.asarray(weights))
        mean_rv /= np.linalg.norm(mean_rv) + 1e-12
        target = self.temp_rv_vec - np.dot(self.temp_rv_vec, self.temp_axis) * self.temp_axis
        target /= np.linalg.norm(target) + 1e-12
        angle = np.arctan2(
            np.dot(np.cross(mean_rv, target), self.temp_axis), np.dot(mean_rv, target)
        )
        rotation = _axis_rotation(self.temp_axis, angle) @ first
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.temp_centroid - rotation @ sub_centroid
        return matrix

    def _refine(self, contours: SliceContours, coarse: np.ndarray) -> np.ndarray:
        rv_world = _stack([points for points in contours.rv if len(points) > 5])
        lv_centers = np.asarray([points.mean(axis=0) for points in contours.lv if len(points) > 5])
        if len(rv_world) < 10 or len(lv_centers) < 2:
            return coarse
        _, subject_center = _fit_line(lv_centers)
        subject_rv = _transform_points(rv_world, coarse)
        source = subject_rv - subject_rv.mean(axis=0)
        target = self.assets.rv_raw - self.assets.rv_raw.mean(axis=0)
        correction = _axis_rotation(
            self.temp_axis,
            _constrained_icp_angle(source, target, self.temp_axis),
        )
        rotation = correction @ coarse[:3, :3]
        result = np.eye(4)
        result[:3, :3] = rotation
        result[:3, 3] = self.temp_centroid - rotation @ subject_center
        return result

    def _score(self, contours: SliceContours, matrix: np.ndarray) -> float:
        scores = []
        for observed_parts, target in (
            (contours.lv, self.assets.lv_norm),
            (contours.rv, self.assets.rv_norm),
        ):
            observed = world_to_norm(_stack(observed_parts), matrix)
            if not len(observed):
                return float("inf")
            forward = cKDTree(target).query(observed)[0]
            backward = cKDTree(observed).query(target)[0]
            scores.append(float(forward.mean() + backward.mean()))
        return float(np.mean(scores))

    def register(self, contours: SliceContours, mode: str) -> RegistrationResult:
        matrix = self._refine(contours, self._coarse(contours))
        if mode == "mirrored":
            reflection = np.eye(4)
            reflection[1, 1] = -1.0
            reflection[:3, 3] = self.temp_centroid - reflection[:3, :3] @ self.temp_centroid
            matrix = reflection @ matrix
        matrix = np.round(matrix, decimals=6)
        inverse = np.linalg.inv(matrix)
        return RegistrationResult(
            rigid=matrix,
            inverse_rigid=inverse,
            scale=SCALE,
            offset=OFFSET.copy(),
            mode=mode,
            score=self._score(contours, matrix),
        )


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, count = ndi.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask, labels, index=np.arange(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def _farthest_pair(points: np.ndarray) -> np.ndarray | None:
    if len(points) < 2:
        return None
    if len(points) > 512:
        points = points[np.linspace(0, len(points) - 1, 512).astype(int)]
    distances = np.sum((points[:, None] - points[None, :]) ** 2, axis=-1)
    return points[list(np.unravel_index(int(np.argmax(distances)), distances.shape))]


def _physical_points(image: sitk.Image, points_zyx: np.ndarray) -> np.ndarray:
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    origin = np.asarray(image.GetOrigin(), dtype=np.float64)
    xyz = np.asarray(points_zyx, dtype=np.float64)[:, [2, 1, 0]]
    return origin + (xyz * spacing) @ direction.T


def extract_insertion_curves(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    image = sitk.ReadImage(str(path))
    mask = sitk.GetArrayFromImage(image).astype(np.int16)
    pairs, valid_z, dilation_by_slice = [], [], {}
    structure = np.ones((3, 3), dtype=bool)
    for z, plane in enumerate(mask):
        rv = plane == 1
        lvm = plane == 2
        if np.count_nonzero(rv) < 10 or np.count_nonzero(lvm) < 10:
            continue
        boundary = lvm & ~ndi.binary_erosion(lvm, structure=structure)
        expanded = rv.copy()
        pair = None
        for dilation in range(1, 5):
            expanded = ndi.binary_dilation(expanded, structure=structure)
            contact = _largest_component(boundary & expanded)
            pair_indices = _farthest_pair(np.argwhere(contact))
            if pair_indices is not None:
                points = np.c_[np.full(2, z), pair_indices]
                pair = _physical_points(image, points)
                dilation_by_slice[str(z)] = dilation
                break
        if pair is not None:
            pairs.append(pair)
            valid_z.append(z)
    if len(pairs) < 2:
        raise ValueError(f"Only {len(pairs)} valid RV-LV insertion slices in {path}")
    ordered = [pairs[0]]
    for pair in pairs[1:]:
        previous = ordered[-1]
        direct = np.linalg.norm(pair - previous, axis=1).sum()
        swapped = np.linalg.norm(pair[::-1] - previous, axis=1).sum()
        ordered.append(pair if direct <= swapped else pair[::-1])
    curves = np.asarray(ordered, dtype=np.float64)
    return curves[:, 0], curves[:, 1], {
        "valid_slices": valid_z,
        "dilation_by_slice": dilation_by_slice,
    }


def _resample_curve(curve: np.ndarray, count: int = 32) -> np.ndarray:
    distances = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    samples = np.linspace(0.0, distances[-1], count)
    return np.column_stack(
        [np.interp(samples, distances, curve[:, axis]) for axis in range(3)]
    )


def _chamfer(first: np.ndarray, second: np.ndarray) -> float:
    distances = np.sum((first[:, None] - second[None, :]) ** 2, axis=-1)
    return float(np.sqrt(distances.min(axis=1)).mean() + np.sqrt(distances.min(axis=0)).mean())


def choose_registration(
    registrar: RigidRegistrar,
    contours: SliceContours,
    reference_path: Path,
    requested_mode: str,
) -> tuple[RegistrationResult, dict]:
    if requested_mode != "auto":
        selected = registrar.register(contours, requested_mode)
        return selected, {
            "requested": requested_mode,
            "selected": requested_mode,
            "low_confidence": False,
            "source": "case_override",
            "registration_score": selected.score,
        }
    normal = registrar.register(contours, "normal")
    mirrored = registrar.register(contours, "mirrored")
    patient_curves, patient_curve_1, details = extract_insertion_curves(reference_path)
    patient_curves = (patient_curves, patient_curve_1)
    template_curves = registrar.assets.insertion_curves

    def candidate_score(result: RegistrationResult, swap: bool) -> float:
        observed = tuple(
            _resample_curve(world_to_norm(curve, result.rigid)) for curve in patient_curves
        )
        targets = template_curves[::-1] if swap else template_curves
        return float(np.mean([
            _chamfer(observed[index], _resample_curve(targets[index]))
            for index in range(2)
        ]))

    normal_score = candidate_score(normal, False)
    mirrored_score = candidate_score(mirrored, True)
    selected = normal if normal_score < mirrored_score else mirrored
    best, worst = sorted((normal_score, mirrored_score))
    margin = float((worst - best) / max(worst, 1e-8))
    low_confidence = margin < LOW_CONFIDENCE_THRESHOLD
    return selected, {
        "requested": "auto",
        "selected": selected.mode,
        "score_normal": normal_score,
        "score_mirrored": mirrored_score,
        "normalized_margin": margin,
        "threshold": LOW_CONFIDENCE_THRESHOLD,
        "low_confidence": low_confidence,
        "source": "minimum_insertion_curve_score",
        **details,
    }


def phase_mapping(phases: list[int], reference_phase: int) -> list[dict]:
    if reference_phase not in phases:
        raise ValueError(f"Reference phase {reference_phase} is absent: {phases}")
    start = phases.index(reference_phase)
    network_order = phases[start:] + phases[:start]
    denominator = max(len(network_order) - 1, 1)
    return [
        {
            "original_phase": phase,
            "internal_phase": internal,
            "t": float(internal / denominator),
        }
        for internal, phase in enumerate(network_order)
    ]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_point_obj(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Invalid point cloud for {path}: {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite point cloud for {path}")
    _atomic_text(path, "".join(f"v {x:.9f} {y:.9f} {z:.9f}\n" for x, y, z in points))


def _read_point_obj(path: Path) -> np.ndarray:
    points = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("v "):
            points.append([float(value) for value in line.split()[1:4]])
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3 or not len(result):
        raise ValueError(f"Empty or invalid point OBJ: {path}")
    return result


def _load_valid_preprocessed_pair(
    obj_path: Path,
    npz_path: Path,
    expected_t: float,
    transform: dict,
) -> np.ndarray | None:
    """Return existing points only when both atomic products are self-consistent."""
    if not obj_path.is_file() or not npz_path.is_file():
        return None
    try:
        points = _read_point_obj(obj_path)
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"pcd", "t", "P", "Pi", "offset", "scale"}:
                return None
            pcd = np.asarray(archive["pcd"])
            if pcd.shape != (len(points), 4) or not np.isfinite(pcd).all():
                return None
            if not np.allclose(pcd[:, :3], points, rtol=0.0, atol=1e-6):
                return None
            if not np.allclose(pcd[:, 3], 0.0, rtol=0.0, atol=0.0):
                return None
            expected_shapes = {
                "P": (4, 4), "Pi": (4, 4), "offset": (3,), "scale": (1,)
            }
            if any(np.asarray(archive[key]).shape != shape for key, shape in expected_shapes.items()):
                return None
            if not np.isclose(float(np.asarray(archive["t"]).reshape(-1)[0]), expected_t):
                return None
            for key in ("P", "Pi", "offset", "scale"):
                if not np.allclose(
                    np.asarray(archive[key]).reshape(-1),
                    np.asarray(transform[key]).reshape(-1),
                    rtol=0.0,
                    atol=1e-8,
                ):
                    return None
        return points
    except (OSError, ValueError, KeyError, IndexError):
        return None


def _transform_text(p: np.ndarray, pi: np.ndarray, offset: np.ndarray, scale: float) -> str:
    rows = ["P:", *["\t".join(f"{value:.6f}" for value in row) for row in p], "", "Pi:"]
    rows.extend("\t".join(f"{value:.6f}" for value in row) for row in pi)
    rows.extend([
        "", "offset:", "\t".join(f"{value:.6f}" for value in offset),
        "", "scale:", f"{scale:.6f}", "",
    ])
    return "\n".join(rows)


def _read_transform(path: Path) -> dict:
    blocks, current, rows = {}, None, []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith(":"):
            if current is not None:
                blocks[current] = np.asarray(rows, dtype=np.float64)
            current, rows = line[:-1], []
        else:
            rows.append([float(value) for value in line.split()])
    if current is not None:
        blocks[current] = np.asarray(rows, dtype=np.float64)
    if set(blocks) != {"P", "Pi", "offset", "scale"}:
        raise ValueError(f"Invalid transform file: {path}")
    transform = {
        "P": blocks["P"], "Pi": blocks["Pi"],
        "offset": blocks["offset"].reshape(3),
        "scale": float(blocks["scale"].reshape(-1)[0]),
    }
    if transform["P"].shape != (4, 4) or transform["Pi"].shape != (4, 4):
        raise ValueError(f"P and Pi must be 4x4 in {path}")
    if not all(np.isfinite(np.asarray(value)).all() for value in transform.values()):
        raise ValueError(f"Transform contains non-finite values: {path}")
    if transform["scale"] <= 0:
        raise ValueError(f"Transform scale must be positive: {path}")
    inverse_error = float(np.max(np.abs(transform["P"] @ transform["Pi"] - np.eye(4))))
    if inverse_error >= 1e-5:
        raise ValueError(f"P/Pi inverse error is {inverse_error} in {path}")
    return transform


def _prepared_sources(config: CaseConfig) -> tuple[list[int], dict, dict[tuple[int, str], Path]]:
    root = config.prepared_root
    point_root = root / "input-points_norm_uv" / config.case_id
    if not point_root.is_dir():
        point_root = root / config.case_id if (root / config.case_id).is_dir() else root
    p_path = root / "P" / "P_lv" / f"{config.case_id}.txt"
    if not p_path.is_file():
        p_path = root / f"{config.case_id}.txt"
    transform = _read_transform(p_path)
    sources = {}
    for surface in ("endo", "epi"):
        for path in point_root.glob(f"*-{surface}.obj"):
            phase = int(path.name[: -len(f"-{surface}.obj")])
            sources[(phase, surface)] = path
    phases = sorted({phase for phase, _ in sources})
    if not phases or any((phase, surface) not in sources for phase in phases for surface in ("endo", "epi")):
        raise ValueError(f"Incomplete prepared point clouds in {point_root}")
    return phases, transform, sources


def run_preprocessing(config: CaseConfig, asset_path: Path = REGISTRATION_ASSET) -> dict:
    preprocess_root = config.output_dir / "artifacts" / "preprocess"
    points_root = preprocess_root / "input-points_norm_uv" / config.case_id
    p_path = preprocess_root / "P" / "P_lv" / f"{config.case_id}.txt"
    npz_root = preprocess_root / "npz"

    labels_by_phase = {}
    orientation = {"requested": "prepared", "selected": "prepared", "low_confidence": False}
    slice_order_info = None
    asset_hash = None
    if config.preprocess_mode == "mask":
        frames = discover_frames(config.input_mask_dir)
        phases = [frame.phase for frame in frames]
        frame_by_phase = {frame.phase: frame for frame in frames}
        if config.reference_phase not in frame_by_phase:
            raise ValueError(f"Reference phase {config.reference_phase} is absent")
        for frame in frames:
            required = (1, 2, 3) if frame.phase == config.reference_phase else (2, 3)
            labels_by_phase[str(frame.phase)] = validate_labels(frame.path, required)
        assets = load_registration_assets(asset_path)
        asset_hash = assets.sha256
        reference_contours = extract_contours(
            frame_by_phase[config.reference_phase].path, config.slice_order
        )
        registrar = RigidRegistrar(assets)
        registration, orientation = choose_registration(
            registrar,
            reference_contours,
            frame_by_phase[config.reference_phase].path,
            config.mirror,
        )
        # The historical data contract stores P at six decimal places.  Pi
        # must be derived from that exact persisted P, not from its unrounded
        # optimizer result, so the text and NPZ representations stay aligned.
        persisted_p = np.round(np.asarray(registration.rigid, dtype=np.float64), 6)
        transform = {
            "P": persisted_p,
            "Pi": np.linalg.inv(persisted_p),
            "offset": registration.offset,
            "scale": registration.scale,
        }
        slice_order_info = reference_contours.slice_order_info
        sources = None
    else:
        phases, transform, sources = _prepared_sources(config)
        if config.reference_phase not in phases:
            raise ValueError(f"Reference phase {config.reference_phase} is absent")

    mapping = phase_mapping(phases, config.reference_phase)
    _atomic_text(
        p_path,
        _transform_text(
            transform["P"], transform["Pi"], transform["offset"], transform["scale"]
        ),
    )
    point_counts = {}
    manifests = {}
    for surface in ("endo", "epi"):
        instances = []
        for item in mapping:
            original_phase = item["original_phase"]
            obj_path = points_root / f"{original_phase:02d}-{surface}.obj"
            npz_path = npz_root / f"{config.case_id}_{original_phase:02d}-{surface}.npz"
            points = None
            if config.skip_existing:
                points = _load_valid_preprocessed_pair(
                    obj_path, npz_path, float(item["t"]), transform
                )
            if points is None:
                if config.preprocess_mode == "mask":
                    frame = frame_by_phase[original_phase]
                    contours = extract_contours(frame.path, config.slice_order)
                    world = _stack(contours.endo if surface == "endo" else contours.epi)
                    points = world_to_norm(world, transform["P"])
                else:
                    points = _read_point_obj(sources[(original_phase, surface)])
                _write_point_obj(obj_path, points)
                pcd = np.column_stack(
                    (points.astype(np.float32), np.zeros(len(points), dtype=np.float32))
                )
                _atomic_npz(
                    npz_path,
                    pcd=pcd,
                    t=np.float32(item["t"]),
                    P=np.asarray(transform["P"], dtype=np.float64),
                    Pi=np.asarray(transform["Pi"], dtype=np.float64),
                    offset=np.asarray(transform["offset"], dtype=np.float64),
                    scale=np.asarray([transform["scale"]], dtype=np.float64),
                )
            instances.append(str(npz_path.resolve()))
            point_counts[f"{original_phase:02d}-{surface}"] = int(len(points))
        manifest = {
            "train": {"case": {}},
            "test": {"case": {f"{config.case_id}-{surface}": {"instance_list": instances}}},
        }
        manifest_path = preprocess_root / f"data_load_{surface}.json"
        _atomic_json(manifest_path, manifest)
        manifests[surface] = str(manifest_path)

    inverse_error = float(
        np.max(np.abs(np.asarray(transform["P"]) @ np.asarray(transform["Pi"]) - np.eye(4)))
    )
    report = {
        "case_id": config.case_id,
        "preprocess_mode": config.preprocess_mode,
        "reference_phase": config.reference_phase,
        "original_phases": phases,
        "phase_mapping": mapping,
        "labels": labels_by_phase,
        "slice_order": slice_order_info,
        "orientation": orientation,
        "registration_asset_sha256": asset_hash,
        "transform": {
            "P": np.asarray(transform["P"]).tolist(),
            "Pi": np.asarray(transform["Pi"]).tolist(),
            "offset": np.asarray(transform["offset"]).tolist(),
            "scale": float(transform["scale"]),
            "inverse_error": inverse_error,
            "determinant": float(np.linalg.det(np.asarray(transform["P"])[:3, :3])),
        },
        "point_counts": point_counts,
        "manifests": manifests,
        "all_passed": inverse_error < 1e-5 and all(count > 0 for count in point_counts.values()),
    }
    report_path = preprocess_root / "preprocess_manifest.json"
    _atomic_json(report_path, report)
    if orientation.get("low_confidence"):
        print(
            f"[warning] {config.case_id}: auto mirror is low confidence; "
            f"selected {orientation['selected']} by minimum score",
            flush=True,
        )
    if not report["all_passed"]:
        raise RuntimeError(f"Preprocessing validation failed: {report_path}")
    return report


def load_preprocess_report(config: CaseConfig) -> dict:
    path = config.output_dir / "artifacts" / "preprocess" / "preprocess_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Preprocess report is missing: {path}")
    with path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("case_id") != config.case_id or not report.get("all_passed"):
        raise ValueError(f"Invalid preprocess report: {path}")
    return report
