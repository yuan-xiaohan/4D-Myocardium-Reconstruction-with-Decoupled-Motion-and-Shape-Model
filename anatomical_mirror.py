"""RV top-view automatic mirror inference with confidence reporting."""

from __future__ import annotations

import numpy as np


AUTO_MIRROR_CONFIDENCE_THRESHOLD = 0.05
AUTO_MIRROR_TEMPLATE_BINS = 12
AUTO_MIRROR_MIN_RV_SLICES = 2
AUTO_MIRROR_SOURCE = "rv_topview_auto_v1"

NORM_TOP_CAMERA_POSITION = [
    (0.6206287314762191, -0.8323725809982682, -5.525099835545793),
    (0.3755530905565364, -0.10349126494400263, 0.07629029356987682),
    (-0.1802627417710753, 0.9743551637668214, -0.13467501167664836),
]

def _topview_basis() -> np.ndarray:
    """Return the two screen axes of the canonical normalized top-view camera."""
    position, focal_point, view_up = [
        np.asarray(item, dtype=np.float64) for item in NORM_TOP_CAMERA_POSITION
    ]
    direction = focal_point - position
    direction /= np.linalg.norm(direction)
    view_up = view_up - direction * np.dot(view_up, direction)
    view_up /= np.linalg.norm(view_up)
    screen_right = np.cross(direction, view_up)
    screen_right /= np.linalg.norm(screen_right)
    return np.column_stack([screen_right, view_up])


def _template_trajectory(registrar, basis: np.ndarray, bins: int) -> np.ndarray:
    template = np.asarray(registrar.assets.rv_norm, dtype=np.float64)
    axis = np.asarray(registrar.temp_axis, dtype=np.float64)
    centroid = np.asarray(registrar.temp_centroid, dtype=np.float64)
    longitudinal = (template - centroid) @ axis
    edges = np.linspace(float(np.min(longitudinal)), float(np.max(longitudinal)), bins + 1)
    points = []
    for index in range(bins):
        selected = template[
            (longitudinal >= edges[index])
            & (longitudinal <= edges[index + 1] if index == bins - 1
               else longitudinal < edges[index + 1])
        ]
        if len(selected):
            points.append(np.mean(selected, axis=0) @ basis)
    if len(points) < 2:
        raise ValueError("Template RV does not provide two valid top-view bins")
    return np.asarray(points, dtype=np.float64)


def _resample_path(path: np.ndarray, count: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if len(path) == count:
        return path
    if len(path) < AUTO_MIRROR_MIN_RV_SLICES:
        raise ValueError(
            f"Only {len(path)} RV-bearing slices; require at least "
            f"{AUTO_MIRROR_MIN_RV_SLICES}"
        )
    positions = np.linspace(0.0, float(len(path) - 1), count)
    return np.column_stack([
        np.interp(positions, np.arange(len(path), dtype=np.float64), path[:, axis])
        for axis in range(path.shape[1])
    ])


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return (homogeneous @ np.asarray(matrix, dtype=np.float64).T)[:, :3]


def _world_to_norm(points: np.ndarray, registration) -> np.ndarray:
    transformed = _transform_points(points, registration.rigid)
    return (transformed + np.asarray(registration.offset, dtype=np.float64)) * float(
        registration.scale
    )


def _patient_trajectory(contours, registration, basis: np.ndarray,
                        axis: np.ndarray, template_count: int) -> tuple[np.ndarray, int]:
    centroids = []
    for points in contours.rv:
        if len(points) == 0:
            continue
        center = np.mean(_world_to_norm(points, registration), axis=0)
        centroids.append(np.asarray([*(center @ basis), float(center @ axis)], dtype=np.float64))
    if len(centroids) < AUTO_MIRROR_MIN_RV_SLICES:
        raise ValueError(
            f"Only {len(centroids)} RV-bearing slices; require at least "
            f"{AUTO_MIRROR_MIN_RV_SLICES}"
        )
    ordered = np.asarray(centroids, dtype=np.float64)
    ordered = ordered[np.argsort(ordered[:, 2])]
    return _resample_path(ordered[:, :2], template_count), len(ordered)


def _trajectory_errors(patient: np.ndarray, template: np.ndarray) -> dict[str, float]:
    patient = np.asarray(patient, dtype=np.float64)
    template = np.asarray(template, dtype=np.float64)
    raw = float(np.mean(np.linalg.norm(patient - template, axis=1)))
    patient_centered = patient - np.mean(patient, axis=0)
    template_centered = template - np.mean(template, axis=0)
    shape = float(np.mean(np.linalg.norm(patient_centered - template_centered, axis=1)))
    denominator = float(np.sum(patient_centered * patient_centered))
    scalar = (
        float(np.sum(patient_centered * template_centered) / denominator)
        if denominator > 1e-12 else 1.0
    )
    scale_fit = float(np.mean(
        np.linalg.norm(scalar * patient_centered - template_centered, axis=1)
    ))
    return {
        "projection_error": raw,
        "shape_error": shape,
        "scale_fit_error": scale_fit,
    }


def score_topview_candidates(contours, registrar,
                             template_bins: int = AUTO_MIRROR_TEMPLATE_BINS) -> dict:
    """Score normal and mirrored rigid candidates using projected RV shape."""
    if template_bins < 2:
        raise ValueError("template_bins must be at least 2")
    basis = _topview_basis()
    template = _template_trajectory(registrar, basis, template_bins)
    axis = np.asarray(registrar.temp_axis, dtype=np.float64)
    candidates = {}
    valid_slices = None
    for mode in ("normal", "mirrored"):
        registration = registrar.register(contours, mode=mode)
        trajectory, candidate_slices = _patient_trajectory(
            contours, registration, basis, axis, len(template)
        )
        if valid_slices is None:
            valid_slices = candidate_slices
        candidates[mode] = {
            **_trajectory_errors(trajectory, template),
            "registration_score": float(registration.score),
        }
    return {
        "valid_rv_slices": int(valid_slices or 0),
        "template_bins": int(template_bins),
        "candidates": candidates,
    }


def infer_auto_mirror(
    contours,
    registrar,
    confidence_threshold: float = AUTO_MIRROR_CONFIDENCE_THRESHOLD,
    template_bins: int = AUTO_MIRROR_TEMPLATE_BINS,
) -> dict:
    """Return the production RV top-view mirror decision.

    Lower centered trajectory-shape error wins.  Confidence is the normalized
    margin between the two candidate errors.  Rejected decisions retain the
    lower-scoring candidate for QC but require an explicit caller fallback.
    """
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    try:
        scoring = score_topview_candidates(contours, registrar, template_bins=template_bins)
    except ValueError as error:
        return {
            "status": "insufficient_evidence",
            "accepted": False,
            "mirror": None,
            "confidence": None,
            "normalized_margin": None,
            "threshold": float(confidence_threshold),
            "template_bins": int(template_bins),
            "min_rv_slices": AUTO_MIRROR_MIN_RV_SLICES,
            "camera": "legacy_norm_top",
            "metric": "mean_centered_rv_centroid_trajectory_distance",
            "source": AUTO_MIRROR_SOURCE,
            "reason": str(error),
        }

    normal_score = float(scoring["candidates"]["normal"]["shape_error"])
    mirrored_score = float(scoring["candidates"]["mirrored"]["shape_error"])
    selected = "normal" if normal_score <= mirrored_score else "mirrored"
    worse = max(normal_score, mirrored_score)
    margin = abs(normal_score - mirrored_score) / max(worse, 1e-12)
    accepted = bool(margin >= confidence_threshold)
    return {
        "status": "accepted" if accepted else "low_confidence",
        "accepted": accepted,
        "mirror": selected,
        "confidence": float(margin),
        "normalized_margin": float(margin),
        "threshold": float(confidence_threshold),
        "score_normal": normal_score,
        "score_mirrored": mirrored_score,
        "valid_rv_slices": scoring["valid_rv_slices"],
        "template_bins": scoring["template_bins"],
        "min_rv_slices": AUTO_MIRROR_MIN_RV_SLICES,
        "camera": "legacy_norm_top",
        "metric": "mean_centered_rv_centroid_trajectory_distance",
        "candidates": scoring["candidates"],
        "source": AUTO_MIRROR_SOURCE,
        "reason": None if accepted else (
            f"Normalized score margin {margin:.6f} is below the "
            f"{confidence_threshold:.6f} threshold"
        ),
    }
