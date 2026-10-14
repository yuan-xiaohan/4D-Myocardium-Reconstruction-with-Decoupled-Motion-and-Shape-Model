"""Postprocess output metadata and validation contracts."""

from __future__ import annotations

import numpy as np


EPI_LABEL = 1
ENDO_LABEL = 2
MIN_NORMAL_ORIENTATION_SCORE = 0.10
POSTPROCESS_ALGORITHM_VERSION = 4
POSTPROCESS_ALGORITHM_NAME = "independent_base_zero_level_loops"
SUPPORTED_POSTPROCESS_ALGORITHM_VERSIONS = frozenset({3, 4})
RAW_CONCAT_ALGORITHM_VERSION = 1
RAW_CONCAT_ALGORITHM_NAME = "raw_surface_concat"


class GeometryError(RuntimeError):
    """Raised when a surface violates the postprocess geometry contract."""


def triangle_faces(mesh):
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0 or faces.size % 4:
        raise GeometryError("Expected a non-empty triangular PolyData surface")
    faces = faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise GeometryError("Non-triangular faces remain after triangulation")
    return faces[:, 1:]


def postprocess_algorithm_name(mesh) -> str | None:
    values = mesh.field_data.get("postprocess_algorithm_name")
    if values is None or not len(values):
        return None
    value = np.asarray(values).reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def _surface_from_id(mesh, surface_id):
    ids = np.asarray(mesh.cell_data["surface_id"])
    cell_ids = np.flatnonzero(ids == surface_id)
    if cell_ids.size == 0:
        raise GeometryError(f"Missing surface_id={surface_id}")
    return (
        mesh.extract_cells(cell_ids)
        .extract_surface()
        .triangulate()
        .clean()
    )


def radial_normal_score(mesh, center=None):
    oriented = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
        non_manifold_traversal=True,
    )
    cell_centers = np.asarray(oriented.cell_centers().points)
    normals = np.asarray(oriented.cell_data["Normals"])
    if center is None:
        center = np.asarray(mesh.center)
    radial = cell_centers - np.asarray(center).reshape(1, 3)
    lengths = np.linalg.norm(radial, axis=1)
    valid = lengths > 1.0e-12
    radial[valid] /= lengths[valid, None]
    return float(np.mean(np.sum(normals[valid] * radial[valid], axis=1)))


def validate_combined_mesh(mesh):
    """Validate a successful strict postprocess output."""
    mesh = mesh.extract_surface().triangulate()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise GeometryError("Combined mesh is empty")
    points = np.asarray(mesh.points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise GeometryError("Combined mesh contains NaN or infinite points")
    faces = triangle_faces(mesh)
    if np.any(faces < 0) or np.any(faces >= mesh.n_points):
        raise GeometryError("Combined mesh contains an invalid face index")
    if mesh.n_open_edges != 0:
        raise GeometryError(f"Combined mesh has {mesh.n_open_edges} open edges")
    if not mesh.is_manifold:
        raise GeometryError("Combined mesh is not manifold")
    algorithm_name = postprocess_algorithm_name(mesh)
    if algorithm_name != POSTPROCESS_ALGORITHM_NAME:
        raise GeometryError(
            "Combined mesh has an unknown postprocess algorithm: "
            f"{algorithm_name!r}"
        )
    if "postprocess_algorithm_version" not in mesh.field_data:
        raise GeometryError(
            "Combined mesh was produced by an obsolete postprocess algorithm"
        )
    algorithm_version = int(
        np.asarray(mesh.field_data["postprocess_algorithm_version"])
        .reshape(-1)[0]
    )
    if algorithm_version not in SUPPORTED_POSTPROCESS_ALGORITHM_VERSIONS:
        raise GeometryError(
            "Combined mesh postprocess algorithm version is obsolete: "
            f"found={algorithm_version}, supported="
            f"{sorted(SUPPORTED_POSTPROCESS_ALGORITHM_VERSIONS)}"
        )
    if "surface_id" not in mesh.cell_data:
        raise GeometryError("Combined mesh has no surface_id cell array")
    surface_ids = np.asarray(mesh.cell_data["surface_id"]).astype(int)
    unique_ids = set(int(value) for value in np.unique(surface_ids))
    if unique_ids != {0, 1}:
        raise GeometryError(
            f"Expected surface_id values {{0, 1}}, got {unique_ids}"
        )
    if "label" not in mesh.cell_data:
        raise GeometryError("Combined mesh has no cell label array")
    cell_labels = np.asarray(mesh.cell_data["label"]).astype(int)
    if not np.array_equal(cell_labels, surface_ids + EPI_LABEL):
        raise GeometryError(
            "Cell label is inconsistent with surface_id; expected epi=1 and endo=2"
        )
    if "label" not in mesh.point_data:
        raise GeometryError("Combined mesh has no point label array")
    point_labels = np.asarray(mesh.point_data["label"]).astype(int)
    point_label_values = set(int(value) for value in np.unique(point_labels))
    if point_label_values != {EPI_LABEL, ENDO_LABEL}:
        raise GeometryError(
            f"Expected point label values {{1, 2}}, got {point_label_values}"
        )
    epi = _surface_from_id(mesh, 0)
    endo = _surface_from_id(mesh, 1)
    epi_score = radial_normal_score(epi)
    endo_score = radial_normal_score(endo)
    if epi_score < MIN_NORMAL_ORIENTATION_SCORE:
        raise GeometryError(f"Epi normals are not outward: score={epi_score:.6f}")
    if endo_score > -MIN_NORMAL_ORIENTATION_SCORE:
        raise GeometryError(
            "Endo normals are not directed toward the chamber: "
            f"score={endo_score:.6f}"
        )
    return {
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "finite_points": True,
        "triangular_faces": True,
        "valid_face_indices": True,
        "n_open_edges": int(mesh.n_open_edges),
        "is_manifold": bool(mesh.is_manifold),
        "epi_cells": int(epi.n_cells),
        "endo_cells": int(endo.n_cells),
        "epi_label": EPI_LABEL,
        "endo_label": ENDO_LABEL,
        "epi_outward_score": epi_score,
        "endo_outward_score": endo_score,
    }


def validate_raw_concatenated_mesh(mesh):
    """Validate serialization invariants for a diagnostic raw concatenation."""
    algorithm_name = postprocess_algorithm_name(mesh)
    if algorithm_name != RAW_CONCAT_ALGORITHM_NAME:
        raise GeometryError(
            "Raw concatenated VTK has an unknown algorithm: "
            f"{algorithm_name!r}"
        )
    if "postprocess_algorithm_version" not in mesh.field_data:
        raise GeometryError("Raw concatenated VTK has no algorithm version")
    algorithm_version = int(
        np.asarray(mesh.field_data["postprocess_algorithm_version"])
        .reshape(-1)[0]
    )
    if algorithm_version != RAW_CONCAT_ALGORITHM_VERSION:
        raise GeometryError(
            "Raw concatenated VTK algorithm version is unsupported: "
            f"found={algorithm_version}, expected={RAW_CONCAT_ALGORITHM_VERSION}"
        )
    points = np.asarray(mesh.points, dtype=np.float64)
    faces = triangle_faces(mesh)
    surface_ids = np.asarray(mesh.cell_data.get("surface_id", [])).reshape(-1)
    labels = np.asarray(mesh.cell_data.get("label", [])).reshape(-1)
    point_labels = np.asarray(mesh.point_data.get("label", [])).reshape(-1)
    if len(surface_ids) != mesh.n_cells or set(np.unique(surface_ids)) != {0, 1}:
        raise GeometryError("Raw concatenated VTK has invalid surface_id values")
    if len(labels) != mesh.n_cells or not np.array_equal(
        labels.astype(np.int32), surface_ids.astype(np.int32) + EPI_LABEL
    ):
        raise GeometryError("Raw concatenated VTK has invalid cell labels")
    if len(point_labels) != mesh.n_points or not set(
        np.unique(point_labels).astype(int)
    ).issubset({EPI_LABEL, ENDO_LABEL}):
        raise GeometryError("Raw concatenated VTK has invalid point labels")
    if not np.isfinite(points).all():
        raise GeometryError("Raw concatenated VTK contains non-finite points")
    valid_face_indices = bool(
        faces.size and faces.min() >= 0 and faces.max() < mesh.n_points
    )
    if not valid_face_indices:
        raise GeometryError("Raw concatenated VTK has invalid face indices")
    return {
        "finite_points": True,
        "triangular_faces": True,
        "valid_face_indices": True,
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "n_open_edges": int(mesh.n_open_edges),
        "is_manifold": bool(mesh.is_manifold),
        "epi_cells": int(np.count_nonzero(surface_ids == 0)),
        "endo_cells": int(np.count_nonzero(surface_ids == 1)),
    }


def classify_postprocess_mesh(mesh) -> tuple[str, dict]:
    """Return the persisted output kind and its matching validation report."""
    algorithm = postprocess_algorithm_name(mesh)
    if algorithm == RAW_CONCAT_ALGORITHM_NAME:
        return "raw_concat", validate_raw_concatenated_mesh(mesh)
    return "strict", validate_combined_mesh(mesh)
