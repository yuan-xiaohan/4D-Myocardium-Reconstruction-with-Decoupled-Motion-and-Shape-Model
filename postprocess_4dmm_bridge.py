from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


# Intentional seam vertices are exact coordinate copies. Keep the merge
# tolerance well below the narrowest observed epi/endo bridge width so a
# near-coincident source-boundary vertex is not collapsed into the midpoint
# loop and does not create a four-edge hole.
POINT_MERGE_TOLERANCE = 1.0e-10
CONTOUR_MAP_TOLERANCE_RATIO = 1.0e-6
MAX_SECONDARY_CONTOUR_LENGTH_RATIO = 0.25
MAX_ARTIFACT_AREA_RATIO = 0.005
MAX_ARTIFACT_COMPONENTS = 12
MIN_BOUNDARY_TOUCH_FRACTION = 0.95
MAX_NONBOUNDARY_TOUCH_FRACTION = 0.05
MIN_NORMAL_ORIENTATION_SCORE = 0.10
EPI_LABEL = 1
ENDO_LABEL = 2
APPROXIMATE_GAP95_EDGE_RATIO = 3.0
APPROXIMATE_GAP_MAX_EDGE_RATIO = 6.0
INDEPENDENT_GAP95_EDGE_RATIO = 3.0
INDEPENDENT_GAP_MAX_EDGE_RATIO = 8.0
POSTPROCESS_ALGORITHM_VERSION = 3
POSTPROCESS_ALGORITHM_NAME = "independent_base_zero_level_loops"


class GeometryError(RuntimeError):
    pass


def _load_surface(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    mesh = pv.read(str(path)).extract_surface().triangulate().clean()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise GeometryError(f"Empty mesh: {path}")
    return mesh


def _triangle_faces(mesh):
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if faces.size == 0 or faces.size % 4:
        raise GeometryError("Expected a non-empty triangular PolyData surface")
    faces = faces.reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise GeometryError("Non-triangular faces remain after triangulation")
    return faces[:, 1:]


def _line_segments(polyline):
    lines = np.asarray(polyline.lines, dtype=np.int64)
    segments = []
    cursor = 0
    while cursor < lines.size:
        count = int(lines[cursor])
        point_ids = lines[cursor + 1 : cursor + 1 + count]
        segments.extend(
            (int(point_ids[index]), int(point_ids[index + 1]))
            for index in range(count - 1)
        )
        cursor += count + 1
    if cursor != lines.size or not segments:
        raise GeometryError("Intersection did not produce valid line segments")
    return segments


def _extract_main_closed_contour(intersection):
    segments = _line_segments(intersection)
    adjacency = [[] for _ in range(intersection.n_points)]
    incident_segments = [[] for _ in range(intersection.n_points)]
    for segment_id, (first, second) in enumerate(segments):
        adjacency[first].append(second)
        adjacency[second].append(first)
        incident_segments[first].append(segment_id)
        incident_segments[second].append(segment_id)

    visited = np.zeros(intersection.n_points, dtype=bool)
    components = []
    for start in range(intersection.n_points):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        point_ids = []
        segment_ids = set()
        while stack:
            current = stack.pop()
            point_ids.append(current)
            segment_ids.update(incident_segments[current])
            for neighbor in adjacency[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)

        degrees = [len(adjacency[point_id]) for point_id in point_ids]
        if not all(degree == 2 for degree in degrees):
            raise GeometryError(
                "Intersection contains a non-closed component: "
                f"degree_range=({min(degrees)}, {max(degrees)})"
            )
        length = float(
            sum(
                np.linalg.norm(
                    intersection.points[segments[segment_id][0]]
                    - intersection.points[segments[segment_id][1]]
                )
                for segment_id in segment_ids
            )
        )
        components.append(
            {
                "point_ids": point_ids,
                "segment_ids": sorted(segment_ids),
                "length": length,
            }
        )

    components.sort(key=lambda item: item["length"], reverse=True)
    main = components[0]
    secondary_length = float(
        sum(component["length"] for component in components[1:])
    )
    secondary_ratio = (
        secondary_length / main["length"] if main["length"] > 0 else np.inf
    )
    if secondary_ratio > MAX_SECONDARY_CONTOUR_LENGTH_RATIO:
        raise GeometryError(
            "Secondary intersection contours are too large: "
            f"ratio={secondary_ratio:.6f}, "
            f"limit={MAX_SECONDARY_CONTOUR_LENGTH_RATIO:.6f}"
        )

    old_to_new = {
        old_point_id: new_point_id
        for new_point_id, old_point_id in enumerate(main["point_ids"])
    }
    contour = pv.PolyData(
        np.asarray(intersection.points)[main["point_ids"]].copy()
    )
    contour.lines = np.asarray(
        [
            [
                2,
                old_to_new[segments[segment_id][0]],
                old_to_new[segments[segment_id][1]],
            ]
            for segment_id in main["segment_ids"]
        ],
        dtype=np.int64,
    )
    diagnostics = {
        "intersection_component_count": len(components),
        "main_contour_points": int(contour.n_points),
        "main_contour_segments": int(len(main["segment_ids"])),
        "main_contour_length": float(main["length"]),
        "secondary_contour_length_ratio": float(secondary_ratio),
    }
    return contour, diagnostics


def _partition_surface_by_contour(split_surface, contour):
    split_surface = split_surface.extract_surface().triangulate()
    distances, mapped_point_ids = cKDTree(
        np.asarray(split_surface.points)
    ).query(np.asarray(contour.points), k=1)
    tolerance = max(
        float(split_surface.length) * CONTOUR_MAP_TOLERANCE_RATIO,
        1.0e-10,
    )
    if float(distances.max()) > tolerance:
        raise GeometryError(
            "Cannot map the contour onto the split surface: "
            f"max_distance={float(distances.max()):.8g}, "
            f"tolerance={tolerance:.8g}"
        )

    blocked_edges = {
        tuple(
            sorted(
                (
                    int(mapped_point_ids[first]),
                    int(mapped_point_ids[second]),
                )
            )
        )
        for first, second in _line_segments(contour)
    }
    faces = _triangle_faces(split_surface)
    edge_to_faces = defaultdict(list)
    for face_id, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_to_faces[
                tuple(sorted((int(first), int(second))))
            ].append(face_id)

    missing_edges = blocked_edges.difference(edge_to_faces)
    invalid_edges = [
        edge
        for edge in blocked_edges
        if edge in edge_to_faces and len(edge_to_faces[edge]) != 2
    ]
    if missing_edges or invalid_edges:
        raise GeometryError(
            "The main contour is not a valid two-sided edge loop on the "
            f"split surface: missing={len(missing_edges)}, "
            f"invalid={len(invalid_edges)}"
        )

    face_neighbors = [[] for _ in range(faces.shape[0])]
    for edge, face_ids in edge_to_faces.items():
        if edge in blocked_edges or len(face_ids) != 2:
            continue
        first, second = face_ids
        face_neighbors[first].append(second)
        face_neighbors[second].append(first)

    visited = np.zeros(faces.shape[0], dtype=bool)
    components = []
    for start in range(faces.shape[0]):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        face_ids = []
        while stack:
            current = stack.pop()
            face_ids.append(current)
            for neighbor in face_neighbors[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(np.asarray(face_ids, dtype=np.int64))

    # vtkPolyData cell IDs are ordered as verts, lines, strips, then polys.
    polygon_cell_offset = (
        split_surface.n_verts
        + split_surface.n_lines
        + split_surface.n_strips
    )
    parts = [
        split_surface.extract_cells(face_ids + polygon_cell_offset)
        .extract_surface()
        .triangulate()
        .clean()
        for face_ids in components
    ]
    parts = [part for part in parts if part.n_cells]
    parts.sort(key=lambda part: float(part.area), reverse=True)
    if len(parts) < 2:
        raise GeometryError(
            f"Contour split produced only {len(parts)} surface component(s)"
        )
    return parts


def _original_boundary_points(mesh):
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    if boundary.n_points == 0:
        raise GeometryError("The original endo surface has no open boundary")
    return np.asarray(boundary.points)


def _select_primary_parts(epi_parts, endo_parts, endo_boundary_points):
    epi_total_area = float(sum(part.area for part in epi_parts))
    endo_total_area = float(sum(part.area for part in endo_parts))
    epi_artifact_ratio = float(
        sum(part.area for part in epi_parts[2:]) / epi_total_area
    )
    endo_artifact_ratio = float(
        sum(part.area for part in endo_parts[2:]) / endo_total_area
    )
    if epi_artifact_ratio > MAX_ARTIFACT_AREA_RATIO:
        raise GeometryError(
            "Epi split contains non-negligible extra components: "
            f"area_ratio={epi_artifact_ratio:.6f}"
        )
    if endo_artifact_ratio > MAX_ARTIFACT_AREA_RATIO:
        raise GeometryError(
            "Endo split contains non-negligible extra components: "
            f"area_ratio={endo_artifact_ratio:.6f}"
        )

    artifact_count = len(epi_parts[2:]) + len(endo_parts[2:])
    if artifact_count > MAX_ARTIFACT_COMPONENTS:
        raise GeometryError(
            "Too many numerical artifact components for deterministic "
            f"repair: count={artifact_count}"
        )

    epi_kept = epi_parts[0]
    endo_primary = endo_parts[:2]
    boundary_tolerance = max(
        max(float(part.length) for part in endo_primary)
        * CONTOUR_MAP_TOLERANCE_RATIO,
        1.0e-10,
    )
    touch_fractions = []
    mean_distances = []
    for part in endo_primary:
        distances = cKDTree(np.asarray(part.points)).query(
            endo_boundary_points, k=1
        )[0]
        touch_fractions.append(
            float(np.mean(distances <= boundary_tolerance))
        )
        mean_distances.append(float(np.mean(distances)))

    endo_removed_index = int(np.argmax(touch_fractions))
    endo_kept_index = 1 - endo_removed_index
    if (
        touch_fractions[endo_removed_index]
        < MIN_BOUNDARY_TOUCH_FRACTION
    ):
        raise GeometryError(
            "Neither dominant endo component contains the original open "
            f"boundary: touch_fractions={touch_fractions}"
        )
    if (
        touch_fractions[endo_kept_index]
        > MAX_NONBOUNDARY_TOUCH_FRACTION
    ):
        raise GeometryError(
            "Both dominant endo components touch the original open "
            f"boundary: touch_fractions={touch_fractions}"
        )

    return {
        "epi_kept": epi_kept,
        "endo_kept": endo_primary[endo_kept_index],
        "epi_artifacts": epi_parts[2:],
        "endo_artifacts": endo_parts[2:],
        "diagnostics": {
            "epi_component_count": len(epi_parts),
            "endo_component_count": len(endo_parts),
            "epi_artifact_area_ratio": epi_artifact_ratio,
            "endo_artifact_area_ratio": endo_artifact_ratio,
            "endo_boundary_touch_fractions": touch_fractions,
            "endo_boundary_mean_distances": mean_distances,
        },
    }


def _merge_mesh_collection(base, extras, include_bits):
    result = base.copy()
    for mesh, include in zip(extras, include_bits):
        if not include:
            continue
        result = (
            result.merge(
                mesh,
                merge_points=True,
                tolerance=POINT_MERGE_TOLERANCE,
            )
            .extract_surface()
            .triangulate()
            .clean(
                point_merging=True,
                tolerance=POINT_MERGE_TOLERANCE,
                absolute=True,
            )
        )
    return result


def _merge_two_surfaces(epi, endo):
    return (
        epi.merge(
            endo,
            merge_points=True,
            tolerance=POINT_MERGE_TOLERANCE,
        )
        .extract_surface()
        .triangulate()
        .clean(
            point_merging=True,
            tolerance=POINT_MERGE_TOLERANCE,
            absolute=True,
        )
    )


def _assign_artifacts_for_closed_merge(selection):
    epi_artifacts = selection["epi_artifacts"]
    endo_artifacts = selection["endo_artifacts"]
    candidates = []
    for epi_bits in product((0, 1), repeat=len(epi_artifacts)):
        epi_candidate = _merge_mesh_collection(
            selection["epi_kept"], epi_artifacts, epi_bits
        )
        for endo_bits in product((0, 1), repeat=len(endo_artifacts)):
            endo_candidate = _merge_mesh_collection(
                selection["endo_kept"], endo_artifacts, endo_bits
            )
            combined = _merge_two_surfaces(epi_candidate, endo_candidate)
            if combined.n_open_edges != 0 or not combined.is_manifold:
                continue
            added_area = float(
                sum(
                    mesh.area
                    for mesh, include in zip(epi_artifacts, epi_bits)
                    if include
                )
                + sum(
                    mesh.area
                    for mesh, include in zip(endo_artifacts, endo_bits)
                    if include
                )
            )
            candidates.append(
                (
                    added_area,
                    tuple(epi_bits),
                    tuple(endo_bits),
                    epi_candidate,
                    endo_candidate,
                )
            )

    if not candidates:
        raise GeometryError(
            "No assignment of numerical fragments produces a closed, "
            "manifold epi/endo merge"
        )
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    added_area, epi_bits, endo_bits, epi_kept, endo_kept = candidates[0]
    diagnostics = {
        "artifact_added_area": added_area,
        "epi_artifact_keep_bits": list(epi_bits),
        "endo_artifact_keep_bits": list(endo_bits),
        "valid_artifact_assignments": len(candidates),
    }
    return epi_kept, endo_kept, diagnostics


def _fit_contour_plane(contour):
    points = np.asarray(contour.points, dtype=np.float64)
    origin = points.mean(axis=0)
    _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
    normal = vh[-1]
    normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
    if normal[2] < 0:
        normal = -normal
    residuals = np.abs((points - origin) @ normal)
    projected = points - np.outer((points - origin) @ normal, normal)
    return origin, normal, projected, {
        "contour_plane_residual_mean": float(np.mean(residuals)),
        "contour_plane_residual_max": float(np.max(residuals)),
    }


def _surface_components(mesh):
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    # vtkClipPolyData can return inconsistent auxiliary cell arrays on
    # difficult cases. Calling vtkCleanPolyData on that object may segfault;
    # rebuild the largest polygon component below instead.
    mesh = mesh.triangulate()
    if mesh.n_cells == 0:
        raise GeometryError("Plane clipping produced an empty surface")
    faces = _triangle_faces(mesh)
    edge_to_faces = defaultdict(list)
    for face_id, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge_to_faces[
                tuple(sorted((int(first), int(second))))
            ].append(face_id)
    neighbors = [[] for _ in range(len(faces))]
    for face_ids in edge_to_faces.values():
        if len(face_ids) == 2:
            first, second = face_ids
            neighbors[first].append(second)
            neighbors[second].append(first)

    visited = np.zeros(len(faces), dtype=bool)
    components = []
    for start in range(len(faces)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(np.asarray(component, dtype=np.int64))
    result = []
    for face_ids in components:
        component_faces = faces[face_ids]
        used_points = np.unique(component_faces)
        old_to_new = np.full(mesh.n_points, -1, dtype=np.int64)
        old_to_new[used_points] = np.arange(
            len(used_points), dtype=np.int64
        )
        remapped_faces = old_to_new[component_faces]
        faces_pv = np.c_[
            np.full(remapped_faces.shape[0], 3, dtype=np.int64),
            remapped_faces,
        ].reshape(-1)
        result.append(
            pv.PolyData(
                np.asarray(mesh.points)[used_points].copy(), faces_pv
            )
        )
    result.sort(key=lambda component: float(component.area), reverse=True)
    return result


def _largest_surface_component(mesh):
    return _surface_components(mesh)[0]


def _plane_clip_candidates(mesh, origin, normal):
    return [
        _largest_surface_component(
            mesh.clip(normal=normal, origin=origin, invert=invert)
        )
        for invert in (False, True)
    ]


def _select_approximate_clipped_surfaces(
    raw_epi,
    raw_endo,
    plane_origin,
    plane_normal,
):
    epi_candidates = _plane_clip_candidates(
        raw_epi, plane_origin, plane_normal
    )
    epi_kept_index = int(
        np.argmax([float(candidate.area) for candidate in epi_candidates])
    )
    epi_kept = epi_candidates[epi_kept_index]

    boundary_points = _original_boundary_points(raw_endo)
    endo_candidates = _plane_clip_candidates(
        raw_endo, plane_origin, plane_normal
    )
    endo_boundary_mean_distances = [
        float(
            np.mean(
                cKDTree(np.asarray(candidate.points)).query(
                    boundary_points, k=1
                )[0]
            )
        )
        for candidate in endo_candidates
    ]
    endo_kept_index = int(np.argmax(endo_boundary_mean_distances))
    endo_kept = endo_candidates[endo_kept_index]
    return epi_kept, endo_kept, {
        "approximate_epi_kept_clip_index": epi_kept_index,
        "approximate_endo_kept_clip_index": endo_kept_index,
        "approximate_endo_boundary_mean_distances": (
            endo_boundary_mean_distances
        ),
    }


def _ordered_boundary_loops(mesh):
    boundary = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        non_manifold_edges=False,
        manifold_edges=False,
    )
    if boundary.n_points == 0:
        raise GeometryError("Clipped surface has no boundary loop")
    mapped_distances, mapped_ids = cKDTree(
        np.asarray(mesh.points)
    ).query(np.asarray(boundary.points), k=1)
    tolerance = max(
        float(mesh.length) * CONTOUR_MAP_TOLERANCE_RATIO, 1.0e-10
    )
    if float(mapped_distances.max()) > tolerance:
        raise GeometryError(
            "Cannot map clipped boundary points back to the surface"
        )

    adjacency = defaultdict(set)
    for first, second in _line_segments(boundary):
        first_id = int(mapped_ids[first])
        second_id = int(mapped_ids[second])
        if first_id == second_id:
            continue
        adjacency[first_id].add(second_id)
        adjacency[second_id].add(first_id)

    unvisited = set(adjacency)
    loops = []
    while unvisited:
        component_start = min(unvisited)
        stack = [component_start]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        unvisited.difference_update(component)
        degrees = [len(adjacency[point_id]) for point_id in component]
        if not all(degree == 2 for degree in degrees):
            raise GeometryError(
                "Clipped boundary is not a collection of closed loops: "
                f"degree_range=({min(degrees)}, {max(degrees)})"
            )

        ordered = [min(component)]
        previous = None
        current = ordered[0]
        while True:
            neighbors = sorted(adjacency[current])
            next_id = (
                neighbors[0]
                if neighbors[0] != previous
                else neighbors[1]
            )
            if next_id == ordered[0]:
                break
            if next_id in ordered:
                raise GeometryError(
                    "Clipped boundary loop revisits a vertex"
                )
            ordered.append(next_id)
            previous, current = current, next_id
        loops.append(np.asarray(ordered, dtype=np.int64))

    loops.sort(
        key=lambda ids: float(
            np.sum(
                np.linalg.norm(
                    np.asarray(mesh.points)[ids]
                    - np.roll(np.asarray(mesh.points)[ids], -1, axis=0),
                    axis=1,
                )
            )
        ),
        reverse=True,
    )
    return loops


def _ordered_contour_points(contour, projected_points):
    segments = _line_segments(contour)
    adjacency = defaultdict(set)
    for first, second in segments:
        adjacency[first].add(second)
        adjacency[second].add(first)
    if len(adjacency) != contour.n_points:
        raise GeometryError("Contour contains unused points")
    if not all(len(neighbors) == 2 for neighbors in adjacency.values()):
        raise GeometryError("Contour is not a simple closed loop")

    ordered = [min(adjacency)]
    previous = None
    current = ordered[0]
    while True:
        neighbors = sorted(adjacency[current])
        next_id = (
            neighbors[0] if neighbors[0] != previous else neighbors[1]
        )
        if next_id == ordered[0]:
            break
        if next_id in ordered:
            raise GeometryError("Contour loop revisits a vertex")
        ordered.append(next_id)
        previous, current = current, next_id
    return np.asarray(projected_points)[ordered]


def _median_mesh_edge_length(mesh):
    faces = _triangle_faces(mesh)
    edges = np.vstack(
        [
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(
        np.asarray(mesh.points)[edges[:, 0]]
        - np.asarray(mesh.points)[edges[:, 1]],
        axis=1,
    )
    positive = lengths[lengths > 1.0e-12]
    if positive.size == 0:
        raise GeometryError("Cannot estimate local mesh edge length")
    return float(np.median(positive))


def _closed_loop_parameters(points):
    points = np.asarray(points, dtype=np.float64)
    edge_lengths = np.linalg.norm(
        np.roll(points, -1, axis=0) - points, axis=1
    )
    total = float(np.sum(edge_lengths))
    if total <= 1.0e-12:
        raise GeometryError("Degenerate closed boundary loop")
    return np.r_[0.0, np.cumsum(edge_lengths)] / total


def _resample_closed_loop(points, count):
    points = np.asarray(points, dtype=np.float64)
    parameters = _closed_loop_parameters(points)
    extended = np.vstack([points, points[0]])
    targets = np.arange(count, dtype=np.float64) / count
    result = np.empty((count, 3), dtype=np.float64)
    for axis in range(3):
        result[:, axis] = np.interp(
            targets, parameters, extended[:, axis]
        )
    return result


def _align_common_loop(boundary_ids, mesh_points, common_points):
    boundary_ids = np.asarray(boundary_ids, dtype=np.int64)
    boundary_points = np.asarray(mesh_points)[boundary_ids]
    distances, common_ids = cKDTree(common_points).query(
        boundary_points, k=1
    )
    boundary_start = int(np.argmin(distances))
    common_start = int(common_ids[boundary_start])
    boundary_ids = np.roll(boundary_ids, -boundary_start)
    boundary_points = np.asarray(mesh_points)[boundary_ids]
    common_forward = np.roll(common_points, -common_start, axis=0)
    common_reverse = np.vstack(
        [
            common_points[common_start],
            common_points[:common_start][::-1],
            common_points[common_start + 1 :][::-1],
        ]
    )
    sample_count = max(
        64, min(512, 2 * max(len(boundary_ids), len(common_points)))
    )
    boundary_sample = _resample_closed_loop(
        boundary_points, sample_count
    )
    forward_cost = float(
        np.mean(
            np.linalg.norm(
                boundary_sample
                - _resample_closed_loop(common_forward, sample_count),
                axis=1,
            )
        )
    )
    reverse_cost = float(
        np.mean(
            np.linalg.norm(
                boundary_sample
                - _resample_closed_loop(common_reverse, sample_count),
                axis=1,
            )
        )
    )
    aligned_common = (
        common_forward if forward_cost <= reverse_cost else common_reverse
    )
    return boundary_ids, aligned_common


def _zipper_strip_faces(boundary_ids, common_ids, boundary_points, common_points):
    boundary_parameters = _closed_loop_parameters(boundary_points)
    common_parameters = _closed_loop_parameters(common_points)
    boundary_count = len(boundary_ids)
    common_count = len(common_ids)
    boundary_index = 0
    common_index = 0
    triangles = []
    while boundary_index < boundary_count or common_index < common_count:
        boundary_next = (
            boundary_parameters[boundary_index + 1]
            if boundary_index < boundary_count
            else np.inf
        )
        common_next = (
            common_parameters[common_index + 1]
            if common_index < common_count
            else np.inf
        )
        current_boundary = int(
            boundary_ids[boundary_index % boundary_count]
        )
        current_common = int(common_ids[common_index % common_count])
        if boundary_next <= common_next:
            next_boundary = int(
                boundary_ids[(boundary_index + 1) % boundary_count]
            )
            triangles.append(
                [current_boundary, next_boundary, current_common]
            )
            boundary_index += 1
        else:
            next_common = int(
                common_ids[(common_index + 1) % common_count]
            )
            triangles.append(
                [current_boundary, next_common, current_common]
            )
            common_index += 1
    return np.asarray(triangles, dtype=np.int64)


def _bridge_surface_to_common_loop(mesh, boundary_ids, common_points):
    boundary_ids, common_points = _align_common_loop(
        boundary_ids, np.asarray(mesh.points), common_points
    )
    boundary_points = np.asarray(mesh.points)[boundary_ids]
    common_start = mesh.n_points
    common_ids = np.arange(
        common_start,
        common_start + len(common_points),
        dtype=np.int64,
    )
    strip_faces = _zipper_strip_faces(
        boundary_ids,
        common_ids,
        boundary_points,
        common_points,
    )
    original_faces = _triangle_faces(mesh)
    all_faces = np.vstack([original_faces, strip_faces])
    faces_pv = np.c_[
        np.full(all_faces.shape[0], 3, dtype=np.int64),
        all_faces,
    ].reshape(-1)
    bridged = pv.PolyData(
        np.vstack([np.asarray(mesh.points), common_points]),
        faces_pv,
    )
    bridged = bridged.clean(
        point_merging=True,
        tolerance=POINT_MERGE_TOLERANCE,
        absolute=True,
        lines_to_points=False,
        polys_to_lines=False,
    )
    return bridged.extract_surface().triangulate()


def _boundary_to_contour_gap(boundary_points, common_points):
    boundary_to_common = cKDTree(common_points).query(
        boundary_points, k=1
    )[0]
    common_to_boundary = cKDTree(boundary_points).query(
        common_points, k=1
    )[0]
    combined = np.r_[boundary_to_common, common_to_boundary]
    return {
        "mean": float(np.mean(combined)),
        "p95": float(np.percentile(combined, 95)),
        "max": float(np.max(combined)),
    }


def _geometry_only_surface(mesh):
    mesh = mesh.extract_surface().triangulate()
    return pv.PolyData(
        np.asarray(mesh.points).copy(),
        np.asarray(mesh.faces, dtype=np.int64).copy(),
    )


def _independent_zero_level_components(source, target):
    source = _geometry_only_surface(source)
    target = _geometry_only_surface(target).compute_normals(
        cell_normals=True,
        point_normals=False,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
        non_manifold_traversal=True,
    )
    distance_surface = source.compute_implicit_distance(
        target, inplace=False
    )
    signed_distances = np.asarray(
        distance_surface.point_data["implicit_distance"],
        dtype=np.float64,
    )
    negative_count = int(np.count_nonzero(signed_distances < 0.0))
    positive_count = int(np.count_nonzero(signed_distances > 0.0))
    if negative_count == 0 or positive_count == 0:
        raise GeometryError(
            "The independent signed-distance field has no zero crossing: "
            f"negative={negative_count}, positive={positive_count}"
        )

    side_components = []
    for invert in (False, True):
        clipped = (
            distance_surface.clip_scalar(
                scalars="implicit_distance",
                value=0.0,
                invert=invert,
            )
            .extract_surface()
            .triangulate()
        )
        # OBJ readers and VTK filters may carry inconsistent GroupIds arrays.
        # Rebuild pure geometry before connectivity/topology operations.
        clipped = _geometry_only_surface(clipped)
        side_components.append(_surface_components(clipped))
    return side_components, {
        "signed_distance_negative_points": negative_count,
        "signed_distance_positive_points": positive_count,
        "zero_level_side_component_areas": [
            [float(component.area) for component in components]
            for components in side_components
        ],
    }


def _flatten_side_components(side_components):
    return [
        (side_index, component_index, component)
        for side_index, components in enumerate(side_components)
        for component_index, component in enumerate(components)
    ]


def _merge_surface_components(components):
    if not components:
        raise GeometryError("No surface components remain after base removal")
    merged = components[0].copy()
    for component in components[1:]:
        merged = (
            merged.merge(
                component,
                merge_points=True,
                tolerance=POINT_MERGE_TOLERANCE,
            )
            .extract_surface()
            .triangulate()
            .clean(
                point_merging=True,
                tolerance=POINT_MERGE_TOLERANCE,
                absolute=True,
            )
        )
    merged_components = _surface_components(merged)
    if len(merged_components) != 1:
        raise GeometryError(
            "The retained zero-level components did not reconnect into "
            f"one surface: components={len(merged_components)}"
        )
    return merged_components[0]


def _loop_length(mesh, point_ids):
    points = np.asarray(mesh.points)[point_ids]
    return float(
        np.sum(
            np.linalg.norm(
                points - np.roll(points, -1, axis=0), axis=1
            )
        )
    )


def _select_independent_base_parts(raw_epi, raw_endo):
    epi_sides, epi_diagnostics = (
        _independent_zero_level_components(raw_epi, raw_endo)
    )
    endo_sides, endo_diagnostics = (
        _independent_zero_level_components(raw_endo, raw_epi)
    )
    epi_components = _flatten_side_components(epi_sides)
    endo_components = _flatten_side_components(endo_sides)

    # The endo component containing its original open boundary is the basal
    # extension that must be removed. Other components, including any apical
    # protrusion cut off by a second zero-level loop, are reattached.
    original_endo_boundary = _original_boundary_points(raw_endo)
    boundary_tolerance = max(
        float(raw_endo.length) * CONTOUR_MAP_TOLERANCE_RATIO,
        1.0e-10,
    )
    endo_touch_fractions = []
    endo_boundary_mean_distances = []
    for _, _, component in endo_components:
        distances = cKDTree(np.asarray(component.points)).query(
            original_endo_boundary, k=1
        )[0]
        endo_touch_fractions.append(
            float(np.mean(distances <= boundary_tolerance))
        )
        endo_boundary_mean_distances.append(float(np.mean(distances)))

    endo_removed_flat_index = int(np.argmax(endo_touch_fractions))
    if (
        endo_touch_fractions[endo_removed_flat_index]
        < MIN_BOUNDARY_TOUCH_FRACTION
    ):
        raise GeometryError(
            "Neither independent endo side contains the original open "
            f"boundary: touch_fractions={endo_touch_fractions}"
        )
    if (
        max(
            fraction
            for index, fraction in enumerate(endo_touch_fractions)
            if index != endo_removed_flat_index
        )
        > MAX_NONBOUNDARY_TOUCH_FRACTION
    ):
        raise GeometryError(
            "Multiple independent endo components touch the original open "
            f"boundary: touch_fractions={endo_touch_fractions}"
        )
    (
        endo_removed_side,
        endo_removed_component,
        endo_extension,
    ) = endo_components[endo_removed_flat_index]
    extension_loops = _ordered_boundary_loops(endo_extension)
    if len(extension_loops) < 2:
        raise GeometryError(
            "The removed endo basal extension does not contain both its "
            "original boundary and a base cut loop"
        )
    original_boundary_tree = cKDTree(original_endo_boundary)
    extension_loop_boundary_distances = [
        float(
            np.mean(
                original_boundary_tree.query(
                    np.asarray(endo_extension.points)[point_ids], k=1
                )[0]
            )
        )
        for point_ids in extension_loops
    ]
    original_loop_index = int(
        np.argmin(extension_loop_boundary_distances)
    )
    base_loop_candidates = [
        index
        for index in range(len(extension_loops))
        if index != original_loop_index
    ]
    endo_base_loop_index = max(
        base_loop_candidates,
        key=lambda index: _loop_length(
            endo_extension, extension_loops[index]
        ),
    )
    endo_base_points = np.asarray(endo_extension.points)[
        extension_loops[endo_base_loop_index]
    ]

    # A closed epi is cut into a body and one cap per intersection loop.
    # Select only the one-loop cap matching the endo basal cut. With one
    # intersection, both epi sides share that loop, so remove the smaller one.
    epi_edge_length = _median_mesh_edge_length(raw_epi)
    epi_base_cap_candidates = []
    epi_component_loop_counts = []
    for flat_index, (side_index, component_index, component) in enumerate(
        epi_components
    ):
        loops = _ordered_boundary_loops(component)
        epi_component_loop_counts.append(len(loops))
        if len(loops) != 1:
            continue
        loop_points = np.asarray(component.points)[loops[0]]
        gap = _boundary_to_contour_gap(
            loop_points, endo_base_points
        )
        if (
            gap["p95"]
            <= INDEPENDENT_GAP95_EDGE_RATIO * epi_edge_length
        ):
            epi_base_cap_candidates.append(
                {
                    "flat_index": flat_index,
                    "side_index": side_index,
                    "component_index": component_index,
                    "area": float(component.area),
                    "gap": gap,
                }
            )
    if not epi_base_cap_candidates:
        raise GeometryError(
            "Cannot identify an epi base cap matching the endo base loop"
        )
    epi_base_cap = min(
        epi_base_cap_candidates, key=lambda item: item["area"]
    )
    epi_removed_flat_index = epi_base_cap["flat_index"]

    epi_kept = _merge_surface_components(
        [
            component
            for flat_index, (_, _, component) in enumerate(epi_components)
            if flat_index != epi_removed_flat_index
        ]
    )
    endo_kept = _merge_surface_components(
        [
            component
            for flat_index, (_, _, component) in enumerate(endo_components)
            if flat_index != endo_removed_flat_index
        ]
    )
    epi_loops = _ordered_boundary_loops(epi_kept)
    endo_loops = _ordered_boundary_loops(endo_kept)
    if len(epi_loops) != 1 or len(endo_loops) != 1:
        raise GeometryError(
            "Independent zero-level clipping did not leave exactly one "
            f"boundary loop per kept surface: epi={len(epi_loops)}, "
            f"endo={len(endo_loops)}"
        )

    diagnostics = {
        "epi_signed_distance": epi_diagnostics,
        "endo_signed_distance": endo_diagnostics,
        "independent_epi_component_loop_counts": (
            epi_component_loop_counts
        ),
        "independent_epi_removed_side": epi_base_cap["side_index"],
        "independent_epi_removed_component": (
            epi_base_cap["component_index"]
        ),
        "independent_epi_removed_area": epi_base_cap["area"],
        "independent_epi_base_match_gap": epi_base_cap["gap"],
        "independent_endo_removed_side": endo_removed_side,
        "independent_endo_removed_component": endo_removed_component,
        "independent_endo_removed_area": float(endo_extension.area),
        "independent_endo_boundary_touch_fractions": (
            endo_touch_fractions
        ),
        "independent_endo_boundary_mean_distances": (
            endo_boundary_mean_distances
        ),
        "independent_endo_extension_loop_count": int(
            len(extension_loops)
        ),
        "independent_endo_extension_loop_boundary_distances": (
            extension_loop_boundary_distances
        ),
        "independent_endo_base_loop_index": endo_base_loop_index,
        "independent_epi_boundary_points": int(len(epi_loops[0])),
        "independent_endo_boundary_points": int(len(endo_loops[0])),
    }
    return (
        epi_kept,
        endo_kept,
        epi_loops[0],
        endo_loops[0],
        diagnostics,
    )


def _midpoint_common_loop(
    epi_mesh,
    epi_boundary_ids,
    endo_mesh,
    endo_boundary_ids,
):
    endo_boundary_points = np.asarray(endo_mesh.points)[
        endo_boundary_ids
    ]
    aligned_epi_ids, aligned_endo_points = _align_common_loop(
        epi_boundary_ids,
        np.asarray(epi_mesh.points),
        endo_boundary_points,
    )
    aligned_epi_points = np.asarray(epi_mesh.points)[aligned_epi_ids]
    sample_count = max(
        len(aligned_epi_points), len(aligned_endo_points), 3
    )
    epi_samples = _resample_closed_loop(
        aligned_epi_points, sample_count
    )
    endo_samples = _resample_closed_loop(
        aligned_endo_points, sample_count
    )
    return 0.5 * (epi_samples + endo_samples)


def _independent_zero_level_clip_and_bridge(raw_epi, raw_endo):
    (
        epi_kept,
        endo_kept,
        epi_boundary_ids,
        endo_boundary_ids,
        selection_diagnostics,
    ) = _select_independent_base_parts(raw_epi, raw_endo)
    epi_boundary_points = np.asarray(epi_kept.points)[
        epi_boundary_ids
    ]
    endo_boundary_points = np.asarray(endo_kept.points)[
        endo_boundary_ids
    ]
    boundary_gap = _boundary_to_contour_gap(
        epi_boundary_points, endo_boundary_points
    )
    epi_edge_length = _median_mesh_edge_length(epi_kept)
    endo_edge_length = _median_mesh_edge_length(endo_kept)
    reference_edge_length = 0.5 * (
        epi_edge_length + endo_edge_length
    )
    if (
        boundary_gap["p95"]
        > INDEPENDENT_GAP95_EDGE_RATIO * reference_edge_length
        or boundary_gap["max"]
        > INDEPENDENT_GAP_MAX_EDGE_RATIO * reference_edge_length
    ):
        raise GeometryError(
            "The independent epi/endo zero-level loops are too far apart: "
            f"p95/edge={boundary_gap['p95'] / reference_edge_length:.3f}, "
            f"max/edge={boundary_gap['max'] / reference_edge_length:.3f}"
        )

    common_points = _midpoint_common_loop(
        epi_kept,
        epi_boundary_ids,
        endo_kept,
        endo_boundary_ids,
    )
    epi_bridged = _bridge_surface_to_common_loop(
        epi_kept, epi_boundary_ids, common_points
    )
    endo_bridged = _bridge_surface_to_common_loop(
        endo_kept, endo_boundary_ids, common_points
    )
    test_merge = _merge_two_surfaces(epi_bridged, endo_bridged)
    if test_merge.n_open_edges != 0 or not test_merge.is_manifold:
        raise GeometryError(
            "Independent zero-level loop bridge is not closed and "
            f"manifold: open_edges={test_merge.n_open_edges}, "
            f"manifold={test_merge.is_manifold}"
        )

    diagnostics = {
        **selection_diagnostics,
        "independent_epi_edge_length": epi_edge_length,
        "independent_endo_edge_length": endo_edge_length,
        "independent_reference_edge_length": reference_edge_length,
        "independent_boundary_gap": boundary_gap,
        "independent_boundary_gap_p95_over_edge": (
            boundary_gap["p95"] / reference_edge_length
        ),
        "independent_boundary_gap_max_over_edge": (
            boundary_gap["max"] / reference_edge_length
        ),
        "independent_common_loop_points": int(len(common_points)),
    }
    return epi_bridged, endo_bridged, diagnostics


def _approximate_clip_and_bridge(raw_epi, raw_endo, contour):
    (
        plane_origin,
        plane_normal,
        projected_contour,
        plane_diagnostics,
    ) = _fit_contour_plane(contour)
    common_points = _ordered_contour_points(
        contour, projected_contour
    )
    epi_clipped, endo_clipped, clip_diagnostics = (
        _select_approximate_clipped_surfaces(
            raw_epi,
            raw_endo,
            plane_origin,
            plane_normal,
        )
    )
    epi_loops = _ordered_boundary_loops(epi_clipped)
    endo_loops = _ordered_boundary_loops(endo_clipped)
    if len(epi_loops) != 1 or len(endo_loops) != 1:
        raise GeometryError(
            "Approximate plane clipping did not produce exactly one "
            f"boundary loop per surface: epi={len(epi_loops)}, "
            f"endo={len(endo_loops)}"
        )

    epi_boundary_points = np.asarray(epi_clipped.points)[epi_loops[0]]
    endo_boundary_points = np.asarray(endo_clipped.points)[endo_loops[0]]
    epi_gap = _boundary_to_contour_gap(
        epi_boundary_points, common_points
    )
    endo_gap = _boundary_to_contour_gap(
        endo_boundary_points, common_points
    )
    epi_edge_length = _median_mesh_edge_length(epi_clipped)
    endo_edge_length = _median_mesh_edge_length(endo_clipped)
    for surface, gap, edge_length in (
        ("epi", epi_gap, epi_edge_length),
        ("endo", endo_gap, endo_edge_length),
    ):
        if (
            gap["p95"]
            > APPROXIMATE_GAP95_EDGE_RATIO * edge_length
            or gap["max"]
            > APPROXIMATE_GAP_MAX_EDGE_RATIO * edge_length
        ):
            raise GeometryError(
                f"Approximate {surface} boundary is too far from the "
                "common contour: "
                f"p95/edge={gap['p95'] / edge_length:.3f}, "
                f"max/edge={gap['max'] / edge_length:.3f}"
            )

    epi_bridged = _bridge_surface_to_common_loop(
        epi_clipped, epi_loops[0], common_points
    )
    endo_bridged = _bridge_surface_to_common_loop(
        endo_clipped, endo_loops[0], common_points
    )
    test_merge = _merge_two_surfaces(epi_bridged, endo_bridged)
    if test_merge.n_open_edges != 0 or not test_merge.is_manifold:
        raise GeometryError(
            "Approximate common-contour bridge is not closed and "
            f"manifold: open_edges={test_merge.n_open_edges}, "
            f"manifold={test_merge.is_manifold}"
        )
    diagnostics = {
        **plane_diagnostics,
        **clip_diagnostics,
        "approximate_epi_boundary_points": len(epi_loops[0]),
        "approximate_endo_boundary_points": len(endo_loops[0]),
        "approximate_common_contour_points": len(common_points),
        "approximate_epi_edge_length": epi_edge_length,
        "approximate_endo_edge_length": endo_edge_length,
        "approximate_epi_gap": epi_gap,
        "approximate_endo_gap": endo_gap,
    }
    return epi_bridged, endo_bridged, diagnostics


def _radial_normal_score(mesh, center=None):
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


def _orient_surface(mesh, outward):
    oriented = mesh.compute_normals(
        cell_normals=True,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
        non_manifold_traversal=True,
    )
    score = _radial_normal_score(oriented)
    if (score > 0) != bool(outward):
        oriented = oriented.flip_faces().compute_normals(
            cell_normals=True,
            point_normals=True,
            split_vertices=False,
            consistent_normals=True,
            auto_orient_normals=False,
            non_manifold_traversal=True,
        )
        score = -score
    return oriented, score


def _assign_surface_ids(merged, epi, endo):
    merged_centers = np.asarray(merged.cell_centers().points)
    epi_centers = np.asarray(epi.cell_centers().points)
    endo_centers = np.asarray(endo.cell_centers().points)
    epi_distances = cKDTree(epi_centers).query(merged_centers, k=1)[0]
    endo_distances = cKDTree(endo_centers).query(merged_centers, k=1)[0]
    surface_ids = (endo_distances < epi_distances).astype(np.int32)

    expected_counts = {0: int(epi.n_cells), 1: int(endo.n_cells)}
    actual_counts = {
        surface_id: int(np.count_nonzero(surface_ids == surface_id))
        for surface_id in (0, 1)
    }
    assigned_distances = np.where(
        surface_ids == 0, epi_distances, endo_distances
    )
    tolerance = max(
        float(merged.length) * CONTOUR_MAP_TOLERANCE_RATIO,
        1.0e-10,
    )
    if actual_counts != expected_counts:
        raise GeometryError(
            "Cannot preserve epi/endo cell identity after merge: "
            f"expected={expected_counts}, actual={actual_counts}"
        )
    if float(assigned_distances.max()) > tolerance:
        raise GeometryError(
            "Merged cells cannot be mapped back to their source surface: "
            f"max_distance={float(assigned_distances.max()):.8g}"
        )

    # Drop arrays produced internally by vtkIntersectionPolyDataFilter and
    # compute_normals. Keep the saved VTK interface small and predictable.
    for array_name in list(merged.point_data.keys()):
        del merged.point_data[array_name]
    for array_name in list(merged.cell_data.keys()):
        del merged.cell_data[array_name]
    for array_name in list(merged.field_data.keys()):
        del merged.field_data[array_name]

    merged.cell_data["surface_id"] = surface_ids
    merged.cell_data["label"] = (
        surface_ids + EPI_LABEL
    ).astype(np.int32)

    faces = _triangle_faces(merged)
    epi_incident = np.zeros(merged.n_points, dtype=bool)
    endo_incident = np.zeros(merged.n_points, dtype=bool)
    epi_incident[
        np.unique(faces[surface_ids == 0].reshape(-1))
    ] = True
    endo_incident[
        np.unique(faces[surface_ids == 1].reshape(-1))
    ] = True
    if not np.all(epi_incident | endo_incident):
        raise GeometryError("Combined mesh contains unlabeled points")

    # A welded interface point is incident to both surfaces and cannot carry
    # two scalar values. Match the existing obj2vtk convention, where epi is
    # concatenated first, by giving shared contour points epi precedence.
    point_labels = np.full(
        merged.n_points, ENDO_LABEL, dtype=np.int32
    )
    point_labels[epi_incident] = EPI_LABEL
    merged.point_data["label"] = point_labels
    merged.field_data["postprocess_algorithm_version"] = np.asarray(
        [POSTPROCESS_ALGORITHM_VERSION], dtype=np.int32
    )
    merged.field_data["postprocess_algorithm_name"] = np.asarray(
        [POSTPROCESS_ALGORITHM_NAME]
    )
    return merged


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


def validate_combined_mesh(mesh):
    mesh = mesh.extract_surface().triangulate()
    if mesh.n_open_edges != 0:
        raise GeometryError(
            f"Combined mesh has {mesh.n_open_edges} open edges"
        )
    if not mesh.is_manifold:
        raise GeometryError("Combined mesh is not manifold")
    if "postprocess_algorithm_version" not in mesh.field_data:
        raise GeometryError(
            "Combined mesh was produced by an obsolete postprocess "
            "algorithm"
        )
    algorithm_version = int(
        np.asarray(
            mesh.field_data["postprocess_algorithm_version"]
        ).reshape(-1)[0]
    )
    if algorithm_version != POSTPROCESS_ALGORITHM_VERSION:
        raise GeometryError(
            "Combined mesh postprocess algorithm version is obsolete: "
            f"found={algorithm_version}, "
            f"expected={POSTPROCESS_ALGORITHM_VERSION}"
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
            "Cell label is inconsistent with surface_id; expected "
            "epi=1 and endo=2"
        )
    if "label" not in mesh.point_data:
        raise GeometryError("Combined mesh has no point label array")
    point_labels = np.asarray(mesh.point_data["label"]).astype(int)
    point_label_values = set(
        int(value) for value in np.unique(point_labels)
    )
    if point_label_values != {EPI_LABEL, ENDO_LABEL}:
        raise GeometryError(
            "Expected point label values {1, 2}, got "
            f"{point_label_values}"
        )

    epi = _surface_from_id(mesh, 0)
    endo = _surface_from_id(mesh, 1)
    epi_score = _radial_normal_score(epi)
    endo_score = _radial_normal_score(endo)
    if epi_score < MIN_NORMAL_ORIENTATION_SCORE:
        raise GeometryError(
            f"Epi normals are not outward: score={epi_score:.6f}"
        )
    if endo_score > -MIN_NORMAL_ORIENTATION_SCORE:
        raise GeometryError(
            f"Endo normals are not directed toward the chamber: "
            f"score={endo_score:.6f}"
        )
    return {
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "n_open_edges": int(mesh.n_open_edges),
        "is_manifold": bool(mesh.is_manifold),
        "epi_cells": int(epi.n_cells),
        "endo_cells": int(endo.n_cells),
        "epi_label": EPI_LABEL,
        "endo_label": ENDO_LABEL,
        "epi_outward_score": epi_score,
        "endo_outward_score": endo_score,
    }


def _validate_raw_surface_roles(raw_epi, raw_endo):
    if raw_epi.n_open_edges != 0 or not raw_epi.is_manifold:
        raise GeometryError(
            "Expected a closed manifold epi input: "
            f"open_edges={raw_epi.n_open_edges}, "
            f"manifold={raw_epi.is_manifold}"
        )
    if raw_endo.n_open_edges == 0:
        raise GeometryError("Expected an open endo input")


def _finalize_surface_pair(
    raw_epi,
    raw_endo,
    epi_kept,
    endo_kept,
    diagnostics,
):
    epi_oriented, epi_score = _orient_surface(epi_kept, outward=True)
    endo_oriented, endo_score = _orient_surface(
        endo_kept, outward=False
    )
    combined = _merge_two_surfaces(epi_oriented, endo_oriented)
    combined = combined.compute_normals(
        cell_normals=True,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=False,
        non_manifold_traversal=True,
    )
    combined = _assign_surface_ids(
        combined, epi_oriented, endo_oriented
    )
    validation = validate_combined_mesh(combined)
    diagnostics.update(
        {
            "raw_epi_open_edges": int(raw_epi.n_open_edges),
            "raw_endo_open_edges": int(raw_endo.n_open_edges),
            "premerge_epi_outward_score": float(epi_score),
            "premerge_endo_outward_score": float(endo_score),
            **validation,
        }
    )
    return combined, diagnostics


def clip_and_merge_surface_pair(epi_path, endo_path):
    raw_epi = _load_surface(epi_path)
    raw_endo = _load_surface(endo_path)
    _validate_raw_surface_roles(raw_epi, raw_endo)
    epi_kept, endo_kept, independent_diagnostics = (
        _independent_zero_level_clip_and_bridge(raw_epi, raw_endo)
    )
    diagnostics = {
        **independent_diagnostics,
        "processing_mode": POSTPROCESS_ALGORITHM_NAME,
        "postprocess_algorithm_version": POSTPROCESS_ALGORITHM_VERSION,
    }
    return _finalize_surface_pair(
        raw_epi,
        raw_endo,
        epi_kept,
        endo_kept,
        diagnostics,
    )

