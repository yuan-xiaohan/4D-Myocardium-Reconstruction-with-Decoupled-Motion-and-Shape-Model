"""Optionally assign the fixed epi/endo labels to one LV VTK mesh.

This is a mask-guided helper, not part of the training pipeline.  It keeps
the input geometry unchanged and only writes the required cell arrays:
``surface_id`` (0=epi, 1=endo) and ``label`` (1=epi, 2=endo).  The result
must be reviewed by the user before it is used for training.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

from preprocess_mask import extract_contours


ALGORITHM_VERSION = "mask_guided_epi_endo_v1"


class SplitError(RuntimeError):
    """The input cannot be assigned two reliable surface roles."""


def _triangles(mesh: pv.PolyData, path: Path) -> np.ndarray:
    raw = np.asarray(mesh.faces, dtype=np.int64)
    faces = []
    cursor = 0
    while cursor < len(raw):
        count = int(raw[cursor])
        end = cursor + count + 1
        if count != 3 or end > len(raw):
            raise SplitError(f"{path} must contain only triangular cells")
        faces.append(raw[cursor + 1 : end])
        cursor = end
    if cursor != len(raw) or not faces:
        raise SplitError(f"{path} contains no triangular cells")
    if len(faces) != mesh.n_cells:
        raise SplitError(f"{path} contains non-triangular cells")
    faces = np.asarray(faces, dtype=np.int64)
    if faces.min() < 0 or faces.max() >= mesh.n_points:
        raise SplitError(f"{path} contains an invalid face index")
    return faces


def _read_mesh(path: Path) -> tuple[pv.PolyData, np.ndarray]:
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        mesh = mesh.combine()
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise SplitError(f"Empty mesh: {path}")
    points = np.asarray(mesh.points, dtype=np.float64)
    if not np.isfinite(points).all():
        raise SplitError(f"Mesh contains non-finite points: {path}")
    return mesh.copy(deep=True), _triangles(mesh, path)


def _edge_faces(faces: np.ndarray) -> tuple[dict[tuple[int, int], list[int]], list[list[int]]]:
    edges: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edges[tuple(sorted((int(first), int(second))))].append(face_index)
    bad = {edge: cells for edge, cells in edges.items() if len(cells) > 2}
    if bad:
        raise SplitError(f"Non-manifold edge(s): {list(bad.items())[:3]}")
    adjacency = [[] for _ in range(len(faces))]
    for cells in edges.values():
        if len(cells) == 2:
            first, second = cells
            adjacency[first].append(second)
            adjacency[second].append(first)
    return dict(edges), adjacency


def _components(adjacency: list[list[int]]) -> list[np.ndarray]:
    unseen = set(range(len(adjacency)))
    result = []
    while unseen:
        start = unseen.pop()
        queue = [start]
        members = [start]
        while queue:
            current = queue.pop()
            for neighbour in adjacency[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    members.append(neighbour)
                    queue.append(neighbour)
        result.append(np.asarray(members, dtype=np.int64))
    return sorted(result, key=len, reverse=True)


def _boundary_loop(edges: list[tuple[int, int]]) -> np.ndarray:
    graph: dict[int, set[int]] = defaultdict(set)
    for first, second in edges:
        graph[first].add(second)
        graph[second].add(first)
    if not graph:
        raise SplitError("Surface has no Base boundary")
    bad = {vertex: len(neighbours) for vertex, neighbours in graph.items() if len(neighbours) != 2}
    if bad:
        raise SplitError(f"Base boundary is branched: {list(bad.items())[:3]}")
    unseen = set(graph)
    loops = 0
    while unseen:
        loops += 1
        queue = [unseen.pop()]
        while queue:
            current = queue.pop()
            for neighbour in graph[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    if loops != 1:
        raise SplitError(f"Surface has {loops} Base boundary loops; expected one")
    return np.asarray(sorted(graph), dtype=np.int64)


def _topology(faces: np.ndarray) -> dict:
    edge_faces, adjacency = _edge_faces(faces)
    components = _components(adjacency)
    boundary = [edge for edge, cells in edge_faces.items() if len(cells) == 1]
    if len(components) == 1 and not boundary:
        return {"mode": "closed_combined_shell", "edges": edge_faces, "adjacency": adjacency, "components": components}
    if len(components) != 2:
        raise SplitError(
            "Expected one closed LV shell or two open epi/endo surfaces; "
            f"found {len(components)} components"
        )
    boundaries = []
    for component in components:
        members = set(component.tolist())
        component_edges = [edge for edge, cells in edge_faces.items() if len(cells) == 1 and cells[0] in members]
        boundaries.append(_boundary_loop(component_edges))
    return {
        "mode": "two_open_surfaces",
        "edges": edge_faces,
        "adjacency": adjacency,
        "components": components,
        "boundaries": boundaries,
    }


def _mask_points(mask: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    contours = extract_contours(mask, "auto")
    epi = np.vstack([np.asarray(item, dtype=np.float64) for item in contours.epi if len(item)])
    endo = np.vstack([np.asarray(item, dtype=np.float64) for item in contours.endo if len(item)])
    if len(epi) == 0 or len(endo) == 0:
        raise SplitError(f"Mask has no epi or endo contours: {mask}")
    centers = np.vstack([
        np.asarray(item, dtype=np.float64).mean(axis=0)
        for item in contours.endo
        if len(item)
    ])
    if len(centers) < 2:
        raise SplitError(f"Mask has fewer than two LV cavity slices: {mask}")
    dense = []
    for first, second in zip(centers[:-1], centers[1:]):
        length = float(np.linalg.norm(second - first))
        count = max(1, int(np.ceil(length / 1.0)))
        values = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
        dense.append((1.0 - values) * first + values * second)
    dense.append(centers[-1:])
    return np.vstack(dense), epi, endo


def _face_centers(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64)[faces].mean(axis=1)


def _label_components(labels: np.ndarray, value: int, adjacency: list[list[int]]) -> list[np.ndarray]:
    unseen = set(np.flatnonzero(labels == value).tolist())
    result = []
    while unseen:
        start = unseen.pop()
        queue = [start]
        members = [start]
        while queue:
            current = queue.pop()
            for neighbour in adjacency[current]:
                if neighbour in unseen and labels[neighbour] == value:
                    unseen.remove(neighbour)
                    members.append(neighbour)
                    queue.append(neighbour)
        result.append(np.asarray(members, dtype=np.int64))
    return sorted(result, key=len, reverse=True)


def _interface_stats(labels: np.ndarray, edges: dict[tuple[int, int], list[int]]) -> dict:
    interface = [edge for edge, cells in edges.items() if len(cells) == 2 and labels[cells[0]] != labels[cells[1]]]
    graph: dict[int, set[int]] = defaultdict(set)
    for first, second in interface:
        graph[first].add(second)
        graph[second].add(first)
    if not graph:
        raise SplitError("Epi and endo have no interface boundary")
    bad = {vertex: len(neighbours) for vertex, neighbours in graph.items() if len(neighbours) != 2}
    if bad:
        raise SplitError(f"Epi/endo interface is branched: {list(bad.items())[:3]}")
    unseen = set(graph)
    loops = 0
    while unseen:
        loops += 1
        queue = [unseen.pop()]
        while queue:
            current = queue.pop()
            for neighbour in graph[current]:
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    if loops != 1:
        raise SplitError(f"Epi/endo interface has {loops} loops; expected one")
    return {"interface_edges": len(interface), "interface_vertices": len(graph), "interface_loops": loops}


def _closed_labels(points: np.ndarray, faces: np.ndarray, mask: Path, topology: dict) -> tuple[np.ndarray, dict]:
    centerline, epi_points, endo_points = _mask_points(mask)
    poly = pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)).astype(np.int64).ravel())
    oriented = poly.compute_normals(
        cell_normals=True, point_normals=False, consistent_normals=True,
        auto_orient_normals=True, inplace=False,
    )
    normals = np.asarray(oriented.cell_data["Normals"], dtype=np.float64)
    centers = _face_centers(points, faces)
    nearest = centerline[cKDTree(centerline).query(centers, k=1)[1]]
    radial = centers - nearest
    norm = np.linalg.norm(radial, axis=1)
    if np.any(norm <= 1e-9):
        raise SplitError("A triangle center lies on the Mask-derived LV centerline")
    scores = np.einsum("ij,ij->i", normals, radial / norm[:, None])
    labels = (scores < 0.0).astype(np.int8)
    ambiguous = np.abs(scores) < 0.08
    if np.any(ambiguous):
        epi_distance = cKDTree(epi_points).query(centers[ambiguous], k=1)[0]
        endo_distance = cKDTree(endo_points).query(centers[ambiguous], k=1)[0]
        labels[ambiguous] = (endo_distance < epi_distance).astype(np.int8)
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise SplitError("Mask-guided normal classification produced one surface only")
    if np.median(scores[labels == 0]) < np.median(scores[labels == 1]):
        labels = 1 - labels
        scores = -scores
    for _ in range(8):
        changed = 0
        for value in (0, 1):
            groups = _label_components(labels, value, topology["adjacency"])
            if not groups:
                raise SplitError(f"Surface {value} disappeared during cleanup")
            for group in groups[1:]:
                labels[group] = 1 - value
                changed += len(group)
        if changed == 0:
            break
    groups = {str(value): len(_label_components(labels, value, topology["adjacency"])) for value in (0, 1)}
    if groups != {"0": 1, "1": 1}:
        raise SplitError(f"Labels remain disconnected after cleanup: {groups}")
    diagnostics = {
        "normal_score_epi_median": float(np.median(scores[labels == 0])),
        "normal_score_endo_median": float(np.median(scores[labels == 1])),
        "ambiguous_faces": int(np.count_nonzero(ambiguous)),
        "components": groups,
    }
    diagnostics.update(_interface_stats(labels, topology["edges"]))
    return labels, diagnostics


def _open_labels(points: np.ndarray, faces: np.ndarray, mask: Path, topology: dict) -> tuple[np.ndarray, dict]:
    _, epi_points, endo_points = _mask_points(mask)
    centers = _face_centers(points, faces)
    epi_tree, endo_tree = cKDTree(epi_points), cKDTree(endo_points)
    scores = []
    for component in topology["components"]:
        component_centers = centers[component]
        scores.append((
            float(np.median(epi_tree.query(component_centers, k=1)[0])),
            float(np.median(endo_tree.query(component_centers, k=1)[0])),
        ))
    direct = scores[0][0] + scores[1][1]
    reverse = scores[0][1] + scores[1][0]
    epi_component, endo_component = (0, 1) if direct <= reverse else (1, 0)
    labels = np.empty(len(faces), dtype=np.int8)
    labels[topology["components"][epi_component]] = 0
    labels[topology["components"][endo_component]] = 1
    epi_boundary = topology["boundaries"][epi_component]
    endo_boundary = topology["boundaries"][endo_component]
    gap = np.concatenate((
        cKDTree(points[endo_boundary]).query(points[epi_boundary], k=1)[0],
        cKDTree(points[epi_boundary]).query(points[endo_boundary], k=1)[0],
    ))
    return labels, {
        "component_epi": epi_component,
        "component_endo": endo_component,
        "base_gap_mean": float(np.mean(gap)),
        "base_gap_p95": float(np.percentile(gap, 95)),
        "base_gap_max": float(np.max(gap)),
        "base_vertices_epi": int(len(epi_boundary)),
        "base_vertices_endo": int(len(endo_boundary)),
    }


def classify(mesh: pv.PolyData, faces: np.ndarray, mask: Path) -> tuple[np.ndarray, dict]:
    topology = _topology(faces)
    if topology["mode"] == "closed_combined_shell":
        labels, diagnostics = _closed_labels(np.asarray(mesh.points), faces, mask, topology)
    else:
        labels, diagnostics = _open_labels(np.asarray(mesh.points), faces, mask, topology)
    diagnostics["topology_mode"] = topology["mode"]
    diagnostics["n_points"] = int(mesh.n_points)
    diagnostics["n_faces"] = int(len(faces))
    return labels, diagnostics


def write_labels(mesh: pv.PolyData, labels: np.ndarray, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    result = mesh.copy(deep=True)
    result.cell_data["surface_id"] = labels.astype(np.int32)
    result.cell_data["label"] = np.where(labels == 0, 1, 2).astype(np.int32)
    handle = tempfile.NamedTemporaryFile(prefix=f".{output.stem}.", suffix=".vtk", dir=output.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        result.save(temporary, binary=True)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--force", action="store_true", help="recompute existing surface labels")
    args = parser.parse_args(argv)
    try:
        mesh, faces = _read_mesh(args.mesh)
        has_surface = "surface_id" in mesh.cell_data
        has_label = "label" in mesh.cell_data
        if (has_surface or has_label) and not args.force:
            raise SplitError(
                "Input already contains surface_id/label; review it directly or use --force to recompute"
            )
        if has_surface:
            mesh.cell_data.pop("surface_id")
        if has_label:
            mesh.cell_data.pop("label")
        labels, diagnostics = classify(mesh, faces, args.mask)
        write_labels(mesh, labels, args.output)
        report = {
            "status": "success",
            "algorithm": ALGORITHM_VERSION,
            "input_mesh": str(args.mesh),
            "input_mask": str(args.mask),
            "output": str(args.output),
            "geometry_unchanged": True,
            "diagnostics": diagnostics,
        }
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0
    except Exception as error:
        report = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
