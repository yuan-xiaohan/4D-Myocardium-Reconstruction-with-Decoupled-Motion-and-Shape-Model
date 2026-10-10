"""Render one postprocessed 4DMM case with input-observation overlays."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pyvista as pv
import vtk
import pyvista.plotting._vtk as pyvista_vtk
import pyvista.plotting.plotter as plotter_module

from case_config import CaseConfig, load_case_configs
from postprocess_4dmm_bridge import validate_combined_mesh
from preprocess_mask import discover_frames, extract_contours, load_preprocess_report


FPS = 10.0
VIDEO_CODEC = "mp4v"
FLAT_CAMERA_POSITION = (
    (2.6591486617344593, -4.618242962652609, -0.8886649007239069),
    (0.27005286001644213, -0.020398898908829843, 0.061032234855901424),
    (-0.0747343779305607, 0.1643263291005936, -0.9835708567864622),
)
MESH_ZOOM = 1.8
RECON_COLOR = "#f2a6a6"
EPI_OPACITY = 0.42
ENDO_OPACITY = 0.55
EPI_CONTOUR_COLOR = "#0066ff"
ENDO_CONTOUR_COLOR = "#00c853"
EPI_POINT_COLOR = "#0066ff"
ENDO_POINT_COLOR = "#00c853"
EPI_BGR = (255, 102, 0)
ENDO_BGR = (83, 200, 0)
RENDER_WINDOW_SIZE = (1200, 800)
PANEL_IMAGE_SIZE = (480, 320)
LABEL_HEIGHT = 42
VIDEO_SIZE = (PANEL_IMAGE_SIZE[0], PANEL_IMAGE_SIZE[1] + LABEL_HEIGHT)


def _configure_offscreen_backend() -> str:
    """Select a headless VTK backend before creating any render window.

    VTK's default X/OpenGL window can segfault on an SSH host without a
    usable visual.  The bundled VTK wheels expose OSMesa, which renders the
    same fixed-camera frames entirely off-screen and works without DISPLAY.
    """
    osmesa_window = getattr(vtk, "vtkOSOpenGLRenderWindow", None)
    if osmesa_window is not None:
        plotter_module._vtk.vtkRenderWindow = osmesa_window
        pyvista_vtk.vtkRenderWindow = osmesa_window
        return "osmesa"
    pv.OFF_SCREEN = True
    return "default-offscreen"


def _load_stitched_phase(config: CaseConfig, phase: int) -> dict[str, pv.PolyData]:
    path = config.output_dir / "vtk" / f"frame_{phase:02d}.vtk"
    if not path.is_file():
        raise FileNotFoundError(path)
    combined = pv.read(str(path)).extract_surface().triangulate()
    validate_combined_mesh(combined)
    surface_ids = np.asarray(combined.cell_data["surface_id"]).reshape(-1)
    surfaces = {}
    for surface, surface_id in (("epi", 0), ("endo", 1)):
        selected = (
            combined.extract_cells(np.flatnonzero(surface_ids == surface_id))
            .extract_surface().triangulate().clean()
        )
        if not selected.n_points or not selected.n_cells:
            raise RuntimeError(f"Empty {surface} surface in {path}")
        surfaces[surface] = selected
    return surfaces


def _world_to_normalized(meshes, transform, offset, scale):
    normalized = {}
    for surface, mesh in meshes.items():
        copied = mesh.copy(deep=True)
        copied.points = _world_points_to_normalized(
            np.asarray(copied.points, dtype=np.float64), transform, offset, scale
        )
        normalized[surface] = copied
    return normalized


def _world_points_to_normalized(
    points: np.ndarray, transform: np.ndarray, offset: np.ndarray, scale: float
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.c_[points, np.ones(len(points), dtype=np.float64)]
    return ((homogeneous @ transform.T)[:, :3] + offset) * scale


def _center(meshes) -> np.ndarray:
    points = np.vstack([np.asarray(mesh.points) for mesh in meshes.values()])
    return 0.5 * (points.min(axis=0) + points.max(axis=0))


def _apply_reference_camera(plotter, reference_center):
    position, focal_point, view_up = (
        np.asarray(value, dtype=np.float64) for value in FLAT_CAMERA_POSITION
    )
    reference_center = np.asarray(reference_center, dtype=np.float64)
    plotter.camera.position = reference_center + (position - focal_point)
    plotter.camera.focal_point = reference_center
    plotter.camera.up = view_up
    plotter.camera.zoom(MESH_ZOOM)


def _render(meshes, reference_center):
    plotter = pv.Plotter(off_screen=True, window_size=RENDER_WINDOW_SIZE)
    plotter.set_background("white")
    plotter.add_mesh(
        meshes["epi"], color=RECON_COLOR, opacity=EPI_OPACITY,
        smooth_shading=True, show_edges=False,
    )
    plotter.add_mesh(
        meshes["endo"], color=RECON_COLOR, opacity=ENDO_OPACITY,
        smooth_shading=True, show_edges=False,
    )
    _apply_reference_camera(plotter, reference_center)
    image = plotter.screenshot(return_img=True, transparent_background=True)
    plotter.close()
    return image


def _line_polydata(points: np.ndarray) -> pv.PolyData | None:
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3 or not np.isfinite(points).all():
        return None
    closed = np.vstack((points, points[0]))
    line = pv.PolyData(closed)
    line.lines = np.hstack(([len(closed)], np.arange(len(closed), dtype=np.int64)))
    return line


def _add_contours(
    plotter: pv.Plotter,
    contours: list[list[np.ndarray]],
    transform: np.ndarray,
    offset: np.ndarray,
    scale: float,
    color: str,
) -> None:
    for slice_contours in contours:
        for points in slice_contours:
            points = np.asarray(points, dtype=np.float64)
            if len(points) < 3:
                continue
            normalized = _world_points_to_normalized(points, transform, offset, scale)
            line = _line_polydata(normalized)
            if line is not None:
                plotter.add_mesh(
                    line,
                    color=color,
                    line_width=5.0,
                    render_lines_as_tubes=True,
                    lighting=False,
                    show_edges=False,
                )


def _add_points(plotter: pv.Plotter, points: np.ndarray, color: str) -> None:
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Input overlay points must be a finite N x 3 array")
    plotter.add_mesh(
        pv.PolyData(points),
        color=color,
        point_size=5.0,
        render_points_as_spheres=True,
        lighting=False,
        show_edges=False,
    )


def _render_overlay(
    meshes: dict[str, pv.PolyData],
    observation: dict,
    transform: np.ndarray,
    offset: np.ndarray,
    scale: float,
    reference_center: np.ndarray,
) -> np.ndarray:
    """Render a mesh together with the observations used by inference.

    Mask inputs are represented by one closed line loop per slice. Prepared
    inputs are already normalized point clouds and are rendered as points.
    Keeping both paths in this renderer makes the repository video output
    useful for either public preprocessing mode without an external script.
    """
    plotter = pv.Plotter(off_screen=True, window_size=RENDER_WINDOW_SIZE)
    plotter.set_background("white")
    plotter.add_mesh(
        meshes["epi"], color=RECON_COLOR, opacity=EPI_OPACITY,
        smooth_shading=True, show_edges=False,
    )
    plotter.add_mesh(
        meshes["endo"], color=RECON_COLOR, opacity=ENDO_OPACITY,
        smooth_shading=True, show_edges=False,
    )
    if observation["mode"] == "mask_contours":
        _add_contours(
            plotter, observation["epi"], transform, offset, scale, EPI_CONTOUR_COLOR
        )
        _add_contours(
            plotter, observation["endo"], transform, offset, scale, ENDO_CONTOUR_COLOR
        )
    elif observation["mode"] == "prepared_points":
        _add_points(plotter, observation["epi"], EPI_POINT_COLOR)
        _add_points(plotter, observation["endo"], ENDO_POINT_COLOR)
    else:
        raise ValueError(f"Unknown video observation mode: {observation['mode']}")
    _apply_reference_camera(plotter, reference_center)
    image = plotter.screenshot(return_img=True, transparent_background=True)
    plotter.close()
    return image


def _image_on_white(image):
    image = cv2.resize(image, PANEL_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.shape[2] != 4:
        raise RuntimeError(f"Unsupported screenshot shape: {image.shape}")
    color = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    return np.clip(color * alpha + 255.0 * (1.0 - alpha), 0, 255).astype(np.uint8)


def _labeled_frame(image, case_id: str, phase: int):
    frame = np.full((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), 255, dtype=np.uint8)
    frame[LABEL_HEIGHT:] = _image_on_white(image)
    label = f"4DMM + input | {case_id} | phase {phase:02d}"
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2
    cv2.putText(frame, label, (10, 28), font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)
    cv2.line(frame, (315, 20), (333, 20), EPI_BGR, 4)
    cv2.putText(frame, "epi", (338, 26), font, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.line(frame, (391, 20), (409, 20), ENDO_BGR, 4)
    cv2.putText(frame, "endo", (414, 26), font, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return frame


def _validate_video(path: Path, expected_count: int) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot reopen rendered video: {path}")
    count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    decoded = 0
    previous = None
    has_frame_change = False
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if previous is not None and not np.array_equal(previous, frame):
            has_frame_change = True
        previous = frame
        decoded += 1
    capture.release()
    problems = []
    if count != expected_count:
        problems.append(f"frames={count}, expected={expected_count}")
    if abs(fps - FPS) > 0.01:
        problems.append(f"fps={fps}, expected={FPS}")
    if (width, height) != VIDEO_SIZE:
        problems.append(f"size={(width, height)}, expected={VIDEO_SIZE}")
    if decoded != expected_count:
        problems.append(f"decoded_frames={decoded}, expected={expected_count}")
    if expected_count > 1 and not has_frame_change:
        problems.append("all adjacent frames are identical")
    if problems:
        raise RuntimeError("Video validation failed: " + "; ".join(problems))
    return {
        "frames": count,
        "fps": fps,
        "size": [width, height],
        "duration_seconds": count / fps,
        "has_frame_change": has_frame_change,
    }


def _mask_observations(config: CaseConfig, preprocess_report: dict) -> dict[int, dict]:
    frames = {frame.phase: frame for frame in discover_frames(config.input_mask_dir)}
    selected_order = config.slice_order
    report_order = preprocess_report.get("slice_order")
    if isinstance(report_order, dict) and report_order.get("selected"):
        selected_order = report_order["selected"]
    observations = {}
    for phase in (int(value) for value in preprocess_report["original_phases"]):
        if phase not in frames:
            raise FileNotFoundError(
                f"Mask phase {phase} is absent from {config.input_mask_dir}"
            )
        contours = extract_contours(frames[phase].path, selected_order)
        observations[phase] = {
            "mode": "mask_contours",
            "epi": [[points] for points in contours.epi if len(points)],
            "endo": [[points] for points in contours.endo if len(points)],
        }
    return observations


def _prepared_observations(preprocess_report: dict) -> dict[int, dict]:
    manifests = preprocess_report.get("manifests") or {}
    if set(manifests) != {"endo", "epi"}:
        raise RuntimeError(
            "Prepared overlay requires preprocessing manifests; run video before "
            "artifacts are cleaned or set retain_artifacts=true"
        )
    by_surface = {}
    for surface in ("endo", "epi"):
        with Path(manifests[surface]).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        sequences = payload.get("test", {}).get("case", {})
        case_names = list(sequences)
        if len(case_names) != 1:
            raise ValueError(f"Prepared manifest has unexpected cases: {case_names}")
        instances = sequences[case_names[0]].get("instance_list", [])
        by_surface[surface] = []
        for path in instances:
            with np.load(path, allow_pickle=False) as archive:
                points = np.asarray(archive["pcd"], dtype=np.float64)[:, :3]
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError(f"Invalid prepared overlay point cloud: {path}")
            by_surface[surface].append(points)
    mapping = preprocess_report["phase_mapping"]
    if any(len(by_surface[surface]) != len(mapping) for surface in ("endo", "epi")):
        raise ValueError("Prepared overlay point counts do not match phase mapping")
    observations = {}
    for index, item in enumerate(mapping):
        observations[int(item["original_phase"])] = {
            "mode": "prepared_points",
            "epi": by_surface["epi"][index],
            "endo": by_surface["endo"][index],
        }
    return observations


def _load_observations(config: CaseConfig, preprocess_report: dict) -> dict[int, dict]:
    if config.preprocess_mode == "mask":
        return _mask_observations(config, preprocess_report)
    if config.preprocess_mode == "prepared":
        return _prepared_observations(preprocess_report)
    raise ValueError(f"Unsupported preprocess mode for video: {config.preprocess_mode}")


def run_case_video(config: CaseConfig, preprocess_report: dict) -> dict:
    backend = _configure_offscreen_backend()
    mapping = preprocess_report["phase_mapping"]
    if config.video_order == "reference":
        phases = [int(item["original_phase"]) for item in mapping]
    else:
        phases = sorted(int(value) for value in preprocess_report["original_phases"])
    transform_info = preprocess_report["transform"]
    transform = np.asarray(transform_info["P"], dtype=np.float64)
    offset = np.asarray(transform_info["offset"], dtype=np.float64).reshape(1, 3)
    scale = float(transform_info["scale"])
    observations = _load_observations(config, preprocess_report)
    reference_phase = int(config.reference_phase)
    reference_meshes = _world_to_normalized(
        _load_stitched_phase(config, reference_phase), transform, offset, scale
    )
    reference_center = _center(reference_meshes)

    video_root = config.output_dir / "Mesh_video"
    video_root.mkdir(parents=True, exist_ok=True)
    output = video_root / f"{config.case_id}_4DMM.mp4"
    preview = video_root / f"{config.case_id}_phase_{reference_phase:02d}.png"
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    writer = cv2.VideoWriter(
        str(partial), cv2.VideoWriter_fourcc(*VIDEO_CODEC), FPS, VIDEO_SIZE
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot initialize video writer: {partial}")
    try:
        for index, phase in enumerate(phases, start=1):
            print(
                f"[video] {config.case_id} {index:02d}/{len(phases)} phase={phase:02d}",
                flush=True,
            )
            meshes = reference_meshes if phase == reference_phase else _world_to_normalized(
                _load_stitched_phase(config, phase), transform, offset, scale
            )
            if phase not in observations:
                raise ValueError(f"No input observation for phase {phase}")
            frame = _labeled_frame(
                _render_overlay(
                    meshes,
                    observations[phase],
                    transform,
                    offset,
                    scale,
                    reference_center,
                ),
                config.case_id,
                phase,
            )
            writer.write(frame)
            if phase == reference_phase and not cv2.imwrite(str(preview), frame):
                raise RuntimeError(f"Could not write preview: {preview}")
    finally:
        writer.release()
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(f"Video writer produced no data: {partial}")
    summary = _validate_video(partial, len(phases))
    partial.replace(output)
    return {
        "case_id": config.case_id,
        "video": str(output),
        "preview": str(preview),
        "phase_order": phases,
        "render_mode": "input_overlay",
        "observation_mode": observations[phases[0]]["mode"],
        "backend": backend,
        "camera": "UVRecons fixed reference-phase paper view",
        "zoom": MESH_ZOOM,
        **summary,
        "all_passed": True,
    }


def main() -> None:
    for config in load_case_configs():
        if config.enabled and config.stages["video"]:
            print(run_case_video(config, load_preprocess_report(config)))


if __name__ == "__main__":
    main()
