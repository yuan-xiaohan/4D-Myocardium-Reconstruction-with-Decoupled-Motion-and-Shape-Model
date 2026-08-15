"""Render one postprocessed 4DMM case in the UVRecons reference style."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pyvista as pv

from case_config import CaseConfig, load_case_configs
from postprocess_4dmm_bridge import validate_combined_mesh
from preprocess_mask import load_preprocess_report


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
RENDER_WINDOW_SIZE = (1200, 800)
PANEL_IMAGE_SIZE = (480, 320)
LABEL_HEIGHT = 42
VIDEO_SIZE = (PANEL_IMAGE_SIZE[0], PANEL_IMAGE_SIZE[1] + LABEL_HEIGHT)


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
        points = np.asarray(copied.points, dtype=np.float64)
        homogeneous = np.c_[points, np.ones(len(points), dtype=np.float64)]
        copied.points = ((homogeneous @ transform.T)[:, :3] + offset) * scale
        normalized[surface] = copied
    return normalized


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
    label = f"4DMM | {case_id} | phase {phase:02d}"
    font, font_scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
    cv2.putText(
        frame,
        label,
        ((VIDEO_SIZE[0] - text_width) // 2, (LABEL_HEIGHT + text_height) // 2 - 1),
        font,
        font_scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )
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


def run_case_video(config: CaseConfig, preprocess_report: dict) -> dict:
    mapping = preprocess_report["phase_mapping"]
    if config.video_order == "reference":
        phases = [int(item["original_phase"]) for item in mapping]
    else:
        phases = sorted(int(value) for value in preprocess_report["original_phases"])
    transform_info = preprocess_report["transform"]
    transform = np.asarray(transform_info["P"], dtype=np.float64)
    offset = np.asarray(transform_info["offset"], dtype=np.float64).reshape(1, 3)
    scale = float(transform_info["scale"])
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
            frame = _labeled_frame(_render(meshes, reference_center), config.case_id, phase)
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
