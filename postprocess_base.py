"""Apply the original 4DMM base clip-and-stitch algorithm to one case."""

from __future__ import annotations

from pathlib import Path

import pyvista as pv

from case_config import CaseConfig, load_case_configs
from postprocess_4dmm_bridge import (
    POSTPROCESS_ALGORITHM_NAME,
    POSTPROCESS_ALGORITHM_VERSION,
    clip_and_merge_surface_pair,
    validate_combined_mesh,
)
from preprocess_mask import load_preprocess_report


def _surface_path(config: CaseConfig, surface: str, phase: int) -> Path:
    return (
        config.output_dir / "artifacts" / "raw" / surface
        / f"{config.case_id}-{surface}-{phase:02d}.obj"
    )


def _vtk_path(config: CaseConfig, phase: int) -> Path:
    return config.output_dir / "vtk" / f"frame_{phase:02d}.vtk"


def _save_atomically(mesh, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.tmp.vtk")
    mesh.save(str(temporary), binary=True)
    saved = pv.read(str(temporary)).extract_surface().triangulate()
    validation = validate_combined_mesh(saved)
    temporary.replace(output_path)
    return validation


def process_phase(config: CaseConfig, phase: int) -> dict:
    output_path = _vtk_path(config, phase)
    if config.skip_existing and output_path.is_file():
        try:
            validation = validate_combined_mesh(
                pv.read(str(output_path)).extract_surface().triangulate()
            )
            return {
                "phase": phase,
                "status": "skipped_valid",
                "output": str(output_path),
                **validation,
            }
        except Exception:
            pass
    epi_path = _surface_path(config, "epi", phase)
    endo_path = _surface_path(config, "endo", phase)
    if not epi_path.is_file() or not endo_path.is_file():
        raise FileNotFoundError(f"Missing raw surface pair: {epi_path}, {endo_path}")
    mesh, diagnostics = clip_and_merge_surface_pair(epi_path, endo_path)
    validation = _save_atomically(mesh, output_path)
    return {
        "phase": phase,
        "status": "processed",
        "output": str(output_path),
        **validation,
        "diagnostics": diagnostics,
    }


def run_case_postprocess(config: CaseConfig, preprocess_report: dict) -> dict:
    phases = sorted(int(value) for value in preprocess_report["original_phases"])
    results = [process_phase(config, phase) for phase in phases]
    report = {
        "case_id": config.case_id,
        "phases": phases,
        "algorithm": POSTPROCESS_ALGORITHM_NAME,
        "algorithm_version": POSTPROCESS_ALGORITHM_VERSION,
        "results": results,
        "all_passed": all(
            item["n_open_edges"] == 0 and item["is_manifold"] for item in results
        ),
    }
    if not report["all_passed"]:
        raise RuntimeError(
            f"4DMM postprocess validation failed: {config.output_dir / 'report.json'}"
        )
    return report


def main() -> None:
    for config in load_case_configs():
        if config.enabled and config.stages["postprocess"]:
            completed = run_case_postprocess(config, load_preprocess_report(config))
            print({
                "case_id": config.case_id,
                "phases": len(completed["results"]),
                "all_passed": completed["all_passed"],
            })


if __name__ == "__main__":
    main()
