"""Run strict 4DMM base stitching with a diagnostic failure output."""

from __future__ import annotations

from pathlib import Path

import pyvista as pv

from case_config import CaseConfig, load_case_configs
from preprocess_mask import load_preprocess_report

from .contract import (
    POSTPROCESS_ALGORITHM_NAME,
    POSTPROCESS_ALGORITHM_VERSION,
    classify_postprocess_mesh,
    validate_combined_mesh,
    validate_raw_concatenated_mesh,
)
from .diagnostic import concatenate_raw_surface_pair
from .geometry import clip_and_merge_surface_pair


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
    try:
        mesh.save(str(temporary), binary=True)
        saved = pv.read(str(temporary)).extract_surface().triangulate()
        validation = validate_combined_mesh(saved)
        temporary.replace(output_path)
        return validation
    finally:
        temporary.unlink(missing_ok=True)


def _save_raw_concat_atomically(mesh, output_path: Path) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.stem}.raw_concat.tmp.vtk")
    try:
        mesh.save(str(temporary), binary=True)
        saved = pv.read(str(temporary)).extract_surface().triangulate()
        validation = validate_raw_concatenated_mesh(saved)
        temporary.replace(output_path)
        return validation
    finally:
        temporary.unlink(missing_ok=True)


def process_phase(config: CaseConfig, phase: int) -> dict:
    output_path = _vtk_path(config, phase)
    if config.skip_existing and output_path.is_file():
        try:
            output_kind, validation = classify_postprocess_mesh(
                pv.read(str(output_path)).extract_surface().triangulate()
            )
            return {
                "phase": phase,
                "status": (
                    "skipped_valid"
                    if output_kind == "strict"
                    else "skipped_raw_concat"
                ),
                "strict_postprocess_passed": output_kind == "strict",
                "output_kind": output_kind,
                "reused_existing": True,
                "output": str(output_path),
                **validation,
            }
        except Exception:
            pass
    epi_path = _surface_path(config, "epi", phase)
    endo_path = _surface_path(config, "endo", phase)
    if not epi_path.is_file() or not endo_path.is_file():
        raise FileNotFoundError(f"Missing raw surface pair: {epi_path}, {endo_path}")
    try:
        mesh, diagnostics = clip_and_merge_surface_pair(epi_path, endo_path)
        validation = _save_atomically(mesh, output_path)
        return {
            "phase": phase,
            "status": "processed",
            "strict_postprocess_passed": True,
            "output_kind": "strict",
            "reused_existing": False,
            "output": str(output_path),
            **validation,
            "diagnostics": diagnostics,
        }
    except Exception as strict_error:
        print(
            f"[postprocess] case={config.case_id} phase={phase:02d} "
            f"strict_failed: {type(strict_error).__name__}: {strict_error}",
            flush=True,
        )
        try:
            fallback, fallback_diagnostics = concatenate_raw_surface_pair(
                epi_path, endo_path
            )
            fallback_validation = _save_raw_concat_atomically(
                fallback, output_path
            )
            print(
                f"[postprocess] case={config.case_id} phase={phase:02d} "
                f"raw_concat_written: {output_path}",
                flush=True,
            )
            return {
                "phase": phase,
                "status": "strict_failed_raw_concat_written",
                "strict_postprocess_passed": False,
                "output_kind": "raw_concat",
                "reused_existing": False,
                "output": str(output_path),
                "strict_error_type": type(strict_error).__name__,
                "strict_error": str(strict_error),
                **fallback_validation,
                "diagnostics": fallback_diagnostics,
            }
        except Exception as fallback_error:
            print(
                f"[postprocess] case={config.case_id} phase={phase:02d} "
                f"raw_concat_failed: {type(fallback_error).__name__}: "
                f"{fallback_error}",
                flush=True,
            )
            return {
                "phase": phase,
                "status": "strict_failed_raw_concat_failed",
                "strict_postprocess_passed": False,
                "output_kind": None,
                "reused_existing": False,
                "output": None,
                "strict_error_type": type(strict_error).__name__,
                "strict_error": str(strict_error),
                "fallback_error_type": type(fallback_error).__name__,
                "fallback_error": str(fallback_error),
            }


def run_case_postprocess(config: CaseConfig, preprocess_report: dict) -> dict:
    phases = sorted(int(value) for value in preprocess_report["original_phases"])
    results = []
    for phase in phases:
        try:
            results.append(process_phase(config, phase))
        except Exception as error:
            print(
                f"[postprocess] case={config.case_id} phase={phase:02d} "
                f"unexpected_failure: {type(error).__name__}: {error}",
                flush=True,
            )
            results.append({
                "phase": phase,
                "status": "unexpected_failure",
                "strict_postprocess_passed": False,
                "output_kind": None,
                "reused_existing": False,
                "output": None,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    strict_passed = [
        item for item in results
        if item.get("strict_postprocess_passed", item["status"] == "skipped_valid")
    ]
    raw_concat = [
        item for item in results
        if item.get("output_kind") == "raw_concat"
    ]
    missing_outputs = [
        item for item in results if not item.get("output")
    ]
    report = {
        "case_id": config.case_id,
        "phases": phases,
        "algorithm": POSTPROCESS_ALGORITHM_NAME,
        "algorithm_version": POSTPROCESS_ALGORITHM_VERSION,
        "results": results,
        "strict_passed_count": len(strict_passed),
        "raw_concat_count": len(raw_concat),
        "missing_output_count": len(missing_outputs),
        "all_outputs_written": not missing_outputs,
        "all_passed": len(strict_passed) == len(results),
    }
    print(
        f"[postprocess] case={config.case_id} complete: "
        f"strict={len(strict_passed)}/{len(results)}, "
        f"raw_concat={len(raw_concat)}, missing={len(missing_outputs)}",
        flush=True,
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
