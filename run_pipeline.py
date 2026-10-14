"""Public command-line entry point for mask-to-4DMM case inference."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
import traceback
from pathlib import Path

from case_config import CaseConfig, load_case_configs
from postprocess import run_case_postprocess
from postprocess.video import run_case_video
from preprocess_mask import load_preprocess_report, run_preprocessing
from project_config import (
    CASE_CONFIG_PATH,
    CONTINUE_ON_CASE_ERROR,
    INPUT_ROOT,
    OUTPUT_ROOT,
)
from reconstruct_4dmm import run_case_inference


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _compact_preprocess_report(report: dict, retain_artifacts: bool) -> dict:
    result = dict(report)
    result["artifacts_retained"] = retain_artifacts
    result.pop("registration_asset_sha256", None)
    if not retain_artifacts:
        result["manifests"] = {}
    return result


def _compact_inference_report(report: dict | None, retain_artifacts: bool) -> dict | None:
    if report is None:
        return None
    result = dict(report)
    result["artifacts_retained"] = retain_artifacts
    result["models"] = {
        surface: {
            key: value for key, value in metadata.items() if key != "sha256"
        }
        for surface, metadata in result.get("models", {}).items()
    }
    if not retain_artifacts:
        result["results"] = [
            {key: value for key, value in item.items() if key != "mesh"}
            for item in result.get("results", [])
        ]
    return result


def run_case(config: CaseConfig) -> dict:
    started = time.perf_counter()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    existing_report = {}
    report_path = config.output_dir / "report.json"
    if report_path.is_file():
        try:
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_report = {}
    stages = dict(existing_report.get("stages") or {})
    inference_report = existing_report.get("inference")
    postprocess_report = existing_report.get("postprocess")
    video_report = existing_report.get("video")
    postprocess_has_failures = bool(
        postprocess_report is not None
        and not postprocess_report.get("all_passed", True)
    )

    def persist_report(status: str = "running") -> dict:
        retain_artifacts = config.retain_artifacts or postprocess_has_failures
        report = {
            "case_id": config.case_id,
            "status": status,
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "config": config.to_json(),
            "phase_mapping": preprocess_report["phase_mapping"],
            "orientation": preprocess_report["orientation"],
            "stages": stages,
            "preprocess": _compact_preprocess_report(
                preprocess_report, retain_artifacts
            ),
            "inference": _compact_inference_report(
                inference_report, retain_artifacts
            ),
            "postprocess": postprocess_report,
            "video": video_report,
        }
        # A successful retry replaces, rather than inherits, a previous
        # run-level failure.  Keep this explicit so error metadata cannot
        # survive if report assembly changes to merge existing fields later.
        if status in {"success", "completed_with_postprocess_failures"}:
            for key in ("error", "error_type", "traceback"):
                report.pop(key, None)
        _atomic_json(config.output_dir / "report.json", report)
        return report

    if config.stages["preprocess"]:
        preprocess_report = run_preprocessing(config)
        stages["preprocess"] = {
            "status": "completed",
            "low_confidence_mirror": bool(
                preprocess_report["orientation"].get("low_confidence", False)
            ),
        }
    else:
        preprocess_report = load_preprocess_report(config)
        stages["preprocess"] = {"status": "reused"}

    # Persist every completed stage so a later failure cannot discard the
    # transform, latent-code report, or per-frame postprocess diagnostics.
    persist_report()

    if config.stages["inference"]:
        inference_report = run_case_inference(config, preprocess_report)
        if not inference_report["all_passed"]:
            raise RuntimeError(f"Inference validation failed for {config.case_id}")
        stages["inference"] = {
            "status": "completed",
            "results": len(inference_report["results"]),
        }
    else:
        stages["inference"] = {"status": "disabled"}
        if inference_report is not None:
            stages["inference"] = existing_report.get("stages", {}).get(
                "inference", {"status": "reused"}
            )
    persist_report()

    if config.stages["postprocess"]:
        postprocess_report = run_case_postprocess(config, preprocess_report)
        postprocess_has_failures = not postprocess_report["all_passed"]
        stages["postprocess"] = {
            "status": (
                "completed_with_failures"
                if postprocess_has_failures
                else "completed"
            ),
            "phases": len(postprocess_report["results"]),
            "strict_passed": postprocess_report.get("strict_passed_count"),
            "raw_concat": postprocess_report.get("raw_concat_count"),
            "missing_outputs": postprocess_report.get("missing_output_count"),
        }
    else:
        stages["postprocess"] = {"status": "disabled"}
        if postprocess_report is not None:
            stages["postprocess"] = existing_report.get("stages", {}).get(
                "postprocess", {"status": "reused"}
            )
    persist_report()

    if config.stages["video"]:
        try:
            video_report = run_case_video(config, preprocess_report)
            stages["video"] = {"status": "completed", **video_report}
        except Exception as error:
            stages["video"] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            persist_report(status="failed")
            raise
    else:
        stages["video"] = {"status": "disabled"}
        if video_report is not None:
            stages["video"] = existing_report.get("stages", {}).get(
                "video", {"status": "reused"}
            )
    report = persist_report(
        status=(
            "completed_with_postprocess_failures"
            if postprocess_has_failures
            else "success"
        )
    )
    if (
        not config.retain_artifacts
        and config.stages["inference"]
        and config.stages["postprocess"]
        and not postprocess_has_failures
    ):
        shutil.rmtree(config.output_dir / "artifacts", ignore_errors=True)
    return report


def run_all_cases(
    config_path: Path = CASE_CONFIG_PATH,
    input_root: Path = INPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    case_ids: list[str] | None = None,
    continue_on_error: bool = CONTINUE_ON_CASE_ERROR,
) -> dict:
    started = time.perf_counter()
    configs = load_case_configs(
        config_path=Path(config_path),
        input_root=Path(input_root),
        output_root=Path(output_root),
    )
    if case_ids:
        requested = set(case_ids)
        available = {config.case_id for config in configs}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested cases were not discovered: {missing}")
        configs = [config for config in configs if config.case_id in requested]
    results, failures, skipped = [], [], []
    for config in configs:
        if not config.enabled:
            skipped.append(config.case_id)
            continue
        print(f"[case] starting {config.case_id}", flush=True)
        try:
            result = run_case(config)
            if result["status"] == "success":
                results.append(result)
                print(
                    f"[case] completed {config.case_id} in "
                    f"{result['seconds']:.1f}s",
                    flush=True,
                )
            else:
                postprocess = result.get("postprocess") or {}
                failure = {
                    "case_id": config.case_id,
                    "status": "failed",
                    "error_type": "PostprocessFailure",
                    "error": (
                        "Strict postprocess failed for "
                        f"{postprocess.get('raw_concat_count', 0)} frame(s); "
                        f"missing outputs={postprocess.get('missing_output_count', 0)}"
                    ),
                    "report": str(config.output_dir / "report.json"),
                }
                failures.append(failure)
                print(
                    f"[case] completed_with_postprocess_failures "
                    f"{config.case_id}: {failure['error']}",
                    flush=True,
                )
                if not continue_on_error:
                    break
        except Exception as error:
            failure = {
                "case_id": config.case_id,
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            partial_report = {}
            report_path = config.output_dir / "report.json"
            if report_path.is_file():
                try:
                    with report_path.open(encoding="utf-8") as handle:
                        partial_report = json.load(handle)
                except (OSError, json.JSONDecodeError):
                    partial_report = {}
            for key in (
                "phase_mapping",
                "orientation",
                "preprocess",
                "inference",
                "postprocess",
                "video",
                "stages",
            ):
                if partial_report.get(key) is not None:
                    failure[key] = partial_report[key]
            failures.append(failure)
            failure["config"] = config.to_json()
            _atomic_json(config.output_dir / "report.json", failure)
            print(f"[case] failed {config.case_id}: {error}", flush=True)
            if not continue_on_error:
                break
    summary = {
        "status": "success" if not failures else "failed",
        "seconds": time.perf_counter() - started,
        "discovered_cases": [config.case_id for config in configs],
        "successful_cases": [item["case_id"] for item in results],
        "skipped_cases": skipped,
        "failures": failures,
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the mask-to-4DMM pipeline for one or more cases."
    )
    parser.add_argument(
        "--config", type=Path, default=CASE_CONFIG_PATH,
        help="case configuration JSON (default: configs/cases.json)",
    )
    parser.add_argument(
        "--input-root", type=Path, default=INPUT_ROOT,
        help=(
            "directory containing <case>/mask(s) or mask(s)/<case> "
            "(default: inputs)"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=OUTPUT_ROOT,
        help="output directory (default: outputs)",
    )
    parser.add_argument(
        "--case", dest="case_ids", action="append",
        help="run only this case; repeat the option for multiple cases",
    )
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="stop after the first failed case",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    summary = run_all_cases(
        config_path=args.config,
        input_root=args.input_root,
        output_root=args.output_root,
        case_ids=args.case_ids,
        continue_on_error=not args.fail_fast,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
