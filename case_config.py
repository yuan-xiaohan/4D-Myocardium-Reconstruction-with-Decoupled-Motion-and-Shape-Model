"""Discover cases and resolve default-plus-per-case pipeline configuration."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from project_config import CASE_CONFIG_PATH, INPUT_ROOT, OUTPUT_ROOT, REPO_ROOT


SURFACES = ("endo", "epi")
MIRROR_MODES = {"auto", "normal", "mirrored"}
SLICE_ORDERS = {"auto", "atob", "btoa"}
PREPROCESS_MODES = {"mask", "prepared"}
VIDEO_ORDERS = {"original", "reference"}
STAGE_NAMES = ("preprocess", "inference", "postprocess", "video")

DEFAULTS = {
    "enabled": True,
    "input_mask_dir": None,
    "reference_phase": 0,
    "mirror": "auto",
    "slice_order": "auto",
    "num_iterations": 2000,
    "phase_batch_size": 4,
    "resolution": 64,
    "preprocess_mode": "mask",
    "prepared_root": None,
    "device": "auto",
    "skip_existing": True,
    "retain_artifacts": False,
    "video_order": "original",
    "checkpoints": {
        "endo": "examples/acdc/4DMM/endo",
        "epi": "examples/acdc/4DMM/epi",
    },
    "stages": {
        "preprocess": True,
        "inference": True,
        "postprocess": True,
        "video": True,
    },
}


@dataclass(frozen=True)
class CaseConfig:
    case_id: str
    input_mask_dir: Path
    output_dir: Path
    enabled: bool
    reference_phase: int
    mirror: str
    slice_order: str
    num_iterations: int
    phase_batch_size: int
    resolution: int
    preprocess_mode: str
    prepared_root: Path | None
    device: str
    skip_existing: bool
    retain_artifacts: bool
    video_order: str
    checkpoints: dict[str, Path]
    stages: dict[str, bool]

    def to_json(self) -> dict:
        result = asdict(self)
        result["input_mask_dir"] = str(self.input_mask_dir)
        result["output_dir"] = str(self.output_dir)
        result["prepared_root"] = (
            str(self.prepared_root) if self.prepared_root is not None else None
        )
        result["checkpoints"] = {
            surface: str(path) for surface, path in self.checkpoints.items()
        }
        return result


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Case configuration does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError("Case configuration root must be an object")
    unknown = set(payload) - {"default", "cases"}
    if unknown:
        raise ValueError(f"Unknown top-level case configuration fields: {sorted(unknown)}")
    if not isinstance(payload.get("default", {}), dict):
        raise TypeError("default must be an object")
    if not isinstance(payload.get("cases", {}), dict):
        raise TypeError("cases must be an object")
    return payload


def _merge_known(base: dict, override: dict, location: str) -> dict:
    if not isinstance(override, dict):
        raise TypeError(f"{location} must be an object")
    unknown = set(override) - set(base)
    if unknown:
        raise ValueError(f"Unknown fields in {location}: {sorted(unknown)}")
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(result[key], dict):
            result[key] = _merge_known(result[key], value, f"{location}.{key}")
        else:
            result[key] = value
    return result


def _resolve_path(value: str | Path | None, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def discover_case_ids(input_root: Path, case_overrides: dict) -> list[str]:
    discovered = set()
    if input_root.is_dir():
        discovered.update(
            path.name for path in input_root.iterdir()
            if path.is_dir() and (path / "masks").is_dir()
        )
    for case_id, override in case_overrides.items():
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"Invalid case id: {case_id!r}")
        if not isinstance(override, dict):
            raise TypeError(f"cases.{case_id} must be an object")
        # Explicit entries remain visible even when disabled or when their
        # input directory is supplied later. Enabled entries are validated
        # below; disabled entries appear in the batch summary as skipped.
        discovered.add(case_id)
    return sorted(discovered)


def _validate_resolved(case_id: str, values: dict) -> None:
    if not isinstance(values["enabled"], bool):
        raise TypeError(f"{case_id}.enabled must be boolean")
    for key in ("reference_phase", "num_iterations", "phase_batch_size", "resolution"):
        if isinstance(values[key], bool) or not isinstance(values[key], int):
            raise TypeError(f"{case_id}.{key} must be an integer")
    if values["reference_phase"] < 0:
        raise ValueError(f"{case_id}.reference_phase must be non-negative")
    if values["num_iterations"] < 1 or values["phase_batch_size"] < 1:
        raise ValueError(f"{case_id} iteration and batch values must be positive")
    if values["resolution"] < 16:
        raise ValueError(f"{case_id}.resolution must be at least 16")
    if values["mirror"] not in MIRROR_MODES:
        raise ValueError(f"{case_id}.mirror must be one of {sorted(MIRROR_MODES)}")
    if values["slice_order"] not in SLICE_ORDERS:
        raise ValueError(f"{case_id}.slice_order must be one of {sorted(SLICE_ORDERS)}")
    if values["preprocess_mode"] not in PREPROCESS_MODES:
        raise ValueError(
            f"{case_id}.preprocess_mode must be one of {sorted(PREPROCESS_MODES)}"
        )
    if values["video_order"] not in VIDEO_ORDERS:
        raise ValueError(f"{case_id}.video_order must be one of {sorted(VIDEO_ORDERS)}")
    if not isinstance(values["device"], str) or not values["device"]:
        raise TypeError(f"{case_id}.device must be a non-empty string")
    if not isinstance(values["skip_existing"], bool):
        raise TypeError(f"{case_id}.skip_existing must be boolean")
    if not isinstance(values["retain_artifacts"], bool):
        raise TypeError(f"{case_id}.retain_artifacts must be boolean")
    for key in ("input_mask_dir", "prepared_root"):
        if values[key] is not None and not isinstance(values[key], str):
            raise TypeError(f"{case_id}.{key} must be a path string or null")
    if set(values["checkpoints"]) != set(SURFACES):
        raise ValueError(f"{case_id}.checkpoints must contain endo and epi")
    if not all(
        isinstance(value, str) and bool(value)
        for value in values["checkpoints"].values()
    ):
        raise TypeError(f"{case_id}.checkpoints values must be non-empty path strings")
    if set(values["stages"]) != set(STAGE_NAMES):
        raise ValueError(f"{case_id}.stages must contain {STAGE_NAMES}")
    if not all(isinstance(value, bool) for value in values["stages"].values()):
        raise TypeError(f"{case_id}.stages values must be boolean")


def load_case_configs(
    config_path: Path = CASE_CONFIG_PATH,
    input_root: Path = INPUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
) -> list[CaseConfig]:
    payload = _read_json(Path(config_path))
    defaults = _merge_known(DEFAULTS, payload.get("default", {}), "default")
    overrides = payload.get("cases", {})
    case_ids = discover_case_ids(Path(input_root), overrides)
    if not case_ids:
        raise RuntimeError(
            f"No cases found below {input_root}; expected inputs/<case>/masks or "
            "a cases.<id>.input_mask_dir override"
        )

    configs = []
    for case_id in case_ids:
        values = _merge_known(defaults, overrides.get(case_id, {}), f"cases.{case_id}")
        _validate_resolved(case_id, values)
        mask_dir = _resolve_path(
            values["input_mask_dir"], Path(input_root) / case_id / "masks"
        )
        prepared_root = _resolve_path(values["prepared_root"])
        checkpoint_dirs = {
            surface: _resolve_path(value)
            for surface, value in values["checkpoints"].items()
        }
        if values["enabled"] and values["preprocess_mode"] == "mask" and not mask_dir.is_dir():
            raise FileNotFoundError(f"Mask directory for {case_id} does not exist: {mask_dir}")
        if values["enabled"] and values["preprocess_mode"] == "prepared":
            if prepared_root is None or not prepared_root.is_dir():
                raise FileNotFoundError(
                    f"Prepared root for {case_id} does not exist: {prepared_root}"
                )
        if values["enabled"] and values["stages"]["inference"]:
            for surface, path in checkpoint_dirs.items():
                if not path.is_dir():
                    raise FileNotFoundError(
                        f"Checkpoint directory for {case_id}/{surface} does not exist: {path}"
                    )
        configs.append(CaseConfig(
            case_id=case_id,
            input_mask_dir=mask_dir,
            output_dir=Path(output_root) / case_id,
            enabled=values["enabled"],
            reference_phase=values["reference_phase"],
            mirror=values["mirror"],
            slice_order=values["slice_order"],
            num_iterations=values["num_iterations"],
            phase_batch_size=values["phase_batch_size"],
            resolution=values["resolution"],
            preprocess_mode=values["preprocess_mode"],
            prepared_root=prepared_root,
            device=values["device"],
            skip_existing=values["skip_existing"],
            retain_artifacts=values["retain_artifacts"],
            video_order=values["video_order"],
            checkpoints=checkpoint_dirs,
            stages=values["stages"],
        ))
    return configs
