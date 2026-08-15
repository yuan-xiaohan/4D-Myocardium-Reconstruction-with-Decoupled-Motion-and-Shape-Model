"""Repository-local paths and user-editable defaults.

All paths are resolved from this file, so scripts can be launched from any working
directory without referring to the legacy cardiac_4d or UVRecons repositories.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# Public inference paths are repository-relative. Users place one ``masks``
# directory below INPUT_ROOT for every case; no source checkout is referenced.
INPUT_ROOT = REPO_ROOT / "inputs"
CASE_CONFIG_PATH = REPO_ROOT / "configs" / "cases.json"
TRAIN_CONFIG_PATH = REPO_ROOT / "configs" / "training.json"
REGISTRATION_ASSET = REPO_ROOT / "assets" / "registration_template.npz"
CONTINUE_ON_CASE_ERROR = True

# Prepared training data is a separate, dataset-level interface. It is kept
# outside the inference case outputs so users can reuse it for new runs.
TRAIN_DATA_ROOT = REPO_ROOT / "training_data"
TRAIN_MESH_ROOT = TRAIN_DATA_ROOT / "obj_norm_uv"
TRANSFORM_ROOT = TRAIN_DATA_ROOT / "P" / "P_lv"
PROCESS_ROOT = TRAIN_DATA_ROOT / "processed"
TRAIN_MANIFEST_ROOT = TRAIN_DATA_ROOT
EXPERIMENT_ROOT = REPO_ROOT / "examples" / "acdc" / "4DMM"
OUTPUT_ROOT = REPO_ROOT / "outputs"

SURFACES = ("endo", "epi")

# Default preprocessing settings. Public entry points can override the common
# values through their command-line interfaces.
SDF_SAMPLE_COUNT = 100_000
SURFACE_SAMPLE_COUNT = 2_000_000
PREPROCESS_WORKERS = 1
MIN_FREE_SPACE_GB = 2
RANDOM_SEED = 31359
