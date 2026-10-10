# 4D Myocardium Reconstruction (4DMM)

[4D Myocardium Reconstruction with Decoupled Motion and Shape Model (ICCV 2023)](https://openaccess.thecvf.com/content/ICCV2023/papers/Yuan_4D_Myocardium_Reconstruction_with_Decoupled_Motion_and_Shape_Model_ICCV_2023_paper.pdf)

<div align="center">
  <img src="images/teaser.jpg" alt="4D myocardium reconstruction with decoupled motion and shape models" width="50%">
</div>

> [!IMPORTANT]
> **🚀 NEW — Endo/epi 4DMM release pipeline**
>
> This release keeps the original 4DMM decoder and latent optimization while
> adding independent endocardial and epicardial models, mask preprocessing,
> base stitching, labeled VTK output, and fixed-camera videos.

Given a sequence of 3D cardiac segmentation masks, the pipeline reconstructs
the left-ventricular myocardium as a 3D mesh at every cardiac phase.

## Installation

```bash
conda env create -f environment.yml
conda activate 4DMM-release
```

Run all commands from the repository root.

## Inference

### Input

Create one `masks` directory for each case:

```text
inputs/
└── CaseA/
    └── masks/
        ├── 00.nii.gz
        ├── 01.nii.gz
        └── ...
```

Mask labels are `0=background`, `1=RV cavity`, `2=LV myocardium`, and
`3=LV cavity`. All frames in a case must have matching image geometry. The
validated workflow uses phase `00` as the end-diastolic temporal reference.

### Run

Run every discovered case:

```bash
python run_pipeline.py
```

Run one case:

```bash
python run_pipeline.py --case CaseA
```

Use `--config`, `--input-root`, and `--output-root` for custom locations. Run
`python run_pipeline.py --help` for the complete command-line interface.

The pipeline preprocesses the masks, reconstructs endo and epi, stitches the
base boundaries, validates the meshes, and renders the overlay video. Invalid
meshes cause the case to fail instead of being written as successful output.

### Configuration

Defaults and per-case overrides are read from `configs/cases.json`:

```json
{
  "default": {
    "reference_phase": 0,
    "mirror": "auto",
    "slice_order": "auto",
    "num_iterations": 2000,
    "phase_batch_size": 4,
    "resolution": 64,
    "device": "auto",
    "retain_artifacts": false
  },
  "cases": {
    "CaseA": {"mirror": "normal"}
  }
}
```

`mirror` controls whether the anatomy is reflected during alignment.
`slice_order` controls whether the slices run from apex to base (`atob`) or
base to apex (`btoa`). Both accept `auto`.

> **Troubleshooting:** If the input contours look correct but the reconstructed
> mesh is severely distorted or misplaced, check `mirror` and `slice_order`
> first. An incorrect value or automatic estimate can cause major
> reconstruction errors. The selected settings are recorded in `report.json`.

### Output

```text
outputs/CaseA/
├── vtk/frame_*.vtk
├── Mesh_video/
│   ├── CaseA_4DMM.mp4
│   └── CaseA_phase_00.png
├── codes/
│   ├── endo/CaseA_endo_cs.pth
│   ├── endo/CaseA_endo_cm.pth
│   ├── epi/CaseA_epi_cs.pth
│   └── epi/CaseA_epi_cm.pth
└── report.json
```

Each VTK contains both surfaces: `surface_id=0` is epi and `surface_id=1` is
endo. The corresponding `label` values are `1` and `2`. The normalization
transform (`P`, `Pi`, `offset`, and `scale`) is stored under
`preprocess.transform` in `report.json`.

Latent codes are retained under `codes/`. Intermediate point clouds, NPZ files,
and raw OBJ files are removed after a successful run unless
`retain_artifacts=true`. The video stage requires a working off-screen OpenGL
backend on headless systems.

## Training

Training uses canonical triangle meshes and their normalization transforms:

```text
training_data/
├── obj_norm_uv/CaseA/
│   ├── 00-endo.obj
│   ├── 00-epi.obj
│   └── ...
└── P/P_lv/CaseA.txt
```

Prepare the data and start training:

```bash
python process_data.py
python train_4dmm.py
```

Training options are defined in `configs/training.json` and are also available
through `python train_4dmm.py --help`. Endo and epi train independently.

## Attribution

```bibtex
@inproceedings{yuan2023myo4d,
  title={4D Myocardium Reconstruction with Decoupled Motion and Shape Model},
  author={Yuan, Xiaohan and Liu, Cong and Wang, Yangang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
