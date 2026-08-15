# 4D Myocardium Reconstruction

> [!IMPORTANT]
> **🚀 NEW — Endo/epi 4DMM release pipeline**
>
> This release keeps the original 4DMM decoder and latent optimization while
> adding independent endocardial and epicardial models, mask preprocessing,
> base stitching, labeled VTK output, and fixed-camera videos.

The repository provides both model training and case inference. Endo and epi
are always separate models, checkpoints, and latent codes; users do not need to
run two commands to process them.

## Environment

```bash
conda env create -f environment.yml
conda activate 4DMM-release
```

Run commands from the repository root. Each public entry point supports
`--help` and can be used with or without a configuration file override.

## Inference from masks

Put one case below `inputs/`:

```text
inputs/
└── CaseA/
    └── masks/
        ├── 00.nii.gz
        ├── 01.nii.gz
        └── ...
```

Each NIfTI frame must use the labels `0=background`, `1=RV cavity`,
`2=LV myocardium`, and `3=LV cavity`. Frames from one case must have matching
image geometry. The default reference frame is `0`; change it per case in
`configs/cases.json` when ED has another frame number.

The default model settings are 2000 optimization iterations, motion batch size
4, and mesh resolution 64. They can be overridden independently for each case.
The bundled ACDC models are inference-only. To run all discovered cases:

```bash
python run_pipeline.py
```

To run only one discovered case:

```bash
python run_pipeline.py --case CaseA
```

Endo and epi inference runs automatically on the selected device. The pipeline
preprocesses the masks, optimizes both surfaces, stitches their base boundaries,
and writes one labeled VTK mesh per frame.

Default output:

```text
outputs/CaseA/
├── vtk/frame_*.vtk
├── Mesh_video/CaseA_4DMM.mp4   # when video is enabled
└── report.json
```

Intermediate point clouds, NPZ files, raw OBJ files, and latent codes are
removed after a successful run. Set `retain_artifacts` to `true` in the case
configuration when those files are needed for debugging or a staged rerun.

## Case configuration

`configs/cases.json` supplies defaults and optional per-case overrides:

```json
{
  "default": {
    "reference_phase": 0,
    "mirror": "auto",
    "slice_order": "auto",
    "num_iterations": 2000,
    "phase_batch_size": 4,
    "resolution": 64,
    "retain_artifacts": false,
    "stages": {
      "preprocess": true,
      "inference": true,
      "postprocess": true,
      "video": true
    }
  },
  "cases": {
    "CaseB": {"reference_phase": 2},
    "CaseC": {"num_iterations": 1000, "stages": {"video": false}}
  }
}
```

The model paths are already set to `examples/acdc/4DMM/{endo,epi}`. A trained
user model can be selected by changing the two checkpoint paths for a case.

## Training

Training uses prepared canonical triangle meshes, not inference masks. Place the
training data in this layout:

```text
training_data/
├── obj_norm_uv/CaseA/
│   ├── 00-endo.obj
│   ├── 00-epi.obj
│   └── ...
└── P/P_lv/CaseA.txt
```

Run the preparation step first. It creates the training files used by both
surfaces:

```bash
python process_data.py
```

Use `--case`, `--phase`, or `--surface` to prepare a subset when needed.

Then edit `configs/training.json` if needed and start the unified trainer:

```bash
python train_4dmm.py
```

Common settings can also be overridden directly, for example:

```bash
python train_4dmm.py --target-epoch 100 --batch-size 8
```

The default target is 50 epochs, batch size 16, and 8000 samples per scene.
With two visible GPUs, endo and epi train independently in parallel; with one
GPU or CPU they run sequentially. A training run is written to:

```text
outputs/training/<run-id>/
├── endo/
│   ├── ModelParameters/latest.pth
│   ├── OptimizerParameters/latest.pth
│   ├── LatentCodes/latest_cs.pth
│   ├── LatentCodes/latest_cm.pth
│   ├── specs.json
│   ├── history.csv
│   └── loss_curve.png
├── epi/
│   └── ...
└── report.json
```

Training starts from scratch by default; set `initialization_mode` to `resume`
only to continue one of your own saved runs. The bundled ACDC models do not
contain optimizer state and cannot be used as training resume checkpoints.

## Attribution

The decoder and core 4DMM formulation are based on:

```bibtex
@inproceedings{yuan2023myo4d,
  title={4D Myocardium Reconstruction with Decoupled Motion and Shape Model},
  author={Yuan, Xiaohan and Liu, Cong and Wang, Yangang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```
