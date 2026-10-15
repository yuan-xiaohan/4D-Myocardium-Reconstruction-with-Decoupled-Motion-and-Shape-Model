# Prepared training data

The preferred input is a labeled VTK dataset. Both layouts below are accepted:

```text
4DM Dataset/<case>/mesh/frame_*.vtk
4DM Dataset/<case>/mask/*.nii.gz
4DM Dataset/<case>/P.txt
```

or:

```text
4DM Dataset/mesh/<case>/frame_*.vtk
4DM Dataset/mask/<case>/*.nii.gz
4DM Dataset/P/<case>.txt
```

`meshes/` is accepted in place of `mesh/`, and `masks/` in place of `mask/`.
Each VTK must contain `surface_id=0/1` and `label=1/2` for epi/endo.

If `P.txt` is absent, `process_data.py` generates it once from the
reference-phase Mask and uses it for every frame. An existing `P.txt` is loaded
directly. If neither the transform nor the Mask is available, processing stops
with an explicit error.

Automatic RV top-view mirror scoring fails on low-confidence anatomy unless
`--mirror-fallback normal|mirrored` is set explicitly.

Run `process_data.py --dataset-root ... --training-data-root ...` to create
the SDF archives and manifests. The transform is read once per case and used
for every frame. The generated `processed/` archives are the training cache;
the processor does not create normalized OBJ files.

Users who already have separated normalized OBJ files may continue to use:

```text
training_data/normalized_meshes/<case>/<phase>-endo.obj
training_data/normalized_meshes/<case>/<phase>-epi.obj
training_data/P/<case>.txt
```
