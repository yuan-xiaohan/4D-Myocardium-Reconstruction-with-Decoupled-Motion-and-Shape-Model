# Prepared training data

Training expects canonical normalized triangle meshes under
`obj_norm_uv/<case>/` and the matching transform under `P/P_lv/<case>.txt`.
Each case contains contiguous `<phase>-endo.obj` and `<phase>-epi.obj` files.
`process_data.py` creates the ignored `processed/` archives and surface
manifests. Raw and generated datasets remain ignored by Git.
