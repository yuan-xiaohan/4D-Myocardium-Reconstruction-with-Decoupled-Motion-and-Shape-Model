# EXP-001: Port `rv_topview_auto_v1` mirror inference

## Goal

Replace production anatomical insertion voting with the standalone
`rv_topview_auto_v1` decision while preserving explicit normal/mirrored
overrides and low-confidence fallback behavior.

## Protocol

- Reuse the bundled registration template, which is numerically identical to
  the standalone centerline and normalized LV/RV assets.
- Score both rigid candidates using the centered per-slice RV-centroid
  trajectory in the canonical normalized top view.
- Accept the lower-error candidate at normalized margin `>= 0.05`; otherwise
  fail or use the configured explicit fallback.
- Remove the older LPS insertion voter and its registration-asset dependency.

## Results

- Syntax checks passed for the mirror, preprocessing, training-data CLI, and tests.
- Targeted mirror/registration tests: `6 passed`.
- Full suite: `38 passed, 1 failed`; the failure is the existing
  `test_explicit_disabled_case_is_reportable_without_input` mismatch between
  the committed test and `discover_inference_masks`, unrelated to mirror logic.
- ACDC `000`, MMs `A0S9V9`, MMs2 `001`, and NUH
  `IMMC001_20150716` matched standalone direction, status, valid slices, both
  candidate scores, and normalized margin exactly (`max delta = 0`).
- ACDC `000` completed real transform generation, atomic `P.txt` write, and
  readback; it selected `mirrored` with margin `0.7335621659590649`.
- NUH `IMMC070_20170629` exercised real low-confidence behavior: strict mode
  failed explicitly and `mirror_fallback=normal` continued with margin
  `0.027584387812323995`.

## Status

Completed. No full reconstruction or training job was launched.
