# Current Status

- Completed integration: `EXP-001` ports `rv_topview_auto_v1` into mask preprocessing.
- Production `mirror=auto` now scores normal/mirrored RV centroid trajectories in
  the canonical normalized top view and requires a normalized margin of `0.05`.
- Validation status: targeted tests pass; four real masks match standalone
  decisions and scores exactly; real transform write/read and fallback paths pass.
- The full suite has one unrelated baseline failure for a disabled case with a
  missing input root (`38 passed, 1 failed`).
