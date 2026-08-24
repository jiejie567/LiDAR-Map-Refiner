# Oriented-surface graph and bundle-adjustment contract

This document is the implementation contract for the post-GhostLoop research
branch.  The physical object carried between modules is an **oriented surface
observation**, not an unoriented point or a freshly re-detected map layer.

## Scope

The method has three coupled modules:

1. observation-oriented registration rejects correspondences between opposite
   faces of thin structure;
2. the accepted correspondences produce both an SE(3) measurement and a full,
   anisotropic information matrix for sequential and loop factors;
3. observation-oriented BALM keeps the same two faces in separate plane units.

The current ICRA branch restores item 3 only. Items 1--2 remain archived for
the journal branch and are not implied by the `Double-sided BALM` switch.

Ghost/map-inconsistency detection is an application and diagnostic.  It is not
used as proof that a proposed loop is correct.

## Frame and tangent conventions

- A factor between nodes `target` and `source` stores
  `T_target_source = inverse(T_world_target) @ T_world_source`.
- A local point is transformed as `p_target = R_target_source p_source + t`.
- G2O `EDGE_SE3:QUAT` information is serialized in translation--rotation
  tangent order `[tx, ty, tz, rx, ry, rz]`.
- GTSAM `Pose3` uses rotation--translation tangent order
  `[rx, ry, rz, tx, ty, tz]`.
- Every boundary between the two formats performs an explicit permutation;
  anisotropic matrices are never passed through unchanged.

## Surface identity

For a point `p` observed from sensor center `s`, the normal is oriented so that
`dot(n, s - p) >= 0`.  A correspondence is admissible only if the transformed
source normal and target normal agree within the configured angular gate.
Consequently, opposite wall faces cannot generate one registration residual or
one BALM plane unit, while repeated observations of the same face can.

The observing side is derived from the original keyframe and is immutable for
an experiment.  It must not be recomputed from an optimized map pose and used
to silently change point identity.

## Factor information

For accepted point-to-plane residuals

`r_k = n_k^T (T_target_source p_k - q_k)`,

the factor builder records the robust normal matrix and a residual-scale
estimate.  Information is anisotropic: weak eigen-directions remain weak.
Correspondence count is capped when scaling information so denser sampling of
the same surface does not create arbitrary confidence.

The matrix describes local uncertainty around the selected registration mode;
it is **not** a place-recognition probability.  Loop factors therefore retain
independent proposal, trial-PGO, and post-solve outlier auditing.

## Graph policy

- Sequential factors are never augmented, because the front end and ICP reuse
  the same observations. The preregistered ablation compares full measurement
  replacement against preserving the scan-to-map front-end measurement while
  rebuilding only its information matrix. Loop factors rebuild both quantities.
- If registration is invalid or underconstrained, retain the original relative
  measurement with an explicitly weakened information matrix.
- Sequential and loop factors use the same measurement builder and matrix
  convention.  Only their proposal and acceptance policies differ.
- BALM is evaluated after PGO as a distinct refinement stage and never changes
  which loop factors were accepted.

## Required ablation

The frozen comparison is: original graph; fully remeasured ICP graph;
front-end measurements with rebuilt anisotropic weights; observation-oriented
weights and remeasured loops; and the complete method with oriented BALM. It
reports ATE/RPE, factor calibration, degeneracy detection, wrong-loop
acceptance, thin-wall thickness, map surface error, and runtime.
