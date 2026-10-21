# Changelog

## 0.1.0 (2026-10-21)

- First version, built to `docs/spec.md`: open zarr v2, zarr v3, N5 and Neuroglancer
  precomputed arrays and pyramids through tensorstore; one metadata precedence per format
  with the voxel-centre rule for translations; index-space and world-box reads with
  padding; level selection; nearest and linear resampling; retries; shared and capped
  tensorstore contexts.
  Built and trialled in one day against the readers it replaces in cellmap-flow,
  cellmap-analyze, mesh-n-bone and chunkmirage, and checked on their own data by the
  CineMap, liconn-err, registration-station and salamander repos; see `docs/spec.md`,
  "Trial status". Hot-path reads cost 0.1 to 0.2 ms over a raw tensorstore read.
