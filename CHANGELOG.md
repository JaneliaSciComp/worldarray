# Changelog

## Unreleased

- First version, built to `docs/spec.md`: open zarr v2, zarr v3, N5 and Neuroglancer
  precomputed arrays and pyramids through tensorstore; one metadata precedence per format
  with the voxel-centre rule for translations; index-space and world-box reads with
  padding; level selection; nearest and linear resampling; retries; shared and capped
  tensorstore contexts.
