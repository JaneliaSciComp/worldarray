# Changelog

## 0.1.1 (2026-10-22)

- funlib `voxel_size`/`offset` on the parent group place an array that carries none, for
  zarr and N5 alike; funlib's keys on an N5 array are read in the array's C order, as
  zarr-python's N5 store writes them (both from mesh-n-bone's review of its PR #48).
- A precomputed path that names a scale is that scale under `level=0` or `"first"` too;
  other levels warn and are ignored, as on an array path.
- The GCE metadata guard is applied only off Google Cloud (decided from the DMI product
  name and the `GCE_METADATA_*` variables), so metadata-server credentials keep working on
  a Google VM.
- `Pyramid.placements()`, `voxel_sizes()` and `translations()` read the group document when
  the pyramid was opened through one of its level paths too; they used to open every level
  from there, and a missing level made them raise.

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
