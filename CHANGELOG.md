# Changelog

## 0.2.0 (2026-10-22)

Metadata and placed creation, built to spec draft 3 (Behaviours 10 to 16). Nothing on the
read side changes.

- `ome_multiscales(axes, units, levels, version)` builds the complete OME-NGFF 0.4 or 0.5
  group document with one explicit placement per level; `level_placement` and
  `level_shape` derive a level from factors and an origin (Convention 1's rule inverted,
    so a level built from a ROI origin is placed from it; `level_shape` absorbs float-noise
  factors such as `0.3 / 0.1`). Units are written in OME's names (`"nm"` becomes
  `"nanometer"`; `units.ome_name`), an unknown unit as given; the multiscale's `type` is
  written when given. Expected documents are pinned in `tests/fixtures/write.json`, shared
  with chunkmirage's TypeScript builder, each case with its resolved per-axis units and
  types, fifteen cases.
- `create(path, shape, dtype, voxel_size, translation, chunks=..., ...)` writes the group
  document (made when absent, else its one dataset entry added or replaced, every other
  attribute kept) and then the array, as zarr v2 with OME 0.4 or zarr v3 with OME 0.5
  following the group's format. `chunks` is required, codecs are explicit with a zstd
  level 3 default, shards are a zarr v3 option. `exists` is `"error"` or `"adopt"` (an
  identical array is opened for writing, a different one raises naming every difference,
  a creation lost to another process is adopted after a short backoff). Nothing is ever
    deleted, truncated or overwritten. Write units equal to the fill value are not stored
  unless `store_fill=True`, and a stored unit written so loses its key, as zarr-python's
  `write_empty_chunks=False` does. `create` also writes a bare group document for every
  missing ancestor down from the `.zarr` root and follows the nearest existing ancestor's
  zarr version, so a level made in a zarr v2 container is zarr v2 with OME 0.4 and the
  hierarchy browses (cellmap-analyze, cellmap-flow). Adopting an identical array writes no
  group document; after a write the document is read again and a lost entry is written
  again, then raises (registration-station's concurrent block jobs).
- `ensure_group_entry` is public: list an array the package did not create, such as a
  symlinked `s0` (mesh-n-bone), or a level filled before it is listed.
- `open(path, mode="r+")` opens an existing array for writing; a `WorldArray` pickles by
  its arguments and its context's spec and reopens in the receiving process, a capped
  context staying capped (mesh-n-bone's workers). `write.codecs_of` takes a path.
- `write_index(begin, end, data)` writes whole write units only (the shard when sharded,
  else the chunk; the clipped last unit allowed), so writers in different processes need
  no coordination; `partial=True` does tensorstore's read-modify-write. `write(box, data)`
  maps a world box by the centre rule first. No cast, no padding, no resampling on write.
- `open_pyramid(path, mode="r+").add_level(name, factors=, origin=)` creates a level's
  entry and empty array, placed and sized from the level below with its dtype, chunks,
  shards, codecs and fill value copied; the downsampling itself stays with the caller.
- `chunk_key` and `parse_chunk_key` for zarr v2, zarr v3 (both encodings, sharded arrays
  keyed by shard), N5 and precomputed, as free functions and on `WorldArray`;
  `list_chunks` uses the same parser. `write_shape`, `ArrayMeta.shard_shape`,
  `chunks_covering(write=True)`.
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
- **Placement change.** An N5 level with `downsamplingFactors` and no offset now sits with
  its corner at the origin at every level, as Neuroglancer draws N5: translation
  `f / 2 * base`. 0.1.0 followed BigDataViewer for `pixelResolution` pyramids, centring the
  base's voxel 0 on the origin (`(f - 1) / 2 * base`, half a base voxel lower), while
  placing `resolution` pyramids with the corner at the origin, so the two disagreed. Two
  shared fixture cases move and are renamed: `n5_group_pixelresolution_and_downsampling_factors_corner_at_origin`
  (translation 4, 4, 4) and `n5_bigstitcher_resolution_with_downsampling_factors_corner_at_origin`
  (0.5, 0.157, 0.157).

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
