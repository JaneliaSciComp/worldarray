# Changelog

## 0.5.0 (unreleased)

Two bugs that hung or crashed cellmap-analyze 0.6 on /nrs (salamander's and c-elegans'
runs, 2026-10-22), and tensorstore's threads sized to the job. No placement change and no
unit table change.

- A read whose first attempt timed out raised `IndexError` instead of retrying: the
  warning took the first line of the error's message, and tensorstore's `TimeoutError()`
  has none. The warning now names the error's type when the message is empty, and the
  error after the last attempt names the operation, the path and the attempts.
- Every context the package makes or opens a store on is watched: once a second a daemon
  thread gives its decoding and file I/O pools one no-op task each (an in-memory copy and
  a local read of a missing key; no store is touched). tensorstore up to at least 0.1.85
  can strand a queued task: a pool worker idle for 20 ms exits as the task arrives, and
  the task never runs (fixed upstream in google/tensorstore ed51718085, in no release
  yet). With `data_copy` and `file_io` capped at 1, as cellmap-analyze's workers were,
  dozens of workers per run waited forever in `write_index` with no tensorstore thread
  alive, and reads timed out into the `IndexError` above. One more task on the same pool
  wakes it, which salamander's and c-elegans' sessions proved on live stuck workers, so a
  stranded task now runs within about a second, at any concurrency, whoever waits on it:
  worldarray, `vol.ts` directly or `await`. Reproduced on purpose on tensorstore 0.1.75,
  0.1.78 and 0.1.85 (opt-in `tests/test_lost_wakeup.py`): cellmap-analyze's block loop
  gave 29 `IndexError`s and 55 writes stuck past 20 s on 0.4.0, none on 0.5.0, slowest
  0.98 s. Spec open question 14 says when this goes.
- `write_index`, `write` and `write_chunk` are timed out and retried as reads are
  (Behaviour 6), by a new `retries=` argument or the handle's `retries`. A write used to
  wait on its future with no timeout at all.
- Without `context`, the shared context's decoding and file I/O pools are limited to the
  new `wa.thread_budget()`: `OMP_NUM_THREADS` when set (dask sets 1 in each worker), else
  the job's slots (`LSB_DJOB_NUMPROC`, `SLURM_CPUS_PER_TASK`, `NSLOTS`), else the CPUs the
  process may run on. A server that wants more threads than its job's slots asks for more
  slots or passes its own context.
  tensorstore alone sizes both to every CPU on the node, whatever `bsub -n` asked for, so
  a dask worker's tensorstore now takes the same one thread as its numpy. Writes stay faster
  than cellmap-analyze's zarr-python writer on NFS, at one scale on 128^3 uint64 label
  chunks: 2.3 to 5 times at one thread per pool (7 threads per process against
  zarr-python's 24 to 31), 2 to 7 times with the pools at the machine's CPUs.

## 0.4.0 (2026-10-22)

cellmap-analyze's three asks, so it can call worldarray directly with no wrapper, and two
of cellmap-flow's. No placement change and no unit table change.

- `vol.read_onto(grid, begin=None, end=None, ...)` reads voxels `[begin, end)` of a
  `Grid`'s spatial axes from the array, exactly that shape (Behaviour 20). It is the
  `read` of the grid's index box at the grid's voxel size anchored at its corner, so at
  the array's own voxel size nothing is resampled, whatever the sub-voxel offset, and
  otherwise the array is resampled onto the grid. The grid's units are converted.
- A pickled handle, and `reopen`, use the path resolved against the working directory at
  open, so a worker that changes directory before unpickling opens the same store; it
  used to raise `FileNotFoundError` for a relative path.
- `write_index`, `write` and `write_chunk` take a scalar, which fills the box. It is
  converted to the dtype only when its value survives (`0` into `uint8`; `300`, `-1`, `0.5`
  and NaN raise), and the fill value over whole units leaves them absent.
- `vol.source_translation`: the stored level's translation in `units`, beside
  `source_voxel_size`, under resample and relabel too (cellmap-flow's `stored_placement`).
- `create(exists="adopt")` also compares the keys of `attrs` with the existing array's
  attributes and raises naming each that differs or is missing; keys not given are not
  compared. An adopt given different `attrs` used to keep the old ones silently, as funlib's
  `prepare_ds` kept an old offset (cellmap-flow's `_refuse_another_region`).

## 0.3.1 (2026-10-22)

- `source_voxel_size` reports the stored level's voxel size under `on_mismatch="relabel"`,
  as the spec says; it reported the relabelled size (cellmap-flow's report). Reads,
  `voxel_size`, `translation` and `corner` are unchanged.
- `box` of an array opened with a resample target is the extent of the target voxels
  `shape` counts, as the spec's `[corner, corner + shape * voxel_size)` says and as
  `Grid.from_meta(meta, voxel_size=t).box` already was; it was the stored data's extent, so
  `read(box)` could return fewer voxels than `shape` (cellmap-flow's 7 voxels of 8 nm at
  24 nm: box ended at 56 nm, now 72). No voxel moves.
- **Unit table change.** The unit table knows kilometres (`kilometer`, `kilometre`, `km`,
  1e3 m), so they convert, abbreviate to `km` and are written as OME's `kilometer`
  (chunkmirage's ask). This should have been a minor release (spec section 8): cellmap-flow
  exports the table and its CI failed until it moved its pin to `>=0.3.1`.
- README: the placement rule gains the corner formula, a worked example, why the centre
  is the model, and how the read rule follows from it.

## 0.3.0 (2026-10-22)

Grids and blocks, built to spec draft 3 (Behaviours 17 to 19). Nothing on the read or
write side changes.

- `blocks(shape, block_shape, context=0, box=None, clip=True)`: the array (or a voxel
  box, which may reach past the array) cut into write boxes that tile it with no overlap
  and no gap, each with a read box grown by `context` (one int, one per axis, or
  `(before, after)` pairs) and clipped to the array when `clip`; C order, `len`,
  `blocks[id]` in O(1), `index_of(begin)`; a `Block` carries `index`, `id`, `begin`,
  `end`, `read_begin`, `read_end`, `at_start` and `at_end` per axis and `inner`, the write
  box's slices inside the read box. Plain picklable records; nothing is scheduled.
  `vol.blocks()` defaults the block to `write_shape`, so default blocks are aligned writes.
  `vol.write_chunk(index, data)` writes one whole unit.
- `Grid(shape, voxel_size, translation, axes, units, chunks, shards)`: a `WorldArray`
  without a store, with `box`, `corner`, `index_box`, `chunks_covering`, `write_shape`,
  `blocks` and `meta(dtype)`; `Grid.from_meta(meta, voxel_size=)` is an existing array's
  grid or the resample target grid over it. `create(path, grid=g, dtype=)` creates from it.
- `level_factors(voxel_size, ratio=1.5, ...)`: the anisotropy-aware pyramid schedule
  mesh-n-bone and cellmap-analyze share, per level.
- `units.abbreviation`, and `store.join`, `store.parent`, `store.basename` as public
  URL-aware path helpers.

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
