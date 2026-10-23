# worldarray

Chunked arrays in world coordinates, through tensorstore.

```bash
pip install worldarray
```

## Why

Volumetric data lives in zarr v2, zarr v3, N5 and Neuroglancer precomputed stores, on local disks, S3, GCS and HTTP. Each format records *where the array sits*,
its voxel size, origin, units and axes, in its own dialect, and the dialects disagree on
the one thing that matters most: whether a stored origin means the centre of the first
voxel or its corner. Get that wrong and every object is half a voxel out of place, which
is invisible in a viewer and ruinous in a measurement.

Reading such a volume correctly means parsing all those dialects, knowing the half-voxel
rule for each, picking a pyramid level for a wanted resolution, and reading a region
given in physical units as a plain numpy block, padded where the region leaves the data,
resampled when no level matches, fast enough for a model server or a thousand cluster
workers, with retries for flaky object stores and thread caps for shared hosts. No one
package did all of that, so every tool wrote its own reader, and the readers drifted apart. worldarray is that reader, written once, with the rules stated and tested.

Its dependencies are `tensorstore` and `numpy`. Since 0.2 it also writes the one thing
every tool wrote for itself and got wrong in its own way: the OME-NGFF metadata of a placed
array or pyramid level, and the array itself, created once and filled by aligned writes
from any number of jobs. It never deletes, overwrites or downsamples. The
[specification](https://janeliascicomp.github.io/worldarray/spec/) is the contract it is
built to; every rule below is a test.

## Quick start

```python
import worldarray as wa

vol = wa.open("s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8", level="s1")
vol.voxel_size      # (8.0, 8.0, 8.0) nm, spatial axes
vol.translation     # the centre of voxel 0, in nm
vol.corner          # the lower corner of voxel 0
vol.shape, vol.dtype, vol.axes, vol.levels

box = ((1000.0, 2000.0, 3000.0), (1512.0, 2512.0, 3512.0))   # [begin, end) in nm, half-open
arr = vol.read(box)                            # numpy, z y x; a voxel is read iff its centre lies in the box
arr = vol.read(box, padding="edge")            # a box past the data comes back padded, never raises
arr = vol.read(box, voxel_size=(16, 16, 16))   # resampled on the fly
arr, inside = vol.read(box, mask=True)         # True where the data was

# a volume at a resolution the store does not have: the best level, resampled
six = wa.open(path, voxel_size=(6, 6, 6), on_mismatch="resample")

# one volume read onto another volume's grid, for masks, contact sites, statistics
onto = raw.read(box, voxel_size=labels.voxel_size, anchor=labels.corner)
onto = raw.read_onto(grid, begin, end)   # voxels [begin, end) of a Grid, exactly that shape

# index space, native units: what an engine or a viewer wants
vol = wa.open(path, units=None)
vol.meta            # shape, dtype, chunk_shape, axes, units, voxel_size, translation, attrs, ...
vol.ts              # the tensorstore itself, C order, origin 0
vol.read_index((0, 0, 0), (64, 64, 64), padding=0)

pyr = wa.open_pyramid(path)             # every level, opened on first use
pyr.select((16, 16, 16), mode="floor")  # the level to read at 16 nm

# writing: a placed level, its OME-NGFF group document written for it (zarr v3 + OME 0.5 here)
out = wa.create("out.zarr/s0", shape=(512, 512, 512), dtype="uint64",
                voxel_size=(8, 8, 8), translation=(4, 4, 4), chunks=(64, 64, 64))
out.write_index((0, 0, 0), (64, 64, 64), block)       # whole chunks only: many jobs, no locks
out.write(box, block)                                  # or by world box
s1 = wa.open_pyramid("out.zarr", mode="r+").add_level("s1", factors=2)   # placed and sized; you fill it
```

## What it does

- **Opens** zarr v2, zarr v3 (sharded or not), N5 and precomputed (sharded or not) arrays
  and pyramids by path, local or `s3://`, `gs://`, `http(s)://`, through tensorstore only.
  Public buckets are read anonymously first, credentials only when refused; the way in is
  remembered. `AWS_ENDPOINT_URL` points at S3-compatible stores.
- **Places** every array by one written rule per format (below), reads axes and units as
  stored, converts only where asked (nanometres by default on spatial axes), and keeps
  voxel sizes as floats, never rounded.
- **Chooses a level** by voxel size from the pyramid's own metadata, without opening the
  levels, with four policies for a size no level has: error, resample, relabel (voxel `i`
  stays voxel `i`, the grid rescaled), or keep the level.
- **Reads** voxel boxes or world boxes as C-contiguous numpy in native byte order, padded
  with any value or the edge voxel, with an optional mask of what was data, a per-read
  `transform` applied before padding (or `post_transform` after resampling), and resampling by nearest neighbour or linear
  interpolation anchored where you say. Adjoining block reads agree with one read across
  both.
- **Behaves on a cluster**: one shared tensorstore context per process, its threads sized
  from `OMP_NUM_THREADS` or the LSF job's slots instead of the node's cores, or a capped one
  per worker; reads and writes retried with backoff and size-aware timeouts on transient
  store errors, a cancelled future on timeout, and every context's thread pools nudged once
  a second against a tensorstore bug that can leave a queued task with no thread, so a
  stuck operation recovers or fails loudly instead of hanging the job.
- **Carries the geometry**: `Box` with `intersect`, `union`, `contains`, `grow`, `shift`,
  `snap` to a grid, and the voxel indices a world box selects.
- **Writes placed arrays** (0.2): the OME-NGFF 0.4 or 0.5 document from voxel size, centre
    translation, axes and units, with each level's placement explicit or derived from
  factors and an origin; `create` following the container's format and writing the group
  documents an output needs down from its `.zarr` root, with explicit chunks and codecs,
  and only two answers to "it exists": raise, or adopt an identical array after checking;
  whole-unit writes from many processes with no coordination; `add_level` as metadata plus
  an empty array; chunk keys for every format. Nothing is ever deleted or overwritten.
- **Plans outputs and blocks** (0.3): a store-less `Grid` with the same geometry as an
  array, so an output is planned, cut into blocks and only then created; `blocks` tiling an
  array or a box into write boxes with context-grown read boxes, in C order, as plain
  picklable records a cluster job takes by id; the anisotropy-aware pyramid schedule; every
  input read onto that grid by its voxel indices (0.4). Nothing is scheduled or run.

## The placement rule

A translation is where the centre of voxel 0 is. Index `i` sits at
`translation + i * voxel_size` and covers half a voxel either side, so the array's corner
is `corner = translation - voxel_size / 2`. With 8 nm voxels and the corner at 0, the
translation is 4: voxel 0 covers 0 to 8 nm around its centre at 4, voxel 1 covers 8 to
16 nm around 12. That is how OME-NGFF defines it and how Neuroglancer draws it.

The centre is the model because a voxel is a sample taken at a point. Everything that
compares or resamples grids (picking a level, reading one volume onto another's grid,
interpolating) works on those points, and the box a voxel covers follows from its point.
The read rule follows too: a voxel is read iff its centre lies in the box, so boxes that
adjoin never share a voxel or drop one.

Dialects that store the corner are moved half a voxel on reading:

| Stored | Means | Read as `translation` |
|---|---|---|
| OME-NGFF `translation` | centre | as is |
| COSEM `transform.translate` | centre | as is |
| `offset` beside `resolution` / `voxel_size` (`offset` missing: 0) | corner | `offset + voxel_size / 2` |
| N5 `pixelResolution` or `resolution` with `downsamplingFactors f`, no offset | corner at the origin at every level, as Neuroglancer draws N5 | `f / 2 * base` |
| precomputed `voxel_offset` | corner | `(voxel_offset + 1/2) * resolution` |
| nothing | a plain array | voxels of 1, corner at the origin |

Axes are the store's C order; N5 and precomputed are transposed to it. Unlisted or
conflicting attributes warn; a voxel size that disagrees with the OME scale raises, since
that is how stale metadata is caught. Metadata the parser did not consume stays available
raw. The full precedence per format, the half-voxel rules, and every read behaviour are
in the [specification](https://janeliascicomp.github.io/worldarray/spec/).

## API in brief

```python
wa.open(path, *, level=None, voxel_size=None, level_mode="floor", on_mismatch="error",
        padding=0, method="nearest", select=None, units="nanometer", context=None, retries=None,
        metadata_conflicts="error", base_voxel_size=None, base_units=None, anchor=None,
        transform=None, strict_units=True) -> WorldArray
wa.open_pyramid(path, ...) -> Pyramid          # .names .levels .by_name() .select() .placements() .metas(): no opens needed
wa.metadata(path, level=None, units=None, open=True, retries=None) -> ArrayMeta   # open=False: from documents alone
wa.detect(path, missing_ok=False, retries=None) -> Node      # format, array or group, raw attrs
wa.open_tensorstore(path, level=None, context=None, retries=None)
                    # retries= times out and retries document reads and the open, as it does data reads
wa.shared_context(cache_bytes=0, threads=None, fsync=False, locking="none") ; wa.Retry(...)
wa.capped_context(cache_bytes=0, concurrency=1, network=True, file_io=None, fsync=False, locking="none")
wa.thread_budget()                              # the shared context's decoding threads: OMP_NUM_THREADS, the job's slots, or CPUs;
                                                #   file I/O gets at least 8; threads=N limits both to N, threads=0 leaves tensorstore's sizes
                                                # contexts made here write without fsync or OS file locks; fsync=True, locking="os" restore them
wa.list_chunk_keys(path, separator=".")        # chunk indices from keys alone, no metadata needed
wa.Box ; wa.units.TABLE

# writing (0.2)
wa.create(path, shape, dtype, voxel_size, translation, *, axes=None, units="nanometer", chunks=..., shards=None,
          fill_value=0, codecs="zstd", format=None, attrs=None, group_attrs=None, exists="error", store_fill=False) -> WorldArray
wa.open(path, mode="r+") ; wa.open_pyramid(path, mode="r+").add_level(name, factors=, origin=0, ...)
wa.ome_multiscales(axes, units, levels, version="0.5", type=None) ; wa.level_placement(voxel_size, translation, factors, origin=0)
wa.level_shape(shape, factors, origin=0) ; wa.chunk_key(index, format, ...) ; wa.parse_chunk_key(key, format, ...)
wa.ensure_group_entry(group_path, name, voxel_size, translation, axes, units="nanometer")   # list an array you made yourself
wa.units.ome_name("nm") -> "nanometer" ; wa.write.codecs_of(path) -> (codecs, shards)

# grids and blocks (0.3)
g = wa.Grid(shape, voxel_size, translation, axes=None, units="nanometer", chunks=None, shards=None)   # or Grid.from_meta(meta, voxel_size=)
wa.create(path, grid=g, dtype="uint64")
for b in wa.blocks(shape, block_shape, context=0, box=None, clip=True):   # also vol.blocks() and g.blocks()
    b.begin, b.end, b.read_begin, b.read_end, b.inner, b.at_end, b.id    # plain records; blocks[id] is O(1)
wa.level_factors(voxel_size, ratio=1.5, max_levels=None, stop_shape=None, shape=None)
wa.units.abbreviation("nanometer") ; wa.store.join(path, "s0") ; wa.store.parent(path) ; wa.store.basename(path)
```

`WorldArray`: `ts`, `meta`, `grid`, `shape`, `dtype`, `axes`, `voxel_size`, `translation`,
`corner`, `box`, `levels`, `level_name`, `level_path`, `source_voxel_size`, `source_translation`, `resampled`,
`writable`, `write_shape`;
`read(box, voxel_size=, padding=, method=, mask=, anchor=, transform=, post_transform=, select=, strict=)`,
`read_index(begin, end, padding=, mask=, select=, transform=)`, `read_onto(grid, begin=, end=, ...)`, `index_box(box)`,
`chunks_covering(begin, end, write=)`, `list_chunks()`, `chunk_key(index)`, `parse_chunk_key(key)`,
`write_index(begin, end, data, partial=, retries=)` and `write(box, data, partial=, retries=)` (`data` an array or a
scalar), `reopen(context=)`.

A box is a `(begin, end)` pair of floats in your units, half-open per axis, over the
spatial axes or every axis; any object with `.begin`/`.end` or `.offset`/`.shape` also
works. There is no `offset` attribute, because that word means the corner in some dialects
and the centre in others: `translation` and `corner` are the two names for the two things.

## Origins

The vocabulary, an array that knows its `voxel_size`, where it sits and its axis names, and
is read by a box in world units, follows funlib.persistence's `Array`, which many of the
tools this serves were built on. The placement rules were consolidated from the readers
those tools had each written, in cellmap-flow, cellmap-analyze, mesh-n-bone and
chunkmirage, whose conventions document and conformance tests seeded this one; their
authors' sessions trialled every change here on their own data. The voxel-centre
convention is OME-NGFF's and ITK's, and is what Neuroglancer draws. tensorstore does the
work underneath.

## Development

```bash
uv sync --group dev --group docs
uv run pytest -q
uv run ruff check src tests && uv run ruff format src tests
uv run mkdocs serve
```

BSD-3-Clause, Howard Hughes Medical Institute.
