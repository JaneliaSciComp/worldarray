# worldarray

Chunked arrays in world coordinates, through tensorstore.

```bash
pip install worldarray
```

## Why

Volumetric data at Janelia lives in zarr v2, zarr v3, N5 and Neuroglancer precomputed
stores, on local disks, S3, GCS and HTTP. Each format records *where the array sits*,
its voxel size, origin, units and axes, in its own dialect, and the dialects disagree on
the one thing that matters most: whether a stored origin means the centre of the first
voxel or its corner. Get that wrong and every object is half a voxel out of place, which
is invisible in a viewer and ruinous in a measurement.

Reading such a volume correctly means parsing all those dialects, knowing the half-voxel
rule for each, picking a pyramid level for a wanted resolution, and reading a region
given in physical units as a plain numpy block, padded where the region leaves the data,
resampled when no level matches, fast enough for a model server or a thousand cluster
workers, with retries for flaky object stores and thread caps for shared hosts. No one
package did all of that, so every tool here wrote its own reader, and the readers drifted
apart. worldarray is that reader, written once, with the rules stated and tested.

Its dependencies are `tensorstore` and `numpy`. It is read-only in this version. The
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

# index space, native units: what an engine or a viewer wants
vol = wa.open(path, units=None)
vol.meta            # shape, dtype, chunk_shape, axes, units, voxel_size, translation, attrs, ...
vol.ts              # the tensorstore itself, C order, origin 0
vol.read_index((0, 0, 0), (64, 64, 64), padding=0)

pyr = wa.open_pyramid(path)             # every level, opened on first use
pyr.select((16, 16, 16), mode="floor")  # the level to read at 16 nm
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
  `transform` applied before padding, and resampling by nearest neighbour or linear
  interpolation anchored where you say. Adjoining block reads agree with one read across
  both.
- **Behaves on a cluster**: one shared tensorstore context per process or a capped one per
  worker, retries with backoff and size-aware timeouts on transient store errors, and a
  cancelled future on timeout.
- **Carries the geometry**: `Box` with `intersect`, `union`, `contains`, `grow`, `shift`,
  `snap` to a grid, and the voxel indices a world box selects.

## The placement rule

A translation is where the centre of voxel 0 is. Index `i` sits at
`translation + i * voxel_size`; voxel `i` covers half a voxel either side. That is how
OME-NGFF defines it and how Neuroglancer draws it. Dialects that store the corner are
moved half a voxel on reading:

| Stored | Means | Read as `translation` |
|---|---|---|
| OME-NGFF `translation` | centre | as is |
| COSEM `transform.translate` | centre | as is |
| `offset` beside `resolution` / `voxel_size` (`offset` missing: 0) | corner | `offset + voxel_size / 2` |
| N5 `pixelResolution` or `resolution` with `downsamplingFactors f`, no offset | BigDataViewer | `(f - 1) / 2 * base` |
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
wa.open_pyramid(path, ...) -> Pyramid          # .names .levels .by_name() .select() .placements() from the group document
wa.metadata(path, level=None, units=None) -> ArrayMeta
wa.detect(path, missing_ok=False) -> Node      # format, array or group, raw attrs
wa.open_tensorstore(path, level=None, context=None)
wa.shared_context(cache_bytes=0) ; wa.capped_context(cache_bytes=0, concurrency=1, network=True) ; wa.Retry(...)
wa.list_chunk_keys(path, separator=".")        # chunk indices from keys alone, no metadata needed
wa.Box ; wa.units.TABLE
```

`WorldArray`: `ts`, `meta`, `grid`, `shape`, `dtype`, `axes`, `voxel_size`, `translation`,
`corner`, `box`, `levels`, `level_name`, `level_path`, `source_voxel_size`, `resampled`;
`read(box, voxel_size=, padding=, method=, mask=, anchor=, transform=, strict=)`,
`read_index(begin, end, padding=, mask=, select=, transform=)`, `index_box(box)`,
`chunks_covering(begin, end)`, `list_chunks()`, `reopen(context=)`.

A box is a `(begin, end)` pair of floats in your units, half-open per axis, over the
spatial axes or every axis; any object with `.begin`/`.end` or `.offset`/`.shape` also
works. There is no `offset` attribute, because that word means the corner in some dialects
and the centre in others: `translation` and `corner` are the two names for the two things.

## Development

```bash
uv sync --group dev --group docs
uv run pytest -q
uv run ruff check src tests && uv run ruff format src tests
uv run mkdocs serve
```

BSD-3-Clause, Howard Hughes Medical Institute.
