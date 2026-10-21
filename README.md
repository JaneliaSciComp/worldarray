# worldarray

Chunked arrays in world coordinates, through tensorstore.

Open a zarr v2, zarr v3, N5 or Neuroglancer precomputed array or pyramid by path (local,
`s3://`, `gs://`, `http(s)://`), know where it sits (voxel size, the centre of voxel 0,
units, axes), pick a level, and read a voxel box or a **world box** as a numpy array, padded
where the box leaves the data and optionally resampled to another voxel size. Any number
of axes, any units: what the store says is what you get, converted only where you ask.

Dependencies: `tensorstore` and `numpy`. No zarr-python, no dask, no funlib. Read-only in
v1. The [specification](https://janeliascicomp.github.io/worldarray/spec/) (`docs/spec.md`
in the repo) is the contract this package is built to; this README is its first sections.

```bash
pip install worldarray
```

## Quick start

```python
import worldarray as wa

vol = wa.open("s3://janelia-cosem-datasets/jrc_hela-2/jrc_hela-2.zarr/recon-1/em/fibsem-uint8", level="s1")
vol.voxel_size      # (8.0, 8.0, 8.0)  nm, spatial axes, the array's order (z, y, x)
vol.translation     # the centre of voxel 0, in nm
vol.corner          # translation - voxel_size / 2: what funlib, daisy and Neuroglancer call the offset
vol.shape, vol.dtype, vol.axes, vol.levels

box = ((1000.0, 2000.0, 3000.0), (1512.0, 2512.0, 3512.0))   # [begin, end) in nm, half-open
arr = vol.read(box)                                           # numpy, zyx; a voxel is read iff its centre lies in the box
arr = vol.read(box, padding="edge")                           # a box past the data comes back padded, never raises
arr = vol.read(box, voxel_size=(16, 16, 16), method="linear")  # resampled (block mean here)
arr, inside = vol.read(box, mask=True)                        # True where the data was

# index space, native units: what an engine or a viewer wants
raw = wa.open(path, units=None)
raw.meta            # ArrayMeta: shape, dtype, chunk_shape, axes, units, voxel_size, translation, attrs, ...
raw.ts              # the tensorstore itself, C order, origin 0
raw.read_index((0, 0, 0), (64, 64, 64), padding=0)

pyr = wa.open_pyramid(path)             # every level, opened on first use
pyr.names                               # ['s0', 's1', ...]
pyr.select((16, 16, 16), mode="floor")  # the level to read at 16 nm
```

## API

```python
wa.open(path, *, level=None, voxel_size=None, level_mode="floor", on_mismatch="error",
        padding=0, method="nearest", select=None, units="nanometer", context=None, retries=None) -> WorldArray
wa.open_pyramid(path, *, context=None, units="nanometer", select=None) -> Pyramid
wa.metadata(path, *, level=None, units=None) -> ArrayMeta
wa.open_tensorstore(path, *, level=None, context=None) -> tensorstore.TensorStore
wa.select_level(levels, voxel_size, mode) ; wa.shared_context(cache_bytes=0) ; wa.capped_context(cache_bytes=0, concurrency=1)
wa.Retry(max_attempts=10, timeout=None, base_delay=1.0)
```

| `wa.open(...)` argument | Meaning |
|---|---|
| `path` | an array, a level under a multiscale group, or a group (then `level` or `voxel_size` picks the level; a group with neither raises, listing its levels) |
| `level` | a level name (`"s1"`); ignored with a warning for an array path |
| `voxel_size` | the size to read at, in `units`: picks the level by `level_mode`; see `on_mismatch` when no level has it |
| `level_mode` | `floor` (the coarsest level no coarser than the target on any axis, else the first), `exact`, `nearest`, `resample` |
| `on_mismatch` | when no level has `voxel_size`: `error` (list the levels), `resample` (read the chosen level and resample it), `relabel` (read it voxel for voxel and call its voxels `voxel_size`; no resampling) |
| `padding` | what fills the part of a box outside the data: any value the dtype holds, or `"edge"` for the nearest voxel |
| `method` | the resampling kernel: `nearest` (ties to the lower index; invents no values) or `linear` (block mean for whole factors, else interpolation). No dtype heuristic: the package never decides an array holds labels |
| `select` | `{"c": 0, "t": 0}`: pin non-spatial axes to one index; else they are kept whole |
| `units` | what world coordinates are in: a unit for the spatial axes (`"nanometer"`), a dict per axis kind (`{"space": "micrometer", "time": "second"}`), or `None` for the store's own units |
| `context` | a `tensorstore.Context`; default `shared_context(0)`, one per process, cache off |
| `retries` | a `Retry`; also settable later on the array, and per read |

`WorldArray` has `ts`, `meta` (native units), `grid` (requested units, all axes), `shape`,
`dtype`, `axes`, `spatial`, `voxel_size`, `translation`, `corner`, `levels`, `level_name`,
`read_index(begin, end, padding=None, mask=False, retries=None)`,
`read(box, voxel_size=None, padding=None, method=None, mask=False, retries=None, strict=False)`,
`chunks_covering(begin, end)`, `reopen(context=None, **overrides)` and, for the migration,
`to_ndarray_ts(roi)`. A **box** is a `(begin, end)` pair of float tuples over the spatial
axes (or over every axis), anything with `.begin`/`.end` or `.offset`/`.shape` (a funlib
`Roi`, duck-typed), or a `wa.Box`. There is no `offset` attribute: that word means the
corner in funlib and the centre in OME, so the two things have two names, `corner` and
`translation`.

`ArrayMeta` (frozen; one entry per axis in the array's C order; the same field names as
chunkmirage's `ArrayInfo`): `shape`, `dtype`, `chunk_shape`, `axes`, `axes_named`, `units`
(`""` when none), `voxel_size`, `translation` (the centre of voxel 0), `format`, `path`,
`parent`, `attrs`, `group_attrs`; `spatial`, `corner`, `in_units(space=, time=)`, `in_nm()`.

## Conventions

These are the rules the package exists to hold in one place.

1. **A translation is where voxel 0's centre is**, as in OME-NGFF, COSEM's `transform`, CF
   coordinates and Neuroglancer's own zarr reader (its `ome.ts`: "In OME's coordinate
   space, the origin of a voxel is its center, while in Neuroglancer it is the lower
   corner"). Index `i` sits at `translation + i * voxel_size`; voxel `i` covers
   `[translation + (i - 1/2) * voxel_size, translation + (i + 1/2) * voxel_size)`. Formats
   that store the **corner** are moved half a voxel along the spatial axes on reading:

   | Stored | Means | Read as `translation` |
   |---|---|---|
   | OME-NGFF `translation` (zarr v2/v3) | centre of voxel 0 | as is |
   | COSEM `transform.translate` (N5 or zarr) | centre | as is |
   | funlib `offset` with `resolution`/`voxel_size` (`offset` missing = 0) | corner | `offset + voxel_size / 2` |
   | N5 `pixelResolution` x `downsamplingFactors f`, no offset | BigDataViewer | `(f - 1) / 2 * pixelResolution` |
   | precomputed `voxel_offset` x `resolution` | corner | `+ resolution / 2` |

   The shift applies only along axes named `z`, `y` or `x`, or the last three when axes are
   unnamed; never along channel or time.

2. **Axis order is the store's C order; N5 and precomputed are transposed to it** on read,
   so `(z, y, x)` for a volume, `(c, z, y, x)` with channels, `(t, c, z, y, x)` for OME 0.5.
   A precomputed single channel is dropped; a non-zero origin is moved to 0 in `.ts` and
   carried in `translation`. Unnamed axes are called `("t", "c", "z", "y", "x")[-ndim:]`
   and `axes_named` is False.

3. **Units are kept as stored and converted only where asked.** Any unit string on any axis
   is accepted. `units="nanometer"` converts the spatial axes through one table (American
   and British spellings, `nm`, `um`, `µm`, `μm`, `mm`, `m`, `Å`, `pm`; and for time `s`,
   `ms`, `us`, `ns`, `minute`, `hour`, `day`). A spatial axis with no unit is taken as
   nanometres; an unknown unit on an axis being converted is a `ValueError`, never silently
   the target unit. Voxel sizes and translations are floats and are never rounded.

4. **Metadata precedence per format, first match wins.** zarr: the parent group's OME-NGFF
   `multiscales[0].datasets` entry whose `path` is this array, matched by name never by
   position, composed with the multiscale-level `coordinateTransformations` (0.4, 0.5; an
   0.6 `coordinateSystems` group is read for its intrinsic system); else the array's own
   funlib (`resolution`/`voxel_size`, `offset`, `units`, `axis_names`) or COSEM
   (`transform`) attributes; axes from OME `axes`, zarr v3 `dimension_names`, xarray
   `_ARRAY_DIMENSIONS` or `axis_names`. N5: the array's `transform`; else the group's
   `multiscales[].datasets[].transform` for this path; else `pixelResolution` (dict or bare
   list) or `resolution`, array else group, times `downsamplingFactors`, plus `offset`
   (x-first lists). precomputed: the scale's `resolution` and `voxel_offset`. An OME
   `translation` beside an `offset` equal to it, or half a voxel below it, passes silently;
   any other pair warns. An OME `scale` disagreeing with a per-array `voxel_size` raises.
   Everything unconsumed stays in `attrs` and `group_attrs`.

5. **Stores are read through tensorstore's kvstore only.** Format detection probes
   `zarr.json`, `.zarray`, `.zgroup`, `info`, `attributes.json`, then the parent's `info`
   with the last path segment as a precomputed scale key; the `.n5` or `.zarr` suffix is
   never consulted. `s3://` is tried anonymously then with default credentials, `gs://`
   with default credentials then through the bucket's public https URL; the way that
   worked is remembered per bucket. Metadata reads give up after about five seconds. A
   zarr v2 compressor carrying members tensorstore rejects (`checksum`) is opened without
   them.

## Behaviour, in brief

- `read_index` clips, reads, pads; a box entirely outside returns the fill without touching
  storage; results are C-contiguous in native byte order.
- `read` selects voxels whose **centres** lie in the half-open box: a box of extent
  `m * voxel_size` returns exactly `m` voxels whatever its alignment, and boxes aligned to
  voxel corners or to voxel centres map to whole voxels with nothing rounded.
- Resampling reads the source voxels the target centres need (plus the kernel radius),
  never the box edges, so adjoining reads agree with one read across both.
- Data reads retry on timeouts and transient store errors with mesh-n-bone's backoff; the
  timed-out future is cancelled first.

The full statements, each pinned by a test, are in the
[specification](https://janeliascicomp.github.io/worldarray/spec/).

## Development

```bash
uv sync --group dev --group docs
uv run pytest -q
uv run ruff check src tests && uv run ruff format src tests
uv run mkdocs serve
```

BSD-3-Clause, Howard Hughes Medical Institute.
