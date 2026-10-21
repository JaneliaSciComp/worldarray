# Specification

worldarray is built to this document. Sections 1 to 5 are also the README; every
statement under "Behaviour" and "Tests" is something a test pins. Code that is not needed
to make a statement here true does not go in; a behaviour not stated here is not
implemented without adding the statement first.

Draft 2, 2026-10-21. Written from a reading of every existing reader in David Ackerman's
repos and the answers of the chunkmirage, cellmap-flow, cellmap-analyze, mesh-n-bone,
CineMap, liconn-err, registration-station, salamander and cellmap-analyze-scripts
sessions. Decisions still open are in section 7.

## 0. Why this exists

Seven readers across those repos open the same zarr v2, zarr v3, N5 and precomputed
volumes through tensorstore and each parses the OME-NGFF, COSEM N5 and precomputed
metadata by hand: cellmap-flow (`image_data_interface.py` plus `io/`, 2.7k lines),
cellmap-analyze (765), mesh-n-bone (346), chunkmirage (`tensorstore_source.py`, 649),
CineMap (`slice_loader.py`), liconn-err (`dataset.py`) and registration-station
(zarr-python, by hand); salamander's analysis scripts parse OME multiscales by hand three
more times. They disagree on the one rule that matters, the half voxel between a stored
corner and a stored centre: all of them place an OME translation at voxel 0's centre, but
mesh-n-bone also reads funlib's `offset` and precomputed's `voxel_offset` as centres (both
store the corner) and ignores COSEM `translate` and `downsamplingFactors`, CineMap's
`extent_nm()` and c-elegans' camera code treat the translation as a corner while their
reads treat it as the centre, cellmap-flow learned the OME rule in PR #103 after shipping
without it (salamander's predictions were half a voxel off until then), and chunkmirage's
TypeScript twin handles N5 differently from its Python.

funlib.persistence has the right vocabulary but the wrong backend for these consumers.
What its `Array` lacks, per cellmap-flow's session: tensorstore (s3, gs and http
throughput, the precomputed and zarr v3 drivers; `Array` reads through zarr-python and
dask, and a dask scheduler has no place in a per-chunk model server); reads that run past
the array and come back padded (`Array.__getitem__` raises on a Roi outside the data;
model halos at a volume's edge need the padding); level selection by target voxel size
with an explicit mismatch policy; the OME centre rule (funlib's Roi offset is a corner);
per-read retries and one shared, capped tensorstore context. Add the zarr-python
major-version split across the repos (cellmap-flow pins 2.18.4; cellmap-analyze and
mesh-n-bone pin >= 3.0.8; cellmap-analyze dropped funlib.persistence in 2026-04 because it
lacked zarr v3), the integer-only `Coordinate`, and pymongo, psycopg2 and dask as
dependencies of a reader. Nothing outside Janelia fills the niche: webknossos-libs,
neu-vol, cloud-volume, brainhops, yaozarrs and xarray-tensorstore were all checked on
2026-10-21 and each is a pipeline or a different model, not this dependency.

So: one small package, one class that serves both kinds of consumer, one written rule for
the half voxel, and conformance tests the TypeScript side can run too.

## 1. Scope

One class, `WorldArray`, with two faces. No legacy names: `ImageDataInterface`,
`open_ds_tensorstore` and `to_ndarray_ts` are not in this package; each migrating repo keeps
whatever shim it needs in its own subclass, for as long as it needs it.

- **Index space, native units** (`units=None`): `.ts`, `.meta` and `read_index`. For
  chunkmirage, CineMap, liconn-err, registration-station and any tool that works in its own
  coordinate frame. The metadata is read exactly as chunkmirage's
  `docs/concepts/formats.md` "Stored sources" specifies.
- **World coordinates** (`units="nanometer"` by default, or any unit per axis kind):
  `voxel_size`, `translation`, `corner`, level selection by voxel size, `read` of world
  boxes with padding, optional resampling and retries. For cellmap-flow, cellmap-analyze and
  mesh-n-bone, which all speak nm and Rois today, and equally for a `(t, c, z, y, x)`
  light-microscopy series or a `(time, lat, lon)` climate array.

Neither face assumes microscopy or three dimensions. Any number of axes, any axis names,
any units: what the store says is what the package reports, converted only on the axes it
is asked to convert.

Read-only in v1. Dependencies: `tensorstore>=0.1.60`, `numpy`. **No `zarr`, no `dask`, no
`pydantic`, no `funlib`.** The package is float-only; a consumer that works in integer Rois
builds them in its own subclass (section 6). Python >= 3.10 with no 3.11-only syntax, no
upper bounds on numpy or tensorstore.

Not in v1, by decision, each with its consumer named so it is not forgotten:

| Out of scope | Who asked | Where it lives instead |
|---|---|---|
| Writes (OME 0.5 multiscales, lock-free chunk writes) | registration-station, cellmap-analyze | each repo's writer; a `prepare` API is v2 |
| Arbitrary affines, BigStitcher XML | liconn-err | liconn-err; the package gives it zarr3 and N5 metadata and index reads |
| Oblique-plane sampling, chunk-grouped gather | CineMap | CineMap, on top of `.ts` and `chunks_covering` |
| CF packing (`scale_factor`, `add_offset`, `_FillValue`), xarray coordinate arrays | chunkmirage | chunkmirage, from `ArrayMeta.attrs`, `group_attrs` and `parent`, which the package exposes raw |
| HDF5 | chunkmirage | chunkmirage (h5py) |
| Normalisation chains, model-facing defaults | cellmap-flow | a `cellmap_flow` subclass of `WorldArray` |
| The integer-scaled frame, the zarr-python write handle | cellmap-analyze | a `cellmap_analyze` subclass of `WorldArray` |
| Decoded-chunk caches beyond tensorstore's pool, demand queues, serving | chunkmirage | chunkmirage |

## 2. Public API

```python
import worldarray as wa

vol = wa.open(
    path,                # an array, a level under a multiscale group, or a group
    level=None,          # a level name ("s1"), a position (0 = the first listed, "first"), or None;
                         #   ignored, with a warning, for an array path
    voxel_size=None,     # the size to read at, in `units`: picks the level (level_mode); see on_mismatch
    level_mode="floor",  # "floor" | "finer" | "exact" | "nearest" | "resample" (Behaviour 3)
    on_mismatch="error", # when no level has `voxel_size`: "error" (list the levels) | "resample" (read the
                         #   level `level_mode` picks and resample it, Behaviour 4) | "relabel" (read it voxel
                         #   for voxel and call its voxels `voxel_size`; no resampling, the grid is relabelled;
                         #   cellmap-flow's default, which its finetune reads rely on) | "keep" (the level
                         #   `level_mode` picks, at its own voxel size: cellmap-analyze's target_voxel_size)
    padding=0,           # any value the dtype holds (0, NaN, a uint64 sentinel) | "edge": what fills the
                         #   part of a box outside the data
    method="nearest",    # "nearest" | "linear": the resampling kernel; no dtype heuristic (Behaviour 5)
    select=None,         # {"c": 0, "t": 0}: pin non-spatial axes to one index; else they are kept whole
    units="nanometer",   # what world coordinates are in on spatial axes; a dict per axis kind
                         #   ({"space": "micrometer", "time": "second"}); None keeps the store's own units
    context=None,        # a tensorstore.Context; default shared_context(0)
    retries=None,        # a Retry; default Retry()
    metadata_conflicts="error",  # a per-array voxel_size disagreeing with the OME scale: "error" | "warn"
    base_voxel_size=None,  # declare level 0's true voxel size (in base_units): every level's spatial voxel
    base_units=None,       #   size and translation scale with it, for stores whose metadata is in voxels or unitless
    anchor=None,           # where a target voxel corner lies when resampling; the source corner by default
    transform=None,        # a callable applied to stored values before resampling and padding (a norm chain)
                         #   read(..., post_transform=) runs one after resampling instead, still before padding
    strict_units=True,     # False: an unknown unit ("pixel") is taken as the target unit, with a warning
)

vol.ts             # tensorstore.TensorStore: C-order axes, origin 0, selected axes dropped
vol.meta           # ArrayMeta in the store's own units
vol.grid           # ArrayMeta in the requested units, every axis
vol.shape, vol.dtype, vol.axes, vol.spatial, vol.level_name, vol.levels, vol.remote
                   # shape is the stored shape, or with a resample target the target voxels covering the data
vol.source_voxel_size, vol.resampled   # the stored level's voxel size; whether reads resample to a target
vol.list_chunks()  # the chunk indices that exist in storage, by listing keys (zarr v2/v3, N5, unsharded precomputed)
vol.path           # what the caller passed (a group, say); vol.level_path is the array actually opened
vol.voxel_size     # tuple[float] in `units`, spatial axes, the array's order (zyx for a volume)
vol.translation    # tuple[float] in `units`: the CENTRE of voxel 0        (OME-NGFF's word, OME-NGFF's meaning)
vol.corner         # tuple[float] in `units`: translation - voxel_size / 2 (what funlib, daisy, Neuroglancer call the offset)
vol.retries        # settable after construction (cellmap-analyze scales the timeout per handle)
vol.padding, vol.method   # the defaults `read` uses; settable

vol.read_index(begin, end, padding=None, mask=False, retries=None, select=None, transform=None)
                                    # select pins axes for this read only; begin/end cover the rest
vol.read(box, voxel_size=None, padding=None, method=None, mask=False, retries=None, strict=False, anchor=None,
         transform=None, post_transform=None, select=None)   # select pins non-spatial axes for this read
vol.index_box(box) -> (begin, end)  # the voxels a world box selects on this grid, for snapping a box first
vol.chunks_covering(begin, end)     # chunk indices a voxel box touches
vol.reopen(context=None, **overrides) -> WorldArray   # the same array on another context (a worker changing its concurrency)

pyr = wa.open_pyramid(path, context=None, units="nanometer", select=None)
pyr.names; pyr.levels; pyr.by_name("s1"); pyr.select(voxel_size, mode="floor")
pyr.metas() ; pyr.level_meta("s1")   # ArrayMeta per level from the documents alone: shape, dtype, chunks (a sharded
                    #   array's inner chunk), fill and placement, with no tensorstore opened
pyr.placements()    # (name, spatial voxel size, spatial translation) per level from the group's own document
pyr.voxel_sizes() ; pyr.translations()   #   where it lists the levels (OME-NGFF, COSEM N5, precomputed), so
                    #   choosing or placing a level opens nothing; else by opening each level

wa.metadata(path, level=None, units=None, open=True) -> ArrayMeta   # open=False: from the documents alone
wa.detect(path, missing_ok=False) -> Node | None   # format, kind (array|group), path, docs, scale_key; .attrs (raw user
                                                   #   attributes), .multiscales (OME 0.5's `ome` unwrapped)
wa.open_tensorstore(path, level=None, context=None) -> tensorstore.TensorStore   # reads no world metadata, so an
                                                   #   array with odd placement attributes (a coordinate array) still opens
wa.corner_to_centre(corner, voxel_size, axes=None) ; wa.default_axes(ndim) ; wa.index_range(begin, end, voxel_size, translation)
wa.units.TABLE      # the unit tables, public: {"length": {...metres per unit}, "time": {...seconds per unit}}
wa.select_level(levels, voxel_size, mode) ; wa.Box ; wa.Retry(max_attempts=10, timeout=None, base_delay=1.0)
wa.shared_context(cache_bytes=0) ; wa.capped_context(cache_bytes=0, concurrency=1, network=True) ; wa.capped_context_spec(...)
                    # network=False caps decoding and local file I/O only, leaving tensorstore's request pools alone
wa.list_chunk_keys(path, separator=".", ndim=None)   # chunk indices present under a path, from its keys alone: no
                    #   metadata document needed, so a directory of chunk files with no .zarray lists too
```

`ArrayMeta` is a frozen dataclass with one entry per axis, in the array's C order, and
deliberately the same field names as chunkmirage's `ArrayInfo`: `shape`, `dtype`,
`chunk_shape` (tensorstore's read chunk), `axes` (as stored; `("t","c","z","y","x")[-ndim:]`
when the store names none), `axes_named`, `units` (as stored, any string; `""` when none),
`voxel_size` (one index step, in `units`; 1.0 on unitless axes), `translation` (the centre
of voxel 0, in `units`; 0.0 where the store gives none), `format`, `path` (normalised),
`parent`, `attrs` and `group_attrs` (raw), and `placed_by`: which rule gave `voxel_size` and
`translation`, one of `ome`, `cosem`, `funlib`, `n5`, `precomputed`, `dimension_units`, or
None when nothing placed the array (then voxel size is 1 with the **corner at the origin**,
translation 0.5, as Neuroglancer draws a plain zarr), and `fill_value` as the store reports
it. Derived: `ndim`, `spatial` (indices of the axes
named z, y or x when any are; else the last three), `corner`, `in_units(space=, time=)`,
`in_nm()`, `drop_axis`.

A **box** for `read` is any of: a `(begin, end)` pair of float tuples in `units`, half-open
per axis, over the spatial axes, or over every axis when it has one entry per axis (a time
window and a spatial box at once); anything with `.begin` and `.end`, or `.offset` and
`.shape` (so a funlib `Roi` works, duck-typed, without the package importing funlib); or
`wa.Box`. `Box` carries funlib's Roi vocabulary on floats, half-open: `shape`, `centre`,
`empty`, `intersect`, `union`, `contains` (a point or a box), `grow(before, after=None)`,
`shift`, `snap(voxel_size, translation, mode="grow" | "shrink" | "closest")` to a grid's
voxel corners, `to_index` (the centre rule) and `Box.from_index`; `vol.box` is the array's
own extent on the spatial axes, `[corner, corner + shape * voxel_size)`, what funlib calls
the array's roi. So `vol.read(vol.box.intersect(roi).snap(vol.voxel_size, vol.translation))`
is the migration of a clipped, snapped Roi read, without funlib.

There is no `offset` attribute anywhere in this API, because that word means the corner in
funlib and the centre in OME; `translation` and `corner` are the two names for the two
things. `path` names the array that is read: an array path is that array whatever `voxel_size`
asks for (it resamples or relabels in place, never hops to a sibling level); only a group
path chooses a level. `level="first"` or `0` on an array path is silently the array.
There is no funlib `Roi` on the way out either: an integer corner-based Roi cannot
hold a 10.48 nm grid, nor a 5 nm grid with translation 0 (corner -2.5), so a bridge that
returned one would raise on inputs cellmap-analyze reads every day. A consumer that works
in integer Rois builds them from `corner`, `voxel_size` and `shape` in its own subclass,
where it also chooses the integer scale factor that makes the corner whole.

`shared_context(cache_bytes)` is one `tensorstore.Context` per budget for the process, so
the levels of a pyramid and repeat opens share one cache pool; the default budget is 0.
`capped_context` is mesh-n-bone's worker-process context (every tensorstore concurrency
resource limited, cache off); it returns a **new** context per call, as those workers
inline the spec into every open today, and `capped_context_spec` exports the dict for
callers building their own opens, writers included. `Retry` sizes the per-attempt timeout
from the read when none is given (Behaviour 6).

## 3. Conventions

These are the rules the package exists to hold in one place. They match chunkmirage's
`docs/concepts/formats.md`, "Stored sources", which its TypeScript twin follows too; where
this document and that one differ, fix one of them.

1. **A translation is where voxel 0's centre is**, as in OME-NGFF, COSEM's `transform` and
   CF coordinates. Index `i` on an axis sits at `translation + i * voxel_size`; voxel `i`
   covers `[translation + (i - 1/2) * voxel_size, translation + (i + 1/2) * voxel_size)`.
   Formats that store voxel 0's **corner** are moved half a voxel along the spatial axes
   on reading. N5 levels with `pixelResolution` and `downsamplingFactors` and no offset
   follow BigDataViewer: level-L voxel `i` is centred where full-resolution voxels
   `f*i .. f*i+f-1` are, that is `(f - 1) / 2` full-resolution voxels on from voxel 0's
   centre (the imglib2 rule liconn-err and registration-station both use). The shift
   applies only along axes named `z`, `y` or `x`, or the last three when axes are unnamed;
   never along channel or time.

   | Stored | Means | Read as `translation` |
   |---|---|---|
   | OME-NGFF `translation` (zarr v2/v3) | centre of voxel 0 | as is |
   | COSEM `transform.translate` (N5 or zarr) | centre (fibsem-tools wrote it as the first sample's position; Neuroglancer ignores `transform` and `offset` on N5 entirely) | as is |
   | funlib `offset` with `resolution`/`voxel_size` (N5 or zarr; `offset` missing = 0) | corner | `offset + voxel_size / 2` |
   | N5 `pixelResolution` x `downsamplingFactors f`, no offset | BigDataViewer | `(f - 1) / 2 * pixelResolution` |
   | precomputed `voxel_offset` x `resolution` | corner | `+ resolution / 2` |
   | CF coordinate array (xarray zarr) | sample position = centre | as is (chunkmirage layers this) |
   | nothing at all | Neuroglancer draws a plain array with its corner at the origin | voxels of 1, translation 0.5 |

   Stored numbers are never rounded onto a grid: a funlib `offset` of 4 at 8 nm stays 4
   (cellmap-flow rounded it to 8). A group's `multiscales` transform for an array outranks
   the array's own `resolution` (chunkmirage's documented order), and an OME scale outranks
   a per-array `voxel_size`.

   Neuroglancer does the same, so placement by `translation` matches what Neuroglancer
   shows. Its zarr datasource (`src/datasource/zarr/ome.ts`, verified 2026-10-21) says: "In
   OME's coordinate space, the origin of a voxel is its center, while in Neuroglancer it is
   the 'lower' (in coordinates) corner. Translate by the physical size of half a voxel",
   and does so. Its precomputed datasource draws voxel `i` over
   `[(voxel_offset + i), (voxel_offset + i + 1)) * resolution`, whose centre is the table's
   `(voxel_offset + 1/2) * resolution`. `corner` is Neuroglancer's internal lower corner of
   voxel 0; a tool computing extents or camera positions from `translation` as if it were a
   corner is half a voxel off from the viewer.

2. **Axis order is the store's C order; N5 and precomputed are transposed to it** on read,
   so `(z, y, x)` for a volume, `(c, z, y, x)` with channels, `(t, c, z, y, x)` for OME
   0.5. Nothing is reordered beyond that transposition; `select` pins non-spatial axes. A
   non-zero stored origin (precomputed `voxel_offset`) is moved to 0 in `.ts` and carried in
   `translation`. \1 Sharded stores (zarr v3 `sharding_indexed`, sharded precomputed) read like any
   other through tensorstore; `chunk_shape` is the inner read chunk, not the shard. Unnamed axes are called `("t", "c", "z", "y", "x")[-ndim:]` (chunkmirage's
   default) with `axes_named` False, so a consumer can apply its own default instead.

3. **Units are kept as stored and converted only on the axes a caller asks to convert**
   (`ArrayMeta.in_units(space=, time=)`, `in_nm()`, the `units` argument), through one table:
   for length, American and British spellings, `nm`, `um`, `µm` (U+00B5), `μm` (U+03BC),
   `mm`, `cm`, `m`, `angstrom`/`Å`, `pm`; for time, `s`/`second`, `ms`, `us`/`µs`, `ns`,
   `minute`, `hour`, `day`. Any unit string on any axis is accepted and never rejected until
   a conversion is asked for (`"s"`, `"m"`, `"degrees_north"`, `""`). A spatial axis with no
   unit is taken as nanometres, with a **warning once per array** (every repo assumed this
   silently; a BigDataViewer N5 whose unitless `resolution` is in micrometres would be read
   a thousandfold too small, and the warning is what tells you; `base_voxel_size` is the
   fix); an unknown unit string on an axis being converted is a `ValueError`, never silently the target unit; an axis not
   being converted keeps its string. Voxel sizes and translations are floats and are never
   rounded or truncated by the package: a 10.48 nm voxel stays 10.48. `snap_integral` is
   used only to absorb float noise such as 9.9999999 before comparing or ceiling.

4. **Metadata precedence per format, first match wins** (chunkmirage's table):
   - zarr v2 / v3: the OME-NGFF `multiscales[0].datasets` entry whose `path` is this array,
     matched **by name never by position**, in the parent group or, when the dataset path
     has more than one component (`"seg/s0"`), in an ancestor up to four levels above; the
     parent is read in both zarr formats, because a zarr v3 array inside a v2 group whose
     `.zattrs` holds the multiscales is a layout in the wild. A parent that lists levels but
     not this array warns, and the array is placed by its own attributes or not at all; composed with the
     multiscale-level `coordinateTransformations` when present (0.4 and 0.5; under
     `attributes.ome` in 0.5). An OME 0.6 group with `coordinateSystems` is read for its
     intrinsic system and never raises, chunkmirage's rule: take the coordinate system named
     by the dataset's first transformation's `output`, else the first system; read axes and
     units from it; compose only the dataset's own `scale` and `translation`, unpacking a
     `sequence`; ignore the multiscale-level transforms (they lead to other systems, which
     chunkmirage's `scene://` applies itself) and any other transform type. Else the array's
     own `resolution` / `voxel_size`, `offset`, `units`, `axis_names` (funlib) or `transform`
     (COSEM), C order; else tensorstore's `dimension_units` attribute (zarr v3). Axes from
     OME `axes`, `dimension_names` (zarr v3), `_ARRAY_DIMENSIONS` (xarray) or `axis_names`.
   - N5: the array's `transform` (COSEM; C order unless `ordering: "F"`); else the parent's
     `multiscales[].datasets[].transform` for this path; else `pixelResolution` (both the
     `{dimensions, unit}` form and a bare list) or `resolution`, from the array else the
     group, times the level's `downsamplingFactors`; plus `offset`; axes from the `axes`
     attribute. These lists are x-first. Without an `offset`, a level with `downsamplingFactors` follows
     BigDataViewer whichever key gave the base (BigStitcher writes `resolution` and
     `downsamplingFactors`), and a funlib `resolution` with no `downsamplingFactors` means a
     corner at 0 (translation half a voxel); funlib never writes `downsamplingFactors`.
   - **Short lists.** A funlib, COSEM or N5 per-axis list shorter than the array (a 3-entry
     `voxel_size` on a `(c, z, y, x)` prediction, which funlib's `prepare_ds(num_channels=)`
     writes) covers the trailing axes; the leading axes get voxel size 1, translation 0,
     unit `""` and the default axis names. OME lists must match the array exactly. A zarr
     `offset` with no voxel size places the array with voxels of 1.
   - precomputed: `resolution` (nm, xyz) and `voxel_offset` (voxels, the corner) of the
     scale; levels are the `scales` entries in order, chosen by `scale_index`, by key, or by
     the last path segment when the path names a scale.
   Everything the parser did not consume stays available in `attrs` and `group_attrs`.
   When an array carries both an OME `translation` and an `offset`, the OME translation
   wins and the `offset` is checked against it. Two values pass silently, because both are
   written in the wild: `offset + voxel_size / 2 == translation` (funlib's `prepare_ds`
   writes the corner `offset` beside the centre `translation` on `s0`) and
   `offset == translation` (cellmap-analyze's `create_multiscale_dataset` writes the centre
   as `offset`). Anything else logs a warning naming the path and both values; it is never
   an error (cellmap-analyze raised on the funlib case until its PR #87 and salamander hit
   that). Voxel size is stricter: an OME `scale` and a per-array `voxel_size` /
   `resolution` that disagree beyond `rtol 1e-6` raise `ValueError`, as cellmap-analyze does
   deliberately, because that is how stale attributes are caught and no choice between
   them is safe; `metadata_conflicts="warn"` lets a server show the image anyway, with the
   OME scale.

5. **Stores are read through tensorstore's kvstore only**, never zarr-python: `zarr.json`,
   `.zarray`, `.zattrs`, `.zgroup`, `attributes.json`, `info`. Format detection probes keys
   in this order: `zarr.json`, `.zarray`, `.zgroup`, `info`, `attributes.json`; when none is
   found, the parent is probed for `info` and the last path segment is taken as a
   precomputed scale key (mesh-n-bone's case), then `s0` is probed so a group with no
   document of its own (an N5 directory holding `s0, s1, ...`) is still a group. The
   filesystem root, a URL's host root and a bucket root are never probed, and an ancestor
   that answers anything but a document (a server root's 301) ends the search for a
   multiscales entry quietly. Off Google Cloud, `GCE_METADATA_ROOT` is pointed at a name that
   fails at once before the first `gs://` open, because tensorstore's probe of the GCE
   metadata server otherwise stalls every open for seconds (`store.GUARD_GCE_METADATA`
   turns this off). A `precomputed://` prefix is accepted and
   dropped. The `.n5` or `.zarr` suffix is never used to decide the format. A zarr v2
   compressor carrying members tensorstore rejects (the `checksum` newer numcodecs writes
   for zstd) is opened without them. Access: a path with no scheme is a local file; `s3://`
   is tried anonymously then with default credentials; `gs://` with default credentials then
   through the bucket's public https URL (`TENSORSTORE_GCS_HTTP_URL` overrides);
   `http(s)://`, `file://`, `memory://` one way. `AWS_ENDPOINT_URL_S3` or `AWS_ENDPOINT_URL` names an
   S3-compatible endpoint (MinIO, moto, a mirror) and `AWS_REGION` or `AWS_DEFAULT_REGION`
   the region, as the AWS tools read them; `TENSORSTORE_GCS_HTTP_URL` is tensorstore's own
   redirect for `gs://`, and the opened-kvstore cache is keyed on it, so a process that
   changes it between opens reads the right host. An N5 `units` written once as a string
   is read as that unit on every axis; tensorstore is given only the array members of
   `attributes.json`, so attributes it would reject do not stop the open. The next way is tried only when the error
   starts `PERMISSION_DENIED` or `UNAUTHENTICATED`; the way that worked is remembered per
   (driver, bucket) for the process. Format probing runs all probe keys under one way before
   moving to the next: a bucket with listing off answers 403 for a missing key, so a 403 on
   one probe key counts as "missing" when another probe key was read under the same way,
   and only when every probe key is refused is the next way tried (see Open question 4).
   Metadata reads retry for about five seconds in total (`max_retries 5, initial_delay
   0.2s, max_delay 2s`); data reads keep tensorstore's defaults plus `Retry`.

## 4. Behaviour (each statement is a test)

1. **Index reads clip, read, pad.** `read_index(begin, end, padding)` returns an array of
   shape `end - begin`; the part inside the data is the stored values; the rest is
   `padding` (a value) or the nearest stored voxel (`"edge"`). Padding is never resampled or
   scaled. A box entirely outside the data returns the fill without touching storage. With
   `mask=True` a boolean array of the same shape, True inside the data, is returned as well
   (cellmap-analyze's `return_valid_mask`). The result is C-contiguous in native byte order
   (mesh-n-bone byte-swaps big-endian stores after every read today). `padding` and the box
   are per-read arguments, never only constructor state, because one cached handle per
   worker serves different kinds of reads (mesh-n-bone's meshing reads blocks it has grown
   by one output voxel per side; its pyramid builder reads ungrown blocks). `padding` is the
   fill only. There is no context or halo argument: a caller that wants context grows its
   own box. `transform` is applied to the stored values before padding.

2. **World boxes select voxels by their centres.** For a box `[b, e)` on an axis with voxel
   size `s` and translation `t`, the voxels read are `i` with `ceil((b - t) / s) <= i <
   ceil((e - t) / s)`, after `snap_integral` on both quotients: a quotient within `1e-6` of a
   whole number of voxels is that whole number, so a centre-aligned begin that computes to
   `k + 1e-12` is `k`, not `k + 1`. A voxel is read iff its centre lies in the box.
   Consequences the tests pin: a box of extent `m * s` returns exactly `m` voxels whatever
   its alignment; a box aligned to voxel **corners** (every cellmap-analyze block, every
   cellmap-flow Roi) maps to exactly the voxels it covers; a box aligned to voxel
   **centres** (mesh-n-bone's blocks begin at `translation + k * s`) maps to voxels
   `k .. k + m - 1`, also exactly, because the interval is half-open at the centres and
   nothing is rounded; the read of the volume's own extent `[corner, corner + shape * s)` is
   the whole array with no padding. A box aligned to neither is still well defined by the
   centre rule; `read(..., strict=True)` raises instead for callers who want to be told.

3. **Level selection** (`select_level`, `Pyramid.select`, `open(voxel_size=)`) follows
   cellmap-flow's four modes plus one: `floor` (the level at that voxel size, else the
   **coarsest** level that is no coarser than the target on any axis, else the first),
   `finer` (as `floor`, but an error instead of the fallback to the first level, which is
   cellmap-analyze's rule), `exact` (that
   level or `ValueError` listing the levels), `nearest` (closest on a log scale summed over
   axes, finer on a tie), `resample` (the level to resample from). Voxel sizes compare as
   floats with `rtol 1e-6`, so 10.48 is not 10. A group path without `level` or
   `voxel_size` raises with the level list; a level name that does not exist raises with
   the list; an array path with `level` warns and ignores it. When no level has the
   requested `voxel_size`, `on_mismatch` decides: `"error"` raises with the list;
   `"resample"` reads the level `level_mode` picks and resamples (Behaviour 4); `"relabel"`
   reads that level voxel for voxel and reports `voxel_size` as the requested size with
   **voxel `i` staying voxel `i`**: every world quantity, the translation and the corner
   included, scales by the ratio of the sizes (cellmap-flow's rule, which chunkmirage's
   server relies on), so a box of `m` requested voxels returns `m` stored voxels of the
   level; `"keep"` reads the chosen level at its own voxel size (a level pick, nothing
   more). `relabel` and `nearest` are different things and both exist: one is a grid
   policy, the other a resampling kernel. `level` also takes a position, `0` or `"first"`
   for the first listed level. The pyramid an array belongs to is found through
   its parent's `multiscales`; a group without `multiscales` is probed for `s0, s1, ...`.

4. **Resampling** (`voxel_size=` with no level at it and `on_mismatch="resample"`, or
   `read(voxel_size=)`) follows cellmap-flow's `io/resample.py`: the target grid's voxel 0
   starts at the source **corner**, has `ceil(n * s / t)` voxels, and target voxel `j` is
   centred at `corner + (j + 1/2) * t`, which is source index `p = (j + 1/2) * t / s - 1/2`.
   `"nearest"` takes the source voxel whose extent `[corner + i * s, corner + (i + 1) * s)`
   holds the target centre: nearest neighbour as ITK defines it, and the same half-open
   rule as Behaviour 2. A centre on a voxel boundary, which every even whole factor
   produces, belongs to the upper voxel, so a factor of 2 reads source voxels `1, 3, 5, ...`
   where a stride read `0, 2, 4, ...`; the stride is a half-block bias and the change is
   intended. The target grid is **anchored** at the source corner unless `anchor` names
   another world position for a target voxel corner; passing another volume's `corner`
   reads this volume onto that volume's grid, which is what contact sites, masks and
   intensity statistics need (cellmap-analyze's case: an 8 nm array with translation 0
   read onto a 4 nm grid with corner -2 gives `0, 1, 1, 2, 2, 3, 3, 4`, where the
   source-anchored read gives `0, 0, 1, 1, 2, 2, 3, 3`, half a voxel off). `"linear"` takes the block mean when `t / s` is a whole number >= 2; a block cut short
   by the end of the data is the mean of what it has (a pyramid's last block), and a block
   that overlaps no data is outside; else linear interpolation between the two neighbours
   with the border voxel repeated past the first and last centre; a whole factor with an anchor off the
   source grid also interpolates. Integer dtypes are rounded back to their
   dtype. The source voxels read are derived from the first and last **target centres** of
   the box, not from the box edges, then grown by the kernel radius (0 for nearest, 1 for
   linear) and clamped to the data with the border repeated; a block mean on a box aligned
   to the target grid needs no margin. Hence a target voxel's value depends only on its
   index and the source array, never on where the read starts or ends: two adjoining reads
   agree with one read across both (the chunk independence property, tested on boundaries
   at every method; cellmap-analyze's current resampler fails it). Target voxels whose
   centres lie outside the data get the fill (with `"edge"`, the border voxel for nearest
   and linear, the last block holding data for the mean), and the mask marks them. With a
   resample target, `shape` reports the target voxels covering the data on each spatial
   axis. A `transform` (a normalisation chain) is applied to the stored values before
   resampling and before padding, so padding is never transformed and the output takes the
   transform's dtype; `post_transform` runs after resampling and the cast back to the
   stored dtype, still before padding, for a normalisation that wants the resampled block.
   A box entirely inside the array is returned as tensorstore's own block, without a copy,
   a mask or a fill, under every padding mode; the mask is built only when asked for.
   Measured on a 4 MiB block with the page cache warm, `read_index` and `read` cost 0.1 to
   0.2 ms over a raw tensorstore read, about half of what the reader they replace cost
   (Tests: the benchmark script in the trial notes, not part of the suite). Block means over
   blocks that tile the read are one `reduceat` pass.

5. **No dtype heuristic.** The package never decides that an array holds labels. `method`
   defaults to `"nearest"`, which invents no values and is correct for labels and acceptable
   for intensities; `"linear"` is opt-in. cellmap-flow's subclass sets `padding="edge"`,
   `on_mismatch="relabel"` and picks `method` with its own `is_label_dtype` rule;
   cellmap-analyze passes `method` from its `interpolation_order`.

6. **Retries** apply to data reads only, with mesh-n-bone's formula, which its tests
   already pin: attempt `k` (from 1) waits `timeout * k` seconds on `.result()`, where
   `timeout` is the given value or, when `None`, `clamp(MiB / 16, 5, 120)` for a remote
   store (`gs://`, `s3://`, `http(s)://`) and `clamp(MiB / 256, 5, 30)` for a local one
   (`file`, `memory`), `MiB` being the size of the in-bounds part of the read after
   clipping; between attempts the sleep is `base_delay * 1.3 ** (k - 1)` plus `U(0, 1)`
   seconds; at most `max_attempts` (default 10). Retried: `TimeoutError`, and a tensorstore
   error whose message starts with `UNAVAILABLE`, `DEADLINE_EXCEEDED` or
   `RESOURCE_EXHAUSTED` or carries an HTTP 5xx or 429. Any other error is raised at once. A
   timed-out tensorstore future is cancelled before the next attempt starts. The policy is a
   per-handle attribute and a per-read argument, never only constructor state.

7. **Contexts.** Without `context`, every open in the process shares `shared_context(0)`:
   tensorstore's cache is off and the levels of one pyramid share one context.
   `capped_context` sets `data_copy_concurrency`, `file_io_concurrency`,
   `gcs_request_concurrency`, `s3_request_concurrency` and `http_request_concurrency` to
   `concurrency` and the cache to `cache_bytes`, and returns a new context each call, so two
   handles never share one capped pool unless the caller passes the same object twice. Both
   are plain tensorstore contexts; a caller may pass its own, at construction or through
   `reopen(context=)` when a long-lived worker changes its concurrency (cellmap-analyze
   reopens tensorstore on every read today for that reason).

8. **Thread safety.** A `WorldArray` may be read from several threads at once; nothing in the
   package holds a lock around a read. No dask anywhere.

9. **Non-spatial axes.** `read_index` works on all axes. `read` takes a box over the spatial
   axes and keeps leading non-spatial axes whole unless `select` pinned them, or a box over
   every axis, in which case Behaviour 2's centre rule applies along each axis in that
   axis's converted unit (a time sample is a point too, so a window `[t0, t1)` in seconds
   reads the samples whose times lie in it). The returned array's axes are `vol.axes`.

## 5. Tests (behavioural; synthetic stores in `tmp_path`; one optional network test)

Conformance, one tiny store each, checking `voxel_size`, `translation`, `axes`,
`axes_named`, `units`, `shape` and, where stated, `dtype`, the raw attributes and a
read-back. Stores and expected values live in `tests/fixtures/metadata.json` so
chunkmirage's TypeScript side can build and run the same cases; a case whose values
chunkmirage derives from raw attributes carries a `layered` note beside the package's
`expect`:

- chunkmirage's cases: N5 transposed to C order with `resolution` and `offset`; N5 COSEM
  `transform` C order; N5 group `pixelResolution` and `downsamplingFactors`
  (BigDataViewer); N5 group `multiscales` transform matched by path; OME levels matched by
  path; OME top-level transforms applied after each dataset; an OME-unlisted array borrows
  no other level's scale; zarr legacy array attrs as a corner; a zarr3 OME 0.5 group with a
  channel axis; precomputed `voxel_offset` as a centre translation; xarray dimensions and
  packing exposed raw with the dtype as stored.
- chunkmirage's additions: N5 with a bare-list `pixelResolution`; an OME 0.6
  `coordinateSystems` group (reads the intrinsic system, never raises); a zstd `.zarray`
  carrying `checksum`; and, from the trials: short funlib, COSEM and N5 lists on a 4-D
  array; N5 `resolution` without `offset`; the N5 `axes` attribute; a zarr `offset` with no
  voxel size; a zarr v3 array in a v2 OME group; a dataset path with two components. Every
  case states `placed_by`.
- units: `µm`, `μm`, `um`, `micrometre`, `nanometer`, missing (nm), unknown (`ValueError`
  only when converted; a factor of 1 with a warning under `strict_units=False`); an N5
  `units` string; a `scale` attribute as a voxel size.
- an array with both an OME `translation` and an `offset` equal to it, or to it less half a
  voxel, reads the translation silently; any other pair warns and still reads the
  translation; an OME `scale` disagreeing with a per-array `voxel_size` raises.
- access fallback, offline, with a fake attempt function: a refusal moves to the next way,
  any other error raises at once, the way that worked is tried first next time for that
  bucket, and a 403 on one probe key with another probe key readable does not trigger the
  fallback; detection ignores the directory suffix and finds a precomputed scale key.

Reads: every statement in Behaviour 1, 2, 3 and 9; a document-less group probed through
`s0`, integer and `"first"` levels, `detect(missing_ok=True)`, `index_box`; `read_index`
with `select`; `base_voxel_size` rescaling a BigStitcher-style pyramid; `AWS_ENDPOINT_URL`,
`AWS_ENDPOINT_URL_S3` and the region reaching the S3 specs and the GCS redirect keying the
kvstore cache; an array path kept under `voxel_size`; relabel keeping the corner; a
`transform` applied before padding and before resampling; `list_chunks` on zarr v2, zarr v3
and N5; a pyramid choosing a level from the group document without opening one; a
precomputed `s<N>` path; the centre rule on a 10.48 nm grid with
translation 3.7 at indices around 1e5, corner- and centre-aligned (mesh-n-bone's float-noise
case); a `(t, c, z, y, x)` OME 0.5 store read with a time window and a spatial box, and with
`select`.

Resampling: `"nearest"` takes the source voxel holding each target centre (factor 2 reads
`1, 3, 5, 7`; a 5 nm read of 4 nm voxels reads `0, 1, 3, 4, ...`); the anchored read onto
another grid from Behaviour 4, and its two halves agree with the whole; `"linear"` at a whole factor is the block mean; two adjoining
reads agree with one read across both at both methods and at whole and non-whole factors; a
2x `"nearest"` read of an integer label volume introduces no new values and pads outside
the data; a resample target set at open reports the target voxel size with the same corner.

Placement: the same cube stored four ways, as OME-NGFF with a `translation`, as N5 with
funlib `resolution`/`offset`, as N5 with a COSEM `transform`, and as precomputed with a
`voxel_offset`, reads back with the same `translation` and the same bytes for the same
world box, within `0.1 * voxel_size`; and an N5 BigDataViewer level 1 sits where the
equivalent OME level 1 does.

Network (skipped without it; not yet written): one public OpenOrganelle volume with a
non-integer z spacing (4.56 or 5.24 nm), read anonymously from s3: the package reports the
float voxel size at every level where cellmap-flow's `Coordinate` wrapping reports 5 and
10, and one corner-aligned box returns the same bytes as cellmap-flow's reader.

Target: about 600 lines of test cases, the store builders in `conftest.py` aside; no
wording pins; no `/nrs`; no cluster.

## 6. Consumers and migration

| Reader today | Face used | What moved into the package from it | Migration |
|---|---|---|---|
| cellmap-flow `image_data_interface.py` + `io/` | world | per its session: `io/paths`, `io/store`, `io/metadata` (ArrayMeta, `snap_integral`, `regularize_offset`, the unit table), `io/multiscale` (`select_level`), `io/geometry` (`Box`), `io/source` (`read_padded`), `io/resample` (`Resampling`), and `io/ome`'s reading half | 77 import sites on `main` at 02fa82e3 (18 of `image_data_interface`, 14 `io.geometry`, 12 `io.metadata`, 11 `io.ome`, 10 the package, 4 `io.source`, 4 `io.multiscale`, 2 `io.store`, 1 `io.resample`, 1 `io.paths`); none in `web/test/fixtures` or `web/build`. `io/*` becomes re-exports for one release. Stays: the input chain (`process_chain`, `apply_norms`, `LazyNormalization`, `with_input_norms`, `norms_to_apply`), `legacy_meta` (deprecated; `viewer/bootstrap.py` still reads it), the deprecated `output_voxel_size` and `custom_fill_value`, the mismatch warning texts, and `io/ome`'s writer half until a v2 writer. `ImageDataInterface` becomes a `WorldArray` subclass setting `padding="edge"`, `on_mismatch="relabel"` and its label heuristic, and applying the chain at read time. Own branch into cellmap-flow's `consolidate` after its wave A; `tests/utils/test_io_placement.py` kept for parity until done |
| cellmap-analyze `util/image_data_interface.py` and the read side of `zarr_io.py`, `voxel_size_utils.py`, `io_util.py`, `precomputed_io.py` | world, through a local subclass | `read_with_retries` (per-handle tunable timeout), `return_valid_mask`, the `"edge"` and uint64-sentinel fills, `concurrency_limit` as a per-worker context, `target_voxel_size` (= `voxel_size` with `level_mode="floor"`) and `output_voxel_size` (= `on_mismatch="resample"`); read-side code to absorb: `_detect_zarr_driver`, `open_ds_tensorstore`, `io_util`'s path and kvstore helpers, `precomputed_io`, `voxel_size_utils` (`get_multiscales` with NGFF 0.5's `ome` nesting, `_select_ome_dataset` by path name, `_read_parent_attrs`, the PR #87 corner acceptance), `zarr_io.resolve_scale_path` and `_read_voxel_size_offset` | **Not an import swap.** cellmap-analyze works in an integer-scaled frame: `voxel_size`, `roi`, `chunk_shape` and `output_voxel_size` are `Coordinate`s in nm times `voxel_size_scale_factor` (62, 21, 7 and 22 sites; `rescale_to_factor` 10 sites). Its IDI is also the write handle: about 17 sites write `output_idi.ds[block.write_roi] = data` on a zarr-python array opened `r+`. So it keeps a local subclass that owns the scaled frame (built from `corner`, `voxel_size` and `shape`, choosing a factor that makes the corner whole), the zarr-python `.ds` writer, the `chunk_shape` override (block size, not storage chunk) and the fill value, and delegates `to_ndarray_ts(roi)` (34 sites) to `read()` after dividing the box by the factor. Drop `to_ndarray_ds`; `interpolation_order` is never passed. Resampled reads change bytes (Behaviour 4), so parity is for unresampled reads only. Uses dask with its own block bookkeeping, not daisy; funlib.geometry is its only funlib dependency |
| mesh-n-bone `util/image_data_interface.py` + the metadata half of `zarr_io.py` | world (functions) | `capped_context` and its spec, the size-and-locality timeout rule, remote detection, the `info`-in-parent probe, and `zarr_io`'s metadata resolution | `open_ds_tensorstore`, `read_with_retries`, `to_ndarray_tensorstore` kept as module functions with their exact current semantics (`index = (p - translation) / voxel_size`, no half-voxel term) for one release; `swap_axes` disappears because the package always returns C order; funlib `offset` and precomputed `voxel_offset` start being read as corners, COSEM `translate` is read, and `downsamplingFactors` levels gain the BDV shift: outputs move on those inputs (Open question 2); its session wants the change and no package-side flag |
| chunkmirage `sources/tensorstore_source.py` | index | access fallbacks, metadata retries, probe order, `corner_to_centre`, the N5 precedence rules, compressor cleaning, `shared_context`, the conformance tests | a thin adapter `ArrayMeta -> ArrayInfo`; CF packing and xarray coordinates layered from `attrs`, `group_attrs` and `parent`; planned, not optional: until then two Python parsers encode the half-voxel rule |
| CineMap `slice_loader.py` | index | zarr v2 `checksum` tolerance | optional: open and metadata through the package, the oblique gather on `.ts` and `chunks_covering`; its `extent_nm()` is half a voxel off from Neuroglancer (Open question 3) |
| liconn-err `dataset.py` | index | nothing new | optional: zarr3 and N5 opens and pyramid factors; affines stay |
| registration-station | index | nothing new | optional: metadata and slab reads; writes stay |
| salamander scripts, c-elegans generators | world | nothing new | optional: replace hand-parsed OME multiscales; both compute camera or extent positions from the translation as a corner today |
| segmentation-challenge-meshes, cellpose scratch, skoots-mito, plasmodesmata, c-elegan-experiments, tier1_exp, dinov3_playground, cellmap-segmentation-challenge | – | nothing | archive or leave; none is live |

Order: the package with tests and CI, tagged 0.1.0 on PyPI (every live consumer is itself
a PyPI package, so a git dependency is not an option); mesh-n-bone and cellmap-analyze
(smallest, and they hold the features the package absorbed, so they find API gaps
cheaply); cellmap-flow; chunkmirage; the rest as their owners choose. Every consumer pins
a version range (`>=0.1,<1`), never a commit; the commit pins that exist today exist only
because chunkmirage was not on PyPI.

### Trial status, 2026-10-21

Each repo's session put worldarray beside its environment and compared it with its own
reader on its own data; none merged or pushed anything. Correct on every store tried, with
byte-identical reads where compared: cellmap-analyze-scripts (one script; drops the
scale-factor bookkeeping and its environment constraint), salamander (three arrays
including a prepare_ds-written prediction; four scripts, 40 to 60 lines),
registration-station (16 OME 0.5 levels; 1.5 to 2x faster than zarr-python cold, more
warm; two parsers and ten diagnostics scripts), liconn-err (three tiles including a BDV
N5; 33 of 148 lines; exposed two stores with wrong or missing units, hence the warning in
Conventions 3 and `base_voxel_size`), CineMap (sharded zarr v3 over https; about 100 of
486 lines), chunkmirage (323 of 326 tests, the other three deleted as covered;
`tensorstore_source.py` 649 to 219 lines; found the short-list blocker and the N5
`resolution` rule, both fixed), mesh-n-bone (320 passed; src +181/-1531 with
`precomputed_io.py` gone and `zarr_io.py` 768 to 16; the five-way cube exact on all five;
found the nested dataset path bug, fixed, and two placement bugs of its own, fixed on its
branch), cellmap-analyze (317 passed, 7 failed of which 5 were worldarray gaps, all fixed
here: the S3 endpoint, short lists, the hybrid layout, `keep` and `finer`, the resample
anchor; about 800 lines deletable now, 640 more with three call-site edits; its re-run on
the fixes: 329 passed, the 4 left being its own pins of intended changes), cellmap-flow
(against the first build: `io/` shrinks by 1,035 of 2,201 lines, but 97 tests failed on
real differences; its three blockers, the S3 endpoint variables, the GCS redirect cache
key and an N5 `units` string, are fixed here, its convention points are decided above with
Neuroglancer and fibsem-tools as the tie-breakers, and its mapping asks are in: an array
path keeps its array, relabel keeps the corner, `shape` reports the target, a `transform`
hook, `list_chunks`, level choice from the group document, `strict_units`, the public unit
table, `fill_value`, `s<N>` precomputed paths). cellmap-flow's third run against the fixed build: 1,141 passed and one known failure of its own; its `io/` 2,201 to 1,065 lines; its 4 MiB block read re-measured within 0.1 to 0.5 ms of raw tensorstore and level with its base on both padding modes. Every live consumer's session has declared worldarray acceptable to publish. Every repo that writes asked for the v2 write API: create an OME
0.5 multiscale group and array from voxel size, centre translation, axes and units;
race-safe region writes from many jobs; add a level to a group; encode a chunk key.

## 7. Open questions

1. **Published mesh sets.** Measured on 2026-10-21 by meshing one cube (4 nm voxels, voxel
   0's corner at the origin, label at voxels 8 to 23, so world 32 to 96 nm) stored five
   ways and run through mesh-n-bone's own `Meshify` with its integration-test settings:
   OME `translation` 2 and cellmap-analyze-style (`translation` 2 beside `offset` 2) meshed
   at exactly 32 to 96; pure funlib (`resolution` + `offset` 0, no OME), prepare_ds-style
   (`translation` 2 beside funlib `offset` 0) and precomputed (`voxel_offset` 0) meshed at
   30 to 94, half a voxel low against where Neuroglancer draws those voxels. The mesher
   itself is right (zmesh puts the surface on voxel faces around sample `i`); the error is
   that mesh-n-bone reads a funlib `offset` and a precomputed `voxel_offset`, both corners,
   as if they were the centre, and reads the array's `offset` before the parent's OME
   translation. Under this spec those three inputs move by half a voxel to match
   Neuroglancer. Must any published mesh set built from such input stay byte-compatible?
   mesh-n-bone's session knows of none and wants the change without a compatibility flag.
   The check script is `check_placement.py` in this session's scratchpad; it belongs in
   mesh-n-bone's tests with a tolerance below half a voxel (its coordinate test uses one
   whole voxel and cannot see this).
2. **CineMap's convention.** Its reads follow the centre rule and match Neuroglancer; its
   `extent_nm()` treats the translation as a corner and does not. Which does CineMap follow?
3. **Listing-off S3 buckets.** The probing rule in Conventions 5 is written from
   chunkmirage's description of the gotcha and tested offline with a fake store, not against
   a real bucket with listing disabled. One network test before 0.1.0 may change the rule.
4. **The network test** against a non-integer-voxel OpenOrganelle volume (Tests) is not yet
   written; it needs the dataset chosen from the catalog.

## 8. Constraints for whoever builds on this

Branch off `main`, draft PRs, no "Test plan" section in PR bodies, never add Claude as git
author or committer, `/nrs/cellmap/data` is read-only, no cluster jobs, few behavioural
tests, minimal reusable code. The README is sections 1 to 5 of this file; a behaviour that
is not stated here is not implemented without adding the statement first.
