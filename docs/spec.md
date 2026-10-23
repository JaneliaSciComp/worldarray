# Specification

worldarray is built to this document. Sections 1 to 5 are also the README; every
statement under "Behaviour" and "Tests" is something a test pins. Code that is not needed
to make a statement here true does not go in; a behaviour not stated here is not
implemented without adding the statement first.

Draft 3, 2026-10-22. Written from a reading of every existing reader in David Ackerman's
repos and the answers of the chunkmirage, cellmap-flow, cellmap-analyze, mesh-n-bone,
CineMap, liconn-err, registration-station, salamander and cellmap-analyze-scripts
sessions; the write side (0.2) and the grid side (0.3) were added on 2026-10-22 from the
five writer sessions' answers and the line census in section 6. Decisions still open are
in section 7. Version map: 0.1 reads (Behaviours 1 to 9); 0.2 metadata and placed
creation (10 to 16); 0.3 grids and blocks (17 to 19). Nothing in 0.2 or 0.3 changes a
0.1 behaviour.

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

The write side has the same disease in a smaller body. Measured on 2026-10-22: five repos
generate OME multiscales documents by hand (cellmap-analyze, mesh-n-bone, cellmap-flow,
registration-station, chunkmirage; 440 lines between them), five create placed arrays with
five different answers to "it already exists" (an unverified adopt zeroed 180 rows of a
registration-station field), four encode chunk keys, and every write-side bug their
sessions could name was a metadata bug: a scale factor that ignored the corner
(cellmap-analyze #93), a level translation that ignored the ROI origin (mesh-n-bone, on
master), OME 0.4 attributes inside zarr v3 groups (two repos), a hard-coded nanometre, a
funlib `offset` holding a centre. The bytes were never the problem: they go through
tensorstore or zarr-python and nobody reported a byte-level bug. So 0.2 takes the metadata
and the placed creation, and nothing else of writing.

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

0.1 reads. 0.2 adds the write side's metadata and placed creation (Behaviours 10 to 16):
the OME document builder and the level rule, chunk keys, `create`, opening for write,
aligned writes, `add_level`. 0.3 adds grids (Behaviours 17 to 19): a store-less `Grid`,
the block iterator every blockwise repo writes for itself, `write_chunk`, the factor
schedule and the public path helpers. Dependencies at every version: `tensorstore>=0.1.60`,
`numpy`. **No `zarr`, no `dask`, no `pydantic`, no `funlib`.** The package is float-only; a
consumer that works in integer Rois builds them in its own subclass (section 6). Python >=
3.10 with no 3.11-only syntax, no upper bounds on numpy or tensorstore.

The package never deletes, truncates or overwrites: there is no `overwrite`, `delete` or
`mode="w"` anywhere in its API (Behaviour 11). What it writes is OME-NGFF zarr, v2 with OME
0.4 or v3 with OME 0.5, and nothing else (Convention 6).

Not in scope, by decision, each with its consumer named so it is not forgotten:

| Out of scope | Who asked | Where it lives instead |
|---|---|---|
| Deleting or overwriting arrays and groups, following symlinks, "rmtree before rerun" | every writer repo, by consensus on 2026-10-22 | each repo's own policy: cellmap-analyze's `delete_tmp_dir_blockwise`, mesh-n-bone's `_safely_remove_pyramid`, cellmap-flow's never-delete rules |
| Downsampling and its reducers (mode, mean, median, min, max, majority vote), the pyramid compute | cellmap-analyze #90, mesh-n-bone, registration-station, cellmap-flow | each repo, per block, through `write_index`; `add_level` writes a level's metadata and empty array only |
| Block scheduling: dask, daisy, LSF, OOM retry, memory sizing | every blockwise repo | each repo; 0.3's iterator yields boxes and runs nothing |
| Writing N5 or precomputed; serving-policy documents (`.zarray` codecs, `info`); writing exactly the bytes served | chunkmirage | chunkmirage's frontends and `materialize`, which call the builder and `chunk_key` for the placement half |
| Arbitrary affines, BigStitcher XML, OME 0.6 coordinate systems and RFC-5 transforms | liconn-err, chunkmirage | liconn-err; chunkmirage's `ngff.py` and `scene://`. The reader takes an OME 0.6 group's intrinsic system (Convention 4) and no more |
| Oblique-plane sampling, chunk-grouped gather | CineMap | CineMap, on top of `.ts` and `chunks_covering` |
| CF packing (`scale_factor`, `add_offset`, `_FillValue`), xarray coordinate arrays | chunkmirage | chunkmirage, from `ArrayMeta.attrs`, `group_attrs` and `parent`, which the package exposes raw |
| HDF5, GeoTIFF | chunkmirage | chunkmirage |
| Normalisation chains, model-facing defaults | cellmap-flow | a `cellmap_flow` subclass of `WorldArray` |
| The integer-scaled frame | cellmap-analyze | dropped, not moved: its commands work in voxel indices of their own output grid and read inputs with `read(box, voxel_size=, anchor=)` (its session's decision, 2026-10-22); mesh-n-bone and cellmap-flow keep their own funlib bridges |
| Legacy write documents (funlib `voxel_size`/`offset`, COSEM `transform`) as package output | cellmap-analyze, whose outputs dacapo may read through funlib | the `attrs` passthrough of `create`; Open question 5 |
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
vol.source_voxel_size, vol.source_translation, vol.resampled
                   # the stored level's voxel size and translation (0.4), also under resample and relabel; whether
                   #   reads resample to a target
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
                    #   choosing or placing a level opens nothing, whether the pyramid was opened by its
                    #   group path or by one of its level paths; else by opening each level

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
the array's roi; with a resample target, the target voxels `shape` counts, so `read(vol.box)`
returns `shape` and equals `Grid.from_meta(meta, voxel_size=t).box`. So `vol.read(vol.box.intersect(roi).snap(vol.voxel_size, vol.translation))`
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
the levels of a pyramid and repeat opens share one cache pool; the default budget is 0. Its
decoding and file I/O pools are sized by `thread_budget()` (0.5, Behaviour 7).
`capped_context` is mesh-n-bone's worker-process context (every tensorstore concurrency
resource limited, cache off); it returns a **new** context per call, as those workers
inline the spec into every open today, and `capped_context_spec` exports the dict for
callers building their own opens, writers included. `Retry` sizes the per-attempt timeout
from the read when none is given (Behaviour 6).

### 0.2: metadata and placed creation

Module `worldarray.write`, re-exported at the top level. Nothing here is imported by the
read side, so a read-only consumer sees no change.

```python
doc = wa.ome_multiscales(
    axes,                 # axis names in the array's C order: ("z","y","x"), ("c","z","y","x"), ("t","c","z","y","x")
        units,                # per axis ("nanometer"; "" on a channel axis) or by kind {"space": "nanometer", "time": "second"};
                          #   written in OME's names ("nm" -> "nanometer", wa.units.ome_name), an unknown unit as given
    levels,               # [("s0", voxel_size, translation), ...]: one explicit placement per level over the spatial axes
                          #   (or every axis). Never derived here: levels in the wild carry their own translations and
                          #   non-whole factors (chunkmirage's pyramids)
    version="0.5",        # "0.5" (zarr v3: attributes["ome"]) | "0.4" (zarr v2: attributes["multiscales"])
    axis_types=None,      # per axis "space" | "time" | "channel"; default from the name (z y x space, t time, c channel)
        name="", metadata=None, type=None,   # the multiscale's own fields; `type` (the downsampling method) written when given
) -> dict                 # the complete group attributes to write; `wa.metadata` reads them back exactly (Behaviour 10)

wa.level_placement(voxel_size, translation, factors, origin=0) -> (voxel_size, translation)
                          # the level whose voxel j covers source voxels origin + j f .. origin + (j + 1) f - 1 on each axis:
                          #   (voxel_size * f, translation + voxel_size * (origin + (f - 1) / 2)). Convention 1's BigDataViewer
                          #   rule inverted, and the centre Behaviour 4's block mean produces
wa.level_shape(shape, factors, origin=0) -> shape    # ceil((n - origin) / f) per axis, the quotient snapped to a whole
                                                     #   number within 1e-6 first (0.3 / 0.1 is a factor of 3)
wa.units.ome_name(unit) -> str   # OME's name for any spelling the unit table knows ("nanometer" for "nm"); unknown raises
wa.ensure_group_entry(group_path, name, voxel_size, translation, axes, units="nanometer", axis_types=None,
                      group_attrs=None, format=None)   # list an array the package did not create (a symlinked s0, a level
                          #   filled before it is listed): the group document's entry for `name` added or replaced, the
                          #   document and missing ancestors created, nothing written when the entry already says the same
                          #   (Behaviour 12). `create` and `add_level` call this
wa.write.codecs_of(path_or_node) -> (codecs, shards)   # an array's codecs in its format's dialect and its shard shape, for
                          #   a mirror created like it (cellmap-flow's sync)

vol = wa.create(
    path,                 # ".../vol.zarr/s0": the last component is the level, its parent the multiscale group, made if absent
    shape, dtype,
    voxel_size, translation,   # spatial placement in `units`, or one entry per axis
    axes=None, units="nanometer",   # as ome_multiscales; axes default to default_axes(ndim)
    chunks=...,           # required: the storage chunk (the inner chunk when sharded). The package never picks a chunk shape
    shards=None,          # zarr v3 only: the shard shape, a whole number of chunks per axis
    fill_value=0,
    codecs="zstd",        # "zstd" (level 3, which is what zarr-python's default `zstd(level=0)` means) | "none" | the format's
                          #   own JSON: a zarr v2 `compressor` dict or a zarr v3 `codecs` list
        format=None,          # "zarr3" | "zarr2"; default the nearest existing ancestor group's format, else "zarr3"; missing
                          #   ancestors up to the ".zarr" root get bare group documents (Behaviour 11).
                          #   OME 0.5 goes with v3 and 0.4 with v2, always
    attrs=None,           # extra array attributes, merged into the array's own
    group_attrs=None,     # extra group attributes, merged; no one else's key is dropped or rewritten (Behaviour 12)
    exists="error",       # "error": anything already at `path` raises | "adopt": an identical array (shape, dtype, chunks,
                          #   shards, placement within rtol 1e-6, and from 0.4 every key of `attrs` holding the value given)
                          #   is opened for writing; a different one raises naming each
                                                    #   difference. There is no third value. An adopt writes no group document when the entry
                          #   already says the same (Behaviour 12)
    store_fill=False,     # True stores write units equal to fill_value instead of leaving them absent
    separator=None,       # zarr v2 dimension_separator, zarr v3 chunk-key separator; the format's default when None
    axis_types=None,      # as ome_multiscales
    context=None, retries=None,
) -> WorldArray           # writable; every read-side property and method as before

vol = wa.open(path, mode="r+", store_fill=False, ...)   # an existing array for writing; vol.writable; the metadata
                                                                                                                #   rules of mode "r" unchanged; a handle pickles by its arguments
                                                        #   and its context's spec (Behaviour 11)
vol.write_index(begin, end, data, partial=False)   # whole write units only (Behaviour 13) unless partial
vol.write(box, data, partial=False)                # a world box by Behaviour 2, then write_index; no padding, no resampling
vol.write_shape                                    # the unit one writer may own: the shard shape when sharded, else the chunk shape
vol.chunks_covering(begin, end, write=False)       # write=True: on the write-unit grid
vol.chunk_key(index) -> str ; vol.parse_chunk_key(key) -> tuple[int, ...] | None   # storage keys of write units (Behaviour 15)
wa.chunk_key(index, format, separator="/", encoding="default", chunk_shape=None, shape=None, origin=None) -> str
wa.parse_chunk_key(key, format, ...) -> tuple[int, ...] | None    # the same from the array's facts alone, for an array that is
                                                   #   served from nowhere (chunkmirage) or not yet created

pyr = wa.open_pyramid(path, mode="r+")
pyr.add_level(name, factors=None, origin=0, voxel_size=None, translation=None, shape=None, dtype=None, chunks=None,
              shards=None, codecs=None, fill_value=None, attrs=None, exists="error", from_level=None) -> WorldArray
                          # the level's dataset entry and its empty array: placement by level_placement over `from_level`
                          #   (the finest listed when None) unless given explicitly, shape by level_shape, dtype, chunks,
                          #   shards, codecs and fill copied from that level unless given; no voxel data; one process at a time
```

### 0.3: grids and blocks

Module `worldarray.grid`, re-exported at the top level.

```python
g = wa.Grid(shape, voxel_size, translation, axes=None, units="nanometer", chunks=None, shards=None)
                          # a store-less grid with box, corner, index_box, chunks_covering, write_shape and blocks as on a
                          #   WorldArray, and meta(dtype) -> ArrayMeta; wa.create(path, grid=g, dtype=...) creates from it
g = wa.Grid.from_meta(meta, voxel_size=None)   # another array's grid, or Behaviour 4's target grid over it at `voxel_size`

for b in wa.blocks(shape, block_shape, context=0, box=None, clip=True):   # also vol.blocks(...) and g.blocks(...)
    b.index, b.id         # position on the block grid in C order, and its linear id
    b.begin, b.end        # the write box: these tile [0, shape), or `box`, with no overlap and no gap; the last one clipped
    b.read_begin, b.read_end   # the write box grown by `context` on each side, clipped to the array when `clip`
len(blocks) ; blocks[id] ; blocks.index_of(begin)   # count, by linear id, by a block's first voxel
vol.write_chunk(index, data)       # one whole write unit by its index; data's shape is that unit's (clipped) shape
wa.level_factors(voxel_size, ratio=1.5, max_levels=None, stop_shape=None, shape=None) -> list[tuple[int, ...]]
                          # per level, factor 2 on every axis within `ratio` of the finest voxel size and 1 elsewhere, compounding,
                          #   until max_levels or until a level fits stop_shape (mesh-n-bone's schedule, adopted by cellmap-analyze #90)
wa.store.join(path, *parts) ; wa.store.parent(path) ; wa.store.basename(path)   # URL-aware path helpers, public
wa.units.abbreviation(unit) -> str   # the OME short form, "nm" for any spelling of nanometre the table knows
```

### 0.4: reading onto a grid

```python
vol.read_onto(grid, begin=None, end=None, padding=None, method=None, mask=False, retries=None, transform=None,
              post_transform=None, select=None)
                          # voxels [begin, end) of a Grid's spatial axes (all of it when None), read from this array:
                          #   exactly that shape, by the centre rule at the array's voxel size, else resampled onto the
                          #   grid anchored at its corner (Behaviour 20)
vol.write_index(begin, end, 0) ; vol.write(box, 0) ; vol.write_chunk(index, 0)
                          # a scalar fills the box, converted to the dtype only when its value survives (Behaviour 13)
vol.source_translation    # the stored level's translation in `units`, beside source_voxel_size, under resample and relabel
wa.create(..., attrs={...}, exists="adopt")   # an adopt also compares the keys of `attrs` (Behaviour 11)
```

### 0.5: writes that cannot hang

```python
vol.write_index(begin, end, data, partial=False, retries=None)
vol.write(box, data, partial=False, retries=None) ; vol.write_chunk(index, data, retries=None)
                          # timed out and retried as reads are: `retries`, else the handle's (Behaviour 6)
wa.thread_budget() -> int # threads for tensorstore's decoding and file I/O in this process: OMP_NUM_THREADS, else
                          #   LSB_DJOB_NUMPROC, else the CPUs it may run on (Behaviour 7)
                          # every wait on a data read, write or open kicks its context's pools after each second
                          #   it is still pending (Behaviour 6)
```

## 3. Conventions

These are the rules the package exists to hold in one place. They match chunkmirage's
`docs/concepts/formats.md`, "Stored sources", which its TypeScript twin follows too; where
this document and that one differ, fix one of them.

1. **A translation is where voxel 0's centre is**, as in OME-NGFF, COSEM's `transform` and
   CF coordinates. Index `i` on an axis sits at `translation + i * voxel_size`; voxel `i`
   covers `[translation + (i - 1/2) * voxel_size, translation + (i + 1/2) * voxel_size)`.
   Formats that store voxel 0's **corner** are moved half a voxel along the spatial axes
   on reading. An N5 level with `downsamplingFactors` and no offset sits as Neuroglancer
   draws it, a pure scale of the base grid with every level's corner at the origin: level-L
   voxel `i` covers full-resolution voxels `f*i .. f*i+f-1`, so its centre is `f*i + f/2`
   full-resolution voxels from the origin and its translation `f / 2 * base`. BigDataViewer
   centres voxel 0 on the origin instead, half a base voxel lower at every level; the shift
   between levels, `(f - 1) / 2` base voxels, is the same in both and is the imglib2 rule
   liconn-err and registration-station use. (Decided 2026-10-22, when mesh-n-bone found
   that `pixelResolution` and `resolution` pyramids disagreed; 0.1.0 followed BigDataViewer
   for `pixelResolution`.) The shift
   applies only along axes named `z`, `y` or `x`, or the last three when axes are unnamed;
   never along channel or time.

   | Stored | Means | Read as `translation` |
   |---|---|---|
   | OME-NGFF `translation` (zarr v2/v3) | centre of voxel 0 | as is |
   | COSEM `transform.translate` (N5 or zarr) | centre (fibsem-tools wrote it as the first sample's position; Neuroglancer ignores `transform` and `offset` on N5 entirely) | as is |
   | funlib `offset` with `resolution`/`voxel_size` (N5 or zarr; `offset` missing = 0) | corner | `offset + voxel_size / 2` |
   | N5 `pixelResolution` or `resolution` x `downsamplingFactors f`, no offset | corner at the origin at every level, as Neuroglancer draws N5 | `f / 2 * pixelResolution` |
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
   `mm`, `cm`, `m`, `km`, `angstrom`/`Å`, `pm`; for time, `s`/`second`, `ms`, `us`/`µs`, `ns`,
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
     (COSEM), C order; else the same funlib keys on the parent group when the array carries
     none (mesh-n-bone's inputs); else tensorstore's `dimension_units` attribute (zarr v3). Axes from
     OME `axes`, `dimension_names` (zarr v3), `_ARRAY_DIMENSIONS` (xarray) or `axis_names`.
   - N5: the array's `transform` (COSEM; C order unless `ordering: "F"`); else the parent's
     `multiscales[].datasets[].transform` for this path; else funlib's `voxel_size` and
     `offset` (array, else group), which zarr-python's N5 store leaves in the array's C
     order, read by the zarr funlib rule above; else `pixelResolution` (both the
     `{dimensions, unit}` form and a bare list) or `resolution`, from the array else the
     group, times the level's `downsamplingFactors`; plus `offset`; axes from the `axes`
     attribute. These lists are x-first. Without an `offset` the corner is at the origin at
     every level whichever key gave the base (BigStitcher writes `resolution` and
     `downsamplingFactors`; funlib writes `resolution` alone): translation `f / 2 * base`,
     half a voxel of the level (Convention 1).
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
   metadata server otherwise stalls every open for seconds; "off Google Cloud" is decided
   with no network call, from the machine's DMI product name and the `GCE_METADATA_HOST` and
   `GCE_METADATA_ROOT` variables, and on a Google machine the variable is left alone so the
   credentials the real metadata server issues keep working (`store.GUARD_GCE_METADATA`
   turns the guard off). A `precomputed://` prefix is accepted and
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

6. **Writing is reading inverted, in one format.** The package writes OME-NGFF zarr only:
   zarr v2 with OME 0.4 (`.zgroup`, and `.zattrs` holding `multiscales`, each entry with
   `"version": "0.4"`) or zarr v3 with OME 0.5 (`zarr.json` with `attributes.ome.version`
   `"0.5"` and `attributes.ome.multiscales`). Never OME 0.4 inside zarr v3, which two repos
   write today and both Neuroglancer (`ome.ts` line 731 falls back to `attrs.multiscales`)
   and this reader tolerate; never N5 or precomputed, because Convention 4's N5 precedence
   has no unambiguous inverse and chunkmirage must write exactly the documents it serves.
   A written `translation` is voxel 0's centre, written always, zeros included; `scale` is
   the voxel size; every number is written as the float it was given, never rounded. Axes
   get `type` from their name unless told otherwise and `unit` on space and time axes only;
   a zarr v3 array's `dimension_names` are the axis names. No legacy document is written
   unless the caller passes one through `attrs`. Reading back what was written returns the
   same `voxel_size`, `translation`, `axes` and `units` with `placed_by == "ome"`: that round
   trip is the test every writer repo lacked.

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
   `finer` (as `floor`, but an error instead of the fallback to the first level, for
   callers that would rather fail than read coarser data than asked), `exact` (that level or
   `ValueError` listing the levels), `nearest` (closest on a log scale summed over
   axes, finer on a tie), `resample` (the level to resample from). Voxel sizes compare as
   floats with `rtol 1e-6`, so 10.48 is not 10. A group path without `level` or
   `voxel_size` raises with the level list; a level name that does not exist raises with
   the list; an array path, or a precomputed path naming a scale, is that array or scale
   under `level=0` or `"first"` silently and warns and ignores any other `level`. When no level has the
   requested `voxel_size`, `on_mismatch` decides: `"error"` raises with the list;
   `"resample"` reads the level `level_mode` picks and resamples (Behaviour 4); `"relabel"`
   reads that level voxel for voxel and reports `voxel_size` as the requested size with
   **voxel `i` staying voxel `i`**: every world quantity, the translation and the corner
   included, scales by the ratio of the sizes (cellmap-flow's rule, which chunkmirage's
   server relies on), so a box of `m` requested voxels returns `m` stored voxels of the level,
   and `source_voxel_size` still reports the level's own size; `"keep"` reads the chosen level at its own voxel size (a level pick, nothing
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

6. **Retries** apply to data reads and data writes, with mesh-n-bone's formula, which its
   tests already pin: attempt `k` (from 1) waits `timeout * k` seconds on `.result()`, where
   `timeout` is the given value or, when `None`, `clamp(MiB / 16, 5, 120)` for a remote
   store (`gs://`, `s3://`, `http(s)://`) and `clamp(MiB / 256, 5, 30)` for a local one
   (`file`, `memory`), `MiB` being the size of the in-bounds part of the read after
   clipping, or of the box written; between attempts the sleep is `base_delay * 1.3 ** (k - 1)` plus `U(0, 1)`
   seconds; at most `max_attempts` (default 10). Retried: `TimeoutError`, and a tensorstore
   error whose message starts with `UNAVAILABLE`, `DEADLINE_EXCEEDED` or
   `RESOURCE_EXHAUSTED` or carries an HTTP 5xx or 429. Any other error is raised at once. A
   timed-out tensorstore future is cancelled before the next attempt starts; a write may
   still land after its cancel, with the same bytes the next attempt writes. Each retry is
   logged as a warning naming the error's first line, or its type when the message is
   empty (tensorstore's timeout is `TimeoutError()`); when every attempt times out the
   `TimeoutError` names the operation, the path and the attempts. Metadata documents are not
   retried this way (Convention 5). The policy is a per-handle attribute and a per-read and
   per-write argument, never only constructor state.

   The kick (0.5): tensorstore up to at least 0.1.85 can strand a queued task. A pool
   worker idle for 20 ms exits, and a task queued as it exits gets no thread while the pool
   counts itself full (fixed upstream in google/tensorstore ed51718085, in no release yet).
   The future then waits forever with no tensorstore thread alive, on a read, a write or an
   open. A pool limited to one thread meets it most; cellmap-analyze 0.6 hung on /nrs with
   `data_copy` and `file_io` at 1, and one more task on the same context woke the pool
   within seconds (salamander's and c-elegans' live tests, 2026-10-22, with un-kicked
   workers staying stuck). So every wait on a data read, a data write or a store open
   gives the context's decoding and file I/O pools one no-op task each (an in-memory array
   copy and a local read of a key that does not exist; no store is touched) after every
   second the future is still pending. A stranded operation finishes within about a
   second, at any concurrency, and the retry stays the outer net. Open question 14 says
   when the kick goes.

7. **Contexts.** Without `context`, every open in the process shares `shared_context(0)`:
   tensorstore's cache is off and the levels of one pyramid share one context. Its
   `data_copy_concurrency` and `file_io_concurrency` limits are `thread_budget()`: the
   first of `OMP_NUM_THREADS` (dask sets 1 in every worker it spawns; a cluster job script
   sets its slots) and `LSB_DJOB_NUMPROC` (an LSF job's slots) that holds a positive
   integer, else the CPUs `sched_getaffinity` allows: the cap every other threaded library
   in the job already takes, as Janelia's cluster guidance asks. tensorstore alone sizes
   both pools to every CPU of the node, whatever the job's slots, so each of twelve dask
   workers in a twelve-slot job on a 96-core node could start 96 threads (0.5). A dask
   worker therefore runs tensorstore on one thread of each pool, which Behaviour 6's kick
   makes safe.
   `capped_context` sets `data_copy_concurrency`, `file_io_concurrency`,
   `gcs_request_concurrency`, `s3_request_concurrency` and `http_request_concurrency` to
   `concurrency` and the cache to `cache_bytes`, and returns a new context each call, so two
   handles never share one capped pool unless the caller passes the same object twice; a
   `concurrency` of 1 is safe under Behaviour 6's kick. Both are plain tensorstore contexts; a caller may pass its own, at construction or through
   `reopen(context=)` when a long-lived worker changes its concurrency (cellmap-analyze
   reopens tensorstore on every read today for that reason).

8. **Thread safety.** A `WorldArray` may be read from several threads at once; nothing in the
   package holds a lock around a read. No dask anywhere.

9. **Non-spatial axes.** `read_index` works on all axes. `read` takes a box over the spatial
   axes and keeps leading non-spatial axes whole unless `select` pinned them, or a box over
   every axis, in which case Behaviour 2's centre rule applies along each axis in that
   axis's converted unit (a time sample is a point too, so a window `[t0, t1)` in seconds
   reads the samples whose times lie in it). The returned array's axes are `vol.axes`.

Behaviours 10 to 16 are 0.2; 17 to 19 are 0.3.

10. **Documents round-trip.** `ome_multiscales(axes, units, levels, version)` returns the
    complete group attributes. Written beside arrays and read back through `wa.metadata`,
    any level reports the voxel size, translation, axes and units it was given, exactly,
    for OME 0.4 and 0.5, for `(z, y, x)`, `(c, z, y, x)` and `(t, c, z, y, x)`, and for
    levels with non-whole factors and their own translations (chunkmirage's case). The
    documents equal the expected documents in `tests/fixtures/write.json` after JSON
        normalisation, so chunkmirage's TypeScript builder can be held to the same output; each
    case also carries its resolved per-axis units and types, and the call with those gives
    the same document, so the TypeScript side need not reimplement the broadcasting. Units
    are written in OME's names: any spelling the unit table knows (`"nm"`, `"µm"`,
    `"nanometre"`) becomes `"nanometer"`, `"micrometer"` and so on (`wa.units.ome_name`),
    and a unit the table does not know is written as given, Convention 3's welcome kept.
    The multiscale's `name`, `metadata` and `type` are written when given. `level_shape`
    snaps the quotient `(n - origin) / f` to a whole number within `1e-6` before the
    ceiling, so a factor that is a ratio of float voxel sizes (`0.3 / 0.1 =
    2.9999999999999996`, chunkmirage's case) gives 3 voxels for 9, not 4.
    `level_placement` and `level_shape` invert Convention 1's BigDataViewer rule and agree
    with Behaviour 4: the level a block mean by factor `f` from source voxel `origin`
    produces has voxel 0 centred at `translation + voxel_size * (origin + (f - 1) / 2)`,
    and a `read(voxel_size=f * s, method="linear")` of the source anchored at that level's
    corner returns the level's values. With `origin > 0` the translation moves by `origin`
    source voxels: the mesh-n-bone bug that sized levels from an aligned ROI origin but
    placed them from voxel 0 cannot be expressed through this API.

11. **`create` makes a placed level, and nothing is ever destroyed.** `create` writes, in
    this order, the group document (created when absent; when present, read, the dataset
    entry for this level added or replaced with the datasets kept in order of voxel size,
    everything else kept as it was) and then the array with its chunks, shards, codecs,
        fill value, dimension names and `attrs`. The format follows the container: walking up
    from the level's parent, the nearest existing node decides, a `.zgroup` group meaning
    zarr v2 and OME 0.4, a `zarr.json` group zarr v3 and OME 0.5, an array, an N5 or a
    precomputed node a `ValueError`; a `format` that contradicts it raises rather than
    mixing versions. When no ancestor exists, the walk ends at the nearest ancestor named
    `*.zarr`, the store root, `format` or zarr v3 decides, and every missing group from
    that root down to the level's parent gets a document, bare above the parent, so
    zarr-python browses the hierarchy and a level made in a zarr v2 container is zarr v2
    (cellmap-analyze's `prepare_ds`, cellmap-flow's `box_n` groups); with no `.zarr`
    ancestor either, only the parent gets a document. `chunks` is required (alignment is the
    whole safety argument, so the package never picks a chunk shape); `shards` that are not
    a whole number of chunks on every axis raise. `codecs` defaults to zstd level 3 in the
    format's own dialect, what zarr-python's default `zstd(level=0)` means, so
    cellmap-analyze's, mesh-n-bone's and registration-station's outputs keep their
    compression; cellmap-flow's blosc stays blosc by passing it. There is no overwrite: with
    `exists="error"` any document already at `path` raises `FileExistsError`; with
    `exists="adopt"` an array whose shape, dtype, chunks, shards and placement match
    (placement within `rtol 1e-6`), and whose attributes hold every key of `attrs` with the
    value given as JSON stores it (keys not given are someone else's and not compared), is
    opened for writing and its dataset entry ensured, and one that differs raises
    `ValueError` listing every differing field and key with the value found and the value
    wanted (cellmap-flow's `_refuse_another_region`, registration-station's
    missing check, which let a shorter last band's chunk grid win a creation race). A
    creation that loses a race, the store reporting the array present after the probe said
    absent, is re-read with backoff, 5 attempts at `0.5 s * attempt`, and then treated as an
    adopt (registration-station's `shared_array`, written because metadata appears late on
    NRS). The package deletes, truncates and follows no symlink, ever; a caller that wants a
        fresh array removes the old one with its own code first. Write units whose values all
    equal `fill_value` are not stored unless `store_fill=True`, so chunk presence stays the
    signal cellmap-flow's trainer and manifest read; a stored unit written so loses its
    key, as zarr-python's `write_empty_chunks=False` does, the one key a write may remove
    being the one it replaces. `create` returns a writable `WorldArray`;
    `wa.open(path, mode="r+")` opens an existing one with every rule of Conventions 3 to 5
    unchanged, and a handle pickles by its arguments and its context's spec, so each worker
    process opens its own cheaply (a local path goes resolved against the working directory
    at open, so a worker that changes directory before unpickling, as cellmap-analyze's dask
    workers do, opens the same store; `reopen` uses it too), on the shared context when the handle had one, else on
    a context rebuilt once per process from the same spec, so a capped context stays
    capped (mesh-n-bone's per-process cache and thread caps).

12. **Group attributes belong to everyone.** The package rewrites only the `multiscales`
    list (0.4) or `ome.multiscales` (0.5), and within it only the dataset entry whose `path`
    is the level being created or added; other multiscales, other datasets and every other
    key of the group and of the array (`fs`, a registry record, `y0`/`y1`, anything) come
        back unchanged. The group's attributes before and after a `create` differ in that one
    entry, and not at all when the entry already says the same: adopting an identical array
    (placement within rtol 1e-6) reads the document, compares and writes nothing, so
    registration-station's twenty block jobs calling `create(exists="adopt")` on one output
    write the document in the first creation only. Rewriting a group document is a
    read-modify-write with no lock, so `create` and `add_level` of different levels in one
    group are called from one process at a time, the coordinator, never from block jobs;
    the docstrings say so. After a write the document is read again and, when this level's
    entry is missing because another process's write landed in between, merged and written
    again, up to 5 times, then `RuntimeError`: a lost update becomes a repair or an error
    where the store shows it, never a misplaced level. `ensure_group_entry` is this listing
    step on its own, public, for an array the package did not create (mesh-n-bone's
    symlinked `s0`) or one listed only after it is filled; it follows the same rules,
    ancestors included.

13. **Aligned writes need no coordination.** `write_index(begin, end, data)` requires
    `data.shape == end - begin`, `data.dtype == vol.dtype` (no silent cast; `ValueError`),
    or `data` a scalar (a number or a 0-d array) that fills the box: it is converted to the
    dtype only when its value survives, so an integer or bool array takes whole numbers in
    its range and a float array any number it does not overflow, rounded to its precision
    (`0` into `uint8` is written; `300`, `-1`, `0.5` and NaN raise), and a scalar equal to
    `fill_value` over whole units leaves them absent as an all-fill array does (cellmap-analyze
    writes a constant for every empty or masked-out block). The box must be a union of whole write units: every bound a multiple of
    `write_shape` on its axis or the array's end on that axis, so the clipped last unit is
    allowed; otherwise `ValueError` names the axis, the bound and the unit size. A unit
    written whole is one put, so writers in different processes writing different units
    need no lock and no ordering, which is what every writer repo does today by
    construction. `partial=True` lifts the check and does tensorstore's read-modify-write
    of the touched units, documented as unsafe while anything else writes the same unit
    (cellmap-flow's paint slabs, which it serialises itself). A write commits before the
    call returns, timed out and retried by Behaviour 6 (0.5; before, it was not, and a
    stranded write waited forever). `write(box,
    data)` maps a world box by Behaviour 2 and then calls `write_index`; a box reaching
    outside the array raises (nothing is padded on write) and `voxel_size` is not an
    argument (nothing is resampled on write).

14. **`add_level` is metadata plus an empty array.** On a pyramid opened for writing,
    `add_level(name, factors=, origin=)` creates `group/name` through `create` with
    placement from `level_placement` over `from_level` (the finest listed when None) and
    shape from `level_shape`, copying dtype, chunks, shards, codecs and fill value from that
    level unless given, or with an explicit `voxel_size` and `translation`. The dataset
    entry is inserted so the list stays ordered finest to coarsest and an entry of the same
    name is replaced; the group's other content is untouched (Behaviour 12). It writes no
    voxel data: downsampling is the caller's, per block, through `write_index`, labels by
    mode and images by mean as each repo decides. `exists` means what it means for `create`.

15. **Chunk keys.** `chunk_key(index)` is the key, relative to the array, of the write unit
    at `index`: zarr v2 indices joined by the array's `dimension_separator` (`0.1.2`); zarr
    v3 by the stored `chunk_key_encoding`, `default` giving `c/0/1/2` with its separator and
    `v2` giving `0.1.2`; N5 `2/1/0`, x first; precomputed `x0-x1_y0-y1_z0-z1` in the scale's
    voxel coordinates with `voxel_offset` included and the last range clipped to the shape;
    a sharded zarr v3 array keys its shards, so `index` is on the shard grid.
    `parse_chunk_key` inverts each and returns None for a key that is not a chunk
    (`zarr.json`, `.zarray`, `attributes.json`, `info`, a `c/` key of the wrong arity).
    `list_chunk_keys` and `list_chunks` use the same parser. The free functions take the
    format and the array's facts instead of an open array, for an array served from nowhere
    (chunkmirage's frontends) or about to be created.

16. **Write units.** `write_shape` is the shard shape of a sharded zarr v3 array, else the
    chunk shape; `ArrayMeta.shard_shape` is None when unsharded; `chunk_shape` stays the
    inner read chunk (Convention 2). `chunks_covering(begin, end, write=True)`,
    `list_chunks` and `chunk_key` run on the write grid. A planner that sizes jobs from
    `write_shape` (registration-station's y-bands, cellmap-analyze's blocks) produces only
    aligned writes.

17. **Blocks tile, context grows, nothing runs.** `blocks(shape, block_shape, context, box,
    clip)` yields, in C order, blocks whose write boxes `[begin, end)` partition `[0,
    shape)`, or `box` (a voxel box, the grid anchored at its begin) when given, with no
    overlap and no gap, the last block on an axis clipped to the box. `box` may reach past
    the array on any side (a ROI rounded outward to a factor multiple, mesh-n-bone's halo
    mode): the write boxes tile it exactly, outside the array too. A block's read box is
    its write box grown by `context` on each side, where `context` is one int, one per
    axis, or a `(before, after)` pair per axis, clipped to the **array** (never to `box`,
    never a write box) when `clip=True` (the default; cellmap-analyze's `create_block` edge
    rule) and left unclipped when `clip=False` for a padded `read_index` (its
    `neg_with_edge_pos` mode, which reads past the array for fill). A block also carries
    `at_start` and `at_end` per axis, whether it is the first or last block on that axis,
    so a caller adds context at the array's far faces itself, and `inner`, the slices of
    the write box inside the read box, correct also when `clip=False` lets `read_begin`
    go below 0. `len(blocks)`, `blocks[b.id]` (O(1)), `blocks.index_of(b.begin)` and
    `b.index` agree. `vol.blocks()` defaults `block_shape` to `write_shape`, so default
    blocks are aligned writes. Nothing is scheduled: the iterable and its blocks are plain,
    picklable records, and the iterable pickles without its blocks, so a worker takes an
    id. One grid per call: a consumer that writes several levels from one block derives
    the per-level boxes itself (mesh-n-bone's direct mode, two lines). `write_chunk(index,
    data)` writes the unit at `index` whole; `data.shape` must equal that unit's clipped
    shape.

18. **A `Grid` is a `WorldArray` without a store.** `Grid(shape, voxel_size, translation,
    axes, units, chunks, shards)` answers `box`, `corner`, `index_box`, `chunks_covering`,
    `write_shape` and `blocks` exactly as a `WorldArray` of those facts does, and
    `g.meta(dtype)` agrees with `wa.create(path, grid=g, dtype=dtype).meta` in every
    field a grid has: shape, dtype, chunk and shard shape, axes, units, voxel size and
    translation (a grid has no path, format, documents or fill value), so a pipeline plans
    its output before anything exists. `Grid.from_meta(meta, voxel_size=t)` is Behaviour
    4's target grid over that array: anchored at its corner with `ceil(n * s / t)` voxels,
    so a `read(voxel_size=t)` of the source fills it exactly, and its `box` is the target
    grid's extent, which may reach past the source's own `box` by less than one target
    voxel; this is the output grid cellmap-analyze's commands compute today in the integer
    frame.

19. **Schedules and helpers.** `level_factors(voxel_size, ratio, max_levels, stop_shape,
    shape)` returns per level the per-axis factors, 2 on every axis whose current voxel size
    is within `ratio` of the smallest and 1 elsewhere, compounding level over level until
    `max_levels` or until the level's shape (from `shape`) fits within `stop_shape`;
    isotropic data gives all 2s at every level. `store.join`, `parent` and `basename` are
    the URL-aware path helpers Convention 5 already uses, made public. `units.abbreviation`
    returns the OME short form for any spelling the table knows and raises for one it does
    not.

20. **Reading onto a grid.** `vol.read_onto(grid, begin, end)` returns voxels `[begin, end)`
    of a `Grid`'s spatial axes, all of it when `begin` and `end` are None, read from this
    array, with exactly `end - begin` voxels on those axes; the array's other axes are read
    as `read` reads them (whole, or pinned by `select`), the open-time `padding`, `method`,
    `retries` and `transform` apply, and `mask` marks the grid voxels that held data. It is
    `read(Box.from_index(begin, end, grid voxel size, grid translation), voxel_size=grid
    voxel size, anchor=grid corner)`, so Behaviours 2 and 4 decide every voxel: at the
    array's own voxel size (within level selection's rtol 1e-6) nothing is resampled,
    whatever the offset between the grids, and each grid voxel takes the stored voxel whose
    centre lies in it, never an interpolation, with `method="linear"` too; at any other
    voxel size the array is resampled onto the grid, anchored at its corner. Adjoining
    `[begin, end)` reads therefore agree with one read across both. The grid's spatial axes
    must be named as the array's are (`ValueError` otherwise), and its units are converted
    to the array's `units`. This is cellmap-analyze's per-operation grid: blocks come from
    one `Grid`, every input is read onto it, and outputs are created on it.

## 5. Tests (behavioural; synthetic stores in `tmp_path`; one optional network test)

Conformance, one tiny store each, checking `voxel_size`, `translation`, `axes`,
`axes_named`, `units`, `shape` and, where stated, `dtype`, the raw attributes and a
read-back. Stores and expected values live in `tests/fixtures/metadata.json` so
chunkmirage's TypeScript side can build and run the same cases; a case whose values
chunkmirage derives from raw attributes carries a `layered` note beside the package's
`expect`:

- chunkmirage's cases: N5 transposed to C order with `resolution` and `offset`; N5 COSEM
  `transform` C order; N5 group `pixelResolution` and `downsamplingFactors`
  (the corner at the origin at every level, as Neuroglancer); N5 group `multiscales` transform matched by path; OME levels matched by
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
`s0`, integer and `"first"` levels, `detect(missing_ok=True)`, `index_box`; funlib keys on
the parent group of an attribute-less zarr and N5 array, and funlib `voxel_size`/`offset` on
an N5 array in C order; a precomputed path naming a scale kept under `level="first"`; the
GCE guard left alone on a Google machine; a pyramid opened through one of its level paths
placing every level from the group document while one level's array is missing; `read_index`
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
world box, within `0.1 * voxel_size`; and an N5 level 1 with `downsamplingFactors` sits where the
equivalent OME level 1 does, both pyramids' s0 corner at the origin.

Network (skipped without it; not yet written): one public OpenOrganelle volume with a
non-integer z spacing (4.56 or 5.24 nm), read anonymously from s3: the package reports the
float voxel size at every level where cellmap-flow's `Coordinate` wrapping reports 5 and
10, and one corner-aligned box returns the same bytes as cellmap-flow's reader.

Writing (0.2): Behaviour 10's round trip for both OME versions and the three axis sets,
against `tests/fixtures/write.json`, which the TypeScript side also checks; `level_placement`
against the reader's BigDataViewer case and against a block-mean `read` anchored at the
level's corner, with origin 0 and 16; `create` following a v2 and a v3 group, refusing a
contradicting `format`, requiring `chunks`, rejecting shards that are not whole chunks,
raising on an existing path, adopting an identical array, listing every difference of a
different one, keeping foreign group and array attributes byte for byte, storing no
fill-only unit; a lost creation race simulated by a store that reports the array present
after the probe; `write_index` accepting whole units and the clipped end, naming the axis
of a misaligned bound, refusing a dtype cast, `partial=True` writing an arbitrary box; two
processes writing disjoint units of one array (multiprocessing) produce the expected
array; `write(box)` by the centre rule, raising outside the array; `add_level` placing and
sizing from factors and origin, inserting in order, replacing a same-named entry, copying
dtype, chunks and codecs, writing no data; `chunk_key` and `parse_chunk_key` round trips
for zarr v2 with `.` and `/`, zarr v3 `default` with both separators and `v2`, N5,
precomputed with a `voxel_offset` and a clipped last chunk, and sharded zarr v3, with
`list_chunk_keys` agreeing; `write_shape` on sharded and unsharded arrays; units written in
OME's names and an unknown unit as given, `type` when given, `level_shape` on a float-noise
factor; missing ancestor documents down from the `.zarr` root and none above it, the nearest
ancestor's format followed and a contradiction refused, an array in the way refused; an
adopt writing no document and a changed placement writing one; a lost entry written again
and the error after the attempts; `ensure_group_entry` on a foreign array, idempotent; a
pickled handle keeping a capped context's spec and sharing the shared one, and a relative
path reopening after a `chdir`; `codecs_of` by path; an all-fill write removing its own key;
a scalar filling a box through `write_index`, `write` and `write_chunk`, refused when it
does not fit the dtype. 0.5: a read and a write whose first attempts never finish (a future
nothing resolves) cancelled and retried, the warning naming `TimeoutError`, and the error
after the last attempt; a pending future kicked until it resolves, a wait's timeout, and a
kick landing on a capped context's own pools; `thread_budget` from each variable, a bad
value skipped, and the shared context's limits.

Grids (0.3): `blocks` partitioning with and without a box, `context` as an int, per axis
and as before/after pairs, clipping on and off, `len`, `blocks[id]` and `index_of`;
`write_chunk` of a middle and of a clipped unit; a `Grid` agreeing with the created array's
`meta`, and `Grid.from_meta` agreeing with Behaviour 4's target shape; `level_factors`
reproducing mesh-n-bone's anisotropic table (8, 8, 30 nm: z stays at 1 until y and x
reach 32) and all 2s for isotropic data; `abbreviation` for every spelling in the table.

Reading onto a grid (0.4): `read_onto` at the array's voxel size returning stored values
under a sub-voxel offset with `method="linear"`, and the next voxel past a half-voxel one;
onto a finer grid equal to the `read` it is defined as, in micrometres too, adjoining
reads agreeing, the mask marking grid voxels before the data; its own grid at voxel sizes
3.54, 4 and 5.3 returning the stored array exactly; a grid with other axis names refused.
`source_translation` under relabel and resample and in micrometres; an adopt keeping keys
it was not given and refusing a different or missing one, naming both values.

Target: about 600 lines of test cases for 0.1 and about 300 more for 0.2 and 0.3, the
store builders in `conftest.py` aside; no wording pins; no `/nrs`; no cluster.

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
race-safe region writes from many jobs; add a level to a group; encode a chunk key. That
became 0.2, below. 0.1.0 was published on PyPI on 2026-10-21; cellmap-flow merged its trial
into `consolidate` (PR #150), cellmap-analyze merged #92, chunkmirage merged #48, and
mesh-n-bone's PR #48 is open.

### Writers, 2026-10-22

What each writer repo's own code does today and what 0.2 replaces, from the five sessions'
answers to one questionnaire and the census below. Line counts are from the census.

| Writer today | Moves into the package | Lines | Stays in the repo |
|---|---|---|---|
| cellmap-analyze `zarr_io.prepare_ds`, `open_dataset`'s write role, `zarr_util.create_multiscale_dataset`, `generate_` and `write_multiscales_metadata`, `CellMapArray.__setitem__`, the metadata half of #90's `Multiscale`, `get_zarr_chunk_path_from_block_index`; 18 `output_idi.ds[roi] = data` sites | `create`, `open(mode="r+")`, `write`/`write_index`, `add_level(factors)`, `chunk_key`. Its outputs keep zstd and OME 0.4 in v2 containers, 0.5 in v3 | 336 (and 705 that 0.1 already covers, 189 of them dead) | block planning and dask, rmtree-before-rerun, the tmp dataset lifecycle, reducers and the factor choice; the integer frame is dropped |
| mesh-n-bone `_create_zarr_array` and its two dtype maps, `build_` and `write_multiscales_metadata`, `_ts_handle_for_output`, three clip-to-bounds writes, the metadata half of `prepare_pyramid_metadata_and_arrays`, `_existing_pyramid_scales` | `create`, `add_level(factors, origin)` (its ROI-origin bug becomes inexpressible), `open(mode="r+")`, `write_index`, `open_pyramid` | 438 (and 121) | the factor schedule and the super-chunk grids until 0.3, snap/halo alignment, reducers, dispatch, OOM retry, the s0 symlink and format matching, safe removal, the reuse policy. A v2 pyramid stays v2 because its s0 is a symlinked v2 array: `format` follows the group |
| cellmap-flow blockwise `_create_output`, `_refuse_another_region` and `process_fn`'s funlib write; `create_volume_zarr`'s array and multiscales; `io/ome.py`'s writer half; `sync_zarr_group_metadata`; `CHUNK_KEY_RE`; the fill, instance and carry writes | `create(exists="adopt")` naming what it found, `write_index`, the writable `.ts` for its unaligned read-modify-write, `chunk_key`, and `write_chunk` in 0.3; funlib.persistence leaves its blockwise path, zarr-python stays for s3fs until worldarray writes reach MinIO (they do: Convention 5's endpoints apply to writes) | 508 (and 603) | the volume root attributes, MinIO sync and ETag diffing, the never-delete rules, daisy, slab threading, the refusal wording; a channel axis is first-class in the builder |
| registration-station `shared_array`, `field_ome`, `write_volume`, `write_multiscale_pyramid`'s metadata, `apply_field_to_s0`'s check-then-create, eight OME literals, five band planners | `create(exists="adopt")` with the backoff it wrote, `ome_multiscales`, `add_level`, `write_index`, `write_shape` for the planner; reads through 0.1 as well | 571 (and 339 by adopting 0.1 at all) | the GPU warp, seam blending, the affine `.npy`, the 2x mean, Neuroglancer links; its `fs` and `fix_origin_um` sidecars become redundant (Open question 7) |
| chunkmirage `frontends/_ome.py`, `ArrayInfo.rescaled`, `scene._explicit_grids`, `stitching.grids` and `level_to_base`, four `chunk_key`s and five chunk regexes, `materialize`'s create and metadata-first write | `ome_multiscales`, `level_placement`, `chunk_key` and `parse_chunk_key`, `create` for `materialize`'s target; the half-voxel rule, written at 15 sites in 12 of its functions, collapses to calls | 337 (and 382, 179 of them partial matches) | serving-policy documents, codecs, N5 and precomputed generation, exact-bytes writes, the OME 0.6 graph, GeoTIFF, HDF5 |

### Census, 2026-10-22

Every function of I/O-type code in the five repos, classified against one rubric on the
trees as they stood after the 0.1 merges (cellmap-analyze main after #92, mesh-n-bone PR
#48, cellmap-flow main after #150, chunkmirage main after #48, registration-station
unadopted). Raw lines, docstrings included; treat as plus or minus ten percent.

| Lines | cellmap-analyze | mesh-n-bone | cellmap-flow | registration-station | chunkmirage | total |
|---|---|---|---|---|---|---|
| already delegates to worldarray | 154 | 237 | 712 | 0 | 71 | 1,174 |
| duplicates worldarray 0.1 | 705 | 121 | 603 | 339 | 382 | 2,150 |
| goes with 0.2 | 336 | 438 | 508 | 571 | 337 | 2,190 |
| needs a further feature | 170 | 662 | 28 | 17 | 683 | 1,560 |
| rightly local | 481 | 541 | 1,623 | 247 | 780 | 3,672 |

0.1 adoption had already removed about 3,400 lines from the four repos that took it. Of
the 1,560 "further" lines, about 400 live ones in four repos are the block grid with
context that 0.3's `blocks` replaces (cellmap-analyze 140, mesh-n-bone 240 plus 320
reachable only from tests, cellmap-flow 20, chunkmirage 9), about 150 are the small
helpers of Behaviour 19, and chunkmirage's 446 lines of OME 0.6, GeoTIFF and HDF5 are its
own domain and stay. Three repos keep 80 to 140 lines each of float-to-integer-Roi bridge
over funlib geometry, by this spec's own assignment.

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
5. **Legacy attributes on new outputs.** cellmap-analyze writes funlib `voxel_size` and
   `offset` (holding the centre) on every output and nothing in the five repos reads them;
   dacapo's checkout has nine `open_ds` users, and cellmap-data and fibsem-tools were not
   checked. Does any live consumer of cellmap-analyze or cellmap-flow outputs read funlib
   attributes? If so, should the package offer `funlib_attrs(voxel_size, translation, axes,
   units)`, writing `offset` as the corner, funlib's meaning, for `create(attrs=)`, or do
   the repos keep writing their own?
6. **The block iterator's shape** (Behaviour 17). Resolved 2026-10-22: both sessions
   confirmed it. mesh-n-bone asked that `box` may reach past the array and that `clip`
   clips read boxes only, and derives its per-level boxes from one grid itself;
   cellmap-analyze asked for `at_start`/`at_end` per axis and `inner`, and confirmed that
   `(before, after)` pairs cover its context modes and that `clip=False` with a padded
   `read_index` is its edge-reading mode. All are in Behaviour 17.
7. **Sidecar attributes.** Resolved 2026-10-22: registration-station keeps `fs` and
   `fix_origin_um` on the group through `group_attrs` and the part attributes (`y0`, `y1`,
   `y0_core`, `y1_core`, `feather_vox`) on the array through `attrs`, all known at creation;
   merge-on-create carries them and no attribute-update call is wanted.
8. **Creation on NRS.** The adopt-with-backoff numbers in Behaviour 11 (5 attempts, 0.5 s
   times the attempt) are registration-station's and are tested here only against a fake
   store. Validate from two LSF jobs on the cluster before anyone relies on them.
9. **The codec default.** Resolved 2026-10-22: cellmap-flow pins blosc explicitly (lz4
   level 5 for blockwise outputs, zstd level 3 for annotation volumes) as zarr v2 compressor
   dicts; cellmap-analyze, mesh-n-bone and registration-station take the zstd default, which
   cellmap-analyze checked byte for byte against zarr-python's. No shortcut name is wanted.
10. **Sharding.** Nobody writes shards today. registration-station wants them because a
   2.4 GB chunk makes Neuroglancer decompress a whole band for one slice; mesh-n-bone would
   take them as an opt-in for v3 sources; cellmap-analyze would size blocks to shards. 0.2
      offers `shards=` at creation and `write_shape` for planners and nothing more; whether
   that is enough is learned from the first repo that uses it. registration-station checked
   that a shard larger than the array is accepted and that band writes on shard edges,
   the clipped last band included, go through.
11. **Listing after filling.** cellmap-analyze's #90 creates a level's array, fills it, and
   lists it last, so a crashed job never leaves a half-filled level listed. `create` and
   `add_level` list first. `ensure_group_entry` is the listing step, but a level created
   unlisted has its placement nowhere until it is listed, so a `create(listed=False)`
   handle, and its pickles, would carry the placement from its arguments alone. Decide
   whether that is worth having, or whether an unlisted level should describe itself on
   the array (open question 5's funlib attributes, or an OME-shaped array attribute).
12. **Per-open object-store credentials.** cellmap-flow's MinIO writers open with a
   per-session key, secret and endpoint while the same process reads public S3
   anonymously. tensorstore 0.1.85's `aws_credentials` takes no inline key and secret, and
   its endpoint and profile settings are process-wide through the environment, so those
   three writers stay on zarr-python and s3fs. Revisit when tensorstore takes static
   credentials per open.
13. **World boxes on blocks.** cellmap-flow reads its input at one voxel size by a world
   box grown by a context in nanometres and writes at another; `blocks` (0.3) gives index
   boxes on one grid. `Grid.blocks` could carry world boxes too (`b.box`, `b.read_box` as
   `Box`, a context in world units). Two lines on cellmap-flow's side today. Not planned
   (2026-10-22): daisy does cellmap-flow's tiling and stays; reopen if cellmap-flow drops it.
14. **When the kick goes.** Behaviour 6's kick works around tensorstore's lost wakeup,
   fixed upstream in google/tensorstore ed51718085 (2026-09-30, "Fix lost wakeups in
   TaskGroup when workers block or exit") and in no release up to 0.1.85. When a release
   contains it (`gh api repos/google/tensorstore/compare/ed51718085...vX.Y.Z` reports
   "ahead" or "identical"), require that release, remove `context.kick` and `KICK_AFTER`
   (waits go back to `future.result(timeout)`), and keep the write retries. That release
   needs Python 3.11 or later, as tensorstore has since 0.1.81, so it also ends support for
   3.10.

## 8. Constraints for whoever builds on this

Branch off `main`, draft PRs, no "Test plan" section in PR bodies, never add Claude as git
author or committer, `/nrs/cellmap/data` is read-only, no cluster jobs, few behavioural
tests, minimal reusable code. The README is sections 1 to 5 of this file; a behaviour that
is not stated here is not implemented without adding the statement first.

Versioning. Every consumer pins a floating range and its CI installs the newest version in
it within the hour of a release, so the version number says what a release can break. A
**patch** release brings the code back to what this spec says, or reads more stores or
reads them faster; it places nothing differently and adds nothing public. A **minor**
release is anything else: it may move voxels or add API, and an entry added to, renamed in
or removed from the unit table is API (`wa.units.TABLE` is public, and cellmap-flow exports
it to its browser app behind a freshness test). Both were once shipped as patches and both
turned cellmap-flow's CI red on every open PR within the hour: 0.1.1 moved offset-less N5
pyramids by half a base voxel, and 0.3.1 added kilometres. A changelog entry starts with
**Placement change** when voxels move and **Unit table change** when the table does, and the
release message to each consumer repeats those lines; nothing before 1.0 promises more.
Consumers that cannot absorb a move between CI runs pin `>=0.x,<0.(x+1)`.
