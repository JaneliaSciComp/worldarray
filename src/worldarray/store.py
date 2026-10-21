"""Where an array's files are, reading its metadata documents, and opening it.

Everything goes through tensorstore's kvstore: ``zarr.json``, ``.zarray``, ``.zattrs``,
``.zgroup``, ``attributes.json`` and ``info`` are read as bytes and parsed here, so no
zarr-python is involved. Remote stores are read the way a public dataset most likely lets
one in, with credentials only when a public read is refused (spec, Conventions 5).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

import tensorstore as ts

logger = logging.getLogger(__name__)
T = TypeVar("T")

PRECOMPUTED_PREFIX = "precomputed://"
GCS_PUBLIC_URL = "https://storage.googleapis.com"
METADATA_RETRIES = {"max_retries": 5, "initial_delay": "0.2s", "max_delay": "2s"}
PROBE_KEYS = ("zarr.json", ".zarray", ".zgroup", "info", "attributes.json")

_SCHEME = re.compile(r"^(s3|gs|http|https|file|memory)://")
_REMOTE = ("s3://", "gs://", "http://", "https://")


# ---------------------------------------------------------------------------
# Paths


def normalize_path(path: str) -> str:
    """``precomputed://`` dropped, trailing slashes dropped, a local path made absolute."""
    path = str(path)
    if path.startswith(PRECOMPUTED_PREFIX):
        path = path[len(PRECOMPUTED_PREFIX) :]
    path = path.rstrip("/") or "/"
    if not _SCHEME.match(path):
        path = os.path.abspath(path)
    return path


def is_remote(path: str) -> bool:
    return normalize_path(path).startswith(_REMOTE)


def parent(path: str) -> str | None:
    """The containing directory or URL prefix; None at a root (``/``, ``s3://bucket``)."""
    path = normalize_path(path)
    m = _SCHEME.match(path)
    head = m.group(0) if m else ""
    rest = path[len(head) :]
    if "/" not in rest.strip("/"):
        return None if (head or rest == "/") else "/"
    return head + rest.rsplit("/", 1)[0]


def basename(path: str) -> str:
    return normalize_path(path).rsplit("/", 1)[-1]


def join(base: str, *parts: str) -> str:
    return "/".join([normalize_path(base).rstrip("/"), *(p.strip("/") for p in parts)])


# ---------------------------------------------------------------------------
# Access


def kvstore_specs(root: str) -> list[dict]:
    """The tensorstore kvstore specs that read the directory ``root``, in the order to try
    them. ``s3://``: anonymously, then AWS's default credentials (S3 refuses even a public
    read signed with a stale or foreign key). ``gs://``: Google's default credentials
    (anonymous without any), then the bucket's public https URL for credentials that are
    there but broken (``TENSORSTORE_GCS_HTTP_URL`` moves it). Anything else: one way. Each
    root ends in ``/`` because tensorstore concatenates keys literally."""
    root = normalize_path(root).rstrip("/") + "/"  # "/" stays "/", never "//"
    if not _SCHEME.match(root):
        return [{"driver": "file", "path": root}]
    spec = ts.KvStore.Spec(root).to_json()
    if spec["driver"] == "s3":
        extra: dict = {}
        endpoint = os.environ.get("AWS_ENDPOINT_URL_S3") or os.environ.get("AWS_ENDPOINT_URL")
        if endpoint:  # an S3-compatible endpoint (MinIO, moto, a mirror), as the AWS tools read it
            extra["endpoint"] = endpoint
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            extra["aws_region"] = region
        return [{**spec, **extra, "aws_credentials": {"type": t}} for t in ("anonymous", "default")]
    if spec["driver"] == "gcs":
        public = (os.environ.get("TENSORSTORE_GCS_HTTP_URL") or GCS_PUBLIC_URL).rstrip("/")
        return [
            spec,
            {
                "driver": "http",
                "base_url": public,
                "path": f"/{spec['bucket']}/{spec.get('path', '')}",
            },
        ]
    return [spec]


class Refused(Exception):
    """A read was refused for want of credentials (not a missing key, not a network error)."""


def is_refused(error: BaseException) -> bool:
    return isinstance(error, Refused) or str(error).startswith(
        ("PERMISSION_DENIED", "UNAUTHENTICATED")
    )


_admitted: dict[tuple[str, str], int] = {}  # (driver, bucket) -> index of the spec let in
_admitted_lock = threading.Lock()


def with_access(specs: Sequence[dict], attempt: Callable[[dict], T]) -> T:
    """``attempt(spec)`` with each spec in turn, moving on only while access is refused; the
    way that worked is tried first next time, for its bucket. Any other error, and the last
    refusal, is raised."""
    if len(specs) == 1:
        return attempt(specs[0])
    key = (specs[0]["driver"], specs[0].get("bucket", ""))
    with _admitted_lock:
        first = _admitted.get(key, 0)
    order = [first] + [i for i in range(len(specs)) if i != first]
    for i in order[:-1]:
        try:
            result = attempt(specs[i])
        except Exception as e:
            if not is_refused(e):
                raise
            logger.info(
                "%s refused a read (%s); trying the next way in", key, str(e).splitlines()[0][:200]
            )
            continue
        with _admitted_lock:
            _admitted[key] = i
        return result
    result = attempt(specs[order[-1]])
    with _admitted_lock:
        _admitted[key] = order[-1]
    return result


def metadata_spec(spec: dict) -> dict:
    """``spec`` with its requests' retries capped, so a mistyped host fails in seconds."""
    if spec["driver"] not in ("s3", "gcs", "http"):
        return spec
    return {**spec, "context": {f"{spec['driver']}_request_retries": METADATA_RETRIES}}


@functools.lru_cache(maxsize=512)
def _metadata_kvstore(spec_json: str, gcs_url: str | None) -> ts.KvStore:
    """One opened kvstore per spec, and per ``TENSORSTORE_GCS_HTTP_URL``: tensorstore's gcs
    driver reads that variable when the store is opened, so a cached store would keep
    talking to the host the variable named then."""
    return ts.KvStore.open(metadata_spec(json.loads(spec_json))).result()


def _read_key(spec: dict, key: str) -> bytes | None:
    kv = _metadata_kvstore(
        json.dumps(spec, sort_keys=True), os.environ.get("TENSORSTORE_GCS_HTTP_URL")
    )
    result = kv.read(key).result()
    return None if result.state == "missing" else bytes(result.value)


class Location:
    """A directory (local or a URL) and the ways in to it."""

    def __init__(self, root: str):
        self.root = normalize_path(root)
        self.specs = kvstore_specs(self.root)

    def read(self, key: str) -> bytes | None:
        return with_access(self.specs, lambda spec: _read_key(spec, key))

    def read_json(self, key: str) -> dict | None:
        data = self.read(key)
        return None if data is None else json.loads(data)

    def read_many(self, keys: Sequence[str]) -> dict[str, bytes | None]:
        """All of ``keys`` under one way in. A bucket with listing off answers a refusal for a
        missing key, so a refusal on one key counts as missing when another key was read
        under the same way; only when every key is refused is the next way tried."""

        def attempt(spec: dict) -> dict[str, bytes | None]:
            found: dict[str, bytes | None] = {}
            refusals: dict[str, BaseException] = {}
            for key in keys:
                try:
                    found[key] = _read_key(spec, key)
                except Exception as e:
                    if not is_refused(e):
                        raise
                    refusals[key] = e
            if refusals and len(refusals) == len(keys):
                raise Refused(next(iter(refusals.values())).args[0] if refusals else "refused")
            for key in refusals:
                found[key] = None
            return found

        return with_access(self.specs, attempt)


# ---------------------------------------------------------------------------
# Detection


@dataclass(frozen=True)
class Node:
    """What is at a path: its format, whether it is an array or a group, and the JSON
    documents found there. For a precomputed scale named by its key, ``path`` is the
    volume root and ``scale_key`` the key."""

    format: str  # zarr2 | zarr3 | n5 | precomputed
    kind: str  # array | group
    path: str
    docs: dict[str, Any] = field(default_factory=dict)
    scale_key: str | None = None

    @property
    def location(self) -> Location:
        return Location(self.path)

    @property
    def attrs(self) -> dict:
        """The node's user attributes, raw: zarr v3 ``attributes``, zarr v2 ``.zattrs``, the N5
        ``attributes.json``, a precomputed ``info``. ``{}`` for a group with no document."""
        if self.format == "zarr3":
            return dict(self.docs.get("zarr.json", {}).get("attributes", {}) or {})
        if self.format == "zarr2":
            return dict(self.location.read_json(".zattrs") or {})
        if self.format == "n5":
            return dict(self.docs.get("attributes.json", {}) or {})
        return dict(self.docs.get("info", {}) or {})

    @property
    def ome_attrs(self) -> dict:
        """``attrs`` with OME-NGFF 0.5's ``ome`` member unwrapped onto the top level, so
        ``multiscales``, ``coordinateSystems``, ``coordinateTransformations`` and the rest
        read the same for 0.4 and 0.5."""
        attrs = self.attrs
        ome = attrs.get("ome")
        return {**attrs, **ome} if isinstance(ome, dict) else attrs

    @property
    def multiscales(self) -> list[dict]:
        """The OME-NGFF ``multiscales`` list (0.5's ``ome`` nesting unwrapped), else ``[]``."""
        return multiscales_of(self.attrs)


def multiscales_of(attrs: dict) -> list[dict]:
    ome = attrs.get("ome")
    if isinstance(ome, dict) and "multiscales" in ome:
        return list(ome["multiscales"])
    return list(attrs.get("multiscales", []) or [])


def _decode(docs: dict[str, bytes | None]) -> dict[str, Any]:
    return {k: json.loads(v) for k, v in docs.items() if v is not None}


def detect(path: str, *, missing_ok: bool = False) -> Node | None:
    """Probe ``zarr.json``, ``.zarray``, ``.zgroup``, ``info``, ``attributes.json`` in that
    order; failing all, probe the parent for ``info`` and take the last path segment as a
    precomputed scale key, then probe ``s0`` for a group that has no document of its own.
    The ``.n5`` or ``.zarr`` suffix is never consulted. Nothing found raises
    ``FileNotFoundError``, or returns None with ``missing_ok``."""
    path = normalize_path(path)
    docs = _decode(Location(path).read_many(PROBE_KEYS))
    if "zarr.json" in docs:
        node_type = docs["zarr.json"].get("node_type", "array")
        return Node("zarr3", "array" if node_type == "array" else "group", path, docs)
    if ".zarray" in docs:
        return Node("zarr2", "array", path, docs)
    if ".zgroup" in docs:
        return Node("zarr2", "group", path, docs)
    if "info" in docs:
        return Node("precomputed", "group", path, docs)
    if "attributes.json" in docs:
        kind = "array" if "dimensions" in docs["attributes.json"] else "group"
        return Node("n5", kind, path, docs)
    up = parent(path)
    if up is not None:
        info = Location(up).read_json("info")
        if info is not None:
            return Node("precomputed", "group", up, {"info": info}, scale_key=basename(path))
    child = _decode(
        Location(join(path, "s0")).read_many(("zarr.json", ".zarray", "attributes.json"))
    )
    if child:
        fmt = "zarr3" if "zarr.json" in child else "zarr2" if ".zarray" in child else "n5"
        return Node(fmt, "group", path, {})
    if missing_ok:
        return None
    raise FileNotFoundError(
        f"{path}: none of {', '.join(PROBE_KEYS)} found, the parent holds no precomputed info, and there is no s0"
    )


# ---------------------------------------------------------------------------
# Opening


def precomputed_scale_index(info: dict, key: str | int) -> int:
    """The index of a precomputed scale named by its key (``"8_8_8"``), by position
    (``1``), or by cellmap-flow's ``s<N>`` convention when no scale is keyed that way."""
    keys = [str(s.get("key", i)) for i, s in enumerate(info["scales"])]
    if isinstance(key, int) and not isinstance(key, bool):
        if not -len(keys) <= key < len(keys):
            raise ValueError(f"no precomputed scale {key}; there are {len(keys)}: {keys}")
        return key % len(keys)
    if key in keys:
        return keys.index(key)
    m = re.fullmatch(r"s(\d+)", str(key))
    if m and int(m.group(1)) < len(keys):
        return int(m.group(1))
    raise ValueError(f"no precomputed scale {key!r}; scales: {keys}")


_KNOWN_COMPRESSOR_MEMBERS = {
    "zstd": {"id", "level"},
    "blosc": {"id", "cname", "clevel", "shuffle", "blocksize"},
    "gzip": {"id", "level"},
    "zlib": {"id", "level"},
    "bz2": {"id", "level"},
}


def cleaned_zarray(zarray: dict) -> dict:
    """A zarr v2 ``.zarray`` with compressor members tensorstore rejects dropped, such as the
    ``checksum`` newer numcodecs writes for zstd."""
    compressor = zarray.get("compressor")
    if isinstance(compressor, dict) and compressor.get("id") in _KNOWN_COMPRESSOR_MEMBERS:
        keep = _KNOWN_COMPRESSOR_MEMBERS[compressor["id"]]
        compressor = {k: v for k, v in compressor.items() if k in keep}
    return {**zarray, "compressor": compressor}


_DRIVER = {"zarr2": "zarr", "zarr3": "zarr3", "n5": "n5", "precomputed": "neuroglancer_precomputed"}


def open_array(
    node: Node,
    *,
    context: ts.Context,
    scale_index: int | None = None,
    squeeze_channel: bool = True,
) -> ts.TensorStore:
    """The tensorstore of an array node (or one scale of a precomputed node), in C order
    with its origin at 0: N5 and precomputed are transposed, a precomputed ``voxel_offset``
    is moved to 0 (and carried in the metadata as the translation), and a precomputed
    single channel is dropped."""
    if node.kind != "array" and node.format != "precomputed":
        raise ValueError(f"{node.path} is a {node.format} group, not an array")

    def attempt(kvstore: dict) -> ts.TensorStore:
        spec: dict[str, Any] = {"driver": _DRIVER[node.format], "kvstore": kvstore}
        if node.format == "zarr2":
            spec["metadata"] = cleaned_zarray(node.docs[".zarray"])
            spec["assume_metadata"] = True
        if node.format == "n5":
            # only the array members: tensorstore rejects attributes it validates and finds
            # odd (a "units" string instead of a list); everything else is read here, not by it
            core = node.docs["attributes.json"]
            spec["metadata"] = {
                k: core[k]
                for k in ("dimensions", "blockSize", "dataType", "compression")
                if k in core
            }
            spec["assume_metadata"] = True
        if node.format == "precomputed" and scale_index is not None:
            spec["scale_index"] = scale_index
        return ts.open(spec, read=True, context=context).result()

    store = with_access(kvstore_specs(node.path), attempt)
    if node.format in ("n5", "precomputed"):
        store = store.T
    if node.format == "precomputed":
        store = store[ts.d[:].translate_to[0]]
        if squeeze_channel and store.shape[0] == 1:
            store = store[0]
    return store
