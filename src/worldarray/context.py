"""Tensorstore contexts: one shared per cache budget, or a capped one per call.

``shared_context(cache_bytes)`` is one ``tensorstore.Context`` per budget for the whole
process, so the levels of a pyramid and repeat opens of an array draw on one cache pool.
The default budget is 0: tensorstore's decoded-chunk cache is off, because every consumer
either reads each region once or caches decoded chunks itself.

``capped_context`` is mesh-n-bone's worker-process context: every tensorstore concurrency
resource limited, so dozens of worker processes on one host do not multiply into thousands
of threads. It returns a **new** context each call, as those workers inline the spec into
every open today; pass the same object twice to share pools (spec, Behaviour 7).
"""

from __future__ import annotations

import json
import threading
from typing import Any

import tensorstore as ts

_shared: dict[int, ts.Context] = {}
_rebuilt: dict[str, ts.Context] = {}
_lock = threading.Lock()


def shared_context(cache_bytes: int = 0) -> ts.Context:
    with _lock:
        ctx = _shared.get(int(cache_bytes))
        if ctx is None:
            ctx = ts.Context({"cache_pool": {"total_bytes_limit": int(cache_bytes)}})
            _shared[int(cache_bytes)] = ctx
        return ctx


def capped_context_spec(cache_bytes: int = 0, concurrency: int = 1, network: bool = True) -> dict:
    """The spec dict behind ``capped_context``, for callers building their own opens.
    ``network=False`` caps only decoding and local file I/O, leaving tensorstore's request
    pools alone (cellmap-analyze's choice for remote reads)."""
    limit = {"limit": int(concurrency)}
    spec = {
        "data_copy_concurrency": limit,
        "file_io_concurrency": limit,
        "cache_pool": {"total_bytes_limit": int(cache_bytes)},
    }
    if network:
        spec.update(
            gcs_request_concurrency=limit,
            s3_request_concurrency=limit,
            http_request_concurrency=limit,
        )
    return spec


def capped_context(cache_bytes: int = 0, concurrency: int = 1, network: bool = True) -> ts.Context:
    return ts.Context(capped_context_spec(cache_bytes, concurrency, network))


def recipe_of(context: ts.Context) -> tuple[str, Any]:
    """How another process rebuilds ``context``: ``("shared", cache_bytes)`` for a shared
    context, else ``("spec", json)`` from the context's own spec, so a capped context stays
    capped across a pickle (Behaviour 11, mesh-n-bone's workers)."""
    with _lock:
        for cache_bytes, ctx in _shared.items():
            if ctx is context:
                return ("shared", cache_bytes)
    spec = context.spec
    return ("spec", json.dumps(spec.to_json() if spec is not None else {}, sort_keys=True))


def from_recipe(recipe: tuple[str, Any] | None) -> ts.Context | None:
    """``recipe_of`` inverted; one rebuilt context per spec per process, so handles that
    shared a context keep sharing one."""
    if recipe is None:
        return None
    kind, value = recipe
    if kind == "shared":
        return shared_context(int(value))
    with _lock:
        ctx = _rebuilt.get(value)
        if ctx is None:
            ctx = ts.Context(ts.Context.Spec(json.loads(value)))
            _rebuilt[value] = ctx
        return ctx
