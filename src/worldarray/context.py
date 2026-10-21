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

import threading

import tensorstore as ts

_shared: dict[int, ts.Context] = {}
_lock = threading.Lock()


def shared_context(cache_bytes: int = 0) -> ts.Context:
    with _lock:
        ctx = _shared.get(int(cache_bytes))
        if ctx is None:
            ctx = ts.Context({"cache_pool": {"total_bytes_limit": int(cache_bytes)}})
            _shared[int(cache_bytes)] = ctx
        return ctx


def capped_context_spec(cache_bytes: int = 0, concurrency: int = 1) -> dict:
    """The spec dict behind ``capped_context``, for callers building their own opens."""
    limit = {"limit": int(concurrency)}
    return {
        "data_copy_concurrency": limit,
        "file_io_concurrency": limit,
        "gcs_request_concurrency": limit,
        "s3_request_concurrency": limit,
        "http_request_concurrency": limit,
        "cache_pool": {"total_bytes_limit": int(cache_bytes)},
    }


def capped_context(cache_bytes: int = 0, concurrency: int = 1) -> ts.Context:
    return ts.Context(capped_context_spec(cache_bytes, concurrency))
