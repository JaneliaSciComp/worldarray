"""Tensorstore contexts: one shared per cache budget, or a capped one per call.

``shared_context(cache_bytes)`` is one ``tensorstore.Context`` per budget for the whole
process, so the levels of a pyramid and repeat opens of an array draw on one cache pool.
The default budget is 0: tensorstore's decoded-chunk cache is off, because every consumer
either reads each region once or caches decoded chunks itself. Its decoding and file I/O
pools are sized by ``thread_budget()``, from the variables a cluster job sets, where
tensorstore alone would size them to every CPU on the node.

``capped_context`` is mesh-n-bone's worker-process context: every tensorstore concurrency
resource limited, so dozens of worker processes on one host do not multiply into thousands
of threads. It returns a **new** context each call, as those workers inline the spec into
every open today; pass the same object twice to share pools (spec, Behaviour 7).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any

import tensorstore as ts

_shared: dict[int, ts.Context] = {}
_rebuilt: dict[str, ts.Context] = {}
_lock = threading.Lock()


# tensorstore's lost wakeup. In every release up to at least 0.1.85, a pool worker idle for
# 20 ms exits, and a task queued as it exits gets no thread while the pool counts itself
# full: the task never runs and its future waits forever, with no tensorstore thread alive.
# A pool limited to one thread meets it most (cellmap-analyze's workers on /nrs,
# 2026-10-22). Any later task on the same pool wakes it, so ``wait`` gives a context's pools
# a no-op task (``kick``) every KICK_AFTER seconds a future is pending. Fixed upstream in
# google/tensorstore ed51718085 (2026-09-30, "Fix lost wakeups in TaskGroup when workers
# block or exit"). Once a release contains it, require that release and remove ``kick``
# (spec, open question 14): `gh api repos/google/tensorstore/compare/ed51718085...vX.Y.Z`
# says "ahead" or "identical" for a release that has it.
KICK_AFTER = 1.0
_KICK_DIR = tempfile.gettempdir()
_KICK_KEY = "worldarray-kick-no-such-key"


def kick(context: ts.Context) -> list:
    """One no-op task on each of ``context``'s decoding and file I/O pools: an in-memory
    array copy and a local read of a key that does not exist. No store is touched. Returns
    the futures, which the caller holds until done (a dropped future may be cancelled)."""
    futures = []
    try:
        array = ts.open(
            {"driver": "array", "array": [0], "dtype": "uint8"}, context=context
        ).result(timeout=KICK_AFTER)
        futures.append(array.read())
        kv = ts.KvStore.open({"driver": "file", "path": _KICK_DIR + "/"}, context=context).result(
            timeout=KICK_AFTER
        )
        futures.append(kv.read(_KICK_KEY))
    except Exception:  # noqa: BLE001  a kick that cannot start changes nothing
        pass
    return futures


def wait(future, context: ts.Context | None, timeout: float | None = None):
    """``future.result()``, kicking ``context``'s pools after every ``KICK_AFTER`` seconds it
    is still pending; ``TimeoutError`` once ``timeout`` seconds have passed (never when
    None). A future that is not a tensorstore one, or no context, is simply waited on."""
    deadline = None if timeout is None else time.monotonic() + timeout
    kicks: list = []
    while True:
        step = KICK_AFTER if deadline is None else min(KICK_AFTER, deadline - time.monotonic())
        try:
            return future.result(timeout=max(step, 0.0))
        except TimeoutError:
            if deadline is not None and time.monotonic() >= deadline:
                raise
            if context is not None:
                kicks = [f for f in kicks if not f.done()] + kick(context)


def thread_budget() -> int:
    """Threads for this process's tensorstore decoding and file I/O: ``OMP_NUM_THREADS``
    when set (dask sets 1 in every worker it spawns; a job script sets its slots), else
    ``LSB_DJOB_NUMPROC`` (an LSF job's slots), else the CPUs this process may run on."""
    for name in ("OMP_NUM_THREADS", "LSB_DJOB_NUMPROC"):
        try:
            n = int(os.environ.get(name, ""))
        except ValueError:
            continue
        if n > 0:
            return n
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def shared_context(cache_bytes: int = 0) -> ts.Context:
    with _lock:
        ctx = _shared.get(int(cache_bytes))
        if ctx is None:
            limit = {"limit": thread_budget()}
            ctx = ts.Context(
                {
                    "cache_pool": {"total_bytes_limit": int(cache_bytes)},
                    "data_copy_concurrency": limit,
                    "file_io_concurrency": limit,
                }
            )
            _shared[int(cache_bytes)] = ctx
        return ctx


def capped_context_spec(cache_bytes: int = 0, concurrency: int = 1, network: bool = True) -> dict:
    """The spec dict behind ``capped_context``, for callers building their own opens.
    ``network=False`` caps only decoding and local file I/O, leaving tensorstore's request
    pools alone (cellmap-analyze's choice for remote reads). A ``concurrency`` of 1 is safe:
    every wait in the package kicks a pool that stranded a task (see ``KICK_AFTER``)."""
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
