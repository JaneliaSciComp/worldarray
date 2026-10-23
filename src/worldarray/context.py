"""Tensorstore contexts: one shared per cache budget, or a capped one per call.

``shared_context(cache_bytes)`` is one ``tensorstore.Context`` per budget for the whole
process, so the levels of a pyramid and repeat opens of an array draw on one cache pool.
The default budget is 0: tensorstore's decoded-chunk cache is off, because every consumer
either reads each region once or caches decoded chunks itself. Its decoding and file I/O
pools are sized by ``thread_budget()``, from the variables a cluster job sets, where
tensorstore alone would size them to every CPU on the node. Every context here, and every
one a store is opened on, is watched for tensorstore's lost wakeup (``watch``).

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
import weakref
from typing import Any

import tensorstore as ts

_shared: dict[int, ts.Context] = {}
_rebuilt: dict[str, ts.Context] = {}
_lock = threading.Lock()


# tensorstore's lost wakeup. In every release up to at least 0.1.85, a pool worker idle for
# 20 ms exits, and a task queued as it exits gets no thread while the pool counts itself
# full: the task never runs and its future waits forever, with no tensorstore thread alive.
# A pool limited to one thread meets it most (cellmap-analyze's workers on /nrs,
# 2026-10-22; its read retries of June 2025 were working around the same thing). Any later
# task on the same pool wakes it. So every context the package makes or opens a store on is
# watched: a daemon thread gives each one's decoding and file I/O pools a no-op task every
# KICK_EVERY seconds, and a stranded task, whoever waits on it (worldarray, ``vol.ts``,
# ``await``), runs within that. Fixed upstream in google/tensorstore ed51718085 (2026-09-30,
# "Fix lost wakeups in TaskGroup when workers block or exit"). Once a release contains it,
# require that release and remove the watchdog (spec, open question 14):
# `gh api repos/google/tensorstore/compare/ed51718085...vX.Y.Z` says "ahead" or "identical"
# for a release that has it, and `WORLDARRAY_STRESS=1 pytest tests/test_lost_wakeup.py -s`
# then strands nothing on a plain context.
KICK_EVERY = 1.0
_KICK_DIR = tempfile.gettempdir()
_KICK_KEY = "worldarray-kick-no-such-key"
_watched: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
_watch_lock = threading.Lock()
_watchdog: threading.Thread | None = None


def _kickers(context: ts.Context) -> tuple:
    """An in-memory array and a local kvstore on ``context``: reading the one copies on its
    decoding pool, reading a missing key from the other runs on its file I/O pool."""
    array = ts.open({"driver": "array", "array": [0], "dtype": "uint8"}, context=context)
    kv = ts.KvStore.open({"driver": "file", "path": _KICK_DIR + "/"}, context=context)
    return array.result(timeout=10), kv.result(timeout=10)


def kick(context: ts.Context) -> list:
    """One no-op task on each of ``context``'s decoding and file I/O pools; no store is
    touched. Returns the futures, which the caller holds until done (a dropped future may
    be cancelled before it is queued)."""
    try:
        with _watch_lock:
            kickers = _watched.get(context)
        if kickers is None:
            kickers = _kickers(context)
            with _watch_lock:
                if context in _watched:
                    _watched[context] = kickers
        array, kv = kickers
        return [array.read(), kv.read(_KICK_KEY)]
    except Exception:  # noqa: BLE001  a kick that cannot start changes nothing
        return []


def watch(context: ts.Context | None) -> ts.Context | None:
    """Kick ``context``'s pools every ``KICK_EVERY`` seconds for as long as it lives, starting
    the watchdog thread on first use. Returns ``context``."""
    global _watchdog
    if context is None:
        return None
    with _watch_lock:
        if context not in _watched:
            _watched[context] = None
        if _watchdog is None or not _watchdog.is_alive():
            _watchdog = threading.Thread(target=_watch, name="worldarray-kick", daemon=True)
            _watchdog.start()
    return context


def _watch() -> None:
    held: list = []
    while True:
        time.sleep(KICK_EVERY)
        with _watch_lock:
            contexts = list(_watched.keys())
        held = [f for f in held if not f.done()]
        for context in contexts:
            held += kick(context)
        del contexts


def _forget_watchdog() -> None:
    global _watchdog
    _watchdog = None  # a forked child has no such thread; the next watch() starts one


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_watchdog)


_SLOTS = ("OMP_NUM_THREADS", "LSB_DJOB_NUMPROC", "SLURM_CPUS_PER_TASK", "NSLOTS")


def thread_budget() -> int:
    """Threads for this process's tensorstore decoding and file I/O: ``OMP_NUM_THREADS``
    when set (dask sets 1 in every worker it spawns; a job script sets its slots), else the
    job's slots (LSF's ``LSB_DJOB_NUMPROC``, SLURM's ``SLURM_CPUS_PER_TASK``, SGE's
    ``NSLOTS``), else the CPUs this process may run on."""
    for name in _SLOTS:
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
            watch(ctx)
            _shared[int(cache_bytes)] = ctx
        return ctx


def capped_context_spec(cache_bytes: int = 0, concurrency: int = 1, network: bool = True) -> dict:
    """The spec dict behind ``capped_context``, for callers building their own opens.
    ``network=False`` caps only decoding and local file I/O, leaving tensorstore's request
    pools alone (cellmap-analyze's choice for remote reads). A ``concurrency`` of 1 is safe:
    the context is watched (see ``KICK_EVERY``)."""
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
    return watch(ts.Context(capped_context_spec(cache_bytes, concurrency, network)))


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
            ctx = watch(ts.Context(ts.Context.Spec(json.loads(value))))
            _rebuilt[value] = ctx
        return ctx
