"""tensorstore's lost wakeup, reproduced on purpose (spec, Behaviour 6 and open question 14).

Opt-in, about a minute: ``WORLDARRAY_STRESS=1 pytest tests/test_lost_wakeup.py -s``. Worker
processes give a pool limited to one thread its next task 19.5-20.5 ms after the last one
finished, when the pool's only worker exits idle. On an affected tensorstore (every release
up to at least 0.1.85) about 0.3% of those tasks are stranded on a plain context and never
run; on a context the package watches, every one finishes within about a second. When a
release says it carries google/tensorstore ed51718085, run this: no strands on the plain
context means the watchdog can go.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import random
import tempfile
import time

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("WORLDARRAY_STRESS"), reason="set WORLDARRAY_STRESS=1 to run"
)

PROCS, SECONDS, GIVE_UP = 8, 20.0, 10.0


def _aim(args):
    worker, watched = args
    import tensorstore as ts

    import worldarray as wa

    limits = {"data_copy_concurrency": {"limit": 1}, "file_io_concurrency": {"limit": 1}}
    ctx = wa.capped_context(concurrency=1, network=False) if watched else ts.Context(limits)
    kv = ts.KvStore.open({"driver": "file", "path": tempfile.mkdtemp() + "/"}, context=ctx)
    kv = kv.result()
    rng = random.Random(worker)
    slowest, stranded = 0.0, 0
    end = time.monotonic() + SECONDS
    while time.monotonic() < end:
        time.sleep(rng.uniform(0.0195, 0.0205))
        future, start = kv.read("missing"), time.monotonic()
        try:
            future.result(timeout=GIVE_UP)
        except TimeoutError:
            stranded += 1
            kv.read("other").result(timeout=GIVE_UP)  # any task on the pool frees it
            future.result(timeout=GIVE_UP)
        slowest = max(slowest, time.monotonic() - start)
    return stranded, slowest


def _run(watched: bool) -> tuple[int, float]:
    with mp.get_context("spawn").Pool(PROCS) as pool:
        results = pool.map(_aim, [(w, watched) for w in range(PROCS)])
    return sum(r[0] for r in results), max(r[1] for r in results)


def test_a_watched_context_never_strands_a_task():
    import importlib.metadata

    version = importlib.metadata.version("tensorstore")
    stranded, _ = _run(watched=False)
    watched_stranded, slowest = _run(watched=True)
    print(
        f"\ntensorstore {version}: plain context stranded {stranded} tasks for {GIVE_UP:g}s+; "
        f"watched context stranded {watched_stranded}, slowest {slowest:.2f}s"
    )
    assert watched_stranded == 0 and slowest < 2.0
    if stranded == 0:
        pytest.skip(
            f"tensorstore {version} stranded nothing: if it carries ed51718085, "
            "the watchdog can go (open question 14)"
        )
