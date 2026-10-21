"""Retrying a data read on a transient failure.

mesh-n-bone's policy, which its tests pin: attempt ``k`` (from 1) waits ``timeout * k``
seconds; ``timeout`` is given, or sized from the read and where it comes from:
``clamp(MiB / 16, 5, 120)`` for a remote store and ``clamp(MiB / 256, 5, 30)`` for a local
one. Between attempts the sleep is ``base_delay * 1.3 ** (k - 1)`` plus up to one second of
jitter. A timed-out future is cancelled before the next attempt (spec, Behaviour 6).
"""

from __future__ import annotations

import logging
import random
import re
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_TRANSIENT_PREFIXES = ("UNAVAILABLE", "DEADLINE_EXCEEDED", "RESOURCE_EXHAUSTED")
_HTTP_TRANSIENT = re.compile(r"HTTP (?:5\d\d|429)\b")


@dataclass
class Retry:
    max_attempts: int = 10
    timeout: float | None = None  # seconds per attempt before scaling; None: from the read's size
    base_delay: float = 1.0

    def timeout_for(self, nbytes: int, remote: bool) -> float:
        if self.timeout is not None:
            return float(self.timeout)
        mib = nbytes / 2**20
        return min(120.0, max(5.0, mib / 16.0)) if remote else min(30.0, max(5.0, mib / 256.0))


def is_transient(error: BaseException) -> bool:
    if isinstance(error, TimeoutError):
        return True
    message = str(error)
    return message.startswith(_TRANSIENT_PREFIXES) or bool(_HTTP_TRANSIENT.search(message))


def read_with_retries(view, retry: Retry, *, remote: bool, what: str = "") -> np.ndarray:
    """``view.read()`` as a numpy array, retried per ``retry``. ``view`` is a tensorstore
    (or any object whose ``read()`` returns a future with ``result(timeout)`` and
    ``cancel()``)."""
    nbytes = int(np.prod(view.shape, dtype=np.int64)) * np.dtype(view.dtype.numpy_dtype).itemsize
    timeout = retry.timeout_for(nbytes, remote)
    last: BaseException | None = None
    for attempt in range(1, max(1, retry.max_attempts) + 1):
        future = view.read()
        try:
            return future.result(timeout=timeout * attempt)
        except TimeoutError as e:
            future.cancel()
            last = e
        except Exception as e:
            if not is_transient(e):
                raise
            last = e
        if attempt < retry.max_attempts:
            delay = retry.base_delay * 1.3 ** (attempt - 1) + random.random()
            logger.warning(
                "read %s failed on attempt %d/%d (%s); retrying in %.1fs",
                what,
                attempt,
                retry.max_attempts,
                str(last).splitlines()[0][:120],
                delay,
            )
            time.sleep(delay)
    assert last is not None
    raise last
