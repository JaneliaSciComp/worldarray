"""Units: kept as stored, converted only on the axes a caller asks to convert.

One table, accepting American and British spellings, ``um``, ``µm`` (U+00B5) and ``μm``
(U+03BC), and the usual time units. A missing unit is the empty string. Nothing here is
consulted until a conversion is asked for, so an array with ``"degrees_north"`` axes is as
welcome as one in nanometres (spec, Conventions 3).
"""

from __future__ import annotations

NANOMETER = "nanometer"
SECOND = "second"

_LENGTH_M = {
    "picometer": 1e-12, "picometre": 1e-12, "pm": 1e-12,
    "angstrom": 1e-10, "ångström": 1e-10, "å": 1e-10, "Å": 1e-10,
    "nanometer": 1e-9, "nanometre": 1e-9, "nm": 1e-9,
    "micrometer": 1e-6, "micrometre": 1e-6, "micron": 1e-6, "um": 1e-6, "µm": 1e-6, "μm": 1e-6,
    "millimeter": 1e-3, "millimetre": 1e-3, "mm": 1e-3,
    "centimeter": 1e-2, "centimetre": 1e-2, "cm": 1e-2,
    "meter": 1.0, "metre": 1.0, "m": 1.0,
}  # fmt: skip
_TIME_S = {
    "nanosecond": 1e-9, "ns": 1e-9,
    "microsecond": 1e-6, "us": 1e-6, "µs": 1e-6, "μs": 1e-6,
    "millisecond": 1e-3, "ms": 1e-3,
    "second": 1.0, "sec": 1.0, "s": 1.0,
    "minute": 60.0, "min": 60.0,
    "hour": 3600.0, "h": 3600.0,
    "day": 86400.0, "d": 86400.0,
}  # fmt: skip


def _lookup(unit: str) -> tuple[str, float] | None:
    """``(kind, size in the kind's base unit)`` of a unit string, or None when unknown.
    Plurals and case are forgiven; the Greek letters are matched as they are."""
    for key in (unit, unit.strip(), unit.strip().lower(), unit.strip().lower().rstrip("s")):
        if key in _LENGTH_M:
            return "space", _LENGTH_M[key]
        if key in _TIME_S:
            return "time", _TIME_S[key]
    return None


def kind(unit: str) -> str:
    """``"space"``, ``"time"`` or ``"unknown"`` for a unit string; ``""`` is unknown."""
    found = _lookup(unit) if unit else None
    return found[0] if found else "unknown"


def factor(unit_from: str, unit_to: str, *, missing_as: str | None = None) -> float:
    """How many ``unit_to`` one ``unit_from`` is.

    ``unit_from == ""`` (nothing stored) is read as ``missing_as`` when that is given, else
    it is an error: the caller decides what an unlabelled axis means, not this table. An
    unknown unit, or a length asked for in a time unit, is a ``ValueError``.
    """
    if unit_from == "":
        if missing_as is None:
            raise ValueError("no unit stored on this axis and no default given")
        unit_from = missing_as
    a, b = _lookup(unit_from), _lookup(unit_to)
    if a is None:
        raise ValueError(f"unknown unit {unit_from!r}")
    if b is None:
        raise ValueError(f"unknown unit {unit_to!r}")
    if a[0] != b[0]:
        raise ValueError(f"cannot convert {unit_from!r} ({a[0]}) to {unit_to!r} ({b[0]})")
    return a[1] / b[1]
