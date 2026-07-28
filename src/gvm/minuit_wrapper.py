"""Compatibility wrapper around iminuit."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from iminuit import Minuit  # type: ignore[import-untyped]


ArrayFunction = Callable[[Sequence[float]], float]


def minimize(
    array_func: ArrayFunction,
    x0: Sequence[float],
    names: Sequence[str],
    errordef: float = 0.5,
) -> Minuit:
    """Minimise an objective with iminuit."""
    if hasattr(Minuit, "from_array_func"):
        m = Minuit.from_array_func(
            array_func,
            x0,
            name=names,
            errordef=errordef,
        )
    else:
        m = Minuit(
            array_func,
            x0,
            name=names,
        )
        m.errordef = errordef
    m.migrad()
    return m
