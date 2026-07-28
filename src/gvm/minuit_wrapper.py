"""Compatibility wrapper around iminuit."""

from iminuit import Minuit


def minimize(array_func, x0, names, errordef=0.5):
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
