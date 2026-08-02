"""Tools for combining correlated measurements with the Gamma Variance Model."""

from importlib.metadata import PackageNotFoundError, version

from .combination import GVMCombination
from .config import InputData, build_input_data, validate_input_data
from .fit_results import FitResult
from .plotting import plot_combination_summary

try:
    __version__ = version("gvm-combinations")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "GVMCombination",
    "InputData",
    "build_input_data",
    "validate_input_data",
    "FitResult",
    "plot_combination_summary",
]
