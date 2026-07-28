"""Fit results."""

from dataclasses import dataclass
import numpy as np


@dataclass
class FitResult:
    """Result of a likelihood fit."""

    mu: float
    thetas: np.ndarray
    nll: float
