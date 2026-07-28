"""Fit results."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FitResult:
    """Result of a likelihood fit."""

    mu: float
    thetas: np.ndarray
    nll: float
