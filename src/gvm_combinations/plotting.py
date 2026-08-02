"""Plot combination results."""

from __future__ import annotations

from typing import Optional

import numpy as np
from matplotlib.axes import Axes

from .config import InputData


def plot_combination_summary(
    ax: Axes,
    data: InputData,
    mu: float,
    ci_68: tuple[float, float],
    ci_95: tuple[float, float],
    *,
    title: Optional[str] = None,
    ylabel: str = "Value",
    label_rotation: float = 0.0,
) -> Axes:
    """Plot input measurements and fitted confidence intervals."""
    labels = data.labels
    centrals = np.array([data.measurements[label] for label in labels], dtype=float)
    stat_errors = np.sqrt(np.diag(data.V_stat))

    syst_variance = np.zeros(data.n_meas)
    for sigma in data.syst.values():
        syst_variance += sigma ** 2
    total_errors = np.sqrt(stat_errors ** 2 + syst_variance)

    x_positions = np.arange(data.n_meas)
    ax.axhspan(ci_95[0], ci_95[1], color="yellow", alpha=0.25, label="95.5% CI")
    ax.axhspan(ci_68[0], ci_68[1], color="green", alpha=0.5, label="68.3% CI")
    ax.axhline(mu, color="red", linewidth=1.2, label=r"MLE for $\mu$")
    ax.errorbar(
        x_positions,
        centrals,
        yerr=total_errors,
        fmt="o",
        color="blue",
        capsize=5,
        markersize=7,
        label="Data Points",
    )

    label_alignment = "right" if label_rotation else "center"
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        labels,
        fontsize=14,
        rotation=label_rotation,
        ha=label_alignment,
        rotation_mode="anchor",
    )
    ax.set_ylabel(ylabel, fontsize=16)
    if title is not None:
        ax.set_title(title)
    ax.legend(fontsize=12, loc="upper right")

    lower = min(ci_95[0], float(np.min(centrals - total_errors)))
    upper = max(ci_95[1], float(np.max(centrals + total_errors)))
    span = upper - lower
    padding = 0.15 * span if span > 0.0 else 1.0
    ax.set_ylim(lower - padding, upper + padding)
    ax.grid(True, linestyle="--", linewidth=0.5)
    return ax
