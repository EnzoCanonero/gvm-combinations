"""Run a GVM combination and write summary results."""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from scipy.stats import chi2, norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gvm import GVMCombination, build_input_data, plot_combination_summary


def find_default_config() -> str:
    """Return the only YAML configuration in the current directory."""
    yamls = sorted(globmod.glob("*.yaml")) + sorted(globmod.glob("*.yml"))
    if len(yamls) == 0:
        raise FileNotFoundError("No .yaml files found in the current directory.")
    if len(yamls) > 1:
        raise RuntimeError(
            f"Multiple .yaml files found in the current directory: {yamls}. "
            "Use --config to specify which one."
        )
    return yamls[0]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a GVM combination.")
    p.add_argument("--config", type=str, default=None,
                   help="Path to the YAML config (default: only YAML in the current directory).")
    p.add_argument("--output", type=str, default="output",
                   help="Output directory (default: output/).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = Path(args.config).resolve() if args.config else Path(find_default_config()).resolve()
    output = Path(args.output).resolve()
    output.mkdir(exist_ok=True)

    data = build_input_data(str(config))
    comb = GVMCombination(data)
    fit = comb.fit()

    mu = fit.mu
    lo_1, hi_1, hw_1 = comb.confidence_interval(cl_val=0.683)
    lo_2, hi_2, hw_2 = comb.confidence_interval(cl_val=0.955)
    gof = comb.goodness_of_fit()
    p_val = 1 - chi2.cdf(gof, df=data.n_meas - 1)
    sig = norm.ppf(1 - p_val / 2)

    print(f"\n=== GVM Combination: {data.name} ===")
    print(f"mu_hat      = {mu:.4f}")
    print(f"68.3% CI    = ({lo_1:.4f}, {hi_1:.4f}),  half-width = {hw_1:.4f}")
    print(f"95.5% CI    = ({lo_2:.4f}, {hi_2:.4f}),  half-width = {hw_2:.4f}")
    print(f"GOF chi2    = {gof:.3f}")
    print(f"p-value     = {p_val:.4f}")
    print(f"significance = {sig:.2f} sigma")

    results = {
        "name": data.name,
        "mu_hat": float(mu),
        "confidence_interval_68": {
            "lower": float(lo_1), "upper": float(hi_1),
            "half_width": float(hw_1),
        },
        "confidence_interval_95": {
            "lower": float(lo_2), "upper": float(hi_2),
            "half_width": float(hw_2),
        },
        "goodness_of_fit": {
            "chi2": float(gof),
            "p_value": float(p_val),
            "significance": float(sig),
        },
    }
    with open(output / "results.yaml", "w") as f:
        yaml.dump(results, f, default_flow_style=False, sort_keys=False)

    figure_width = max(10.0, 0.7 * data.n_meas)
    label_rotation = 50.0 if data.n_meas > 8 else 0.0
    fig, ax = plt.subplots(figsize=(figure_width, 6))
    plot_combination_summary(
        ax,
        data,
        mu,
        (lo_1, hi_1),
        (lo_2, hi_2),
        label_rotation=label_rotation,
    )
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {output / 'summary.png'}")


if __name__ == "__main__":
    main()
