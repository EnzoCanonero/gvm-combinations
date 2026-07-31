# GVM Combination Toolkit

A Python toolkit for combining correlated measurements with errors-on-errors.

This repository provides a simple implementation of the Gamma Variance Model (GVM) for
combining correlated measurements. The `GVMCombination` class constructs the likelihood,
performs the minimisation with *Minuit*, and computes confidence intervals using an
analytic Bartlett correction.

## Contents

- [Statistical overview](#statistical-overview)
- [Bartlett correction](#bartlett-correction)
- [Installation](#installation)
- [Testing](#testing)
- [Quick start](#quick-start)
- [Configuration file](#configuration-file)
- [Runs](#runs)
- [Tutorials](#tutorials)
- [Repository layout](#repository-layout)
- [Companion paper](#companion-paper)
- [Citation](#citation)
- [License](#license)

## Statistical overview

The likelihood is defined as

$$
\begin{aligned}
\ell_p(\mu,\boldsymbol{\theta})
&= -\frac{1}{2}\sum_{i,j=1}^N
\left(y_i-\mu-\sum_{p=1} \Gamma_i^{p}\theta^i_p\right)W_{ij}^{-1}
\left(y_j-\mu-\sum_{p=1} \Gamma_j^{p}\theta^j_p\right) \\
&\quad -\frac{1}{2}\sum_{p=1}
\left(N+\frac{1}{2\varepsilon_p^2}\right)
\log\left[1 + 2\varepsilon_p^2 \sum_{i,j=1}^N \theta^i_p
\left(\rho^{(p)}\right)_{ij}^{-1} \theta^j_p\right] .
\end{aligned}
$$

Here, $\mu$ is the parameter of interest, and $\boldsymbol{\theta}$ denotes the nuisance
parameters associated with each systematic source $p$, for which the corresponding
error-on-error $\varepsilon_p$ is greater than zero. The coefficients $\Gamma_i^{p}$ quantify
the uncertainty induced by systematic source $p$ on measurement $i$, while $\rho^{(p)}$ is
the correlation matrix describing correlations among measurements due to source $p$.

All other sources of systematic uncertainty, together with the statistical errors,
are encoded in a BLUE-like covariance matrix defined as

$$
W_{ij}=V_{ij}+ \sum_{s\in\{\varepsilon_s=0\}} U_{ij}^{(s)} ,
$$

with

$$
U_{ij}^{(s)}=\rho^{(s)}_{ij}\Gamma_i^{s}\Gamma_j^{s}.
$$

Here, $V$ is the statistical covariance matrix of the measurements,
and $U^{(s)}_{ij}$ is the contribution from systematic source $s$, with
$\rho^{(s)}$ again the corresponding correlation matrix. The sum runs over
systematic sources with vanishing error-on-error $(\varepsilon_s = 0)$; their
nuisance parameters are profiled out, as in a BLUE-like combination.

Further details can be found in [arXiv:2407.05322](https://arxiv.org/abs/2407.05322).

## Bartlett correction

Profile likelihood ratios and goodness‑of‑fit (GOF) statistics can deviate from
their asymptotic chi‑square behaviour when error‑on‑error terms are present.
To improve accuracy without resorting to expensive profiling or toy MC, the toolkit
applies Bartlett correction factors computed analytically within the model.

- Confidence intervals: the profile LR, `q(mu)`, is compared against a
  Bartlett‑scaled threshold `b_profile × chi2_1(CL)`.
- Goodness of fit: the overall fit statistic is rescaled as
  `q* = q × (N−1) / b_chi2` and then interpreted against a `chi2_{N−1}`
  distribution.

The correction factors are computed automatically from the fitted point and the
model's information matrices; no user action is required.

## Installation

Python 3.11 or newer is required. Install the released package with:

```bash
python -m pip install gvm-combinations
```

To work from the repository, install it in editable mode:

```bash
python -m pip install -e .
```

This installs the `gvm_combinations` package and its dependencies: NumPy,
PyYAML, iminuit, SciPy and Matplotlib.

## Testing

For development, install the optional test and type-checking dependencies:

```bash
pip install -e ".[dev]"
python -m pytest
python -m mypy
python -m ruff check src/gvm_combinations scripts tests
```

## Quick start

```python
import matplotlib.pyplot as plt

from gvm_combinations import GVMCombination, build_input_data, plot_combination_summary

data = build_input_data("path/to/config.yaml")
comb = GVMCombination(data)
fit = comb.fit()

lo_68, hi_68, _ = comb.confidence_interval(cl_val=0.683)
lo_95, hi_95, _ = comb.confidence_interval(cl_val=0.955)
print(f"mu = {fit.mu:.4f}  68% CI = ({lo_68:.4f}, {hi_68:.4f})")

fig, ax = plt.subplots(figsize=(10, 6))
plot_combination_summary(ax, data, fit.mu, (lo_68, hi_68), (lo_95, hi_95))
fig.tight_layout()
```

## Configuration file

The combination is driven by a YAML configuration file with three main
sections:

* `global` – combination metadata. It must define `name`, `n_meas` and
  `n_syst`, giving the combination name and the expected numbers of
  measurements and systematic sources. The optional `matrix_dir` sets the
  base directory for correlation matrices and statistical covariance files.
  Relative file paths are resolved from `matrix_dir` when set, or from the
  YAML file's directory otherwise.
* `data` – the measurement names and central values together with their
  statistical uncertainties.  Statistical errors may be given explicitly or a
  `stat_cov_path` can provide a covariance matrix.  In either case the
  toolkit constructs the full covariance matrix.
* `syst` – list of systematics. Each has a `shift` block with `value`
  listing the shifts for each measurement and `correlation` specifying
  `diagonal`, `ones` or a path to a correlation matrix. An optional
  `error-on-error` block specifies the `value` and `type`
  (`dependent` or `independent`). For `independent` systematics the
  `value` may be either a list giving one epsilon per measurement or a
  single number which is applied to all measurements. If omitted, the value
  defaults to `0` and the type to `dependent`.

A minimal configuration:

```yaml
global:
  name: Example
  n_meas: 2
  n_syst: 1

data:
  measurements:
    - label: A
      central: 1.0
      stat_error: 0.1
    - label: B
      central: 1.2
      stat_error: 0.1

syst:
  - name: scale
    shift:
      value: [0.05, -0.02]
      correlation: diagonal
    error-on-error:
      value: 0.02
      type: dependent
```

More information can be found in the [toy tutorial](notebooks/toy/toy_tutorial.ipynb).

## Runs

The `scripts/` directory contains two reusable command-line drivers, while
`runs/` contains ready-to-use example cases with their inputs and generated
outputs:

* **`scripts/run_combination.py`** — baseline fit: prints the MLE, confidence
  intervals, and GOF; saves `results.yaml` and `summary.png`.
* **`scripts/scan_error_on_error.py`** — lambda scan: multiplies all
  error-on-error values by a global factor $\lambda$ and tracks how $\mu$, the
  CI, and the GOF evolve; saves `scan.png` and `gof.png`.

Both accept `--config` and `--output` (defaults: the single YAML in the current
run directory and its `output/` directory).  `scan_error_on_error.py` also
accepts:

* `--lambda-range MIN MAX` — scan range (default: `0.0 1.2`).
* `--n-points N` — number of scan points (default: `14`).

### Available examples

| Directory | Description |
|-----------|-------------|
| `toy_2meas` | Two incompatible measurements ($y=\pm2.5$) with correlated statistical and systematic uncertainties. Dependent error-on-error $\varepsilon=0.5$. |
| `toy_4meas` | Four measurements with one outlier ($y_1=16$). Independent error-on-error $\varepsilon=0.5$. |
| `top_mass_outlier` | LHC top-mass combination (15 measurements) plus a fictitious outlier at 174.5 GeV. Error-on-error $\varepsilon=0.5$ on the NEW systematic associated with the fake measurement. |

```bash
# Baseline fit
python scripts/run_combination.py --config runs/toy_2meas/toy2.yaml --output runs/toy_2meas/output
python scripts/run_combination.py --config runs/toy_4meas/toy4.yaml --output runs/toy_4meas/output
python scripts/run_combination.py --config runs/top_mass_outlier/LHC_comb_fictitious_meas.yaml --output runs/top_mass_outlier/output

# Lambda scan
python scripts/scan_error_on_error.py --config runs/toy_2meas/toy2.yaml --output runs/toy_2meas/output
python scripts/scan_error_on_error.py --config runs/toy_4meas/toy4.yaml --output runs/toy_4meas/output
python scripts/scan_error_on_error.py --config runs/top_mass_outlier/LHC_comb_fictitious_meas.yaml --output runs/top_mass_outlier/output

# Or cd into a run directory (auto-discovers its YAML, outputs to output/)
cd runs/toy_2meas
python ../../scripts/run_combination.py
python ../../scripts/scan_error_on_error.py
```

## Tutorials

The `notebooks/` directory contains educational material that explains and
showcases the model step by step.  Unlike the runs, which produce standalone
results, the notebooks are meant to be read interactively and walk the reader
through the theory, the API, and the interpretation of the outputs.

- [Toy GVM tutorial](notebooks/toy/toy_tutorial.ipynb) — Step-by-step introduction
  to the GVM using toy examples.
- [Top-mass combination tutorial](notebooks/top-mass/top_mass_combination.ipynb) —
  Application to the combination from [arXiv:2407.05322](https://arxiv.org/abs/2407.05322).

## Repository layout

```
GVM-Combinations/
├── .github/workflows/ci.yml   # Tests and type checking
├── pyproject.toml              # Package metadata & dependencies
├── PYPI_README.md              # PyPI package description
├── src/gvm_combinations/       # The statistical engine
│   ├── __init__.py
│   ├── combination.py          # GVMCombination class
│   ├── config.py               # YAML parsing & validation
│   ├── likelihood.py           # Log-likelihood construction
│   ├── fit_results.py          # FitResult dataclass
│   ├── plotting.py             # Summary plot
│   └── minuit_wrapper.py       # iMinuit interface
├── scripts/
│   ├── run_combination.py      # Generic fit script
│   └── scan_error_on_error.py  # Generic lambda scan script
├── tests/
│   └── test_mathematics.py     # Mathematical stability tests
├── runs/
│   ├── toy_2meas/              # 2-measurement example (incompatible, correlated)
│   │   ├── toy2.yaml           # Run configuration
│   │   ├── correlations/       # Correlation matrices
│   │   └── output/             # Generated results
│   ├── toy_4meas/              # 4-measurement example (outlier, independent)
│   │   ├── toy4.yaml           # Run configuration
│   │   └── output/             # Generated results
│   └── top_mass_outlier/       # Top-mass combination with fictitious outlier
│       ├── LHC_comb_fictitious_meas.yaml  # Run configuration
│       ├── correlations_fict/              # Correlation matrices
│       └── output/                         # Generated results
└── notebooks/
    ├── toy/                    # Introductory tutorial
    │   └── toy_tutorial.ipynb
    └── top-mass/               # Top-mass combination tutorial
        └── top_mass_combination.ipynb
```

## Companion paper

Further methodological details and the top-mass application are available in
[arXiv:2407.05322](https://arxiv.org/abs/2407.05322).

## Citation

If you use GVM Combinations, please cite the software using the metadata in
[`CITATION.cff`](CITATION.cff) and cite the accompanying paper.

## License

GVM Combinations is distributed under the [MIT License](LICENSE).
