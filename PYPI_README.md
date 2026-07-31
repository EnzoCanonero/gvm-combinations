# GVM Combinations

`gvm-combinations` provides tools for combining correlated measurements with
errors-on-errors using the Gamma Variance Model.

It supports:

- statistical covariance matrices and correlated systematic uncertainties;
- dependent and independent errors-on-errors;
- likelihood fits with iminuit;
- Bartlett-corrected confidence intervals and goodness-of-fit statistics;
- YAML configuration files and combination summary plots.

## Installation

The package requires Python 3.11 or newer.

```bash
python -m pip install gvm-combinations
```

## Example

```python
from gvm_combinations import GVMCombination, build_input_data

data = build_input_data("combination.yaml")
combination = GVMCombination(data)
fit = combination.fit()
low, high, _ = combination.confidence_interval()

print(f"mu = {fit.mu:.4f}")
print(f"68.3% CI = ({low:.4f}, {high:.4f})")
```

The public Python namespace is `gvm_combinations`. Combination inputs are
defined in YAML files containing the measurements, statistical covariance and
systematic uncertainty model.

## Documentation and examples

The [GitHub repository](https://github.com/EnzoCanonero/GVM-Combinations)
contains the full mathematical overview, configuration reference, example
runs and tutorials:

- [Toy GVM tutorial](https://github.com/EnzoCanonero/GVM-Combinations/blob/main/notebooks/toy/toy_tutorial.ipynb)
- [Top-mass combination tutorial](https://github.com/EnzoCanonero/GVM-Combinations/blob/main/notebooks/top-mass/top_mass_combination.ipynb)

Methodological details are available in
[arXiv:2407.05322](https://arxiv.org/abs/2407.05322).

## Citation

Citation metadata are provided in
[`CITATION.cff`](https://github.com/EnzoCanonero/GVM-Combinations/blob/main/CITATION.cff).

## License

GVM Combinations is distributed under the
[MIT License](https://github.com/EnzoCanonero/GVM-Combinations/blob/main/LICENSE).
