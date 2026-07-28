"""GVM fitting, intervals and goodness-of-fit."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Optional

import numpy as np
from scipy.stats import norm

from .config import (
    ErrorOnError,
    ErrorOnErrorType,
    input_data as InputData,
    validate_input_data,
)
from .fit_results import FitResult
from .likelihood import nll as _nll_fn, bartlett_correction as _bartlett_correction_fn
from .minuit_wrapper import minimize as _minimize


class GVMCombination:
    """Combine correlated measurements with the Gamma Variance Model."""

    def __init__(self, data: InputData) -> None:
        validate_input_data(data)

        self._input_data: InputData = data

        self.V_inv: np.ndarray
        self.C_inv: dict[str, np.ndarray] = {}
        self.Gamma: dict[str, np.ndarray] = {}
        self.prepare()

        self.fit_results: Optional[FitResult] = None
    
    # Input data
    
    @property
    def name(self) -> str:
        return self._input_data.name

    @property
    def n_meas(self) -> int:
        return self._input_data.n_meas

    @property
    def n_syst(self) -> int:
        return self._input_data.n_syst

    @property
    def measurements(self) -> dict[str, float]:
        return self._input_data.measurements

    @property
    def V_stat(self) -> np.ndarray:
        return self._input_data.V_stat

    @property
    def syst(self) -> dict[str, np.ndarray]:
        return self._input_data.syst

    @property
    def corr(self) -> dict[str, np.ndarray]:
        return self._input_data.corr

    @property
    def eoe_type(self) -> dict[str, ErrorOnErrorType]:
        return self._input_data.eoe_type

    @property
    def uncertain_systematics(self) -> dict[str, ErrorOnError]:
        return self._input_data.uncertain_systematics

    def get_input_data(self, copy: bool = False) -> InputData:
        """Return the input data, optionally as a shallow copy with copied arrays."""
        if not copy:
            return self._input_data
        s = self._input_data
        return replace(
            s,
            measurements=dict(s.measurements),
            V_stat=np.array(s.V_stat, copy=True),
            syst={k: np.array(v, copy=True) for k, v in s.syst.items()},
            corr={k: np.array(v, copy=True) for k, v in s.corr.items()},
            eoe_type=dict(s.eoe_type),
            uncertain_systematics={k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                                   for k, v in s.uncertain_systematics.items()},
        )

    def set_input_data(self, data: InputData, refit: bool = True) -> GVMCombination:
        """Replace the input data, rebuild the matrices and optionally refit."""
        validate_input_data(data)
        self._input_data = data
        self.V_inv, self.C_inv, self.Gamma = self._compute_likelihood_matrices()
        self.fit_results = None
        if refit:
            self.fit_results = self.minimize()
        return self

    # Likelihood matrices
    
    def prepare(self) -> GVMCombination:
        """Validate the input data and rebuild the likelihood matrices."""
        validate_input_data(self._input_data)
        self.V_inv, self.C_inv, self.Gamma = self._compute_likelihood_matrices()
        return self
    
    def _compute_likelihood_matrices(
        self,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Build likelihood matrices, dropping nuisance parameters with zero shifts."""
        n = len(self.measurements)
        V_stat = self.V_stat
        V_syst = np.zeros((n, n))
        for src, rho in self.corr.items():
            if src not in self.uncertain_systematics:
                sigma = self.syst[src]
                V_syst += np.outer(sigma, sigma) * rho
        V_blue = V_stat + V_syst
        V_inv = np.linalg.inv(V_blue)

        C_inv: dict[str, np.ndarray] = {}
        Gamma_factors: dict[str, np.ndarray] = {}
        for src, sigma in self.syst.items():
            if src in self.uncertain_systematics:
                rho = self.corr[src]
                red, Gamma = self._reduce_corr(rho, src_name=src)
                for i in range(Gamma.shape[0]):
                    for j in range(Gamma.shape[1]):
                        if Gamma[i, j] != 0:
                            Gamma[i, j] *= sigma[i]
                zero_cols = np.all(Gamma == 0, axis=0)
                Gamma = Gamma[:, ~zero_cols]
                if np.any(zero_cols):
                    red = red[~zero_cols][:, ~zero_cols]
                C_inv[src] = np.linalg.inv(red)
                Gamma_factors[src] = Gamma
        return V_inv, C_inv, Gamma_factors
    
    def _reduce_corr(
        self,
        rho: np.ndarray,
        src_name: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collapse fully correlated or anticorrelated entries into one nuisance parameter."""
        n = rho.shape[0]
        groups = []
        visited = set()
        for i in range(n):
            if i not in visited:
                group = [i]
                for j in range(i + 1, n):
                    if abs(rho[i, j]) == 1:
                        group.append(j)
                        visited.add(j)
                groups.append(group)
                visited.add(i)

        rsize = len(groups)
        reduced = np.zeros((rsize, rsize))
        Gamma = np.zeros((n, rsize))

        for new_i, group in enumerate(groups):
            for j in group:
                sign = 1.0
                if rho[group[0], j] == -1:
                    sign = -1.0
                Gamma[j, new_i] = sign

        for new_i, gi in enumerate(groups):
            for new_j, gj in enumerate(groups):
                vec = [rho[i, j] * Gamma[i, new_i] * Gamma[j, new_j]
                       for i in gi for j in gj]
                reduced[new_i, new_j] = np.mean(vec)

        eig = np.linalg.eigvalsh(reduced)
        m = eig.min()
        if m <= 0:
            offset = abs(m) + 0.01
            np.fill_diagonal(reduced, reduced.diagonal() + offset)
            if src_name:
                warnings.warn(
                    f'Negative eigenvalue {m:.4e} in systematic "{src_name}"; '
                    f'adding {offset:.4e} to diagonal for regularisation.')
        return reduced, Gamma
    
    # Fitting

    def minimize(
        self,
        fixed: Optional[Mapping[str, float]] = None,
        update: bool = True,
    ) -> FitResult:
        """Minimise the likelihood, optionally fixing parameters and storing the result."""
        fixed_params: Mapping[str, float] = fixed or {}

        names = ['mu']
        for key in self.Gamma:
            for j in range(self.Gamma[key].shape[1]):
                names.append(f'{key}_{j}')

        y_vals = np.fromiter(self.measurements.values(), dtype=float)
        initial = [np.mean(y_vals)] + [0.] * (len(names) - 1)

        free_idx = []
        free_names = []
        x0 = []
        for i, n in enumerate(names):
            if n not in fixed_params:
                free_idx.append(i)
                free_names.append(n)
                x0.append(initial[i])

        # Evaluate directly when every parameter is fixed.
        if len(x0) == 0:
            params = list(initial)
            for n, val in fixed_params.items():
                params[names.index(n)] = val
            mu = params[0]
            theta_flat = params[1:]
            thetas = []
            i = 0
            for key in self.Gamma:
                npar = self.Gamma[key].shape[1]
                thetas.append(np.array(theta_flat[i:i+npar]))
                i += npar
            nll_val = _nll_fn(self, mu, *thetas)
            result = FitResult(
                mu=mu,
                thetas=np.array(theta_flat),
                nll=nll_val,
            )
            if update:
                self.fit_results = result
            return result

        def f(arr: Sequence[float]) -> float:
            params = list(initial)
            for val, idx in zip(arr, free_idx):
                params[idx] = val
            for n, val in fixed_params.items():
                params[names.index(n)] = val
            mu = params[0]
            theta_flat = params[1:]
            thetas = []
            i = 0
            for key in self.Gamma:
                npar = self.Gamma[key].shape[1]
                thetas.append(np.array(theta_flat[i:i+npar]))
                i += npar
            return _nll_fn(self, mu, *thetas)

        m = _minimize(f, x0, free_names, errordef=0.5)

        values = dict(zip(names, initial))
        for val, idx in zip(m.values, free_idx):
            values[names[idx]] = val
        for n, v in fixed_params.items():
            values[n] = v

        result = FitResult(
            mu=values['mu'],
            thetas=np.array([values[n] for n in names[1:]]),
            nll=m.fval,
        )
        if update:
            self.fit_results = result
        return result
    
    def fit(
        self,
        fixed: Optional[Mapping[str, float]] = None,
        update: bool = True,
    ) -> FitResult:
        """Prepare the model if needed, then minimise the likelihood."""
        if not self.Gamma:
            self.prepare()
        return self.minimize(fixed=fixed, update=update)
    
    # Confidence intervals

    def likelihood_ratio(self, mu: float) -> float:
        """Return the profile likelihood-ratio statistic at mu."""
        best = self.fit_results or self.minimize()
        nll_best = best.nll
        res_mu = self.minimize(fixed={'mu': mu}, update=False)
        nll_mu = res_mu.nll
        return 2 * (nll_mu - nll_best)

    def confidence_interval(
        self,
        step: float = 0.01,
        tol: float = 0.001,
        max_iter: int = 1000,
        cl_val: float = 0.683,
    ) -> tuple[float, float, float]:
        """Return the Bartlett-corrected interval as (lower, upper, half_width)."""
        fit = self.fit_results or self.minimize()
        b_profile, _ = _bartlett_correction_fn(self)
        thr = b_profile * (norm.ppf(0.5 * (1.0 + cl_val)) ** 2)
        mu_hat = fit.mu
        q0 = self.likelihood_ratio(mu_hat)
        up = mu_hat
        q_up = q0
        down = mu_hat
        q_down = q0
        it = 0
        while q_up <= thr and it < max_iter:
            up += step
            q_up = self.likelihood_ratio(up)
            it += 1
        it = 0
        while q_down <= thr and it < max_iter:
            down -= step
            q_down = self.likelihood_ratio(down)
            it += 1
        step /= 2
        it = 0
        while abs(q_up - thr) > tol and it < max_iter:
            if q_up > thr:
                up -= step
            else:
                up += step
            q_up = self.likelihood_ratio(up)
            step /= 2
            it += 1
        step = step if step > 0 else 0.001
        it = 0
        while abs(q_down - thr) > tol and it < max_iter:
            if q_down > thr:
                down += step
            else:
                down -= step
            q_down = self.likelihood_ratio(down)
            step /= 2
            it += 1
        return down, up, 0.5*(up - down)
    
    # Goodness of fit

    def goodness_of_fit(self) -> float:
        """Return the Bartlett-corrected goodness-of-fit statistic."""
        fit = self.fit_results or self.minimize()
        mu = fit.mu

        # Split the flat nuisance-parameter array by systematic source.
        theta_flat = np.asarray(fit.thetas)
        keys = list(self.C_inv)
        sizes = [self.C_inv[key].shape[0] for key in keys]
        idx = np.cumsum([0] + sizes)
        thetas = [theta_flat[idx[i]:idx[i + 1]] for i in range(len(keys))]

        q = 2 * _nll_fn(self, mu, *thetas)
        _, b_chi2 = _bartlett_correction_fn(self)
        return q * (len(self.measurements) - 1) / b_chi2
