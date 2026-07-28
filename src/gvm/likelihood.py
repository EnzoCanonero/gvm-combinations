"""Likelihood calculations and Bartlett corrections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .combination import GVMCombination


def nll(comb: GVMCombination, mu: float, *thetas: np.ndarray) -> float:
    """Return the negative log-likelihood for a combination."""
    theta_values = list(thetas)

    adj = np.sum([comb.Gamma[k] @ theta_values[i]
                  for i, k in enumerate(comb.Gamma)], axis=0) if theta_values else 0
    y_vals = np.fromiter(comb.measurements.values(), dtype=float)
    v = y_vals - mu - adj
    chi2_y = float(v @ comb.V_inv @ v)

    chi2_u = 0.0
    keys = list(comb.Gamma.keys())
    for i, k in enumerate(keys):
        theta = np.asarray(theta_values[i])
        eps_value = comb.uncertain_systematics[k]
        if comb.eoe_type.get(k, 'dependent') == 'dependent':
            eps_scalar = float(eps_value)
            Cinv = comb.C_inv[k]
            N_s = len(theta)
            if eps_scalar > 0:
                chi2_u += float(
                    (N_s + 1.0 / (2.0 * eps_scalar ** 2))
                    * np.log(
                        1. + 2. * eps_scalar ** 2 * theta @ Cinv @ theta
                    )
                )
            else:
                chi2_u += float(theta @ Cinv @ theta)
        else:
            eps_array = np.asarray(eps_value)
            mask = eps_array > 0
            if np.any(mask):
                chi2_u += float(
                    np.sum(
                        (1 + 1.0 / (2.0 * eps_array[mask] ** 2))
                        * np.log(
                            1. + 2. * eps_array[mask] ** 2 * theta[mask] ** 2
                        )
                    )
                )
            if np.any(~mask):
                chi2_u += float(np.sum(theta[~mask] ** 2))
    return float(0.5 * (chi2_y + chi2_u))


def compute_FIM(
    comb: GVMCombination,
    S: Sequence[np.ndarray],
) -> np.ndarray:
    """Return the Fisher information matrix."""
    keys = list(comb.C_inv.keys())
    sizes = [comb.C_inv[k].shape[0] for k in keys]

    tot = sum(sizes)
    F = np.zeros((1 + tot, 1 + tot))
    F[0, 0] = np.sum(comb.V_inv)
    start_idx = np.cumsum([0] + sizes[:-1])
    idxs = [np.arange(sz) + s + 1 for sz, s in zip(sizes, start_idx)]

    V_G = {k: comb.V_inv @ comb.Gamma[k] for k in keys}
    for i, k in enumerate(keys):
        idx = idxs[i]
        F[0, idx] = F[idx, 0] = V_G[k].sum(axis=0)

    for i, ks in enumerate(keys):
        idx_s = idxs[i]
        Gs = comb.Gamma[ks]
        Cinv_s = comb.C_inv[ks]
        S_s = S[i]
        for j, kp in enumerate(keys):
            idx_p = idxs[j]
            GsVinGp = Gs.T @ V_G[kp]
            if ks == kp:
                if comb.eoe_type.get(kp, 'dependent') == 'dependent':
                    scale = float(S_s)
                    F[np.ix_(idx_s, idx_p)] = (
                        GsVinGp + (1.0 / scale) * Cinv_s
                    )
                else:
                    F[np.ix_(idx_s, idx_p)] = (
                        GsVinGp + Cinv_s * (1.0 / S_s)[:, None]
                    )
            else:
                F[np.ix_(idx_s, idx_p)] = GsVinGp
    return F


def bartlett_correction(comb: GVMCombination) -> tuple[float, float]:
    """Return profile-likelihood and goodness-of-fit Bartlett corrections."""
    if len(comb.C_inv) == 0:
        return 1.0, float(len(comb.measurements) - 1)

    fit = comb.fit_results or comb.minimize()
    theta_values = fit.thetas
    keys = list(comb.C_inv.keys())
    sizes = [comb.C_inv[k].shape[0] for k in keys]
    idx = np.cumsum([0] + sizes)
    thetas = [
        np.asarray(theta_values[idx[i]:idx[i+1]])
        for i in range(len(keys))
    ]
    eps = [np.asarray(comb.uncertain_systematics[k], dtype=float) for k in keys]
    C_inv_list = [comb.C_inv[k] for k in keys]
    N_s = sizes
    S: list[np.ndarray] = []
    for th, k, C, e, N in zip(thetas, keys, C_inv_list, eps, N_s):
        if comb.eoe_type.get(k, 'dependent') == 'dependent':
            eps_scalar = float(e)
            scale = (
                (1 + 2 * eps_scalar ** 2 * th @ C @ th)
                / (1 + 2 * eps_scalar ** 2 * N)
            )
        else:
            scale = (1 + 2 * e ** 2 * th ** 2) / (1 + 2 * e ** 2)
        S.append(np.asarray(scale, dtype=float))
    F = compute_FIM(comb, S)
    W_full = np.linalg.inv(F)[1:, 1:]
    W_theta = np.linalg.inv(F[1:, 1:])
    start_idx = idx[:-1]
    b_lik = b_theta = b_chi2 = 0.0
    for s, (th, Cinv, e, N, S_s) in enumerate(zip(thetas, C_inv_list, eps, N_s, S)):
        si = start_idx[s]
        ei = si + N
        W_s = W_full[si:ei, si:ei]
        Wt_s = W_theta[si:ei, si:ei]
        if comb.eoe_type.get(keys[s], 'dependent') == 'dependent':
            eps_scalar = float(e)
            scalar_scale = float(S_s)
            trWC = np.trace(W_s @ Cinv)
            trWCWC = np.trace(W_s @ Cinv @ W_s @ Cinv)
            trW_t_C = np.trace(Wt_s @ Cinv)
            trW_t_CWC = np.trace(Wt_s @ Cinv @ Wt_s @ Cinv)
            b_lik += float(
                (4 * eps_scalar ** 2 / scalar_scale) * trWC
                - (2 * eps_scalar ** 2 / scalar_scale ** 2) * trWCWC
                + (eps_scalar ** 2 / scalar_scale ** 2) * (trWC ** 2)
            )
            b_theta += float(
                (4 * eps_scalar ** 2 / scalar_scale) * trW_t_C
                - (2 * eps_scalar ** 2 / scalar_scale ** 2) * trW_t_CWC
                + (eps_scalar ** 2 / scalar_scale ** 2) * (trW_t_C ** 2)
            )
            b_chi2 += float((2 * N + N ** 2) * eps_scalar ** 2)
        else:
            diag_W = np.diag(W_s)
            diag_W_sq = diag_W ** 2
            diag_Wt = np.diag(Wt_s)
            diag_Wt_sq = diag_Wt ** 2
            e2 = e ** 2
            b_lik += float(np.sum(
                (4 * e2 / S_s) * diag_W
                - (e2 / S_s ** 2) * diag_W_sq
            ))
            b_theta += float(np.sum(
                (4 * e2 / S_s) * diag_Wt
                - (e2 / S_s ** 2) * diag_Wt_sq
            ))
            b_chi2 += float(np.sum(3 * e2))

    b_profile = 1 + b_lik - b_theta
    b_chi2 = float(len(comb.measurements) - 1) + b_chi2 - b_lik
    return float(b_profile), float(b_chi2)
