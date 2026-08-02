from typing import Literal, Optional, Union

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

from gvm_combinations import GVMCombination, InputData


def _blue_data() -> InputData:
    return InputData(
        name="blue",
        n_meas=2,
        n_syst=0,
        labels=["a", "b"],
        measurements={"a": 1.0, "b": 3.0},
        V_stat=np.diag([1.0, 4.0]),
        syst={},
        corr={},
        eoe_type={},
        uncertain_systematics={},
    )


def _blue_with_systematic(
    epsilon: Optional[float],
    eoe_type: Literal["dependent", "independent"] = "independent",
) -> InputData:
    uncertain: dict[str, Union[float, np.ndarray]] = {}
    if epsilon is not None:
        uncertain = {
            "scale": epsilon if eoe_type == "dependent" else np.full(2, epsilon)
        }
    return InputData(
        name="blue_with_systematic",
        n_meas=2,
        n_syst=1,
        labels=["a", "b"],
        measurements={"a": 1.0, "b": 3.0},
        V_stat=np.diag([1.0, 4.0]),
        syst={"scale": np.array([0.5, 1.0])},
        corr={"scale": np.eye(2)},
        eoe_type={"scale": eoe_type},
        uncertain_systematics=uncertain,
    )


def _structured_data(
    offset: float = 0.0,
    order: Optional[np.ndarray] = None,
) -> InputData:
    labels = np.array(["y1", "y2", "y3", "y4"])
    values = np.array([16.0, 10.5, 9.5, 9.0]) + offset
    stat = np.array([1.0, 0.8, 1.2, 0.9])
    shifts = np.array([1.0, 0.7, 1.1, 0.9])
    epsilon = np.array([0.5, 0.3, 0.4, 0.2])
    if order is not None:
        labels = labels[order]
        values = values[order]
        stat = stat[order]
        shifts = shifts[order]
        epsilon = epsilon[order]
    label_list = labels.tolist()
    return InputData(
        name="structured",
        n_meas=4,
        n_syst=1,
        labels=label_list,
        measurements=dict(zip(label_list, values.tolist())),
        V_stat=np.diag(stat ** 2),
        syst={"scale": shifts},
        corr={"scale": np.eye(4)},
        eoe_type={"scale": "independent"},
        uncertain_systematics={"scale": epsilon},
    )


def _toy_outlier_data() -> InputData:
    labels = ["y1", "y2", "y3", "y4"]
    return InputData(
        name="toy4_outlier",
        n_meas=4,
        n_syst=1,
        labels=labels,
        measurements=dict(zip(labels, [16.0, 10.5, 9.5, 9.0])),
        V_stat=np.eye(4),
        syst={"sys1": np.ones(4)},
        corr={"sys1": np.eye(4)},
        eoe_type={"sys1": "independent"},
        uncertain_systematics={"sys1": np.full(4, 0.5)},
    )


def test_blue_limit_matches_analytic_result() -> None:
    combination = GVMCombination(_blue_data())
    fit = combination.fit()
    cl_one_sigma = float(2.0 * norm.cdf(1.0) - 1.0)
    low, high, half_width = combination.confidence_interval(cl_val=cl_one_sigma)

    assert_allclose(fit.mu, 1.4, atol=1e-8)
    assert_allclose(fit.nll, 0.4, atol=1e-10)
    assert_allclose(combination.goodness_of_fit(), 0.8, atol=1e-10)
    expected_interval = [1.4 - np.sqrt(0.8), 1.4 + np.sqrt(0.8)]
    assert_allclose([low, high], expected_interval, atol=2e-3)
    assert_allclose(half_width, np.sqrt(0.8), atol=2e-3)


def test_tiny_error_on_error_recovers_blue_limit() -> None:
    nominal = GVMCombination(_blue_with_systematic(None))
    nominal_fit = nominal.fit()
    nominal_gof = nominal.goodness_of_fit()

    for limiting_data in (
        _blue_with_systematic(1e-8, "dependent"),
        _blue_with_systematic(1e-8, "independent"),
    ):
        limiting = GVMCombination(limiting_data)
        limiting_fit = limiting.fit()
        assert_allclose(limiting_fit.mu, nominal_fit.mu, atol=1e-7)
        assert_allclose(limiting_fit.nll, nominal_fit.nll, atol=1e-9)
        assert_allclose(limiting.goodness_of_fit(), nominal_gof, atol=1e-8)


def test_toy_outlier_regression() -> None:
    combination = GVMCombination(_toy_outlier_data())
    fit = combination.fit()
    low, high, half_width = combination.confidence_interval()

    assert_allclose(fit.mu, 9.95969888, atol=1e-5)
    assert_allclose(fit.nll, 4.73848825, atol=1e-6)
    assert_allclose(combination.goodness_of_fit(), 6.92523951, atol=1e-5)
    expected_interval = [9.06673013, 10.87880044, 0.90603516]
    assert_allclose([low, high, half_width], expected_interval, atol=2e-3)


def test_fit_is_translation_invariant() -> None:
    base = GVMCombination(_structured_data())
    shifted = GVMCombination(_structured_data(offset=100.0))
    base_fit = base.fit()
    shifted_fit = shifted.fit()

    assert_allclose(shifted_fit.mu, base_fit.mu + 100.0, atol=1e-6)
    assert_allclose(shifted_fit.nll, base_fit.nll, atol=1e-8)
    assert_allclose(shifted.goodness_of_fit(), base.goodness_of_fit(), atol=1e-7)


def test_fit_is_permutation_invariant() -> None:
    base = GVMCombination(_structured_data())
    permuted = GVMCombination(_structured_data(order=np.array([2, 0, 3, 1])))
    base_fit = base.fit()
    permuted_fit = permuted.fit()

    assert_allclose(permuted_fit.mu, base_fit.mu, atol=1e-7)
    assert_allclose(permuted_fit.nll, base_fit.nll, atol=1e-8)
    assert_allclose(permuted.goodness_of_fit(), base.goodness_of_fit(), atol=1e-7)
