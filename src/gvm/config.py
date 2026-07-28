"""Parse and validate combination configuration."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import yaml


ErrorOnError = Union[float, np.ndarray]
ErrorOnErrorType = Literal['dependent', 'independent']


@dataclass
class input_data:
    """Input data for a measurement combination."""

    name: str
    n_meas: int
    n_syst: int
    labels: list[str]
    measurements: dict[str, float]
    V_stat: np.ndarray
    syst: dict[str, np.ndarray]
    corr: dict[str, np.ndarray]
    eoe_type: dict[str, ErrorOnErrorType]
    uncertain_systematics: dict[str, ErrorOnError]


def build_input_data(path: str) -> input_data:
    """Build combination input data from a YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    yaml_dir = os.path.dirname(os.path.abspath(path))

    try:
        glob = data['global']
        corr_dir = glob.get('corr_dir', '')
        if corr_dir and not os.path.isabs(corr_dir):
            corr_dir = os.path.join(yaml_dir, corr_dir)
        name = glob['name']
        n_meas = int(glob['n_meas'])
        n_syst = int(glob['n_syst'])
    except KeyError as exc:
        raise KeyError('Global configuration must define "name", "n_meas" and "n_syst"') from exc

    try:
        combo = data['data']
        meas_entries = combo['measurements']
    except KeyError as exc:
        raise KeyError('Data configuration must define "measurements"') from exc

    labels: list[str] = []
    measurements: dict[str, float] = {}
    stat_err: list[float] = []
    for m in meas_entries:
        try:
            label = m['label']
        except KeyError as exc:
            raise KeyError('Each measurement requires a "label"') from exc
        try:
            cent = float(m['central'])
        except KeyError as exc:
            raise KeyError(f'Measurement "{label}" must define "central"') from exc
        labels.append(label)
        measurements[label] = cent
        if 'stat_error' in m:
            stat_err.append(float(m['stat_error']))

    stat_cov_path = combo.get('stat_cov_path')
    if stat_cov_path:
        stat_cov_path = stat_cov_path.replace('${global.corr_dir}', corr_dir)
        if not os.path.isabs(stat_cov_path):
            stat_cov_path = os.path.join(corr_dir, stat_cov_path)
        V_stat = np.loadtxt(stat_cov_path, dtype=float)
    elif stat_err:
        V_stat = np.diag(np.array(stat_err, dtype=float) ** 2)
    else:
        raise KeyError('Measurement stat errors or covariance required')

    try:
        syst_entries = data['syst']
    except KeyError as exc:
        raise KeyError('Configuration must define "syst" section') from exc

    meas_map = {m: i for i, m in enumerate(labels)}
    syst: dict[str, np.ndarray] = {}
    corr: dict[str, np.ndarray] = {}
    eoe_type: dict[str, ErrorOnErrorType] = {}
    uncertain_systematics: dict[str, ErrorOnError] = {}

    for item in syst_entries:
        sname = item['name']
        try:
            shift_vals = item['shift']['value']
        except KeyError as exc:
            raise KeyError(f'Systematic "{sname}" must define "shift.value"') from exc
        shifts = [float(x) for x in shift_vals]

        try:
            corr_spec = item['shift']['correlation']
        except KeyError as exc:
            raise KeyError(f'Systematic "{sname}" must define "shift.correlation"') from exc
        if corr_spec == 'diagonal':
            corr_mat = np.eye(n_meas)
        elif corr_spec == 'ones':
            corr_mat = np.ones((n_meas, n_meas))
        else:
            path_corr = corr_spec.replace('${global.corr_dir}', corr_dir)
            if not os.path.isabs(path_corr):
                path_corr = os.path.join(corr_dir, path_corr)
            corr_mat = np.loadtxt(path_corr, dtype=float)

        eoe = item.get('error-on-error', {})
        eps_val = eoe.get('value', 0.0)
        eps_typ = eoe.get('type', 'dependent')
        if eps_typ not in ('dependent', 'independent'):
            raise ValueError(f'Systematic "{sname}" has unrecognised error-on-error type "{eps_typ}"')

        if eps_typ == 'independent':
            if isinstance(eps_val, (list, tuple, np.ndarray)):
                eps_list = [float(x) for x in eps_val]
            else:
                eps_list = [float(eps_val)]
            if len(eps_list) == 1:
                eps_list *= n_meas
            eps_val = eps_list
        else:
            eps_val = float(eps_val)

        val_map = {lab: shifts[meas_map[lab]] for lab in labels}
        syst[sname] = np.array([val_map[m] for m in labels], dtype=float)
        corr[sname] = np.asarray(corr_mat, dtype=float)
        eoe_type[sname] = eps_typ

        if eps_typ == 'independent':
            eps = np.asarray(eps_val, dtype=float)
            sigma = syst[sname]
            mask = sigma != 0.0
            eps = eps[mask]
            if eps.size > 0 and np.any(eps != 0.0):
                uncertain_systematics[sname] = eps
        else:
            epsf = float(eps_val)
            if epsf != 0.0:
                uncertain_systematics[sname] = epsf

    return input_data(
        name=name,
        n_meas=n_meas,
        n_syst=n_syst,
        labels=labels,
        measurements=measurements,
        V_stat=V_stat,
        syst=syst,
        corr=corr,
        eoe_type=eoe_type,
        uncertain_systematics=uncertain_systematics,
    )


def validate_input_data(input_data: input_data) -> None:
    """Validate the consistency of parsed input data."""
    meas_names = list(input_data.measurements)
    if len(meas_names) != input_data.n_meas:
        raise ValueError(f'Expected {input_data.n_meas} measurements, got {len(meas_names)}')

    if len(input_data.syst) != input_data.n_syst:
        raise ValueError(f'Expected {input_data.n_syst} systematics, got {len(input_data.syst)}')

    if input_data.V_stat.shape != (input_data.n_meas, input_data.n_meas):
        raise ValueError(f'Stat covariance must be {input_data.n_meas}x{input_data.n_meas}')
        
    diff = np.argwhere(~np.isclose(input_data.V_stat, input_data.V_stat.T, rtol=1e-7, atol=1e-8))
    for i, j in diff:
        if i < j:
            warnings.warn(
                f'Stat covariance asymmetric for measurements '
                f'{meas_names[i]} and {meas_names[j]}: '
                f'{input_data.V_stat[i, j]} vs {input_data.V_stat[j, i]}')
    
    for name, arr in input_data.syst.items():
        if arr.shape[0] != input_data.n_meas:
            raise ValueError(f'Systematic {name} must have {input_data.n_meas} values')

    if len(input_data.corr) != input_data.n_syst:
        raise ValueError(f'Expected {input_data.n_syst} correlation matrices, got {len(input_data.corr)}')

    for name, mat in input_data.corr.items():
        if mat.shape != (input_data.n_meas, input_data.n_meas):
            raise ValueError(f'Correlation matrix {name} must be {input_data.n_meas}x{input_data.n_meas}')
        diff = np.argwhere(~np.isclose(mat, mat.T, rtol=1e-7, atol=1e-8))
        for i, j in diff:
            if i < j:
                warnings.warn(
                    f'Correlation matrix "{name}" asymmetric for measurements '
                    f'{meas_names[i]} and {meas_names[j]}: '
                    f'{mat[i, j]} vs {mat[j, i]}')
        # Independent error-on-error terms require a diagonal correlation matrix.
        if input_data.eoe_type.get(name, 'dependent') == 'independent':
            if not np.allclose(mat, np.eye(input_data.n_meas)):
                raise ValueError(
                    f'Systematic {name} has independent error-on-error but correlation is not diagonal')

    # Dependent systematics use a single epsilon.
    for name, typ in input_data.eoe_type.items():
        if typ != 'dependent':
            continue
        if name in input_data.uncertain_systematics:
            eps_val = input_data.uncertain_systematics[name]
            if isinstance(eps_val, (list, tuple, np.ndarray)):
                raise ValueError(
                    f"Systematic {name} has dependent error-on-error but epsilon is not a single number")
            try:
                epsf = float(eps_val)
            except Exception:
                epsf = None
            if epsf is not None and epsf == 0.0:
                input_data.uncertain_systematics.pop(name, None)
                warnings.warn(
                    f"Systematic '{name}' has epsilon 0.0; removing from uncertain_systematics.")

    # Independent systematics use one epsilon per active shift.
    for name, typ in input_data.eoe_type.items():
        if typ != 'independent':
            continue
        expected = np.count_nonzero(input_data.syst[name])
        if name in input_data.uncertain_systematics:
            val = input_data.uncertain_systematics[name]
            if not isinstance(val, (list, tuple, np.ndarray)):
                input_data.uncertain_systematics[name] = np.repeat(float(val), expected)
        eps_raw = input_data.uncertain_systematics.get(name, np.zeros(expected))
        eps = np.asarray(eps_raw, dtype=float)
        if eps.shape[0] != expected:
            if eps.shape[0] == input_data.n_meas:
                mask = input_data.syst[name] != 0.0
                eps = eps[mask]
                input_data.uncertain_systematics[name] = eps
                warnings.warn(
                    f"Systematic '{name}' epsilon vector included zero-shift entries; dropping them to match active components.")
            else:
                raise ValueError(
                    f"Systematic {name} has independent error-on-error but epsilon has {eps.shape[0]} values (expected {expected} or {input_data.n_meas})")
        if expected == 0 or (eps.size == expected and not np.any(eps != 0.0)):
            input_data.uncertain_systematics.pop(name, None)
            warnings.warn(
                f"Systematic '{name}' has all-zero epsilons; removing from uncertain_systematics.")
