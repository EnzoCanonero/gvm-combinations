"""Parse and validate combination configuration."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import numpy as np
import yaml

ErrorOnError = Union[float, np.ndarray]
ErrorOnErrorType = Literal['dependent', 'independent']


@dataclass
class InputData:
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


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_input_data(path: str) -> InputData:
    """Build combination input data from a YAML file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open('r') as f:
        data = yaml.safe_load(f)

    yaml_dir = config_path.parent

    try:
        glob = data['global']
        matrix_dir = glob.get('matrix_dir')
        if matrix_dir is None and 'corr_dir' in glob:
            matrix_dir = glob['corr_dir']
            warnings.warn(
                '"global.corr_dir" is deprecated; use "global.matrix_dir" instead',
                DeprecationWarning,
                stacklevel=2,
            )
        matrix_root = _resolve_path(matrix_dir, yaml_dir) if matrix_dir else yaml_dir
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
        V_stat = np.loadtxt(_resolve_path(stat_cov_path, matrix_root), dtype=float)
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
            corr_mat = np.loadtxt(_resolve_path(corr_spec, matrix_root), dtype=float)

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

    return InputData(
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


def validate_input_data(data: InputData) -> None:
    """Validate the consistency of parsed input data."""
    meas_names = list(data.measurements)
    if len(meas_names) != data.n_meas:
        raise ValueError(f'Expected {data.n_meas} measurements, got {len(meas_names)}')

    if len(data.syst) != data.n_syst:
        raise ValueError(f'Expected {data.n_syst} systematics, got {len(data.syst)}')

    if data.V_stat.shape != (data.n_meas, data.n_meas):
        raise ValueError(f'Stat covariance must be {data.n_meas}x{data.n_meas}')
        
    diff = np.argwhere(~np.isclose(data.V_stat, data.V_stat.T, rtol=1e-7, atol=1e-8))
    for i, j in diff:
        if i < j:
            warnings.warn(
                f'Stat covariance asymmetric for measurements '
                f'{meas_names[i]} and {meas_names[j]}: '
                f'{data.V_stat[i, j]} vs {data.V_stat[j, i]}')
    
    for name, arr in data.syst.items():
        if arr.shape[0] != data.n_meas:
            raise ValueError(f'Systematic {name} must have {data.n_meas} values')

    if len(data.corr) != data.n_syst:
        raise ValueError(f'Expected {data.n_syst} correlation matrices, got {len(data.corr)}')

    for name, mat in data.corr.items():
        if mat.shape != (data.n_meas, data.n_meas):
            raise ValueError(f'Correlation matrix {name} must be {data.n_meas}x{data.n_meas}')
        diff = np.argwhere(~np.isclose(mat, mat.T, rtol=1e-7, atol=1e-8))
        for i, j in diff:
            if i < j:
                warnings.warn(
                    f'Correlation matrix "{name}" asymmetric for measurements '
                    f'{meas_names[i]} and {meas_names[j]}: '
                    f'{mat[i, j]} vs {mat[j, i]}')
        # Independent error-on-error terms require a diagonal correlation matrix.
        if data.eoe_type.get(name, 'dependent') == 'independent':
            if not np.allclose(mat, np.eye(data.n_meas)):
                raise ValueError(
                    f'Systematic {name} has independent error-on-error but correlation is not diagonal')

    # Dependent systematics use a single epsilon.
    for name, typ in data.eoe_type.items():
        if typ != 'dependent':
            continue
        if name in data.uncertain_systematics:
            eps_val = data.uncertain_systematics[name]
            if isinstance(eps_val, (list, tuple, np.ndarray)):
                raise ValueError(
                    f"Systematic {name} has dependent error-on-error but epsilon is not a single number")
            try:
                epsf = float(eps_val)
            except Exception:
                epsf = None
            if epsf is not None and epsf == 0.0:
                data.uncertain_systematics.pop(name, None)
                warnings.warn(
                    f"Systematic '{name}' has epsilon 0.0; removing from uncertain_systematics.")

    # Independent systematics use one epsilon per active shift.
    for name, typ in data.eoe_type.items():
        if typ != 'independent':
            continue
        if name not in data.uncertain_systematics:
            continue
        expected = np.count_nonzero(data.syst[name])
        val = data.uncertain_systematics[name]
        if not isinstance(val, (list, tuple, np.ndarray)):
            data.uncertain_systematics[name] = np.repeat(float(val), expected)
        eps_raw = data.uncertain_systematics[name]
        eps = np.asarray(eps_raw, dtype=float)
        if eps.shape[0] != expected:
            if eps.shape[0] == data.n_meas:
                mask = data.syst[name] != 0.0
                eps = eps[mask]
                data.uncertain_systematics[name] = eps
                warnings.warn(
                    f"Systematic '{name}' epsilon vector included zero-shift entries; dropping them to match active components.")
            else:
                raise ValueError(
                    f"Systematic {name} has independent error-on-error but epsilon has {eps.shape[0]} values (expected {expected} or {data.n_meas})")
        if expected == 0 or (eps.size == expected and not np.any(eps != 0.0)):
            data.uncertain_systematics.pop(name, None)
            warnings.warn(
                f"Systematic '{name}' has all-zero epsilons; removing from uncertain_systematics.")
