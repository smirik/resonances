"""The response of e and i to a secular angle (the column `check_secular_e_response`; DECISIONS: "Secular: the
response of e (or i) to sigma, e_dot ~ sin sigma, against a circle of constant radius (kinematics)").

A term R = K e^|p| sin^|q|(i/2) cos(sigma) of the averaged disturbing function, sigma = p varpi + q Omega + (planets),
drives e_dot ~ p sin(sigma) and d sin(i/2)/dt ~ q sin(sigma) (Lagrange's planetary equations, Murray & Dermott 1999,
§6.8, p. 251). The fit x_dot = A sin(sigma_f) + B cos(sigma_f) + C on the analysis window, with x and sigma at the same
cutoff, measures it: R^2, the amplitude sqrt(A^2 + B^2) per kyr and the phase atan2(-B, A) (0 for a pure sin sigma).
The variables are those the angle contains: e for a body g-term, sin(i/2) for a body s-term.

A column only: it changes no status and no confidence until the author decides (Q11).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from resonances.resonance.analysis.filtering import lowpass_or_none, time_derivative, window_mask
from resonances.resonance.analysis.filtering import r2 as filtering_r2
from resonances.resonance.analysis.secular.models import LongPeriodSeries, Response
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

NAN = float('nan')
PER_KYR = 1e3
MIN_SAMPLES = 8  # the three-parameter fit needs more points than parameters, with room for the residual


def _fit(t: np.ndarray, x: np.ndarray, sigma_f: np.ndarray, mask: np.ndarray, cutoff_yr: float, settings) -> Tuple[float, float, float]:
    """(R^2, amplitude per kyr, phase deg) of x_dot = A sin sigma + B cos sigma + C on `mask`."""
    x_f = lowpass_or_none(t, x, cutoff_yr, settings)
    if x_f is None:
        return NAN, NAN, NAN
    rate = time_derivative(t, x_f)[mask]
    sigma = sigma_f[mask]
    if len(rate) < MIN_SAMPLES or not np.all(np.isfinite(rate)):
        return NAN, NAN, NAN
    design = np.column_stack([np.sin(sigma), np.cos(sigma), np.ones_like(sigma)])
    coef, *_ = np.linalg.lstsq(design, rate, rcond=None)
    residual = rate - design @ coef
    r2 = filtering_r2(rate, residual)
    amplitude = float(np.hypot(coef[0], coef[1]) * PER_KYR)
    phase = float(np.degrees(np.arctan2(-coef[1], coef[0])))
    return r2, amplitude, phase


def analyse_response(series: LongPeriodSeries, angle, settings: AnalysisSettings) -> Response:
    """The response of e and sin(i/2) to the filtered secular angle `angle` (a `FilteredAngle`)."""
    variables = tuple(
        name for name, mode in (('e', 'g'), ('i', 's')) if any(term.index is None and term.mode == mode for term in series.terms)
    )
    response = Response(variables=variables)
    if angle is None or not variables:
        response.check = str(CheckValue.NOT_MEASURED)
        return response
    t = np.asarray(angle.t_yr, dtype=float)
    mask = window_mask(t, angle.window) & np.isfinite(angle.angle_f)
    if 'e' in variables:
        response.r2_e, response.amplitude_e_per_kyr, response.phase_e_deg = _fit(
            t, np.asarray(series.e, dtype=float), angle.angle_f, mask, angle.cutoff_yr, settings
        )
    if 'i' in variables:
        sin_half = np.sin(np.asarray(series.inc, dtype=float) / 2)
        response.r2_i, response.amplitude_i_per_kyr, response.phase_i_deg = _fit(
            t, sin_half, angle.angle_f, mask, angle.cutoff_yr, settings
        )
    values = [response.r2_e if name == 'e' else response.r2_i for name in variables]
    if not all(np.isfinite(value) for value in values):
        response.check = str(CheckValue.NOT_MEASURED)
    else:
        response.check = str(CheckValue.PASS if min(values) >= settings.secular_response_r2_min else CheckValue.FAIL)
    return response
