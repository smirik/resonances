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

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from resonances.resonance.analysis.filtering import lowpass_or_none, time_derivative, window_mask
from resonances.resonance.analysis.filtering import r2 as filtering_r2
from resonances.resonance.analysis.secular.models import LongPeriodSeries, Response
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

NAN = float('nan')
PER_KYR = 1e3
MIN_SAMPLES = 8  # the three-parameter fit needs more points than parameters, with room for the residual


@dataclass
class ResponseFit:
    """One fit x_dot = A sin(sigma_f) + B cos(sigma_f) + C: x_dot (x filtered at the angle's cutoff) on the whole record,
    the fitted model, the samples fitted (`mask`) and the numbers of `Response`. Panel (f) of the secular figure draws
    these arrays, so its curve and its printed R^2 are one fit."""

    rate: np.ndarray  # per yr
    model: np.ndarray  # A sin + B cos + C on every sample
    mask: np.ndarray
    r2: float
    amplitude_per_kyr: float
    phase_deg: float


def fit_response(t: np.ndarray, x: np.ndarray, angle, settings: AnalysisSettings) -> Optional[ResponseFit]:
    """The fit of x_dot on the filtered angle `angle` (a `FilteredAngle`: sigma_f, its cutoff and window), x filtered
    at the same cutoff; None when x cannot be filtered or the window holds too few finite samples."""
    t = np.asarray(t, dtype=float)
    x_f = lowpass_or_none(t, np.asarray(x, dtype=float), angle.cutoff_yr, settings)
    if x_f is None:
        return None
    sigma = np.asarray(angle.angle_f, dtype=float)
    mask = window_mask(t, angle.window) & np.isfinite(sigma)
    rate = time_derivative(t, x_f)
    if mask.sum() < MIN_SAMPLES or not np.all(np.isfinite(rate[mask])):
        return None
    design = np.column_stack([np.sin(sigma), np.cos(sigma), np.ones_like(sigma)])
    coef, *_ = np.linalg.lstsq(design[mask], rate[mask], rcond=None)
    model = design @ coef
    r2 = filtering_r2(rate[mask], rate[mask] - model[mask])
    amplitude = float(np.hypot(coef[0], coef[1]) * PER_KYR)
    phase = float(np.degrees(np.arctan2(-coef[1], coef[0])))
    return ResponseFit(rate, model, mask, r2, amplitude, phase)


def _fit(t: np.ndarray, x: np.ndarray, angle, settings) -> Tuple[float, float, float]:
    """(R^2, amplitude per kyr, phase deg) of `fit_response`; nan when it cannot be fitted."""
    fit = fit_response(t, x, angle, settings)
    return (fit.r2, fit.amplitude_per_kyr, fit.phase_deg) if fit is not None else (NAN, NAN, NAN)


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
    if 'e' in variables:
        response.r2_e, response.amplitude_e_per_kyr, response.phase_e_deg = _fit(t, response_series(series, 'e'), angle, settings)
    if 'i' in variables:
        response.r2_i, response.amplitude_i_per_kyr, response.phase_i_deg = _fit(t, response_series(series, 'i'), angle, settings)
    values = [response.r2_e if name == 'e' else response.r2_i for name in variables]
    if not all(np.isfinite(value) for value in values):
        response.check = str(CheckValue.NOT_MEASURED)
    else:
        response.check = str(CheckValue.PASS if min(values) >= settings.secular_response_r2_min else CheckValue.FAIL)
    return response


def response_series(series: LongPeriodSeries, variable: str) -> np.ndarray:
    """The element whose rate answers the angle: e ('e') or sin(i/2) ('i')."""
    if variable == 'e':
        return np.asarray(series.e, dtype=float)
    return np.sin(np.asarray(series.inc, dtype=float) / 2)
