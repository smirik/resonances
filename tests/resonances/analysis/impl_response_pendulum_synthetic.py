"""Synthetic MMR records built by hand for the response/kinematics tests (no other analysis module is run).

A 'pendulum' record: sigma_ddot = -omega0^2 sin(sigma - sigma_s), a from the identity a = a* + sigma_dot / c, so that
a_dot = -(omega0^2 / c) sin(sigma - sigma_s): the force law holds exactly with K = omega0^2 / |c| and phi0 = sigma_s
(c < 0) or sigma_s + pi (c > 0). A 'kinematic' record: sigma swings by its varpi part while a stays flat.
"""

import typing
from dataclasses import MISSING, fields

import numpy as np
from scipy import integrate, signal

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis import pendulum as pend
from resonances.resonance.analysis.mmr import models as m
from resonances.resonance.analysis.mmr.context import mean_motion

NAN = float('nan')
G5 = PLANETARY_FREQUENCIES['g5'] * np.pi / 180 / 3600


def make(cls, **given):
    """A dataclass with every field not given set to its 'not measured' value (nan, empty, None)."""
    hints = typing.get_type_hints(cls, vars(m) | vars(geo) | vars(pend) | {'np': np})
    values = {}
    for f in fields(cls):
        if f.name in given or f.default is not MISSING or f.default_factory is not MISSING:
            continue
        values[f.name] = _empty(hints[f.name])
    values.update(given)
    return cls(**values)


def _empty(hint):
    origin = typing.get_origin(hint)
    if hint is float:
        return NAN
    if hint is bool:
        return False
    if hint is int:
        return 0
    if hint is str:
        return ''
    if hint is np.ndarray:
        return np.array([])
    if origin in (list, typing.List):
        return []
    if origin in (tuple, typing.Tuple):
        return ()
    if origin in (dict, typing.Dict):
        return {}
    return None


def identity_coefficient(k, a):
    return -1.5 * k * mean_motion(a) / a


def turning_points(t, angle, prominence=0.05):
    maxima, _ = signal.find_peaks(angle, prominence=prominence)
    minima, _ = signal.find_peaks(-angle, prominence=prominence)
    index = np.sort(np.concatenate([maxima, minima]))
    kinds = np.where(np.isin(index, maxima), 1, -1)
    return geo.TurningPoints(index, t[index], angle[index], kinds, prominence)


def episode_from(tp, episode_id=0, accepted=True):
    half = np.abs(np.diff(tp.values_rad)) / 2
    periods = tp.times_yr[2:] - tp.times_yr[:-2]
    return make(
        geo.Episode,
        episode_id=episode_id,
        tp_times_yr=tp.times_yr,
        tp_values_rad=tp.values_rad,
        accepted=accepted,
        shape=geo.EpisodeShape.STATIONARY,
        reject_reason=geo.RejectReason.NONE,
        period_med_yr=float(np.median(periods)),
        period_min_yr=float(np.min(periods)),
        period_max_yr=float(np.max(periods)),
        half_amp_min_rad=float(half.min()),
        half_amp_med_rad=float(np.median(half)),
        half_amp_max_rad=float(half.max()),
        bands_overlap=True,
    )


class Record:
    """A synthetic record with every input of analyse_response / analyse_episodes / analyse_kinematics."""

    def __init__(
        self,
        t,
        angle,
        rate,
        a,
        k,
        a_star,
        varpi,
        e,
        omega,
        spec,
        window,
        centre=None,
        cutoff_yr=10.0,
        planet_rates=None,
        slow_window_yr=None,
    ):
        self.t, self.angle, self.rate, self.a, self.k = t, angle, rate, a, k
        self.c = identity_coefficient(k, a_star)
        adot = np.gradient(a, t)
        self.prep = make(
            m.Preprocessed,
            cutoff=m.Cutoff(cutoff_yr, 'floor', 0.0, cutoff_yr, cutoff_yr / (t[1] - t[0])),
            t_yr=t,
            angle_unwrapped=angle,
            angle_clean=angle,
            angle_f=angle,
            rate_f=rate,
            angle_f_half=angle,
            angle_f_double=angle,
            a_f=a,
            adot_f=adot,
            e_f=e,
            inc_f=np.zeros_like(t),
            window=window,
            ripple_removed=False,
            ripple_half_amp_rad=0.0,
            ripple_corr=NAN,
            unremoved_ripple_rad=0.0,
        )
        tp = turning_points(t, angle)
        inside = (tp.times_yr >= window[0]) & (tp.times_yr <= window[1])
        tp = geo.TurningPoints(tp.indices[inside], tp.times_yr[inside], tp.values_rad[inside], tp.kinds[inside], tp.prominence_rad)
        runs = [episode_from(tp)] if len(tp) >= 3 else []
        centre = np.full_like(t, np.mean(angle)) if centre is None else centre
        track = make(geo.CentreTrack, t_yr=t, centre_rad=centre, window_yr=NAN, turns=turning_points(t, centre, 10.0))
        period = runs[0].period_med_yr if runs else NAN
        self.geometry = make(
            geo.Geometry, window=window, regime_start_yr=t[0], turning_points=tp, period_cycle_yr=period, centre=track, runs=runs
        )
        trim = slow_window_yr or (3 * period if np.isfinite(period) else 3 * cutoff_yr)
        slow_window = (t[0] + trim, t[-1] - trim)
        self.slow = m.SlowComponents(
            window_yr=trim,
            t_yr=t,
            angle_slow=centre,
            a_slow=np.full_like(t, np.mean(a)),
            adot_slow=1e-12 * np.random.default_rng(2).standard_normal(len(t)),  # a does not answer: noise only
            varpi_slow=varpi,
            e_slow=e,
            window=slow_window,
        )
        self.context = make(
            m.Context,
            planet_mean_motion_rad_yr={},
            planet_mean_motion_source={},
            planet_mass={},
            planet_varpi_rate_rad_yr=planet_rates or {},
            great_inequality=None,
            order=abs(spec.body_varpi_coefficient) + sum(abs(x) for x in spec.planet_varpi_coefficients),
            a_mean_au=a_star,
            e_med=float(np.median(e)),
            crosses=(),
            mean_motion_rad_yr=mean_motion(a_star),
            identity_coefficient=self.c,
            mu_scale_no_eq=1e-3,
        )
        self.sampling = make(m.Sampling, verdict=m.StepVerdict.RESOLVED)
        self.spec = spec
        n = len(t)
        self.series = m.MMRSeries(
            t_yr=t,
            a=a,
            e=e,
            inc=np.zeros(n),
            Omega=np.zeros(n),
            omega=omega,
            varpi=np.mod(varpi, 2 * np.pi),
            angle_wrapped=np.mod(angle, 2 * np.pi),
            resonance=spec,
            regime=m.Regime(step_yr=t[1] - t[0], span_yr=t[-1] - t[0], n_rows=n),
        )


def pendulum_record(omega0=2 * np.pi / 300.0, h=0.5, well=np.pi, k=-2, a_star=4.0, span=6000.0, step=0.5):
    """A Hilda-like k = -2 pendulum (c > 0) with T0 = 300 yr at h (A = 90 deg for h = 0.5)."""
    t = np.arange(0.0, span, step)
    sol = integrate.solve_ivp(
        lambda _, y: [y[1], -(omega0**2) * np.sin(y[0] - well)],
        (0, t[-1]),
        [well, 2 * omega0 * np.sqrt(h)],
        t_eval=t,
        rtol=1e-11,
        atol=1e-13,
    )
    angle, rate = sol.y
    c = identity_coefficient(k, a_star)
    a = a_star + rate / c
    varpi = 1e-4 * t  # a slow perihelion; the angle does not contain it here
    spec = m.ResonanceSpec('3J-2+0+0', m.ResonanceKind.TWO_BODY, ('Jupiter',), (3,), k, -1, (0,))
    return Record(t, angle, rate, a, k, a_star, varpi, np.full_like(t, 0.1), 3e-4 * t, spec, (t[0] + 100, t[-1] - 100))


def kinematic_record(p=3, k=-4, a_star=30.0, span=200000.0, step=10.0):
    """1N-4+0+3-like angle swinging only by p varpi: varpi_dot = g + dg cos(nu t); the lambda part cancels <p varpi_dot>.
    a is flat (tiny noise), so the slow a_dot does not answer sin sigma."""
    t = np.arange(0.0, span, step)
    g, dg = 2e-4, 1.5e-4
    nu = g - G5  # the varpi_dot modulation of a free/forced e-vector has the period 2 pi / (g - g5)
    varpi = g * t + dg / nu * np.sin(nu * t)
    angle = -p * g * t + p * varpi + np.pi  # sigma = psi_lambda + p varpi, psi_lambda = -p g t + pi
    rate = np.gradient(angle, t)
    rng = np.random.default_rng(1)
    a = a_star + 1e-7 * rng.standard_normal(len(t))
    spec = m.ResonanceSpec('1N-4+0+3', m.ResonanceKind.TWO_BODY, ('Neptune',), (1,), k, p, (0,))
    window = (t[0] + 1000, t[-1] - 1000)
    record = Record(t, angle, rate, a, k, a_star, varpi, np.full_like(t, 0.1), 1e-5 * t, spec, window, centre=angle, slow_window_yr=10000.0)
    record.nu, record.g = nu, g
    return record
