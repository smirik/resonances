"""G5w (W28 B6c, cards g07-5/g07-6 A): is an unfinished slow swing of sigma a swing in the resonance's own well?

A record whose slow centre (`Geometry.centre`, sigma low-passed at W_c) turns back twice inside the record holds one
measured half-swing between the turns. If sigma is an incipient slow libration of this resonance, that half-swing must
obey the resonance's pendulum (Murray & Dermott 1999, sect. 8.6-8.7; the well from direct averaging,
`strength.theory_period`):

- the identity on the slow scale: sigma_dot_s = c (a_s - a*), c = -1.5 k n / a, with sigma_s and a_s low-passed at
  W_c (`response.identity`, MMR_IDENTITY_SLOPE_TOL and MMR_IDENTITY_CORR_MIN, over the window trimmed by W_c);
- the swing of a: between two turns sigma_dot changes sign once, so a runs from a* to a* + delta a and back, and
  delta a_meas = ptp(a_s) there equals the pendulum's half-excursion 2 omega0 sin(A/2) / |c| at the half-swing's
  amplitude A (`pendulum.excursion`, omega0 = 2 pi / T0_th);
- the period: 2 (t2 - t1) against T0_th F(A), F(A) = (2/pi) K(sin^2(A/2)) (`pendulum.amplitude_factor`).

Both ratios must lie within [1/sqrt(x), sqrt(x)], x = MMR_STRENGTH_CHECK_RATIO (the strength check's own bound on
mu: omega0 and delta a scale as sqrt(mu)); no new threshold. A drift between two turns (the author's drift of 101703
and 142694 1V+3S-7) fails the slow identity (slope / c -0.27 and -0.25) and is three-body (no direct theory); 523683
2U-9 passes all three (1.00, a 1.07, period 1.00). Anything unmeasured leaves G5w unmet (`SlowSwing.holds` False).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.filtering import is_finite
from resonances.resonance.analysis.geometry import Geometry
from resonances.resonance.analysis.mmr import strength as strength_module
from resonances.resonance.analysis.mmr.models import Context, MMRSeries, Preprocessed
from resonances.resonance.analysis.mmr.response import identity
from resonances.resonance.analysis.pendulum import amplitude_factor, excursion
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

NAN = float('nan')
TWO_PI = 2 * math.pi

# SlowSwing.reason: why G5w is not measured ('' when every number is)
REASON_OFF = 'off'  # MMR_SLOW_SWING_IN_WELL off
REASON_TURNS = 'not_two_turns'  # the slow centre does not turn back exactly twice inside the record within one turn
REASON_COORBITAL = 'coorbital'  # a 1:1 angle has its own force law (`coorbital.py`)
REASON_IDENTITY = 'identity_not_measured'
REASON_THEORY = 'theory_not_measured'  # no T0_th (or the strength module's own reason: not_two_body, no_planet_orbit, ...)


@dataclass(frozen=True)
class SlowSwing:
    """The G5w numbers (module docstring): the half-swing between the slow centre's two interior turns
    [t_start_yr, t_end_yr], its amplitude A [rad] (half the range of sigma_s there), the period it implies
    2 (t_end - t_start) [yr], the slow identity (slope / c, correlation, check), the swing of a there (ptp of a_s, au)
    and its theory at A (au), T0_th and T0_th F(A) [yr]; `reason` says what is missing: '' when G5w can be judged
    (everything measured, or the slow identity measured and failed, when the theory is not computed)."""

    t_start_yr: float = NAN
    t_end_yr: float = NAN
    amplitude_rad: float = NAN
    period_yr: float = NAN
    identity_slope_ratio: float = NAN
    identity_corr: float = NAN
    identity: CheckValue = CheckValue.NOT_MEASURED
    a_swing_au: float = NAN
    a_swing_theory_au: float = NAN
    small_period_theory_yr: float = NAN
    period_theory_yr: float = NAN
    reason: str = REASON_TURNS

    @property
    def a_ratio(self) -> float:
        """delta a_meas / delta a_th(A)."""
        return _ratio(self.a_swing_au, self.a_swing_theory_au)

    @property
    def period_ratio(self) -> float:
        """2 (t2 - t1) / (T0_th F(A))."""
        return _ratio(self.period_yr, self.period_theory_yr)

    @property
    def measured(self) -> bool:
        return self.reason == ''


def _ratio(value: float, theory: float) -> float:
    return float(value / theory) if is_finite(value) and is_finite(theory) and theory > 0 else NAN


def within_well(ratio: float, settings: AnalysisSettings) -> bool:
    """`ratio` within [1/sqrt(x), sqrt(x)], x = mmr_strength_check_ratio (False when unmeasured)."""
    bound = math.sqrt(settings.mmr_strength_check_ratio)
    return is_finite(ratio) and 1 / bound <= ratio <= bound


def slow_turns(geometry: Geometry) -> np.ndarray:
    """The slow centre's turns farther than W_c from both window edges (`decide.interior_turn_times`, repeated here
    so that this module does not import the decision code)."""
    centre = geometry.centre
    times = np.sort(np.asarray(centre.turns.times_yr, dtype=float))
    start, end = geometry.window
    return times[(times - start > centre.window_yr) & (end - times > centre.window_yr)]


def half_swing(
    geometry: Optional[Geometry], t_yr: np.ndarray, a_au: np.ndarray, identity_coefficient: float, settings: AnalysisSettings
) -> SlowSwing:
    """The numbers of the half-swing alone, before the theory (`measure_slow_swing`): `a_au` is the record's a
    (filtered at P_c) at `t_yr`, low-passed here at the centre's W_c like sigma_s; c = `identity_coefficient`. Reason
    not_two_turns unless the slow centre turns back exactly twice away from the edges within one turn (one half-swing
    measured, no full slow cycle), coorbital for a 1:1 angle, identity_not_measured when the slow identity has no
    number (a flat a), '' otherwise (the theory is still missing then: `with_theory`)."""
    if not settings.mmr_slow_swing_in_well:
        return SlowSwing(reason=REASON_OFF)
    if geometry is None or geometry.coorbital:
        return SlowSwing(reason=REASON_COORBITAL if geometry is not None else REASON_TURNS)
    centre = geometry.centre
    turns = slow_turns(geometry)
    if len(turns) != 2 or not (is_finite(centre.range_rad) and centre.range_rad < TWO_PI):
        return SlowSwing()
    t = np.asarray(centre.t_yr, dtype=float)
    sigma = np.asarray(centre.centre_rad, dtype=float)
    t_all = np.asarray(t_yr, dtype=float)
    a = np.asarray(a_au, dtype=float)[int(np.searchsorted(t_all, t[0])) :][: len(t)]
    a_slow = filtering.lowpass(t, a, centre.window_yr, settings.filter_order)
    rate = np.gradient(sigma, t)
    start, end = geometry.window
    inside = (t >= start + centre.window_yr) & (t <= end - centre.window_yr)
    ratio, corr, check = identity(rate[inside], a_slow[inside], identity_coefficient, settings)
    t1, t2 = (float(x) for x in turns)
    half = (t >= t1) & (t <= t2) & np.isfinite(sigma) & np.isfinite(a_slow)
    return SlowSwing(
        t_start_yr=t1,
        t_end_yr=t2,
        amplitude_rad=float(np.ptp(sigma[half]) / 2) if np.any(half) else NAN,
        period_yr=2 * (t2 - t1),
        identity_slope_ratio=ratio,
        identity_corr=corr,
        identity=check,
        a_swing_au=float(np.ptp(a_slow[half])) if np.any(half) else NAN,
        reason=REASON_IDENTITY if check == CheckValue.NOT_MEASURED else '',
    )


def with_theory(swing: SlowSwing, small_period_yr: float, identity_coefficient: float, reason: str = '') -> SlowSwing:
    """`swing` with the pendulum of the resonance's well: T0_th = `small_period_yr` (the directly averaged
    potential, `strength.theory_period`), the a half-excursion 2 omega0 sin(A/2) / |c| and the period T0_th F(A) at
    the half-swing's amplitude A; without a T0_th the reason `reason` (or theory_not_measured)."""
    if not (is_finite(small_period_yr) and small_period_yr > 0):
        return replace(swing, reason=reason or REASON_THEORY)
    amplitude = swing.amplitude_rad
    return replace(
        swing,
        a_swing_theory_au=excursion(TWO_PI / small_period_yr, amplitude, identity_coefficient),
        small_period_theory_yr=float(small_period_yr),
        period_theory_yr=float(small_period_yr * amplitude_factor(amplitude)),
    )


def measure_slow_swing(
    series: MMRSeries, context: Context, prep: Preprocessed, geometry: Optional[Geometry], settings: AnalysisSettings
) -> SlowSwing:
    """The G5w numbers of the record (module docstring): `half_swing`, then, only when the slow identity passes (it
    fails for a drift, and the averaging costs), T0_th of the directly averaged well at the half-swing's snapshots
    (`strength.theory_period` on `strength.record_swings` of sigma_s; two-body, not 1:1: three-body angles have no
    direct theory, reason not_two_body) and `with_theory`."""
    swing = half_swing(geometry, prep.t_yr, prep.a_f, context.identity_coefficient, settings)
    if not swing.measured or swing.identity != CheckValue.PASS:
        return swing
    centre = geometry.centre
    parts = strength_module.record_swings(centre.t_yr, centre.centre_rad, swing.t_start_yr, swing.t_end_yr, settings.mmr_strength_snapshots)
    if parts is None:
        return replace(swing, reason=REASON_THEORY)
    theory = strength_module.theory_period(series, context, parts, prep.cutoff.cutoff_yr, settings)
    return with_theory(swing, theory.period_min_yr, context.identity_coefficient, theory.reason)


def holds(swing: Optional[SlowSwing], settings: AnalysisSettings) -> bool:
    """G5w: the slow identity passes and both the a swing and the period agree with the well within
    [1/sqrt(x), sqrt(x)] (`within_well`). False when anything is unmeasured (the row stays as it was)."""
    if swing is None or not swing.measured or swing.identity != CheckValue.PASS:
        return False
    return within_well(swing.a_ratio, settings) and within_well(swing.period_ratio, settings)


def slow_swing_columns(swing: Optional[SlowSwing]) -> dict:
    """The geo_slow_swing_* columns (mmr_numbers.csv): the half-swing, the slow identity, the a and period ratios and
    why G5w was not measured (empty when it was)."""
    swing = swing if swing is not None else SlowSwing()
    return {
        'geo_slow_swing_t0_yr': swing.t_start_yr,
        'geo_slow_swing_t1_yr': swing.t_end_yr,
        'geo_slow_swing_amp_deg': math.degrees(swing.amplitude_rad) if is_finite(swing.amplitude_rad) else NAN,
        'geo_slow_swing_period_yr': swing.period_yr,
        'geo_slow_swing_identity_slope_ratio': swing.identity_slope_ratio,
        'geo_slow_swing_identity_corr': swing.identity_corr,
        'geo_slow_swing_a_au': swing.a_swing_au,
        'geo_slow_swing_a_theory_au': swing.a_swing_theory_au,
        'geo_slow_swing_t0_theory_yr': swing.small_period_theory_yr,
        'geo_slow_swing_period_theory_yr': swing.period_theory_yr,
        'geo_slow_swing_a_ratio': swing.a_ratio,
        'geo_slow_swing_period_ratio': swing.period_ratio,
        'geo_slow_swing_reason': swing.reason,
    }
