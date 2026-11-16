"""W36 F5 (143662 s+g5-g8-s3, Claude's finding): a secular angle with too few turning points for one cycle whose
filtered angle spans at least one turn to the record's end is decided now as non_resonant (`policy._insufficient`);
when that travel lies within CONFIDENCE_THIN_BAND of one turn the level is at most medium (`policy._travel_thin`).
The travel runs from the last reversal of sigma_f (`policy._end_witness`, the author 2026-11-15: any run's turning
points, but a pair that only stalls the circulation is no reversal, `geometry.last_reversal_yr`).

The synthetic angle imitates 143662 (20 Myr at 2 kyr steps; the forced waves of g5, g8 and s3 set the cutoff as
there): sigma(t) = -2 pi k t / T - b sin(2 pi t / W), a slow retrograde drift with a wobble. With W = 20 Myr and b a
little above k the drift stalls once mid-record and turns back ~11 deg between two turning points (143662: 110 deg
down, 20 deg back, 330 deg down), a stall: the travel is read from the window's start. The expected travel is
computed by hand from the raw formula over the analysis window [P_c, T - P_c] (P_c ~ 709 kyr, the slowest forced
wave s - s3 doubled): the 709-kyr low-pass leaves the wobble and the drift intact, so the package's sigma_f range
must agree to 0.1 %:

    k = 1.25, b = 1.5 rad   ->  range 1.0545 turns: thin (< 1.1), non_resonant / medium
    k = 1.35, b = 1.6 rad   ->  range 1.1409 turns: clear,        non_resonant / high

With W = 10 Myr (k = 0.95, b = -0.9 rad) the drift stalls twice: 203 deg down, 32 back, 203 down, and the record ends
0.9 Myr after a turning point, with 6 deg after it. That last turn ends a full swing and nothing shows what follows:
uncertain, extend (it was read from the window's start, non_resonant / medium, while only accepted runs counted).
"""

import numpy as np
import pytest

from resonances.data.const import PLANETARY_FREQUENCIES as PF
from resonances.resonance.analysis import policy
from resonances.resonance.analysis.secular import AngleTerm, LongPeriodSeries, analyse_secular
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Action, AngleStatus, Confidence, Reason, ResonanceType, Status

SETTINGS = AnalysisSettings()
RAD = 2 * np.pi / 1296000.0  # rad/yr per arcsec/yr
SPAN = 20e6


def raw_angle(t, k, b, wobble_yr=SPAN):
    return -2 * np.pi * k * t / SPAN - b * np.sin(2 * np.pi * t / wobble_yr)


def short_record_series(k, b, wobble_yr=SPAN, n=10001) -> LongPeriodSeries:
    """The proper angle s + g5 - g8 - s3 = raw_angle (+ 1.1 rad): Omega carries it, varpi circulates freely."""
    t = np.linspace(0.0, SPAN, n)
    Omega = (PF['s3'] - PF['g5'] + PF['g8']) * RAD * t + raw_angle(t, k, b, wobble_yr) + 1.1
    varpi = 14.19 * RAD * t + 0.3
    sigma = Omega + (PF['g5'] - PF['g8'] - PF['s3']) * RAD * t
    terms = (AngleTerm('s', None, 1), AngleTerm('g', 5, 1), AngleTerm('g', 8, -1), AngleTerm('s', 3, -1))
    return LongPeriodSeries(
        ResonanceType.SECULAR,
        's+g5-g8-s3',
        'synthetic',
        t,
        sigma,
        np.full(n, 2.73),
        np.full(n, 0.05),
        np.full(n, np.radians(5.0)),
        varpi=varpi,
        Omega=Omega,
        terms=terms,
        planet_frequencies=PF,
    )


def hand_travel(k, b, cutoff_yr):
    t = np.linspace(cutoff_yr, SPAN - cutoff_yr, 200001)
    values = raw_angle(t, k, b)
    return (values.max() - values.min()) / (2 * np.pi)


@pytest.mark.parametrize(
    'k, b, travel, level, action, thin',
    [
        (1.25, 1.5, 1.0545, Confidence.MEDIUM, Action.SPOT_CHECK, True),
        (1.35, 1.6, 1.1409, Confidence.HIGH, Action.NONE, False),
    ],
)
def test_a_short_record_that_ends_in_circulation(k, b, travel, level, action, thin):
    analysis = analyse_secular(short_record_series(k, b), SETTINGS)
    v = analysis.verdict
    assert v.v2_verdict.startswith('uncertain//baseline_needed/extend/insufficient_time')  # the tree: too short
    assert (v.status, v.reason, v.angle_status) == (Status.NON_RESONANT, Reason.CIRCULATION, AngleStatus.INSUFFICIENT_TIME)
    assert v.pol_decided_by == policy.DECIDED_INSUFFICIENT and len(analysis.geometry.turning_points) == 2
    # the two turning points only stall the drift: the travel is read from the window's start, P_c after the record's start
    assert v.pol_t_last_tp_yr == pytest.approx(analysis.geometry.window[0])
    assert v.pol_end_travel_rev == pytest.approx(hand_travel(k, b, analysis.cutoff.cutoff_yr), rel=1e-3)
    assert v.pol_end_travel_rev == pytest.approx(travel, abs=2e-4)
    assert (v.confidence, v.action) == (level, action)
    assert ('end_travel:thin:medium' in v.pol_level_inputs) == thin


def test_a_record_that_ends_after_a_full_swing_is_too_short():
    analysis = analyse_secular(short_record_series(0.95, -0.9, wobble_yr=10e6), SETTINGS)
    v, tp = analysis.verdict, analysis.geometry.turning_points
    assert len(tp) == 4 and np.degrees(np.abs(np.diff(tp.values_rad))) == pytest.approx([203.4, 32.4, 203.4], abs=0.5)
    assert v.pol_t_last_tp_yr == pytest.approx(tp.times_yr[-1]) and v.pol_end_travel_rev < 0.02
    assert (v.status, v.reason, v.action) == (Status.UNCERTAIN, Reason.BASELINE_NEEDED, Action.EXTEND)


def test_the_thin_band_is_the_setting():
    # CONFIDENCE_THIN_BAND = 0.05: 1.0545 turns is clear of the band, the row is high again
    v = analyse_secular(short_record_series(1.25, 1.5), AnalysisSettings(confidence_thin_band=0.05)).verdict
    assert (v.status, v.confidence) == (Status.NON_RESONANT, Confidence.HIGH)
    assert policy.END_TRAVEL_INPUT not in v.pol_level_inputs
