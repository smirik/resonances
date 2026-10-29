"""The output step as a validity condition of the MMR filter (inv-09, inv-15; docs/physics/signal.md).

The cutoff no longer grows with the step: P_c = max(1.5 P_syn, P_floor / 2). The step only decides
- whether the record can be filtered at all: P_c <= 2 dt -> invalid / undersampled, resample at the largest round step
  <= P_floor / MMR_RESAMPLE_STEPS_PER_CYCLE (a Jupiter Trojan: P_floor = 0.5 x 147.75 yr, P_c = 36.9 yr, invalid from
  dt = 18.47 yr, resample at 73.9 / 20 = 3.69 -> 2.5 yr);
- whether short-period terms fold into slow ones: dt >= min(P_orb, P_syn) / 2 -> flag short_period_aliased, and the
  cap (confidence <= medium) unless the fast residual of sigma_f is small against the episode's half-amplitude (W20).

Hand values: P_J = 4332.59 d / 365.2569 d = 11.8617 Gaussian yr, m_Sun / m_J = 1047.3486, tadpole period
P_J / sqrt(27 mu / 4) = 147.75 yr; a Trojan has Jupiter's a = 5.2026 au (P_orb = 11.866 yr).
"""

import math

import numpy as np
import pytest

from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.mmr import preprocess
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.mmr.preprocess import DecisiveQuantities
from resonances.resonance.analysis.mmr.sampling import short_period_step
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Action, CheckValue, Confidence, Flag, Reason, Status, default_action

S = settings_from_config()
GAUSS_YEAR_DAYS = 365.2569
N_JUPITER = 2 * math.pi * GAUSS_YEAR_DAYS / 4332.59  # rad / yr
TADPOLE_YR = 147.75


def trojan(step_yr: float, span_yr: float = 20000.0, period_yr: float = 147.77) -> MMRSeries:
    """A linear tadpole about L4, sigma = lambda_J - lambda = pi/3 + 0.05 cos(w t) with a answering it
    (sigma_dot = c (a - a*)), sampled at `step_yr` (instantaneous samples: nothing faster than the libration)."""
    t = np.arange(0.0, span_yr + 0.5 * step_yr, step_yr)
    w = 2 * math.pi / period_yr
    a0 = (2 * math.pi / N_JUPITER) ** (2 / 3)
    c = 3 * math.pi * a0**-2.5
    zero = np.zeros(t.size)
    sigma = math.pi / 3 + 0.05 * np.cos(w * t)
    return MMRSeries(
        t,
        a0 - 0.05 * w * np.sin(w * t) / c,
        np.full(t.size, 0.01),
        zero,
        zero,
        zero,
        zero,
        np.mod(sigma, 2 * math.pi),
        ResonanceSpec.from_key('1J-1'),
        Regime(step_yr, t[-1], t.size),
    )


def analyse(series):
    return analyse_mmr(series, S, scan_owner=False, scan_subresonances=False)


def test_a_trojan_step_above_half_the_cutoff_is_undersampled():
    analysis = analyse(trojan(20.0))
    verdict = analysis.verdict
    assert (verdict.status, verdict.reason) == (Status.INVALID, Reason.UNDERSAMPLED)
    assert verdict.action == Action.RESAMPLE and verdict.resample_step_yr == 2.5
    assert verdict.checks['check_mmr_filter'] == CheckValue.FAIL
    assert all(value == CheckValue.NOT_MEASURED for name, value in verdict.checks.items() if name != 'check_mmr_filter')
    entry = verdict.decision_path[0]
    assert entry.column == 'flt_cutoff_yr' and entry.result is False
    assert entry.value == pytest.approx(TADPOLE_YR / 4, rel=2e-3) and entry.threshold == pytest.approx(40.0, rel=1e-6)
    assert analysis.prep is None and analysis.extra_columns['flt_period_floor_yr'] == pytest.approx(TADPOLE_YR / 2, rel=2e-3)
    assert str(Flag.SHORT_PERIOD_ALIASED) in verdict.flags  # 20 yr >= 11.87 / 2


@pytest.mark.slow
@pytest.mark.parametrize('step, flagged', [(1.0, False), (5.0, False), (15.92, True), (18.0, True)])
def test_a_trojan_below_the_limit_keeps_its_libration(step, flagged):
    analysis = analyse(trojan(step))
    verdict = analysis.verdict
    assert verdict.status == Status.RESONANT
    assert analysis.prep.cutoff.binding == 'floor' and analysis.prep.cutoff.cutoff_yr == pytest.approx(TADPOLE_YR / 4, rel=2e-3)
    assert not analysis.prep.filter_too_slow
    assert (str(Flag.SHORT_PERIOD_ALIASED) in verdict.flags) is flagged  # P_orb / 2 = 5.93 yr
    # W20 round D: the record never lifts the cap (MMR_SHORT_PERIOD_CAP_RATIO = 0): a flagged record keeps it, however
    # clean (the fast residual is not even compared). The folded lines of P_orb = 11.862 yr up to the second harmonic:
    # at 15.92 yr |1/11.862 - 1/15.92|^-1 = 46.5 yr and |2/11.862 - 3/15.92|^-1 = 50.5 yr; at 18 yr the harmonic folds
    # to |2/11.862 - 3/18|^-1 = 507 yr
    assert (str(Cap.SHORT_PERIOD_ALIASED) in verdict.caps) is flagged
    if flagged:
        expected = max(1 / abs(m / 11.862 - round(m * step / 11.862) / step) for m in (1, 2))
        assert analysis.sampling.folded_slowest_yr == pytest.approx(expected, rel=0.01)
    assert not [e for e in verdict.decision_path if e.setting == 'mmr_short_period_cap_ratio']
    assert analysis.sampling.short_period_step_yr == pytest.approx(5.2026**1.5 / 2, rel=2e-3)


@pytest.mark.slow
@pytest.mark.parametrize('step, flagged', [(5.0, False), (15.92, True)])
def test_the_short_period_cap_with_the_opt_in_lift(step, flagged):
    # MMR_SHORT_PERIOD_CAP_RATIO = 0.1 (the W20 F5 lift, opt-in since round D): the clean tadpole's fast residual is far
    # below 0.1 of its 2.86-deg half-amplitude, so the cap depends on the scatter of a against its swing only; default:
    # medium at most when flagged
    lift = settings_from_config(mmr_short_period_cap_ratio=0.1)
    analysis = analyse_mmr(trojan(step), lift, scan_owner=False, scan_subresonances=False)
    verdict = analysis.verdict
    assert verdict.status == Status.RESONANT
    ratio = analysis.numbers_row()['smp_da_sp_over_a_excursion']
    assert (str(Cap.SHORT_PERIOD_ALIASED) in verdict.caps) is (flagged and not ratio <= lift.mmr_short_period_a_ratio)
    default = analyse_mmr(trojan(step), S, scan_owner=False, scan_subresonances=False).verdict
    assert (str(Cap.SHORT_PERIOD_ALIASED) in default.caps) is flagged
    if flagged:
        assert default.confidence in (Confidence.MEDIUM, Confidence.LOW)


def test_the_resample_step_also_filters_an_explicit_cutoff():
    # MMR_FILTER_CUTOFF_YR = 1 yr at dt = 16 yr: P_c <= 2 dt, undersampled; the floor bound alone would give 2.5 yr,
    # which cannot filter 1 yr either: the step must stay below P_c / 2 = 0.5 yr -> 0.25 yr (Codex, W16 review)
    settings = settings_from_config(mmr_filter_cutoff_yr=1.0)
    verdict = analyse_mmr(trojan(16.0), settings, scan_owner=False, scan_subresonances=False).verdict
    assert (verdict.status, verdict.reason) == (Status.INVALID, Reason.UNDERSAMPLED)
    assert verdict.resample_step_yr == 0.25 and preprocess.filter_realizable(1.0, verdict.resample_step_yr)


def test_short_period_step_by_hand():
    # Hilda a = 3.9684: P = 7.905 yr, P_syn,J = 23.70 yr -> 3.953 yr; a Trojan: P_syn,J ~ infinite -> P_orb / 2
    assert short_period_step(3.9684**1.5, {'Jupiter': 23.70}) == pytest.approx(3.9526, abs=1e-3)
    assert short_period_step(11.866, {'Jupiter': 1e9}) == pytest.approx(5.933)
    # a three-body angle: the shortest of P_orb and the synodic periods with its planets
    assert short_period_step(5.13, {'Mars': 3.05, 'Jupiter': 9.6}) == pytest.approx(1.525)
    assert short_period_step(float('nan'), {}) == math.inf


def test_undersampled_asks_to_resample():
    assert default_action(Status.INVALID, Reason.UNDERSAMPLED, Confidence.NONE) == Action.RESAMPLE
    assert default_action(Status.INVALID, Reason.UNPHYSICAL, Confidence.NONE) == Action.REVIEW


def test_the_x2_guard_compares_the_shape_by_the_digital_gain():
    # 588 at dt = 15.9185 yr (Codex, inv-15): a 9-step cycle of 143.27 yr against the x2 cutoff 73.885 yr. The ratio
    # rule (143.27 < 2 x 73.885) would not compare its shape; the digital gain 0.9596 >= 0.94 does
    dt = 15.918509
    same = dict(n_tp=10, n_episodes=1, stall=False, stall_fraction=0.0, h_med=0.3, period_yr=9 * dt, half_amp_max_rad=0.08)
    first = DecisiveQuantities(cutoff_yr=36.94, centre_shift_over_amp=0.1, **same)
    second = DecisiveQuantities(cutoff_yr=73.885, centre_shift_over_amp=0.1, **same)
    assert preprocess.compare_quantities(first, second, S, dt).not_checked == ()
    assert preprocess.compare_quantities(first, second, S).not_checked == ('energy', 'shape')  # the analog limit: G = 0.93
