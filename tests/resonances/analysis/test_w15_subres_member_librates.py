"""Test L of the subresonance scan: an accepted member must librate itself in the scan window.

The case (EXPLANATIONS 2026-10-17 Q5; impl/w13/explain/agent1.md): 63047 3J+7S-2+0+0-8, scan window 42.4-44.0 kyr where
sigma0 has 24 turning points; the member 3J+7S-2-8+0+0 passed test A (phi0 coherence 0.86 > 0.8 with a force-law R^2
of 0.16) although it has no turning point in that window (its own row circulates, 3.8 turns in 50 kyr). The 130-yr
oscillation of sigma0 there is the short-period part of the body's own 8 varpi, which the member does not contain.

Independent truths: synthetic angles whose turning points are counted by hand (extrema of a sine), the numbers of the
63047 explanation (0 TPs of the member in the window, 24 of sigma0), and the prototypes of SISTERS §2 (65366, 31935)
and the 82831 case the author and Codex checked (the member p = 2 librates; its catalogue angle circulates)."""

import math
from dataclasses import fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import io, subresonances as sr
from resonances.resonance.analysis.mmr.models import ResonanceSpec, SubresonanceCandidate, SubresonanceTrigger
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue
from tests.resonances.analysis.builders import make_geometry, make_turning_points

S = settings_from_config()
NAN = float('nan')


def _member(key='9M-16+2+5', coherence=0.95, member_tp=-1, member_episodes=0):
    spec = ResonanceSpec.from_key(key)
    base = dict(
        spec=spec,
        j=int(sum(spec.planet_varpi_coefficients)),
        nodal=False,
        centre_range_rad=0.9,
        centre_rate_rad_yr=0.0,
        centre_max_pass_rad=0.9,
        centre_mean_rad=math.pi,
        phi0_coherence=coherence,
        coefficient_rel=NAN,
        composite_rms_rad=NAN,
        period_corr=NAN,
        period_ratio=NAN,
        rank=1,
        accepted=False,
        reason='',
    )
    assert set(base) == {f.name for f in fields(SubresonanceCandidate)}
    return sr.SubresonanceMember(**base, kept=True, member_tp=member_tp, member_episodes=member_episodes)


# --- the decision ----------------------------------------------------------------------------------------


def test_a_coherent_member_without_turning_points_is_not_accepted():
    # 63047: coherence 0.86 against 0.26 for sigma0 (A passes), 0 turning points of its own in the window
    creeping = _member('3J+7S-2-8+0+0', coherence=0.86, member_tp=0)
    ledger = Ledger.from_settings(S)
    outcome = sr.evaluate([creeping], 0.26, 0.03, 0.1, S, ledger=ledger)
    assert outcome.winner is None and outcome.reasons['3J+7S-2-8+0+0'] == sr.REASON_L
    failed = [entry for entry in ledger if entry.name.startswith('L ') and not entry]
    assert len(failed) == 1 and failed[0].value == 0 and failed[0].threshold == S.libration_min_turning_points


def test_turning_points_without_an_accepted_episode_are_not_a_libration():
    # enough turning points, but none of the runs passes the geometry (centre moves, no closed cycle)
    member = _member(member_tp=12, member_episodes=0)
    outcome = sr.evaluate([member], 0.3, 0.16, 0.1, S, ledger=Ledger.from_settings(S))
    assert outcome.winner is None and outcome.reasons[member.spec.key] == sr.REASON_L


def test_a_member_that_librates_itself_is_accepted_and_one_not_measured_is_not():
    librating = _member(member_tp=50, member_episodes=1)  # 31935 9M-16+2+5: 50 TPs, one episode over the window
    assert sr.decide([librating], 0.51, 0.16, 0.1, S)[0] is librating
    unmeasured = _member(member_tp=-1)
    outcome = sr.evaluate([unmeasured], 0.51, 0.16, 0.1, S)
    assert outcome.winner is None and outcome.reasons[unmeasured.spec.key] == sr.REASON_L_NOT_MEASURED


def test_test_l_lets_the_librating_member_win_over_a_creeping_one_with_a_smaller_range():
    creeping = sr.SubresonanceMember(**{**_member('9M-16+1+6', member_tp=0).__dict__, 'centre_range_rad': 0.5})
    librating = _member('9M-16+2+5', member_tp=40, member_episodes=1)
    outcome = sr.evaluate([creeping, librating], 0.3, 0.16, 0.1, S)
    assert outcome.winner is librating and outcome.reasons['9M-16+1+6'] == sr.REASON_L


def test_columns_of_test_l():
    member = _member(member_tp=0)
    columns = member.extra_columns()
    assert columns['member_tp'] == 0 and columns['member_episodes'] == 0
    assert _member().extra_columns()['member_tp'] is None  # not measured: empty, not -1


# --- the member's own angle (synthetic) ------------------------------------------------------------------


def _synthetic_scan(sigma0, offset, centre0, window, prominence_deg=20.0):
    t = np.arange(len(sigma0), dtype=float)
    scan = SimpleNamespace(
        t=t,
        window=window,
        settings=S,
        series=SimpleNamespace(resonance=ResonanceSpec.from_key('9M-16+0+7')),
        geometry=make_geometry(turning_points=make_turning_points(prominence_rad=math.radians(prominence_deg))),
        offset=lambda spec, member, nodal, smoothed=True: offset,
        centre0=centre0,
        screen_index=np.arange(len(t)),
    )
    prep = SimpleNamespace(angle_f=sigma0, cutoff=SimpleNamespace(cutoff_yr=20.0))
    return scan, prep


T = np.arange(20001, dtype=float)
MEMBER = ResonanceSpec.from_key('9M-16+2+5')


def test_member_libration_counts_the_turning_points_of_the_members_own_angle():
    # the member librates about pi with a 500-yr period; sigma0 = member - Delta drifts with Delta = 8 x 2 pi / 40 kyr
    delta = 8 * 2 * np.pi * T / 40000
    member = np.pi + 0.8 * np.sin(2 * np.pi * T / 500)
    scan, prep = _synthetic_scan(member - delta, delta, np.pi - delta, (1000.0, 19000.0))
    found = sr.member_libration(scan, prep, MEMBER, False)
    # extrema of sin(2 pi t / 500) at t = 125 + 250 k: k = 4 (1125 yr) .. 75 (18875 yr) -> 72, one episode over them
    assert found.n_tp == 72 and found.n_episodes == 1 and found.reject == '' and found.librates
    assert found.window == pytest.approx((1125.0, 18875.0), abs=2.0)


def test_member_libration_of_a_creeping_member_whose_oscillation_is_in_delta():
    # 63047-like: the 130-yr wave of sigma0 is the short-period part of 8 varpi (inside Delta); the member only creeps
    creep = 2 * np.pi * T / 40000
    delta = 8 * 2 * np.pi * T / 40000 + 0.7 * np.sin(2 * np.pi * T / 130)
    scan, prep = _synthetic_scan(creep - delta, delta, creep - 8 * 2 * np.pi * T / 40000, (1000.0, 19000.0))
    sigma0_tp = sr.find_turning_points(scan.t, prep.angle_f, scan.geometry.turning_points.prominence_rad, scan.window)
    assert len(sigma0_tp) > 100  # sigma0 turns back every 65 yr (the wave's slope 0.034 rad/yr beats the drift 0.0011)
    found = sr.member_libration(scan, prep, MEMBER, False)
    assert found.n_tp == 0 and found.n_episodes == 0 and not found.librates


def test_a_forced_wave_of_the_member_faster_than_the_period_floor_is_no_libration():
    # the member's own oscillation has a 100-yr period, below MMR_LIBRATION_PERIOD_MIN_YR = 150 yr: the geometry
    # accepts the run (closed cycles about a still centre) but the member's row would not count it (period floor)
    assert S.mmr_libration_period_min_yr > 100
    delta = 8 * 2 * np.pi * T / 40000
    member = np.pi + 0.8 * np.sin(2 * np.pi * T / 100)
    scan, prep = _synthetic_scan(member - delta, delta, np.pi - delta, (1000.0, 19000.0))
    found = sr.member_libration(scan, prep, MEMBER, False)
    assert found.n_tp == 360 and found.n_episodes == 0 and found.reject == sr.REJECT_PERIOD  # extrema 1025 .. 18975


# --- real records (benchmark data, no integration) ----------------------------------------------------


def _scan(example_id, settings=S):
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example_id).exists():
        pytest.skip('benchmark data not available')
    series = io.load_series(example_id)
    analysis = analyse_mmr(series, settings, scan_subresonances=False, scan_owner=False)
    triggered = (SubresonanceTrigger.T1,)
    return sr.run_subresonances(series, analysis.prep, analysis.geometry, analysis.response, analysis.context, triggered, settings)


def test_63047_member_without_turning_points_is_no_longer_accepted():
    # at the physical cutoff (75 yr) the member is not kept at all (test A); test L is shown at the cutoff of the W15
    # study, 20 x 0.6 yr = 12 yr, where A still passes
    physical = _scan('mmr3__63047__3J+7S-2+0+0-8')
    assert physical.accepted is None and not any(c.kept for c in physical.candidates if c.spec.key == '3J+7S-2-8+0+0')
    result = _scan('mmr3__63047__3J+7S-2+0+0-8', replace(S, mmr_filter_cutoff_yr=12.0))
    member = next(c for c in result.candidates if c.spec.key == '3J+7S-2-8+0+0')
    # A still passes (the explanation's 0.86 against 0.26) and the centre stands: only L rejects it
    assert member.kept and member.phi0_coherence >= S.mmr_subresonance_phi0_coh_min
    assert 42000 < result.window[0] < 43000 and 43500 < result.window[1] < 44500  # 42.4-44.0 kyr
    assert member.member_tp == 0 and member.member_episodes == 0 and member.reason == sr.REASON_L
    assert result.accepted is None and result.check == CheckValue.PASS
    assert result.to_columns()['sub_best_member_tp'] == 0


@pytest.mark.parametrize(
    'example_id, member',
    [('mmr2__65366__6M-11+0+5', '6M-11+1+4'), ('mmr2__31935__9M-16+0+7', '9M-16+2+5'), ('mmr2__82831__13M-24+0+11', '13M-24+2+9')],
)
def test_members_that_librate_stay_accepted(example_id, member):
    result = _scan(example_id)
    accepted = result.accepted
    assert accepted is not None and accepted.spec.key == member
    assert accepted.member_tp >= S.libration_min_turning_points and accepted.member_episodes >= 1
    # the member's own episode covers the scan window: the shared oscillation is the member's throughout
    span = result.window[1] - result.window[0]
    assert accepted.member_t1_yr - accepted.member_t0_yr > 0.9 * span
