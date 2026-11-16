"""W4 status: episodes before the last jump of a (G1e) and the near-episode cap.

G1e (P0 of W4-TASKS): 2021 BA 1E-1 and 524522 1V-1 20 kyr librated (accepted episodes) and then, after a jump of a,
circulated; the last-regime window threw the episodes away and the verdict read non_resonant/circulation/high, while the
author's verdict is transient. Status follows the observed behaviour (DELTA 13b): the episodes make the status at least
transient, angle_status stays the last regime's (rule 10), the last regime's response and energy are not read for them.

Near-episode (2016 CA138: one half-cycle of capture, 2 TPs, author transient): the 3-TP threshold stays, but a verdict
without libration next to a run one TP short of an episode is capped at medium with the flag near_episode.

The records are synthetic: a libration of 40 deg and 500 yr, a jump of a at 10 kyr, then a circulation of one turn per
1000 yr; and a circulation of one turn per 400 yr whose rate w (1 - D exp(-((t - 3000)/150)^2)) turns negative for a
while. For D = 1.6 the angle runs back while D e^{-x^2} > 1, |x| < 150 sqrt(ln 1.6) = 102.8 yr, by
w [1.6 * 150 sqrt(pi) erf(sqrt(ln 1.6)) - 2 * 102.8] = 0.015708 * 78.6 = 1.235 rad: two TPs, half-amplitude 35.4 deg.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import confidence as conf
from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue, Confidence, Flag, LibrationOrigin, Status
from tests.resonances.analysis.test_impl_preprocess_geometry_geometry import ONE_DEG, filtered
from tests.resonances.analysis.builders import make_energy, make_response, make_sampling

S = settings_from_config()
SAMPLING = make_sampling(
    unwrap_resid_p99_rad=0.0, unwrap_resid_max_rad=0.0, unwrap_ok=True, undefined_phase=False, verdict='resolved', alias_numpy_differs=False
)
PERIOD = 500.0


def jump_record(after: str = 'circulation'):
    """Libration (40 deg, 500 yr) to 10 kyr, a jump of a by 1e-3 au, then `after`: a circulation of a turn per 1000 yr,
    or a libration of 100 deg around another centre. Geometry finished with cycle energies 0.2 before and 1.5 (a
    rotation) or 0.8 after."""
    t = np.arange(0, 20000, 1.0)
    early = np.pi + np.radians(40) * np.sin(2 * np.pi * t / PERIOD)
    if after == 'circulation':
        late, energy_after = np.pi + 2 * np.pi * (t - 10000) / 1000, 1.5
    else:
        late, energy_after = np.pi + 2 + np.radians(100) * np.sin(2 * np.pi * t / PERIOD), 0.8
    angle = np.where(t < 10000, early, late)
    slow = 1.0 + 1e-4 * np.cos(2 * np.pi * t / PERIOD) + 1e-3 * (t >= 10000)
    filtered_angle = filtered(t, angle, 50.0, slow=slow)
    geometry = g.measure_geometry(filtered_angle, ONE_DEG, S)
    cycles = np.arange(250, 20000, PERIOD)
    energies = np.where(cycles < 10000, 0.2, energy_after)
    return g.finish_geometry(geometry, filtered_angle, PERIOD, np.nan, np.nan, 0.02, cycles, energies, 5e-4, PERIOD, [], S)


def decide_all(geometry, settings=S, response=None, energy=None):
    ledger = Ledger.from_settings(settings)
    angle = decide.decide_angle(SAMPLING, False, geometry, energy, settings, ledger)
    decision = decide.decide_status(angle, SAMPLING, geometry, energy, response, None, None, None, None, settings, ledger)
    return angle, decision, ledger


# --- G1e ---------------------------------------------------------------------------------------------------------


def test_episodes_before_the_last_jump_make_the_status_transient():
    geometry = jump_record('circulation')
    assert str(Flag.LAST_REGIME_AFTER_JUMP) in geometry.flags
    (episode,) = geometry.episodes_before_jump
    assert episode.t_end_yr < 10000 and episode.n_tp > 30  # 40 half-cycles of 250 yr in 10 kyr, minus the edges
    angle, decision, ledger = decide_all(geometry)
    # angle_status is the last regime's (a circulation), the status that of the observed episode
    assert (angle.angle_status, angle.rule, angle.earlier_episode_id) == (AngleStatus.CIRCULATION, 'G4', episode.episode_id)
    assert decide.status_basis(angle) == AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G4', 0, 0)
    # nothing is measured on the episode (no response, energy or strength: rule 5u/6, W18): origin unknown
    assert (decision.status, decision.origin, decision.rule) == (Status.TRANSIENT, LibrationOrigin.UNKNOWN, '5u/6')
    assert 'geo_n_episodes_before_jump' in ledger.columns()


def test_the_last_regime_physics_is_not_read_for_an_earlier_episode():
    """A failed force law and potential of the circulation after the jump do not demote the episode before it."""
    geometry = jump_record('circulation')
    response = make_response(a_response=CheckValue.FAIL, a_amplitude=CheckValue.FAIL, force_r2_h1=0.0, force_r2_h2=0.0, b1_ratio=0.0)
    energy = make_energy(sigma_only=CheckValue.FAIL, with_slow=CheckValue.FAIL, ve_vf_agree_fraction=0.0)
    angle, decision, _ = decide_all(geometry, response=response, energy=energy)
    assert decision.status == Status.TRANSIENT
    assert decide.regime_physics(angle, response, energy) == (None, None)
    caps = decide.collect_caps(decide.tree_view(angle, None, energy), decision, geometry, {}, S)
    assert str(conf.Cap.EPISODE_BEFORE_JUMP) in caps
    assert conf.apply_caps(Confidence.HIGH, [str(conf.Cap.EPISODE_BEFORE_JUMP)]) == Confidence.MEDIUM
    # nothing measured on the episode itself (rule 5u/6, W18): the geometry alone, at most low
    assert str(conf.Cap.GEOMETRY_ONLY) in caps and conf.apply_caps(Confidence.HIGH, caps) == Confidence.LOW


def test_a_librating_last_regime_keeps_its_own_verdict():
    geometry = jump_record('libration')
    angle, decision, _ = decide_all(geometry)
    assert angle.angle_status == AngleStatus.LIBRATION and angle.earlier_episode_id is None
    assert decision.status == Status.RESONANT


def test_an_aliased_circulation_is_not_rescued_by_earlier_episodes():
    geometry = jump_record('circulation')
    ledger = Ledger.from_settings(S)
    aliased = AngleDecision(AngleStatus.CIRCULATION, AngleSubtype.ALIAS, 'G4')
    assert decide._g1e(aliased, geometry, S, ledger) == aliased
    assert decide.status_basis(aliased) == aliased


# --- near-episode ------------------------------------------------------------------------------------------------


def reversal_record(depth: float, width_yr: float = 150.0):
    t = np.arange(0, 6000, 1.0)
    rate = 2 * np.pi / 400 * (1 - depth * np.exp(-(((t - 3000) / width_yr) ** 2)))
    angle = np.concatenate([[0.0], np.cumsum((rate[1:] + rate[:-1]) / 2)])
    return g.measure_geometry(filtered(t, angle, 20.0), ONE_DEG, S)


def test_a_half_cycle_of_capture_is_a_near_episode():
    geometry = reversal_record(1.6)
    (run,) = geometry.runs
    assert run.n_tp == 2 and np.degrees(run.half_amp_max_rad) == pytest.approx(35.4, abs=1.0)
    assert decide.near_episode_tp(geometry, S) == 2
    angle, decision, ledger = decide_all(geometry)
    assert (angle.angle_status, decision.status) == (AngleStatus.CIRCULATION, Status.NON_RESONANT)
    assert str(Flag.NEAR_EPISODE) in decision.flags and str(conf.Cap.NEAR_EPISODE) in decision.caps
    assert conf.apply_caps(Confidence.HIGH, decision.caps) == Confidence.MEDIUM
    (entry,) = [e for e in ledger if e.column == 'geo_near_episode_n_tp']
    assert (entry.value, entry.threshold, entry.setting) == (2.0, 2, 'libration_near_episode_tp_deficit')
    margins = decide.evaluate_margins(
        decide.tree_view(angle, None, None), decision, CONTEXT, SAMPLING_FULL, geometry, None, None, None, [], S
    )
    assert margins['N'].margin == 'thin' and margins['N'].value == 2


def test_a_reversal_faster_than_the_shortest_libration_is_no_near_episode():
    """Width 30 yr: the angle runs back for 2 x 30 sqrt(ln 1.6) = 41 yr, a 'period' of 82 yr below the 150-yr floor
    (mmr_libration_period_min_yr): a forced wave, not half a libration (the episode criterion of `librating_episodes`)."""
    geometry = reversal_record(1.6, width_yr=30.0)
    (run,) = g.short_runs(geometry.runs)
    half_cycle = run.tp_times_yr[1] - run.tp_times_yr[0]
    assert 2 * half_cycle == pytest.approx(82.0, abs=6.0) and 2 * half_cycle < S.mmr_libration_period_min_yr
    assert decide.near_episode_tp(geometry, S) == 0
    _, decision, _ = decide_all(geometry)
    assert decision.status == Status.NON_RESONANT and str(Flag.NEAR_EPISODE) not in decision.flags


def test_the_setting_switches_the_near_episode_off():
    off = dataclasses.replace(S, libration_near_episode_tp_deficit=0)
    _, decision, _ = decide_all(reversal_record(1.6), settings=off)
    assert decision.status == Status.NON_RESONANT and str(Flag.NEAR_EPISODE) not in decision.flags


def test_a_reversal_over_the_top_is_no_near_episode():
    """D = 4: the angle runs back by more than a turn (half-cycle above pi): two over-the-top runs, nothing near."""
    geometry = reversal_record(4.0)
    assert all(run.reject_reason == g.RejectReason.OVER_THE_TOP for run in geometry.runs)
    assert decide.near_episode_tp(geometry, S) == 0
    _, decision, _ = decide_all(geometry)
    assert str(Flag.NEAR_EPISODE) not in decision.flags


CONTEXT = SimpleNamespace(rate_from_mean_longitude_rad_yr=np.nan, rate_expected_rad_yr=np.nan, rate_measured_rad_yr=np.nan)
SAMPLING_FULL = make_sampling(
    step_rotation_max_rad=0.01, aliased_lines_yr=[], verdict='resolved', far_ratio=float('nan'), branch_margin_turns=float('nan')
)
