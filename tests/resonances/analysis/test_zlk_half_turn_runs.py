"""W28 D3 (card g09-2): a Lidov-Kozai omega breaks its runs at half a turn; the circulation margin P has a thin band.

The averaged ZLK Hamiltonian depends on 2 omega, so an island (omega about 90 or 270 deg) is at most half a turn wide:
a half-cycle that swings half a turn or more has left the island (ZLK_RUN_BREAK_HALF_AMPLITUDE_DEG = 90). MMR and
secular angles keep pi. 88899 over 500 kyr: a run at pi joined the two islands into one rejected run and the angle
read circulation on a centre pass of 1.04 rev.

Expectations are computed by hand from the synthetic angle: islands of half-amplitude 20 deg about 90 and 270 deg
joined by a smooth half-turn step; the half-cycle across the step swings from 90 - 20 to 270 + 20 deg, a half-amplitude
of (180 + 40) / 2 = 110 deg: at least 90 (a break), below 180 (no break at pi).
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.ledger import Ledger, Margin
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.secular import LongPeriodSeries, decide, preprocess
from resonances.resonance.analysis.secular.analyse import analyse_long_period
from resonances.resonance.analysis.settings import settings_from_config, validate_settings
from resonances.resonance.analysis.verdict import AngleStatus, ResonanceType, Status
from tests.resonances.analysis.builders import make_centre, make_geometry, make_turning_points

SETTINGS = settings_from_config()
T = np.arange(0.0, 300_001.0, 100.0)
ONE = np.ones_like(T)
ISLAND_HALF_DEG = 20.0
ISLAND_PERIOD_YR = 20_000.0  # well above the ZLK floor (2 kyr) and its filter (P_c = 2 x 937 yr)


def _series(omega_deg: np.ndarray) -> LongPeriodSeries:
    return LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', T, np.radians(omega_deg), 5.2 * ONE, 0.05 * ONE, 0.3 * ONE)


def _island_switch() -> np.ndarray:
    """omega about 90 deg, a smooth half-turn step at 150 kyr (the record passes e ~ 0), then about 270 deg."""
    step = 0.5 * (1.0 + np.tanh((T - 150_000.0) / 3000.0))
    return 90.0 + 180.0 * step + ISLAND_HALF_DEG * np.sin(2 * np.pi * T / ISLAND_PERIOD_YR)


def _accepted(analysis):
    return [run for run in analysis.geometry.runs if run.accepted]


def test_the_default_is_half_a_turn_and_validated():
    assert SETTINGS.zlk_run_break_half_amplitude_deg == 90.0
    with pytest.raises(ValueError):
        validate_settings(replace(SETTINGS, zlk_run_break_half_amplitude_deg=0.0))
    with pytest.raises(ValueError):
        validate_settings(replace(SETTINGS, zlk_run_break_half_amplitude_deg=181.0))


def test_two_islands_through_e_zero_are_two_episodes():
    analysis = analyse_long_period(_series(_island_switch()), SETTINGS)
    accepted = _accepted(analysis)
    assert len(accepted) == 2
    first, second = accepted
    assert np.degrees(first.centre_mod_2pi_rad) == pytest.approx(90.0, abs=5.0)
    assert np.degrees(second.centre_mod_2pi_rad) == pytest.approx(270.0, abs=5.0)
    assert first.t_end_yr < 150_000.0 < second.t_start_yr
    for run in accepted:
        assert np.degrees(run.half_amp_max_rad) == pytest.approx(ISLAND_HALF_DEG, abs=2.0)
    assert analysis.verdict.angle_status == AngleStatus.EPISODIC
    assert analysis.verdict.status == Status.TRANSIENT  # no gate here: the geometry's status
    assert str(Rule.ZLK_HALF_TURN_CUT) in analysis.verdict.rules_fired


def test_at_pi_the_islands_are_one_run():
    """The rule before D3: the 110-deg half-cycle stays in the run, which spans both islands."""
    analysis = analyse_long_period(_series(_island_switch()), replace(SETTINGS, zlk_run_break_half_amplitude_deg=180.0))
    assert not _accepted(analysis)
    assert len(analysis.geometry.runs) == 1
    assert np.degrees(analysis.geometry.runs[0].half_amp_max_rad) == pytest.approx(110.0, abs=3.0)
    assert str(Rule.ZLK_HALF_TURN_CUT) not in analysis.verdict.rules_fired


def test_one_wide_island_stays_one_run():
    """An island at 80 deg of half-amplitude 60 deg (below half a turn): one libration, nothing cut."""
    omega = 80.0 + 60.0 * np.sin(2 * np.pi * T / ISLAND_PERIOD_YR)
    analysis = analyse_long_period(_series(omega), SETTINGS)
    assert len(analysis.geometry.runs) == 1
    (run,) = _accepted(analysis)
    assert np.degrees(run.centre_mod_2pi_rad) == pytest.approx(80.0, abs=2.0)
    assert np.degrees(run.half_amp_max_rad) == pytest.approx(60.0, abs=2.0)
    assert analysis.verdict.angle_status == AngleStatus.LIBRATION
    assert str(Rule.ZLK_HALF_TURN_CUT) not in analysis.verdict.rules_fired


def test_mmr_and_secular_angles_keep_pi():
    """The same turning points: a ZLK omega breaks at the 110-deg half-cycle, an MMR or secular angle (pi) does not."""
    analysis = analyse_long_period(_series(_island_switch()), SETTINGS)
    tp = analysis.geometry.turning_points
    assert tp.break_half_amplitude_rad == pytest.approx(np.pi / 2)
    at_pi = replace(tp, break_half_amplitude_rad=np.pi)
    assert len(g.split_runs(at_pi, SETTINGS)) == 1
    assert len(g.split_runs(tp, SETTINGS)) == 2
    assert g.half_turn_cut(tp, SETTINGS) and not g.half_turn_cut(at_pi, SETTINGS)
    # the defaults: TurningPoints and FilteredAngle carry pi; the secular preprocessing keeps it for a secular angle
    assert make_turning_points([0.0, 1.0]).break_half_amplitude_rad == np.pi
    assert g.FilteredAngle.__dataclass_fields__['run_break_half_amplitude_rad'].default == np.pi
    secular = LongPeriodSeries(
        ResonanceType.SECULAR, 'g-g6', 'synthetic', T * 50, np.radians(_island_switch()), 2.4 * ONE, 0.1 * ONE, 0.1 * ONE
    )
    angle = preprocess.filtered_angle(secular, preprocess.choose_cutoff(secular, SETTINGS), SETTINGS)
    assert angle is not None and angle.run_break_half_amplitude_rad == np.pi


def test_the_midpoint_jump_breaks_at_half_a_turn_too():
    # half-amplitudes 1, 1, then the centre moves from 0 to 2 rad (> pi/2, < pi) with the half-amplitude still 1
    values = [1.0, -1.0, 1.0, 3.0, 1.0]
    tp = replace(make_turning_points([0, 10, 20, 30, 40], values), break_half_amplitude_rad=np.pi / 2)
    assert [list(run) for run in g.split_runs(tp, SETTINGS)] == [[0, 1, 2], [3, 4]]
    assert len(g.split_runs(replace(tp, break_half_amplitude_rad=np.pi), SETTINGS)) == 1


@pytest.mark.parametrize('low, margin', [(1.04, Margin.THIN), (1.0, Margin.THIN), (1.5, Margin.FULL), (1.1, Margin.FULL)])
def test_the_circulation_margin_has_a_thin_band(low, margin):
    """P is full only clear of one turn by CONFIDENCE_THIN_BAND (0.1): 1.04 rev (88899) is thin, 1.5 rev full."""
    geometry = make_geometry(centre=make_centre(max_pass_rev_lo=low))
    margins, _, _ = decide.margins_and_caps(
        decide.AngleDecision(AngleStatus.CIRCULATION), geometry, 1000.0, None, SETTINGS, Ledger.from_settings(SETTINGS)
    )
    assert margins['P'].margin == margin
    assert margins['P'].value == pytest.approx(low)
