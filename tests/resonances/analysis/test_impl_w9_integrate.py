"""W9 integration: the regimes of a record with a jump of a do not share turning points or a centre.

- the runs before the last jump keep only their turning points before the last regime's start (jump + tau/2): the jump
  zone stays with the regime before it (2021 BA 1E-1: the horseshoe's third TP 30 yr after the jump), the last
  regime's turning points are not counted twice (277617 1M-1 20 kyr);
- the centre of the last regime is low-passed on its own samples (`geometry._stitched_centre` puts it on the record's
  grid; the synthetic captures after a circulation are in tests/resonances/independent_w9).
"""

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, Flag, Status

S = settings_from_config()


def _analysed(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), S)


def test_a_turning_point_inside_the_jump_zone_stays_before_the_jump():
    # 2021 BA 1E-1, 1 kyr: a horseshoe with TPs at 134, 274 and 407 yr, the jump of a at 377 yr (tau 111 yr), then
    # circulation; 407 yr < 377 + tau/2 = 432 yr (the last regime's start), so the 3-TP episode stays (G1e transient)
    analysis = _analysed('coorbital__2021_BA__1E-1+0+0')
    geometry = analysis.geometry
    (jump,) = [jump for jump in geometry.jumps if not jump.kick]
    assert geometry.regime_start_yr == pytest.approx(jump.time_yr + geometry.jump_tau_yr / 2)
    (episode,) = geometry.episodes_before_jump
    assert episode.n_tp == 3 and jump.time_yr < episode.t_end_yr < geometry.regime_start_yr
    verdict = analysis.verdict
    assert (verdict.status, verdict.angle_status, verdict.final_regime) == (Status.TRANSIENT, AngleStatus.EPISODIC, 'circulation')


def test_runs_before_the_jump_end_before_the_last_regime():
    analysis = _analysed('coorbital__277617__1M-1+0+0__20kyr')
    geometry = analysis.geometry
    before = [run for run in geometry.runs if run.before_last_jump]
    after = [run for run in geometry.runs if not run.before_last_jump]
    assert before and after
    assert max(run.t_end_yr for run in before) < geometry.regime_start_yr <= min(run.t_start_yr for run in after)
    times = np.concatenate([run.tp_times_yr for run in geometry.runs])
    assert len(np.unique(times)) == len(times)  # no turning point in two runs


def test_the_last_regime_centre_is_its_own_low_pass():
    """277617 1M-1 20 kyr (circulation, the jump at 9.2 kyr, then a quasi-satellite): the last regime's centre on the
    record's grid is the low-pass of the regime's own samples only (the synthetic captures: independent_w9)."""
    analysis = _analysed('coorbital__277617__1M-1+0+0__20kyr')
    geometry = analysis.geometry
    t = np.asarray(analysis.prep.t_yr, dtype=float)
    assert Flag.LAST_REGIME_AFTER_JUMP in geometry.flags
    assert len(geometry.centre.t_yr) == len(t) and np.all(np.isfinite(geometry.centre.centre_rad))
    own = t >= geometry.regime_start_yr
    angle_f = np.asarray(analysis.prep.angle_f, dtype=float)
    expected = g.centre_track(
        t[own],
        angle_f[own],
        geometry.period_cycle_yr,
        analysis.prep.cutoff.cutoff_yr,
        geometry.turning_points.prominence_rad,
        geometry.window,
        S,
    )
    np.testing.assert_allclose(geometry.centre.centre_rad[own], expected.centre_rad)
    turns = geometry.centre.turns
    np.testing.assert_allclose(t[turns.indices], turns.times_yr)  # the turn indices point into the record's grid
