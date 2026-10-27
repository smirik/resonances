"""W3 guards, F2: a jump of a needs data on both sides; a candidate at the edge of the window cuts nothing.

Codex sol (REPORT.md, P1): 31935 9M-16+0+7 at the cutoff x2 (P_c 10 -> 20 yr) found 'jumps' 20 yr after the start
and 20 yr before the end of a 100-kyr record (tau = 40 yr), confirmed them by a alone and cut the record to the
inverted window [100000, 99980]: 60 TPs -> 0, circulation -> insufficient_time. A step needs max(tau,
libration_jump_edge_periods x P_cyc) of the window on each side (`geometry.jump_edge_margin_yr`).
"""

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, Flag, Status

S = settings_from_config()
DEG = np.pi / 180
CUTOFF = 50.0  # tau = max(2 x 50, 5 x 1) = 100 yr (LIBRATION_JUMP_TAU_CUTOFFS 2, _STEPS 5)
PERIOD = 200.0


def libration_with_a_step(step_time: float) -> geo.FilteredAngle:
    """A 200-yr libration of 1 rad whose amplitude jumps 0.5 -> 1.5 rad with a 1e-3 au step of a at `step_time`
    (h = sin^2(A/2): 0.061 -> 0.465, dh = 0.40 > LIBRATION_JUMP_DH_MIN 0.1)."""
    rng = np.random.default_rng(3)
    t = np.arange(0.0, 20000.0 + 1e-9, 1.0)
    amplitude = np.where(t < step_time, 0.5, 1.5)
    sigma = np.pi + amplitude * np.sin(2 * np.pi * t / PERIOD)
    a = 2.5 + 1e-3 * 0.5 * (1 + np.tanh((t - step_time) / 5.0)) + 1e-7 * rng.standard_normal(t.size)
    return geo.FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=sigma,
        rate_f=np.gradient(sigma, t),
        angle_f_half=sigma,
        angle_f_double=sigma,
        slow_f=a,
        identity_coefficient=-0.5,
        cutoff_yr=CUTOFF,
        step_yr=1.0,
        window=(CUTOFF, 20000.0 - CUTOFF),
    )


def finished(step_time: float) -> geo.Geometry:
    angle = libration_with_a_step(step_time)
    geometry = geo.measure_geometry(angle, 1 * DEG, S)
    cycles = np.arange(100.0, 20000.0, PERIOD)
    h = np.where(cycles < step_time, np.sin(0.25) ** 2, np.sin(0.75) ** 2)
    return geo.finish_geometry(geometry, angle, 190.0, 0.0, 1.2, 0.02, cycles, h, 0.05, PERIOD, [], S)


def test_the_margin_by_hand():
    assert geo.jump_edge_margin_yr(100.0, 200.0, S) == 200.0  # one period > tau
    assert geo.jump_edge_margin_yr(100.0, 40.0, S) == 100.0  # tau > one period
    assert geo.jump_edge_margin_yr(100.0, float('nan'), S) == 100.0


def test_a_step_inside_the_record_cuts_it():
    geometry = finished(10000.0)
    cutting = [jump for jump in geometry.jumps if not jump.kick]
    assert len(cutting) == 1 and cutting[0].time_yr == pytest.approx(10000.0, abs=100.0)
    assert 10000.0 <= geometry.regime_start_yr <= 10200.0 and geometry.window[0] < geometry.window[1]


@pytest.mark.parametrize('step_time', [19900.0, 100.0])  # 50 yr from the window end / start: < 200 yr margin
def test_a_step_at_the_edge_is_kept_but_cuts_nothing(step_time):
    geometry = finished(step_time)
    assert [jump for jump in geometry.jumps if not jump.kick] == []
    edge = [jump for jump in geometry.jumps if jump.at_edge]
    assert len(edge) == 1 and edge[0].time_yr == pytest.approx(step_time, abs=60.0)
    assert geometry.regime_start_yr == 0.0 and geometry.window == (CUTOFF, 20000.0 - CUTOFF)
    assert str(Flag.JUMP_AT_EDGE) in geometry.flags and str(Flag.LAST_REGIME_AFTER_JUMP) not in geometry.flags
    columns = geometry.to_columns()
    assert columns['jump_at_edge'] == '1' and columns['jump_edge_margin_yr'] == PERIOD


def test_31935_at_the_doubled_cutoff_keeps_its_window():
    example = 'mmr2__31935__9M-16+0+7'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = io.load_series(example)
    base = analyse_mmr(series, S, scan_owner=False, scan_subresonances=False)
    doubled = analyse_mmr(
        series, settings_from_config(mmr_filter_cutoff_yr=2 * base.prep.cutoff.cutoff_yr), scan_owner=False, scan_subresonances=False
    )
    window = doubled.geometry.window
    assert window[0] < window[1] and doubled.geometry.regime_start_yr == series.t_yr[0]
    assert [jump for jump in doubled.geometry.jumps if not jump.kick] == []  # nothing cuts the record
    # sol's 'jumps' next to the edges of the filtered series (20 and 99980 yr at 2 P_c = 20 yr, 150 yr at 2 x 75 yr):
    # kept as at-edge kicks, closer to an edge than the margin a confirmation needs
    edge = [jump.time_yr for jump in doubled.geometry.jumps if jump.at_edge]
    t0, t1 = float(series.t_yr[0]), float(series.t_yr[-1])
    assert edge and all(min(time - t0, t1 - time) <= doubled.geometry.jump_edge_margin_yr for time in edge)
    assert len(doubled.geometry.turning_points) == len(base.geometry.turning_points) == 60
    # the same verdict at both cutoffs. Without the member scan the catalogue angle's own libration about the moving
    # minimum of the averaged potential (W18) makes it a transient: the author's angle label since 2026-10-16 (episodic,
    # DECISIONS.md; the circulation of 2026-10-15 is withdrawn)
    for analysis in (base, doubled):
        assert (analysis.verdict.status, analysis.verdict.angle_status) == (Status.TRANSIENT, AngleStatus.EPISODIC)
        assert any(e.accepted and str(e.shape) == 'moving_centre' and e.t_start_yr > 30000.0 for e in analysis.geometry.runs)
