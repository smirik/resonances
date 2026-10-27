"""Q12: the second scale of G2c, W_c from the turns of the centre itself (`geometry.centre_turns_window_yr`).

sigma = pi + 2.1 sin(2 pi t / 40 kyr) + 0.05 sin(2 pi t / 3.5 kyr + phase), step 50 yr, 200 kyr. The 0.05-rad wave shows
only near the slow extremes (its slope 0.05 x 2pi/3.5 = 0.09 rad/kyr is below the slow 2.1 x 2pi/40 = 0.33 rad/kyr
elsewhere). At phase 1.3 it adds a turning-point pair at two of the ten slow extremes; the fine P_cyc is then ~30 kyr
and 3 P_cyc = 90 kyr is above the libration. The turns of the centre (prominence p_c = 10 deg, the wave pairs swing
2-3 deg) are the ten slow extremes: P_t = 40 kyr, W_c = P_t / 2 = 20 kyr (FILTER_MIN_LIBRATION_CUTOFFS = 2).
"""

from dataclasses import replace

import numpy as np
import pytest

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.independent_w5.test_w5_fixes import _synthetic

S = settings_from_config()
P_SLOW = 40_000.0


def _fine_angle(phase):
    t = np.arange(0.0, 200_000.0 + 25.0, 50.0)
    sigma = np.pi + 2.1 * np.sin(2 * np.pi * t / P_SLOW) + 0.05 * np.sin(2 * np.pi * t / 3500.0 + phase)
    f = filtering.lowpass(t, sigma, 1000.0, S.filter_order)
    return geo.FilteredAngle(
        t_yr=t,
        angle_raw=sigma,
        angle_f=f,
        rate_f=np.gradient(f, t),
        angle_f_half=f,
        angle_f_double=f,
        slow_f=np.zeros_like(t),
        identity_coefficient=float('nan'),
        cutoff_yr=1000.0,
        step_yr=50.0,
        window=filtering.edge_window(t, 1000.0),
    )


@pytest.mark.parametrize('phase', [0.0, 1.3, 2.5])
def test_the_turns_of_the_centre_give_half_the_slow_period(phase):
    # the wave moves each slow extreme by up to ~1 kyr (phase 0: P_t 38.9 kyr), hence 5 %
    window = geo.centre_turns_window_yr(_fine_angle(phase), geo.centre_prominence(np.radians(1.0), S), S)
    assert window == pytest.approx(P_SLOW / S.filter_min_libration_cutoffs, rel=0.05)


def test_no_turns_of_the_centre_no_window():
    angle = _fine_angle(0.0)
    flat = replace(angle, angle_f=np.full_like(angle.t_yr, np.pi))
    assert np.isnan(geo.centre_turns_window_yr(flat, np.radians(10.0), S))
    assert geo.angle_at_centre_window(angle, float('nan'), S) is None
    assert geo.angle_at_centre_window(angle, 1000.0, S) is None  # W_c = P_c


@pytest.mark.parametrize('phase, basis', [(0.0, 'centre_turns'), (1.3, 'centre_turns'), (2.5, 'centre_turns')])
def test_g2c_tries_the_turns_of_the_centre_after_p_cyc(phase, basis):
    """The first scale (~90 kyr) loses the libration and the second (~20 kyr) finds it. Phase 0 used to decide at the
    first scale: one 3-TP wave pair at 8.4-11.0 kyr (298.2, 297.2, 301.1 deg: 1.1 deg back, 3.9 deg on) was an episode
    and gave P_cyc 2.55 kyr. It closes no cycle (net 2.8 deg > mean swing 2.5 deg, W13), so P_cyc is the runs' 29.9 kyr
    as at the other phases."""
    analysis = analyse_mmr(_synthetic('7N-9r6', 2.1, 0.05, phase), S, scan_subresonances=False, scan_owner=False)
    assert analysis.verdict.angle_status == 'libration'
    assert analysis.extra_columns['geo_centre_scale_basis'] == basis
    if basis == 'centre_turns':
        # 5 % as in test_the_turns_of_the_centre_give_half_the_slow_period (phase 0: P_t 38.9 kyr, W_c 19.45 kyr)
        assert analysis.extra_columns['geo_centre_scale_cutoff_yr'] == pytest.approx(P_SLOW / 2, rel=0.05)
        # the fine P_cyc: the runs' 29.9 kyr, or, where a slow half-swing pair is closed by its phase return (W20, a
        # cycle of two turning points), that cycle's own period, P_t
        fine = analysis.extra_columns['geo_fine_period_cycle_yr']
        assert fine == pytest.approx(29_900.0, rel=0.02) or fine == pytest.approx(P_SLOW, rel=0.05)
