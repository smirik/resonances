"""W4 status (W3-G7): the slope d(J/J_s)/dh and the inverse J/J_s -> h in a multi-harmonic potential.

`classify_cycles` and `edge_cycles` turned the scatter of h into a scatter of J/<J_s> with the cosine slope K(h)/2 and
the zone's h_lo came from the cosine inverse; for V = -cos s - 0.2 cos 2s that is the wrong pendulum. The general slope
is dJ/dE = T/(2 pi) (Landau & Lifshitz, Mechanics §49), so d(J/J_s)/dh = depth T(h) / (2 pi J_s). Checked here against
a finite difference of the quadrature `action_ratio_in_potential` (an independent route: it never uses T), against the
closed form for one harmonic, and by the round trip of the inverse.
"""

import numpy as np
import pytest
from scipy.special import ellipk

from resonances.resonance.analysis import pendulum as p
from resonances.resonance.analysis.pendulum import CycleClass, CycleEnergy, Zone, ZoneMethod
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
TWO = p.potential_from_coefficients([-1.0, -0.2], [0.0, 0.0])  # V = -cos s - 0.2 cos 2s
ONE = p.potential_from_coefficients([-1.0], [0.0])


@pytest.mark.parametrize('h', [0.1, 0.3, 0.5, 0.7])
def test_slope_is_the_derivative_of_the_action_ratio(h):
    step = 1e-4
    numeric = (p.action_ratio_in_potential(TWO, h + step) - p.action_ratio_in_potential(TWO, h - step)) / (2 * step)
    assert p.action_slope_in_potential(TWO, h) == pytest.approx(numeric, rel=1e-4)


@pytest.mark.parametrize('h', [0.1, 0.5, 0.9])
def test_one_harmonic_slope_is_k_over_two(h):
    assert p.action_slope_in_potential(ONE, h) == pytest.approx(ellipk(h) / 2, rel=1e-12)
    assert p.action_slope_in_potential(None, h) == pytest.approx(ellipk(h) / 2, rel=1e-12)


def test_two_harmonics_differ_from_the_cosine():
    """At h = 0.5 the second harmonic makes the well flatter at the top: slope and inverse are not the cosine's."""
    assert abs(p.action_slope_in_potential(TWO, 0.5) / (ellipk(0.5) / 2) - 1) > 0.02
    assert abs(p.h_from_action_ratio_in_potential(TWO, 0.5) - p.h_from_action_ratio(0.5)) > 0.01


@pytest.mark.parametrize('j', [0.05, 0.3, 0.6, 0.95])
def test_inverse_round_trip(j):
    h = p.h_from_action_ratio_in_potential(TWO, j)
    assert p.action_ratio_in_potential(TWO, h) == pytest.approx(j, abs=1e-9)
    assert p.h_from_action_ratio_in_potential(ONE, j) == pytest.approx(p.h_from_action_ratio(j), abs=1e-12)


def test_ends_and_missing_values():
    assert p.h_from_action_ratio_in_potential(TWO, 0.0) == 0.0
    assert np.isnan(p.h_from_action_ratio_in_potential(TWO, np.nan))
    assert np.isnan(p.action_slope_in_potential(TWO, np.nan))
    assert p.action_slope_in_potential(TWO, 1.0) == np.inf  # at V_max: at or above the well's separatrix


def test_classification_reads_the_slope_of_the_potential():
    """A libration cycle at J/<J_s> = 0.80 with sigma_h = 0.05 next to J_lo = 0.85 (margin 1 sigma): deep or zone
    depends on sigma_J, i.e. on the slope of the potential the cycle was measured in."""
    h = p.h_from_action_ratio_in_potential(TWO, 0.80)
    cycle = CycleEnergy(0.0, 1.0, False, h, 0.80, 1.0, 1.0)
    slope_two, slope_one = p.action_slope_in_potential(TWO, h), ellipk(h) / 2
    assert slope_two != pytest.approx(slope_one, rel=0.02)
    sigma_h = 0.05
    j_lo = 0.80 + S.separatrix_sigma_margin * sigma_h * (slope_two + slope_one) / 2  # between the two answers
    zone = Zone(ZoneMethod.NUMERIC, j_lo, 0.0, 1.5, j_lo, 1.0, False)
    two = p.classify_cycles([cycle], zone, sigma_h, S, TWO)[0]
    one = p.classify_cycles([cycle], zone, sigma_h, S)[0]
    assert {two, one} == {CycleClass.DEEP, CycleClass.ZONE}
    assert (two == CycleClass.DEEP) == (slope_two < slope_one)
