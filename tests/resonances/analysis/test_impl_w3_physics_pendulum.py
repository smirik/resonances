"""W3 G7: the action and the half-width of a multi-harmonic V (analysis.pendulum).

Truths: Codex astra §8 (`action_probe.py`, the closed contour integrated directly): V = -cos s - 0.2 cos 2s gives
J/J_s = 0.3841667556 at h = .5 and 0.0706589394 at h = .1, where the cosine formula E(h) - (1-h) K(h) gives
0.4236065424 and 0.0795604230. Here the same numbers come from a brute-force trapezoid over a fine grid (no root
finding, no change of variable) and from a trajectory integrated by solve_ivp. The separatrix rate of V = -cos 2s is
sqrt(2 x depth 2) = 2, while 2 sqrt(V'') = 4.
"""

import numpy as np
import pytest
from scipy import integrate

from resonances.resonance.analysis import pendulum as pend

GRID = 400001


def _brute_force_ratio(potential, h):
    """J/J_s by the trapezoid rule on a 4e5-point grid: the well's own separatrix is the lower of its two barriers."""
    x = potential.well_rad + np.linspace(-np.pi, np.pi, GRID)
    v = potential.value(x)
    centre = GRID // 2

    def barrier(step):
        i = centre
        while 0 < i + step < GRID - 1 and v[i + step] > v[i]:
            i += step
        return i

    right, left = barrier(1), barrier(-1)
    separatrix = min(v[right], v[left])

    def action(energy):
        inside = np.zeros(GRID, dtype=bool)
        i = centre
        while i < GRID - 1 and v[i] < energy:
            inside[i] = True
            i += 1
        i = centre
        while i > 0 and v[i] < energy:
            inside[i] = True
            i -= 1
        p = np.where(inside, np.sqrt(2 * np.clip(energy - v, 0.0, None)), 0.0)
        return np.trapezoid(p, x) / np.pi

    energy = potential.v_min + h * (potential.v_max - potential.v_min)
    return action(energy) / action(separatrix)


def test_codex_two_harmonic_action():
    potential = pend.potential_from_coefficients([-1.0, -0.2], [0.0, 0.0])
    assert pend.action_ratio_in_potential(potential, 0.5) == pytest.approx(0.3841667556, abs=1e-9)
    assert pend.action_ratio_in_potential(potential, 0.1) == pytest.approx(0.0706589394, abs=1e-9)
    assert pend.action_ratio_in_potential(potential, 0.5) == pytest.approx(_brute_force_ratio(potential, 0.5), abs=1e-5)
    # the cosine formula is 10 % off here
    assert pend.action_ratio(0.5) == pytest.approx(0.4236065424, abs=1e-9)
    assert pend.action_ratio_in_potential(potential, 1.0) == 1.0 and pend.action_ratio_in_potential(potential, 0.0) == 0.0
    assert np.isnan(pend.action_ratio_in_potential(potential, float('nan')))


def test_one_harmonic_is_the_closed_form():
    cosine = pend.potential_from_coefficients([-1.0], [0.0])
    almost = pend.potential_from_coefficients([-1.0, -1e-12], [0.0, 0.0])  # the quadrature path on the same V
    # J_s = 8 Omega / pi (Codex astra §4) by the closed form and by the quadrature
    assert pend.separatrix_action(cosine) == pytest.approx(8 / np.pi)
    assert pend.separatrix_action(almost) == pytest.approx(8 / np.pi, rel=1e-9)
    for h in (0.05, 0.5, 0.95):
        assert pend.action_ratio_in_potential(cosine, h) == pend.action_ratio(h)
        assert pend.action_ratio_in_potential(almost, h) == pytest.approx(pend.action_ratio(h), abs=1e-9)


def test_asymmetric_well_uses_its_own_lower_barrier():
    # V = -cos s - 0.8 cos 2s + 0.3 sin s: the deep well at s ~ 0 is bounded by two maxima of different height; the lower
    # one is the separatrix of this well (the global top is higher)
    potential = pend.potential_from_coefficients([-1.0, -0.8], [0.3, 0.0])
    left, right, separatrix = pend.well_separatrix(potential)
    v_left, v_right = (float(potential.value(np.atleast_1d(x))[0]) for x in (left, right))
    assert separatrix == pytest.approx(min(v_left, v_right)) and abs(v_left - v_right) > 0.1
    assert separatrix < potential.v_max
    for h in (0.2, 0.5):
        assert pend.action_ratio_in_potential(potential, h) == pytest.approx(_brute_force_ratio(potential, h), abs=1e-5)


def test_half_width_of_a_multi_harmonic_well():
    double = pend.potential_from_coefficients([0.0, -1.0], [0.0, 0.0])  # V = -cos 2s: depth 2, V'' = 4 at the well
    assert double.omega_rad_yr == pytest.approx(2.0)
    assert pend.half_width_in_potential(double, 1.0) == pytest.approx(2.0)
    assert pend.half_width(double.omega_rad_yr, 1.0) == pytest.approx(4.0)  # the cosine formula does not apply
    cosine = pend.potential_from_coefficients([-0.25], [0.0])  # Omega = 0.5: 2 Omega / |c| = 2 with c = 0.5
    assert pend.half_width_in_potential(cosine, -0.5) == pytest.approx(2.0)
    assert pend.half_width(cosine.omega_rad_yr, -0.5) == pytest.approx(2.0)


def test_cycle_energies_take_the_action_of_the_fitted_potential():
    potential = pend.potential_from_coefficients([-1.0, -0.2], [0.0, 0.0])

    def rhs(_, y):
        return [y[1], -(np.sin(y[0]) + 0.4 * np.sin(2 * y[0]))]  # -V'(s)

    h = 0.5
    rate0 = np.sqrt(2 * h * potential.depth)  # at the bottom of the well, V = V_min
    period = pend.period_in_potential(potential, h)
    t = np.linspace(0.0, 3 * period, 30001)
    sol = integrate.solve_ivp(rhs, (0.0, t[-1]), [0.0, rate0], t_eval=t, rtol=1e-11, atol=1e-12)
    cycles = pend.cycle_energies(t, sol.y[0], sol.y[1], [(0.0, period, False), (period, 2 * period, False)], potential)
    assert len(cycles) == 2
    for cycle in cycles:
        assert cycle.h == pytest.approx(h, abs=1e-6)
        assert cycle.action_ratio == pytest.approx(0.3841667556, abs=1e-6)
        assert cycle.period_law_ratio == pytest.approx(1.0, abs=1e-6)
