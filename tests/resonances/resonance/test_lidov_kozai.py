from types import SimpleNamespace

import numpy as np
import pytest

from resonances import LidovKozaiResonance, LidovKozaiParameters


def _omega_span_deg(e0, inc0_deg, omega0_deg, tau_max=40.0, dtau=0.01):
    """Span of omega in the quadrupole secular problem, by RK4 on the standard equations
    (test particle, circular perturber, time in units of n / n_pert^2):

        de/dtau     = (15/8) e sqrt(1-e^2) sin^2 i sin 2w
        dw/dtau     = (3/4) [2(1-e^2) + 5 sin^2 w (e^2 - sin^2 i)] / sqrt(1-e^2)

    with i from the conserved (1-e^2) cos^2 i. Written here independently of the package.
    """
    e = np.atleast_1d(np.asarray(e0, dtype=float))
    w = np.deg2rad(np.atleast_1d(np.asarray(omega0_deg, dtype=float)))
    theta = (1 - e**2) * np.cos(np.deg2rad(inc0_deg)) ** 2

    def rhs(e, w):
        sin2i = 1 - theta / (1 - e**2)
        de = 15 / 8 * e * np.sqrt(1 - e**2) * sin2i * np.sin(2 * w)
        dw = 3 / 4 * (2 * (1 - e**2) + 5 * np.sin(w) ** 2 * (e**2 - sin2i)) / np.sqrt(1 - e**2)
        return de, dw

    w_min, w_max = w.copy(), w.copy()
    for _ in range(int(tau_max / dtau)):
        k1 = rhs(e, w)
        k2 = rhs(e + dtau / 2 * k1[0], w + dtau / 2 * k1[1])
        k3 = rhs(e + dtau / 2 * k2[0], w + dtau / 2 * k2[1])
        k4 = rhs(e + dtau * k3[0], w + dtau * k3[1])
        e = e + dtau / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w = w + dtau / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        w_min, w_max = np.minimum(w_min, w), np.maximum(w_max, w)
    return np.rad2deg(w_max - w_min)


@pytest.mark.parametrize(
    'e, inc, omega, c1, c2, h_quad',
    [
        # Values worked out by hand from Shevchenko (2017) eqs. 3.35, 3.43 and 3.41.
        # e=0.3, i=60, w=90: c1 = 0.91*0.25; c2 = 0.09*(0.4-0.75); h = 2.27*(-0.25) - 15*0.09*0.75
        (0.3, 60.0, 90.0, 0.2275, -0.0315, -1.58),
        # e=0.1, i=70, w=0: c1 = 0.99*cos^2(70) = 0.99*0.116978; c2 = 0.01*0.4; h = 2.03*(-0.649067) + 0.15*0.883022
        (0.1, 70.0, 0.0, 0.115808, 0.004, -1.185152),
        # e=0.3, i=45, w=90: c1 = 0.91*0.5; c2 = 0.09*(0.4-0.5); h = 2.27*0.5 - 15*0.09*0.5
        (0.3, 45.0, 90.0, 0.455, -0.009, 0.46),
        # e=0: c2 = 0 and h = 2(3 cos^2 i - 1) = 2(3 c1 - 1), the separatrix value (Shevchenko 2017, p. 42)
        (0.0, 50.0, 30.0, 0.413176, 0.0, 0.479055),
    ],
)
def test_lidov_integrals(e, inc, omega, c1, c2, h_quad):
    values = LidovKozaiParameters.evaluate(e, np.deg2rad(inc), np.deg2rad(omega))

    assert values == pytest.approx((c1, c2, h_quad), abs=1e-5)


def test_the_hamiltonian_is_a_combination_of_the_integrals():
    # Substituting cos 2w = 1 - 2 sin^2 w into h_quad gives h_quad = 6 c1 - 2 + 30 c2.
    rng = np.random.default_rng(1)
    e, inc, omega = rng.uniform(0, 0.95, 200), rng.uniform(0, np.pi, 200), rng.uniform(0, 2 * np.pi, 200)

    c1, c2, h_quad = LidovKozaiParameters.evaluate(e, inc, omega)

    assert h_quad == pytest.approx(6 * c1 - 2 + 30 * c2, abs=1e-12)


def test_the_resonance_centre_has_negative_c2():
    # Centre of the LK resonance at c1 = 0.25 (Shevchenko 2017, p. 34): w = 90,
    # e = sqrt(1 - sqrt(5 c1 / 3)) = 0.595401, cos i = (3 c1 / 5)^(1/4) = 0.622333.
    c1, c2, _ = LidovKozaiParameters.evaluate(0.595401, np.arccos(0.622333), np.pi / 2)

    assert c1 == pytest.approx(0.25, abs=1e-5)
    assert c2 < 0


@pytest.mark.parametrize(
    'e, inc, omega, librates',
    [
        (0.1, 70.0, 0.0, False),  # h_quad = -1.19 < 0, yet omega circulates: c2 = +0.004
        (0.3, 45.0, 90.0, True),  # h_quad = +0.46 > 0, yet omega librates: c2 = -0.009
        (0.3, 60.0, 90.0, True),
        (0.5, 50.0, 90.0, True),
        (0.5, 50.0, 0.0, False),
        (0.2, 30.0, 90.0, False),  # c1 > 3/5: always circulates
    ],
)
def test_the_libration_criterion_is_c1_below_three_fifths_and_c2_negative(e, inc, omega, librates):
    c1, c2, _ = LidovKozaiParameters.evaluate(e, np.deg2rad(inc), np.deg2rad(omega))
    span = _omega_span_deg(e, inc, omega)[0]

    assert bool(span < 360.0) == librates
    assert bool(c1 < 0.6 and c2 < 0) == librates


def test_lidov_kozai_resonance_angle_direct_value():
    resonance = LidovKozaiResonance()
    orbit = SimpleNamespace(omega=1.234)
    assert np.isclose(resonance.calc_angle(orbit, None), 1.234)
