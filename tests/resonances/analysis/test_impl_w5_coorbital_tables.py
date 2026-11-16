"""W5: a co-orbital force table is rebuilt when the orbit's orientation turns, also at i = 0 (codex astra W5 finding 2).

At i = 0 a turn of the node turns the perihelion (varpi = omega + Omega) while e exp(i omega) and sin i exp(i Omega)
stay put; the reused table then describes another orbit. Numbers from the review: planet Orbit(0.1, 0, 0, 0), body
Orbit(0.2, 0, 0, 0) and Orbit(0.2, 0, 0, pi/2), 720 nodes, softening 0.001: F~(60 deg) = -0.0488978 and -0.15059082."""

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import coorbital as co

PLANET = co.Orbit(0.1, 0.0, 0.0, 0.0)
TOL_E, TOL_I = 0.01, np.radians(1.0)


def test_a_node_turn_at_zero_inclination_rebuilds_the_table():
    before, after = co.Orbit(0.2, 0.0, 0.0, 0.0), co.Orbit(0.2, 0.0, 0.0, np.pi / 2)
    assert co.needs_new_table((before, PLANET), (after, PLANET), TOL_E, TOL_I)
    forces = [co.averaged_table(body, PLANET, 720, 0.001).force[120] for body in (before, after)]  # theta = 60 deg
    assert forces == pytest.approx([-0.0488978, -0.15059082], rel=1e-5)


def test_prograde_and_retrograde_planes_are_told_apart():
    """sin i is the same for i = 80 and 100 deg; the orbit normal is not."""
    prograde, retrograde = co.Orbit(0.2, np.radians(80), 0.3, 1.0), co.Orbit(0.2, np.radians(100), 0.3, 1.0)
    assert co.needs_new_table((prograde, PLANET), (retrograde, PLANET), TOL_E, TOL_I)


def test_the_same_orbit_described_twice_keeps_its_table():
    """At i = 0 only varpi = omega + Omega matters: (omega, Omega) = (1, 0) and (0.4, 0.6) are one orbit."""
    one, same = co.Orbit(0.2, 0.0, 1.0, 0.0), co.Orbit(0.2, 0.0, 0.4, 0.6)
    assert not co.needs_new_table((one, PLANET), (same, PLANET), TOL_E, TOL_I)
