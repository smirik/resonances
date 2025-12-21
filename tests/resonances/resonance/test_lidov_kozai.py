from types import SimpleNamespace

import numpy as np

from resonances import LidovKozaiResonance, LidovKozaiParameters


def test_lidov_kozai_parameters_scalar():
    e = 0.3
    inc = np.deg2rad(60.0)
    omega = np.deg2rad(90.0)

    c1, c2, c = LidovKozaiParameters.evaluate(e, inc, omega)

    assert np.isclose(c1, 0.2275, atol=1e-6)
    assert np.isclose(c2, -1.58, atol=1e-2)
    assert np.isclose(c, -0.0315, atol=1e-4)
    assert c1 < 3.0 / 5.0
    assert c2 < 0.0


def test_lidov_kozai_resonance_angle_direct_value():
    resonance = LidovKozaiResonance()
    orbit = SimpleNamespace(omega=1.234)
    assert np.isclose(resonance.calc_angle(orbit, None), 1.234)
