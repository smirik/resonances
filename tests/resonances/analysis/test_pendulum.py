"""pendulum: V_F without a free constant and the eligible-window count of R0 with a (W28 B3)."""

import numpy as np
import pytest

from resonances.resonance.analysis import pendulum
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue

S = AnalysisSettings()


def test_force_fit_without_constant_recovers_two_harmonics_on_a_short_arc():
    # c x_dot = -V'(sigma) for V = a1 cos s + b1 sin s + a2 cos 2s + b2 sin 2s, sampled on 100 deg only: the force
    # columns alone determine V exactly (a constant column would be nearly collinear with them on this arc)
    a1, b1, a2, b2, c = -1.0e-4, 2.0e-5, 3.0e-5, -1.0e-5, -0.7
    sigma = np.radians(np.linspace(230.0, 330.0, 400))
    minus_v_prime = a1 * np.sin(sigma) - b1 * np.cos(sigma) + 2 * a2 * np.sin(2 * sigma) - 2 * b2 * np.cos(2 * sigma)
    fitted = pendulum.fit_potential_from_force(sigma, minus_v_prime / c, c, 2)
    assert np.asarray(fitted.cos_coeffs) == pytest.approx([a1, a2], rel=1e-8)
    assert np.asarray(fitted.sin_coeffs) == pytest.approx([b1, b2], rel=1e-8)
    # a known frame term is taken off the target first: c x_dot = -V' + s'' gives back the same V
    shifted = pendulum.fit_potential_from_force(sigma, (minus_v_prime + 3.0e-6) / c, c, 2, frame_term=np.full_like(sigma, 3.0e-6))
    assert np.asarray(shifted.cos_coeffs) == pytest.approx([a1, a2], rel=1e-8)


def test_agree_fraction_counts_eligible_windows_only():
    r = [0.95, 0.5, 0.97, 0.6]
    assert pendulum.agree_fraction(r, S) == pytest.approx(0.5)  # 2 of 4
    assert pendulum.agree_fraction(r, S, [True, False, True, False]) == pytest.approx(1.0)  # 2 of 2
    assert pendulum.with_slow_check(CheckValue.PASS, r, S) == CheckValue.FAIL
    assert pendulum.with_slow_check(CheckValue.PASS, r, S, [True, False, True, False]) == CheckValue.PASS
    assert pendulum.with_slow_check(CheckValue.PASS, r, S, [False] * 4) == CheckValue.NOT_MEASURED
    assert pendulum.with_slow_check(CheckValue.FAIL, r, S, [True] * 4) == CheckValue.FAIL  # sigma-only passed through
