"""W3 G1: the free/forced branches of the eccentricity vector (analysis.mmr.kinematics).

Truth: z = e_free exp(i g t) + e_forced exp(i g5 t), varpi_dot = Im(conj(z) z_dot)/|z|^2 taken numerically from the
argument of z built here; the closed ends g5 + (g - g5) e_f/(e_f +- e_F) at cos theta = +-1 (Codex astra §6:
e_free = .05, e_forced = .2, g = 1, g5 = 0 gives [-1/3, .2], the old code [.8, 4/3]).
"""

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import kinematics as kin
from resonances.resonance.analysis.mmr import models as m
from resonances.resonance.analysis.mmr import response as r
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.impl_response_pendulum_synthetic import Record

SETTINGS = settings_from_config()
ARCSEC = np.pi / 180 / 3600


def _vector(e_free, e_forced, g, g5, cycles=4.0, n=200001, phase=0.3):
    t = np.linspace(0.0, cycles * 2 * np.pi / abs(g - g5), n)
    z = e_free * np.exp(1j * (g * t + phase)) + e_forced * np.exp(1j * g5 * t)
    return t, z


def _numeric_rate_range(t, z):
    rate = np.gradient(np.unwrap(np.angle(z)), t)
    return rate[5:-5].min(), rate[5:-5].max()


def test_codex_counterexample_forced_dominant_branch():
    # e_free = .05 < e_forced = .2: the vector turns at g5 = 0 on average; varpi_dot in [-1/3, .2]
    t, z = _vector(0.05, 0.2, 1.0, 0.0)
    low_true, high_true = _numeric_rate_range(t, z)
    assert (low_true, high_true) == pytest.approx((-1 / 3, 0.2), abs=1e-4)
    e_free, e_forced, g = kin.free_forced_split(t, np.abs(z), np.angle(z), 0.0)
    assert (e_free, e_forced, g) == pytest.approx((0.05, 0.2, 1.0), rel=1e-6)
    assert kin.varpi_rate_range(e_free, e_forced, g, 0.0) == pytest.approx((-1 / 3, 0.2), rel=1e-6)


def test_the_same_e_series_with_the_branches_swapped():
    # |z| is symmetric in the swap: e alone cannot tell; the vector can ([.8, 4/3] for e_free = .2 > e_forced = .05)
    t, swapped = _vector(0.2, 0.05, 1.0, 0.0)
    _, forced_dominant = _vector(0.05, 0.2, 1.0, 0.0)
    assert np.abs(swapped).min() == pytest.approx(np.abs(forced_dominant).min(), abs=1e-9)
    assert np.abs(swapped).max() == pytest.approx(np.abs(forced_dominant).max(), abs=1e-9)
    split = kin.free_forced_split(t, np.abs(swapped), np.angle(swapped), 0.0)
    assert split == pytest.approx((0.2, 0.05, 1.0), rel=1e-6)
    assert kin.varpi_rate_range(*split, 0.0) == pytest.approx(_numeric_rate_range(t, swapped), abs=1e-4)
    assert kin.varpi_rate_range(*split, 0.0) == pytest.approx((0.8, 4 / 3), rel=1e-6)


@pytest.mark.parametrize('e_free, e_forced, g, g5', [(0.05, 0.2, -0.5, 0.2), (0.15, 0.05, 2e-4, 2.0640e-5), (0.03, 0.1, 5e-4, 2.0640e-5)])
def test_split_and_range_against_the_vector(e_free, e_forced, g, g5):
    t, z = _vector(e_free, e_forced, g, g5)
    split = kin.free_forced_split(t, np.abs(z), np.angle(z), g5)
    assert split == pytest.approx((e_free, e_forced, g), rel=1e-6)
    assert kin.varpi_rate_range(*split, g5) == pytest.approx(_numeric_rate_range(t, z), rel=1e-3, abs=1e-12)


def test_a_record_shorter_than_one_modulation_cycle():
    # 0.6 of 2 pi/|g - g5|: the projection alone is biased; the least-squares g is not
    t, z = _vector(0.05, 0.2, 1.0, 0.1, cycles=0.6, n=40001)
    assert kin.free_forced_split(t, np.abs(z), np.angle(z), 0.1) == pytest.approx((0.05, 0.2, 1.0), rel=1e-6)


def test_e_through_zero_and_degenerate_vectors():
    # e_min = 0 (e_free = e_forced): varpi undefined at e = 0, its rate unbounded; never a ZeroDivisionError
    e_free, e_forced, low, high, period = kin.varpi_rate_modulation(np.array([0.0, 0.1, 0.2]), 1.0, 0.0)
    assert (e_free, e_forced) == (0.1, 0.1) and (low, high) == (-np.inf, np.inf) and period == pytest.approx(2 * np.pi)
    assert kin.varpi_rate_range(0.1, 0.1, 1.0, 0.0) == (-np.inf, np.inf)
    # a forced vector alone turns at g5 whatever g; nothing at all: undefined
    assert kin.varpi_rate_range(0.0, 0.2, np.nan, 0.3) == (0.3, 0.3)
    assert np.isnan(kin.varpi_rate_range(0.0, 0.0, 1.0, 0.0)).all()
    t = np.linspace(0.0, 100.0, 1001)
    e_free, e_forced, g = kin.free_forced_split(t, np.full_like(t, 0.2), 0.3 * t, 0.3)
    assert e_free == 0.0 and e_forced == pytest.approx(0.2) and np.isnan(g)
    assert kin.varpi_rate_range(e_free, e_forced, g, 0.3) == (0.3, 0.3)


def test_analyse_kinematics_reads_the_branch_from_the_slow_vector():
    # a forced-dominant record: varpi turns at g5 on average, so its slope is NOT the free frequency
    g5, g = 2.0640e-5, 5e-4
    t = np.arange(0.0, 200000.0, 10.0)
    z = 0.05 * np.exp(1j * (g * t + 0.3)) + 0.2 * np.exp(1j * g5 * t)
    varpi, e = np.unwrap(np.angle(z)), np.abs(z)
    assert np.polyfit(t, varpi, 1)[0] == pytest.approx(g5, rel=0.05)  # the slope says g5
    angle = 3 * varpi + np.pi
    spec = m.ResonanceSpec('1N-4+0+3', m.ResonanceKind.TWO_BODY, ('Neptune',), (1,), -4, 3, (0,))
    rec = Record(
        t, angle, np.gradient(angle, t), np.full_like(t, 30.0), -4, 30.0, varpi, e, 1e-5 * t, spec,
        (t[0] + 1000, t[-1] - 1000), centre=angle, slow_window_yr=10000.0, planet_rates={'Jupiter': g5},
    )  # fmt: skip
    resp = r.analyse_response(rec.prep, rec.slow, rec.geometry, rec.context, rec.sampling, SETTINGS, spec=rec.spec)
    k = kin.analyse_kinematics(rec.series, rec.prep, rec.slow, rec.geometry, rec.context, resp, SETTINGS)
    assert (k.e_free, k.e_forced) == pytest.approx((0.05, 0.2), rel=1e-4)
    low = g5 + (g - g5) * 0.05 / (0.05 - 0.2)
    high = g5 + (g - g5) * 0.05 / (0.05 + 0.2)
    assert (k.varpi_rate_pred_min_rad_yr, k.varpi_rate_pred_max_rad_yr) == pytest.approx((low, high), rel=1e-4)
    assert k.modulation_period_yr == pytest.approx(2 * np.pi / (g - g5), rel=1e-4)
    columns = kin.kinematics_columns(k)
    assert columns['kin_varpi_dot_pred_min_arcsec_yr'] == pytest.approx(low / ARCSEC, rel=1e-4)
