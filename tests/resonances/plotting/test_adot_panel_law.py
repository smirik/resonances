"""Panel (e) of the combined MMR figure draws the force law that decided, on the series it read (dev/status.md, W26).

Truths computed here, independently of the drawing code:
- a multiplet a_dot = 2e-6 sin(sigma) + 1e-6 cos(sigma + D) - 3e-7 sin(sigma - D) + 4e-7 (D = varpi - varpi_P) is
  recovered exactly by the multiplet fit: C_0 = 2e-6, C_1 = 1e-6 i, C_-1 = -3e-7, C = 4e-7 (C_j = A_j + i B_j);
- the multiplet law drawn in (e) is numpy's least squares of a_dot on [sin psi_j, cos psi_j, 1] over the union of the
  law's windows, and 1 - SS_res / SS_tot of the drawn curve against the drawn a_dot there is the law's R^2;
- under the line rule (a raised cutoff) the black a_dot is a_dot low-passed at that cutoff, not at the response
  cutoff, and the cutoff and the line are printed.
"""

import dataclasses

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting import style  # noqa: E402
from resonances.resonance.analysis.mmr import response as response_module  # noqa: E402
from resonances.resonance.analysis.mmr.analyse import analyse_mmr  # noqa: E402
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402

LAYOUT = mc.LAYOUT_V1
S = settings_from_config()


def teardown_function():
    plt.close('all')


def test_the_multiplet_fit_keeps_the_law_it_measured():
    t = np.linspace(0.0, 20000.0, 4001)
    sigma = 0.8 * np.sin(2 * np.pi * t / 700.0) + 0.3 * np.cos(2 * np.pi * t / 2300.0)
    offset = 2 * np.pi * t / 9000.0  # varpi - varpi_P
    adot = 2e-6 * np.sin(sigma) + 1e-6 * np.cos(sigma + offset) - 3e-7 * np.sin(sigma - offset) + 4e-7
    windows = [(0.0, 10000.0), (10000.0, 20000.0)]
    fit = response_module.multiplet_fit(t, sigma, adot, offset, (-1, 0, 1), windows, 0.8)
    assert fit.pooled == pytest.approx(np.array([-3e-7, 2e-6, 1e-6j]), abs=1e-15)
    assert fit.constant_au_yr == pytest.approx(4e-7, abs=1e-15)
    assert fit.windows == ((0.0, 10000.0), (10000.0, 20000.0)) and fit.r2 == pytest.approx(1.0, abs=1e-12)
    model = response_module.members_model(sigma, offset, fit.pooled, fit.harmonics, fit.constant_au_yr)
    assert model == pytest.approx(adot, abs=1e-15)


def _pendulum():
    """sigma = 1 rad x cos(2 pi t / 1000) about 0 and a answering it near 2:1 with Jupiter (2 kyr at 1 yr), varpi
    turning at 10 deg/kyr so that varpi - varpi_J moves."""
    n = 2001
    t = np.arange(n, dtype=float)
    n_jupiter = np.radians(30.3496)
    a_star = (np.pi / n_jupiter) ** (2 / 3)
    c = 3 * np.pi * a_star**-2.5
    omega = 2 * np.pi / 1000.0
    sigma = np.cos(omega * t)
    varpi = np.radians(10.0) * t / 1000.0
    # the pendulum's a plus a member of the multiplet (sigma + varpi - varpi_J): R^2 of the multiplet law below 1
    a = a_star - omega * np.sin(omega * t) / c + 2e-5 * np.sin(sigma + varpi + np.radians(30.3) * t / 1000.0)
    zero = np.zeros(n)
    return MMRSeries(
        t, a, np.full(n, 0.1), zero, zero, varpi, varpi, sigma % (2 * np.pi), ResonanceSpec.from_key('2J-1'), Regime(1.0, t[-1], n)
    )


def _model_lines(ax):
    return [line for line in ax.lines if to_hex(line.get_color()) == style.MODEL]


def _ink_line(ax):
    return next(line for line in ax.lines if to_hex(line.get_color()) == to_hex(style.INK) and line.get_linestyle() == '-')


def test_the_multiplet_law_is_drawn_on_its_windows():
    series = _pendulum()
    analysis = analyse_mmr(series, S, scan_subresonances=False, scan_owner=False)
    prep, t = analysis.prep, np.asarray(series.t_yr, float)
    psi = np.asarray(prep.angle_response, float)
    adot = np.asarray(prep.adot_response, float)
    rate = analysis.context.planet_varpi_rate_rad_yr['Jupiter']
    offset = response_module.multiplet_offset(analysis.slow, t, rate)
    windows = [(200.0, 1000.0), (1000.0, 1800.0)]
    fit = response_module.multiplet_fit(t, psi, adot, offset, (0, 1), windows, 0.8)
    fit = dataclasses.replace(fit, planet='Jupiter', planet_varpi_rate_rad_yr=rate)
    response = dataclasses.replace(analysis.response, force_law='multiplet', multiplet=fit, force_line_binding_yr=float('nan'))
    analysis = dataclasses.replace(analysis, response=response)
    frame = mc._Frame(analysis, series, LAYOUT, settings=S)
    assert np.all(mc._law_shift(frame) == 0.0)  # a stationary centre: psi is sigma

    fig = plt.figure(figsize=LAYOUT.size, dpi=LAYOUT.dpi)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.2))
    mc.draw_adot_panel(ax, frame, LAYOUT)

    lines = _model_lines(ax)
    assert len(lines) == 1  # one law, no single-sine windows
    drawn = np.asarray(lines[0].get_ydata(), float)
    black = np.asarray(_ink_line(ax).get_ydata(), float)
    inside = (t >= 200.0) & (t <= 1800.0) & frame.inside
    assert np.all(np.isnan(drawn[~inside]))
    # numpy's own least squares over the union of the windows
    design = np.column_stack([np.sin(psi), np.cos(psi), np.sin(psi + offset), np.cos(psi + offset), np.ones_like(psi)])
    coef, *_ = np.linalg.lstsq(design[inside], adot[inside], rcond=None)
    scale = drawn[inside][0] / (design[inside] @ coef)[0]
    assert drawn[inside] == pytest.approx(scale * (design[inside] @ coef), rel=1e-9)
    residual = black[inside] - drawn[inside]
    r2 = 1 - np.sum(residual**2) / np.sum((black[inside] - black[inside].mean()) ** 2)
    assert r2 == pytest.approx(fit.r2, abs=1e-9)
    halves = [value for value in (response.force_r2_h1, response.force_r2_h2) if np.isfinite(value)]
    single = f'{min(halves):.2f}' if halves else '–'  # two cycles: too short for single-law windows here
    assert f'multiplet law j = 0, 1: R² {fit.r2:.2f} (single {single})' in [text.get_text() for text in ax.texts]


def test_a_raised_cutoff_draws_the_series_the_law_read():
    series = _pendulum()
    analysis = analyse_mmr(series, S, scan_subresonances=False, scan_owner=False)
    raised = 3 * analysis.prep.response_cutoff_yr
    response = dataclasses.replace(analysis.response, force_law='single', force_cutoff_yr=raised, force_line_binding_yr=42.0)
    analysis = dataclasses.replace(analysis, response=response)
    frame = mc._Frame(analysis, series, LAYOUT, settings=S)
    fig = plt.figure(figsize=LAYOUT.size, dpi=LAYOUT.dpi)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.2))
    mc.draw_adot_panel(ax, frame, LAYOUT)

    expected = response_module.smoothed_series(analysis.prep, raised, S)[3]
    black = np.asarray(_ink_line(ax).get_ydata(), float)
    shown = np.isfinite(black)
    ratio = black[shown] / expected[shown]
    assert ratio == pytest.approx(np.full(shown.sum(), np.median(ratio)), rel=1e-9)  # the same curve, one scale
    base = np.asarray(analysis.prep.adot_response, float)
    assert not np.allclose(base[shown] * np.median(ratio), black[shown])  # not the response cutoff's a_dot
    assert f'cutoff {raised:.4g} yr (line 42 yr)' in [text.get_text() for text in ax.texts]
