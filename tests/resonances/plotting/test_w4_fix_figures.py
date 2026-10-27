"""The two display fixes of W4: a 1:1 angle's panel (e) shows its own law, not the harmonic windows, and the secular/ZLK
panel (b) labels its right-angle ticks in (-180, 180] (its window starts wherever the centre puts it, often below 0)."""

import math
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from scipy.integrate import solve_ivp

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting.secular_plots import draw_long_period_combined  # noqa: E402
from resonances.resonance.analysis.mmr.analyse import analyse_mmr  # noqa: E402
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec  # noqa: E402
from resonances.resonance.analysis.secular import LongPeriodSeries  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import ResonanceType  # noqa: E402
from resonances.resonance.analysis.zlk import analyse_zlk  # noqa: E402

SETTINGS = settings_from_config()
MU_EARTH = 1.0 / 332946.05
N_EARTH = math.radians(360.000505)


def _tadpole_series():
    """Planar circular Earth tadpole: theta'' = -3 mu n^2 dR/dtheta, R = 1/(2|sin(theta/2)|) - cos theta (Murray &
    Dermott 1999, eqs. 3.178-3.179), from theta = 30 deg at rest; a from theta_dot = n(a) - n_E."""

    def force(theta):
        s = np.sin(theta / 2)
        return -0.25 * np.sign(s) * np.cos(theta / 2) / s**2 + np.sin(theta)

    t = np.arange(0.0, 3000.0, 1.0)
    rhs = lambda _, y: [y[1], -3 * MU_EARTH * N_EARTH**2 * force(y[0])]  # noqa: E731
    sol = solve_ivp(rhs, (0.0, t[-1]), [math.radians(30.0), 0.0], t_eval=t, rtol=1e-11, atol=1e-13)
    theta, theta_dot = sol.y
    a = (2 * np.pi / (N_EARTH + theta_dot)) ** (2.0 / 3.0)
    nr = len(t)
    lam_p = N_EARTH * t
    lam = lam_p + theta
    varpi = np.full(nr, 1.0)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full(nr, 1e-3),
        inc=np.full(nr, 1e-3),
        Omega=np.zeros(nr),
        omega=varpi.copy(),
        varpi=varpi,
        angle_wrapped=np.mod(lam_p - lam, 2 * np.pi),
        resonance=ResonanceSpec.from_key('1E-1'),
        regime=Regime(step_yr=1.0, span_yr=float(t[-1]), n_rows=nr),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name='tadpole',
    )


def _texts(ax):
    return [text.get_text() for text in ax.texts]


def test_coorbital_panel_e_shows_the_1_1_law_and_no_harmonic_window():
    series = _tadpole_series()
    analysis = analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False)
    law = analysis.response.coorbital
    assert law is not None and analysis.response.force_law == 'coorbital'
    fig = plt.figure(figsize=(12, 10))
    axes = mc.draw_time_panels(fig, analysis, series, mc.LAYOUT_V1)
    panel_e = axes[4]
    # only the numbers of the 1:1 law (no words: the regime is in the header, the check in the right column)
    notes = [text for text in _texts(panel_e) if text.startswith('R² ')]
    assert len(notes) == 1
    assert notes[0] == f'R² {law.r2:.2f} ({law.r2_h1:.2f} / {law.r2_h2:.2f})   K/K_th {law.scale:.2f}'
    # only a_dot itself is drawn (solid, and pale dashed in the filter's edge spread): no fitted sine of the windows
    assert [line.get_linestyle() for line in panel_e.lines] == ['-', '--']
    assert {line.get_color() for line in panel_e.lines} == {mc.INK}
    plt.close(fig)


def test_secular_panel_b_ticks_are_wrapped_right_angles():
    # a libration about 10 deg: the 360-degree window around the centre starts at -170 deg; its ticks are right angles
    # labelled in [0, 360) (the MMR figure's convention, author 2026-10-23: 270, not -90), not the window's raw values
    t = np.linspace(0.0, 1e5, 5001)
    angle = np.radians(10.0) + 0.5 * np.sin(2 * np.pi * t / 1e4)
    series = LongPeriodSeries(
        ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, angle, np.full(5001, 2.5), np.full(5001, 0.2), np.full(5001, 0.6)
    )
    analysis = analyse_zlk(series, SETTINGS, gate_config=SimpleNamespace(free_gate_enabled=False, free_gate_min_cycles=3.0))
    fig = draw_long_period_combined(analysis)
    fig.canvas.draw()
    panel_b = next(ax for ax in fig.axes if ax.get_ylabel() == 'σ mod 360, °')
    low, _ = panel_b.get_ylim()
    assert low < 0  # the window really starts below 0
    labels = [tick.get_text() for tick in panel_b.get_yticklabels() if tick.get_text()]
    assert labels == ['270', '0', '90']
    assert all(0 <= float(label) < 360 and float(label) % 90 == 0 for label in labels)
    plt.close(fig)


def test_energy_panel_with_cycles_without_a_potential_top():
    # every cycle of 1998 QA62 1V-5M+3 has h = inf (no separatrix to scale by): the panel draws, its limits stay finite
    cycles = [SimpleNamespace(t_start_yr=100.0 * i, t_end_yr=100.0 * (i + 1), h=math.inf, rotation=True) for i in range(5)]
    frame = SimpleNamespace(
        analysis=SimpleNamespace(energy=SimpleNamespace(cycles=cycles, zone=None)), times=lambda t: np.asarray(t, float) / 1e3
    )
    fig, ax = plt.subplots()
    mc.draw_energy_panel(ax, frame, mc.LAYOUT_V1)
    assert ax.get_ylim() == pytest.approx((1 / 3, 3.0))  # log axis framed by h = 1 alone; inf h are edge triangles
    plt.close(fig)
