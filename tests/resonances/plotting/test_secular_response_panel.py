"""Panel (f) of the secular figure draws the fit whose R² it prints (docs/reading-figures.md, secular panel (f)).

The case: a 518 z1-like libration (3.3 Myr, 80°) carrying 12° waves of 430 kyr, so the centre scale decides
(`analyse_secular`, step 4), and an e that answers sin σ (Lagrange: e_dot ~ sin σ) plus its own 430-kyr wave. At the
centre's cutoff the wave is filtered out of e; at the fine P_c (100 kyr) it stays and the refit the panel used to draw
had another R². The test reads the drawn arrays back from the figure and computes R² by hand.
"""

import re

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.plotting.secular_plots import draw_long_period_combined  # noqa: E402
from resonances.plotting.style import INK, MODEL  # noqa: E402
from resonances.resonance.analysis.filtering import lowpass, time_derivative  # noqa: E402
from resonances.resonance.analysis.secular import AngleTerm, LongPeriodSeries, analyse_secular  # noqa: E402
from resonances.resonance.analysis.settings import AnalysisSettings  # noqa: E402
from resonances.resonance.analysis.verdict import ResonanceType, Status  # noqa: E402

SETTINGS = AnalysisSettings()
T = np.linspace(0.0, 1e7, 20001)  # 10 Myr at 500 yr


def _r2(target, model):
    return 1 - np.sum((target - model) ** 2) / np.sum((target - target.mean()) ** 2)


@pytest.fixture(scope='module')
def analysis():
    sigma = np.radians(80.0 * np.sin(2 * np.pi * T / 3.3e6) + 12.0 * np.sin(2 * np.pi * T / 4.3e5))
    e = 0.1 + 1e-9 * np.cumsum(np.sin(sigma)) * (T[1] - T[0]) + 3e-3 * np.sin(2 * np.pi * T / 4.3e5)
    n = len(T)
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -1))
    series = LongPeriodSeries(ResonanceType.SECULAR, 'g-g6', 'z1-like', T, sigma, np.full(n, 2.5), e, np.full(n, 0.1), terms=terms)
    return analyse_secular(series, SETTINGS)


def test_the_drawn_curve_is_the_fit_of_the_printed_r2(analysis):
    assert analysis.verdict.status == Status.RESONANT and analysis.scale == 'centre'
    assert analysis.angle.cutoff_yr > 5 * analysis.cutoff.cutoff_yr  # the centre's cutoff, not P_c
    fig = draw_long_period_combined(analysis, settings=SETTINGS)
    try:
        panel = next(ax for ax in fig.axes if any(text.get_text().startswith('R²') for text in ax.texts))
        printed = next(text.get_text() for text in panel.texts if text.get_text().startswith('R²'))
        model = next(line for line in panel.get_lines() if line.get_color() == MODEL).get_ydata().astype(float)
        solid = [line for line in panel.get_lines() if line.get_color() == INK and line.get_linestyle() == '-']
        rate = np.nanmax(np.vstack([np.where(np.isfinite(y), y, -np.inf) for y in (ln.get_ydata().astype(float) for ln in solid)]), axis=0)
    finally:
        plt.close(fig)
    both = np.isfinite(model) & np.isfinite(rate)
    assert both.sum() > 100
    drawn = _r2(rate[both], model[both])
    assert drawn == pytest.approx(analysis.response.r2_e, abs=1e-9)
    assert float(re.search(r'R² ([0-9.]+)', printed).group(1)) == pytest.approx(analysis.response.r2_e, abs=5e-3)
    assert 'centre scale' in printed

    # the refit the panel drew before (e filtered at the fine P_c) is another fit: its R² differs
    fine = time_derivative(T, lowpass(T, analysis.series.e, analysis.cutoff.cutoff_yr, SETTINGS.filter_order))
    sigma = analysis.angle.angle_f
    design = np.column_stack([np.sin(sigma), np.cos(sigma), np.ones_like(sigma)])
    coef, *_ = np.linalg.lstsq(design[both], fine[both], rcond=None)
    assert abs(_r2(fine[both], design[both] @ coef) - analysis.response.r2_e) > 0.05
