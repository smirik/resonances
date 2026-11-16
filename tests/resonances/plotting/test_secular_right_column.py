"""The secular/ZLK right column: the FREQUENCIES row a period is named by (`verdict_text._identify`) and the figure
drawing the column the package export builds.

Expectations by hand: g5 turns in 1 296 000 / 4.2575 = 304.4 kyr. The synthetic ZLK body (591986's analogue) has a
free circle at 222″/yr about a forced vector at g5. The slotted bullets and sentences of the old column (W28 E9) were
removed with the engine that drew them (2026-11-14).
"""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.data.const import PLANETARY_FREQUENCIES  # noqa: E402
from resonances.plotting import verdict_text as vt  # noqa: E402
from resonances.plotting.secular_plots import Frequency, draw_long_period_combined  # noqa: E402
from resonances.resonance.analysis.secular import LongPeriodSeries  # noqa: E402
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from resonances.resonance.analysis.verdict import ResonanceType  # noqa: E402
from resonances.resonance.analysis.zlk import analyse_zlk  # noqa: E402
from resonances.secular import free_elements as fe  # noqa: E402

SETTINGS = settings_from_config()
RAD = 2 * np.pi / 1296000.0  # rad/yr per arcsec/yr
G5, G6, S6 = PLANETARY_FREQUENCIES['g5'], PLANETARY_FREQUENCIES['g6'], PLANETARY_FREQUENCIES['s6']


@pytest.fixture(scope='module')
def zlk():
    """591986's analogue over 500 kyr: forced 0.045 at g5 and 0.017 at g6, the free circle 0.011 at 222″/yr, the node
    at g5, i 39° (a kinematic libration of ω)."""
    t = np.arange(0.0, 500_001.0, 100.0)
    z_e = 0.045 * np.exp(1j * (G5 * RAD * t + np.radians(83.0))) + 0.017 * np.exp(1j * (G6 * RAD * t + np.radians(3.2)))
    z_e = z_e + 0.011 * np.exp(1j * 222.0 * RAD * t)
    Omega = G5 * RAD * t + np.radians(146.0) + np.radians(8.0) * np.sin(2 * np.pi * t / 1.5e6)
    e, varpi = np.abs(z_e), np.angle(z_e)
    inc = np.full_like(t, np.radians(39.0))
    omega = np.unwrap(varpi - Omega)
    series = LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, 5.2 + 0.05 * np.sin(2 * np.pi * t / 1e4), e, inc)
    elements = fe.free_elements(t, e, inc, Omega, omega)
    analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
    row = {**analysis.summary_row(), **fe.summary_fields(elements)}
    return analysis, row, elements


# --------------------------------------------------------------------------------------------- time scales


def test_identify_names_a_row_within_one_over_t():
    """g5 turns in 1296000/4.2575 = 304.4 kyr: a period of 300 kyr (290–310) over 1 Myr (1/T 1.296″/yr) is g5; 2306's
    221 kyr (218–257 kyr: 5.04–5.94″/yr, 1/T 0.13″/yr over 10 Myr) is none of g5, g6, s6, and g5 is the nearest."""
    rows = [Frequency('g5 Jup', G5, 1296000 / G5), Frequency('g6 Sat', G6, 1296000 / G6), Frequency('s6 Sat', S6, 1296000 / abs(S6))]
    assert vt._identify(3.0e5, 2.9e5, 3.1e5, 1.296, rows) == ('g5 Jup', 'g5 Jup')
    # 46.5 kyr (27.87″/yr) over 100 kyr (1/T 12.96″/yr): g6 (28.25) and s6 (26.34) are both within 1/T, nearest first
    assert vt._identify(4.65e4, float('nan'), float('nan'), 12.96, rows) == ('g6 Sat or s6 Sat', 'g6 Sat')
    assert vt._identify(2.21e5, 2.185e5, 2.57e5, 0.1296, rows) == ('', 'g5 Jup')
    assert vt._identify(1296000 / 26.0, float('nan'), float('nan'), 0.1296, rows) == ('', 's6 Sat')  # 26.000 vs 26.345


# --------------------------------------------------------------------------------------------- the decisive argument


# --------------------------------------------------------------------------------------------- the figure


def test_the_figure_draws_the_column(zlk):
    """The figure draws the ten-block column (`verdict_column`, 2026-11-04) that the package export builds
    (`verdict_column_long.figure_column`): the same column, its status word and Why drawn, no '↑' pointer (D1)."""
    from resonances.plotting import verdict_column as vc
    from resonances.plotting.verdict_column_long import figure_column

    analysis, row, elements = zlk
    extra = {key: value for key, value in row.items() if key.startswith('free_')}
    fig = draw_long_period_combined(analysis, row=extra, free_elements=elements, settings=SETTINGS)
    try:
        column = fig.text_stats['column']
        assert column == figure_column(analysis, extra, elements, SETTINGS) and vc.problems(column) == []
        assert fig.text_stats['overflow'] == 0
        drawn = ' '.join(t.get_text() for t in fig.texts if t.get_gid() == 'verdict_column')
        assert column.header.word in drawn and vc.plain(column.why.sentences[0]).split()[0] in drawn
        assert '↑' not in drawn
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------------------- benchmark records
