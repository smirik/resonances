"""The FREQUENCIES block and the a of the ORBIT block of the secular/ZLK figure (docs/reading-figures.md, "The
secular and Lidov-Kozai figure").

Every expectation is computed by hand from the synthetic vectors and the planetary constants of Laskar's table as the
package lists them (g5 4.2575, g6 28.2455, s6 −26.3450 ″/yr): a period is 1 296 000″ / |f|, so g5 turns in
1296000 / 4.25749 = 304.4 kyr, s6 in 49.2 kyr and g6 − g5 = 23.988″/yr beats in 54.0 kyr.
"""

import re

import numpy as np
import pytest

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.plotting.secular_plots import long_period_frequency_lines
from resonances.resonance.analysis.secular import AngleTerm, LongPeriodSeries, analyse_secular
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import ResonanceType
from resonances.resonance.analysis.zlk import analyse_zlk
from resonances.secular import free_elements as fe

SETTINGS = settings_from_config()
RAD = 2 * np.pi / 1296000.0  # rad/yr per arcsec/yr
G5, G6, S6 = PLANETARY_FREQUENCIES['g5'], PLANETARY_FREQUENCIES['g6'], PLANETARY_FREQUENCIES['s6']
ROW = re.compile(r'^(?P<name>.{8})(?P<rate>.{9}) (?P<period>.{9})(?:  (?P<note>.*))?$')


def _rows(lines):
    """{name: (rate text, period text, note)} of the block's rows (the first is the column header)."""
    rows = {}
    for line in lines[1:]:
        match = ROW.match(line.value)
        assert match, line.value
        rows[match['name'].strip()] = (match['rate'].strip(), match['period'].strip(), (match['note'] or '').strip())
    return rows


def _rate(text: str) -> float:
    return float(text.replace('−', '-'))


def _period_yr(text: str) -> float:
    value, unit = text.split()
    return float(value) * {'yr': 1.0, 'kyr': 1e3, 'Myr': 1e6}[unit]


@pytest.fixture(scope='module')
def zlk():
    """591986's analogue over 500 kyr (tests/resonances/analysis/test_zlk_node_lock.py, trojan_500kyr): forced 0.045
    at g5 and 0.017 at g6, the free circle 0.011 at 222″/yr, the node at g5."""
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


def test_zlk_rows_give_the_body_rates_the_modes_and_their_beats(zlk):
    analysis, row, elements = zlk
    lines = long_period_frequency_lines(analysis, row, elements)
    assert lines[0].value.split() == ['arcsec/yr', 'period', '|c|']
    rows = _rows(lines)
    # the body: the free circle turns at 222"/yr (5.84 kyr), the node at g5 (304 kyr); omega = g - s
    assert _rate(rows['g'][0]) == pytest.approx(222.0, abs=0.05) and rows['g'][1:] == ('5.84 kyr', 'free ϖ')
    # the node: g5 plus the slope of its 8-deg wave over a third of its 1.5-Myr cycle (a least-squares line through
    # the node alone gives 4.31"/yr); the printed rate is the free_b_Omega column, its period 1296000 / rate
    assert _rate(rows['s'][0]) == pytest.approx(row['free_b_Omega'], abs=5e-4) and rows['s'][2] == 'free Ω'
    assert _rate(rows['s'][0]) == pytest.approx(4.2575, abs=0.15)
    assert _period_yr(rows['s'][1]) == pytest.approx(1296000 / row['free_b_Omega'], rel=5e-3)
    assert _rate(rows['g − s'][0]) == pytest.approx(_rate(rows['g'][0]) - _rate(rows['s'][0]), abs=0.1)
    assert _period_yr(rows['g − s'][1]) == pytest.approx(1296000 / 217.6, rel=2e-3)
    # the modes: Laskar's constants with their periods and the fitted |c|; g5 holds varpi (attributed)
    assert row['zlk_attribution'] == 'attributed'
    assert rows['g5 Jup'] == ('+4.2575', '304 kyr', '0.045 ◀')
    assert rows['g6 Sat'][:2] == ('+28.2455', '45.9 kyr') and rows['g6 Sat'][2] == '0.017'
    assert rows['s6 Sat'][:2] == ('−26.3450', '49.2 kyr')
    # the beats: g - g5 = 217.74"/yr -> 5.95 kyr (the kinematic omega cycle), g - g6 = 193.75"/yr -> 6.69 kyr, and the
    # forced vector's own beat g6 - g5 = 23.988"/yr -> 54.0 kyr
    assert _period_yr(rows['g − g5'][1]) == pytest.approx(1296000 / (222.0 - 4.25749), rel=5e-3)
    assert rows['g − g5'][2] == 'kinematic'
    assert _period_yr(rows['g − g6'][1]) == pytest.approx(1296000 / (222.0 - 28.24553), rel=5e-3)
    assert rows['g6 − g5'] == ('+23.9880', '54 kyr', 'beat')
    assert 'Ω − ϖ5' not in rows  # no lock flagged over 500 kyr
    # the kinematic period printed is the attribution's own beat column
    assert _period_yr(rows['g − g5'][1]) == pytest.approx(row['zlk_beat_period_yr'], rel=5e-3)


def test_a_flagged_node_lock_adds_its_period(zlk):
    analysis, row, elements = zlk
    locked = {
        **row,
        'zlk_node_lock': 'g5',
        'zlk_node_lock_period_yr': 1.404e6,
        'zlk_node_lock_min_deg': 50.3,
        'zlk_node_lock_max_deg': 130.4,
    }
    assert _rows(long_period_frequency_lines(analysis, locked, elements))['Ω − ϖ5'] == ('lock', '1.4 Myr', '50–130°')


def test_zlk_without_free_elements_lists_jupiter_and_saturn(zlk):
    analysis, _, _ = zlk
    rows = _rows(long_period_frequency_lines(analysis, {}, None))
    assert list(rows) == ['g5 Jup', 'g6 Sat', 's6 Sat']
    assert rows['g5 Jup'] == ('+4.2575', '304 kyr', 'not fitted')


def test_secular_rows_give_the_angle_rate_and_the_forced_waves():
    """2306-like g − g6 + s − s6: free vectors at g = 57.8 and s = −56.1″/yr. sigma's rate from them is
    57.8 − 28.2455 − 56.1 + 26.3450 = −0.2005″/yr (6.46 Myr); the forced waves g − g6 = 29.554″/yr (43.9 kyr) and
    s − s6 = −29.755″/yr (43.6 kyr)."""
    t = np.linspace(0.0, 2e6, 4001)
    g, s = 57.8, -56.1
    varpi, Omega = g * RAD * t + 0.3, s * RAD * t + 1.1
    sigma = varpi - G6 * RAD * t + Omega - S6 * RAD * t
    n = len(t)
    a = 2.73 + 0.002 * np.sin(2 * np.pi * t / 3e4)
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -1), AngleTerm('s', None, 1), AngleTerm('s', 6, -1))
    series = LongPeriodSeries(
        ResonanceType.SECULAR,
        'g-g6+s-s6',
        'synthetic',
        t,
        sigma,
        a,
        np.full(n, 0.05),
        np.full(n, np.radians(5.0)),
        varpi=varpi,
        Omega=Omega,
        terms=terms,
        planet_frequencies=PLANETARY_FREQUENCIES,
    )
    analysis = analyse_secular(series, SETTINGS)
    row = analysis.summary_row()
    lines = long_period_frequency_lines(analysis, row)
    assert lines[0].value.split() == ['arcsec/yr', 'period']
    rows = _rows(lines)
    assert _rate(rows['g'][0]) == pytest.approx(g, abs=0.01) and rows['g'][1:] == ('22.4 kyr', 'free ϖ')
    assert _rate(rows['s'][0]) == pytest.approx(s, abs=0.01) and rows['s'][1:] == ('23.1 kyr', 'free Ω')
    assert _rate(rows['σ'][0]) == pytest.approx(-0.2005, abs=0.01) and rows['σ'][2] == 'mean rates'
    assert rows['g5 Jup'] == ('+4.2575', '304 kyr', '')
    assert rows['g6 Sat'] == ('+28.2455', '45.9 kyr', 'in σ −1')
    assert rows['s6 Sat'] == ('−26.3450', '49.2 kyr', 'in σ −1')
    assert rows['g − g6'][1:] == ('43.9 kyr', 'forced') and _rate(rows['g − g6'][0]) == pytest.approx(29.554, abs=0.01)
    assert rows['s − s6'][1:] == ('43.6 kyr', 'forced') and _rate(rows['s − s6'][0]) == pytest.approx(-29.755, abs=0.01)
    # ORBIT: a's median and range (2.73 -/+ 0.002 au)
    assert row['ctx_a_med_au'] == pytest.approx(2.73, abs=1e-4)
    assert (row['ctx_a_min_au'], row['ctx_a_max_au']) == pytest.approx((2.728, 2.732), abs=1e-5)


def test_the_table_stays_in_one_number_column(zlk):
    """The rows of the block are one unbreakable item of the explained column: the column header is never separated
    from the rows it names (combined_numbers._items)."""
    from resonances.plotting import combined_numbers as numbers

    analysis, row, elements = zlk
    lines = long_period_frequency_lines(analysis, row, elements)
    items = numbers._items(numbers._number_rows(lines, width=3.0, size=8.0))
    assert len(items) == 1 and sum(1 for kind, _, _ in items[0] if kind in ('line', 'cont')) == len(lines)
