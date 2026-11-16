"""W28 E6 (card g10-6, the author's option B): the radial overlap of the body's band q_min..Q_max with the band
q_j..Q_j of each of the eight planets, columns ctx_orbit_overlap and ctx_orbit_overlap_source, for MMR, secular and
ZLK rows. ctx_crosses (the angle's planets only) is not changed.

Expected planets are computed by hand from the J2000 bands a_j (1 -/+ e_j) of const.PLANETS_AXIS and
const.PLANETS_ECCENTRICITY:

    Mercury 0.30758..0.46669   Venus 0.71840..0.72824   Earth 0.98330..1.01670   Mars 1.38140..1.66603
    Jupiter 4.95068..5.45428   Saturn 9.02720..10.05579  Uranus 18.28040..20.09556 Neptune 29.81093..30.32812

and the body bands of the card's examples (min q, max Q of the saved records, benchmark/data, read on 2026-10-31):
2020 CB 0.7093..1.3053, 2000 SF8 1.0057..2.2907, 41588 1.5093..3.1719, 285637 1.7401..2.1815, 137295 33.1928..63.5135.
"""

import math

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import context as ctx
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec, orbit_overlap_columns
from resonances.resonance.analysis.secular import LongPeriodSeries
from resonances.resonance.analysis.secular.analyse import analyse_long_period
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import ResonanceType

SETTINGS = settings_from_config()


@pytest.mark.parametrize(
    'q_min, big_q_max, expected',
    [
        (0.7093, 1.3053, ('Venus', 'Earth')),  # 2020 CB: q < Q_Venus 0.72824, Q < q_Mars 1.38140
        (1.0057, 2.2907, ('Earth', 'Mars')),  # 2000 SF8: q < Q_Earth 1.01670 (option A, the angle's planet, gave Mars only)
        (1.5093, 3.1719, ('Mars',)),  # 41588: q < Q_Mars 1.66603, Q < q_Jupiter 4.95068
        (1.7401, 2.1815, ()),  # 285637: q > Q_Mars
        (33.1928, 63.5135, ()),  # 137295: q > Q_Neptune 30.32812
    ],
)
def test_overlap_with_the_j2000_bands(q_min, big_q_max, expected):
    assert ctx.orbit_overlap(q_min, big_q_max) == (expected, 'J2000')


def test_no_data_is_not_none():
    assert ctx.Context.__dataclass_fields__['orbit_overlap'].default is None  # a Context built by hand: no data
    assert ctx.orbit_overlap(float('nan'), 2.0) == (None, '')
    # the CSV form: blank for no data, 'none' for no overlap
    assert orbit_overlap_columns(None, '') == {'ctx_orbit_overlap': None, 'ctx_orbit_overlap_source': None}
    assert orbit_overlap_columns((), 'J2000') == {'ctx_orbit_overlap': 'none', 'ctx_orbit_overlap_source': 'J2000'}
    assert orbit_overlap_columns(('Venus', 'Earth'), 'saved') == {
        'ctx_orbit_overlap': 'Venus;Earth',
        'ctx_orbit_overlap_source': 'saved',
    }


def _planet(name, t, a, e):
    ones = np.ones_like(t)
    return PlanetSeries(name, t, a * ones, np.asarray(e) * ones, 0.0 * ones, 0.0 * ones, 0.0 * ones, None)


def _hungaria(planets):
    """A 285637-like body (a 1.94, e 0.1035: q 1.7392, Q 2.1408) in 3M+7J-6 over 2 kyr."""
    t = np.arange(0.0, 2000.0, 1.0)
    ones = np.ones_like(t)
    return MMRSeries(
        t_yr=t,
        a=1.94 * ones,
        e=0.1035 * ones,
        inc=0.4 * ones,
        Omega=0.5 + 1e-4 * t,
        omega=2.0 - 2e-4 * t,
        varpi=np.mod(2.5 - 1e-4 * t, 2 * np.pi),
        angle_wrapped=np.mod(1.0 + 2 * np.pi * t / 300.0, 2 * np.pi),
        resonance=ResonanceSpec.from_key('3M+7J-6+0+0-4'),
        regime=Regime(1.0, 2000.0, len(t)),
        planets=planets,
    )


def test_saved_planet_bands_replace_j2000():
    t = np.arange(0.0, 2000.0, 1.0)
    # Mars saved with e up to 0.125: Q_Mars = 1.52371243 x 1.125 = 1.71418 < q 1.7392 -> still none
    near = {'Mars': _planet('Mars', t, 1.52371243, np.linspace(0.09, 0.125, len(t))), 'Jupiter': _planet('Jupiter', t, 5.2025, 0.0484)}
    orbit = ctx._orbit_columns(_hungaria(near))
    assert (orbit['orbit_overlap'], orbit['orbit_overlap_source']) == ((), 'saved:Mars+Jupiter')
    # e up to 0.145: Q_Mars = 1.52371243 x 1.145 = 1.74465 > q 1.7392 -> Mars, which J2000 (Q_Mars 1.66603) misses
    far = {'Mars': _planet('Mars', t, 1.52371243, np.linspace(0.09, 0.145, len(t))), 'Jupiter': _planet('Jupiter', t, 5.2025, 0.0484)}
    orbit = ctx._orbit_columns(_hungaria(far))
    assert (orbit['orbit_overlap'], orbit['orbit_overlap_source']) == (('Mars',), 'saved:Mars+Jupiter')
    # ctx_crosses reads the same saved bands of the angle's planets (Mars, Jupiter): unchanged by E6
    assert orbit['crosses'] == ('Mars',)
    unsaved = ctx._orbit_columns(_hungaria({}))
    assert (unsaved['orbit_overlap'], unsaved['orbit_overlap_source'], unsaved['crosses']) == ((), 'J2000', ())


def test_all_eight_saved_is_marked_saved():
    t = np.arange(0.0, 10.0, 1.0)
    planets = {
        name: _planet(name, t, a, 0.0) for name, a in zip(ctx.PLANET_NAMES_ORDER, (0.387, 0.723, 1.0, 1.524, 5.2, 9.54, 19.19, 30.07))
    }
    # circular planets: Mars band 1.524..1.524 inside 1.5..1.6, Earth 1.0 outside
    assert ctx.orbit_overlap(1.5, 1.6, planets) == (('Mars',), 'saved')


def test_crosses_keeps_the_angles_planets_only():
    # 2000 SF8 in 8M-9 with the J2000 bands: crosses Mars only, the radial overlap adds Earth
    spec_planets = {'Mars': ctx._planet_orbit_band(None, 'Mars')}
    assert ctx.crossed_planets(1.0057, 2.2907, spec_planets) == ('Mars',)
    assert ctx.orbit_overlap(1.0057, 2.2907)[0] == ('Earth', 'Mars')


def test_mmr_context_columns():
    context = ctx.build_context(_hungaria({}), SETTINGS)
    columns = context.to_columns()
    assert columns['ctx_orbit_overlap'] == 'none'
    assert columns['ctx_orbit_overlap_source'] == 'J2000'
    assert columns['ctx_crosses'] == ''
    assert math.isclose(columns['ctx_q_min_au'], 1.94 * (1 - 0.1035))


def test_long_period_columns_use_the_j2000_bands():
    # a 1.2, e 0.45: q 0.66, Q 1.74 -> Venus, Earth, Mars (Q_Mercury 0.46669 < q; Q < q_Jupiter)
    t = np.arange(0.0, 300_001.0, 100.0)
    ones = np.ones_like(t)
    omega = np.radians(90.0 + 20.0 * np.sin(2 * np.pi * t / 20_000.0))
    series = LongPeriodSeries(ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, 1.2 * ones, 0.45 * ones, 0.3 * ones)
    row = analyse_long_period(series, SETTINGS).numbers_row()
    assert row['ctx_orbit_overlap'] == 'Venus;Earth;Mars'
    assert row['ctx_orbit_overlap_source'] == 'J2000'
