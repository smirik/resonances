"""The orbit bound of the MMR cutoff (W28 B6a, cards g07-5/g07-6): P_c >= MMR_FILTER_ORBIT_FACTOR x the body's P_orb.

Heliocentric elements osculate about GM_sun alone, so a distant body's a and lambda carry a wave at its own orbital
period; beyond a planet's 1:2 (P_orb > P_syn) the synodic bound let it through as libration cycles (523683 2U-9:
379 yr). The expected numbers are worked out by hand from Kepler's third law P = a^1.5 [yr] and the planets' periods
(Uranus 84.019 yr, Neptune 164.767 yr, Jupiter 11.862 yr: the package's table rates).
"""

import math

import pytest

from resonances.resonance.analysis import rules
from resonances.resonance.analysis.mmr import preprocess as pre
from resonances.resonance.analysis.mmr.models import Context, Preprocessed, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from tests.resonances.analysis.builders import make

DEFAULT = settings_from_config()
P_URANUS, P_NEPTUNE, P_JUPITER = 84.019, 164.767, 11.862  # yr
MU_NEPTUNE, MU_JUPITER = 5.1514e-5, 9.5479e-4


def _context(a_au: float, spec: ResonanceSpec, periods: dict, masses: dict | None = None) -> Context:
    """A Context with what the cutoff reads: n(<a>), the planets' n and masses, the synodic periods of the angle."""
    n = 2 * math.pi / a_au**1.5
    planet_n = {name: 2 * math.pi / period for name, period in periods.items()}
    synodic = {name: 2 * math.pi / abs(n - planet_n[name]) if n != planet_n[name] else math.inf for name in spec.planets}
    return make(
        Context,
        mean_motion_rad_yr=n,
        orbital_period_yr=2 * math.pi / n,
        planet_mean_motion_rad_yr=planet_n,
        planet_mass=dict(masses or {}),
        synodic_periods_yr=synodic,
    )


def test_a_distant_body_beyond_the_uranus_1_2_is_bound_by_its_own_orbit():
    # a = 52.5 au: P_orb = 52.5^1.5 = 380.40 yr; P_syn,U = 1 / (1/84.019 - 1/380.40) = 107.83 yr -> synodic bound
    # 161.7 yr, floor 150 / 2 = 75 yr; the orbit bound 1.5 x 380.40 = 570.6 yr binds
    spec = ResonanceSpec.from_key('9U-2+0-7')
    cutoff, floor, _ = pre.angle_cutoff(spec, _context(52.5, spec, {'Uranus': P_URANUS}), 1.0, DEFAULT)
    assert cutoff.binding == 'orbit'
    assert cutoff.orbit_bound_yr == pytest.approx(570.6, abs=0.1)
    assert cutoff.cutoff_yr >= 570.0 and cutoff.cutoff_yr == cutoff.orbit_bound_yr
    assert cutoff.synodic_bound_yr == pytest.approx(161.7, abs=0.1)
    assert cutoff.floor_bound_yr == floor / 2 == 75.0
    assert pre.cutoff_rules(cutoff) == {rules.Rule.ORBIT_CUTOFF}
    # the column of the CSV
    prep = make(Preprocessed, cutoff=cutoff, window=(0.0, 1.0))
    assert pre.preprocessed_columns(prep)['flt_cutoff_orbit_bound_yr'] == pytest.approx(570.6, abs=0.1)
    # MMR_FILTER_ORBIT_FACTOR = 0 restores the synodic bound
    off = pre.angle_cutoff(spec, _context(52.5, spec, {'Uranus': P_URANUS}), 1.0, settings_from_config(mmr_filter_orbit_factor=0.0))[0]
    assert off.binding == 'synodic' and off.cutoff_yr == pytest.approx(161.7, abs=0.1) and pre.cutoff_rules(off) == frozenset()


def test_the_orbit_bound_binds_only_beyond_the_planets_1_2():
    # P_syn = P P_N / (P - P_N) > P while P < 2 P_N: a Plutino (P = 1.5 P_N = 247.15 yr) has P_syn = 3 P_N = 494.3 yr,
    # so 1.5 P_syn = 741.5 yr against 1.5 P_orb = 370.7 yr; at the 1:2 itself P_syn = P_orb (tie: synodic)
    settings = settings_from_config(mmr_libration_period_min_yr=0.0)
    for ratio, binding, value in (
        (1.5, 'synodic', 1.5 * 3 * P_NEPTUNE),
        (2.0, 'synodic', 1.5 * 2 * P_NEPTUNE),
        (3.0, 'orbit', 1.5 * 3 * P_NEPTUNE),
    ):
        period = ratio * P_NEPTUNE
        synodic = {'Neptune': period * P_NEPTUNE / (period - P_NEPTUNE)}
        cutoff = pre.choose_cutoff(synodic, 1.0, settings, None, period)
        assert cutoff.binding == binding and cutoff.cutoff_yr == pytest.approx(value, rel=1e-9)


def test_a_main_belt_body_is_unchanged():
    # a = 2.5 au: P_orb = 3.953 yr, P_syn,J = 1 / (1/3.953 - 1/11.862) = 5.929 yr; bounds 8.89 (synodic), 5.93 (orbit),
    # 75 (floor): the floor binds, as without the orbit bound
    spec = ResonanceSpec.from_key('3J-1+0-2')
    context = _context(2.5, spec, {'Jupiter': P_JUPITER})
    cutoff = pre.angle_cutoff(spec, context, 0.1, DEFAULT)[0]
    assert (cutoff.binding, cutoff.cutoff_yr) == ('floor', 75.0)
    assert cutoff.orbit_bound_yr == pytest.approx(1.5 * 3.953, abs=0.01)
    assert cutoff.synodic_bound_yr == pytest.approx(1.5 * 5.929, abs=0.01)
    off = pre.angle_cutoff(spec, context, 0.1, settings_from_config(mmr_filter_orbit_factor=0.0))[0]
    assert (off.binding, off.cutoff_yr, off.synodic_bound_yr) == (cutoff.binding, cutoff.cutoff_yr, cutoff.synodic_bound_yr)
    # and without the floor the synodic bound still wins over the orbit (an inner body: P_syn > P_orb)
    bare = pre.angle_cutoff(spec, context, 0.1, settings_from_config(mmr_libration_period_min_yr=0.0))[0]
    assert bare.binding == 'synodic'


@pytest.mark.parametrize(
    'key, a_au, period, mu, floor_cutoff',
    [
        # Neptune Trojan: floor 0.5 x 164.767 / sqrt(6.75 x 5.1514e-5) = 4418 yr -> P_c 2209 yr >> 1.5 x 164.8 = 247 yr
        ('1N-1+0+0', 164.767 ** (2 / 3), P_NEPTUNE, MU_NEPTUNE, 0.25 * P_NEPTUNE / math.sqrt(6.75 * MU_NEPTUNE)),
        # Jupiter Trojan: floor 73.88 yr -> P_c 36.94 yr > 1.5 x 11.862 = 17.79 yr
        ('1J-1+0+0', 11.862 ** (2 / 3), P_JUPITER, MU_JUPITER, 36.94),
    ],
)
def test_a_coorbital_keeps_its_floor(key, a_au, period, mu, floor_cutoff):
    spec = ResonanceSpec.from_key(key)
    planet = spec.planets[0]
    cutoff = pre.angle_cutoff(spec, _context(a_au, spec, {planet: period}, {planet: mu}), 1.0, DEFAULT)[0]
    assert cutoff.binding == 'floor' and cutoff.cutoff_yr == pytest.approx(floor_cutoff, rel=1e-3)
    assert cutoff.orbit_bound_yr == pytest.approx(1.5 * period, rel=1e-3) and cutoff.orbit_bound_yr < cutoff.cutoff_yr
    assert pre.cutoff_rules(cutoff) == frozenset()


def test_the_explicit_cutoff_still_wins_and_a_missing_orbit_is_no_bound():
    user = pre.choose_cutoff({'Neptune': 200.0}, 1.0, settings_from_config(mmr_filter_cutoff_yr=100.0), None, 400.0)
    assert user.binding == 'user' and user.cutoff_yr == 100.0 and user.orbit_bound_yr == 600.0
    for missing in (None, float('nan'), 0.0):
        cutoff = pre.choose_cutoff({'Neptune': 200.0}, 1.0, DEFAULT, None, missing)
        assert cutoff.orbit_bound_yr == 0.0 and cutoff.binding == 'synodic' and cutoff.cutoff_yr == 300.0
