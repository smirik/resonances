"""W3 guards, G3: no universal 150-yr 'physical minimum' of the libration period.

Codex astra (PHYSICS-MATH §7): the small Trojan libration of Jupiter has P_orb / sqrt(27 mu / 4) = 147.77 yr, below
the old MMR_LIBRATION_PERIOD_MIN_YR = 150, and a clean synthetic Trojan came out uncertain/insufficient_time; in the
benchmark the Jupiter Trojans 588, 659, 884 (148-149 yr) and the Earth horseshoes 2013 BS45, 2014 UR, 2018 PN22
(145-148 yr) lost their episodes the same way. A co-orbital angle now gets MMR_COORBITAL_PERIOD_FLOOR_FACTOR x the
tadpole period of its planet; any other angle keeps the convention MMR_LIBRATION_PERIOD_MIN_YR.

Hand values (literature, not the package tables): P_J = 4332.59 d / 365.2569 d = 11.8617 Gaussian yr, m_Sun / m_J =
1047.3486 -> sqrt(27 mu / 4) = 0.080280, P = 147.75 yr; Earth 365.2564 d, 332946.05 -> 222.09 yr; Mars 686.98 d,
3098703.6 -> 1274.3 yr.
"""

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import decide, preprocess
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, Status
from tests.resonances.analysis.builders import make_episode, make_geometry

S = settings_from_config()
GAUSS_YEAR_DAYS = 365.2569
PLANETS = {  # sidereal period [d], m_Sun / m_planet
    'Jupiter': (4332.59, 1047.3486),
    'Earth': (365.2564, 332946.05),
    'Mars': (686.98, 3098703.6),
}
TADPOLE_YR = {'Jupiter': 147.75, 'Earth': 222.09, 'Mars': 1274.3}


def context_of(*names):
    return SimpleNamespace(
        planet_mean_motion_rad_yr={n: 2 * math.pi * GAUSS_YEAR_DAYS / PLANETS[n][0] for n in names},
        planet_mass={n: 1 / PLANETS[n][1] for n in names},
    )


@pytest.mark.parametrize('planet, key', [('Jupiter', '1J-1'), ('Earth', '1E-1'), ('Mars', '1M-1')])
def test_a_coorbital_floor_is_its_planets_tadpole_period(planet, key):
    floor, setting = preprocess.libration_period_floor(ResonanceSpec.from_key(key), context_of(planet), S)
    assert setting == 'mmr_coorbital_period_floor_factor'
    assert floor == pytest.approx(S.mmr_coorbital_period_floor_factor * TADPOLE_YR[planet], rel=2e-3)


@pytest.mark.parametrize('key', ['2J-1', '3J-2', '3J+8S-3', '1J-3r0'])
def test_other_angles_keep_the_convention(key):
    floor, setting = preprocess.libration_period_floor(ResonanceSpec.from_key(key), context_of('Jupiter'), S)
    assert (floor, setting) == (S.mmr_libration_period_min_yr, 'mmr_libration_period_min_yr')


def test_the_coorbital_cutoff_follows_the_floor():
    floor, _ = preprocess.libration_period_floor(ResonanceSpec.from_key('1J-1'), context_of('Jupiter'), S)
    cutoff = preprocess.choose_cutoff({}, 0.1, S, period_floor_yr=floor)
    assert cutoff.binding == 'floor' and cutoff.cutoff_yr == pytest.approx(0.25 * TADPOLE_YR['Jupiter'], rel=2e-3)


def test_the_episode_floor_is_the_angles_own():
    episode = make_episode(tp_times_yr=np.array([0.0, 1000.0]), period_med_yr=147.8)
    trojan = make_geometry(runs=[episode], period_floor_yr=73.9, period_floor_setting='mmr_coorbital_period_floor_factor')
    assert decide.librating_episodes(trojan, S) == [episode]
    default = make_geometry(runs=[episode], period_floor_yr=float('nan'), period_floor_setting='')
    assert decide.librating_episodes(default, S) == []  # 147.8 < 150: the convention of non-co-orbitals


def trojan_series() -> MMRSeries:
    """A linear tadpole about L4 (sigma = lambda - lambda_J = pi/3 + 0.05 cos(w t), w = 2 pi / 147.77 yr) with
    a answering it (sigma_dot = c (a - a*)), 2 kyr at 0.1 yr (Codex astra's probe)."""
    n_jupiter = 2 * math.pi * GAUSS_YEAR_DAYS / PLANETS['Jupiter'][0]
    period = 147.77
    t = np.arange(20001) * 0.1
    w = 2 * math.pi / period
    a0 = (2 * math.pi / n_jupiter) ** (2 / 3)
    c = 3 * math.pi * a0**-2.5
    sigma = math.pi / 3 + 0.05 * np.cos(w * t)
    zero = np.zeros(t.size)
    return MMRSeries(
        t,
        a0 - 0.05 * w * np.sin(w * t) / c,
        np.full(t.size, 0.01),
        zero,
        zero,
        zero,
        zero,
        sigma,
        ResonanceSpec.from_key('1J-1'),
        Regime(0.1, t[-1], t.size),
    )


def test_a_small_trojan_libration_of_148_yr_is_a_libration():
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(trojan_series(), S, scan_owner=False, scan_subresonances=False)
    assert analysis.prep.period_floor_basis == 'mmr_coorbital_period_floor_factor'
    assert analysis.geometry.period_cycle_yr == pytest.approx(147.77, abs=0.5)
    assert analysis.extra_columns['geo_n_librating_episodes'] >= 1
    assert (analysis.verdict.status, analysis.verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    numbers = analysis.numbers_row()
    assert numbers['flt_period_floor_yr'] == pytest.approx(S.mmr_coorbital_period_floor_factor * TADPOLE_YR['Jupiter'], rel=5e-3)


def test_the_floor_reaches_the_decision():
    """A floor above the libration period (factor 1.1: 162.5 yr > 147.8 yr) makes the libration 'faster than any
    libration of the angle': the TP prominence covers it (`analyse.fast_residual`) and no episode remains (the cutoff
    is held at 37 yr: the co-orbital rule would raise it to 81 yr with the floor and damp the libration anyway)."""
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    strict = dataclasses.replace(S, mmr_coorbital_period_floor_factor=1.1, mmr_filter_cutoff_yr=37.0)
    analysis = analyse_mmr(trojan_series(), strict, scan_owner=False, scan_subresonances=False)
    numbers = analysis.numbers_row()
    assert numbers['flt_period_floor_yr'] == pytest.approx(1.1 * TADPOLE_YR['Jupiter'], rel=5e-3)
    assert numbers['flt_fast_residual_deg'] > 1.0  # most of the 2.9-deg libration sits above the floor's frequency
    assert analysis.extra_columns['geo_n_librating_episodes'] == 0 and analysis.verdict.status != Status.RESONANT
