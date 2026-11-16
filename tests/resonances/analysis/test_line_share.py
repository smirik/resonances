"""W28 B8b (card g07-3): which lattice lines carry a_dot on the decisive episode (`forced_lines.line_share`, columns
`rsp_line_*`).

The expected numbers are computed by hand, not by the package: a sine of amplitude A carries A^2 / 2 of the variance,
so on a_dot = A1 sin(2 pi t / T1) + A2 cos(2 pi t / T2) + white noise of variance s^2 the first line's share is
(A1^2 / 2) / (A1^2 / 2 + A2^2 / 2 + s^2). Periods: Neptune's sidereal period 164.79 yr (Mars 1.8808 yr); a two-body
p:q line sits at j / P_conj with the conjunction cycle P_conj = q P_planet (8N-13: 13 x 164.79 = 2142.3 yr; 40M-57:
57 x 1.8808 = 107.2 yr).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import forced_lines as FL
from resonances.resonance.analysis.mmr.models import ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
P_NEPTUNE = 164.79  # yr, sidereal
P_MARS = 1.8808  # yr, sidereal
SPEC_8_13 = ResonanceSpec.from_key('8N-13+0+5')
N_NEPTUNE = 2 * math.pi / P_NEPTUNE
P_CONJ = 13 * P_NEPTUNE  # 2142.3 yr
# 8N-13 at its commensurability: n_c = 8/13 n_N, P_orb = 13/8 P_N = 267.8 yr
LINES_8_13 = FL.lattice_lines(SPEC_8_13, {'Neptune': N_NEPTUNE}, 0.0, S.matrix_2body_order_max, S.matrix_3body_order_max)
T = np.arange(0.0, 84_000.0, 10.0)  # a 59358-like episode of 84 kyr at a 10-yr step
SHORTEST = 300.0  # the band's short end [yr]


def _two_lines(a1: float, a2: float, noise: float, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return a1 * np.sin(2 * np.pi * T / P_CONJ + 0.3) + a2 * np.cos(2 * np.pi * T / (P_CONJ / 2)) + noise * rng.standard_normal(len(T))


def test_the_share_of_two_known_lines_in_noise_is_the_analytic_one():
    a1, a2, noise = 1.0, 0.6, 0.5
    expected_share = (a1**2 / 2) / (a1**2 / 2 + a2**2 / 2 + noise**2)  # 0.538
    expected_all = (a1**2 / 2 + a2**2 / 2) / (a1**2 / 2 + a2**2 / 2 + noise**2)  # 0.731
    share = FL.line_share(T, 1e-6 * _two_lines(a1, a2, noise), LINES_8_13, SPEC_8_13, float('nan'), SHORTEST)
    assert share.reason == ''
    assert share.period_yr == pytest.approx(P_CONJ, rel=0.01)
    # 3 sigma of the share over noise draws (std 0.014 on 40 seeds, mean 0.541: the noise the fit absorbs)
    assert share.share == pytest.approx(expected_share, abs=0.045)
    assert share.share_all == pytest.approx(expected_all, abs=0.045)
    # the conjunction cycle, and both coinciding lattice lines (5 n_c - 3 n_N = -(8 n_c - 5 n_N) = n_N / 13), lowest
    # order first
    assert share.ids == '13 P_N = P_conj; 3N-5; 5N-8'


def test_the_stronger_line_dominates_when_the_amplitudes_swap():
    a1, a2, noise = 0.4, 1.0, 0.2
    expected = (a2**2 / 2) / (a1**2 / 2 + a2**2 / 2 + noise**2)  # 0.725
    share = FL.line_share(T, _two_lines(a1, a2, noise), LINES_8_13, SPEC_8_13, float('nan'), SHORTEST)
    assert share.period_yr == pytest.approx(P_CONJ / 2, rel=0.01)
    assert share.share == pytest.approx(expected, abs=0.03)
    assert share.ids.startswith('P_conj/2; 2N-3')  # 13/2 P_N is no whole multiple of P_N


def test_a_line_off_by_a_fraction_of_the_resolution_is_still_found():
    # the body's mean motion is known only to the episode's resolution: a line 0.3 / T off its lattice value
    shifted = 1 / (1 / P_CONJ + 0.3 / np.ptp(T))
    share = FL.line_share(T, np.sin(2 * np.pi * T / shifted), LINES_8_13, SPEC_8_13, float('nan'), SHORTEST)
    assert share.share > 0.95 and share.ids.startswith('13 P_N')


def test_power_off_the_lattice_is_unidentified_or_the_own_libration():
    off = np.sin(2 * np.pi * T / 3000.0) + 0.3 * np.sin(2 * np.pi * T / P_CONJ)  # 3000 yr lies between 0 and 1 / P_conj
    share = FL.line_share(T, off, LINES_8_13, SPEC_8_13, float('nan'), SHORTEST)
    assert share.ids == FL.UNIDENTIFIED and share.period_yr == pytest.approx(3000.0, rel=0.02)
    assert share.share == pytest.approx(0.5 / (0.5 + 0.045), abs=0.03)
    assert share.share_all == pytest.approx(0.045 / (0.5 + 0.045), abs=0.03)  # the lattice holds the 2142-yr line only
    own = FL.line_share(T, off, LINES_8_13, SPEC_8_13, 3000.0, SHORTEST)  # the libration's own period
    assert own.ids == 'P_lib'


def test_the_orbital_period_is_a_known_line():
    t = np.arange(0.0, 20_000.0, 5.0)
    orbit = 13 / 8 * P_NEPTUNE  # 267.8 yr, the body at 8:13
    share = FL.line_share(t, np.cos(2 * np.pi * t / orbit), (), SPEC_8_13, float('nan'), 100.0, orbital_period_yr=orbit)
    assert share.ids == 'P_orb' and share.share > 0.99
    # in a two-body lattice it is the comb's line j = |k_P| (P_conj / 8): named with it
    lattice = FL.line_share(t, np.cos(2 * np.pi * t / orbit), LINES_8_13, SPEC_8_13, float('nan'), 100.0, orbital_period_yr=orbit)
    assert lattice.ids.startswith('P_conj/8') and lattice.ids.endswith('P_orb')


def test_period_names_of_two_body_lines():
    spec = ResonanceSpec.from_key('40M-57+0+17')
    # 7M-10: j = |(-7)(-57) - 10 x 40| = 1, the conjunction cycle 57 P_M; 21M-30: j = 3, 57/3 = 19 P_M; 14M-20: j = 2,
    # 57/2 is no whole number
    line = FL.ForcedLine(57 * P_MARS, 10, (-7,), 3, ('Mars',))
    assert line.label == '7M-10' and FL.period_name(line, spec) == '57 P_M = P_conj'
    assert FL.period_name(FL.ForcedLine(19 * P_MARS, 30, (-21,), 9, ('Mars',)), spec) == '19 P_M'
    assert FL.period_name(FL.ForcedLine(57 / 2 * P_MARS, 20, (-14,), 6, ('Mars',)), spec) == 'P_conj/2'
    three = ResonanceSpec.from_key('4J-2S-1+0+0-1')
    assert FL.period_name(FL.ForcedLine(44.0, 1, (3, 0), 2, ('Jupiter', 'Saturn')), three) == ''


def test_lattice_periods_at_the_commensurability_are_the_conjunction_comb():
    periods = sorted({round(line.period_yr, 6) for line in LINES_8_13}, reverse=True)
    assert periods[0] == pytest.approx(P_CONJ, rel=1e-9)
    assert all((P_CONJ / p) == pytest.approx(round(P_CONJ / p), rel=1e-6) for p in periods)


def test_too_short_or_empty_episodes_are_blank():
    assert FL.line_share(T[:5], np.ones(5), LINES_8_13, SPEC_8_13, float('nan'), SHORTEST).reason == 'no_samples'
    assert FL.line_share(T[:100], np.sin(T[:100]), LINES_8_13, SPEC_8_13, float('nan'), 5_000.0).reason == 'no_band'
    retro = ResonanceSpec.from_key('1J-3r0')
    assert FL.lattice_lines_at(retro, {'Jupiter': 0.53}, 1.0, 10, 6) == ()


# --- real records (benchmark data) ---------------------------------------------------------------------------------
# 2005 GQ107 40M-57 (author 14.10, kept 31.10, card g07-3 A: uncertain/episodic): the right column explains its a by the
# 107-yr line 7M-10 = 57 P_M (Claude's g07 reading); 59358 8N-13: 2142 yr = 13 P_N with 3N-5 and 5N-8, and the lines
# of the lattice carry most of a_dot (77 % for 2142/1071/714 yr in the g07 reading, Butterworth at 640 yr).


@lru_cache(maxsize=None)
def _analysis(example):
    from resonances.resonance.analysis.mmr import analyse as analyse_module, io

    return analyse_module.analyse_mmr(io.load_series(example), S)


@pytest.mark.benchmark
def test_gq107_a_dot_is_the_107_yr_line():
    columns = _analysis('mmr2__2005_GQ107__40M-57+0+17').response.to_columns()
    assert '7M-10' in columns['rsp_line_ids'] and '57 P_M = P_conj' in columns['rsp_line_ids']
    assert columns['rsp_line_period_yr'] == pytest.approx(57 * P_MARS, rel=0.01)
    assert columns['rsp_line_share'] > 0.5


@pytest.mark.benchmark
def test_59358_a_dot_is_the_conjunction_cycle():
    columns = _analysis('mmr2__59358__8N-13+0+5').response.to_columns()
    assert columns['rsp_line_ids'].startswith('13 P_N = P_conj; 3N-5; 5N-8')
    assert columns['rsp_line_period_yr'] == pytest.approx(P_CONJ, rel=0.01)
    assert columns['rsp_line_share_all'] > 0.7
