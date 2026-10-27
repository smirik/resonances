"""The well rule of the two-body strength (analysis.mmr.averaged_potential.swing_in_well, strength.direct_theory): an
evaluation of the direct theory counts only when the observed half-cycle librates in the theoretical well.

Independent truths:
- planar exterior 2:3 with Neptune, circular planet: sigma = 3 lambda - 2 lambda_N - varpi librates about 180 deg (the
  Plutinos; M&D 1999 Table 8.1, f_31 > 0: <R> ~ f_31 e cos sigma has its minimum at 180 deg), the top of <R> is the
  conjunction at perihelion, sigma = 0 (Gallardo 2006, Icarus 184, sect. 3.3);
- planar exterior 1:2, e = 0.3: two asymmetric wells either side of 180 deg, a minor maximum at 180 deg and the global
  one at 0 deg (Gallardo 2006, sect. 3.1, Fig. 3; the package's 119979 1N-2 sits at 284.8 deg); a horseshoe around
  both wells crosses the minor maximum and is a libration;
- 482824 19N-32: its observed centre is 178 deg from the minimum and every half-cycle crosses the global maximum near
  350 deg (W7 strength re-check, Claude + Codex); the ratio 7.8 is in range by accident, F must fail.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.geometry import Episode, EpisodeShape, RejectReason
from resonances.resonance.analysis.mmr import averaged_potential as ap
from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue

SETTINGS = settings_from_config()
RULE = st.quadrature(SETTINGS)
NEPTUNE_AXIS = 30.07
NEPTUNE_MASS = 5.1503e-5
CIRCLE = Orbit(0.0, 0.0, 0.0, 0.0)
PLUTINO_AXIS = NEPTUNE_AXIS / (2 / 3) ** (2 / 3)
TWOTINO_AXIS = NEPTUNE_AXIS * 2 ** (2 / 3)
DEG = np.pi / 180


def _plutino_well():
    body = ap.Ellipse(PLUTINO_AXIS, Orbit(0.1, 0.0, 0.0, 0.0))
    return ap.centre_curvature(body, ap.Ellipse(NEPTUNE_AXIS, CIRCLE), 3, -2, 0.0, np.pi, RULE)


def test_plutino_well_centre_and_top():
    result = _plutino_well()
    assert result.measured
    assert np.degrees(result.centre_rad) == pytest.approx(180.0, abs=0.5)
    assert float(ap.circular_distance(result.global_max_rad, 0.0)) < 360 / SETTINGS.mmr_strength_sigma_grid * DEG
    centre, top = result.centre_rad, result.global_max_rad
    assert ap.swing_in_well(centre, top, 170 * DEG, 190 * DEG)  # a small libration about 180 deg
    assert ap.swing_in_well(centre, top, 100 * DEG, 300 * DEG)  # a large one, still below the top
    assert not ap.swing_in_well(centre, top, -10 * DEG, 10 * DEG)  # about the top: neither condition holds
    assert not ap.swing_in_well(centre, top, 90 * DEG, 430 * DEG)  # holds the centre but passes the top (482824)
    # the same swings one turn up the unwrapped axis
    assert ap.swing_in_well(centre, top, 530 * DEG, 550 * DEG)
    assert not ap.swing_in_well(centre, top, 450 * DEG, 790 * DEG)
    assert not ap.swing_in_well(np.nan, top, 170 * DEG, 190 * DEG)


def test_twotino_horseshoe_crosses_only_the_inner_maximum():
    body = ap.Ellipse(TWOTINO_AXIS, Orbit(0.3, 0.0, 0.0, 0.0))
    planet = ap.Ellipse(NEPTUNE_AXIS, CIRCLE)
    trailing = ap.centre_curvature(body, planet, 2, -1, 0.0, 285 * DEG, RULE)
    leading = ap.centre_curvature(body, planet, 2, -1, 0.0, 75 * DEG, RULE)
    assert trailing.measured and leading.measured
    assert np.degrees(trailing.centre_rad) == pytest.approx(288.7, abs=1.0)  # asymmetric wells, not 180 deg
    assert np.degrees(leading.centre_rad) == pytest.approx(71.3, abs=1.0)
    assert np.degrees(trailing.global_max_rad) == pytest.approx(0.0, abs=1e-9)
    top = trailing.global_max_rad
    assert ap.swing_in_well(trailing.centre_rad, top, 20 * DEG, 340 * DEG)  # horseshoe: through 180 deg, not 0
    assert ap.swing_in_well(trailing.centre_rad, top, 260 * DEG, 310 * DEG)  # a tadpole about the trailing well
    assert not ap.swing_in_well(trailing.centre_rad, top, 240 * DEG, 400 * DEG)  # through 0 deg


# --- direct_theory and the check on a synthetic Plutino record ------------------------------------------------------


def _record(turning_points_deg):
    """A Plutino on a fixed orbit (e = 0.1, varpi = 0) and a saved circular Neptune; one episode with these TPs."""
    t = np.arange(0.0, 20000.0, 10.0)
    n = len(t)
    zeros, ones = np.zeros(n), np.ones(n)
    spec = ResonanceSpec.from_key('2N-3+0+1')  # sigma = 2 lambda_N - 3 lambda + varpi: the same centre and top
    neptune = PlanetSeries('Neptune', t, NEPTUNE_AXIS * ones, zeros, zeros, zeros, zeros)
    series = MMRSeries(
        t, PLUTINO_AXIS * ones, 0.1 * ones, zeros, zeros, zeros, zeros, zeros, spec, Regime(10.0, t[-1], n), {'Neptune': neptune}
    )
    values = np.radians(np.asarray(turning_points_deg, float))
    times = np.linspace(2000.0, 18000.0, len(values))
    middles = (values[1:] + values[:-1]) / 2
    episode = Episode(
        1, times, values, True, EpisodeShape.STATIONARY, RejectReason.NONE, *([np.nan] * 6),
        float(np.mod(np.mean(middles), 2 * np.pi)), 0.0, 0.0, True, 0, 0.0, 0.0,
    )  # fmt: skip
    return series, episode


def _strength(series, episode, required):
    context = SimpleNamespace(
        planet_mass={'Neptune': NEPTUNE_MASS},
        e_med=0.1,
        crosses=(),
        mu_scale_no_eq=NEPTUNE_MASS,
        order=1,
        body_coefficient=series.resonance.body_coefficient,
        mean_motion_rad_yr=2 * np.pi * PLUTINO_AXIS**-1.5,
    )
    physics = SimpleNamespace(episode_id=episode.episode_id, mu_eff=required)
    return st.analyse_strength(context, [physics], SETTINGS, series, SimpleNamespace(runs=[episode]), 100.0)


def test_direct_theory_counts_only_swings_in_the_well():
    theory_value = NEPTUNE_MASS * PLUTINO_AXIS * _plutino_well().curvature
    inside, inside_episode = _record([150, 210, 150, 210, 150, 210, 150])
    strength = _strength(inside, inside_episode, theory_value)
    assert strength.check == CheckValue.PASS and strength.not_measured_reason == ''
    assert strength.n_evaluations == strength.n_consistent == SETTINGS.mmr_strength_snapshots
    assert strength.ratio == pytest.approx(1.0, rel=1e-6)

    # the same orbit, the swing through the top of <R>: the strengths agree, but this is no libration in the well
    over, over_episode = _record([90, 430, 90, 430, 90, 430, 90])
    strength = _strength(over, over_episode, theory_value)
    assert strength.check == CheckValue.FAIL and strength.not_measured_reason == st.REASON_CENTRE_OUTSIDE_WELL
    assert strength.n_consistent == 0 and strength.n_evaluations == SETTINGS.mmr_strength_snapshots
    assert strength.ratio == pytest.approx(1.0, rel=1e-6)  # the range over every evaluation stays as a column
    assert strength.to_columns()['str_n_consistent'] == 0

    # a swing about the top itself: the descent finds the 180 deg well, which the swing does not hold
    top, top_episode = _record([-20, 20, -20, 20, -20, 20, -20])
    assert _strength(top, top_episode, theory_value).not_measured_reason == st.REASON_CENTRE_OUTSIDE_WELL


def test_482824_fails_f_outside_the_well():
    example = 'mmr2__482824__19N-32+0+13'
    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    # W20's stall cut (LIBRATION_REGIME_CUTS) splits 482824's long rejected run and the decisive episode moves to the
    # 10-TP stall at 33.7-56.6 kyr; the numbers below are those of the episode before the cut
    strength = analyse_mmr(io.load_series(example), replace(SETTINGS, libration_regime_cuts=False)).strength
    assert strength.check == CheckValue.FAIL and strength.not_measured_reason == st.REASON_CENTRE_OUTSIDE_WELL
    assert strength.n_consistent == 0 and strength.n_evaluations == SETTINGS.mmr_strength_snapshots
    assert strength.ratio_min == pytest.approx(7.81, rel=0.02)  # in range by itself: the well rule decides
