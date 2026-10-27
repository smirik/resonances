"""W20 CS5 (F1): the 1:1 quasi-satellite floor from the averaged potential, rule G1q.

The tadpole floor of a co-orbital angle (111 yr for the Earth) and its 55.5-yr cutoff erase a quasi-satellite
libration of 8-80 yr: 2010 CG236, 2010 NT134 and 2023 FW13 were uncertain/stall, the quasi-satellite part of 469219
and 164207 was invisible. The period comes from the well of the averaged potential about theta = 0
(`coorbital.quasi_satellite_well`); the record is analysed again at the floor 0.5 P_QS when the orbit-averaged angle
stays in the well (`quasi_satellite`, `analyse._quasi_satellite_pass`).
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.geometry import FloorWell
from resonances.resonance.analysis.mmr import coorbital, decide, io
from resonances.resonance.analysis.mmr import analyse as analyse_module
from resonances.resonance.analysis.mmr import context as context_module
from resonances.resonance.analysis.mmr import quasi_satellite as qs_module
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Action, AngleSubtype, Status

from tests.resonances.analysis.test_impl_w3_guards_ledger_columns import missing_numbers

S = settings_from_config()
MU_EARTH_MOON = 3.040e-6  # (M_E + M_Moon) / M_sun
N_EARTH = 2 * math.pi  # rad/yr


def needs(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    return io.load_series(example)


def synthetic_table(potential) -> coorbital.CoorbitalTable:
    theta = 2 * np.pi * np.arange(720) / 720
    orbit = coorbital.Orbit(0.0, 0.0, 0.0, 0.0)
    return coorbital.CoorbitalTable(theta, potential(theta), np.gradient(potential(theta), theta), orbit, orbit)


# --------------------------------------------------------------------------------------------- the well


@pytest.mark.parametrize(
    'potential, centre, curvature, halfwidth',
    [
        # R~ = -cos theta: minimum at 0 (R~'' = 1), maxima at 180 deg
        (lambda x: -np.cos(x), 0.0, 1.0, math.pi),
        # shifted by 0.1 rad: the same well about 0.1
        (lambda x: -np.cos(x - 0.1), 0.1, 1.0, math.pi),
        # R~ = -cos 3 theta: minima at 0 and +-120 deg (R~'' = 9), maxima at +-60 deg and 180 deg
        (lambda x: -np.cos(3 * x), 0.0, 9.0, math.pi / 3),
    ],
)
def test_well_of_analytic_potentials(potential, centre, curvature, halfwidth):
    found = coorbital.quasi_satellite_well(synthetic_table(potential))
    assert found is not None
    assert found[0] == pytest.approx(centre, abs=1e-4)
    assert found[1] == pytest.approx(curvature, rel=1e-3)
    assert found[2] == pytest.approx(halfwidth, rel=1e-3)


def test_no_well_where_the_potential_peaks_at_zero():
    """R~ = cos theta (a maximum at 0, the minimum at 180 deg holds no 0), and the planar circular problem with the
    Hill softening: the planet itself peaks at theta = 0 and the minima nearest 0 are L4/L5 at +-60 deg."""
    assert coorbital.quasi_satellite_well(synthetic_table(np.cos)) is None
    circular = coorbital.Orbit(0.0, 0.0, 0.0, 0.0)
    softening = coorbital.hill_softening(MU_EARTH_MOON, S.mmr_coorbital_softening_hill)
    assert coorbital.quasi_satellite_well(coorbital.averaged_table(circular, circular, 720, softening)) is None


def _rk4_period(force, theta_c: float, amplitude: float, scale: float, dt: float, steps: int) -> float:
    """Period of theta'' = -scale F~(theta) from theta_c + amplitude at rest (RK4), from the upward crossings of theta_c."""
    grid = 2 * np.pi * np.arange(len(force)) / len(force)

    def accel(x):
        return -scale * np.interp(np.mod(x, 2 * np.pi), np.append(grid, 2 * np.pi), np.append(force, force[0]))

    x, v, t, crossings = theta_c + amplitude, 0.0, 0.0, []
    for _ in range(steps):
        k1x, k1v = v, accel(x)
        k2x, k2v = v + dt / 2 * k1v, accel(x + dt / 2 * k1x)
        k3x, k3v = v + dt / 2 * k2v, accel(x + dt / 2 * k2x)
        k4x, k4v = v + dt * k3v, accel(x + dt * k3x)
        nx, nv = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x), v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        if x < theta_c <= nx:
            crossings.append(t + dt * (theta_c - x) / (nx - x))
        x, v, t = nx, nv, t + dt
    return float(np.mean(np.diff(crossings)))


@pytest.mark.parametrize('e, inc_deg', [(0.15, 5.4), (0.10, 8.0), (0.05, 0.7)])
def test_qs_period_against_an_integration_of_the_force(e, inc_deg):
    """Same table, different path: the small-amplitude period of theta'' = -3 mu n^2 F~(theta), integrated by RK4 with
    the table's own force F~ = <dR/dlambda> (a separate quadrature from R~), against 2 pi / sqrt(3 mu n^2 R~''(theta_c))
    from the spline of R~: within 1 % (amplitude 0.05 of the well's half-width; the narrow e = 0.05 well, 4.8 deg on the
    0.5-deg grid, agrees to 0.8 %, and to 0.1 % on a 4x finer grid)."""
    body = coorbital.Orbit(e, math.radians(inc_deg), math.radians(40.0), math.radians(120.0))
    planet = coorbital.Orbit(0.0167, 0.0, math.radians(102.9), 0.0)
    softening = coorbital.hill_softening(MU_EARTH_MOON, S.mmr_coorbital_softening_hill)
    table = coorbital.averaged_table(body, planet, S.mmr_coorbital_grid_points, softening)
    found = coorbital.quasi_satellite_well(table)
    assert found is not None
    theta_c, curvature, halfwidth = found
    scale = 3 * MU_EARTH_MOON * N_EARTH**2
    expected = 2 * math.pi / math.sqrt(scale * curvature)
    period = _rk4_period(table.force, theta_c, 0.05 * halfwidth, scale, expected / 400, 4000)
    assert period == pytest.approx(expected, rel=0.01)


# --------------------------------------------------------------------------------------------- against the records

# measured periods of the quasi-satellite libration (E report and the W19 F1 review, a fine cutoff on the raw angle)
MEASURED_QS_PERIOD_YR = {
    'coorbital__2010_CG236__1E-1+0+0': (8.0, 8.0),
    'coorbital__2010_NT134__1E-1+0+0': (51.0, 55.0),
    'coorbital__2023_FW13__1E-1+0+0': (51.0, 55.0),
    'coorbital__164207__1E-1+0+0': (67.0, 84.0),
}


def _first_pass_well(series):
    context = context_module.build_context(series, S)
    from resonances.resonance.analysis.mmr import preprocess, sampling as sampling_module

    sampling = sampling_module.analyse_sampling(series, context, S)
    prep = preprocess.preprocess(series, context, sampling, S)
    return qs_module.measure(series, context, prep, sampling.step_yr, S)


@pytest.mark.parametrize('example', sorted(MEASURED_QS_PERIOD_YR))
def test_qs_period_matches_the_measured_libration(example):
    """P_QS,min from the averaged potential (the elements in blocks of 55.5 yr, the planet's mean J2000 orbit) is within
    a factor 1.5 of the libration period measured on the record."""
    found = _first_pass_well(needs(example))
    low, high = MEASURED_QS_PERIOD_YR[example]
    assert found is not None
    assert low / 1.5 <= found.period_min_yr <= high * 1.5


# --------------------------------------------------------------------------------------------- verdicts


@pytest.mark.parametrize(
    'example', ['coorbital__2010_CG236__1E-1+0+0', 'coorbital__2010_NT134__1E-1+0+0', 'coorbital__2023_FW13__1E-1+0+0']
)
def test_quasi_satellites_librate(example):
    """uncertain/stall before W20; at the quasi-satellite floor one librating episode covers the record, and the 1:1
    force law of the planet's mass explains a (K/K_theory near 1)."""
    analysis = analyse_mmr(needs(example), S)
    verdict = analysis.verdict
    assert (verdict.status, verdict.angle_status) == ('resonant', 'libration')
    assert analysis.extra_columns['geo_qs_pass'] == 'kept'
    names = [entry.name for entry in verdict.decision_path[:2]]
    assert names == [
        'G1q the orbit-averaged angle stays in the quasi-satellite well',
        'G1q quasi-satellite floor from the averaged potential',
    ]
    assert analysis.numbers_row()['rsp_coorb_regime'] == 'quasi_satellite'
    assert 0.5 <= analysis.numbers_row()['rsp_coorb_K_over_K_theory'] <= 2.0
    assert missing_numbers(analysis) == []
    # W20 round C: the harmonic e^q strength does not judge a 1:1 angle (CG236 1 kyr had ratio 1800 and the flag
    # implausible_strength on this resonant quasi-satellite); its own law above judges the scale
    assert 'implausible_strength' not in verdict.flags and verdict.checks['check_mmr_strength'] == 'not_applicable'


def test_469219_quasi_satellite_then_horseshoe():
    """The quasi-satellite (24-257 yr, period ~39 yr) and the horseshoe (426-950 yr) join without a circulation."""
    analysis = analyse_mmr(needs('coorbital__469219__1E-1+0+0'), S)
    assert (analysis.verdict.status, analysis.verdict.angle_status, analysis.verdict.angle_subtype) == (
        'resonant',
        'libration',
        'regime_change',
    )
    assert analysis.verdict.confidence == 'high'
    periods = sorted(e.period_med_yr for e in decide.librating_episodes(analysis.geometry, S))
    assert periods[0] == pytest.approx(39.0, rel=0.1)  # the quasi-satellite, below the 111-yr tadpole floor


def test_164207_quasi_satellite_episode_is_seen():
    """The quasi-satellite episode (43-521 yr, period ~70 yr) is accepted at its floor. The horseshoe part (turning
    points at 752 and 932 yr) never crosses its mid-level before 752 yr; it closes by its return at 0.2 of its swing
    (W20 FIX2, LIBRATION_RETURN_LEVELS), and the two regimes without a circulation between them cover the record:
    resonant (author 2026-10-19)."""
    analysis = analyse_mmr(needs('coorbital__164207__1E-1+0+0'), S)
    assert analysis.extra_columns['geo_qs_pass'] == 'kept'
    in_well = [e for e in decide.librating_episodes(analysis.geometry, S) if analysis.geometry.floor_well.holds(e)]
    assert len(in_well) == 1 and in_well[0].period_med_yr == pytest.approx(70.0, rel=0.1)
    assert analysis.verdict.status == 'resonant' and analysis.verdict.angle_subtype == 'regime_change'


def test_2021_ed2_is_not_high():
    """The angle stays in the quasi-satellite well for 26 yr, less than 1.5 P_QS (46 yr): no second pass; the one
    horseshoe cycle closed by its return keeps it transient/episodic at medium."""
    analysis = analyse_mmr(needs('coorbital__2021_ED2__1E-1+0+0'), S)
    assert analysis.extra_columns['geo_qs_pass'] == ''
    assert analysis.extra_columns['geo_qs_visit_yr'] < analysis.extra_columns['geo_qs_visit_needed_yr']
    assert (analysis.verdict.status, analysis.verdict.angle_status, analysis.verdict.confidence) == ('transient', 'episodic', 'medium')


def test_2019_gf1_pass_that_loses_the_horseshoe_is_rejected():
    """26 yr in the well (1.07 x 1.5 P_QS): the pass at the 2.6-yr cutoff finds a 14-yr blip of 4 TPs in the well but
    loses the horseshoe cycle the first pass closed by its return: rejected, the first pass stands (transient/high)."""
    analysis = analyse_mmr(needs('coorbital__2019_GF1__1E-1+0+0'), S)
    assert analysis.extra_columns['geo_qs_pass'] == 'rejected'
    assert analysis.prep.cutoff.cutoff_yr == pytest.approx(55.5, rel=0.01)
    assert (analysis.verdict.status, analysis.verdict.angle_status) == ('transient', 'episodic')


@pytest.mark.parametrize('example', ['coorbital__2002_AA29__1E-1+0+0', 'coorbital__3753__1E-1+0+0'])
def test_no_second_pass_on_librating_co_orbitals(example):
    """2002 AA29 (horseshoe and quasi-satellite, a libration over the record) and 3753 (horseshoe, never in the
    well long enough) keep their first pass."""
    analysis = analyse_mmr(needs(example), S)
    assert analysis.extra_columns['geo_qs_pass'] == ''
    assert analysis.verdict.angle_status == 'libration'
    assert not any(entry.name.startswith('G1q') for entry in analysis.verdict.decision_path)


def test_a_trojan_never_visits_the_quasi_satellite_well():
    """617 Patroclus (L5/L4 tadpole of Jupiter, i = 20 deg): in some blocks the planet's softened peak sits at -0.5 deg
    and the L4 basin reaches past 0; that minimum at 60 deg is not a quasi-satellite well, so the angle never visits
    one, and every well found is centred within 3 deg of 0."""
    found = _first_pass_well(needs('coorbital__617__1J-1+0+0'))
    assert found is not None
    assert found.visit_yr == 0.0
    assert np.nanmax(np.abs(found.well.centre_rad)) < math.radians(3.0)


def test_switched_off_the_stall_stays():
    """MMR_COORBITAL_QS_FLOOR=False: 2010 NT134 is the stall of W19 and asks for a longer record."""
    analysis = analyse_mmr(needs('coorbital__2010_NT134__1E-1+0+0'), settings_from_config(mmr_coorbital_qs_floor=False))
    assert (analysis.verdict.status, analysis.verdict.angle_subtype) == ('uncertain', 'stall')
    assert analysis.verdict.action == 'extend'
    assert math.isnan(analysis.extra_columns['flt_qs_period_yr'])


# --------------------------------------------------------------------------------------------- floors and actions


def _well(outside_floor=111.0):
    t = np.linspace(0.0, 1000.0, 1001)
    return FloorWell(t, np.zeros_like(t), np.full_like(t, math.radians(15.0)), outside_floor, 'mmr_coorbital_period_floor_factor')


def _episode(centre_deg, period_yr=40.0):
    return SimpleNamespace(t_start_yr=100.0, t_end_yr=500.0, centre_mod_2pi_rad=math.radians(centre_deg), period_med_yr=period_yr)


def test_episode_floor_inside_and_outside_the_well():
    """In the quasi-satellite well (centre 355 deg, 5 deg from 0 < 15 deg) the pass's own floor holds; a horseshoe
    centred at 180 deg keeps the tadpole floor."""
    geometry = SimpleNamespace(period_floor_yr=20.0, period_floor_setting='mmr_coorbital_period_floor_factor', floor_well=_well())
    assert decide.period_floor(geometry, S, _episode(355.0))[0] == 20.0
    assert decide.period_floor(geometry, S, _episode(180.0))[0] == 111.0
    assert decide.period_floor(geometry, S)[0] == 20.0


def _quasi_satellite(realizable: bool, visit_yr: float = 500.0) -> qs_module.QuasiSatellite:
    """P_QS = 6 yr along a 1-kyr record; the angle in the well for `visit_yr` against 1.5 P_QS = 9 yr."""
    t = np.linspace(0.0, 1000.0, 11)
    well = coorbital.QuasiSatelliteWell(t, np.zeros_like(t), np.full_like(t, 0.2), np.full_like(t, 6.0))
    return qs_module.QuasiSatellite(well, 1, visit_yr, 0.0, 9.0, 3.0, 111.0, 'mmr_coorbital_period_floor_factor', 1.5, realizable, 55.5)


@pytest.mark.parametrize('realizable, action, step', [(True, Action.REVIEW, None), (False, Action.RESAMPLE, 0.25)])
def test_a_stall_faster_than_the_cutoff_is_not_extended(realizable, action, step):
    """P_QS = 6 yr below P_c = 55.5 yr and the angle in the well: a longer record cures nothing. With a step of 1 yr the
    quasi-satellite cutoff (1.5 yr) cannot be filtered (1.5 <= 2 x 1): resample at the largest round step
    <= min(6 / 20, < 0.75) = 0.25 yr. A resonant verdict, or a stall that never enters the well (2015 TL144 1M-1),
    keeps its action."""
    from resonances.resonance.analysis.ledger import Ledger

    angle = SimpleNamespace(subtype=AngleSubtype.STALL)
    prep = SimpleNamespace(cutoff=SimpleNamespace(cutoff_yr=55.5))
    sampling = SimpleNamespace(safe_step_plain_yr=float('nan'), safe_step_pred_yr=float('nan'), step_yr=1.0)

    def action_of(found, status):
        return analyse_module._quasi_satellite_stall_action(
            found, angle, SimpleNamespace(status=status), prep, sampling, S, Ledger.from_settings(S)
        )

    assert action_of(_quasi_satellite(realizable), Status.UNCERTAIN) == (action, step)
    assert action_of(_quasi_satellite(realizable), Status.RESONANT) is None
    assert action_of(_quasi_satellite(realizable, visit_yr=0.0), Status.UNCERTAIN) is None
