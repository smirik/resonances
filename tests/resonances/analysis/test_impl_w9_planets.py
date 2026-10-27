"""W9 planets: the C11 ripple fix, the planets of the NEA production examples, catalogue longitudes in the owner search.

Truth from closed forms and synthetic records built here:

* the great inequality moves Jupiter and Saturn in antiphase, so 5J+6S keeps |5 x 0.34 - 6 x 0.83| = 3.28 deg and
  3J-1S keeps 3 x 0.34 + 0.83 = 1.85 deg (C11);
* a rate error alpha of a phase that is otherwise fixed lowers the coherence of window centres spread uniformly over T
  to |sinc(alpha T / 2)| (0.9939 at 22 deg, 0.9734 at 46 deg, 0.8046 at 128 deg), whose steepest slope is 0.21809 per
  radian at D = 4.163 rad;
* the synthetic owner 2J-1+0-1 is an integrated pendulum, the analysed angle 5J-2+0-3 = 2 phi_o + psi
  (test_w2_owner.py), here with Jupiter on the table rate so that its catalogue line is its exact longitude.
"""

import dataclasses
import json
import math
import types

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from resonances.data import const
from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.confidence import CAP_CEILING, Cap
from resonances.resonance.analysis.mmr import io, owner
from resonances.resonance.analysis.mmr import preprocess as pre
from resonances.resonance.analysis.mmr.context import CATALOGUE_PLANETS, build_context, catalogue_longitudes
from resonances.resonance.analysis.mmr.models import GreatInequality, MMRSeries, OwnerCandidate, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.mmr.sampling import analyse_sampling
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, Confidence
from resonances.simulation import mmr_adapter

S = settings_from_config()
TWO_PI = 2 * np.pi
DEG = math.pi / 180
GI = GreatInequality(period_yr=937.0, dlambda_jupiter_rad=0.34 * DEG, dlambda_saturn_rad=0.83 * DEG, measured=False)


# --------------------------------------------------------------------------------------------- C11


def test_predicted_ripple_signed_sum_of_the_antiphased_waves():
    gain = filtering.butterworth_gain(937.0, 50.0, 2)
    same_sign = pre.predicted_ripple(ResonanceSpec.from_key('5J+6S-2+0+0-9'), GI, 50.0, 2)
    opposite = pre.predicted_ripple(ResonanceSpec.from_key('3J-1S-1+0+0-1'), GI, 50.0, 2)
    assert same_sign == pytest.approx(3.28 * DEG * gain, rel=1e-9)  # |1.70 - 4.98| deg
    assert opposite == pytest.approx(1.85 * DEG * gain, rel=1e-9)  # 1.02 + 0.83 deg
    # 2J+5S (signs alike, near-cancelling): |0.68 - 4.15| = 3.47 deg, not 4.83
    assert pre.predicted_ripple(ResonanceSpec.from_key('2J+5S-1+0+0-6'), GI, 50.0, 2) == pytest.approx(3.47 * DEG * gain, rel=1e-9)


def test_predicted_ripple_is_the_wave_of_antiphased_longitudes():
    # The GI terms dlambda_J = A_J sin(theta), dlambda_S = -A_S sin(theta) in 5 lambda_J + 6 lambda_S, measured on a grid.
    theta = np.linspace(0, TWO_PI, 10001)
    wave = 5 * GI.dlambda_jupiter_rad * np.sin(theta) + 6 * (-GI.dlambda_saturn_rad) * np.sin(theta)
    half_amplitude = np.ptp(wave) / 2
    huge_cutoff_gain = filtering.butterworth_gain(937.0, 1e-6, 2)  # ~1: the analog gain far below the cutoff
    predicted = pre.predicted_ripple(ResonanceSpec.from_key('5J+6S-2+0+0-9'), GI, 1e-6, 2) / huge_cutoff_gain
    assert predicted == pytest.approx(half_amplitude, rel=1e-6)


# --------------------------------------------------------------------------------------------- the coherence bound


def test_coherence_of_a_drifting_phase_is_the_sinc():
    centres = np.linspace(0.0, 1.0, 20001)  # window centres spread uniformly over T = 1
    for drift_deg, expected in ((22, 0.9939), (46, 0.9734), (128, 0.8046)):
        coherence = filtering.circular_resultant(math.radians(drift_deg) * centres)
        assert coherence == pytest.approx(expected, abs=2e-4)
        sinc = abs(math.sin(math.radians(drift_deg) / 2) / (math.radians(drift_deg) / 2))
        assert 1 - coherence <= owner.SINC_HALF_SLOPE_MAX * math.radians(drift_deg) + 1e-12
        assert coherence == pytest.approx(sinc, abs=1e-4)


def test_sinc_slope_constant_bounds_the_steepest_slope():
    d = np.linspace(1e-4, 40.0, 400001)
    x = d / 2
    slope = np.abs((x * np.cos(x) - np.sin(x)) / x**2) / 2
    assert slope.max() == pytest.approx(0.21809, abs=1e-5)
    assert d[slope.argmax()] == pytest.approx(4.163, abs=1e-3)
    assert owner.SINC_HALF_SLOPE_MAX >= slope.max()  # a bound, never below the true slope


def test_catalogue_phase_error_by_hand():
    # Earth over 18 kyr: min(2.4e-5 x 18000, 2 x 0.37 + 1.1e-5 x 18000) = min(0.432, 0.938) deg, x |k| = 5
    assert owner.catalogue_phase_error(ResonanceSpec.from_key('5E-16+0+11'), ('Earth',), 18000.0) == pytest.approx(2.16 * DEG, rel=1e-9)
    # 5M-9S-4: 5 min(0.666, 1.44 + 0.1098) + 9 min(6.66, 9.38 + 0.153) = 3.33 + 59.94 deg = 1.1043 rad
    spec = ResonanceSpec.from_key('5M-9S-4+0+0+8')
    assert owner.catalogue_phase_error(spec, ('Mars', 'Saturn'), 18000.0) == pytest.approx(63.27 * DEG, rel=1e-9)
    assert owner.catalogue_phase_error(spec, ('Mars',), 18000.0) == pytest.approx(3.33 * DEG, rel=1e-9)  # Saturn saved
    # 500 kyr: the slow bound wins: Earth 2 x 0.37 + 1.1e-5 x 5e5 = 6.24 deg (< 2.4e-5 x 5e5 = 12 deg)
    assert owner.catalogue_phase_error(ResonanceSpec.from_key('5E-16+0+11'), ('Earth',), 5e5) == pytest.approx(5 * 6.24 * DEG, rel=1e-9)


def test_catalogue_error_tables_cover_the_catalogue_planets_only():
    assert 'Mercury' not in CATALOGUE_PLANETS  # its table rate is SABA's: 1.5e-3 deg/yr off in ias15
    assert CATALOGUE_PLANETS == ('Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune')
    for table in (const.PLANET_N_SPREAD_DEG_PER_GAUSS_YR, const.PLANET_N_OFFSET_DEG_PER_GAUSS_YR, const.PLANET_LAMBDA_SLOW_MAX_DEG):
        assert set(table) == set(CATALOGUE_PLANETS)


def test_catalogue_longitudes_carry_the_epoch_value_to_the_first_row():
    # Earth 360.000505 deg/yr, 10 yr before the epoch: -3600.00505 deg = -0.00505 deg mod 360
    values = catalogue_longitudes({'Earth': 0.1, 'Mercury': 1.0, 'Venus': float('nan')}, 0.0, -10.0)
    assert set(values) == {'Earth'}  # Mercury is never catalogued, nan is dropped
    assert values['Earth'] == pytest.approx(0.1 - 0.00505 * DEG, abs=1e-9)


# --------------------------------------------------------------------------------------------- ambiguity rules


def _candidate(**overrides):
    values = dict(
        spec=ResonanceSpec.from_key('5E-16+0+11'),
        rate_rad_yr=0.0,
        r2_median=0.9,
        r2_windows=(0.9, 0.9, 0.9),
        phi0_coherence=0.95,
        psi_share=0.02,
        force_au_yr=1e-4,
        force_theory_au_yr=1e-3,
        strength_ratio=0.1,
        rank=1,
        accepted=False,
        reason='',
    )
    values.update(overrides)
    return OwnerCandidate(**values)


def _regime(sigma_r2=0.85, sigma_coherence=0.5, turns=0):
    return owner.RegimeFit(
        owner.REGIME_LIBRATION, (0.0, 1000.0, 2000.0), 0.9, (0.9, 0.9, 0.9), 0.95, 1e-4, sigma_r2, sigma_coherence, 1e-4, turns
    )


def test_catalogue_ambiguous_near_the_coherence_threshold():
    # mmr_owner_phi0_coh_min 0.8: 0.85 within 0.1 of it is open, 0.95 is not (sigma's coherence far below: phase route)
    assert owner.catalogue_ambiguous(_candidate(phi0_coherence=0.85), _regime(), 0.1, S)
    assert owner.catalogue_ambiguous(_candidate(phi0_coherence=0.75), _regime(), 0.1, S)
    assert not owner.catalogue_ambiguous(_candidate(phi0_coherence=0.95), _regime(), 0.1, S)
    assert not owner.catalogue_ambiguous(_candidate(phi0_coherence=0.65), _regime(), 0.1, S)  # fails clearly
    assert not owner.catalogue_ambiguous(_candidate(phi0_coherence=0.85), _regime(), 0.0, S)  # saved planets


def test_catalogue_ambiguous_near_the_phase_gain():
    # phase route: 0.95 - 0.72 = 0.23, within 0.1 of mmr_owner_phi0_coh_gain 0.2 -> open; the R^2 route (gain 0.2 over
    # sigma's R^2 0.6) carries it whatever the phase
    assert owner.catalogue_ambiguous(_candidate(), _regime(sigma_coherence=0.72), 0.1, S)
    assert not owner.catalogue_ambiguous(_candidate(), _regime(sigma_r2=0.6, sigma_coherence=0.72), 0.1, S)
    assert not owner.catalogue_ambiguous(_candidate(), _regime(sigma_coherence=0.5), 0.1, S)


def test_catalogue_ambiguous_only_when_the_other_tests_pass():
    assert not owner.catalogue_ambiguous(_candidate(phi0_coherence=0.85, r2_median=0.5), _regime(), 0.1, S)
    assert not owner.catalogue_ambiguous(_candidate(phi0_coherence=0.85, psi_share=0.5), _regime(), 0.1, S)
    assert not owner.catalogue_ambiguous(_candidate(phi0_coherence=0.85, psi_share=float('nan')), _regime(), 0.1, S)
    forced = dataclasses.replace(_regime(turns=2), regime=owner.REGIME_FORCED_POSITIVE)  # a ripple
    assert not owner.catalogue_ambiguous(_candidate(phi0_coherence=0.85), forced, 0.1, S)


def test_catalogue_cap_is_medium():
    assert CAP_CEILING[Cap.CATALOGUE_LONGITUDES] == Confidence.MEDIUM


# --------------------------------------------------------------------------------------------- synthetic owner search

N_J = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'])  # Jupiter on the table rate: its catalogue line is exact
VARPI_RATE = TWO_PI / 40000.0
T_REC = np.arange(0.0, 12000.0 + 0.5, 1.0)
OMEGA_LIB = TWO_PI / 400.0
P_LIB_40 = 412.5  # 400 yr x (1 + A^2/16 + 11 A^4/3072), A = 40 deg
LAMBDA_J0 = 0.4


def _pendulum():
    sol = solve_ivp(
        lambda t, y: [y[1], -(OMEGA_LIB**2) * np.sin(y[0])],
        (T_REC[0], T_REC[-1]),
        [math.radians(40), 0.0],
        t_eval=T_REC,
        method='DOP853',
        rtol=1e-11,
        atol=1e-13,
    )
    return sol.y[0], sol.y[1]


PHI, PHIDOT = _pendulum()


def _series(jupiter_rate_error=0.0, saved=True):
    """A body driven by 2J-1+0-1 (the pendulum PHI); the analysed angle 5J-2+0-3. Jupiter turns at the table rate +
    `jupiter_rate_error`; `saved` False keeps only its catalogue longitude."""
    t = T_REC
    lam_j = (N_J + jupiter_rate_error) * t + LAMBDA_J0
    varpi = 1.3 + VARPI_RATE * t
    lam = 2 * lam_j - varpi - PHI
    a = (TWO_PI / (2 * (N_J + jupiter_rate_error) - VARPI_RATE - PHIDOT)) ** (2.0 / 3.0)
    sigma = -2 * lam - 3 * varpi + 5 * lam_j
    constant = {'a': 5.2026, 'e': 0.05, 'inc': 0.01, 'Omega': 1.0, 'varpi': 0.3}
    jupiter = PlanetSeries('Jupiter', t, **{k: np.full_like(t, v) for k, v in constant.items()}, mean_longitude=np.mod(lam_j, TWO_PI))
    planets = {'Jupiter': jupiter} if saved else {}
    Omega = np.full_like(t, 0.7)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=np.full_like(t, 0.1),
        inc=np.full_like(t, 0.05),
        Omega=Omega,
        omega=np.mod(varpi - Omega, TWO_PI),
        varpi=np.mod(varpi, TWO_PI),
        angle_wrapped=np.mod(sigma, TWO_PI),
        resonance=ResonanceSpec.from_key('5J-2+0-3'),
        regime=Regime(step_yr=1.0, span_yr=t[-1] - t[0], n_rows=len(t), planets_saved=tuple(planets)),
        planets=planets,
        mean_longitude=np.mod(lam, TWO_PI),
        name='synthetic',
        catalogue_longitudes={} if saved else {'Jupiter': LAMBDA_J0},
    )


def _search(series):
    context = build_context(series, S)
    sampling = analyse_sampling(series, context, S)
    prep = pre.preprocess(series, context, sampling, S)
    return owner.run_owner(series, prep, context, P_LIB_40, S)


def _top(result, key):
    return next(c for c in result.candidates if c.spec.key == key)


def test_catalogue_line_finds_the_owner_of_the_saved_planet():
    saved, catalogue = _search(_series()), _search(_series(saved=False))
    assert saved.check == catalogue.check == CheckValue.FAIL
    assert saved.accepted.spec.key == catalogue.accepted.spec.key == '2J-1+0-1'
    assert catalogue.accepted.r2_median == pytest.approx(saved.accepted.r2_median, abs=1e-6)
    assert catalogue.accepted.phi0_coherence == pytest.approx(saved.accepted.phi0_coherence, abs=1e-6)
    assert owner.FLAG_CATALOGUE in catalogue.flags and owner.FLAG_CATALOGUE not in saved.flags
    assert owner.FLAG_CATALOGUE_AMBIGUOUS not in catalogue.flags


def test_constant_error_of_lambda_j0_changes_nothing():
    series = _series(saved=False)
    shifted = dataclasses.replace(series, catalogue_longitudes={'Jupiter': LAMBDA_J0 + math.radians(30)})
    first, second = _search(series), _search(shifted)
    for key in ('2J-1+0-1',):
        assert _top(second, key).r2_median == pytest.approx(_top(first, key).r2_median, abs=1e-9)
        assert _top(second, key).phi0_coherence == pytest.approx(_top(first, key).phi0_coherence, abs=1e-9)


def test_rate_error_lowers_the_coherence_within_the_bound():
    # The true Jupiter turns 2e-5 rad/yr faster than the table: the catalogue line misses 2 x 2e-5 t on the owner
    # 2J-1+0-1. Over window centres within the 12-kyr record the coherence drops by at most 0.218 x 2 x 2e-5 x 12000.
    error = 2e-5
    true = _top(_search(_series(jupiter_rate_error=error)), '2J-1+0-1')
    line = _top(_search(_series(jupiter_rate_error=error, saved=False)), '2J-1+0-1')
    drop = true.phi0_coherence - line.phi0_coherence
    assert 0 < drop <= owner.SINC_HALF_SLOPE_MAX * 2 * error * (T_REC[-1] - T_REC[0])
    assert line.r2_median == pytest.approx(true.r2_median, abs=0.01)  # within a window the line is good


def test_without_lambda_j0_there_is_no_search():
    series = dataclasses.replace(_series(saved=False), catalogue_longitudes={})
    assert _search(series).check == CheckValue.NOT_MEASURED


# --------------------------------------------------------------------------------------------- the readers


def _write_solar_system(path, longitudes, t=0.0):
    import rebound

    sim = rebound.Simulation()
    sim.add(m=1.0, hash='Sun')
    for index, name in enumerate(('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune')):
        sim.add(m=1e-9, a=0.4 + index, e=0.01, l=longitudes[name], hash=name, primary=sim.particles[0])
    sim.t = t * TWO_PI
    sim.save_to_file(str(path))


LONGITUDES = {'Mercury': 0.1, 'Venus': 0.2, 'Earth': 0.3, 'Mars': 0.4, 'Jupiter': 0.5, 'Saturn': 0.6, 'Uranus': 0.7, 'Neptune': 0.8}


def test_solar_system_cache_gives_the_epoch_longitudes(tmp_path):
    _write_solar_system(tmp_path / 'solar.bin', LONGITUDES)
    t, longitudes = io.solar_system_longitudes(tmp_path / 'solar.bin')
    assert t == 0.0
    assert longitudes == pytest.approx(LONGITUDES, abs=1e-12)
    assert io.solar_system_longitudes(tmp_path / 'missing.bin') is None


def test_adapter_catalogue_from_the_run_cache(tmp_path):
    _write_solar_system(tmp_path / 'solar.bin', LONGITUDES)
    simulation = types.SimpleNamespace(
        config=types.SimpleNamespace(solar_system_moon=False),
        integration_engine=types.SimpleNamespace(_solar_system_filename=lambda moon=False: str(tmp_path / 'solar.bin')),
    )
    t_yr = np.array([-20.0, -10.0, 0.0])  # a backward run, time-ordered: t_yr[0] is 20 yr before the epoch
    saved = {'Earth': PlanetSeries('Earth', t_yr, *(np.zeros(3) for _ in range(5)), np.zeros(3))}
    values = mmr_adapter._catalogue(simulation, saved, t_yr)
    assert 'Earth' not in values and 'Mercury' not in values
    # Jupiter: 0.5 rad - 20 x 30.349646 deg = 0.5 rad - 606.99292 deg
    assert values['Jupiter'] == pytest.approx(np.mod(0.5 - 606.99292 * DEG, TWO_PI), abs=1e-9)
    assert set(values) == {'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'}


def _example(root, example_id, t, meta_extra):
    folder = root / 'examples' / example_id
    folder.mkdir(parents=True)
    n = len(t)
    frame = pd.DataFrame(
        {
            't_yr': t,
            'a': np.full(n, 3.28),
            'e': np.full(n, 0.1),
            'inc': np.full(n, 0.05),
            'Omega': np.full(n, 0.7),
            'omega': np.full(n, 0.6),
            'varpi': np.full(n, 1.3),
            'M': np.zeros(n),
            'mean_longitude': np.mod(0.5 * t, TWO_PI),
            'angle_raw': np.linspace(0, 1, n),
        }
    )
    frame.to_csv(folder / 'series.csv.gz', index=False)
    meta = {
        'name': 'synthetic',
        'resonance': '5J-2+0-3',
        'resonance_type': 'mmr',
        'regime': {'integrator': 'ias15'},
        'provenance': meta_extra,
    }
    (folder / 'meta.json').write_text(json.dumps(meta))


def _run(root, run_id, t, planets=('Jupiter', 'Saturn', 'Earth')):
    folder = root / 'runs' / run_id
    folder.mkdir(parents=True)
    columns = {'t_yr': t}
    for index, name in enumerate(planets):
        for field in ('a', 'e', 'inc', 'Omega', 'omega', 'M', 'varpi'):
            columns[f'{name}.{field}'] = np.full(len(t), 1.0 + index)
        columns[f'{name}.mean_longitude'] = np.mod(0.1 * (index + 1) + math.radians(const.PLANET_N_DEG_PER_GAUSS_YR[name]) * t, TWO_PI)
    pd.DataFrame(columns).to_csv(folder / 'planets.csv.gz', index=False)


def test_planets_run_id_is_read_before_run_id(tmp_path, monkeypatch):
    monkeypatch.setenv(io.DATA_ENV, str(tmp_path))
    t = np.arange(0.0, 50.0, 5.0)
    _run(tmp_path, 'linked', t)
    _example(tmp_path, 'with_link', t, {'run_id': 'no_planets', 'planets_run_id': 'linked'})
    _example(tmp_path, 'without_link', t, {'run_id': 'no_planets'})
    linked = io.load_series('with_link')
    assert set(linked.planets) == {'Earth', 'Jupiter', 'Saturn'}
    assert linked.planets['Jupiter'].mean_longitude[3] == pytest.approx(np.mod(0.1 + math.radians(30.349646) * 15.0, TWO_PI))
    # the planets the link did not hold have no source: the run file lacks them, there is no solar-system cache
    assert linked.catalogue_longitudes == {}
    assert io.load_series('without_link').planets == {}


def test_catalogue_from_the_run_file_on_another_grid(tmp_path, monkeypatch):
    monkeypatch.setenv(io.DATA_ENV, str(tmp_path))
    _run(tmp_path, 'coarse', np.arange(0.0, 100.0, 10.0))
    t = np.arange(20.0, 70.0, 5.0)  # another grid: no planet series, lambda_j0 carried from t = 0 to t = 20 yr
    _example(tmp_path, 'other_grid', t, {'run_id': 'coarse'})
    series = io.load_series('other_grid')
    assert series.planets == {}
    assert set(series.catalogue_longitudes) == {'Earth', 'Jupiter', 'Saturn'}  # the planets the run file holds
    # Saturn (index 1): 0.2 rad + 20 x 12.221327 deg at t = 20 yr
    assert series.catalogue_longitudes['Saturn'] == pytest.approx(np.mod(0.2 + 20 * 12.221327 * DEG, TWO_PI), abs=1e-9)


def test_catalogue_from_the_solar_system_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(io.DATA_ENV, str(tmp_path))
    _write_solar_system(tmp_path / 'solar.bin', LONGITUDES)
    t = np.arange(0.0, 50.0, 5.0)
    _example(tmp_path, 'cache_only', t, {'run_id': 'nothing', 'solar_system_cache': {'path': str(tmp_path / 'solar.bin')}})
    series = io.load_series('cache_only')
    assert series.catalogue_longitudes == pytest.approx({name: LONGITUDES[name] for name in CATALOGUE_PLANETS}, abs=1e-9)
    assert io.load_series('cache_only', with_planets=False).catalogue_longitudes == {}


def test_series_rejects_a_bad_catalogue_longitude():
    with pytest.raises(ValueError, match='catalogue'):
        dataclasses.replace(_series(saved=False), catalogue_longitudes={'Jupiter': float('nan')}).validate()
    with pytest.raises(ValueError, match='catalogue'):
        dataclasses.replace(_series(saved=False), catalogue_longitudes={'Pluto': 0.1}).validate()


# --------------------------------------------------------------------------------------------- benchmark examples

HW7 = 'mmr3__2001_HW7__5J+6S-2+0+0-9'
UG22 = 'mmr3__2003_UG22__8M-4S-7+0+0+3'


def _benchmark(example_id):
    if not (io.example_path(example_id) / 'meta.json').exists():
        pytest.skip('benchmark data not available')
    return io.load_series(example_id)


def _catalogue_only(series, example_id):
    """The example without planets, the owner search on catalogue lines from the linked run's first row."""
    meta = io.read_meta(example_id)
    return dataclasses.replace(series, planets={}, catalogue_longitudes=io._catalogue(meta, float(series.t_yr[0]), {}))


def test_hw7_reads_the_linked_planets_and_the_catalogue_finds_the_same_owner():
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = _benchmark(HW7)
    assert set(series.planets) == {'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune'}
    assert series.catalogue_longitudes == {}
    real = analyse_mmr(series, S)
    assert real.verdict.owner == '5E-16+0+11' and str(real.verdict.status) == 'non_resonant'
    approx = analyse_mmr(_catalogue_only(series, HW7), S)
    assert approx.owner.accepted is not None and approx.owner.accepted.spec.key == '5E-16+0+11'
    assert approx.owner.accepted.r2_median == pytest.approx(real.owner.accepted.r2_median, abs=0.005)
    assert approx.owner.accepted.phi0_coherence == pytest.approx(real.owner.accepted.phi0_coherence, abs=0.005)
    assert owner.FLAG_CATALOGUE in approx.owner.flags
    assert str(approx.verdict.confidence) == 'medium' and 'catalogue_longitudes' in str(approx.verdict.caps)


def test_ug22_catalogue_owner_is_ambiguous_not_accepted():
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = _benchmark(UG22)
    approx = analyse_mmr(_catalogue_only(series, UG22), S)
    assert approx.owner.check == CheckValue.NOT_MEASURED and approx.owner.accepted is None
    assert owner.FLAG_CATALOGUE_AMBIGUOUS in approx.owner.flags
    assert next(c for c in approx.owner.candidates if c.spec.key == '5M-9S-4+0+0+8').reason == owner.REASON_CATALOGUE_AMBIGUOUS
    assert str(approx.verdict.status) == 'uncertain'
