"""W4 integration fixes on bodies built by hand (no integration), with the series of test_restore_and_resume:
LIB (4J-2S-1+0+0-1) is a pendulum libration of known period (8 kyr), so its verdict and spectrum are known.

- restore(recompute_librations=True) rebuilds the periodograms of the recomputed angles (Codex W4 review, P2);
- a saved derived row whose parent was not saved is analysed alone, keeping its parent (P2);
- a planet whose saved times are not the body's is left out of the MMR series, not realigned (P2);
- a secular/ZLK row writes its analysis' numbers, never the TTO columns, and its episodes go to episodes.csv.
"""

import numpy as np
import pandas as pd

from resonances.simulation import SimulationSerializer, mmr_adapter
from tests.resonances.simulation.test_restore_and_resume import LIB, LIBRATION_PERIOD, N, _saved_run, _series, _simulation

DERIVED = '4J-2S-1-1+0+0'  # a planar subresonance of LIB: the same lambda part, varpi_J in place of varpi


def test_recompute_rebuilds_the_spectrum_of_the_angle(tmp_path):
    _saved_run(tmp_path)
    path = tmp_path / 'data' / 'data-librator-periodograms.csv'
    spectra = pd.read_csv(path)
    spectra[f'{LIB}_power'] = -123.0  # a sentinel no periodogram can produce
    spectra.to_csv(path, index=False)

    kept = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
    again = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=True)

    assert np.all(kept.bodies[0].periodogram_power[LIB] == -123.0)  # read back as saved
    body = again.bodies[0]
    power, frequency = body.periodogram_power[LIB], body.periodogram_frequency[LIB]
    assert np.all(power >= 0)
    # the strongest line is the libration itself: 8 kyr within the frequency resolution of the 100 kyr record
    assert abs(1 / frequency[np.argmax(power)] - LIBRATION_PERIOD) < 0.15 * LIBRATION_PERIOD


def test_a_saved_child_of_an_unsaved_parent_is_analysed_alone(tmp_path):
    sim = _simulation(tmp_path)
    sim.body_manager.add_body({'a': 2.6, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, [LIB], name='child')
    sim.times = np.linspace(0.0, sim.config.tmax, N)
    t = sim.times / (2 * np.pi)
    axis, angles = _series(t)
    body = sim.bodies[0]
    body.setup_vars_for_simulation(sim.times)
    body.axis[:] = axis
    body.ecc[:] = 0.1
    body.add_derived_resonance(DERIVED, LIB, angle_wrapped=np.mod(angles[LIB], 2 * np.pi), angle_unwrapped=angles[LIB].copy())

    sim.identify_librations(only={'child': {DERIVED}})  # what a restore passes when only the child was saved

    analysis = body.mmr_analyses[DERIVED]
    assert analysis.parent == LIB and body.derived_resonances[DERIVED] == LIB  # provenance kept
    assert LIB not in body.librations  # the parent's series was not saved: it is not analysed
    assert body.statuses[DERIVED] == str(body.librations[DERIVED].status) == 'resonant'


def test_a_planet_on_another_grid_is_left_out(tmp_path):
    sim = _simulation(tmp_path, save_planets='all')
    sim.body_manager.add_body({'a': 2.6, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, [LIB], name='grid')
    sim.times = np.linspace(0.0, sim.config.tmax, N)
    t = sim.times / (2 * np.pi)
    axis, angles = _series(t)
    body = sim.bodies[0]
    body.setup_vars_for_simulation(sim.times)
    body.axis[:] = axis
    body.ecc[:] = 0.1
    body.angles_unwrapped[LIB] = angles[LIB].copy()
    step = t[1] - t[0]

    def planet(times):
        zeros = np.zeros(N)
        columns = ('a', 'e', 'inc', 'Omega', 'omega', 'M', 'l', 'varpi')
        return {'times': times, **{column: zeros + (5.2 if column == 'a' else 0.0) for column in columns}}

    sim.integration_engine.planets_data = {'Jupiter': planet(t.copy()), 'Saturn': planet(t + 0.5 * step), 'Mars': planet(t[::-1].copy())}
    series = mmr_adapter.series_from_body(body, body.mmrs[0], sim)
    # Saturn is half a step off and Mars in reverse order: both left out; the angle's planets go all or none (J, S)
    assert set(series.planets) == set()
    sim.integration_engine.planets_data = {'Jupiter': planet(t.copy()), 'Saturn': planet(t + 1e-4 * step), 'Mars': planet(t.copy())}
    series = mmr_adapter.series_from_body(body, body.mmrs[0], sim)
    assert set(series.planets) == {'Jupiter', 'Saturn', 'Mars'}  # within 1e-3 of a step: the same instants


def test_a_secular_row_writes_its_numbers_and_episodes(tmp_path):
    """The nu6 librator of test_secular_plot_pipeline (forced e 0.2 against a free 0.07 turning at g6 + 2.2"/yr: the
    angle librates around the forced equilibrium, 17 cycles in 10 Myr), saved: its summary row carries the analysis'
    numbers and no TTO column, its runs go to episodes.csv, and restore(recompute_librations=True) reads the unwrapped
    angle alone and gives the same verdict."""
    from tests.resonances.simulation.test_secular_plot_pipeline import F, synthetic_secular

    sim, body, resonance = synthetic_secular(tmp_path, 'g-g6', g=F['g6'] + 2.2, forced6=0.2)
    sim.config.save, sim.config.plot, sim.config.save_summary = 'all', None, True
    sim.data_manager.save_data(sim.bodies, sim.times, sim)
    key = resonance.to_s()

    summary = pd.read_csv(tmp_path / 'summary.csv')
    row = summary[summary['resonance'] == key].iloc[0]
    assert row['resonance_type'] == 'secular' and row['status'] == body.statuses[key] == 'resonant'
    assert row['flt_cutoff_yr'] > 0 and row['geo_n_tp'] == body.classifier_numbers[key]['geo_n_tp'] > 0
    for legacy in ('chaos_flag', 'n_good_total', 'legacy_type', 'centre_check', 'metrics_trend_to_oscillation'):
        assert legacy not in summary.columns
    episodes = pd.read_csv(tmp_path / 'episodes.csv')
    assert (episodes['resonance'] == key).sum() == len(body.secular_analyses[key].episode_rows()) > 0
    assert not (tmp_path / 'segments.csv').exists()

    again = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=True)
    assert again.bodies[0].statuses[key] == body.statuses[key]
    assert str(again.bodies[0].librations[key].confidence) == str(body.librations[key].confidence)
