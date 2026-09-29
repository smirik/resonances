"""MMR diagnostic plots through Simulation, DataManager and SimulationSerializer.

The body is built by hand, not integrated: an exact inner 2:1 with Jupiter whose angle
librates about pi, so these tests stay in the fast suite.
"""

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.simulation.serializer import SimulationSerializer  # noqa: E402
from resonances.simulation.simulation import Simulation  # noqa: E402

LIBRATION_PERIOD = 400.0  # yr
A_BODY, A_JUPITER = 3.2776, 5.2026
MMR_PLOTS = ['combined', 'recurrence', 'fair', 'portrait', 'cycles']


def synthetic_mmr(tmp_path, resonance='2J-1', n=4000, years=4000.0, planets=True, **config_kwargs):
    """Simulation with one body in `resonance`, its angle librating about pi.

    sigma = 2 lambda_J - lambda - varpi with varpi = 0 and M = lambda, so Jupiter's mean
    longitude follows from sigma exactly: lambda_J = (sigma + lambda) / 2.
    """
    kwargs = dict(
        name='mmr_plots',
        save=None,
        plot='all',
        save_summary=False,
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        integration_years=years,
        Nout=n,
    )
    kwargs.update(config_kwargs)
    sim = Simulation(**kwargs)
    sim.body_manager.add_body({'a': A_BODY, 'e': 0.1, 'inc': 0.05, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, resonance, name='synthetic')
    sim.times = np.linspace(0.0, sim.config.tmax, n)
    body = sim.bodies[0]
    body.times = sim.times
    t = sim.times / (2 * np.pi)
    rng = np.random.default_rng(0)

    phase = 2 * np.pi * t / LIBRATION_PERIOD
    sigma_filtered = np.pi + 0.6 * np.sin(phase)
    sigma_raw = sigma_filtered + 0.01 * rng.standard_normal(n)
    longitude = 2 * np.pi * t / A_BODY**1.5
    body.axis_filtered = A_BODY + 0.002 * np.cos(phase)
    body.axis = body.axis_filtered + 1e-4 * rng.standard_normal(n)
    body.M = np.mod(longitude, 2 * np.pi)
    body.longitude = longitude
    body.ecc = np.full(n, 0.1)
    for name in ('inc', 'Omega', 'omega', 'varpi'):
        setattr(body, name, np.zeros(n))

    resonance = body.resonances()[0]
    key = resonance.to_s()
    body.angles_unwrapped[key] = sigma_raw
    body.angles[key] = np.mod(sigma_raw, 2 * np.pi)
    body.angles_filtered_unwrapped[key] = sigma_filtered
    body.angles_filtered[key] = np.mod(sigma_filtered, 2 * np.pi)
    # Only Jupiter is synthesised; an integrated run would fill every planet.
    sim.integration_engine.planets_data = {}
    if planets:
        sim.integration_engine.planets_data['Jupiter'] = {
            'times': t,
            'l': (sigma_raw + longitude) / 2,
            'a': np.full(n, A_JUPITER),
        }
    return sim, body, resonance


def _finish_run(sim):
    """What Simulation.run() does after integrating, minus prepare_angles (which would
    refilter the synthetic angle): periodograms, classification, saving."""
    sim.build_periodograms()
    sim.identify_librations()
    sim.data_manager.save_data(sim.bodies, sim.times, sim)


class TestDispatch:
    def test_every_mmr_kind_writes_a_file(self, tmp_path):
        sim, body, resonance = synthetic_mmr(tmp_path, plots=MMR_PLOTS)
        sim.data_manager.plot_body(body, sim)
        for kind in MMR_PLOTS:
            assert (tmp_path / f'synthetic-{resonance.to_s()}-{kind}.png').exists(), kind

    def test_kinds_are_opt_in(self, tmp_path):
        sim, body, _ = synthetic_mmr(tmp_path, plots=['portrait'])
        sim.data_manager.plot_body(body, sim)
        assert [p.name.split('-')[-1] for p in tmp_path.glob('*.png')] == ['portrait.png']

    def test_pdf(self, tmp_path):
        sim, body, resonance = synthetic_mmr(tmp_path, plots=['combined'], image_type='pdf', plot_options={'style': 'paper'})
        sim.data_manager.plot_body(body, sim)
        assert (tmp_path / f'synthetic-{resonance.to_s()}-combined.pdf').stat().st_size > 0

    def test_mmr_kinds_skip_other_resonances(self, tmp_path):
        sim, _, _ = synthetic_mmr(tmp_path, plots=MMR_PLOTS)
        sim.body_manager.add_body({'a': 2.5, 'e': 0.2, 'inc': 0.5, 'Omega': 0.0, 'omega': 1.0, 'M': 0.0}, 'LK', name='lk')
        assert sim.bodies[-1].lidov_kozai_resonances
        sim.data_manager.plot_body(sim.bodies[-1], sim)
        assert not list(tmp_path.glob('lk-*'))

    def test_fair_without_planet_data_warns_and_skips(self, tmp_path):
        sim, body, resonance = synthetic_mmr(tmp_path, planets=False, plots=['fair', 'combined'])
        with pytest.warns(UserWarning, match='save_planets'):
            sim.data_manager.plot_body(body, sim)
        assert not (tmp_path / f'synthetic-{resonance.to_s()}-fair.png').exists()
        assert (tmp_path / f'synthetic-{resonance.to_s()}-combined.png').exists()

    def test_coarse_output_warns_but_draws(self, tmp_path):
        # a = 3.2776 au -> P = 5.93 yr; one sample per 5 yr is 0.84 of the orbit.
        sim, body, resonance = synthetic_mmr(tmp_path, n=800, plots=['fair'])
        with pytest.warns(UserWarning, match='aliased'):
            sim.data_manager.plot_body(body, sim)
        assert (tmp_path / f'synthetic-{resonance.to_s()}-fair.png').exists()

    def test_dense_output_draws_fair_without_warning(self, tmp_path, recwarn):
        sim, body, resonance = synthetic_mmr(tmp_path, n=40_000, plots=['fair'])
        sim.data_manager.plot_body(body, sim)
        assert not [w for w in recwarn if 'FAIR' in str(w.message)]
        assert (tmp_path / f'synthetic-{resonance.to_s()}-fair.png').exists()

    def test_three_body_combined_omits_fair_silently(self, tmp_path, recwarn):
        sim, body, resonance = synthetic_mmr(tmp_path, resonance='4J-2S-1', planets=False, plots=['combined'])
        sim.data_manager.plot_body(body, sim)
        assert (tmp_path / f'synthetic-{resonance.to_s()}-combined.png').exists()
        assert not [w for w in recwarn if 'FAIR' in str(w.message)]


class TestRegenerationAfterRestore:
    def test_restored_simulation_redraws_in_paper_style(self, tmp_path):
        """Everything the figures need is saved: a restored run can be redrawn in another
        style, FAIR included, without integrating again."""
        from resonances.plotting import MMRDiagnostics, MMRPlotter

        sim, body, resonance = synthetic_mmr(tmp_path, n=8000, save='all', save_planets=True, plot=None)
        _finish_run(sim)

        restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=False)
        restored_body = restored.bodies[0]
        restored_resonance = restored_body.resonances()[0]

        # The planet series come back as arrays, the same shape as after a live run.
        jupiter = restored.integration_engine.planets_data['Jupiter']
        assert isinstance(jupiter['l'], np.ndarray)
        np.testing.assert_allclose(jupiter['l'], sim.integration_engine.planets_data['Jupiter']['l'])

        original = MMRDiagnostics.from_body(body, resonance, sim)
        again = MMRDiagnostics.from_body(restored_body, restored_resonance, restored)
        assert again.fair_unavailable_reason() is None
        assert len(again.cycles) == len(original.cycles) == 9
        np.testing.assert_allclose(again.sigma_filtered, original.sigma_filtered)

        plotter = MMRPlotter(again, {'style': 'paper'}).plot_combined()
        plotter.save(tmp_path / 'paper.pdf')
        assert plotter.figure.get_size_inches()[0] == pytest.approx(7.09)
        plotter.close()

    def test_restore_ignores_a_removed_plot_kind(self, tmp_path):
        import json

        sim, body, _ = synthetic_mmr(tmp_path, save='all', plot=None)
        _finish_run(sim)
        path = tmp_path / 'simulation.json'
        data = json.loads(path.read_text())
        data['config']['plots'] = ['evolution', 'phase_portrait']
        path.write_text(json.dumps(data))

        restored = SimulationSerializer.restore(str(path), recompute_librations=False)
        assert restored.config.plots == ['evolution']
