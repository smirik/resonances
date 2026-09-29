"""End-to-end tests of the coherence step inside Simulation and DataManager.

Bodies are built by hand rather than integrated, so these stay in the fast suite.
"""

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from resonances.resonance import coherence_analysis as ca  # noqa: E402
from resonances.resonance.classify.models import ResonanceStatus  # noqa: E402
from resonances.data.const import PLANETARY_FREQUENCIES  # noqa: E402
from resonances.resonance.omega_free_gate import GateOutcome  # noqa: E402
from resonances.secular import free_elements  # noqa: E402
from resonances.secular.free_elements import RAD_PER_ARCSEC  # noqa: E402
from resonances.simulation.serializer import SimulationSerializer  # noqa: E402
from resonances.simulation.simulation import Simulation  # noqa: E402


# Chosen so that with n = 30000 samples and n_seg = 4 the line falls exactly on a
# frequency bin (nperseg / period = 7500 / 1250 = 6) and inside the validity band
# (P_max = T / (5 * n_seg) = 1500 yr).
ZLK_PERIOD = 1250.0


def build_simulation(tmp_path, n=30_000, antiphase=True, **config_kwargs):
    """Simulation holding one synthetic Lidov-Kozai body, ready for identify_librations().

    e and i share a ZLK_PERIOD line; omega is bounded, so the classifier reports
    LIBRATION and the gate has something to confirm or reject.
    """
    years = float(n)
    kwargs = dict(
        name='coherence_pipeline',
        save=None,
        plot=None,
        save_summary=False,
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        integration_years=years,
        Nout=n,
        libration_period_min=0,
        coherence_period_min=200.0,
    )
    kwargs.update(config_kwargs)
    sim = Simulation(**kwargs)
    sim.body_manager.add_body(
        {'a': 2.5, 'e': 0.2, 'inc': 0.5, 'Omega': 0.3, 'omega': 1.0, 'M': 0.5},
        'LK',
        name='synthetic',
    )
    sim.times = np.linspace(0.0, sim.config.tmax, sim.config.Nout)

    body = sim.bodies[0]
    body.times = sim.times
    t_years = sim.times / (2 * np.pi)
    rng = np.random.default_rng(0)
    common = np.sin(2 * np.pi * t_years / ZLK_PERIOD)

    body.axis = np.full(n, 2.5)
    body.axis_filtered = body.axis
    body.ecc = 0.2 + 0.05 * common + 0.002 * rng.standard_normal(n)
    body.inc = 0.5 + (-0.05 if antiphase else 0.05) * common + 0.002 * rng.standard_normal(n)
    omega_unwrapped = 1.0 + 0.3 * common  # bounded -> libration
    body.omega = np.mod(omega_unwrapped, 2 * np.pi)
    body.Omega = np.mod(0.3 + 2 * np.pi * t_years / 8000.0, 2 * np.pi)

    resonance = body.resonances()[0]
    key = resonance.to_s()
    body.angles_unwrapped[key] = omega_unwrapped
    body.angles[key] = body.omega
    body.angles_filtered[key] = body.omega
    body.angles_filtered_unwrapped[key] = omega_unwrapped
    return sim, body, resonance


def build_free_gate_simulation(tmp_path, free_rate_e, free_rate_i, **config_kwargs):
    """A body whose element vectors and whose resonant angle are supplied independently.

    The resonant angle is a bounded wobble, so the classifier always reports LIBRATION;
    the element vectors carry free pericentre and node rates chosen per test, so the free
    omega either stands still or winds. What this exercises is the wiring — a LIBRATION
    goes in, the gate's verdict comes out and reaches every place a status is read.

    The two are supplied separately on purpose. A body whose osculating omega is bounded
    while its free omega winds cannot be assembled from planetary fundamentals on a 500 kyr
    window: pinning the osculating angle needs the forced pericentre and node to co-rotate,
    and no pair of real g and s modes does. Real objects manage it — 591986 does it for
    30 Myr — through a forced model this basis does not contain, which is why the physics
    of the case is settled by the fixtures in test_omega_free_gate_reference.py and not here.
    """
    years, n = 5e5, 5000
    kwargs = dict(
        name='free_gate',
        save=None,
        plot=None,
        save_summary=False,
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        integration_years=years,
        Nout=n,
        libration_period_min=0,
        coherence_enabled=False,
    )
    kwargs.update(config_kwargs)
    sim = Simulation(**kwargs)
    sim.body_manager.add_body({'a': 2.5, 'e': 0.2, 'inc': 0.5, 'Omega': 0.3, 'omega': 1.0, 'M': 0.5}, 'LK', name='synthetic')
    sim.times = np.linspace(0.0, sim.config.tmax, sim.config.Nout)

    body = sim.bodies[0]
    body.times = sim.times
    t = sim.times / (2 * np.pi)
    turn = lambda amplitude, arcsec: amplitude * np.exp(1j * arcsec * RAD_PER_ARCSEC * t)  # noqa: E731
    z_e = turn(0.03, PLANETARY_FREQUENCIES['g5']) + turn(0.20, free_rate_e)
    z_i = turn(0.02, PLANETARY_FREQUENCIES['s6']) + turn(0.30, free_rate_i)

    body.axis = np.full(n, 2.5)
    body.axis_filtered = body.axis
    body.ecc, body.inc = np.abs(z_e), 2.0 * np.arcsin(np.abs(z_i))
    # The elements have to round-trip: free_elements rebuilds varpi as Omega + omega, so
    # body.omega must be the true argument of the vectors.
    body.Omega = np.angle(z_i)
    body.omega = np.angle(z_e) - body.Omega

    resonance = body.resonances()[0]
    key = resonance.to_s()
    angle = 1.0 + 0.3 * np.sin(2 * np.pi * t / 40_000.0)  # bounded -> LIBRATION
    body.angles_unwrapped[key] = angle
    body.angles[key] = np.mod(angle, 2 * np.pi)
    body.angles_filtered[key] = body.angles[key]
    body.angles_filtered_unwrapped[key] = angle
    return sim, body, resonance


class TestEiExchangeInPipeline:
    """The coherence check records a flag and leaves every status alone."""

    def test_an_antiphase_line_is_recorded_as_an_exchange(self, tmp_path):
        sim, body, resonance = build_simulation(tmp_path, antiphase=True)

        sim.identify_librations()

        key = resonance.to_s()
        assert body.zlk_gates[key].confirmed_ei is True
        assert body.zlk_gates[key].check == ca.CoherenceCheck.PASSED
        assert body.statuses[key] == ResonanceStatus.LIBRATION

    def test_an_in_phase_line_no_longer_demotes(self, tmp_path):
        # This used to be a 2 -> -2 demotion. The e-i band on a short baseline sits below
        # the real Lidov-Kozai periods, so the measurement is kept and the verdict is not.
        sim, body, resonance = build_simulation(tmp_path, antiphase=False)

        sim.identify_librations()

        key = resonance.to_s()
        assert body.zlk_gates[key].confirmed_ei is False
        assert body.statuses[key] == ResonanceStatus.LIBRATION

    def test_the_flag_reaches_summary_csv(self, tmp_path):
        sim, body, resonance = build_simulation(tmp_path, antiphase=False)

        sim.identify_librations()
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)

        row = summary.iloc[0]
        assert bool(row['confirmed_ei']) is False
        assert row['zlk_check'] == 'failed'
        assert row['zlk_line_period'] == pytest.approx(ZLK_PERIOD, rel=0.02)

    def test_disabled_coherence_computes_nothing(self, tmp_path):
        sim, body, resonance = build_simulation(tmp_path, antiphase=False, coherence_enabled=False)

        sim.identify_librations()

        assert body.coherence == {}
        assert body.zlk_gates == {}
        assert body.statuses[resonance.to_s()] == ResonanceStatus.LIBRATION

    def test_a_short_baseline_records_why_nothing_was_measured(self, tmp_path):
        # P_max = T/(5*n_seg) is 1500 yr at the coarsest level, so demanding periods above
        # 2000 yr leaves no usable band, and "not measured" must not read as "not ZLK".
        sim, body, resonance = build_simulation(tmp_path, antiphase=False, coherence_period_min=2000.0)

        sim.identify_librations()
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)

        assert body.zlk_gates[resonance.to_s()].check == ca.CoherenceCheck.SHORT_BASELINE
        assert summary.iloc[0]['zlk_check'] == 'short_baseline'
        assert summary.iloc[0]['confirmed_ei'] is None


class TestFreeOmegaGateInPipeline:
    def test_a_shared_free_rate_is_confirmed(self, tmp_path):
        sim, body, resonance = build_free_gate_simulation(tmp_path, free_rate_e=-200.0, free_rate_i=-200.0)

        sim.identify_librations()

        key = resonance.to_s()
        assert body.statuses[key] == ResonanceStatus.LIBRATION
        assert body.free_gates[key].outcome is GateOutcome.CONFIRMED
        assert 'gate: 2 confirmed' in body.librations[key].comments

    def test_a_carousel_is_demoted_and_reaches_summary_csv(self, tmp_path):
        # The osculating omega is bounded because both vectors are forced-dominated, but
        # the free pericentre and node turn 400 arcsec/yr apart. summary.csv builds its
        # status from the flattened classification result, so the gate has to update that
        # object too, not only body.statuses.
        sim, body, resonance = build_free_gate_simulation(tmp_path, free_rate_e=200.0, free_rate_i=-200.0)

        sim.identify_librations()
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)

        key = resonance.to_s()
        assert body.statuses[key] == ResonanceStatus.LIBRATION_UNCERTAIN
        row = summary.iloc[0]
        assert row['status'] == ResonanceStatus.LIBRATION_UNCERTAIN
        assert row['free_gate'] == 'kinematic'
        assert row['free_b_omega'] == pytest.approx(400.0, rel=0.05)
        assert 'gate: 2->-2 kinematic libration' in row['comments']

    def test_a_disabled_gate_keeps_the_original_status(self, tmp_path):
        sim, body, resonance = build_free_gate_simulation(tmp_path, free_rate_e=200.0, free_rate_i=-200.0, free_gate_enabled=False)

        sim.identify_librations()

        key = resonance.to_s()
        assert body.statuses[key] == ResonanceStatus.LIBRATION
        assert body.free_gates[key].outcome is GateOutcome.DISABLED

    def test_a_window_too_short_for_the_libration_withholds_the_verdict(self, tmp_path):
        sim, body, resonance = build_free_gate_simulation(tmp_path, free_rate_e=-200.0, free_rate_i=-200.0, free_gate_min_cycles=1000.0)

        sim.identify_librations()

        key = resonance.to_s()
        assert body.statuses[key] == ResonanceStatus.UNCERTAIN
        assert body.free_gates[key].outcome is GateOutcome.SHORT_WINDOW


class TestSummaryAndCoherenceCsv:
    def test_forced_and_free_eccentricity_columns(self, tmp_path):
        sim, body, _ = build_simulation(tmp_path)

        sim.identify_librations()
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)

        expected_forced, expected_free = body.forced_free_eccentricity()
        assert summary.iloc[0]['e_forced'] == pytest.approx(expected_forced)
        assert summary.iloc[0]['e_free'] == pytest.approx(expected_free)

    def test_free_element_columns_report_the_reduced_basis(self, tmp_path):
        # 30 kyr cannot separate any planetary mode from any other, so the whole candidate
        # list collapses to one cluster per vector and a single effective arrow is fitted.
        sim, body, _ = build_simulation(tmp_path)

        sim.identify_librations()
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)

        row = summary.iloc[0]
        assert body.free_elements is not None
        assert row['free_basis_e'] == 'g5'
        assert row['free_basis_i'] == 'const'
        assert set(free_elements.SUMMARY_COLUMNS) <= set(summary.columns)

    def test_disabled_free_elements_compute_nothing(self, tmp_path):
        sim, body, _ = build_simulation(tmp_path, free_elements_enabled=False)

        sim.identify_librations()

        assert body.free_elements is None
        assert body.free_gates == {}

    def test_the_omega_free_series_are_written(self, tmp_path):
        sim, body, _ = build_simulation(tmp_path, save='all')
        sim.identify_librations()

        sim.data_manager.save_body(body, sim.times)

        frame = pd.read_csv(tmp_path / 'synthetic-omega-free.csv')
        assert len(frame) == len(body.free_elements.times)
        assert {'varpi_free_unwrapped', 'Omega_free_unwrapped', 'omega_free_unwrapped', 'masked_e', 'masked_i'} <= set(frame.columns)

    def test_simulation_json_records_what_the_baseline_allowed(self, tmp_path):
        # A verdict read months later is not interpretable without the basis it came from,
        # and the basis follows from the baseline.
        sim, _, _ = build_simulation(tmp_path)
        sim.identify_librations()

        setup = SimulationSerializer._build_simulation_json(sim.config, sim.bodies, sim)['simulation']['free_elements']

        assert setup['basis_e'] == ['g5'] and setup['basis_i'] == ['const']
        assert setup['clusters_e'] == [['g5', 'g6', 'g7', 'g8', 'const']]
        assert setup['resolution_arcsec_per_year'] == pytest.approx(1.296e6 / setup['baseline_years'])
        assert 'g5' in setup['frequencies']

    def test_comments_stays_the_last_column(self, tmp_path):
        sim, _, _ = build_simulation(tmp_path)

        sim.identify_librations()
        summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)

        assert list(summary.columns)[-1] == 'comments'

    def test_coherence_csv_is_written_and_appended(self, tmp_path):
        sim, _, _ = build_simulation(tmp_path)
        sim.identify_librations()

        first = sim.data_manager.save_coherence_summary(sim.bodies)
        sim.data_manager.save_coherence_summary(sim.bodies)

        path = tmp_path / 'coherence.csv'
        assert path.exists()
        assert not first.empty
        assert {'body', 'resonance', 'pair', 'period', 'gamma2', 'phase_deg', 'amp_ratio'} <= set(first.columns)
        # Appended a second time: twice the rows, header written only once.
        assert sum(1 for _ in path.open()) == 2 * len(first) + 1

    def test_empty_coherence_writes_no_file(self, tmp_path):
        sim, _, _ = build_simulation(tmp_path, coherence_enabled=False)
        sim.identify_librations()

        sim.data_manager.save_coherence_summary(sim.bodies)

        assert not (tmp_path / 'coherence.csv').exists()


class TestPlotDispatch:
    def test_new_plot_kinds_write_files(self, tmp_path):
        sim, body, resonance = build_simulation(
            tmp_path,
            n=20_000,
            plot='all',
            plots=['ecc_vector', 'cross_spectrum', 'free_omega'],
            coherence_period_min=200.0,
        )
        sim.identify_librations()

        sim.data_manager.plot_body(body, sim)

        key = resonance.to_s()
        assert (tmp_path / 'synthetic-ecc-vector.png').exists()
        assert (tmp_path / f'synthetic-{key}-coherence-e-i.png').exists()
        assert (tmp_path / f'synthetic-{key}-free-omega-drift.png').exists()
        assert (tmp_path / f'synthetic-{key}-free-omega-portrait.png').exists()
        assert (tmp_path / f'synthetic-{key}-free-omega-vector.png').exists()

    def test_plot_kinds_are_opt_in(self, tmp_path):
        sim, body, _ = build_simulation(tmp_path, plot='all', plots=['evolution'])
        sim.identify_librations()

        sim.data_manager.plot_body(body, sim)

        assert not list(tmp_path.glob('*ecc-vector*'))
        assert not list(tmp_path.glob('*coherence-*'))
        assert not list(tmp_path.glob('*free-omega*'))
