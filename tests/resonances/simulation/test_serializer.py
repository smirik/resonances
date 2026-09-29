import json
from pathlib import Path

import numpy as np

from resonances.simulation import Simulation, DataManager, SimulationSerializer
from resonances.secular.free_elements import FreeElements


def _build_minimal_simulation(tmp_path: Path) -> Simulation:
    sim = Simulation(
        name="test_serializer",
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        save="all",
        plot=None,
        tmax=10,
        Nout=5,
    )

    elem = {
        "name": "test_body",
        "a": 2.5,
        "e": 0.1,
        "inc": 0.2,
        "Omega": 0.3,
        "omega": 0.4,
        "M": 0.5,
    }
    sim.body_manager.add_body(elem, ["2J+3S-1+0+0-4", "LK"], name="test_body")
    body = sim.bodies[0]

    sim.times = np.linspace(0.0, sim.config.tmax, sim.config.Nout)
    body.setup_vars_for_simulation(sim.times)

    body.axis = np.linspace(2.5, 2.6, sim.config.Nout)
    body.ecc = np.linspace(0.1, 0.15, sim.config.Nout)
    body.inc = np.linspace(0.2, 0.25, sim.config.Nout)
    body.Omega = np.linspace(0.3, 0.35, sim.config.Nout)
    body.omega = np.linspace(0.4, 0.45, sim.config.Nout)
    body.M = np.linspace(0.5, 0.55, sim.config.Nout)
    body.longitude = np.linspace(0.6, 0.65, sim.config.Nout)
    body.varpi = np.linspace(0.7, 0.75, sim.config.Nout)
    body.axis_filtered = body.axis.copy()

    for resonance in body.resonances():
        key = resonance.to_s()
        body.angles_unwrapped[key] = np.linspace(0.0, 1.0, sim.config.Nout)
        body.angles[key] = body.angles_unwrapped[key].copy()
        body.angles_filtered_unwrapped[key] = body.angles_unwrapped[key].copy()
        body.angles_filtered[key] = body.angles_unwrapped[key].copy()

    periodogram_freq = np.array([0.1, 0.2])
    periodogram_power = np.array([1.0, 0.5])
    body.axis_periodogram_frequency = periodogram_freq
    body.axis_periodogram_power = periodogram_power
    for resonance in body.resonances():
        key = resonance.to_s()
        body.periodogram_frequency[key] = periodogram_freq
        body.periodogram_power[key] = periodogram_power

    free_times = np.linspace(0.0, 2_000.0, sim.config.Nout)
    free_phase_e = np.linspace(0.1, 0.5, sim.config.Nout)
    free_phase_i = np.linspace(-0.2, 0.1, sim.config.Nout)
    keep_e = np.array([True, True, False, True, True])
    keep_i = np.array([True, False, True, True, True])
    both = keep_e & keep_i
    z_e_free = 0.08 * np.exp(1j * free_phase_e)
    z_i_free = 0.03 * np.exp(1j * free_phase_i)
    z_e_forced = np.full(sim.config.Nout, 0.02 + 0j)
    z_i_forced = np.full(sim.config.Nout, 0.01 + 0j)
    body.free_elements = FreeElements(
        times=free_times,
        z_e=z_e_free + z_e_forced,
        z_i=z_i_free + z_i_forced,
        z_e_free=z_e_free,
        z_i_free=z_i_free,
        z_e_forced=z_e_forced,
        z_i_forced=z_i_forced,
        basis_e={"g5": 4.25749319},
        basis_i={"const": 0.0},
        dropped_e=["g7"],
        dropped_i=["s7"],
        amplitudes_e={},
        amplitudes_i={},
        keep_e=keep_e,
        keep_i=keep_i,
        varpi_free=free_phase_e[keep_e],
        Omega_free=free_phase_i[keep_i],
        omega_free=free_phase_e[both] - free_phase_i[both],
        b_varpi=0.0,
        b_Omega=0.0,
        b_omega=0.0,
        clusters_e=[["g5", "g7"]],
        clusters_i=[["const", "s7"]],
    )

    manager = DataManager(sim.config)
    manager.save_body(body, sim.times)
    manager.save_configuration_details(sim.bodies, sim)

    return sim


def test_restore_simulation_basic(tmp_path):
    original = _build_minimal_simulation(tmp_path)
    json_path = tmp_path / "simulation.json"

    sim = SimulationSerializer.restore(str(json_path), recompute_librations=False)

    assert sim.config.name == original.config.name
    assert len(sim.bodies) == 1
    assert sim.times is not None

    body = sim.bodies[0]
    assert body.axis is not None
    assert body.axis.shape[0] == len(sim.times)
    assert "2J+3S-1+0+0-4" in body.angles
    assert isinstance(body.angles["2J+3S-1+0+0-4"], np.ndarray)


def test_restore_periodograms_loaded(tmp_path):
    _build_minimal_simulation(tmp_path)
    json_path = tmp_path / "simulation.json"

    sim = SimulationSerializer.restore(str(json_path), recompute_librations=False)
    body = sim.bodies[0]
    key = "2J+3S-1+0+0-4"

    assert body.axis_periodogram_frequency is not None
    assert key in body.periodogram_frequency
    assert key in body.periodogram_power


def test_restore_free_omega_series(tmp_path):
    original = _build_minimal_simulation(tmp_path)
    restored = SimulationSerializer.restore(str(tmp_path / "simulation.json"), recompute_librations=False)

    expected = original.bodies[0].free_elements
    actual = restored.bodies[0].free_elements
    assert actual is not None
    np.testing.assert_allclose(actual.times, expected.times)
    np.testing.assert_array_equal(actual.keep_e, expected.keep_e)
    np.testing.assert_array_equal(actual.keep_i, expected.keep_i)
    np.testing.assert_allclose(actual.omega_free, expected.omega_free)
    np.testing.assert_allclose(np.abs(actual.z_e_free), np.abs(expected.z_e_free))


def test_save_simulation_json_roundtrip(tmp_path):
    sim = _build_minimal_simulation(tmp_path)
    sim.config.save_path = str(tmp_path)

    SimulationSerializer.save_simulation_json(sim.config, sim.bodies, sim)

    saved_path = tmp_path / "simulation.json"
    assert saved_path.exists()

    with saved_path.open("r") as f:
        data = json.load(f)

    assert data["config"]["name"] == sim.config.name
    assert data["simulation"]["number_of_bodies"] == len(sim.bodies)
