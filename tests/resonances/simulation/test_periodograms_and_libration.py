import numpy as np

from resonances.simulation import Simulation


def _build_simulation(tmp_path):
    sim = Simulation(
        name="test_periodograms",
        save=None,
        plot=None,
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        tmax=2 * np.pi * 10.0,
        Nout=500,
        libration_period_min=0,
        periodogram_frequency_min=0.0,
        periodogram_frequency_max=1.0,
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

    sim.body_manager.add_body(elem, "2J-1", name="test_body")
    sim.times = np.linspace(0.0, sim.config.tmax, sim.config.Nout)
    body = sim.bodies[0]
    body.times = sim.times

    t_years = sim.times / (2 * np.pi)
    # Use the same frequency for both semi-major axis and resonant angle
    # so that periodogram peaks match for MMR classification
    freq = 0.2
    body.axis = 2.5 + 0.01 * np.sin(2 * np.pi * freq * t_years)
    body.axis_filtered = body.axis  # For test purposes, use same data
    body.ecc = 0.1 + 0.01 * np.sin(2 * np.pi * freq * t_years)

    resonance = body.resonances()[0]
    wrapped_angle = (np.pi + 0.5 * np.sin(2 * np.pi * freq * t_years)) % (2 * np.pi)
    body.angles[resonance.to_s()] = wrapped_angle
    body.angles_filtered[resonance.to_s()] = wrapped_angle
    body.angles_filtered_unwrapped[resonance.to_s()] = np.unwrap(wrapped_angle)

    return sim


def test_build_periodograms_populates_arrays(tmp_path):
    sim = _build_simulation(tmp_path)
    body = sim.bodies[0]
    resonance = body.resonances()[0]

    sim.build_periodograms()

    assert body.axis_periodogram_frequency is not None
    assert body.eccentricity_periodogram_frequency is not None
    assert resonance.to_s() in body.periodogram_frequency


def test_identify_librations_sets_status(tmp_path):
    sim = _build_simulation(tmp_path)
    body = sim.bodies[0]
    resonance = body.resonances()[0]

    # Build periodograms first (required for MMR periodogram overlap check)
    sim.build_periodograms()
    sim.identify_librations()

    assert resonance.to_s() in body.librations
    assert body.statuses[resonance.to_s()] == 2
