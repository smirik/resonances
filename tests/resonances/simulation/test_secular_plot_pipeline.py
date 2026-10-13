"""Secular diagnostics through Simulation, the classifier, the centre check and DataManager.

The bodies are built by hand, not integrated, so these tests stay in the fast suite:
- a nu6 librator like (1222) Tina: the forced term at g6 (0.2) is larger than the free one (0.07),
  which turns 2.2 arcsec/yr faster, so g - g6 librates about pi with a period of 1296000/2.2 = 589 kyr;
- a drift like 2110 in g - g6 - 2g7 + 2g8: g sits 0.04 arcsec/yr off (110 deg in 10 Myr) and a forced
  g6 term half the free one wiggles the angle by 30 deg at |g - g6| (268 kyr), which keeps the
  classifier's trend-to-oscillation ratio low enough for a 2.
"""

import matplotlib

matplotlib.use('Agg')

import os  # noqa: E402

import numpy as np  # noqa: E402

from resonances.data.const import PLANETARY_FREQUENCIES as F  # noqa: E402
from resonances.data.const import PLANETARY_PHASES as PHASES  # noqa: E402
from resonances.plotting.secular_plots import SecularDiagnostics  # noqa: E402
from resonances.simulation.simulation import Simulation  # noqa: E402

YEARS, N = 1e7, 20001
SECULAR_PLOTS = ['combined', 'portrait', 'angle', 'rate']


def _rad(arcsec_per_year):
    return np.deg2rad(arcsec_per_year / 3600.0)


def synthetic_secular(tmp_path, resonance, g, forced6, free=0.07, drift6=0.0, planets=None, **config):
    """A body built by hand; `drift6` (radians, scalar or per sample) shifts the real g6 phase off the
    linear law, `planets` goes to the integration engine's planets_data before the angles are built."""
    sim = Simulation(
        name='secular_plots',
        date='2000-01-01 12:00',  # J2000: the mode phases apply as tabulated
        save=None,
        plot='all',
        plots=SECULAR_PLOTS,
        save_summary=False,
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        integration_years=YEARS,
        Nout=N,
        coherence_enabled=False,
        **config,
    )
    sim.body_manager.add_body({'a': 2.79, 'e': 0.1, 'inc': 0.1, 'Omega': 0.0, 'omega': 0.0, 'M': 0.0}, resonance, name='synthetic')
    sim.times = np.linspace(0.0, sim.config.tmax, N)
    body = sim.bodies[0]
    body.setup_vars_for_simulation(sim.times)
    t = sim.times / (2 * np.pi)
    varpi6 = _rad(F['g6']) * t + np.deg2rad(PHASES['g6']) + drift6
    z_e = forced6 * np.exp(1j * (varpi6 + np.pi)) + free * np.exp(1j * _rad(g) * t)
    Omega = _rad(-40.0) * t
    body.ecc[:] = np.abs(z_e)
    body.varpi[:] = np.angle(z_e)
    body.Omega[:] = Omega
    body.omega[:] = body.varpi - Omega
    body.inc[:] = 0.1
    body.axis[:] = 2.79
    key = body.resonances()[0].to_s()
    body.angles_unwrapped[key] = np.zeros(N)  # the integration's angle; replaced by the proper one
    sim.integration_engine.planets_data.update(planets or {})
    sim.prepare_angles()
    sim.build_periodograms()
    sim.identify_librations()
    return sim, body, body.resonances()[0]


def test_nu6_librator_is_classified_plotted_and_its_libration_followed(tmp_path):
    sim, body, resonance = synthetic_secular(tmp_path, 'g-g6', g=F['g6'] + 2.2, forced6=0.2)
    assert body.statuses['g-g6'] == 2
    d = SecularDiagnostics.from_body(body, resonance, sim.config)
    assert d.centre.drift_deg < sim.config.secular_centre_drift  # the 1 Myr centre stands still
    # the 1 Myr centre stands still, so the curve follows the 589 kyr libration on a quarter of it:
    # 17 cycles in 10 Myr, minus half a window at each end -> about 32 turning points
    assert d.window < 0.2e6
    assert 28 <= len(d.turning) <= 34
    assert abs(d.mean_rate) < 0.01
    # the angle is bounded by the forced g6 term; the free vector turns at g = g6 + 2.2"/yr, and the
    # 589 kyr is 2 pi / 2.2"/yr: a libration around the forced equilibrium, as Tina's
    assert abs(body.secular_free_divisor(resonance) - 2.2) < 0.01

    sim.data_manager.plot_body(body, sim)
    files = {name for _, _, names in os.walk(tmp_path) for name in names}
    for kind in SECULAR_PLOTS:
        assert f'synthetic-g-g6-{kind}.png' in files, kind
    assert os.path.exists(os.path.join(tmp_path, 'combined', 'synthetic-g-g6-combined.png'))


DRIFT_G = F['g6'] + 2 * F['g7'] - 2 * F['g8'] - 0.04


def test_a_drift_is_lowered_by_the_centre_check(tmp_path):
    sim, body, _ = synthetic_secular(tmp_path, 'g-g6-2g7+2g8', g=DRIFT_G, forced6=0.05, free=0.1)
    key = 'g-g6-2g7+2g8'
    assert body.statuses[key] == -6
    assert body.librations[key].status.value == -6
    assert 'centre: drifts' in body.librations[key].comments
    assert body.centre_checks[key].lowered
    assert abs(body.secular_free_divisor(body.resonances()[0]) + 0.04) < 0.01  # built with divisor -0.04"/yr


def test_without_the_centre_check_the_drift_stays_a_libration(tmp_path):
    sim, body, _ = synthetic_secular(tmp_path, 'g-g6-2g7+2g8', g=DRIFT_G, forced6=0.05, free=0.1, secular_centre_check=False)
    assert body.statuses['g-g6-2g7+2g8'] == 2


def test_measured_planet_phases_follow_a_drifting_mode(tmp_path):
    # Saturn's g6 phase drifts 60 deg in 10 Myr (quadratic). The nu6 librator's forced term follows the
    # real Saturn, so against the linear law its centre walks 60 deg; against the measured phase it
    # stays at 180 deg.
    t = np.linspace(0.0, YEARS, N)
    drift = np.radians(60.0) * (t / YEARS) ** 2
    z_saturn = 0.048 * np.exp(1j * (_rad(F['g6']) * t + np.deg2rad(PHASES['g6']) + drift)) + 0.033 * np.exp(1j * (_rad(F['g5']) * t + 0.5))
    zeta_saturn = 0.008 * np.exp(1j * (_rad(F['s6']) * t + 1.0)) + 0.0138 * np.exp(1j * 1.9)
    saturn = {'e': np.abs(z_saturn), 'varpi': np.angle(z_saturn), 'inc': 2 * np.arcsin(np.abs(zeta_saturn)), 'Omega': np.angle(zeta_saturn)}

    centres = {}
    for phases in ('linear', 'measured'):
        _, body, _ = synthetic_secular(
            tmp_path,
            'g-g6',
            g=F['g6'] + 2.2,
            forced6=0.2,
            drift6=drift,
            planets={'Saturn': saturn},
            secular_planet_phases=phases,
            save_planets=True,
        )
        sigma = np.degrees(np.unwrap(body.secular_angles_proper['g-g6']))
        tenth = N // 10
        centres[phases] = (np.mean(sigma[:tenth]), np.mean(sigma[-tenth:]))
    linear, measured = centres['linear'], centres['measured']
    assert 45.0 < linear[1] - linear[0] < 60.0  # the 60 deg drift, seen from the first and the last Myr
    assert abs(measured[1] - measured[0]) < 5.0
