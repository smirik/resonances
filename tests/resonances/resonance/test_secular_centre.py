import numpy as np

from resonances.resonance.classify.models import ResonanceStatus
from resonances.resonance.secular_centre import check_centre, summary_fields
from resonances.simulation.config import SimulationConfig

Config = SimulationConfig(save=None, plot=None, save_summary=False)  # the defaults: T/10, 30 deg, 20 deg


T = np.linspace(0.0, 1e7, 20001)


def test_a_drifting_centre_lowers_a_libration():
    # 110 deg in 10 Myr (0.04 arcsec/yr) with 270-kyr forced wiggles of 30 deg: bounded, never turns
    sigma = np.radians(-110.0 * T / 1e7 + 30.0 * np.sin(2 * np.pi * T / 2.7e5))
    status, check = check_centre(ResonanceStatus.LIBRATION, T, sigma, Config)
    assert status == ResonanceStatus.BASELINE_TOO_SHORT
    assert check.lowered and len(check.turning) == 0 and 80 < check.drift_deg < 110
    assert check.comment.startswith('centre: drifts')
    assert summary_fields(check) == {'centre_check': 'lowered', 'centre_drift_deg': round(check.drift_deg, 1), 'centre_turning_points': 0}
    assert summary_fields(None) == {'centre_check': None, 'centre_drift_deg': None, 'centre_turning_points': None}


def test_a_slow_libration_keeps_its_status():
    # period 4 Myr, amplitude 80 deg: extrema at 1, 3, 5, 7, 9 Myr, all inside the 0.5..9.5 Myr centre
    sigma = np.radians(80.0 * np.sin(2 * np.pi * T / 4e6))
    status, check = check_centre(ResonanceStatus.LIBRATION, T, sigma, Config)
    assert status == ResonanceStatus.LIBRATION and not check.lowered and len(check.turning) == 5


def test_a_cycle_and_a_half_is_the_least_a_moving_centre_needs():
    # period 6.67 Myr: extrema at 1.67, 5.0, 8.33 Myr -> 3 turning points (1020 Arcadia in z1: 1.3, 4.6, 8.0)
    sigma = np.radians(80.0 * np.sin(2 * np.pi * T / (1e7 / 1.5)))
    status, check = check_centre(ResonanceStatus.LIBRATION, T, sigma, Config)
    assert status == ResonanceStatus.LIBRATION and len(check.turning) == Config.secular_min_turning_points == 3


def test_one_cycle_is_too_few():
    # 80 cos(2 pi t / 8 Myr): extrema at 4 and 8 Myr inside the 0.5..9.5 Myr centre -> 2 turning
    # points, one cycle (3701 Purkyne in z1, period ~8 Myr)
    sigma = np.radians(80.0 * np.cos(2 * np.pi * T / 8e6))
    status, check = check_centre(ResonanceStatus.LIBRATION, T, sigma, Config)
    assert status == ResonanceStatus.BASELINE_TOO_SHORT and check.lowered and len(check.turning) == 2
    assert 'turns back 2 time(s), fewer than 3' in check.comment


def test_a_single_u_is_lowered():
    # 2005 GQ107-like: the body's g drifts through the commensurability once; the angle rises
    # 100 deg and comes back (half a sine over the baseline) with 300-kyr forced wiggles of 30 deg
    sigma = np.radians(100.0 * np.sin(np.pi * T / 1e7) + 30.0 * np.sin(2 * np.pi * T / 3e5))
    status, check = check_centre(ResonanceStatus.LIBRATION, T, sigma, Config)
    assert status == ResonanceStatus.BASELINE_TOO_SHORT and len(check.turning) == 1


def test_a_fast_libration_has_a_fixed_centre_and_keeps_its_status():
    # Tina-like: 600 kyr, 35 deg about 180 deg; the 1 Myr centre stands still
    sigma = np.radians(180.0 + 35.0 * np.sin(2 * np.pi * T / 6e5))
    status, check = check_centre(ResonanceStatus.LIBRATION, T, sigma, Config)
    assert status == ResonanceStatus.LIBRATION and check.drift_deg < Config.secular_centre_drift


def test_other_statuses_are_left_alone():
    sigma = np.radians(-110.0 * T / 1e7)
    for status in (ResonanceStatus.TRANSIENT, ResonanceStatus.NON_RESONANT, ResonanceStatus.NEAR_SEPARATRIX):
        assert check_centre(status, T, sigma, Config) == (status, None)
