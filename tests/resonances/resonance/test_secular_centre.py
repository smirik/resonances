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
    # period 5 Myr, amplitude 80 deg: the centre turns twice in the baseline
    sigma = np.radians(80.0 * np.sin(2 * np.pi * T / 5e6))
    status, check = check_centre(ResonanceStatus.LIBRATION, T, sigma, Config)
    assert status == ResonanceStatus.LIBRATION and not check.lowered and len(check.turning) >= 2


def test_a_fast_libration_has_a_fixed_centre_and_keeps_its_status():
    # Tina-like: 600 kyr, 35 deg about 180 deg; the 1 Myr centre stands still
    sigma = np.radians(180.0 + 35.0 * np.sin(2 * np.pi * T / 6e5))
    status, check = check_centre(ResonanceStatus.LIBRATION, T, sigma, Config)
    assert status == ResonanceStatus.LIBRATION and check.drift_deg < Config.secular_centre_drift


def test_other_statuses_are_left_alone():
    sigma = np.radians(-110.0 * T / 1e7)
    for status in (ResonanceStatus.TRANSIENT, ResonanceStatus.NON_RESONANT, ResonanceStatus.NEAR_SEPARATRIX):
        assert check_centre(status, T, sigma, Config) == (status, None)
