import numpy as np
import pytest

import resonances

LK_ASTEROIDS = [
    (1373, 2),
    (3040, 2),
    (15527, 0),
]

LK_SIMULATION_CONFIG = dict(
    tmax=int(50000 * 2 * np.pi),
    integrator='SABA(10,6,4)',
    dt=1.0,
    Nout=5000,
    save=None,
    plot=None,
    save_summary=False,
    libration_period_min=1000,
    libration_period_critical=20000,
    periodogram_frequency_min=1e-5,
    periodogram_frequency_max=0.002,
)


@pytest.mark.slow
def test_real_lidov_kozai_statuses():
    resonance_name = resonances.LidovKozaiResonance().to_s()
    sim = resonances.check([number for number, _ in LK_ASTEROIDS], "lidov-kozai", **LK_SIMULATION_CONFIG)

    sim.run(progress=False)

    summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)
    assert not summary.empty

    for asteroid, expected_status in LK_ASTEROIDS:
        row = summary.loc[(summary['name'] == str(asteroid)) & (summary['resonance'] == resonance_name)]
        assert not row.empty
        status = int(row['status'].iloc[0])

        if expected_status == 0:
            assert status == 0
        else:
            assert abs(status) == expected_status

        assert row['c1'].notna().all()
        assert row['c2'].notna().all()
        assert row['c'].notna().all()
