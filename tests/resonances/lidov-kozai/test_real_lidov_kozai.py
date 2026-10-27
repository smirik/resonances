import numpy as np
import pytest

import resonances
from resonances.resonance.analysis.verdict import AngleStatus, LibrationOrigin, Status

# True: omega librates (the old |status| = 2: confirmed by the free-omega gate, or kinematic);
# False: nothing bounded (the old 0)
LK_ASTEROIDS = [
    (1373, True),
    (3040, True),
    (15527, False),
]

# 500 kyr, the baseline the free-omega gate is calibrated for. Shorter does not just cost
# precision: one Rayleigh width at 50 kyr is 26 arcsec/yr, wider than the whole spread of
# inner Solar System secular rates, so nothing can be told apart from planetary forcing and
# every librator is withheld at -9. See docs/free-elements.md.
# The date is fixed: 3040 crosses the planets' orbits, and over 500 kyr of SABA dt 1 a start a few weeks earlier or later
# is another trajectory (W24, 6 dates: libration 3 times, episodic twice, circulation once). All three cases pass at
# 2026-06-10 (Horizons refused 2026-06-09, the AstDyS epoch, when it was measured). The author's 3040 'resonant' is the
# mercurius record, checked by tests/resonances/resonance/test_omega_free_gate_reference.py.
LK_SIMULATION_CONFIG = dict(
    source='nasa',
    date='2026-06-10',
    tmax=int(500000 * 2 * np.pi),
    integrator='SABA(10,6,4)',
    dt=1.0,
    Nout=5000,
    save=None,
    plot=None,
    save_summary=False,
    libration_period_min=1000,
    periodogram_frequency_min=1e-5,
    periodogram_frequency_max=0.002,
)


@pytest.mark.slow
def test_real_lidov_kozai_statuses():
    resonance_name = resonances.LidovKozaiResonance().to_s()
    sim = resonances.check([number for number, _ in LK_ASTEROIDS], "lidov-kozai", **LK_SIMULATION_CONFIG)

    sim.run(progress=False)

    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    assert not summary.empty

    for asteroid, librates in LK_ASTEROIDS:
        row = summary.loc[(summary['name'] == str(asteroid)) & (summary['resonance'] == resonance_name)]
        assert not row.empty
        status, angle_status, origin = row['status'].iloc[0], row['angle_status'].iloc[0], row['libration_origin'].iloc[0]

        if librates:
            assert angle_status == AngleStatus.LIBRATION, (asteroid, status, angle_status)
            # resonant when the gate confirms it, non_resonant when the libration is kinematic
            assert status == Status.RESONANT or (status == Status.NON_RESONANT and origin == LibrationOrigin.KINEMATIC), (
                asteroid,
                status,
                origin,
            )
        else:
            assert status == Status.NON_RESONANT, (asteroid, status)
            assert not isinstance(origin, str) or origin == LibrationOrigin.NONE, (asteroid, origin)

        assert row['c1'].notna().all()
        assert row['c2'].notna().all()
        assert row['h_quad'].notna().all()
