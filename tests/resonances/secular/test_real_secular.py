import pandas as pd
import pytest

import resonances
from resonances.secular.secular_resonance import SecularResonance

# z1 = g+s-g6-s6 over 10 Myr, measured once by hand at this date and setup (SABA, dt 0.5):
# - (1020) Arcadia librates, period ~7 Myr, divisor 0.000 (one of the z1 bodies of the finder tests);
# - (363) Padua librates, period ~10 Myr: the Padua family is mostly inside z1 (Carruba 2009). Its
#   osculating angle circulates for 6 of the 10 Myr because the forced g5 term of its eccentricity
#   is as large as the free one; the classifier gave it -5 until the angle was reduced;
# - (847) Agnia circulates slowly (+0.25 arcsec/yr), "right outside the resonance" (Vokrouhlicky et
#   al. 2006, Icarus 183, 349, Fig. 4).
# Built from the planets' osculating longitudes instead of their mode longitudes, z1 turned at -26
# arcsec/yr — exactly -s6, because Saturn's node measured from the ecliptic does not circulate.
Z1 = {'1020': 2, '363': 2}


@pytest.mark.slow
def test_real_z1(tmp_path):
    sim = resonances.Simulation(
        date='2026-09-29',  # fixed: the slow tests drift with the Horizons epoch otherwise
        source='nasa',
        integrator='SABA(10,6,4)',
        dt=0.5,
        integration_years=10_000_000,
        Nout=20001,
        save=None,
        plot=None,
        save_summary=True,  # batch workers report through summary.csv, not through the bodies
        save_path=str(tmp_path),
        coherence_enabled=False,
        batch_enabled=True,
        batch_threshold=1,
        batch_size=1,
        n_cores=3,
    )
    sim.create_solar_system()
    for number in ('1020', '363', '847'):
        sim.add_body(int(number), [SecularResonance('g-g6+s-s6'), SecularResonance('s-s6')], name=number)
    sim.run(progress=False)

    summary = pd.read_csv(tmp_path / 'summary.csv', dtype={'name': str})
    status = {(row['name'], row['resonance']): row['status'] for _, row in summary.iterrows()}
    for name in ('1020', '363', '847'):
        if name in Z1:
            assert status[(name, 'g-g6+s-s6')] == Z1[name], name
        else:
            assert status[(name, 'g-g6+s-s6')] != 2, name
        assert status[(name, 's-s6')] <= 0, name
