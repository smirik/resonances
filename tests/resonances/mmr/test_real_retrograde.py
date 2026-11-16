"""Real retrograde resonances (Li, Huang & Gong 2019, Table A.1; Morais & Namouni 2017).

The elements come from Horizons at a pinned date, so the results do not drift with today's epoch.
Centaur libration periods are 0.7-3 kyr, below the default low-pass cutoff (2 kyr): the Jupiter run
filters at 200 yr instead (docs/matrix.md).
"""

import numpy as np
import pytest

import resonances
from resonances.mmr.retrograde_two_body import RetrogradeTwoBody
from resonances.resonance.analysis.verdict import Status
from tests.tools import accepted_episodes

DATE = '2026-10-08'

JUPITER_CONFIG = dict(
    date=DATE,
    source='nasa',
    integrator='ias15',
    integration_years=10000,
    Nout=10000,
    oscillations_cutoff=0.005,
    periodogram_frequency_max=0.005,
    libration_period_min=300,
    save=None,
    plot=None,
    save_summary=False,
)


def row(summary, name, resonance):
    found = summary.loc[(summary['name'] == name) & (summary['resonance'] == resonance)]
    assert len(found) == 1, f'{name} {resonance} not in the summary'
    return found.iloc[0]


def centre_deg(body, key):
    """Median of the unwrapped angle, in [0, 360)."""
    return float(np.degrees(np.median(body.angles_unwrapped[key])) % 360)


def body_named(sim, name):
    return next(b for b in sim.bodies if b.name == name)


@pytest.mark.slow
def test_jupiter_2000_dg8_and_2015_bz509():
    """2000 DG8 in 1/-3 (k=0) about 0 deg and 2015 BZ509 in 1/-1 (k=1) about 180 deg (Li et al. 2019,
    both S = 10); an off-resonance retrograde particle at a = 11.2 circulates at 3 n - n_J (~43 turns).

    DG8 comes through `find`, which must give it the retrograde candidates with Jupiter only.
    """
    sim = resonances.find('2000 DG8', planets=['Jupiter'], type='mmr', **JUPITER_CONFIG)
    dg8_mmrs = sim.bodies[0].mmrs
    assert dg8_mmrs and all(isinstance(m, RetrogradeTwoBody) for m in dg8_mmrs)
    assert {'1J-3r0', '1J-3r1', '1J-3r2'} <= {m.to_s() for m in dg8_mmrs}

    sim.add_body('2015 BZ509', '1J-1r1', name='2015 BZ509')
    sim.add_body(dict(a=11.2, e=0.3, inc=np.radians(150), Omega=1.0, omega=2.0, M=3.0), '1J-3r0', name='particle')
    sim.run()
    summary = sim.data_manager.get_simulation_summary(sim.bodies)

    dg8 = row(summary, '2000 DG8', '1J-3r0')
    assert dg8['status'] == Status.RESONANT
    centre = centre_deg(body_named(sim, '2000 DG8'), '1J-3r0')
    assert min(centre, 360 - centre) < 20  # measured: -8 deg
    assert np.ptp(body_named(sim, '2000 DG8').angles_unwrapped['1J-3r0']) < np.radians(120)  # +-48 deg
    assert 500 < dg8['geo_period_cycle_yr'] < 1000  # measured: 714 yr

    # BZ509 librates about 180 deg with a large amplitude (+-140 deg), and touches the separatrix
    # between 4 and 6 kyr: the classifier may call this run transient rather than resonant.
    bz = row(summary, '2015 BZ509', '1J-1r1')
    assert bz['status'] in (Status.TRANSIENT, Status.RESONANT)
    bz_body = body_named(sim, '2015 BZ509')
    assert abs(centre_deg(bz_body, '1J-1r1') - 180) < 20  # measured: 176 deg
    assert np.ptp(bz_body.angles_unwrapped['1J-1r1']) < np.radians(450)  # no circulation: 389 deg at most
    assert 500 < bz['geo_period_cycle_yr'] < 1000  # measured: 703 yr

    assert row(summary, 'particle', '1J-3r0')['status'] == Status.NON_RESONANT


@pytest.mark.slow
def test_neptune_2011_kt19_polar():
    """2011 KT19 (i = 110 deg) in 7/-9 with Neptune, k = 6, about 180 deg with a period of ~40 kyr
    (Morais & Namouni 2017). SABA(10,6,4) with dt = 1.0 (REBOUND units, 2 pi = 1 yr: a step of
    ~58 d) follows ias15 to better than 0.01 deg over 20 kyr."""
    sim = resonances.check(
        '2011 KT19',
        '7N-9r6',
        date=DATE,
        source='nasa',
        integrator='SABA(10,6,4)',
        dt=1.0,
        integration_years=200000,
        Nout=4000,
        save=None,
        plot=None,
        save_summary=False,
    )
    sim.run()
    summary = sim.data_manager.get_simulation_summary(sim.bodies)

    kt19 = row(summary, '2011 KT19', '7N-9r6')
    assert kt19['status'] == Status.RESONANT
    episodes = accepted_episodes(sim.bodies[0], '7N-9r6')
    assert episodes
    assert all(abs(e['centre_mod_2pi_deg'] - 180) < 15 for e in episodes)  # measured: 179.8 deg
    assert 25000 < kt19['geo_period_cycle_yr'] < 60000  # measured: 40.9 kyr
    angle = np.degrees(sim.bodies[0].angles['7N-9r6'])
    assert angle.min() > 30 and angle.max() < 330  # measured: 61..305 deg


@pytest.mark.slow
def test_the_finder_never_mixes_orbit_directions():
    """A prograde Trojan gets no retrograde resonance, a retrograde co-orbital no prograde one,
    though both sit at 5.1-5.3 au next to Jupiter's 1:1."""
    sim = resonances.find([624, '2015 BZ509'], planets=['Jupiter'], type='mmr', date=DATE, source='nasa')
    mmrs = {b.name: b.mmrs for b in sim.bodies}

    assert mmrs['624'] and not any(m.retrograde for m in mmrs['624'])
    assert '1J-1' in {m.to_short() for m in mmrs['624']}
    assert mmrs['2015 BZ509'] and all(isinstance(m, RetrogradeTwoBody) for m in mmrs['2015 BZ509'])
    assert {'1J-1r0', '1J-1r1'} <= {m.to_s() for m in mmrs['2015 BZ509']}
