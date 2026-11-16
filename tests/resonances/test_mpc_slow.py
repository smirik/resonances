"""DATA_SOURCE=mpc against the real files: MPCORB.DAT.gz (94 MB, downloaded into MPC_CACHE_DIR on first use) and
NEAm00.txt, with the planets from Horizons. Network and date dependent (the MPC moves its standard epoch every
200 days, AstDyS on its own schedule): a comparison that needs both at one epoch is skipped when they are not."""

import math

import astdys
import numpy as np
import pytest

from resonances import mpcorb
from resonances.data.util import mjd_from_datetime
from resonances.simulation.simulation import Simulation


# The comparison is meant for spans like the 138 days it was made on (MPC_PROPAGATE_MAX_DAYS no longer bounds it)
PROPAGATION_SPAN_DAYS = 400.0


def _angle_difference(x, y):
    return (x - y + math.pi) % (2 * math.pi) - math.pi


@pytest.mark.slow
def test_mpc_and_astdys_lola_integrate_alike():
    """463 Lola from MPCORB and from AstDyS allnum.cat at the same epoch: the elements agree within the catalogues'
    precision (different fits: |da| ~ 1e-7 AU, angles ~ 5e-4 deg), and so do a and the angle of 4J-2S-1 over 2 kyr."""
    snapshot = mpcorb.snapshot('mpcorb')
    astdys.set_type('osculating')
    astdys_elem = astdys.search(463)
    if float(astdys_elem['epoch']) != snapshot['standard_epoch_mjd']:
        pytest.skip(f"AstDyS epoch MJD {astdys_elem['epoch']} != MPC standard epoch MJD {snapshot['standard_epoch_mjd']}")

    sim = Simulation(source='mpc', mpc_catalog='mpcorb', save=None, plot=None, save_summary=False)
    assert mjd_from_datetime(sim.config.date) == snapshot['standard_epoch_mjd']
    sim.config.tmax = 2000 * 2 * np.pi
    sim.config.Nout = 201
    sim.create_solar_system()
    sim.add_body(463, '4J-2S-1', name='mpc')
    sim.add_body(dict(astdys_elem), '4J-2S-1', name='astdys')
    from_mpc, from_astdys = sim.bodies
    for key, tolerance in (('a', 2e-7), ('e', 3e-7), ('inc', math.radians(1e-4))):
        assert from_mpc.initial_data[key] == pytest.approx(astdys_elem[key], abs=tolerance)
    for key in ('Omega', 'omega', 'M'):
        assert abs(_angle_difference(from_mpc.initial_data[key], astdys_elem[key])) < math.radians(1e-3)

    sim.body_manager.add_bodies_to_simulation(sim.integration_engine.sim)
    times = np.linspace(0.0, sim.config.tmax, sim.config.Nout)
    sim.integration_engine.run_integration(sim.bodies, times)
    # on 2026-10-26: max |da| 5.8e-7 AU, max |d sigma| 0.012 deg (initial |da| 7.8e-8 AU, |d lambda| 3e-5 deg)
    assert np.max(np.abs(from_mpc.axis - from_astdys.axis)) < 2e-6
    angle_gap = _angle_difference(
        np.asarray(from_mpc.angles_unwrapped['4J-2S-1+0+0-1']), np.asarray(from_astdys.angles_unwrapped['4J-2S-1+0+0-1'])
    )
    assert np.max(np.abs(angle_gap)) < math.radians(0.05)


@pytest.mark.slow
def test_propagate_policy_reproduces_mpc_propagation_of_eros():
    """433 Eros from MPCORB (standard epoch) integrated by the package (MPC_EPOCH_POLICY=propagate) to the epoch of
    NEAm00.txt reproduces the MPC's own propagation there: on 2026-10-25 over 138 days the differences were 5e-8 AU and
    4e-6 deg, the file's rounding (docs/config.md, "Minor Planet Center")."""
    snapshot = mpcorb.snapshot('mpcorb')
    today = mpcorb.lookup([433], catalog='neam00')['433']
    span = abs(float(today['epoch_mjd']) - snapshot['standard_epoch_mjd'])
    if span > PROPAGATION_SPAN_DAYS:
        pytest.skip(f"NEAm00 is {span:.0f} days from the cached MPCORB epoch: run `resonances mpc-update`")

    sim = Simulation(
        source='mpc',
        mpc_catalog='mpcorb',
        mpc_epoch_policy='propagate',
        date=mpcorb.datetime_from_mjd(today['epoch_mjd']),
        save=None,
        plot=None,
    )
    sim.add_body(433, '4J-2S-1', name='433')
    elem = sim.bodies[0].initial_data
    assert (elem['epoch'], elem['propagated_from_epoch']) == (float(today['epoch_mjd']), snapshot['standard_epoch_mjd'])
    reference = mpcorb.elements_from_row(today)
    assert elem['a'] == pytest.approx(reference['a'], abs=5e-7)
    assert elem['e'] == pytest.approx(reference['e'], abs=5e-7)
    assert abs(_angle_difference(elem['inc'], reference['inc'])) < math.radians(1e-4)
    varpi = lambda x: x['Omega'] + x['omega']  # noqa: E731
    longitude = lambda x: x['Omega'] + x['omega'] + x['M']  # noqa: E731
    assert abs(_angle_difference(varpi(elem), varpi(reference))) < math.radians(1e-4)
    assert abs(_angle_difference(longitude(elem), longitude(reference))) < math.radians(1e-4)
