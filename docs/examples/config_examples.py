"""The code of docs/config.md (run by docs/examples/run_all.py; nothing is integrated).

run_all.py points MPC_CATALOG at the 20-row sample tests/fixtures/mpc/MPCORB_sample.DAT; with the defaults the same
code reads the whole MPCORB catalogue.
"""

# --8<-- [start:priority]
import resonances

sim = resonances.Simulation(name='test', integrator='whfast', save='all', save_path='output/my_run')
sim.config.plot = 'nonzero'  # after creation: through sim.config
resonances.config.set('INTEGRATION_DT', '0.5')  # a .env.dist key, as a string; read when a Simulation is created
print(resonances.config.has('INTEGRATION_DT'), resonances.config.get('INTEGRATION_DT'))
# --8<-- [end:priority]
resonances.config.set('INTEGRATION_DT', '1.0')

# --8<-- [start:rebound]
sim = resonances.Simulation()
sim.create_solar_system()
sim.integration_engine.sim.ri_ias15.epsilon = 1e-10
# --8<-- [end:rebound]

# --8<-- [start:mpcorb]
from resonances import mpcorb  # noqa: E402

mpcorb.search(463)  # the elements of (463) Lola (radians, epoch MJD TT); None if absent
mpcorb.search([1, '2001 ND13'])  # {identifier: elements}
mpcorb.search_by_axis(2.4, sigma=0.01)  # DataFrame of the rows with 2.39 <= a <= 2.41 AU
mpcorb.query(a=(0, 1))  # orbits inside 1 AU
mpcorb.query(q=(None, 1.3), H=(None, 18), U=(0, 3), numbered=True)
mpcorb.query(orbit_type=['apollo', 'aten'], pha=True, epoch='standard')
mpcorb.query(node=(350, 10))  # angle ranges may wrap through 0
# --8<-- [end:mpcorb]

# --8<-- [start:mpc_find]
sim = resonances.find(mpcorb.query(a=(2.38, 2.42), numbered=True, U=(0, 2)), planets=['Jupiter', 'Saturn'], type='mmr', source='mpc')
# --8<-- [end:mpc_find]
print([body.name for body in sim.bodies])
