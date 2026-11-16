"""The code of docs/examples.md (run by docs/examples/run_all.py with short spans).

run_all.py points MPC_CATALOG at the 20-row sample tests/fixtures/mpc/MPCORB_sample.DAT, so the MPC recipe runs
without downloading MPCORB; with the defaults the same code reads the whole catalogue.
"""

# --8<-- [start:find_mmrs]
import resonances

# Candidate MMRs near a = 2.3986 au (463 Lola) with Jupiter and Saturn
for mmr in resonances.find_mmrs(2.3986, planets=['Jupiter', 'Saturn']):
    print(mmr.to_short(), round(mmr.resonant_axis, 4))
# --8<-- [end:find_mmrs]

# --8<-- [start:find]
# find: every candidate MMR of each body (type='mmr'; None also adds the secular and Lidov-Kozai candidates)
sim = resonances.find([463, 490], planets=['Jupiter', 'Saturn'], type='mmr', name='find-demo', save='all', integration_years=2_000)
sim.run()
summary = sim.data_manager.get_simulation_summary(sim.bodies)
print(summary[['name', 'resonance', 'status', 'angle_status', 'reason']])
# --8<-- [end:find]

# --8<-- [start:restore]
json_file = f'{sim.config.save_path}/simulation.json'
restored = resonances.SimulationSerializer.restore(json_file)  # the verdicts as they are in summary.csv
print(restored.bodies[0].statuses)

# analyse the saved series again (with the settings of simulation.json; current_settings=True: those of today's .env)
again = resonances.SimulationSerializer.restore(json_file, recompute_librations=True)
print(again.bodies[0].statuses)
# --8<-- [end:restore]

# --8<-- [start:mpc]
# a selection of the MPC catalogue: a DataFrame (a in au, angles in degrees, H, U, MPC's orbit-type flags)
selection = resonances.mpcorb.query(a=(2.3, 2.45), U=(None, 2), numbered=True)
print(selection[['number', 'name', 'a', 'e', 'inc_deg', 'H']])

# elements from MPCORB at its standard epoch (the default date of a source='mpc' run)
sim = resonances.find(selection, source='mpc', planets=['Jupiter', 'Saturn'], type='mmr', name='mpc-demo', integration_years=2_000)
print(sim.config.date, sim.config.mpc_snapshot['sha256'][:12], [body.name for body in sim.bodies])
# --8<-- [end:mpc]

# --8<-- [start:layout]
sim = resonances.Simulation(
    name='layout-demo',
    date='2026-01-01',
    integration_years=2_000,
    plot='all',
    plots=['combined', 'evolution'],
    plot_layout='type/status',  # figures/<type>/<status>/...
    plot_combined_layout='confidence/status',  # combined/<confidence>/<status>/... (the default)
)
sim.create_solar_system()
sim.add_body(463, ['4J-2S-1', 'g-g6'], name='463')
sim.run()
print(sim.config.plot_path)
# --8<-- [end:layout]

# --8<-- [start:integrator]
# IAS15 (adaptive) with the Moon as a separate body: for planet-crossing orbits and close encounters
sim = resonances.Simulation(
    name='ias15-demo',
    date='2026-01-01',
    integration_years=500,
    integrator='ias15',
    dt=0.01,  # initial step, REBOUND units (yr / 2 pi); IAS15 adapts it
    solar_system_moon=True,
    output_step=0.2,  # years; a 1:1 or a near-Earth orbit needs a fine output step
)
sim.create_solar_system()
sim.add_body('2002 AA29', '1E-1', name='2002 AA29')  # a co-orbital of the Earth
sim.run()
print(sim.bodies[0].statuses)
# --8<-- [end:integrator]
