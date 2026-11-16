"""The code of docs/secular.md (run by docs/examples/run_all.py with a short span)."""

# --8<-- [start:create]
import resonances

nu6 = resonances.create_secular_resonance('nu6')  # g-g6
nu14 = resonances.create_secular_resonance('nu14')  # s-s4
z1 = resonances.create_secular_resonance('g-g6+s-s6')
z2 = resonances.create_secular_resonance('2(g-g6)+(s-s6)')
other = resonances.create_secular_resonance('-g+s+g5-s7')
# --8<-- [end:create]

# --8<-- [start:arcadia]
sim = resonances.Simulation(
    date='2026-09-29',
    integrator='SABA(10,6,4)',
    dt=0.5,
    integration_years=20_000,  # the benchmark run: integration_years=10_000_000, Nout=100001
    Nout=201,
)
sim.create_solar_system()
sim.add_body(1020, [resonances.create_secular_resonance('z1')], name='Arcadia')
sim.run()
print(sim.bodies[0].statuses)  # {'g-g6+s-s6': 'resonant'} over 10 Myr in the benchmark; 20 kyr is too short
# --8<-- [end:arcadia]
