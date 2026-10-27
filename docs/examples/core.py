"""The code of docs/core.md (run by docs/examples/run_all.py with short spans)."""

# --8<-- [start:simulation]
import numpy as np

import resonances

sim = resonances.Simulation(
    name='core-demo',  # the output goes to SAVE_PATH/core-demo (cache/core-demo by default)
    date='2026-01-01',  # start of the integration; the planets and the elements are taken at this date
    integration_years=2_000,  # 100_000 years for a real verdict of a main-belt MMR
    integrator='SABA(10,6,4)',  # the default; 'whfast' or 'ias15' for close encounters
    dt=1.0,  # REBOUND time units: 1.0 = 1/(2 pi) yr = 58.1 days
    output_step=1.0,  # years between saved points (default: about 100 REBOUND units = 15.9 yr)
    save='all',  # which (body, resonance) rows to save: none, all, resonant, candidates, nonzero, ...
    plot='resonant',  # which rows to plot, by the same status filters
    plots=['combined'],  # the one-page verdict figure
)
sim.create_solar_system()
# --8<-- [end:simulation]

# --8<-- [start:bodies]
sim.add_body(463, '4J-2S-1', name='463')  # by number, one resonance
sim.add_body('2001 ND13', ['1V-3', 'g-g6'], name='2001 ND13')  # by designation, several resonances
elements = {  # by elements: heliocentric J2000 ecliptic, angles in radians, epoch in MJD (optional)
    'a': 2.395,
    'e': 0.22,
    'inc': np.radians(13.5),
    'Omega': np.radians(97.0),
    'omega': np.radians(180.0),
    'M': np.radians(10.0),
}
sim.add_body(elements, resonances.TwoBody('3J-1'), name='synthetic')
# --8<-- [end:bodies]

# --8<-- [start:objects]
mmr2 = resonances.create_mmr('3J-1')  # two-body: 3 lambda_J - lambda - 2 varpi
mmr3 = resonances.create_mmr('4J-2S-1')  # three-body: 4 lambda_J - 2 lambda_S - lambda + 0 + 0 - varpi
retro = resonances.RetrogradeTwoBody('1J-3r0')  # retrograde p/-q with k (orbits with i > 90 deg)
nu6 = resonances.SecularResonance('g-g6')  # linear secular resonance
z1 = resonances.create_resonance('g-g6+s-s6')  # the factory reads any notation
lk = resonances.LidovKozaiResonance()  # the argument of perihelion omega
print(mmr2.to_s(), mmr3.to_s(), retro.to_s(), nu6.to_s(), z1.to_s(), lk.to_s())
print(round(mmr3.resonant_axis, 4))  # nominal catalogue axis a_res [au]; the analysed a* is ctx_a_star_au
# --8<-- [end:objects]

# --8<-- [start:run]
report = sim.dry_run()  # optional: checks the output step and the integrator before integrating
sim.run()
# --8<-- [end:run]

# --8<-- [start:verdicts]
for body in sim.bodies:
    for key, verdict in body.librations.items():
        print(body.name, key, verdict.status, verdict.angle_status, verdict.reason, verdict.confidence, verdict.action)
    print(body.statuses)  # {key: status string}

body = sim.bodies[0]
t = sim.times / (2 * np.pi)  # years since the start: REBOUND time is years * 2 pi
angle = body.angle(mmr3)  # the resonant angle, radians in [0, 2 pi)
print(len(t), body.axis[:3], angle[:3])
# --8<-- [end:verdicts]

# --8<-- [start:summary]
summary = sim.data_manager.get_simulation_summary(sim.bodies)
print(summary[['name', 'resonance', 'status', 'angle_status', 'confidence', 'reason']])
# --8<-- [end:summary]
