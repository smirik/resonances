"""The code of docs/lidov-kozai.md (run by docs/examples/run_all.py with a short span)."""

# --8<-- [start:integrals]
import numpy as np

import resonances
from resonances import LidovKozaiParameters

# osculating ecliptic e, i, Omega, omega of three main-belt asteroids (the benchmark's epoch)
samples = {
    '1373 Cincinnati': (0.3152, 38.953, -62.561, 98.882),
    '3040 Kozai': (0.2003, 46.647, 143.473, 290.213),
    '15527': (0.1529, 21.259, 173.771, 168.872),
}
for name, (e, inc_deg, node_deg, omega_deg) in samples.items():
    # i and omega read from the invariable plane (an exact rotation), as the summary columns c1, c2, h_quad
    c1, c2, h_quad = LidovKozaiParameters.evaluate_invariable(e, *np.radians([inc_deg, node_deg, omega_deg]))
    print(f'{name:16s} c1 = {c1:.3f}  c2 = {c2:+.4f}  h_quad = {h_quad:.3f}')
# --8<-- [end:integrals]

# --8<-- [start:run]
sim = resonances.Simulation(
    name='lk-demo',
    date='2026-01-01',
    integration_years=5_000,  # 500_000 years for a real verdict: the omega cycles last 10-100 kyr
    integrator='SABA(10,6,4)',
    dt=0.5,  # REBOUND units: 0.5 / (2 pi) yr = 29 days
    output_step=10.0,  # years
    plot_always=['lidov_kozai'],  # draw the omega figure whatever the status
)
sim.create_solar_system()
for number in (1373, 3040, 15527):
    sim.add_body(number, resonances.LidovKozaiResonance(), name=str(number))
sim.run()

for body in sim.bodies:
    verdict = body.verdict('LK')
    print(body.name, verdict.status, verdict.angle_status, verdict.reason, verdict.libration_origin)
# --8<-- [end:run]

# --8<-- [start:angle]
body = sim.bodies[1]  # 3040 Kozai
t = sim.times / (2 * np.pi)  # years
omega = body.angle(resonances.LidovKozaiResonance())  # radians in [0, 2 pi)
print(f'{len(t)} points, omega from {np.degrees(omega.min()):.0f} to {np.degrees(omega.max()):.0f} deg')
# --8<-- [end:angle]
