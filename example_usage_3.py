#!/usr/bin/env python3
"""Example 3: Check multiple asteroids for secular resonance"""

import resonances
import resonances.finder.mmr_finder

v6_asteroids = [
    337335,  # start Huaman2018
    143199,
    295883,
    73415,
    322878,
    371246,
    203236,
    397488,
    248158,
    401320,
    354463,
    364329,
    314787,
    257972,
    179554,
    88064,
    246505,
    320989,
    282444,
    220835,
    121759,
    56932,
    313169,
    344837,
    267506,  # end Huaman2018
]

# ν16 resonance asteroid numbers
nu16_asteroids = [2335, 4177]

# z1 resonance asteroid numbers
z1_asteroids = [633, 847, 3395, 1020, 1228, 363]

# z2 resonance asteroid numbers
z2_asteroids = [163, 8089]

# g+s-g6 resonance asteroid numbers
g_s_g6_asteroids = [221, 339]

# g+s-g5-s7 resonance asteroid numbers
g_s_g5_s7_asteroids = [1075, 320]

g_2g6_g5 = [2198, 5507, 6573, 11791, 41895, 17200, 14611, 47883]

sim = resonances.Simulation(
    integration_years=1000000,
    save='all',
    plot='all',
)

# Create the solar system before adding bodies
sim.create_solar_system()

for asteroid in v6_asteroids:
    sim.add_body(asteroid, 'nu6', name=f'{asteroid}')
for asteroid in nu16_asteroids:
    sim.add_body(asteroid, 'nu16', name=f'{asteroid}')
for asteroid in z1_asteroids:
    sim.add_body(asteroid, 'g-g6+s-s6', name=f'{asteroid}')
for asteroid in z2_asteroids:
    sim.add_body(asteroid, '2g-2g6+s-s6', name=f'{asteroid}')
for asteroid in g_s_g6_asteroids:
    sim.add_body(asteroid, 'g+s-g6', name=f'{asteroid}')
for asteroid in g_s_g5_s7_asteroids:
    sim.add_body(asteroid, 'g+s-g5-s7', name=f'{asteroid}')
for asteroid in g_2g6_g5:
    sim.add_body(asteroid, 'g-2g6+g5', name=f'{asteroid}')

sim.run(progress=True)
summary = sim.data_manager.get_simulation_summary(sim.bodies)
for _, row in summary.iterrows():
    print(f"Asteroid {row['name']}: status = {row['status']}")
