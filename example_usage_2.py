#!/usr/bin/env python3
"""Example 2: Check multiple asteroids for secular resonance"""

import resonances

# Check multiple asteroids for nu6 secular resonance
sim = resonances.secular_finder.check(
    asteroids=[
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
    ],
    resonance='nu6',
    integration_years=200000,
    save='all',
    plot='all',  # Reduced for testing
)

sim.run()
summary = sim.data_manager.get_simulation_summary(sim.bodies)
for _, row in summary.iterrows():
    print(f"Asteroid {row['name']} in nu6: status = {row['status']}")
