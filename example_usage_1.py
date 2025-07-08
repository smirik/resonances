#!/usr/bin/env python3
"""Example 1: Check single asteroid for secular resonance"""

import resonances

asteroid_num = 314787
resonance = 'nu6'
sim = resonances.secular_finder.check(
    asteroids=asteroid_num, resonance=resonance, integration_years=200000, save='all', plot='all'  # Reduced for testing
)

sim.run()
summary = sim.data_manager.get_simulation_summary(sim.bodies)
print(f"Asteroid {asteroid_num} in {resonance}: status = {summary.iloc[0]['status']}")
