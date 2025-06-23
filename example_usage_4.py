#!/usr/bin/env python3
"""Example 4: Complete resonance investigation (MMR + secular)"""

import resonances

# Check asteroid 463 for MMRs
mmr_sim = resonances.check(asteroids=463, resonance='4J-2S-1', integration_years=50000, save='none', plot='none')  # Reduced for testing
mmr_sim.run()

# Check same asteroid for secular resonances
secular_sim = resonances.secular_finder.find(
    asteroids=463, order=2, integration_years=100000, save='none', plot='none'  # Linear secular resonances only  # Reduced for testing
)
secular_sim.run()

# Analyze results
mmr_summary = mmr_sim.data_manager.get_simulation_summary(mmr_sim.bodies)
secular_summary = secular_sim.data_manager.get_simulation_summary(secular_sim.bodies)

print(f"Asteroid 463 resonance analysis:")
print(f"  MMR 4J-2S-1: status = {mmr_summary.iloc[0]['status']}")

secular_resonances = secular_summary[abs(secular_summary['status']) >= 1]
print(f"  Secular resonances found: {len(secular_resonances)}")
for _, row in secular_resonances.iterrows():
    print(f"    {row['resonance']}: status = {row['status']}")
