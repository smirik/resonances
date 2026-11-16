"""The code of docs/quick-start.md (run by docs/examples/run_all.py with short spans)."""

# --8<-- [start:mmr]
import resonances

# (463) Lola in the three-body resonance 4J-2S-1 (4 lambda_J - 2 lambda_S - lambda + ...)
sim = resonances.check(463, '4J-2S-1', integration_years=2_000)  # a real verdict: integration_years=100_000, output_step=1.0
sim.run()

body = sim.bodies[0]
for key, verdict in body.librations.items():  # key: the full angle, 4J-2S-1+0+0-1
    print(key, verdict.status, verdict.angle_status, verdict.confidence, verdict.reason, verdict.action)
print(sim.config.save_path)  # summary.csv, the series and the figures are written here
# --8<-- [end:mmr]

# --8<-- [start:table]
summary = sim.data_manager.get_simulation_summary(sim.bodies)
print(summary[['name', 'resonance', 'status', 'angle_status', 'confidence', 'reason', 'action']])
# --8<-- [end:table]

# --8<-- [start:secular]
# (1222) Tina in the linear secular resonance nu6 (angle varpi - varpi of Saturn's mode g6)
sim = resonances.check(1222, 'g-g6', integration_years=5_000)  # 2-10 Myr for a real verdict
sim.run()
print(sim.bodies[0].statuses)  # {'g-g6': 'uncertain'} on this short run
# --8<-- [end:secular]

# --8<-- [start:zlk]
# (3040) Kozai: the Lidov-Kozai angle is the argument of perihelion omega
sim = resonances.check(3040, resonances.LidovKozaiResonance(), integration_years=5_000)  # 500 kyr for a real verdict
sim.run()
print(sim.bodies[0].statuses)  # {'LK': ...}
# --8<-- [end:zlk]
