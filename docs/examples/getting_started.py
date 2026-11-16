"""The code of docs/index.md and docs/install.md (run by docs/examples/run_all.py with a short span)."""

# --8<-- [start:index]
import resonances

sim = resonances.check(463, '4J-2S-1', integration_years=2_000)  # (463) Lola in 4J-2S-1; 100_000 years for a verdict
sim.run()
# --8<-- [end:index]

# --8<-- [start:install]
import resonances  # noqa: E402,F811

sim = resonances.check(463, '4J-2S-1', integration_years=2_000)  # (463) Lola in 4J-2S-1; 100_000 years for a verdict
sim.run()
print(sim.bodies[0].statuses)
print(sim.config.save_path)
# --8<-- [end:install]
