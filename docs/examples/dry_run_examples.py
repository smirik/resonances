"""The code of docs/dry-run.md (run by docs/examples/run_all.py; only the pilot integrates, a few seconds)."""

# --8<-- [start:dry_run]
import resonances

sim = resonances.Simulation(name='dry-run-demo', integration_years=100_000, output_step='auto')
sim.create_solar_system()
sim.add_body(463, '4J-2S-1', name='463')  # or resonances.find(...) / resonances.check(...)
report = sim.dry_run(level='static')  # elements only: milliseconds per body
report = sim.dry_run(level='pilot')  # a short fine integration per body: 1-3 s each
report.summary()  # tables and warnings
rows, bodies, run = report.rows, report.bodies, report.run  # DataFrame body x MMR, DataFrame of bodies, dict of the run
report.apply()  # set output_step/Nout and the recommended integrator/dt on the config
report.to_csv('out')  # dryrun_rows.csv, dryrun_bodies.csv, dryrun_run.json
# --8<-- [end:dry_run]
