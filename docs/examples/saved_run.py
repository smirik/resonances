"""The code of docs/output.md, docs/plots.md and docs/serialization.md (run by docs/examples/run_all.py).

A short run is saved into output/my_run first; the snippets read it, as they would read a real run.
"""

import resonances

# --8<-- [start:plots_config]
sim = resonances.Simulation(
    plot='nonzero',
    plots=['evolution', 'combined'],
    plot_options={'style': 'screen'},  # optional, see below
)
# --8<-- [end:plots_config]

# The run the snippets below read: (463) Lola in an MMR and in a secular resonance, 2 kyr, everything saved
sim = resonances.Simulation(name='my_run', date='2026-01-01', integration_years=2_000, save='all', save_path='output/my_run')
sim.create_solar_system()
sim.add_body(463, ['4J-2S-1', 'g-g6'], name='463')
sim.run()

# --8<-- [start:summary]
import pandas as pd  # noqa: E402

summary = pd.read_csv('output/my_run/summary.csv')
print(summary.groupby(['status', 'angle_status', 'confidence'], dropna=False).size())  # uncertain/invalid: no level
in_resonance = summary[summary['status'].isin(['resonant', 'transient'])]
# --8<-- [end:summary]

# --8<-- [start:redraw]
from resonances import SimulationSerializer, MMRPlotter  # noqa: E402
from resonances.plotting import SecularPlotter  # noqa: E402

sim = SimulationSerializer.restore('output/my_run/simulation.json')
body = sim.bodies[0]
plotter = MMRPlotter.from_body(body, body.mmrs[0], sim, options={'style': 'paper'})
plotter.plot('recurrence')  # recurrence, fair, portrait or cycles; the combined figure: plot_body below
plotter.save('recurrence.pdf').close()

secular = SecularPlotter.from_body(body, body.secular_resonances[0], sim)  # secular or ZLK rows: pass `sim`
secular.plot('portrait')
secular.save('portrait.png').close()

sim.config.plot, sim.config.plots = 'all', ['combined']
sim.data_manager.plot_body(body, sim)  # the verdict figures, from a re-run analysis
# --8<-- [end:redraw]

resonance = body.mmrs[0]
times = sim.times / (2 * 3.141592653589793)
ecc, omega = body.ecc, body.omega

# --8<-- [start:ecc_vector]
resonances.EccentricityVectorPlotter.from_body(body, resonance, sim).plot().save('kh.png')
resonances.EccentricityVectorPlotter.from_data(times, ecc, omega, body_name='463').plot().save('kh.png')
# --8<-- [end:ecc_vector]

# --8<-- [start:restore]
from resonances import SimulationSerializer  # noqa: E402,F811

sim = SimulationSerializer.restore("output/my_run/simulation.json")
# --8<-- [end:restore]
print(sim.bodies[0].statuses)

# --8<-- [start:recompute]
sim = SimulationSerializer.restore("output/my_run/simulation.json", recompute_librations=True)
# --8<-- [end:recompute]
print(sim.bodies[0].statuses)
