"""The code of docs/coherence.md (run by docs/examples/run_all.py with short spans)."""

# --8<-- [start:pairs]
import resonances

sim = resonances.check(
    463,
    '4J-2S-1',
    integration_years=2_000,  # 100_000 years for a real verdict
    coherence_pairs={'mmr': ['sigma-a', 'sigma-e', 'sigma-i']},
    coherence_phase_targets={'mmr': {'sigma-a': (90.0, 20.0)}},
)
# --8<-- [end:pairs]
sim.run()

# --8<-- [start:plots]
sim = resonances.check(
    162474,
    'lidov-kozai',
    integration_years=20_000,  # 500 kyr or more for the Lidov-Kozai periods
    plots=['evolution', 'ecc_vector', 'cross_spectrum'],
    plot='all',
)
sim.run()
# --8<-- [end:plots]

# --8<-- [start:plotters]
body = sim.bodies[0]
resonance = body.lidov_kozai_resonances[0]  # key 'LK'
resonances.EccentricityVectorPlotter.from_body(body, resonance, sim).plot().save('kh.png')
resonances.CrossSpectrumPlotter.from_body(body, resonance, 'e-i').plot().save('coh.png')
# --8<-- [end:plotters]

# --8<-- [start:primitives]
import numpy as np  # noqa: E402

times_yrs = np.asarray(sim.times) / (2 * np.pi)  # a uniform grid, in years
e, inc = body.ecc, body.inc  # a, e and i are not unwrapped; angles must be (np.unwrap)

result = resonances.cross_spectrum(times_yrs, e, inc, n_seg=4, alpha=0.01, period_min=10_000)
if result.valid and result.top_line:
    line = result.top_line
    print(line.period, line.gamma2, line.phase_deg, line.amp_ratio)
    print(resonances.phase_matches(line.phase_deg, 180.0, 30.0))
# --8<-- [end:primitives]
