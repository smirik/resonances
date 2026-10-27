"""The code of docs/free-elements.md (run by docs/examples/run_all.py with a short span)."""

import resonances

# (3040) Kozai in the Lidov-Kozai resonance; 500 kyr or more for a real verdict
sim = resonances.check(3040, resonances.LidovKozaiResonance(), integration_years=50_000)
sim.run()
body, config = sim.bodies[0], sim.config
libration_period_yr = 200_000.0  # the period of the bounded omega, in years

# --8<-- [start:gate]
from resonances.resonance.omega_free_gate import evaluate_gate  # noqa: E402

elements = body.build_free_elements()
gate = evaluate_gate(elements, libration_period_yr, config)  # an outcome: confirmed, kinematic, or no verdict
print(gate.outcome, elements.b_omega)
# --8<-- [end:gate]

times_years = sim.times / (2 * 3.141592653589793)
ecc, inc, Omega, omega = body.ecc, body.inc, body.Omega, body.omega

# --8<-- [start:arrays]
from resonances.secular.free_elements import free_elements  # noqa: E402

elements = free_elements(times_years, ecc, inc, Omega, omega)
# --8<-- [end:arrays]
print(elements.b_varpi, elements.b_Omega, elements.b_omega)
