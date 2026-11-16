"""The code of docs/matrix.md (run by docs/examples/run_all.py with a short span).

run_all.py points MPC_CATALOG at the 20-row sample tests/fixtures/mpc/MPCORB_sample.DAT; with the defaults the
candidates come from the whole MPCORB catalogue.
"""

# --8<-- [start:retrograde]
import resonances

mmr = resonances.create_mmr('1J-3r0')  # or RetrogradeTwoBody(p=1, q=3, k=0, planet='Jupiter')
# 2_000 years here; 10_000 and more for a verdict
sim = resonances.check('2000 DG8', '1J-3r0', integrator='ias15', integration_years=2_000, Nout=2_001)
sim.run()
# --8<-- [end:retrograde]
print(mmr.to_s(), sim.bodies[0].statuses)

# --8<-- [start:three_body]
from resonances.mmr.three_body_matrix import ThreeBodyMatrix  # noqa: E402

mmrs = ThreeBodyMatrix.find_resonances(2.39, planets=['Jupiter', 'Saturn', 'Uranus'])
for mmr in mmrs:
    print(mmr.to_short())
# --8<-- [end:three_body]

# --8<-- [start:two_body]
from resonances.mmr.two_body_matrix import TwoBodyMatrix  # noqa: E402

mmrs = TwoBodyMatrix.find_resonances(5.2, planets=['Jupiter'])
for mmr in mmrs:
    print(mmr.to_short())
# --8<-- [end:two_body]

# --8<-- [start:two_body_sigma]
mmrs = TwoBodyMatrix.find_resonances(5.2, planets=['Jupiter'], sigma=1.0)
for mmr in mmrs:
    print(mmr.to_short())
# --8<-- [end:two_body_sigma]

# --8<-- [start:candidates]
from resonances import mpcorb  # noqa: E402

df = mpcorb.search_by_axis(resonances.create_mmr('4J-2S-1').resonant_axis)  # or astdys.search_by_axis (AstDyS)
print(df['number'].tolist())
# --8<-- [end:candidates]
