"""Real-integration checks of the cross-spectral criteria.

Two reference objects, chosen because they disagree:

- 162474: omega librates AND e/i exchange in antiphase at the ~52 kyr ZLK cycle.
  A genuine Lidov-Kozai resonance; the gate must confirm it.
- 463 Lola in the 4J-2S-1 mean-motion resonance: sigma and a must share the
  libration frequency roughly in quadrature (a lags the angle by a quarter period),
  and the MMR status must be left alone by the coherence machinery.

162474 (a = 1.24 au, e = 0.36, i = 51 deg) crosses the orbits of Venus, Earth and
Mars. dt = 0.5 gives 17 SABA steps per orbit and reproduces dt = 0.1 (same 52 kyr
cycle, same a range) at a third of the cost; dt = 1.0 gives 8.7 steps per orbit and
ejects the body. 500 kyr is the shortest baseline that puts the 52 kyr line inside
P_max = T/(5*n_seg).
"""

import numpy as np
import pytest

import resonances
from resonances.resonance.classify.models import ResonanceStatus
from resonances.resonance.cross_spectrum import phase_distance


@pytest.mark.slow
def test_zlk_coherence_confirms_162474():
    years, cadence = 500_000, 2.0
    sim = resonances.check(
        162474,
        "lidov-kozai",
        name="test_zlk_162474",
        source="astdys",
        integrator="SABA(10,6,4)",
        dt=0.5,
        integration_years=years,
        Nout=int(years / cadence) + 1,
        save=None,
        plot=None,
        save_summary=False,
    )
    sim.run(progress=False)

    body = sim.bodies[0]
    key = resonances.LidovKozaiResonance().to_s()

    assert body.statuses[key] == ResonanceStatus.LIBRATION

    gate = body.zlk_gates[key]
    assert gate.check == resonances.CoherenceCheck.PASSED
    assert gate.coherent is True
    assert phase_distance(gate.line_phase, 180.0) < 30.0
    assert gate.line_gamma2 > 0.5
    # The exchange condition must clear its threshold with room to spare, not scrape by:
    # 591986, which is not in resonance, never gets past -0.4 on any window.
    assert gate.line_r_band < -0.6

    # Independent confirmation that e and i really do trade against each other.
    assert np.corrcoef(body.ecc, body.inc)[0, 1] < -0.5

    summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)
    row = summary.iloc[0]
    assert row['zlk_check'] == 'passed'
    assert 0.0 < row['e_free'] < 1.0
    assert 0.0 < row['e_forced'] < 1.0


@pytest.mark.slow
def test_mmr_sigma_axis_coherence_for_463():
    # 463 librates in 4J-2S-1 with a period near 10 kyr, so the baseline has to reach
    # P_max = T/(5*n_seg) = T/20 above that: 40 kyr is not enough, 250 kyr is.
    sim = resonances.find(
        463,
        ['Jupiter', 'Saturn'],
        name="test_coherence_463",
        integration_years=250_000,
        type='mmr',
        save=None,
        plot=None,
        save_summary=False,
        coherence_period_min=2000.0,
    )
    sim.run(progress=False)

    body = next(b for b in sim.bodies if b.name == '463')
    resonance = next(r for r in body.resonances() if r.to_s() == '4J-2S-1+0+0-1')
    key = resonance.to_s()

    assert body.statuses[key] == ResonanceStatus.LIBRATION

    analysis = body.coherence[key]['sigma-a']
    assert analysis.top_line is not None, "expected a coherent sigma-a line for a librating MMR"
    # The libration period from the classifier and from the cross spectrum must agree.
    libration_period = body.librations[key].metrics.libration_period_1
    assert analysis.top_line.period == pytest.approx(libration_period, rel=0.25)
    assert phase_distance(analysis.top_line.phase_deg, 90.0) < 45.0

    # Coherence is diagnostic only for MMR: no gate, no status change.
    assert key not in body.zlk_gates
    summary, _ = sim.data_manager.get_simulation_summary(sim.bodies)
    row = summary.loc[(summary['name'] == '463') & (summary['resonance'] == key)].iloc[0]
    assert row['status'] == ResonanceStatus.LIBRATION
    assert row['zlk_check'] is None
