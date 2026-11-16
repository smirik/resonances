"""Real-integration checks of the cross-spectral criteria.

Two reference objects, chosen because they disagree:

- 162474: omega librates AND e/i exchange in antiphase at the ~52 kyr ZLK cycle.
  A genuine Lidov-Kozai libration: about 270 deg to ~0.82 Myr, then about 90 deg (the twin island,
  centre +180 deg). Over the 1.5 Myr record that is a transient (author, 2026-10-17: "an ordinary
  transient"); the first 0.5 Myr alone is a libration about 270 deg that the free-omega gate confirms.
- 463 Lola in the 4J-2S-1 mean-motion resonance: sigma and a must share the
  libration frequency roughly in quadrature (a lags the angle by a quarter period),
  and the MMR status must be left alone by the coherence machinery.

162474 (a = 1.24 au, e = 0.36, i = 51 deg) crosses the orbits of Venus, Earth and
Mars. dt = 0.5 gives 17 SABA steps per orbit and reproduces dt = 0.1 (same 52 kyr
cycle, same a range) at a third of the cost; dt = 1.0 gives 8.7 steps per orbit and
ejects the body.

Baseline and date (2026-10-12). The e-i line must sit inside the validity band
P_max = T/(5*n_seg) = T/20, so a ZLK cycle of 52-75 kyr needs T >= 1.5 Myr. At 500 kyr
(P_max = 25 kyr) the test passed or failed with the epoch of the AstDyS catalogue: the
line was outside the band and was found by luck (2025-05-05: passed, r_band -0.66;
2025-11-21, 2026-06-09: no line, although r(e, i) = -0.92/-0.95). At 1.5 Myr all three
epochs pass with r_band -0.97 and phase within 3 deg of 180. The body is chaotic, so the
initial conditions are pinned: Horizons at 2026-06-09 (the epoch of the current AstDyS
catalogue). The e-i coherence needs the whole 1.5 Myr; the libration of the first 0.5 Myr is
judged on that stretch alone (its own free elements, filter and gate), not on whole-record arrays.
The same record is the benchmark example zlk__162474__LK__1500kyr (stored at 10 yr).
"""

import dataclasses

import numpy as np
import pytest

import resonances
from resonances.resonance.analysis.secular.decide import librating_episodes
from resonances.resonance.analysis.verdict import AngleStatus, Reason, Status
from resonances.resonance.analysis.zlk import analyse_zlk
from resonances.resonance.cross_spectrum import phase_distance
from resonances.resonance.omega_free_gate import GateOutcome
from resonances.secular.free_elements import free_elements
from resonances.simulation.simulation import long_period_series

PREFIX_YR = 500_000.0
ISLAND_DEG = 270.0
ISLAND_TOL_DEG = 36.0  # 0.2 pi: a centre within a fifth of a half-turn of the island's


def analyse_prefix(sim, body, resonance, t_end_yr):
    """The ZLK verdict of the record's first `t_end_yr` years alone: a fresh series, free elements fitted on that
    stretch, the filter and the gate of that stretch (nothing taken from the whole-record analysis)."""
    t_yr = np.asarray(body.times, dtype=float) / (2 * np.pi)
    keep = t_yr <= t_end_yr
    series = long_period_series(body, resonance)
    prefix = dataclasses.replace(
        series,
        t_yr=series.t_yr[keep],
        angle=series.angle[keep],
        a=series.a[keep],
        e=series.e[keep],
        inc=series.inc[keep],
        varpi=series.varpi[keep],
        Omega=series.Omega[keep],
    )
    elements = free_elements(
        times=t_yr[keep],
        ecc=np.asarray(body.ecc)[keep],
        inc=np.asarray(body.inc)[keep],
        Omega=np.asarray(body.Omega)[keep],
        omega=np.asarray(body.omega)[keep],
        sampling_years=sim.config.free_elements_sampling_years,
        mask_quantile=sim.config.free_gate_mask_quantile,
    )
    return analyse_zlk(prefix, sim.config.analysis_settings, free_elements=elements, gate_config=sim.config)


@pytest.mark.slow
def test_zlk_coherence_confirms_162474():
    years, cadence = 1_500_000, 2.0
    sim = resonances.check(
        162474,
        "lidov-kozai",
        name="test_zlk_162474",
        source="nasa",
        date="2026-06-09",  # pinned: a chaotic Earth-crosser drifts with the epoch
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
    resonance = resonances.LidovKozaiResonance()
    key = resonance.to_s()

    # the whole record: two librating episodes about different islands -> transient (author, 2026-10-17)
    verdict = body.librations[key]
    assert body.statuses[key] == Status.TRANSIENT
    assert verdict.angle_status == AngleStatus.EPISODIC
    assert verdict.reason == Reason.DYNAMIC_EPISODE

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

    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    row = summary.iloc[0]
    assert row['zlk_check'] == 'passed'
    assert 0.0 < row['e_free'] < 1.0
    assert 0.0 < row['e_forced'] < 1.0

    # the first 0.5 Myr alone: a libration about 270 deg, confirmed by the free-omega gate
    prefix = analyse_prefix(sim, body, resonance, PREFIX_YR)
    assert prefix.verdict.status == Status.RESONANT
    assert prefix.verdict.angle_status == AngleStatus.LIBRATION
    assert prefix.gate is not None and prefix.gate.outcome == str(GateOutcome.CONFIRMED)
    centres = [np.degrees(episode.centre_mod_2pi_rad) for episode in librating_episodes(prefix.geometry)]
    assert centres
    assert all(abs((centre - ISLAND_DEG + 180.0) % 360.0 - 180.0) < ISLAND_TOL_DEG for centre in centres)


@pytest.mark.slow
def test_mmr_sigma_axis_coherence_for_463():
    # 463 librates in 4J-2S-1 with a period near 10 kyr, so the baseline has to reach
    # P_max = T/(5*n_seg) = T/20 above that: 40 kyr is not enough, 250 kyr is.
    # check() of the one resonance (find ran every J-S candidate of 463: 97 of the 110 s); the date fixed as the
    # backward Lola test
    sim = resonances.check(
        463,
        '4J-2S-1',
        name="test_coherence_463",
        source='nasa',
        date='2026-09-29',
        integration_years=250_000,
        save=None,
        plot=None,
        save_summary=False,
        coherence_period_min=2000.0,
    )
    sim.run(progress=False)

    body = next(b for b in sim.bodies if b.name == '463')
    resonance = next(r for r in body.resonances() if r.to_s() == '4J-2S-1+0+0-1')
    key = resonance.to_s()

    assert body.statuses[key] == Status.RESONANT

    analysis = body.coherence[key]['sigma-a']
    assert analysis.top_line is not None, "expected a coherent sigma-a line for a librating MMR"
    # The libration period from the analysis (median full cycle) and from the cross spectrum must agree.
    libration_period = body.mmr_analyses[key].summary_row()['geo_period_cycle_yr']
    assert analysis.top_line.period == pytest.approx(libration_period, rel=0.25)
    assert phase_distance(analysis.top_line.phase_deg, 90.0) < 45.0

    # Coherence is diagnostic only for MMR: no gate, no status change.
    assert key not in body.zlk_gates
    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    row = summary.loc[(summary['name'] == '463') & (summary['resonance'] == key)].iloc[0]
    assert row['status'] == Status.RESONANT
    assert 'zlk_check' not in row.index or not isinstance(row['zlk_check'], str)
