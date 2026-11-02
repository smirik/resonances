"""W29 minimal policy end to end: `analyse_mmr` on synthetic records and one path through `Simulation` -> summary.csv ->
`status_selected` -> `SimulationSerializer.restore(recompute_librations=True)`.

The records are exact pendulums near 2J-1 with a answering through the identity sigma_dot = c (a - a*), 1-yr step:
12 cycles of 1000 yr over 12 kyr is a libration of 12 complete cycles (certain: N >= 10, every veto clean); the same
pendulum for 6 kyr followed by a circulation of 700 yr per turn is one episode of 5.5 complete cycles (transient, high:
N >= 3) whose record ends in circulation (no extend)."""

import numpy as np
import pandas as pd

from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Action, Confidence, Status
from resonances.simulation import SimulationSerializer
from resonances.simulation.config import status_selected
from tests.resonances.simulation.test_restore_and_resume import CIRC, LIB, _saved_run

MINIMAL = AnalysisSettings(classifier_policy='minimal')
N = 12001
T = np.arange(N, dtype=float)
W = 2 * np.pi / 1000.0


def series(sigma, sigma_dot) -> MMRSeries:
    n_jupiter = np.radians(30.3496)  # rad / Gaussian yr
    a_star = (np.pi / n_jupiter) ** (2 / 3)  # n = 2 n_J
    c = 3 * np.pi * a_star**-2.5
    zero = np.zeros(N)
    return MMRSeries(
        T,
        a_star + sigma_dot / c,
        np.full(N, 0.1),
        zero,
        zero,
        zero,
        zero,
        sigma % (2 * np.pi),
        ResonanceSpec.from_key('2J-1'),
        Regime(1.0, T[-1], N),
    )


def test_a_pendulum_of_twelve_cycles_is_certain():
    analysis = analyse_mmr(series(np.cos(W * T), -W * np.sin(W * T)), MINIMAL, scan_owner=False, scan_subresonances=False)
    v = analysis.verdict
    assert (v.policy, v.status, v.confidence, v.action) == ('minimal', Status.RESONANT, Confidence.CERTAIN, Action.NONE)
    assert v.n_cycles >= 10 and v.n_counted_episodes == 1
    assert analysis.policy is not None and analysis.policy.counted_episode_ids == frozenset(analysis.policy.counted_episode_ids)
    assert sum(bool(row['counted']) for row in analysis.episode_rows()) == v.n_counted_episodes


def test_a_pendulum_then_circulation_is_one_counted_episode():
    sigma = np.where(T < 6000, np.cos(W * T), 1.0 + 2 * np.pi * (T - 6000) / 700.0)
    rate = np.where(T < 6000, -W * np.sin(W * T), 2 * np.pi / 700.0)
    v = analyse_mmr(series(sigma, rate), MINIMAL, scan_owner=False, scan_subresonances=False).verdict
    assert (v.status, v.n_counted_episodes) == (Status.TRANSIENT, 1)
    assert 5.0 <= v.n_cycles < 6.0 and v.confidence == Confidence.HIGH  # 5.5 complete cycles
    assert v.pol_end_travel_rev >= 1 and v.action != Action.EXTEND


def test_the_policy_survives_the_simulation_path(tmp_path):
    sim = _saved_run(tmp_path, classifier_policy='minimal')
    librator = next(body for body in sim.bodies if body.name == 'librator')
    verdict = librator.librations[LIB]
    assert verdict.policy == 'minimal' and librator.statuses[LIB] == str(verdict.status) == 'resonant'
    summary = pd.read_csv(tmp_path / 'summary.csv', dtype=str, keep_default_na=False)
    assert set(summary['policy']) == {'minimal'} and 'confidence_basis' in summary.columns
    # SAVE=nonzero saves the librator's resonant row, not the circulation; the review mode reads the policy's level
    assert status_selected(sim.config.save, str(verdict.status), str(verdict.libration_origin))
    circ = librator.librations[CIRC]
    assert not status_selected(sim.config.save, str(circ.status), str(circ.libration_origin))
    # review also takes a non_resonant row whose level is medium or low (the policy's level, not a cap)
    review = status_selected('review', str(circ.status), str(circ.libration_origin), str(circ.action), str(circ.confidence))
    assert review == (circ.confidence in (Confidence.MEDIUM, Confidence.LOW))
    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=True)
    again = next(body for body in restored.bodies if body.name == 'librator')
    assert again.librations[LIB].policy == 'minimal'
    assert (again.statuses[LIB], again.librations[LIB].confidence) == (librator.statuses[LIB], verdict.confidence)
    assert again.librations[CIRC].status == Status.NON_RESONANT
