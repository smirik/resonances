"""R0 read inside the accepted episodes (W34, `zone.episode_r0`, SEPARATRIX_R0_EPISODES; the author on 79653 3M+3J-7,
checked by Claude and Codex): where one V_E pooled over the record measures nothing, an episode with two disjoint full
cycles is fitted on its own, with the thresholds unchanged. Lift-only: a pass needs no failing episode. And the filter
guard, which compared another episode, withdraws nothing of that R0 (`decide.r0_beside_guard`).

The pendulum is integrated from sigma_ddot = -omega0^2 sin(sigma - pi) with a from the identity (exact R0: V_E = V_F,
T_obs = T_pred); the driven angle has the same sigma while a does not answer it (V_F = 0: R0 with a fails)."""

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as geo
from resonances.resonance.analysis.mmr import decide, zone
from resonances.resonance.analysis.mmr.preprocess import DecisiveQuantities, FilterSensitivity
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import CheckValue
from tests.resonances.analysis.builders import make, make_energy
from tests.resonances.analysis.impl_response_pendulum_synthetic import episode_from, pendulum_record

S = AnalysisSettings()


def _first_cycles(record, n_tp):
    """The record's geometry with one accepted run of its first `n_tp` turning points."""
    tp = record.geometry.turning_points
    keep = slice(0, n_tp)
    part = geo.TurningPoints(tp.indices[keep], tp.times_yr[keep], tp.values_rad[keep], tp.kinds[keep], tp.prominence_rad)
    record.geometry.turning_points = part
    record.geometry.runs = [episode_from(part)]
    return record


def _r0(record):
    modulation = zone.eccentricity_modulation(record.series, record.slow, record.context, record.t)
    return zone.episode_r0(record.t, record.prep, record.geometry, modulation, record.context, S)


def test_a_pendulum_episode_of_two_cycles_passes_on_its_own():
    # T0 = 300 yr at h = 0.5 (A = 90 deg): five turning points are two disjoint full cycles
    found = _r0(_first_cycles(pendulum_record(), 5))
    assert found == (CheckValue.PASS, CheckValue.PASS, 0)


def test_one_cycle_is_not_enough():
    assert _r0(_first_cycles(pendulum_record(), 4)) is None


def test_an_angle_whose_a_does_not_answer_is_no_lift():
    # sigma swings as the pendulum, a stays at a*: V_E fits sigma, V_F is flat, R0 with a cannot pass
    record = _first_cycles(pendulum_record(), 5)
    record.prep.a_f = np.full_like(record.t, 4.0) + 1e-9 * np.random.default_rng(3).standard_normal(len(record.t))
    record.prep.adot_f = np.gradient(record.prep.a_f, record.t)
    assert _r0(record) is None


@pytest.mark.parametrize('local, compared, kept', [(0, 1, True), (1, 1, False), (None, 1, False), (0, None, False)])
def test_the_guard_withdraws_r0_only_from_the_episode_it_compared(local, compared, kept):
    energy = make_energy(with_slow=CheckValue.PASS, r0_episode_id=local)
    quantities = make(DecisiveQuantities)
    sensitivity = FilterSensitivity(quantities, quantities, ('energy',), decisive_episode_id=compared)
    assert decide.r0_beside_guard(energy, sensitivity) == kept
    checks = {'check_mmr_potential': CheckValue.PASS}
    decide.apply_filter_sensitivity(checks, {}, sensitivity, keep=decide.R0_KEPT if kept else ())
    assert checks['check_mmr_potential'] == (CheckValue.PASS if kept else CheckValue.NOT_MEASURED)
