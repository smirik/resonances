"""W2 zone-rules on benchmark records (skipped without the benchmark data; nothing is integrated).

The expected verdicts are the author's decisions of 2026-10-15 (benchmark/DECISIONS.md, evening): 52356 transient/medium
with the flag episodes_in_separatrix_zone, 82831 a circulation in its last regime (final_regime; angle_status episodic since
2026-10-16); and the Hilda reference 153 (clean stationary
libration of 19 deg whose centre follows the forced varpi).
"""

import pytest

from resonances.resonance.analysis.ledger import Margin
from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.pendulum import EPISODES_IN_SEPARATRIX_ZONE
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, Confidence, Status

# rtol 1e-6: the zone agrees with the default 1e-9 within one grid step, at half the cost
S = settings_from_config(mmr_zone_rtol=1e-6)


def analysed(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), S)


def test_52356_is_a_transient_inside_the_chaotic_layer():
    analysis = analysed('mmr3__52356__5J-2S-2+0+0-1')
    verdict, energy = analysis.verdict, analysis.energy
    assert (verdict.status, verdict.angle_status, verdict.confidence) == (Status.TRANSIENT, AngleStatus.EPISODIC, Confidence.MEDIUM)
    assert EPISODES_IN_SEPARATRIX_ZONE in verdict.flags and 'episodes_in_separatrix_zone' in verdict.caps
    # the members' fit resolves the pulsation (SEPARATRIX §5: eps 0.54, Omega/<Omega> 0.65-1.25, lambda 0.52)
    assert energy.modulation_source == 'members' and energy.eps[0] == pytest.approx(0.54, abs=0.08)
    assert energy.omega_min_rel == pytest.approx(0.68, abs=0.08) and energy.omega_max_rel == pytest.approx(1.25, abs=0.08)
    # the zone reaches below the adiabatic edge (PHYSICS B1b: secondary resonances at lambda ~ 0.5)
    assert energy.zone.j_lo < energy.zone.adiabatic_lo - 0.2
    assert energy.episodes_in_separatrix_zone and energy.n_crossings >= 5 and energy.n_outside == 0
    assert energy.with_slow == CheckValue.PASS  # V_E ~ V_F cycle by cycle (29 of 37 windows in SEPARATRIX §5)


def test_82831_is_a_circulation_in_its_last_regime():
    analysis = analysed('mmr2__82831__13M-24+0+11')
    verdict = analysis.verdict
    # over the record a transient (episodes, then the circulation: AUTHOR-FEEDBACK 2026-10-16 C1), ending in circulation
    assert verdict.angle_status == AngleStatus.EPISODIC and verdict.final_regime == AngleStatus.CIRCULATION
    assert analysis.extra_columns.get('rec_rule') != 'G2'
    # rule C (W7 'ug22'): the member 13M-24+2+9 librates for a while, the commensurability is its transient
    assert (verdict.status, verdict.reason, verdict.owner) == (Status.TRANSIENT, 'subresonance_librates', '13M-24+2+9')


def test_153_hilda_is_a_measured_libration():
    analysis = analysed('mmr2__153__3J-2+0-1')
    verdict, energy = analysis.verdict, analysis.energy
    assert (verdict.status, verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    # V_E in the frame of the moving centre: the period law of a pendulum, the well found from sigma alone
    assert energy.frame == 'centre' and energy.potential_e.r2 > 0.9
    assert energy.period_law_med == pytest.approx(1.0, abs=0.1)
    # at 30 deg the identity alone makes V_E ~ V_F: R0 with a is not measured, like the force law
    assert energy.with_slow == CheckValue.NOT_MEASURED
    owner = verdict.margins.get('W')
    if owner is not None and owner.margin in (Margin.FULL, Margin.THIN):
        assert 'all_physics_not_measured' not in verdict.caps
