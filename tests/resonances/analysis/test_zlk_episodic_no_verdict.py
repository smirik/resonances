"""W28 D10 (cards g09-2, g09-6, g09-8; the author's answer q-1 E, 2026-10-31): an episodic ZLK omega whose free-omega
gate gives no verdict is transient only when e and i trade at constant H inside its deciding episode; otherwise it is
uncertain / gate_unavailable (-9). ZLK_EPISODIC_NO_VERDICT = ei (default) | uncertain | keep.

Expectations come from outside the package: the sign of the two laws (di/de = -e cot i / (1 - e^2) at constant
H = sqrt(1-e^2) cos i, +e tan(i/2) / (1 - e^2) at constant K = sqrt(1-e^2)(1 - cos i)) and the synthetic series built
to hold H or K exactly, so H's or K's residual is zero by construction; the status codes are SPEC 5.7 (transient 1,
uncertain gate_unavailable -9, kinematic -2), which the CLAUDE.md rule keeps apart.
"""

import numpy as np
import pytest

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.rules import Rule
from resonances.resonance.analysis.secular import decide
from resonances.resonance.analysis.secular.models import EpisodeCoupling, GateResult, LongPeriodSeries
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, Reason, ResonanceType, Status, status_code
from resonances.resonance.analysis.zlk import analyse_zlk
from resonances.resonance.analysis.zlk.episode_ei import measure_episode_coupling
from tests.resonances.analysis.builders import make_episode
from tests.resonances.analysis.test_zlk_double_scale import G5, G6, RAD, zlk_series

SETTINGS = settings_from_config()
LK = ResonanceType.LIDOV_KOZAI
P = 10_000.0  # the synthetic episode's period [yr]
E0, DE, I0 = 0.30, 0.04, np.radians(40.0)


# ------------------------------------------------------------------------------------------ the measurement


def _series(law: str, t=None):
    """e = E0 + DE sin(2 pi t / P); i such that H (law 'h') or K (law 'k') is exactly constant, i in quadrature with e
    ('quadrature'), or frozen ('frozen')."""
    t = np.arange(0.0, 200_001.0, 50.0) if t is None else t
    phase = 2 * np.pi * t / P
    e = E0 + DE * np.sin(phase)
    g, g0 = np.sqrt(1 - e**2), np.sqrt(1 - E0**2)
    if law == 'h':
        inc = np.arccos(g0 * np.cos(I0) / g)
    elif law == 'k':
        inc = np.arccos(1 - g0 * (1 - np.cos(I0)) / g)
    elif law == 'quadrature':
        inc = I0 + np.radians(0.5) * np.cos(phase)
    else:
        inc = np.full_like(t, I0)
    omega = np.radians(90.0) + 0.3 * np.sin(phase)
    series = LongPeriodSeries(LK, 'LK', 'synthetic', t, omega, np.full_like(t, 2.5), e, inc)
    return series


def _episode(t0=20_000.0, t1=180_000.0, period=P):
    """Turning points every half period from t0 to t1."""
    return make_episode(episode_id=3, tp_times_yr=np.arange(t0, t1 + 1.0, period / 2), period_med_yr=period)


def test_a_trade_at_constant_h_is_zlk():
    ei = measure_episode_coupling(_series('h'), _episode(), SETTINGS)
    assert ei.coupling == 'zlk' and ei.reason == ''
    assert ei.n_periods == pytest.approx(16.0)
    assert abs(ei.phase_deg - 180.0) < 5.0  # di/de < 0 at constant H
    assert ei.coherence < -0.95
    assert ei.eta_h < 0.05  # H held exactly; i frozen would give 1
    assert ei.ratio_h == pytest.approx(1.0, abs=0.1)  # the slope of the law at the mean e, i
    assert ei.episode_id == 3 and ei.period_yr == P


def test_a_trade_at_constant_k_is_a_forced_beat():
    ei = measure_episode_coupling(_series('k'), _episode(), SETTINGS)
    assert ei.coupling == 'forced_beat'
    assert ei.phase_deg < 5.0 or ei.phase_deg > 355.0  # di/de > 0 at constant K
    assert ei.coherence > 0.95 and ei.eta_k < 0.05 and ei.eta_h > SETTINGS.zlk_ei_h_residual_max


def test_i_in_quadrature_with_e_is_none():
    ei = measure_episode_coupling(_series('quadrature'), _episode(), SETTINGS)
    assert ei.coupling == 'none'
    assert abs(ei.phase_deg - 270.0) < 5.0 or abs(ei.phase_deg - 90.0) < 5.0


@pytest.mark.parametrize(
    'series, episode, reason',
    [
        (_series('h'), _episode(20_000.0, 44_000.0), 'short_episode'),  # 2.4 periods < ZLK_EI_MIN_CYCLES (3): RS2
        (_series('h', np.arange(0.0, 200_001.0, 4_500.0)), _episode(period=9_000.0), 'undersampled'),  # 2 per P
        (_series('h'), None, 'no_episode'),
    ],
)
def test_what_cannot_be_measured_is_not_measured(series, episode, reason):
    ei = measure_episode_coupling(series, episode, SETTINGS)
    assert (ei.coupling, ei.reason) == ('not_measured', reason)


def test_a_frozen_inclination_is_no_trade():
    # e moves, i does not: H changes as fully as it can (eta_H = 1), a measurement of no coupling, not a missing one
    ei = measure_episode_coupling(_series('frozen'), _episode(), SETTINGS)
    assert ei.coupling == 'none' and ei.eta_h == pytest.approx(1.0, abs=0.01)


def test_an_orbit_flip_is_not_measured():
    series = _series('h')
    series.inc = np.linspace(np.radians(80.0), np.radians(100.0), len(series.t_yr))
    assert measure_episode_coupling(series, _episode(), SETTINGS).reason == 'orbit_flip'


def test_the_measurement_reads_only_the_episode():
    # constant-H trade inside the episode, i in quadrature outside it: still zlk (the whole record would mix them)
    h, q = _series('h'), _series('quadrature')
    outside = (h.t_yr < 20_000.0) | (h.t_yr > 180_000.0)
    h.inc = np.where(outside, q.inc, h.inc)
    assert measure_episode_coupling(h, _episode(), SETTINGS).coupling == 'zlk'


# ----------------------------------------------------------------------------------------------- the status


def _ei(coupling, n_periods=8.0):
    measured = coupling not in ('not_measured',)
    return EpisodeCoupling(
        coupling=coupling,
        reason='' if measured else 'short_episode',
        n_periods=n_periods,
        phase_deg=180.0 if coupling == 'zlk' else 90.0,
        coherence=-0.99 if coupling == 'zlk' else 0.1,
        eta_h=0.1 if coupling == 'zlk' else 1.2,
        min_cycles=3.0,
        phase_tol_deg=45.0,
        min_coherence=0.5,
        h_residual_max=0.5,
    )


def _status(gate, angle_status=AngleStatus.EPISODIC, ledger=None):
    angle = decide.AngleDecision(angle_status, decisive_episode_id=0)
    return decide.decide_status(LK, angle, gate, ledger if ledger is not None else Ledger.from_settings(SETTINGS))


def _code(decision, angle_status=AngleStatus.EPISODIC):
    return status_code(LK, decision.status, angle_status, decision.reason)


NO_VERDICT = [
    dict(outcome='planetary_residual', n_cycles=15.0, rho_max=13.0),  # 2023 RS2, 88899 500 kyr, 797204
    dict(outcome='short_window', n_cycles=2.0),
    dict(outcome='undefined_phase', n_cycles=10.0),
    dict(outcome='unavailable'),  # no free elements (not the co-orbital detail)
]


@pytest.mark.parametrize('gate_fields', NO_VERDICT)
def test_policy_ei_keeps_a_transient_only_with_zlk_coupling_in_the_episode(gate_fields):
    zlk = _status(GateResult(**gate_fields, min_cycles=3.0, episode_ei=_ei('zlk'), episodic_policy='ei'))
    assert (zlk.status, zlk.reason, _code(zlk)) == (Status.TRANSIENT, Reason.DYNAMIC_EPISODE, 1)
    assert Rule.ZLK_GATE_UNCONFIRMED in zlk.rules_fired  # the rule decided, also when it kept the status
    for coupling in ('none', 'forced_beat', 'not_measured'):  # 88899 500 kyr none, RS2 and 797204 100 kyr not measured
        other = _status(GateResult(**gate_fields, min_cycles=3.0, episode_ei=_ei(coupling), episodic_policy='ei'))
        assert (other.status, other.reason, _code(other)) == (Status.UNCERTAIN, Reason.GATE_UNAVAILABLE, -9), coupling
        assert other.checks['check_zlk_free_omega'] == CheckValue.NOT_MEASURED
        assert Rule.ZLK_GATE_UNCONFIRMED in other.rules_fired


def test_policies_uncertain_and_keep():
    gate = dict(outcome='planetary_residual', n_cycles=15.0, min_cycles=3.0, rho_max=13.0, episode_ei=_ei('zlk'))
    uncertain = _status(GateResult(**gate, episodic_policy='uncertain'))
    assert (uncertain.status, _code(uncertain)) == (Status.UNCERTAIN, -9)
    keep = _status(GateResult(**dict(gate, episode_ei=_ei('none')), episodic_policy='keep'))
    assert (keep.status, keep.reason, _code(keep)) == (Status.TRANSIENT, Reason.DYNAMIC_EPISODE, 1)
    assert Rule.ZLK_GATE_UNCONFIRMED not in keep.rules_fired


def test_the_default_policy_is_the_authors_answer():
    assert SETTINGS.zlk_episodic_no_verdict == 'ei' and GateResult('planetary_residual').episodic_policy == 'ei'
    with pytest.raises(ValueError):
        settings_from_config(zlk_episodic_no_verdict='sometimes')


def test_the_ledger_holds_the_comparisons_behind_the_class():
    ledger = Ledger.from_settings(SETTINGS)
    _status(GateResult('planetary_residual', n_cycles=15.0, min_cycles=3.0, episode_ei=_ei('none')), ledger=ledger)
    entries = {entry.name: entry for entry in ledger.entries}
    assert entries['ei_episode_cycles'].result is True
    assert entries['ei_episode_antiphase'].result is False  # 90 deg off antiphase > 45
    assert entries['ei_episode_h_residual'].result is False and entries['ei_episode_h_residual'].column == 'zlk_ei_episode_eta_h'


def test_other_outcomes_and_angles_do_not_read_the_coupling():
    none = _ei('none')
    # an unfitted co-orbital stays -9 by its own rule, whatever the coupling (W28 g09-6 B: 2020 PP1)
    coorbital = _status(GateResult('unavailable', min_cycles=3.0, detail='coorbital_unfitted_forcing', episode_ei=_ei('zlk')))
    assert _code(coorbital) == -9 and Rule.ZLK_COORBITAL_EPISODIC in coorbital.rules_fired
    assert Rule.ZLK_GATE_UNCONFIRMED not in coorbital.rules_fired
    # a kinematic gate (a measurement): pinned -> -2; not pinned keeps the transient (1866), never -9
    pinned = _status(GateResult('kinematic', n_cycles=10.0, min_cycles=3.0, rho_max=0.5, episode_ei=none))
    assert _code(pinned) == -2
    loose = _status(GateResult('kinematic', n_cycles=10.0, min_cycles=3.0, rho_max=2.0, episode_ei=none))
    assert (loose.status, _code(loose)) == (Status.TRANSIENT, 1)
    # a disabled gate keeps the geometric status (not the same as no verdict)
    assert _status(GateResult('disabled', episode_ei=none)).status == Status.TRANSIENT
    # a drift or too few cycles stays baseline_needed; a libration without a verdict is -9 without the rule
    for angle_status in (AngleStatus.DRIFT, AngleStatus.INSUFFICIENT_TIME):
        decision = _status(GateResult('planetary_residual', n_cycles=4.0, min_cycles=3.0, episode_ei=none), angle_status)
        assert decision.reason == Reason.BASELINE_NEEDED and Rule.ZLK_GATE_UNCONFIRMED not in decision.rules_fired
    libration = _status(GateResult('planetary_residual', n_cycles=15.0, min_cycles=3.0, episode_ei=_ei('zlk')), AngleStatus.LIBRATION)
    assert _code(libration, AngleStatus.LIBRATION) == -9 and Rule.ZLK_GATE_UNCONFIRMED not in libration.rules_fired


# ------------------------------------------------------------------------------------------------ end to end


def _episodic_librator(trade: bool):
    """omega_free = 90 + 25 sin(2 pi t / 40 kyr) until 240 kyr, then circulating; e 0.30 + 0.04 cos at the cycle, i at
    constant H (trade) or fixed at 40 deg; node -20 arcsec/yr, small forced terms at g5 and g6; a = 2.5 au (no planet's
    co-orbital)."""
    t = np.arange(0.0, 500_001.0, 100.0)
    switch = 240_000.0
    librating = np.radians(90.0) + np.radians(25.0) * np.sin(2 * np.pi * t / 40_000.0)
    circulating = np.radians(90.0) + 2 * np.pi * (t - switch) / 30_000.0
    omega_free = np.where(t < switch, librating, circulating)
    Omega = -20.0 * RAD * t + 0.7
    e_free = 0.30 + 0.04 * np.cos(2 * np.pi * t / 40_000.0)
    z_e = e_free * np.exp(1j * (omega_free + Omega))
    z_e = z_e + 0.03 * np.exp(1j * (G5 * RAD * t + 1.0)) + 0.007 * np.exp(1j * (G6 * RAD * t + 2.0))
    e = np.abs(z_e)
    inc = np.arccos(np.sqrt(1 - 0.30**2) * np.cos(I0) / np.sqrt(1 - e**2)) if trade else np.full_like(t, I0)
    return zlk_series(t, z_e, Omega, inc, a=2.5)


def test_end_to_end_the_episode_coupling_decides_a_row_without_a_gate_verdict():
    for trade in (True, False):
        series, elements = _episodic_librator(trade)
        analysis = analyse_zlk(series, SETTINGS, free_elements=elements)
        verdict, gate = analysis.verdict, analysis.gate
        assert verdict.angle_status == AngleStatus.EPISODIC
        row = analysis.summary_row()
        assert row['zlk_ei_episode_coupling'] == gate.episode_ei.coupling
        if gate.outcome not in decide.GATE_NO_VERDICT:
            pytest.skip(f'the synthetic gate gives a verdict ({gate.outcome}); the decision tests cover the branch')
        expected = (Status.TRANSIENT, 1) if trade else (Status.UNCERTAIN, -9)
        assert (verdict.status, verdict.status_code) == expected, (trade, gate.episode_ei)
        assert gate.episode_ei.coupling == ('zlk' if trade else 'none')
        assert 'zlk_gate_unconfirmed' in verdict.rules_fired
        keep = analyse_zlk(series, settings_from_config(zlk_episodic_no_verdict='keep'), free_elements=elements).verdict
        assert keep.status == Status.TRANSIENT and 'zlk_gate_unconfirmed' not in keep.rules_fired
