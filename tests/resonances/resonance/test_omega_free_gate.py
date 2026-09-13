"""Tests for the free-omega gate.

Each case is built from a signal whose answer is known before the code runs: a forced term
at a real planetary frequency plus a free term at a rate chosen by hand. The gate is then
asked what it makes of it.
"""

import numpy as np
import pytest

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.resonance.classify.models import ResonanceStatus
from resonances.resonance.omega_free_gate import (
    SUMMARY_COLUMNS,
    GateOutcome,
    _confirmation_targets,
    apply_gate,
    summary_fields,
)
from resonances.secular.free_elements import ARCSEC_PER_TURN, RAD_PER_ARCSEC, free_elements
from resonances.simulation.config import SimulationConfig

BASELINE = 5e5
P_LIB = 50_000.0  # ten cycles in the baseline, comfortably over min_cycles = 3


@pytest.fixture
def config():
    return SimulationConfig(save=None, plot=None, save_summary=False)


def _wave(times, amplitude, arcsec_per_year, phase=0.0):
    return amplitude * np.exp(1j * (arcsec_per_year * RAD_PER_ARCSEC * times + phase))


def _decompose(z_e, z_i, times, **kwargs):
    ecc, varpi = np.abs(z_e), np.angle(z_e)
    inc, Omega = 2.0 * np.arcsin(np.abs(z_i)), np.angle(z_i)
    return free_elements(times, ecc, inc, Omega, varpi - Omega, sampling_years=None, **kwargs)


@pytest.fixture
def times():
    return np.arange(0.0, BASELINE, 200.0)


def _librator(times, delta_deg=60.0, period=40_000.0, free_rate=-19.0):
    """A Lidov-Kozai lock: varpi_free and Omega_free share a rate, so omega_free is bounded.

    The libration itself is a wobble of amplitude `delta_deg` on top of that shared rate.
    """
    wobble = np.radians(delta_deg) * np.sin(2 * np.pi * times / period)
    z_e = _wave(times, 0.03, PLANETARY_FREQUENCIES['g5']) + 0.20 * np.exp(1j * (free_rate * RAD_PER_ARCSEC * times + wobble))
    z_i = _wave(times, 0.02, PLANETARY_FREQUENCIES['s6']) + _wave(times, 0.30, free_rate)
    return z_e, z_i


class TestConfirmation:
    def test_a_librator_is_confirmed(self, times, config):
        # T1: bounded free omega whose rates are nowhere near a planetary fundamental.
        elements = _decompose(*_librator(times), times)

        status, gate = apply_gate(ResonanceStatus.LIBRATION, elements, P_LIB, config)

        assert status == ResonanceStatus.LIBRATION
        assert gate.outcome is GateOutcome.CONFIRMED
        assert elements.omega_range < 2 * np.pi
        assert gate.comment == 'gate: 2 confirmed'

    def test_a_bounded_residual_at_a_planetary_rate_is_withheld(self, times, config):
        # T13, the case the removed amplitude floor was meant to catch. The free part is
        # tiny and what dominates the residual is an unsubtracted g8 line: over 500 kyr g8
        # turns a quarter of a circle, so the angle looks beautifully confined — and the
        # only thing that gives it away is that it turns at exactly g8.
        z_e = _wave(times, 0.03, PLANETARY_FREQUENCIES['g5']) + _wave(times, 0.05, PLANETARY_FREQUENCIES['g8'])
        z_i = _wave(times, 0.02, PLANETARY_FREQUENCIES['s6']) + _wave(times, 0.30, -19.0)
        elements = _decompose(z_e, z_i, times)

        status, gate = apply_gate(ResonanceStatus.LIBRATION, elements, P_LIB, config)

        assert status == ResonanceStatus.UNCERTAIN
        assert gate.outcome is GateOutcome.PLANETARY_RESIDUAL
        assert any('b_varpi~g8' in match for match in gate.matches)
        assert 'residual at planetary freq' in gate.comment

    def test_the_zero_target_tracks_whether_the_constant_was_fitted(self, times, config):
        # A constant left in the residual is the most dangerous leak of all, since it makes
        # any angle look confined — so zero is a target exactly when it was not subtracted.
        # On the inclination side it always is: 'const' heads PRIORITY_I.
        short = _decompose(*_librator(times), times)
        long_times = np.arange(0.0, 1e7, 200.0)
        full = _decompose(*_librator(long_times), long_times)

        assert 'const' in short.dropped_e and 'const' in _confirmation_targets(short, PLANETARY_FREQUENCIES)[0]
        assert 'const' not in full.dropped_e and 'const' not in _confirmation_targets(full, PLANETARY_FREQUENCIES)[0]
        assert 'const' not in short.dropped_i and 'const' not in full.dropped_i


class TestDemotion:
    def test_a_circulator_clear_of_every_combination_is_demoted(self, times, config):
        # T2: g - s = 150 arcsec/yr, past every g_j - s_k (the largest is g6 - s6 = 54.6).
        z_e = _wave(times, 0.03, PLANETARY_FREQUENCIES['g5']) + _wave(times, 0.20, 130.0)
        z_i = _wave(times, 0.02, PLANETARY_FREQUENCIES['s6']) + _wave(times, 0.30, -20.0)
        elements = _decompose(z_e, z_i, times)

        status, gate = apply_gate(ResonanceStatus.LIBRATION, elements, P_LIB, config)

        assert status == ResonanceStatus.LIBRATION_UNCERTAIN
        assert gate.outcome is GateOutcome.KINEMATIC
        assert elements.b_omega == pytest.approx(150.0, rel=0.05)
        assert gate.comment == 'gate: 2->-2 kinematic libration'

    def test_a_winding_residual_at_a_planetary_rate_is_withheld_not_demoted(self, times, config):
        # A free omega that winds at g6 - s6 could be the object's own rate or a defect of
        # the subtraction. The two are not separable, so no verdict is issued — and -9 must
        # never be collapsed into -2 in any statistic.
        combination = PLANETARY_FREQUENCIES['g6'] - PLANETARY_FREQUENCIES['s6']
        z_e = _wave(times, 0.03, PLANETARY_FREQUENCIES['g5']) + _wave(times, 0.20, combination - 20.0)
        z_i = _wave(times, 0.02, PLANETARY_FREQUENCIES['s6']) + _wave(times, 0.30, -20.0)
        elements = _decompose(z_e, z_i, times)

        status, gate = apply_gate(ResonanceStatus.LIBRATION, elements, P_LIB, config)

        assert status == ResonanceStatus.UNCERTAIN
        assert any('b_omega~g6-s6' in match for match in gate.matches)

    def test_the_sign_of_a_node_rate_matters(self, times, config):
        # Every s_j is retrograde. A free node creeping forwards at +2.99 arcsec/yr has the
        # magnitude of s7 and nothing else in common with it, so it must not match.
        rate = -PLANETARY_FREQUENCIES['s7']
        z_e = _wave(times, 0.03, PLANETARY_FREQUENCIES['g5']) + _wave(times, 0.20, 130.0)
        z_i = _wave(times, 0.02, PLANETARY_FREQUENCIES['s6']) + _wave(times, 0.30, rate)
        elements = _decompose(z_e, z_i, times)

        status, gate = apply_gate(ResonanceStatus.LIBRATION, elements, P_LIB, config)

        assert elements.b_Omega == pytest.approx(abs(PLANETARY_FREQUENCIES['s7']), rel=0.05)
        assert not any('b_Omega~s7' in match for match in gate.matches)
        assert status == ResonanceStatus.LIBRATION_UNCERTAIN


class TestPreconditions:
    def test_a_window_holding_too_few_cycles_yields_no_verdict(self, times, config):
        # T9: 500 kyr against a 200 kyr libration period is 2.5 cycles, under min_cycles = 3.
        elements = _decompose(*_librator(times), times)

        status, gate = apply_gate(ResonanceStatus.LIBRATION, elements, 200_000.0, config)

        assert status == ResonanceStatus.UNCERTAIN
        assert gate.outcome is GateOutcome.SHORT_WINDOW
        assert gate.n_cycles == pytest.approx(2.5, rel=0.01)
        assert '2.5 libration periods' in gate.comment

    def test_an_unknown_libration_period_does_not_block_the_gate(self, times, config):
        # The periodogram can fail to return a period even at status 2; that is not a
        # reason to refuse a verdict the free omega can give on its own.
        elements = _decompose(*_librator(times), times)

        status, gate = apply_gate(ResonanceStatus.LIBRATION, elements, None, config)

        assert status == ResonanceStatus.LIBRATION
        assert gate.n_cycles is None

    def test_a_mostly_masked_series_yields_no_verdict(self, times, config):
        elements = _decompose(*_librator(times), times, mask_quantile=1.5)

        status, gate = apply_gate(ResonanceStatus.LIBRATION, elements, P_LIB, config)

        assert status == ResonanceStatus.UNCERTAIN
        assert gate.outcome is GateOutcome.UNDEFINED_PHASE

    @pytest.mark.parametrize(
        'status',
        [ResonanceStatus.TRANSIENT, ResonanceStatus.NON_RESONANT, ResonanceStatus.NEAR_SEPARATRIX, ResonanceStatus.CHAOTIC],
    )
    def test_only_a_libration_is_ever_touched(self, times, config, status):
        elements = _decompose(*_librator(times), times)

        result, gate = apply_gate(status, elements, P_LIB, config)

        assert result == status
        assert gate.outcome is GateOutcome.NOT_APPLICABLE
        assert gate.comment == ''

    def test_a_disabled_gate_leaves_the_status_alone(self, times):
        config = SimulationConfig(save=None, plot=None, save_summary=False, free_gate_enabled=False)
        elements = _decompose(*_librator(times), times)

        status, gate = apply_gate(ResonanceStatus.LIBRATION, elements, P_LIB, config)

        assert status == ResonanceStatus.LIBRATION
        assert gate.outcome is GateOutcome.DISABLED

    def test_a_missing_decomposition_leaves_the_status_alone(self, config):
        status, gate = apply_gate(ResonanceStatus.LIBRATION, None, P_LIB, config)

        assert status == ResonanceStatus.LIBRATION
        assert gate.outcome is GateOutcome.UNAVAILABLE


class TestInvariance:
    def test_the_verdict_survives_resampling(self, config):
        # T7: the answer must not depend on the cadence the series happen to be saved at.
        fine = np.arange(0.0, BASELINE, 100.0)
        verdicts = set()
        for sampling in (None, 500.0, 1000.0):
            z_e, z_i = _librator(fine)
            ecc, varpi = np.abs(z_e), np.angle(z_e)
            elements = free_elements(fine, ecc, 2.0 * np.arcsin(np.abs(z_i)), np.angle(z_i), varpi - np.angle(z_i), sampling_years=sampling)
            verdicts.add(apply_gate(ResonanceStatus.LIBRATION, elements, P_LIB, config)[0])

        assert verdicts == {ResonanceStatus.LIBRATION}

    def test_the_tolerance_is_one_rayleigh_width(self, times, config):
        elements = _decompose(*_librator(times), times)

        assert elements.resolution == pytest.approx(ARCSEC_PER_TURN / BASELINE, rel=0.01)


class TestSummaryFields:
    def test_a_missing_gate_still_produces_every_column(self):
        fields = summary_fields(None)

        assert set(fields) == set(SUMMARY_COLUMNS)
        assert all(value is None for value in fields.values())

    def test_matches_are_recorded_next_to_the_outcome(self, times, config):
        z_e = _wave(times, 0.03, PLANETARY_FREQUENCIES['g5']) + _wave(times, 0.05, PLANETARY_FREQUENCIES['g8'])
        z_i = _wave(times, 0.02, PLANETARY_FREQUENCIES['s6']) + _wave(times, 0.30, -19.0)
        _, gate = apply_gate(ResonanceStatus.LIBRATION, _decompose(z_e, z_i, times), P_LIB, config)

        fields = summary_fields(gate)

        assert fields['free_gate'] == 'planetary_residual'
        assert 'b_varpi~g8' in fields['free_gate_matches']
        assert fields['free_n_cycles'] == pytest.approx(BASELINE / P_LIB, rel=0.01)
