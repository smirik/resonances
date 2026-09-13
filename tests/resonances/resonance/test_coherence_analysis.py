"""Tests for the body-level coherence orchestration and the Lidov-Kozai status gate."""

import numpy as np
import pytest

import resonances
from resonances.body import Body
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance
from resonances.resonance import coherence_analysis as ca
from resonances.resonance.cross_spectrum import CoherenceLine, CrossSpectrum
from resonances.simulation.config import SimulationConfig


def make_config(**kwargs):
    return SimulationConfig(save=None, plot=None, save_summary=False, **kwargs)


def make_body(n=200_000, dt=1.0, zlk_period=5000.0, antiphase=True, seed=0):
    """Synthetic body whose e and i share one line, in antiphase or not."""
    rng = np.random.default_rng(seed)
    t_years = np.arange(n) * dt
    common = np.sin(2 * np.pi * t_years / zlk_period)
    sign = -1.0 if antiphase else 1.0

    body = Body()
    body.name = 'synthetic'
    body.times = t_years * 2 * np.pi
    body.axis = 2.5 + 0.001 * rng.standard_normal(n)
    body.ecc = 0.2 + 0.05 * common + 0.002 * rng.standard_normal(n)
    body.inc = 0.5 + sign * 0.05 * common + 0.002 * rng.standard_normal(n)
    body.omega = np.mod(1.0 + 0.3 * common, 2 * np.pi)
    resonance = LidovKozaiResonance()
    body.lidov_kozai_resonances = [resonance]
    body.angles_unwrapped[resonance.to_s()] = np.unwrap(body.omega)
    return body, resonance


class TestPresetsAndOverrides:
    def test_defaults_per_resonance_type(self):
        config = make_config()

        assert ca.pairs_for(config, 'lidov_kozai') == ['e-i']
        assert ca.pairs_for(config, 'mmr') == ['sigma-a']
        assert ca.phase_target_for(config, 'lidov_kozai', 'e-i') == (180.0, 30.0)
        assert ca.phase_target_for(config, 'mmr', 'sigma-a') == (90.0, 30.0)

    def test_unknown_resonance_type_has_no_pairs(self):
        assert ca.pairs_for(make_config(), 'unknown') == []

    def test_override_replaces_only_the_named_type(self):
        config = make_config(coherence_pairs={'mmr': ['sigma-a', 'sigma-e']})

        assert ca.pairs_for(config, 'mmr') == ['sigma-a', 'sigma-e']
        assert ca.pairs_for(config, 'lidov_kozai') == ['e-i']  # preset untouched

    def test_phase_target_override(self):
        config = make_config(coherence_phase_targets={'lidov_kozai': {'e-i': (170.0, 10.0)}})

        assert ca.phase_target_for(config, 'lidov_kozai', 'e-i') == (170.0, 10.0)
        assert ca.phase_target_for(config, 'mmr', 'sigma-a') == (90.0, 30.0)

    def test_phase_target_accepts_a_list_from_restored_json(self):
        config = make_config(coherence_phase_targets={'lidov_kozai': {'e-i': [175.0, 25.0]}})

        assert ca.phase_target_for(config, 'lidov_kozai', 'e-i') == (175.0, 25.0)

    def test_missing_phase_target_is_none(self):
        assert ca.phase_target_for(make_config(), 'secular', 'sigma-e') is None

    def test_period_band_preset_and_override(self):
        assert ca.period_band_for(make_config(), 'lidov_kozai') == (10_000.0, None)
        assert ca.period_band_for(make_config(), 'mmr') == (None, None)
        assert ca.period_band_for(make_config(coherence_period_min=500.0), 'lidov_kozai') == (500.0, None)


class TestResolveSeries:
    def test_sigma_is_the_raw_unwrapped_angle(self):
        # The filtered angle must never be used: the low-pass filter reshapes exactly
        # the frequency content the cross spectrum measures.
        body, resonance = make_body(n=2000)
        body.angles_filtered_unwrapped[resonance.to_s()] = np.zeros(2000)

        assert ca.resolve_series(body, resonance, 'sigma') is body.angles_unwrapped[resonance.to_s()]

    def test_elements_are_returned_as_is(self):
        body, resonance = make_body(n=2000)

        assert ca.resolve_series(body, resonance, 'a') is body.axis
        assert ca.resolve_series(body, resonance, 'e') is body.ecc
        assert ca.resolve_series(body, resonance, 'i') is body.inc

    def test_unknown_key_is_none(self):
        body, resonance = make_body(n=2000)

        assert ca.resolve_series(body, resonance, 'nope') is None


class TestAnalyseResonance:
    def test_antiphase_body_passes(self):
        body, resonance = make_body(antiphase=True, seed=1)
        config = make_config(coherence_period_min=1000.0)

        analyses = ca.analyse_resonance(body, resonance, config)

        assert set(analyses) == {'e-i'}
        analysis = analyses['e-i']
        assert analysis.check == ca.CoherenceCheck.PASSED
        assert analysis.coherent is True
        assert analysis.top_line.period == pytest.approx(5000.0, rel=0.02)

    def test_in_phase_body_fails(self):
        body, resonance = make_body(antiphase=False, seed=2)
        config = make_config(coherence_period_min=1000.0)

        analysis = ca.analyse_resonance(body, resonance, config)['e-i']

        assert analysis.check == ca.CoherenceCheck.FAILED
        assert analysis.coherent is False

    def test_short_baseline_reports_short_baseline(self):
        body, resonance = make_body(n=1000, seed=3)
        config = make_config()

        analysis = ca.analyse_resonance(body, resonance, config)['e-i']

        assert analysis.check == ca.CoherenceCheck.SHORT_BASELINE
        assert analysis.coherent is None
        assert analysis.n_levels_valid == 0

    def test_invalid_pair_is_skipped_not_fatal(self):
        body, resonance = make_body(n=2000, seed=4)
        config = make_config(coherence_pairs={'lidov_kozai': ['e-nonsense']})

        assert ca.analyse_resonance(body, resonance, config) == {}


def _analysis(check, phase_target=(180.0, 30.0), line=True, r_band=-0.9):
    """Minimal PairAnalysis for gate testing, without running any spectral maths."""
    coherence_line = (
        CoherenceLine(
            period=54_000.0,
            frequency=1 / 54_000.0,
            gamma2=0.8,
            phase_deg=126.0,
            phase_std=1.0,
            amp_ratio=0.03,
            cross_amplitude=15.0,
            n_bins=1,
            n_seg=4,
            r_band=r_band,
        )
        if line
        else None
    )
    level = CrossSpectrum(pair='e-i', n_seg=4, valid=True, lines=[coherence_line] if line else [])
    return ca.PairAnalysis(
        pair='e-i',
        levels=[level],
        decisive=level,
        top_line=coherence_line,
        phase_target=phase_target,
        check=check,
        n_levels_valid=1,
    )


class TestEiExchange:
    """The e-i coherence check, which reports whether e and i trade — and nothing else.

    It used to decide the Lidov-Kozai status; that moved to `omega_free_gate`, so every
    case here asserts a measurement, never a status.
    """

    def test_a_passing_line_is_an_exchange(self):
        gate = ca.evaluate_ei_exchange({'e-i': _analysis(ca.CoherenceCheck.PASSED)}, make_config())

        assert gate.confirmed_ei is True
        assert gate.coherent is True
        assert gate.line_period == 54_000.0

    def test_a_failing_line_is_not(self):
        gate = ca.evaluate_ei_exchange({'e-i': _analysis(ca.CoherenceCheck.FAILED)}, make_config())

        assert gate.confirmed_ei is False
        assert gate.line_phase == 126.0

    def test_a_short_baseline_reports_no_measurement_rather_than_a_negative_one(self):
        gate = ca.evaluate_ei_exchange({'e-i': _analysis(ca.CoherenceCheck.SHORT_BASELINE)}, make_config())

        assert gate.confirmed_ei is None
        assert gate.check == ca.CoherenceCheck.SHORT_BASELINE

    def test_a_disabled_check_measures_nothing(self):
        gate = ca.evaluate_ei_exchange({'e-i': _analysis(ca.CoherenceCheck.FAILED)}, make_config(zlk_coherence_check=False))

        assert gate.check == ca.CoherenceCheck.DISABLED
        assert gate.confirmed_ei is None

    def test_a_missing_phase_target_measures_nothing(self):
        analyses = {'e-i': _analysis(ca.CoherenceCheck.NO_TARGET, phase_target=None)}

        assert ca.evaluate_ei_exchange(analyses, make_config()).check == ca.CoherenceCheck.NO_TARGET

    def test_a_missing_pair_measures_nothing(self):
        assert ca.evaluate_ei_exchange({}, make_config()).check == ca.CoherenceCheck.NO_TARGET

    def test_antiphase_without_exchange_does_not_count(self):
        # The 591986 failure mode: the phase criterion is satisfied, but e and i do not
        # actually trade in the time domain, so the line is not a Lidov-Kozai exchange.
        gate = ca.evaluate_ei_exchange({'e-i': _analysis(ca.CoherenceCheck.PASSED, r_band=-0.15)}, make_config())

        assert gate.check == ca.CoherenceCheck.WEAK_EXCHANGE
        assert gate.confirmed_ei is False
        assert gate.line_r_band == -0.15

    def test_unmeasurable_correlation_is_not_evidence(self):
        analyses = {'e-i': _analysis(ca.CoherenceCheck.PASSED, r_band=float('nan'))}

        assert ca.evaluate_ei_exchange(analyses, make_config()).check == ca.CoherenceCheck.WEAK_EXCHANGE

    def test_the_threshold_is_configurable(self):
        analyses = {'e-i': _analysis(ca.CoherenceCheck.PASSED, r_band=-0.3)}

        strict = ca.evaluate_ei_exchange(analyses, make_config(zlk_min_anticorrelation=0.5))
        lenient = ca.evaluate_ei_exchange(analyses, make_config(zlk_min_anticorrelation=0.2))

        assert strict.confirmed_ei is False
        assert lenient.confirmed_ei is True

    def test_no_coherent_line_at_all(self):
        gate = ca.evaluate_ei_exchange({'e-i': _analysis(ca.CoherenceCheck.FAILED, line=False)}, make_config())

        assert gate.confirmed_ei is False
        assert gate.line_period is None


class TestReporting:
    def test_summary_fields_without_a_gate_are_empty(self):
        fields = ca.summary_fields(None, None)

        assert set(fields) == set(ca.SUMMARY_COLUMNS)
        assert all(value is None for value in fields.values())

    def test_summary_fields_carry_the_gate(self):
        gate = ca.evaluate_ei_exchange({'e-i': _analysis(ca.CoherenceCheck.FAILED)}, make_config())

        fields = ca.summary_fields({}, gate)

        assert fields['confirmed_ei'] is False
        assert fields['zlk_coherent'] is False
        assert fields['zlk_check'] == 'failed'
        assert fields['zlk_line_gamma2'] == 0.8

    def test_coherence_rows_are_one_per_line(self):
        body, resonance = make_body(antiphase=True, seed=5)
        analyses = ca.analyse_resonance(body, resonance, make_config(coherence_period_min=1000.0))

        rows = ca.coherence_rows('synthetic', 'LK', analyses)

        assert rows
        assert {row['body'] for row in rows} == {'synthetic'}
        assert {row['pair'] for row in rows} == {'e-i'}
        assert sum(1 for row in rows if row['decisive']) == 1
        decisive = next(row for row in rows if row['decisive'])
        assert decisive['phase_target'] == 180.0
        assert decisive['phase_tolerance'] == 30.0
        assert decisive['period'] == pytest.approx(5000.0, rel=0.02)


class TestPublicApi:
    def test_exports(self):
        assert resonances.CoherenceCheck.PASSED == 'passed'
        assert callable(resonances.cross_spectrum)
        assert callable(resonances.forced_free_eccentricity)
