"""Tests for the forced/free decomposition of the eccentricity and inclination vectors.

The synthetic cases are built the other way round from the code under test: a signal is
assembled from a forced term at a *known* planetary frequency plus a free term at a
frequency chosen by hand, and the test checks that the decomposition recovers the free
term's amplitude and rate. Nothing here is compared against the implementation's own
intermediate results.
"""

import numpy as np
import pytest
from scipy.signal import decimate

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.secular.free_elements import (
    ARCSEC_PER_TURN,
    RAD_PER_ARCSEC,
    SUMMARY_COLUMNS,
    candidate_frequencies,
    choose_basis,
    cluster,
    _resample,
    decimation_steps,
    free_elements,
    has_remainder,
    mask_origin,
    rate,
    remove_forced,
    series_rows,
    summary_fields,
)


def _elements_from_vectors(z_e, z_i):
    """Invert z_e = e exp(i varpi), z_i = sin(i/2) exp(i Omega) back to elements."""
    ecc, varpi = np.abs(z_e), np.angle(z_e)
    inc, Omega = 2.0 * np.arcsin(np.abs(z_i)), np.angle(z_i)
    return ecc, inc, Omega, varpi - Omega


def _series(times, terms):
    """Sum of complex exponentials, given as (amplitude, arcsec/yr) pairs."""
    return sum(amplitude * np.exp(1j * value * RAD_PER_ARCSEC * times) for amplitude, value in terms)


@pytest.fixture
def times():
    # 4 Myr at 200 yr separates every mode used below: the tightest pair, g5 and g7, is
    # 1.17 arcsec/yr apart against a 2/T threshold of 0.65.
    return np.arange(0.0, 4e6, 200.0)


class TestCandidateFrequencies:
    def test_g5_and_s6_periods_match_their_textbook_values(self):
        # g5 drives the eccentricity with a 304 kyr period, s6 the node with 49 kyr.
        g5 = candidate_frequencies('e')['g5']
        s6 = candidate_frequencies('i')['s6']

        assert ARCSEC_PER_TURN / g5 == pytest.approx(304_404, rel=1e-3)
        assert abs(ARCSEC_PER_TURN / s6) == pytest.approx(49_193, rel=1e-3)

    def test_the_zero_mode_appears_once_per_vector(self):
        # s5 is identically zero and *is* the constant, so it is carried under that name;
        # the eccentricity vector has no zero fundamental and gets a technical one.
        assert candidate_frequencies('e')['const'] == 0.0
        assert candidate_frequencies('i')['const'] == PLANETARY_FREQUENCIES['s5'] == 0.0
        assert 's5' not in candidate_frequencies('i')

    def test_each_kind_takes_only_its_own_modes(self):
        assert set(candidate_frequencies('e')) == {'const', 'g5', 'g6', 'g7', 'g8'}
        assert set(candidate_frequencies('i')) == {'const', 's6', 's7', 's8'}

    def test_unknown_kind_is_rejected(self):
        with pytest.raises(ValueError):
            candidate_frequencies('a')


class TestCluster:
    def test_a_ten_myr_baseline_separates_every_mode(self):
        # 2/T is 0.26 arcsec/yr there, and the closest pair (const, g8) is 0.67 apart.
        # Groups come out ordered by frequency: const 0, g8 0.67, g7 3.09, g5 4.26, g6 28.25.
        assert cluster(candidate_frequencies('e'), 1e7) == [['const'], ['g8'], ['g7'], ['g5'], ['g6']]

    def test_at_five_hundred_kyr_the_slow_modes_merge_and_g6_stands_alone(self):
        # Threshold 5.18 arcsec/yr. Sorted rates 0, 0.67, 3.09, 4.26, 28.25 have neighbour
        # gaps 0.67, 2.42, 1.17, 23.99 — the first three chain, the last one breaks.
        assert cluster(candidate_frequencies('e'), 5e5) == [['g5', 'g7', 'g8', 'const'], ['g6']]
        assert cluster(candidate_frequencies('i'), 5e5) == [['s6'], ['const', 's7', 's8']]

    def test_at_a_hundred_kyr_everything_collapses_to_one_group(self):
        # Threshold 25.9 arcsec/yr exceeds even the g5-g6 gap of 23.99: the screening regime,
        # where a single effective planetary arrow is subtracted per vector.
        assert cluster(candidate_frequencies('e'), 1e5) == [['g5', 'g6', 'g7', 'g8', 'const']]

    def test_chaining_joins_modes_that_are_not_themselves_close(self):
        # const and g5 are 4.26 apart, over the 3.0 threshold of a 864 kyr window, yet they
        # share a group because g8 and g7 bridge them.
        groups = cluster(candidate_frequencies('e'), 2.0 * ARCSEC_PER_TURN / 3.0)

        assert ['g5', 'g7', 'g8', 'const'] in groups


class TestChooseBasis:
    def test_the_representative_follows_the_documented_priority(self):
        basis_e, dropped_e, _ = choose_basis(candidate_frequencies('e'), 5e5)
        basis_i, dropped_i, _ = choose_basis(candidate_frequencies('i'), 5e5)

        # Jupiter's g5 for the eccentricity vector, the invariable-plane offset for the
        # slow inclination modes — the basis validated by hand against 10 Myr runs.
        assert list(basis_e) == ['g5', 'g6']
        assert list(basis_i) == ['s6', 'const']
        assert set(dropped_e) == {'g7', 'g8', 'const'}
        assert set(dropped_i) == {'s7', 's8'}

    def test_nothing_is_dropped_when_the_baseline_resolves_everything(self):
        basis, dropped, _ = choose_basis(candidate_frequencies('e'), 1e7)

        assert dropped == []
        assert set(basis) == set(candidate_frequencies('e'))


class TestRemoveForced:
    def test_a_pure_forced_signal_is_removed_entirely(self, times):
        z = _series(times, [(0.1, PLANETARY_FREQUENCIES['g5'])])

        free, forced, coefficients = remove_forced(z, times, {'g5': PLANETARY_FREQUENCIES['g5']})

        assert np.abs(free).max() < 1e-6
        assert abs(coefficients['g5']) == pytest.approx(0.1, rel=1e-6)
        np.testing.assert_allclose(forced, z, atol=1e-6)

    def test_a_free_term_survives_untouched(self, times):
        # 10 arcsec/yr is 18 Rayleigh widths from g5 on a 4 Myr baseline, so the fit
        # cannot absorb it.
        z = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5']), (0.05, 10.0)])

        free, _, _ = remove_forced(z, times, candidate_frequencies('e'))

        assert np.abs(free).mean() == pytest.approx(0.05, rel=0.02)

    def test_no_frequencies_leaves_the_series_alone(self, times):
        z = _series(times, [(0.05, 10.0)])

        free, forced, coefficients = remove_forced(z, times, {})

        assert coefficients == {}
        np.testing.assert_allclose(free, z)
        np.testing.assert_allclose(forced, 0.0)


class TestRate:
    def test_recovers_the_rate_of_a_single_rotating_vector(self, times):
        z = _series(times, [(0.05, 7.5)])

        assert rate(times, np.unwrap(np.angle(z))) == pytest.approx(7.5, rel=1e-6)

    def test_a_retrograde_vector_gives_a_negative_rate(self, times):
        z = _series(times, [(0.05, -7.5)])

        assert rate(times, np.unwrap(np.angle(z))) == pytest.approx(-7.5, rel=1e-6)


class TestMaskOrigin:
    def test_samples_near_the_origin_are_dropped(self):
        amplitude = np.array([1.0, 1.0, 1.0, 0.01, 1.0])
        z = amplitude * np.exp(1j * np.linspace(0, 1, 5))

        keep = mask_origin(z, quantile=0.15)

        assert list(keep) == [True, True, True, False, True]

    def test_a_uniform_circle_keeps_everything(self):
        z = np.exp(1j * np.linspace(0, 10, 100))

        assert mask_origin(z).all()

    def test_a_zero_vector_has_no_argument(self):
        assert not mask_origin(np.zeros(10, dtype=complex)).any()
        assert not mask_origin(np.zeros(0, dtype=complex)).any()


class TestDecimationSteps:
    def test_a_small_factor_is_applied_in_one_pass(self):
        assert decimation_steps(5) == [5]

    def test_a_large_factor_is_cascaded_and_the_product_is_exact(self):
        # scipy's FIR design degrades past a factor of about 10 per pass.
        steps = decimation_steps(500)

        assert all(step <= 10 for step in steps)
        assert int(np.prod(steps)) == 500

    def test_no_decimation_below_two(self):
        assert decimation_steps(1) == []

    @pytest.mark.parametrize(
        'factor, expected',
        [
            # worked by hand: 31 and 37 are primes above 10, 62 = 2 x 31, 148 = 4 x 37, 100 = 10 x 10
            (31, [31]),
            (62, [2, 31]),
            (100, [10, 10]),
            (148, [4, 37]),
        ],
    )
    def test_every_factor_is_filtered_and_the_product_is_exact(self, factor, expected):
        # 31 is the factor of a 15.918-yr record resampled to 500 yr (555527): before W28 it gave [] (no filter).
        steps = decimation_steps(factor)

        assert steps == expected
        assert int(np.prod(steps)) == factor
        assert has_remainder(steps) == (factor in (31, 62, 148))

    @pytest.mark.parametrize('factor', [31, 62, 100, 148])
    def test_the_thinned_grid_is_times_every_factor(self, factor):
        # A slow vector (period 50 kyr) passes the filter unchanged, so the resampled value at sample k is the
        # vector at times[::factor][k]: a shifted or stretched grid would show as a phase error of
        # 2 pi * shift / 50 kyr (one input step of 15.9 yr is already 2e-3 rad). The middle half of the record is
        # read: near the ends the filters see padding (next test).
        dt = 15.918
        times = np.arange(0.0, 200_000.0, dt)
        z = 0.05 * np.exp(2j * np.pi * times / 50_000.0)

        steps = decimation_steps(factor)
        resampled = _resample(z, steps)
        grid = times[::factor][: len(resampled)]

        assert len(resampled) == len(grid) == int(np.ceil(len(times) / factor))
        quarter = len(grid) // 4
        inner = slice(quarter, -quarter)  # the FIR cascades' edge transients reach ~20 coarse samples in
        truth = 0.05 * np.exp(2j * np.pi * grid / 50_000.0)
        assert np.max(np.abs(np.angle(resampled[inner] / truth[inner]))) < 2e-4
        # gain: the FIR passband of two 10-passes costs 0.2 % at this period (factor 100, as before W28)
        assert np.max(np.abs(np.abs(resampled[inner]) / 0.05 - 1.0)) < 5e-3

    @pytest.mark.parametrize('factor', [5, 64, 100, 500])
    def test_factors_of_passes_up_to_ten_are_bit_identical_to_the_fir_cascade(self, factor):
        # The polyphase pass is only for a leftover above 10: a factor made of passes <= 10 is resampled exactly as
        # before (scipy's zero-phase FIR decimate, pass by pass), so no record whose factor was handled moves.
        rng = np.random.default_rng(7)
        z = rng.normal(size=4000) + 1j * rng.normal(size=4000)
        steps = decimation_steps(factor)
        expected = z
        for step in steps:
            expected = decimate(expected.real, step, ftype='fir', zero_phase=True) + 1j * decimate(
                expected.imag, step, ftype='fir', zero_phase=True
            )

        assert not has_remainder(steps)
        assert np.array_equal(_resample(z, steps), expected)


class TestRemainderAntiAlias:
    """555527 (g09-1): at a 15.918-yr step the factor to 500 yr is 31; before W28 it skipped the filter, Jupiter's
    11.86-yr term folded into a ~46-yr alias (|1/11.862 - 1/15.918| = 1/46.6 yr) and, where it is larger than the free
    circle, it wound the free varpi by whole turns: steps."""

    DT = 15.918
    RATE = 50.0  # arcsec/yr of the free vector, chosen by hand (not a planetary mode)

    def _elements(self):
        times = np.arange(0.0, 100_000.0, self.DT)
        z_e = _series(times, [(0.01, self.RATE), (0.03, PLANETARY_FREQUENCIES['g5'])])
        z_e = z_e + 0.3 * np.abs(z_e) * np.exp(2j * np.pi * times / 11.862)  # ripple 0.3 |z| > |z_free| = 0.01
        z_i = _series(times, [(0.1, -20.0)])
        return times, _elements_from_vectors(z_e, z_i)

    def test_the_ripple_leaves_no_step_in_the_free_varpi(self):
        times, elements = self._elements()

        result = free_elements(times, *elements)

        assert result.resample_steps == (31,)
        assert result.resample_remainder
        assert np.median(np.diff(result.times)) == pytest.approx(31 * self.DT)
        t = result.times[result.keep_e]
        excess = (np.diff(result.varpi_free) - self.RATE * RAD_PER_ARCSEC * np.diff(t)) / (2.0 * np.pi)
        assert np.max(np.abs(excess)) < 0.25  # turns per sample; unfiltered, 0.50 (a half-turn ambiguity per sample)

    def test_the_free_rate_is_recovered_within_the_resolution(self):
        times, elements = self._elements()

        result = free_elements(times, *elements)

        # unfiltered this came out at 79 arcsec/yr, two resolutions (13 arcsec/yr at 100 kyr) away
        assert abs(result.b_varpi - self.RATE) < result.resolution / 10.0

    def test_the_record_ends_carry_no_larger_transient_than_the_fir_cascade(self):
        # padtype='line' pads along the line from the first to the last sample, not with zeros. Measured against the
        # clean vector: from the third sample on, the error is < 0.1 |z_free|; the first two samples carry an edge
        # transient, smaller than the FIR cascade's at the neighbouring factor 30 = 10 x 3 (the path every composite
        # factor has always taken: 1.8 |z_free| at the first sample against 0.6).
        dt = self.DT
        times = np.arange(0.0, 100_000.0, dt)
        clean = _series(times, [(0.01, self.RATE), (0.03, PLANETARY_FREQUENCIES['g5'])])
        rippled = clean + 0.3 * np.abs(clean) * np.exp(2j * np.pi * times / 11.862)

        def error(factor):
            resampled = _resample(rippled, decimation_steps(factor))
            grid = times[::factor][: len(resampled)]
            return np.abs(resampled - _series(grid, [(0.01, self.RATE), (0.03, PLANETARY_FREQUENCIES['g5'])])) / 0.01

        polyphase, fir = error(31), error(30)
        assert np.max(polyphase[2:-2]) < 0.1
        assert polyphase[0] < fir[0] and polyphase[-1] < fir[-1]

    def test_no_resampling_records_no_steps(self):
        times, elements = self._elements()

        result = free_elements(times, *elements, sampling_years=None)

        assert result.resample_steps == ()
        assert not result.resample_remainder


class TestFreeElements:
    def test_equal_free_rates_make_omega_librate(self, times):
        # varpi_free and Omega_free turning at the same rate is exactly what a Lidov-Kozai
        # lock does: omega = varpi - Omega then stands still.
        z_e = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5']), (0.20, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -12.0)])

        result = free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)

        assert result.librates is True
        assert result.revolutions == pytest.approx(0.0, abs=0.05)
        assert result.b_omega == pytest.approx(0.0, abs=0.05)

    def test_different_free_rates_make_omega_circulate(self, times):
        # g - s = -12 - (-20) = 8 arcsec/yr over 4 Myr is 8 * 4e6 / 1296000 = 24.7 turns.
        z_e = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5']), (0.20, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -20.0)])

        result = free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)

        assert result.librates is False
        assert result.b_varpi == pytest.approx(-12.0, rel=1e-2)
        assert result.b_Omega == pytest.approx(-20.0, rel=1e-2)
        assert result.b_omega == pytest.approx(8.0, rel=1e-2)
        assert result.revolutions == pytest.approx(8.0 * 4e6 / ARCSEC_PER_TURN, rel=1e-2)

    def test_the_three_slopes_stay_consistent(self, times):
        z_e = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5']), (0.20, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -20.0)])

        result = free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)

        # Not an identity — the three are fitted on three grids — but a large gap would
        # mean the two masks disagree badly enough that the rates are incomparable.
        assert result.slope_consistency < 0.1

    def test_omega_free_lives_on_the_intersection_of_both_masks(self, times):
        z_e = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5']), (0.20, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -20.0)])

        result = free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)

        assert len(result.omega_free) == int((result.keep_e & result.keep_i).sum())
        assert len(result.varpi_free) == int(result.keep_e.sum())
        assert len(result.omega_times) == len(result.omega_free)

    def test_the_mask_drops_samples_whose_argument_is_undefined(self):
        # A perfect librator — varpi_free and Omega_free both turn at -12 arcsec/yr, so
        # omega_free stands still — with a scattering of samples pushed to within a
        # ten-thousandth of the origin and given arbitrary directions. There the argument
        # is not a measurement, and unwrapping through it accumulates turns the dynamics
        # never made. Their neighbours are untouched, so dropping them restores continuity.
        times = np.arange(0.0, 1e6, 200.0)
        rng = np.random.default_rng(0)
        forced = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5'])])
        free = _series(times, [(0.20, -12.0)])
        corrupt = np.concatenate([np.arange(start, start + 20) for start in range(50, times.size - 20, 120)])
        free[corrupt] = 1e-4 * np.exp(2j * np.pi * rng.random(corrupt.size))
        elements = _elements_from_vectors(forced + free, _series(times, [(0.30, -12.0)]))

        masked = free_elements(times, *elements, sampling_years=None)
        unmasked = free_elements(times, *elements, sampling_years=None, mask_quantile=0.0)

        assert masked.mask_fraction_e == pytest.approx(corrupt.size / times.size, rel=0.05)
        # Unmasked, the walk runs past a full turn and the librator is reported as
        # circulating; masked, the span is a few degrees and the verdict survives.
        assert unmasked.librates is False
        assert masked.omega_range < np.radians(30.0)
        assert masked.librates is True

    def test_resampling_does_not_move_the_answer(self, times):
        z_e = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5']), (0.20, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -20.0)])
        elements = _elements_from_vectors(z_e, z_i)

        raw = free_elements(times, *elements, sampling_years=None)
        resampled = free_elements(times, *elements, sampling_years=1000.0)

        assert len(resampled.times) < len(raw.times)
        assert resampled.b_omega == pytest.approx(raw.b_omega, rel=1e-2)
        assert resampled.librates == raw.librates

    def test_truncation_changes_the_basis_but_not_the_verdict(self, times):
        z_e = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5']), (0.20, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -20.0)])
        elements = _elements_from_vectors(z_e, z_i)
        short = times <= 5e5

        full = free_elements(times, *elements, sampling_years=None)
        cut = free_elements(times[short], *(array[short] for array in elements), sampling_years=None)

        assert len(cut.basis_e) < len(full.basis_e)
        assert cut.librates == full.librates is False
        assert cut.b_omega == pytest.approx(full.b_omega, rel=0.05)

    def test_rho_is_above_one_when_the_free_circle_encloses_the_origin(self, times):
        z_e = _series(times, [(0.05, PLANETARY_FREQUENCIES['g5']), (0.30, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -12.0)])

        result = free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)

        assert result.rho > 1.0
        low, high = result.rho_envelope()
        assert low <= result.rho <= high or np.isclose(low, high)

    def test_rho_is_below_one_for_a_carousel(self, times):
        # Forced part six times the free one: the cloud never reaches the origin, and the
        # osculating pericentre is pinned whatever the dynamics.
        z_e = _series(times, [(0.30, PLANETARY_FREQUENCIES['g5']), (0.05, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -20.0)])

        result = free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)

        assert result.rho < 1.0

    def test_too_few_samples_returns_nothing(self):
        short = np.arange(0.0, 10.0)

        assert free_elements(short, *(np.full(10, 0.1),) * 4, sampling_years=None) is None


class TestOutputs:
    def test_missing_elements_still_produce_every_column(self):
        fields = summary_fields(None)

        assert set(fields) == set(SUMMARY_COLUMNS)
        assert all(value is None for value in fields.values())
        assert series_rows(None) is None

    def test_columns_match_the_declared_names(self, times):
        z_e = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5']), (0.20, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -12.0)])

        fields = summary_fields(free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None))

        assert set(fields) == set(SUMMARY_COLUMNS)
        assert fields['free_basis_e'] == 'const+g8+g7+g5+g6'
        assert fields['free_omega_range_deg'] < 360.0

    def test_series_rows_are_all_the_same_length_on_the_full_grid(self, times):
        z_e = _series(times, [(0.10, PLANETARY_FREQUENCIES['g5']), (0.20, -12.0)])
        z_i = _series(times, [(0.05, PLANETARY_FREQUENCIES['s6']), (0.30, -20.0)])
        elements = free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)

        columns = series_rows(elements)

        assert {len(values) for values in columns.values()} == {len(elements.times)}
        # Masked samples carry NaN in the angle columns rather than being dropped, so the
        # file stays aligned with the times column.
        assert np.isnan(columns['omega_free_unwrapped']).sum() == int((~(elements.keep_e & elements.keep_i)).sum())


class TestRemainderRuleFlag:
    """The rule free_resample_remainder (`analysis.rules`) names a ZLK row whose free elements were resampled with a
    polyphase leftover pass; it changes no status."""

    @staticmethod
    def _zlk(dt):
        from resonances.resonance.analysis.secular import LongPeriodSeries
        from resonances.resonance.analysis.settings import settings_from_config
        from resonances.resonance.analysis.verdict import ResonanceType
        from resonances.resonance.analysis.zlk import analyse_zlk

        t = np.arange(0.0, 60_000.0, dt)
        z_e = _series(t, [(0.03, PLANETARY_FREQUENCIES['g5']), (0.1, 40.0)])
        Omega = -30.0 * RAD_PER_ARCSEC * t
        e, varpi = np.abs(z_e), np.angle(z_e)
        inc = np.full_like(t, np.radians(30.0))
        omega = np.unwrap(varpi - Omega)
        series = LongPeriodSeries(
            ResonanceType.LIDOV_KOZAI, 'LK', 'synthetic', t, omega, np.full_like(t, 2.5), e, inc, varpi=np.unwrap(varpi), Omega=Omega
        )
        elements = free_elements(t, e, inc, Omega, omega)
        return analyse_zlk(series, settings_from_config(), free_elements=elements), elements

    def test_a_factor_of_31_fires_the_rule(self):
        analysis, elements = self._zlk(15.918)

        assert elements.resample_steps == (31,)
        assert 'free_resample_remainder' in analysis.verdict.rules_fired

    def test_a_factor_of_passes_up_to_ten_does_not(self):
        analysis, elements = self._zlk(10.0)  # 500 / 10 = 50 = 10 x 5

        assert elements.resample_steps == (10, 5)
        assert 'free_resample_remainder' not in analysis.verdict.rules_fired


class TestRhoI:
    """FreeElements.rho_i_local_max (W28 D8, g09-7): the free inclination over the turning forced one, the constant
    (s5 = 0, the reference plane's tilt) left out. Expected ratios are the amplitudes put into the signal."""

    @staticmethod
    def _split(times, inclination_terms):
        z_e = _series(times, [(0.03, PLANETARY_FREQUENCIES['g5']), (0.10, 40.0)])
        z_i = _series(times, inclination_terms)
        return free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)

    @pytest.mark.parametrize('free, forced, expected', [(0.05, 0.01, 5.0), (0.01, 0.02, 0.5)])
    def test_the_ratio_is_the_free_over_the_turning_forced_amplitude(self, times, free, forced, expected):
        # a constant of 0.02 sits in the signal as well: counted with the forced part it would give 0.05 / 0.03 or less
        result = self._split(times, [(0.02, 0.0), (forced, PLANETARY_FREQUENCIES['s6']), (free, -50.0)])

        assert 's6' in result.basis_i and 'const' in result.basis_i
        assert result.rho_i_local_max() == pytest.approx(expected, rel=0.02)  # 5 +- 0.1
        assert result.rho_i_local_max(period_yr=30_000.0) == pytest.approx(expected, rel=0.02)

    def test_a_constant_only_forced_part_has_no_ratio(self):
        # over 80 kyr 2/T = 32.4 arcsec/yr chains s6 (-26.35) through s7 and s8 to the constant: one cluster, the
        # constant fitted alone, nothing turning to compare with (373136 over 100 kyr)
        times = np.arange(0.0, 80_000.0, 50.0)
        result = self._split(times, [(0.02, 0.0), (0.05, -50.0)])

        assert list(result.basis_i) == ['const']
        assert result.forced_i_turning() is None
        assert np.isnan(result.rho_i_local_max(period_yr=20_000.0))


class TestNodeNear:
    """A planetary residual whose only match is the free node within a Rayleigh width of s_j (1981 Midas: s6, d 1.85 <
    2.59 arcsec/yr over 500 kyr): `omega_free_gate.node_near` and the columns of `diagnostics.node_near`."""

    @staticmethod
    def _split(node_rate):
        # 500 kyr: resolution 1296000 / 500000 = 2.592 arcsec/yr; varpi and Omega turn together, so omega librates
        times = np.arange(0.0, 500_000.0, 100.0)
        z_e = _series(times, [(0.03, PLANETARY_FREQUENCIES['g5']), (0.20, node_rate)])
        z_i = _series(times, [(0.01, 0.0), (0.30, node_rate)])
        return free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)

    @staticmethod
    def _columns(elements, period=20_000.0):
        from types import SimpleNamespace

        from resonances.resonance import omega_free_gate
        from resonances.resonance.analysis.secular.models import GateResult
        from resonances.resonance.analysis.settings import settings_from_config
        from resonances.resonance.analysis.zlk import diagnostics

        config = SimpleNamespace(free_gate_enabled=True, free_gate_min_cycles=3.0)
        verdict = omega_free_gate.evaluate_gate(elements, period, config)
        gate = GateResult(outcome=str(verdict.outcome), matches=tuple(verdict.matches), libration_period_yr=period)
        return verdict, diagnostics.node_near(elements, gate, settings_from_config()).to_columns()

    def test_a_node_within_a_width_of_s6_is_named(self):
        elements = self._split(PLANETARY_FREQUENCIES['s6'] - 1.85)
        verdict, columns = self._columns(elements)

        assert elements.librates
        assert str(verdict.outcome) == 'planetary_residual'
        assert all(match.startswith('b_Omega~s6') for match in verdict.matches)
        assert columns['zlk_gate_node_near'] == 's6'
        distance = abs(elements.b_Omega - PLANETARY_FREQUENCIES['s6'])
        assert columns['zlk_gate_node_near_d_arcsec_yr'] == pytest.approx(distance)
        assert columns['zlk_gate_node_near_resolution_arcsec_yr'] == pytest.approx(2.592, rel=1e-3)
        assert distance < 2.592
        assert columns['zlk_gate_node_separation_yr'] == pytest.approx(1_296_000.0 / distance)

    def test_a_node_far_from_every_s_mode_is_not(self):
        elements = self._split(-40.0)  # 13.65 arcsec/yr from s6, five widths
        verdict, columns = self._columns(elements)

        assert str(verdict.outcome) == 'confirmed'
        assert columns['zlk_gate_node_near'] == ''
        assert np.isnan(columns['zlk_gate_node_near_d_arcsec_yr'])
        assert np.isfinite(columns['free_rho_i_max'])

    def test_an_eccentricity_match_is_not_a_node_near(self):
        # b_varpi on g5 as well: the residual is not the node alone
        times = np.arange(0.0, 500_000.0, 100.0)
        z_e = _series(times, [(0.20, PLANETARY_FREQUENCIES['g5'] + 1.0)])
        z_i = _series(times, [(0.01, 0.0), (0.30, PLANETARY_FREQUENCIES['s6'] - 1.0)])
        elements = free_elements(*(times,) + _elements_from_vectors(z_e, z_i), sampling_years=None)
        from resonances.resonance import omega_free_gate

        mode, distance = omega_free_gate.node_near(elements)
        assert mode == '' and np.isnan(distance)
