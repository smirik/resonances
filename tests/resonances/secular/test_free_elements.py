"""Tests for the forced/free decomposition of the eccentricity and inclination vectors.

The synthetic cases are built the other way round from the code under test: a signal is
assembled from a forced term at a *known* planetary frequency plus a free term at a
frequency chosen by hand, and the test checks that the decomposition recovers the free
term's amplitude and rate. Nothing here is compared against the implementation's own
intermediate results.
"""

import numpy as np
import pytest

from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.secular.free_elements import (
    ARCSEC_PER_TURN,
    RAD_PER_ARCSEC,
    SUMMARY_COLUMNS,
    candidate_frequencies,
    choose_basis,
    cluster,
    decimation_steps,
    free_elements,
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
