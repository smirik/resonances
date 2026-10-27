"""Wilson score intervals and status shares.

Reference values: Newcombe (1998, Statistics in Medicine 17, 857, Table I, method 3 = Wilson without continuity
correction) and scipy's binomtest(...).proportion_ci(method='wilson'), a separate implementation."""

import pytest
from scipy.stats import binomtest

from resonances.clones import ensemble_row, status_shares, wilson_interval


@pytest.mark.parametrize(
    'k, n, low, high',
    [
        (81, 263, 0.2553, 0.3662),  # Newcombe 1998, Table I
        (15, 148, 0.0624, 0.1605),
        (0, 20, 0.0000, 0.1611),
        (1, 29, 0.0061, 0.1718),
    ],
)
def test_wilson_against_newcombe_table(k, n, low, high):
    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(low, abs=6e-5)
    assert hi == pytest.approx(high, abs=6e-5)


@pytest.mark.parametrize('k, n', [(0, 10), (5, 10), (10, 10), (0, 100), (50, 100), (90, 100), (96, 100), (200, 400), (3, 1000)])
def test_wilson_against_scipy(k, n):
    reference = binomtest(k, n).proportion_ci(confidence_level=0.95, method='wilson')
    lo, hi = wilson_interval(k, n)
    assert lo == pytest.approx(reference.low, abs=1e-9)
    assert hi == pytest.approx(reference.high, abs=1e-9)


def test_wilson_half_widths_of_the_documented_clone_counts():
    """docs/clones.md: at p = 0.5 the 95 % half-width is +-10 % for 100 clones, +-5 % for 400, +-3 % for 1000; 0/N has
    an upper bound of 0.037 for N = 100."""
    for n, half in ((100, 0.0963), (400, 0.0488), (1000, 0.0309)):
        lo, hi = wilson_interval(n // 2, n)
        assert (hi - lo) / 2 == pytest.approx(half, abs=2e-4)
    assert wilson_interval(0, 100)[1] == pytest.approx(0.0370, abs=1e-4)
    assert wilson_interval(0, 0) != wilson_interval(0, 0)  # nan, nan


def test_status_shares_and_nominal_agreement():
    statuses = ['resonant'] * 6 + ['transient'] * 3 + ['non_resonant'] + [None]
    row = status_shares(statuses)
    assert row['n_classified'] == 10 and row['n_missing'] == 1
    assert row['n_resonant'] == 6 and row['f_resonant'] == 0.6
    assert row['f_near_separatrix'] == 0 and row['f_near_separatrix_hi'] == pytest.approx(wilson_interval(0, 10)[1])
    assert (row['f_transient_lo'], row['f_transient_hi']) == wilson_interval(3, 10)
    assert row['modal_status'] == 'resonant' and row['modal_share'] == 0.6
    agreement = ensemble_row(statuses, nominal_status='transient')
    assert agreement['nominal_agreement'] == pytest.approx(0.3)
    assert (agreement['nominal_agreement_lo'], agreement['nominal_agreement_hi']) == wilson_interval(3, 10)
    with pytest.raises(ValueError):
        status_shares(['libration'])
