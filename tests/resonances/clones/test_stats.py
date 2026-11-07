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


def test_cloud_row_counts_only_the_bound_draws():
    """4 draws, 1 of them unbound: the shares are over the 3 bound clones (2 resonant, 1 transient), worked by hand:
    f_resonant = 2/3, f_transient = 1/3; the Wilson bounds are those of 2/3 (scipy, a separate implementation);
    f_draw_invalid = 1/4. With CLONES_MAX_INVALID = 0.3 the cloud is usable."""
    from resonances.clones import cloud_row

    row = cloud_row(['resonant', 'transient', 'resonant'], 'resonant', n_clones=4, n_draw_invalid=1, max_invalid=0.3)
    reference = binomtest(2, 3).proportion_ci(confidence_level=0.95, method='wilson')
    assert row['n_classified'] == 3 and row['n_draw_invalid'] == 1 and row['cloud_usable'] is True
    assert row['f_resonant'] == pytest.approx(2 / 3) and row['f_transient'] == pytest.approx(1 / 3)
    assert row['f_invalid'] == 0.0  # an unbound draw is no 'invalid' verdict
    assert row['f_draw_invalid'] == pytest.approx(0.25)
    assert (row['f_resonant_lo'], row['f_resonant_hi']) == pytest.approx((reference.low, reference.high), abs=1e-9)
    assert row['nominal_agreement'] == pytest.approx(2 / 3) and row['modal_status'] == 'resonant'


def test_cloud_row_without_a_usable_cloud_has_no_shares():
    """83 of 100 draws unbound (2010 DH65 from SBDB, a 1-day arc): above CLONES_MAX_INVALID = 0.05 the row keeps its
    counts (17 classified, all non-resonant) but no share, no modal status and no nominal agreement."""
    import math

    from resonances.clones import cloud_row

    row = cloud_row(['non_resonant'] * 17, 'transient', n_clones=100, n_draw_invalid=83, max_invalid=0.05)
    assert row['cloud_usable'] is False and row['n_classified'] == 17 and row['n_non_resonant'] == 17
    assert row['f_draw_invalid'] == pytest.approx(0.83)
    shares = [value for key, value in row.items() if key.startswith('f_') and key != 'f_draw_invalid']
    assert shares and all(math.isnan(value) for value in shares)
    assert row['modal_status'] == '' and math.isnan(row['modal_share']) and math.isnan(row['nominal_agreement'])


def test_cloud_row_at_the_limit_is_usable_and_all_unbound_is_not():
    """5 of 100 unbound = 0.05: at the limit, usable; 100 of 100: no bound draw, unusable."""
    from resonances.clones import cloud_row

    assert cloud_row(['resonant'] * 95, None, 100, 5, 0.05)['cloud_usable'] is True
    assert cloud_row([], None, 100, 100, 0.05)['cloud_usable'] is False


def test_an_old_row_is_read_in_the_current_semantics():
    """A clones.csv row written before 2026-11-06 for 2010 DH65: 100 draws, 83 unbound counted as 'invalid', the 17 bound
    ones non-resonant. Read back: 17 classified, n_invalid 0, no shares (83 % > 5 %). A current row is untouched."""
    import math

    from resonances.clones import upgrade_row

    old = {'n_clones': 100, 'n_draw_invalid': 83, 'n_missing': 0, 'n_invalid': 83, 'n_non_resonant': 17, 'nominal_status': 'transient'}
    old |= {f'n_{s}': 0 for s in ('resonant', 'transient', 'near_separatrix', 'uncertain')}
    row = upgrade_row(old, 0.05)
    assert row['n_classified'] == 17 and row['n_invalid'] == 0 and row['n_non_resonant'] == 17
    assert row['cloud_usable'] is False and math.isnan(row['f_non_resonant'])
    current = dict(row)
    assert upgrade_row(current, 0.05) is current
    usable = upgrade_row({**old, 'n_draw_invalid': 0, 'n_invalid': 0, 'n_non_resonant': 100}, 0.05)
    assert usable['cloud_usable'] is True and usable['f_non_resonant'] == 1.0
