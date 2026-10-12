import pytest

from resonances.data.const import PLANETARY_FREQUENCIES, SECULAR_FORMULAS
from resonances.secular.combinations import (
    PLANETARY_MODES,
    coefficient_key,
    formula_from_coefficients,
    near_resonant,
)
from resonances.secular.secular_resonance import SecularResonance

F = PLANETARY_FREQUENCIES
NO_WINDOW = {2: 1e9}


def _is_admissible(key):
    coeffs = dict(key)
    s_sum = sum(c for label, c in coeffs.items() if label.startswith('s'))
    return sum(coeffs.values()) == 0 and s_sum % 2 == 0


def test_coefficient_key_ignores_spelling_and_sign():
    z1 = coefficient_key('g-g6+s-s6')
    assert z1 == coefficient_key('g+s-s6-g6')
    assert z1 == coefficient_key('-g-s+g6+s6')
    assert z1 == (('g', 1), ('g6', -1), ('s', 1), ('s6', -1))
    assert coefficient_key('-g+s+g5-s7') == coefficient_key('g-s-g5+s7')
    assert coefficient_key('2(g-g6)+(s-s6)') == coefficient_key('2g-2g6+s-s6')
    assert coefficient_key(SecularResonance('g-g6')) == (('g', 1), ('g6', -1))
    assert coefficient_key('g-g6') != coefficient_key('g-g5')


def test_formula_from_coefficients():
    assert formula_from_coefficients(1, 1, {'g6': -1, 's6': -1}) == 'g+s-g6-s6'
    assert formula_from_coefficients(2, -2, {}) == '2g-2s'
    assert formula_from_coefficients(0, 1, {'s4': -1}) == 's-s4'
    assert formula_from_coefficients(1, 0, {'g5': 1, 'g6': -2}) == 'g+g5-2g6'
    # the formula reads back as the same angle
    assert coefficient_key('g+g5-2g6') == coefficient_key('g-2g6+g5')


def test_order_two_is_exactly_the_fifteen_linear_resonances():
    # Counted by hand: g-g_j for j = 1..8 and s-s_j for j = 1..8 without s5. Nothing else of order 2
    # obeys d'Alembert: g-s has an odd number of node terms, g+s and 2g need more terms to sum to zero.
    found = near_resonant(30.0, -30.0, max_order=2, thresholds=NO_WINDOW)
    expected = {f'g-g{j}' for j in range(1, 9)} | {f's-s{j}' for j in range(1, 9) if j != 5}
    assert {c.formula for c in found} == expected
    assert all(c.order == 2 for c in found)


def test_divisors_near_nu6_and_z1():
    # A body 0.1 arcsec/yr above g6 and 0.2 below s6: the divisors follow by hand.
    g, s = F['g6'] + 0.1, F['s6'] - 0.2
    found = {c.formula: c for c in near_resonant(g, s, max_order=4, thresholds={2: 0.5, 4: 0.5})}
    assert found['g-g6'].divisor == pytest.approx(0.1)
    assert found['s-s6'].divisor == pytest.approx(-0.2)
    assert found['g+s-g6-s6'].divisor == pytest.approx(-0.1)
    assert found['g+s-g6-s6'].order == 4
    for c in found.values():
        assert abs(c.divisor) < 0.5
        assert _is_admissible(c.key)


def test_window_and_modes_restrict_the_result():
    g, s = F['g6'] + 0.1, F['s6'] - 0.2
    narrow = {c.formula for c in near_resonant(g, s, max_order=4, thresholds={2: 0.15, 4: 0.15})}
    assert 'g-g6' in narrow and 'g+s-g6-s6' in narrow and 's-s6' not in narrow
    outer = near_resonant(g, s, max_order=4, thresholds={2: 5.0, 4: 5.0}, modes=('g5', 'g6', 's6'))
    assert all(set(dict(c.key)) <= {'g', 's', 'g5', 'g6', 's6'} for c in outer)


def test_every_catalogue_angle_up_to_order_six_is_enumerated():
    # The catalogue comes from the literature; the enumerator from the d'Alembert rules alone.
    # Every catalogue angle must obey the rules and appear when the window is unlimited.
    found = {c.key for c in near_resonant(30.0, -30.0, max_order=6, thresholds=NO_WINDOW)}
    for formula in SECULAR_FORMULAS:
        resonance = SecularResonance(formula)
        if resonance.order() > 6:
            continue
        key = coefficient_key(resonance)
        assert _is_admissible(key), formula
        assert key in found, formula


def test_every_result_is_admissible_and_unique():
    found = near_resonant(32.0, -34.0, max_order=6)
    keys = [c.key for c in found]
    assert len(keys) == len(set(keys))
    for c in found:
        assert _is_admissible(c.key)
        assert c.order == sum(abs(v) for _, v in c.key)
        # the stored divisor is the one of the formula itself
        coeffs = dict(c.key)
        divisor = coeffs.get('g', 0) * 32.0 + coeffs.get('s', 0) * -34.0
        divisor += sum(v * F[label] for label, v in coeffs.items() if label in PLANETARY_MODES)
        assert c.divisor == pytest.approx(divisor)


def test_missing_frequency_raises():
    with pytest.raises(ValueError, match='Missing planetary frequencies'):
        near_resonant(30.0, -30.0, max_order=2, frequencies={'g5': 4.26, 'g6': 28.25})
