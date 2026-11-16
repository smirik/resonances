import pytest

from resonances.secular.secular_resonance_formula import SecularResonanceFormula


SECULAR_RESONANCES = {
    # Linear
    'g-g5': [
        ('g', 1, None),
        ('g', -1, 5),
    ],
    'g-g6': [
        ('g', 1, None),
        ('g', -1, 6),
    ],
    's-s6': [
        ('s', 1, None),
        ('s', -1, 6),
    ],
    's-s7': [
        ('s', 1, None),
        ('s', -1, 7),
    ],
    # Degree 4
    'g5-g6': [
        ('g', 1, 5),
        ('g', -1, 6),
    ],
    's7-s6': [
        ('s', 1, 7),
        ('s', -1, 6),
    ],
    'g+s-s7-g5': [
        ('g', 1, None),
        ('s', 1, None),
        ('s', -1, 7),
        ('g', -1, 5),
    ],
    'g+s-s7-g6': [
        ('g', 1, None),
        ('s', 1, None),
        ('s', -1, 7),
        ('g', -1, 6),
    ],
    'g+s-s6-g5': [
        ('g', 1, None),
        ('s', 1, None),
        ('s', -1, 6),
        ('g', -1, 5),
    ],
    'g+s-s6-g6': [
        ('g', 1, None),
        ('s', 1, None),
        ('s', -1, 6),
        ('g', -1, 6),
    ],
    '2g-2s': [
        ('g', 2, None),
        ('s', -2, None),
    ],
    'g-2g5+g6': [
        ('g', 1, None),
        ('g', -2, 5),
        ('g', 1, 6),
    ],
    'g+g5-2g6': [
        ('g', 1, None),
        ('g', 1, 5),
        ('g', -2, 6),
    ],
    '2g-g5-g6': [
        ('g', 2, None),
        ('g', -1, 5),
        ('g', -1, 6),
    ],
    '-g+s+g5-s7': [
        ('g', -1, None),
        ('s', 1, None),
        ('g', 1, 5),
        ('s', -1, 7),
    ],
    '-g+s+g6-s7': [
        ('g', -1, None),
        ('s', 1, None),
        ('g', 1, 6),
        ('s', -1, 7),
    ],
    '-g+s+g5-s6': [
        ('g', -1, None),
        ('s', 1, None),
        ('g', 1, 5),
        ('s', -1, 6),
    ],
    '-g+s+g6-s6': [
        ('g', -1, None),
        ('s', 1, None),
        ('g', 1, 6),
        ('s', -1, 6),
    ],
    'g-g5+s7-s6': [
        ('g', 1, None),
        ('g', -1, 5),
        ('s', 1, 7),
        ('s', -1, 6),
    ],
    'g-g5-s7+s6': [
        ('g', 1, None),
        ('g', -1, 5),
        ('s', -1, 7),
        ('s', 1, 6),
    ],
    'g-g6+s7-s6': [
        ('g', 1, None),
        ('g', -1, 6),
        ('s', 1, 7),
        ('s', -1, 6),
    ],
    'g-g6-s7+s6': [
        ('g', 1, None),
        ('g', -1, 6),
        ('s', -1, 7),
        ('s', 1, 6),
    ],
    '2g-s-s7': [
        ('g', 2, None),
        ('s', -1, None),
        ('s', -1, 7),
    ],
    '2g-s-s6': [
        ('g', 2, None),
        ('s', -1, None),
        ('s', -1, 6),
    ],
    '-g+2s-g5': [
        ('g', -1, None),
        ('s', 2, None),
        ('g', -1, 5),
    ],
    '-g+2s-g6': [
        ('g', -1, None),
        ('s', 2, None),
        ('g', -1, 6),
    ],
    '2g-2s7': [
        ('g', 2, None),
        ('s', -2, 7),
    ],
    '2g-2s6': [
        ('g', 2, None),
        ('s', -2, 6),
    ],
    '2g-s7-s6': [
        ('g', 2, None),
        ('s', -1, 7),
        ('s', -1, 6),
    ],
    # Degree 6 & parentheses
    'g-2g6+g7': [
        ('g', 1, None),
        ('g', -2, 6),
        ('g', 1, 7),
    ],
    'g-3g6+2g5': [
        ('g', 1, None),
        ('g', -3, 6),
        ('g', 2, 5),
    ],
    '2(g-g6)+(s-s6)': [
        ('g', 2, None),
        ('g', -2, 6),
        ('s', 1, None),
        ('s', -1, 6),
    ],
    '3(g-g6)+(s-s6)': [
        ('g', 3, None),
        ('g', -3, 6),
        ('s', 1, None),
        ('s', -1, 6),
    ],
    'g+g5-g6-g7': [
        ('g', 1, None),
        ('g', 1, 5),
        ('g', -1, 6),
        ('g', -1, 7),
    ],
    'g-g5-g6+g7': [
        ('g', 1, None),
        ('g', -1, 5),
        ('g', -1, 6),
        ('g', 1, 7),
    ],
    'g+g5-2g6-s6+s7': [
        ('g', 1, None),
        ('g', 1, 5),
        ('g', -2, 6),
        ('s', -1, 6),
        ('s', 1, 7),
    ],
}


@pytest.mark.parametrize("formula, expected", SECULAR_RESONANCES.items())
def test_parse_formula(formula, expected):
    f = SecularResonanceFormula(formula)

    parsed = sorted(
        ((t.mode, t.coefficient, t.index) for t in f.terms),
        key=lambda x: (x[0], x[1], x[2] if x[2] is not None else -1),
    )

    expected = sorted(
        expected,
        key=lambda x: (x[0], x[1], x[2] if x[2] is not None else -1),
    )

    assert parsed == expected


def test_no_zero_coefficients():
    for formula in SECULAR_RESONANCES:
        f = SecularResonanceFormula(formula)
        assert all(t.coefficient != 0 for t in f.terms)


def test_term_types_consistency():
    f = SecularResonanceFormula("g-g5")
    proper = [t for t in f.terms if t.index is None]
    forced = [t for t in f.terms if t.index is not None]

    assert len(proper) == 1
    assert len(forced) == 1
