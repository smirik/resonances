import pytest

from resonances.resonance.secular import GeneralSecularResonance


@pytest.mark.parametrize(
    "formula,expected,planets",
    [
        ("g-g6", {"g": 1.0, "g6": -1.0}, ["Saturn"]),
        ("g-g5", {"g": 1.0, "g5": -1.0}, ["Jupiter"]),
        ("g-g6+s-s6", {"g": 1.0, "g6": -1.0, "s": 1.0, "s6": -1.0}, ["Saturn"]),
        ("2g-2g6+s-s6", {"g": 2.0, "g6": -2.0, "s": 1.0, "s6": -1.0}, ["Saturn"]),
        ("2(g-g6)+(s-s6)", {"g": 2.0, "g6": -2.0, "s": 1.0, "s6": -1.0}, ["Saturn"]),
        ("g+s-s7-g5", {"g": 1.0, "s": 1.0, "s7": -1.0, "g5": -1.0}, ["Jupiter", "Uranus"]),
    ],
)
def test_general_secular_parse_formula(formula, expected, planets):
    """Ensure _parse_formula produces correct coefficient dictionaries."""
    parsed = GeneralSecularResonance._parse_formula(formula)
    res = GeneralSecularResonance(formula=formula)

    # Only compare keys that are expected to change for the given formula.
    for key, value in expected.items():
        assert parsed[key] == pytest.approx(value), f"Formula {formula}: coefficient {key} mismatch"

    # All other coefficients should remain zero.
    for key, value in parsed.items():
        if key not in expected:
            assert value == 0.0, f"Formula {formula}: unexpected non-zero coefficient for {key}"

    # Planet list should match expectations derived from the coefficients.
    assert res.planets_names == planets, f"Formula {formula}: planet list mismatch"
