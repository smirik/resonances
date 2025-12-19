import math
import pytest

from resonances.resonance.secular import SecularResonance

# ============================================================
# Dummy orbit class (minimal interface needed by calc_angle)
# ============================================================


class DummyOrbit:
    def __init__(self, Omega, omega):
        self.Omega = Omega
        self.omega = omega


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def body():
    # Fixed but arbitrary orbital angles
    return DummyOrbit(
        Omega=1.0,
        omega=0.5,
    )


@pytest.fixture
def planets():
    # Secular index -> orbit
    return {
        5: DummyOrbit(Omega=0.3, omega=0.2),  # Jupiter
        6: DummyOrbit(Omega=0.4, omega=0.1),  # Saturn
        7: DummyOrbit(Omega=0.6, omega=0.25),  # Uranus
    }


# ============================================================
# Resonant angle tests
# ============================================================


@pytest.mark.parametrize(
    "formula, expected_angle",
    [
        # g - g5
        (
            "g-g5",
            (1.0 + 0.5) - (0.3 + 0.2),
        ),
        # s - s6
        (
            "s-s6",
            1.0 - 0.4,
        ),
        # g + s - s6 - g5
        (
            "g+s-s6-g5",
            (1.0 + 0.5) + 1.0 - 0.4 - (0.3 + 0.2),
        ),
        # g - 2g5 + g6
        (
            "g-2g5+g6",
            (1.0 + 0.5) - 2 * (0.3 + 0.2) + (0.4 + 0.1),
        ),
        # 2(g-g6) + (s-s6)
        (
            "2(g-g6)+(s-s6)",
            2 * ((1.0 + 0.5) - (0.4 + 0.1)) + (1.0 - 0.4),
        ),
    ],
)
def test_calc_angle(formula, expected_angle, body, planets):
    sr = SecularResonance(formula)

    angle = sr.calc_angle(body, planets)

    expected = expected_angle % (2 * math.pi)

    assert math.isclose(angle, expected, rel_tol=1e-12)


# ============================================================
# Structural / API tests
# ============================================================


def test_basic_properties():
    sr = SecularResonance("g-g5")

    assert sr.type == "secular"
    assert sr.to_short() == "g-g5"
    assert "g-g5" in sr.to_s()


def test_order():
    assert SecularResonance("g-g5").order() == 2
    assert SecularResonance("g-2g5+g6").order() == 4
    assert SecularResonance("2(g-g6)+(s-s6)").order() == 6


def test_index_of_planets():
    sr = SecularResonance("g-2g5+g6")
    assert sr.index_of_planets == [5, 6]

    sr = SecularResonance("2g-2s")
    assert sr.index_of_planets == []
