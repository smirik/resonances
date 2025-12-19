import pytest
from resonances.data.const import PLANETARY_FREQUENCIES
from resonances.secular.secular_resonance import SecularResonance
from resonances.secular.secular_resonance_finder import SecularResonanceFinder


def test_find_secular_resonances_from_proper_freqs():
    freqs = PLANETARY_FREQUENCIES.copy()
    proper_freqs = {"g": freqs["g6"], "s": freqs["s6"]}

    finder = SecularResonanceFinder()
    results = finder.find_secular_resonances(proper_freqs=proper_freqs)

    assert "g-g6" in results
    assert "s-s6" in results
    assert "g-g6+s-s6" in results
    assert isinstance(results["g-g6"], SecularResonance)


@pytest.mark.slow
def test_find_real_secular_resonances():
    finder = SecularResonanceFinder(threshold_negative=-6, threshold_positive=3)

    asteroids_in_nu6 = [1222, 337335, 143199, 295883, 73415]
    for asteroid in asteroids_in_nu6:
        results = finder.find_secular_resonances(asteroid)
        assert "g-g6" in results
        assert isinstance(results["g-g6"], SecularResonance)

    asteroids_in_z1 = [847, 1020, 363]
    finder = SecularResonanceFinder(threshold_negative=-6, threshold_positive=3)
    for asteroid in asteroids_in_z1:
        results = finder.find_secular_resonances(asteroid)
        assert "g-g6+s-s6" in results
        assert isinstance(results["g-g6+s-s6"], SecularResonance)


@pytest.mark.slow
def test_find_secular_resonances_for_asteroids():
    finder = SecularResonanceFinder(threshold_negative=-6, threshold_positive=3)

    asteroids = [1222, 847, 363, 143199, 73415]
    results = finder.find_secular_resonances_for_asteroids(asteroids)

    assert isinstance(results, dict)
    assert len(results) > 0

    assert "g-g6" in results["1222"]
    assert "g-g6" in results["143199"]
    assert "g-g6" in results["73415"]
    assert "g-g6+s-s6" in results["847"]
    assert "g-g6+s-s6" in results["363"]
