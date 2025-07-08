import pytest

from resonances.resonance.three_body import ThreeBody
from resonances.resonance.two_body import TwoBody
from resonances.resonance.mmr import MMR
from resonances.resonance.secular import Nu6Resonance, Nu5Resonance, Nu16Resonance, GeneralSecularResonance
from resonances.resonance.factory import create_mmr, detect_resonance_type, create_resonance, create_secular_resonance


def test_create_mmr():
    mmr = create_mmr('4J-2S-1')
    assert isinstance(mmr, ThreeBody) is True

    mmr = create_mmr(mmr)
    assert isinstance(mmr, MMR) is True
    assert isinstance(mmr, ThreeBody) is True

    mmr1 = create_mmr('2J-1')
    assert isinstance(mmr1, TwoBody) is True
    mmr2 = create_mmr([4, -2, -1, 0, 0, -1], planets_names=['Jupiter', 'Saturn'])
    assert isinstance(mmr2, ThreeBody) is True
    mmr3 = create_mmr([2, -1, 0, -1], planets_names=['Jupiter'])
    assert isinstance(mmr3, TwoBody) is True

    mmrs = create_mmr([mmr1, mmr2, mmr3])
    assert 3 == len(mmrs)
    assert isinstance(mmrs[0], TwoBody) is True
    assert isinstance(mmrs[1], ThreeBody) is True
    assert isinstance(mmrs[2], TwoBody) is True

    mmrs = create_mmr(['4J-2S-1', '1J-1'])
    assert 2 == len(mmrs)
    assert isinstance(mmrs[0], ThreeBody) is True
    assert isinstance(mmrs[1], TwoBody) is True

    with pytest.raises(Exception, match='notation is wrong'):
        mmr = create_mmr('5J-2S-1U-1')

    with pytest.raises(Exception, match='notation is wrong'):
        mmr = create_mmr('5J')

    with pytest.raises(Exception, match='number of coefficients'):
        mmr = create_mmr([4, -2, -1, 0, 0, 0, -1], planets_names=['Jupiter', 'Saturn'])

    with pytest.raises(Exception, match='number of coefficients'):
        mmr = create_mmr([4, -2, -1], planets_names=['Jupiter', 'Saturn'])

    with pytest.raises(Exception, match='number of coefficients'):
        mmr = create_mmr([4, -2], planets_names=['Jupiter'])

    with pytest.raises(Exception, match='should be either'):
        mmr = create_mmr(42, planets_names=['Jupiter'])


def test_detect_resonance_type():
    """Test the detect_resonance_type function."""

    # Test MMR string inputs
    assert detect_resonance_type('2J-1') == 'mmr'
    assert detect_resonance_type('4J-2S-1') == 'mmr'
    assert detect_resonance_type('3J-1S-2') == 'mmr'
    assert detect_resonance_type('1J-2') == 'mmr'

    # Test secular resonance string inputs
    assert detect_resonance_type('nu6') == 'secular'
    assert detect_resonance_type('nu5') == 'secular'
    assert detect_resonance_type('nu16') == 'secular'
    assert detect_resonance_type('Nu6') == 'secular'  # Test case insensitive
    assert detect_resonance_type('NU5') == 'secular'  # Test case insensitive

    # Test secular resonances starting with g, s, 2g, 2s
    assert detect_resonance_type('g1') == 'secular'
    assert detect_resonance_type('s2') == 'secular'
    assert detect_resonance_type('2g3') == 'secular'
    assert detect_resonance_type('2s4') == 'secular'

    # Test simple secular resonance patterns that start with supported prefixes
    assert detect_resonance_type('g-g5') == 'secular'  # starts with 'g'
    assert detect_resonance_type('g-g6') == 'secular'  # starts with 'g'
    assert detect_resonance_type('s-s7') == 'secular'  # starts with 's'
    assert detect_resonance_type('2g-2s') == 'secular'  # starts with '2g'
    assert detect_resonance_type('2s-2g') == 'secular'  # starts with '2s'

    # Test patterns that behave based on their starting characters
    assert detect_resonance_type('g5-g6') == 'secular'  # starts with 'g' (even though it's g5)
    assert detect_resonance_type('2*g-2*s') == 'mmr'  # starts with '2*', not '2g'
    assert detect_resonance_type('g+s-s7-g5') == 'secular'  # starts with 'g'
    assert detect_resonance_type('g-2*g5+g6') == 'secular'  # starts with 'g'

    # Test with actual resonance objects
    mmr_obj = create_mmr('2J-1')
    assert detect_resonance_type(mmr_obj) == 'mmr'

    secular_obj = create_secular_resonance('nu6')
    assert detect_resonance_type(secular_obj) == 'secular'

    # Test with GeneralSecularResonance object
    general_secular = GeneralSecularResonance(
        coeffs={'varpi': [1, -2, 1]},  # [body_coeff, saturn_coeff, jupiter_coeff]
        planet_names=['Saturn', 'Jupiter'],
        resonance_name='g-2g6+g5',
    )
    assert detect_resonance_type(general_secular) == 'secular'

    # Test another GeneralSecularResonance with different pattern
    complex_secular = GeneralSecularResonance(
        coeffs={'varpi': [2, -1, -1]},
        planet_names=['Saturn', 'Jupiter'],
        resonance_name='2*g-g5-g6',
    )
    assert detect_resonance_type(complex_secular) == 'secular'

    # Test invalid input type
    with pytest.raises(Exception, match='should be either a string'):
        detect_resonance_type(42)

    with pytest.raises(Exception, match='should be either a string'):
        detect_resonance_type(['2J-1'])


def test_create_resonance():
    """Test the create_resonance function."""

    # Test creating MMR from string
    mmr = create_resonance('2J-1')
    assert isinstance(mmr, TwoBody)
    assert mmr.type == 'mmr'

    mmr_three = create_resonance('4J-2S-1')
    assert isinstance(mmr_three, ThreeBody)
    assert mmr_three.type == 'mmr'

    # Test creating secular resonance from string
    secular = create_resonance('nu6')
    assert isinstance(secular, Nu6Resonance)
    assert secular.type == 'secular'

    secular5 = create_resonance('nu5')
    assert isinstance(secular5, Nu5Resonance)
    assert secular5.type == 'secular'

    secular16 = create_resonance('nu16')
    assert isinstance(secular16, Nu16Resonance)
    assert secular16.type == 'secular'

    # Test case insensitive
    secular_case = create_resonance('Nu6')
    assert isinstance(secular_case, Nu6Resonance)
    assert secular_case.type == 'secular'

    # Test returning existing resonance objects as-is
    existing_mmr = create_mmr('2J-1')
    returned_mmr = create_resonance(existing_mmr)
    assert returned_mmr is existing_mmr

    existing_secular = create_secular_resonance('nu6')
    returned_secular = create_resonance(existing_secular)
    assert returned_secular is existing_secular

    # Test returning GeneralSecularResonance as-is
    general_secular = GeneralSecularResonance(
        coeffs={'varpi': [1, -2, 1]},
        planet_names=['Saturn', 'Jupiter'],
        resonance_name='g-2g6+g5',
    )
    returned_general = create_resonance(general_secular)
    assert returned_general is general_secular
    assert returned_general.type == 'secular'

    # Test that unsupported secular resonance strings raise an exception
    # (even though detect_resonance_type detects them as secular,
    # create_secular_resonance doesn't support creating them from strings)
    with pytest.raises(Exception, match='Unknown variable'):
        create_resonance('g1')

    with pytest.raises(Exception, match='Unknown variable'):
        create_resonance('s2')

    # Test complex secular patterns that are now supported through SecularMatrix
    secular_complex = create_resonance('g-g5')
    assert isinstance(secular_complex, Nu5Resonance)
    assert secular_complex.type == 'secular'

    secular_complex2 = create_resonance('g-2*g5+g6')
    assert isinstance(secular_complex2, GeneralSecularResonance)
    assert secular_complex2.type == 'secular'

    # Test invalid input type
    with pytest.raises(Exception, match='should be either a valid secular resonance'):
        create_resonance(42)

    with pytest.raises(Exception, match='should be either a valid secular resonance'):
        create_resonance(['2J-1'])

    # Test invalid resonance string (this would be caught by the underlying functions)
    with pytest.raises(Exception, match='notation is wrong'):
        create_resonance('invalid_resonance_string')
