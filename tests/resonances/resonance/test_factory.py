import pytest

from resonances.mmr.three_body import ThreeBody
from resonances.mmr.two_body import TwoBody
from resonances.mmr.mmr import MMR
from resonances.secular.secular_resonance import SecularResonance
from resonances import LidovKozaiResonance
from resonances.data.const import SECULAR_FORMULAS
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

    mmr = create_mmr('2J+3S-1+0+0-4')
    assert isinstance(mmr, ThreeBody) is True


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

    # Formulas that do not start with g/s: a leading minus or a bracket (these used to come out as MMRs)
    assert detect_resonance_type('-g+s+g5-s7') == 'secular'
    assert detect_resonance_type('-g+2s-g6') == 'secular'
    assert detect_resonance_type('2(g-g6)+(s-s6)') == 'secular'
    assert detect_resonance_type('4(g-g6)+(s-s6)') == 'secular'
    assert detect_resonance_type('nu14') == 'secular'
    assert detect_resonance_type('NU3') == 'secular'
    for formula in SECULAR_FORMULAS:
        assert detect_resonance_type(formula) == 'secular', formula
        assert isinstance(create_resonance(formula), SecularResonance), formula
    # Capital planet letters stay MMRs
    assert detect_resonance_type('1S-1') == 'mmr'
    assert detect_resonance_type('5J-2S-2') == 'mmr'

    # Test Lidov-Kozai identifiers
    assert detect_resonance_type('lk') == 'lidov_kozai'
    assert detect_resonance_type('Lidov-Kozai') == 'lidov_kozai'

    # Test patterns that behave based on their starting characters
    assert detect_resonance_type('g5-g6') == 'secular'  # starts with 'g' (even though it's g5)
    assert detect_resonance_type('2*g-2*s') == 'secular'  # only g and s: secular (and then rejected by the parser)
    assert detect_resonance_type('g+s-s7-g5') == 'secular'  # starts with 'g'
    assert detect_resonance_type('g-2*g5+g6') == 'secular'  # starts with 'g'

    # Test with actual resonance objects
    mmr_obj = create_mmr('2J-1')
    assert detect_resonance_type(mmr_obj) == 'mmr'

    secular_obj = create_secular_resonance('nu6')
    assert detect_resonance_type(secular_obj) == 'secular'

    lidov_obj = LidovKozaiResonance()
    assert detect_resonance_type(lidov_obj) == 'lidov_kozai'
    # Test with secular resonance object
    general_secular = SecularResonance('g-2g6+g5')
    assert detect_resonance_type(general_secular) == 'secular'

    # Test another secular resonance with different pattern
    complex_secular = SecularResonance('2g-g5-g6')
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
    assert isinstance(secular, SecularResonance)
    assert secular.type == 'secular'
    assert secular.to_short() == 'g-g6'

    secular5 = create_resonance('nu5')
    assert isinstance(secular5, SecularResonance)
    assert secular5.type == 'secular'
    assert secular5.to_short() == 'g-g5'

    secular16 = create_resonance('nu16')
    assert isinstance(secular16, SecularResonance)
    assert secular16.type == 'secular'
    assert secular16.to_short() == 's-s6'

    # Test case insensitive
    secular_case = create_resonance('Nu6')
    assert isinstance(secular_case, SecularResonance)
    assert secular_case.type == 'secular'
    assert secular_case.to_short() == 'g-g6'

    # Test returning existing resonance objects as-is
    existing_mmr = create_mmr('2J-1')
    returned_mmr = create_resonance(existing_mmr)
    assert returned_mmr is existing_mmr

    existing_secular = create_secular_resonance('nu6')
    returned_secular = create_resonance(existing_secular)
    assert returned_secular is existing_secular

    existing_lidov = LidovKozaiResonance('LK-test')
    returned_lidov = create_resonance(existing_lidov)
    assert returned_lidov is existing_lidov

    # Test returning SecularResonance as-is
    general_secular = SecularResonance('g-2g6+g5')
    returned_general = create_resonance(general_secular)
    assert returned_general is general_secular
    assert returned_general.type == 'secular'

    # Planets are numbered 1..8 (Mercury..Neptune); anything else is not a planet.
    with pytest.raises(ValueError, match='unsupported forced index'):
        create_resonance('g-g9')

    with pytest.raises(ValueError, match='unsupported forced index'):
        create_resonance('s-s0')

    # An angle without the body's own g or s is not a resonance of the body.
    with pytest.raises(ValueError, match='no proper term'):
        create_resonance('g1')

    with pytest.raises(ValueError, match='no proper term'):
        create_resonance('s2')

    # Test complex secular patterns that are supported through the formula parser
    secular_complex = create_resonance('g-g5')
    assert isinstance(secular_complex, SecularResonance)
    assert secular_complex.type == 'secular'
    assert secular_complex.to_short() == 'g-g5'

    # '*' notation is intentionally not supported for secular formulas.
    with pytest.raises(ValueError, match='illegal character'):
        create_resonance('g-2*g5+g6')

    lidov = create_resonance('lk')
    assert isinstance(lidov, LidovKozaiResonance)
    assert lidov.type == 'lidov_kozai'
    assert lidov.identifier == 'LK'
    custom_lidov = create_resonance('lk:Jupiter')
    assert isinstance(custom_lidov, LidovKozaiResonance)
    assert custom_lidov.identifier == 'Jupiter'

    # Test invalid input type
    with pytest.raises(Exception, match='should be either a valid secular resonance'):
        create_resonance(42)

    with pytest.raises(Exception, match='should be either a valid secular resonance'):
        create_resonance(['2J-1'])

    # Test invalid resonance string (this would be caught by the underlying functions)
    with pytest.raises(Exception, match='notation is wrong'):
        create_resonance('invalid_resonance_string')


@pytest.mark.parametrize(
    'notation, expected_class, short',
    [
        ('2S-1', TwoBody, '2S-1'),
        ('2S-1+0-1', TwoBody, '2S-1'),
        ('2S-3', TwoBody, '2S-3'),
        ('2S-1U-1', ThreeBody, '2S-1U-1'),
        ('2S-1U-1+0+0+0', ThreeBody, '2S-1U-1'),
        ('4j-2s-1', ThreeBody, '4J-2S-1'),  # lowercase planet letters with a non-secular letter stay MMRs
    ],
)
def test_mmrs_that_start_like_a_secular_formula(notation, expected_class, short):
    # Capital S is Saturn: these used to be sent to the secular parser and raise.
    assert detect_resonance_type(notation) == 'mmr'
    resonance = create_resonance(notation)
    assert isinstance(resonance, expected_class)
    assert resonance.to_short() == short


def test_saturn_mmr_axis_from_kepler():
    # 2S-1: n = 2 n_S, so a = a_S * 2^(-2/3) = 9.5415 AU * 0.62996 = 6.0108 AU
    assert create_resonance('2S-1').resonant_axis == pytest.approx(6.0108, abs=1e-3)


@pytest.mark.parametrize(
    'notation, short',
    [
        ('g-g6', 'g-g6'),
        ('2g-g5-g6', '2g-g5-g6'),
        ('s-s6', 's-s6'),
        ('z1', 'g-g6+s-s6'),
        ('Z2', '2g-2g6+s-s6'),
        ('nu6', 'g-g6'),
        ('NU16', 's-s6'),
        ('2nu6-nu5', 'g-2g6+g5'),
    ],
)
def test_secular_notations(notation, short):
    assert detect_resonance_type(notation) == 'secular'
    resonance = create_resonance(notation)
    assert isinstance(resonance, SecularResonance)
    assert resonance.to_short() == short


@pytest.mark.parametrize('notation', ['G-G6', 'S-S6', '2(G-G6)'])
def test_capitalised_secular_notation_gets_a_hint(notation):
    with pytest.raises(ValueError, match="write g and s in lowercase"):
        create_resonance(notation)
