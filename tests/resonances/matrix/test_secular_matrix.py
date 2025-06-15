import pytest
from resonances.matrix.secular_matrix import SecularMatrix
from resonances.matrix.secular_resonances import SECULAR_RESONANCES, get_available_secular_resonance_formulas, get_available_orders
from resonances.resonance.secular import SecularResonance, Nu5Resonance, Nu6Resonance, Nu16Resonance, GeneralSecularResonance
from resonances.resonance.factory import create_secular_resonance


class TestSecularMatrix:
    """Test the SecularMatrix class functionality."""

    def test_get_available_formulas(self):
        """Test getting available formulas."""
        formulas = get_available_secular_resonance_formulas()
        assert isinstance(formulas, list)
        assert len(formulas) > 0
        assert 'g-g5' in formulas
        assert 'g-g6' in formulas
        assert 's-s6' in formulas

    def test_get_available_formulas_by_order(self):
        """Test getting available formulas filtered by order."""
        order2_formulas = get_available_secular_resonance_formulas(order=2)
        order4_formulas = get_available_secular_resonance_formulas(order=4)

        assert isinstance(order2_formulas, list)
        assert isinstance(order4_formulas, list)
        assert 'g-g5' in order2_formulas
        assert 'g-g6' in order2_formulas
        assert 's-s6' in order2_formulas
        assert 'g+s-s7-g5' in order4_formulas

    def test_get_orders(self):
        """Test getting available orders."""
        orders = get_available_orders()
        assert isinstance(orders, list)
        assert 2 in orders
        assert 4 in orders
        assert 6 in orders

    def test_parse_simple_formulas(self):
        """Test parsing simple formulas with GeneralSecularResonance."""
        # Test g-g5
        res = GeneralSecularResonance(formula='g-g5')
        assert res.resonance_type == 'g-g5'

        # Test g-g6
        res = GeneralSecularResonance(formula='g-g6')
        assert res.resonance_type == 'g-g6'

        # Test s-s6
        res = GeneralSecularResonance(formula='s-s6')
        assert res.resonance_type == 's-s6'

    def test_parse_complex_formulas(self):
        """Test parsing complex formulas with GeneralSecularResonance."""
        # Test 2g-g5-g6
        res = GeneralSecularResonance(formula='2g-g5-g6')
        assert res.resonance_type == '2g-g5-g6'

        # Test g+s-s7-g5
        res = GeneralSecularResonance(formula='g+s-s7-g5')
        assert res.resonance_type == 'g+s-s7-g5'

        # Test -g+s+g5-s7
        res = GeneralSecularResonance(formula='-g+s+g5-s7')
        assert res.resonance_type == '-g+s+g5-s7'

    def test_parse_special_formulas(self):
        """Test parsing special formulas with GeneralSecularResonance."""
        # Test the special case: 2(g-g6)+(s-s6)
        res = GeneralSecularResonance(formula='2(g-g6)+(s-s6)')
        assert res.resonance_type == '2(g-g6)+(s-s6)'

    def test_build_specific_formulas(self):
        """Test building specific formulas."""
        # Test single formula
        resonances = SecularMatrix.build('g-g5')
        assert len(resonances) == 1
        assert isinstance(resonances[0], Nu5Resonance)

        # Test multiple formulas
        resonances = SecularMatrix.build(['g-g5', 'g-g6'])
        assert len(resonances) == 2
        assert isinstance(resonances[0], Nu5Resonance)
        assert isinstance(resonances[1], Nu6Resonance)

        # Test complex formula
        resonances = SecularMatrix.build('g+s-s7-g5')
        assert len(resonances) == 1
        assert isinstance(resonances[0], GeneralSecularResonance)

    def test_build_by_order(self):
        """Test building resonances by order."""
        # Test order 2
        order2_resonances = SecularMatrix.build(order=2)
        assert len(order2_resonances) > 0
        for res in order2_resonances:
            assert isinstance(res, SecularResonance)

        # Test order 4
        order4_resonances = SecularMatrix.build(order=4)
        assert len(order4_resonances) > 0
        for res in order4_resonances:
            assert isinstance(res, SecularResonance)

    def test_build_all(self):
        """Test building all resonances."""
        all_resonances = SecularMatrix.build()
        assert len(all_resonances) > 0
        for res in all_resonances:
            assert isinstance(res, SecularResonance)

    def test_old_way_general_resonance(self):
        """Test creating GeneralSecularResonance with old coefficients way."""
        # Test old way using coefficients
        coeffs = {
            'varpi': [1.0, -1.0],  # Body coefficient, Jupiter coefficient
        }
        planet_names = ['Jupiter']

        res = GeneralSecularResonance(coeffs=coeffs, planet_names=planet_names, resonance_name='test_resonance')
        assert res.resonance_type == 'test_resonance'
        assert 'Jupiter' in res.planet_names
        assert res.coeffs == coeffs

    def test_create_known_resonances(self):
        """Test creating known resonances returns correct types."""
        # Test nu5 (g-g5)
        res = create_secular_resonance('g-g5')
        assert isinstance(res, Nu5Resonance)

        # Test nu6 (g-g6)
        res = create_secular_resonance('g-g6')
        assert isinstance(res, Nu6Resonance)

        # Test nu16 (s-s6)
        res = create_secular_resonance('s-s6')
        assert isinstance(res, Nu16Resonance)

    def test_create_general_resonances(self):
        """Test creating general resonances from complex formulas."""
        # Test complex formula
        res = create_secular_resonance('g+s-s7-g5')
        assert isinstance(res, GeneralSecularResonance)
        assert res.resonance_type == 'g+s-s7-g5'

        # Test another complex formula
        res = create_secular_resonance('2g-s7-s6')
        assert isinstance(res, GeneralSecularResonance)
        assert res.resonance_type == '2g-s7-s6'

    def test_formula_consistency(self):
        """Test that all formulas in SECULAR_RESONANCES can be built."""
        failed_formulas = []

        for formula in SECULAR_RESONANCES.keys():
            try:
                resonances = SecularMatrix.build(formula)
                assert len(resonances) == 1, f"Expected 1 resonance for {formula}, got {len(resonances)}"
                assert isinstance(resonances[0], SecularResonance), f"Expected SecularResonance for {formula}"
            except Exception as e:
                failed_formulas.append((formula, str(e)))

        if failed_formulas:
            pytest.fail(f"Failed to build formulas: {failed_formulas}")

    def test_invalid_formulas(self):
        """Test handling of invalid formulas."""
        # Test completely invalid formula
        resonances = SecularMatrix.build(['invalid_formula'])
        # Should handle gracefully and continue
        assert isinstance(resonances, list)

    def test_resonance_properties(self):
        """Test that created resonances have correct properties."""
        # Test a known simple resonance
        resonances = SecularMatrix.build('g-g5')
        res = resonances[0]

        assert res.type == 'secular'
        assert hasattr(res, 'calc_angle')
        assert hasattr(res, 'to_s')
        assert hasattr(res, 'to_short')

        # Test a complex resonance
        resonances = SecularMatrix.build('g+s-s7-g5')
        res = resonances[0]

        assert res.type == 'secular'
        assert hasattr(res, 'calc_angle')
        assert hasattr(res, 'to_s')

    def test_both_ways_constructor(self):
        """Test that GeneralSecularResonance supports both construction ways."""
        # New way - with formula
        res1 = GeneralSecularResonance(formula='g-g5')
        assert res1.resonance_type == 'g-g5'

        # Old way - with coefficients
        coeffs = {'varpi': [1.0, -1.0]}
        planet_names = ['Jupiter']
        res2 = GeneralSecularResonance(coeffs=coeffs, planet_names=planet_names)
        assert res2.planet_names == ['Jupiter']

        # Both should have the same interface
        assert hasattr(res1, 'calc_angle')
        assert hasattr(res2, 'calc_angle')
        assert res1.type == 'secular'
        assert res2.type == 'secular'

    def test_planetary_frequencies_loading(self):
        """Test that planetary frequencies are loaded correctly."""
        from resonances.matrix.secular_resonances import load_planetary_frequencies

        freqs = load_planetary_frequencies()

        assert isinstance(freqs, dict)
        assert 'g5' in freqs
        assert 'g6' in freqs
        assert 's6' in freqs
        assert 's7' in freqs

        # Check some expected values
        assert freqs['g5'] > 0
        assert freqs['g6'] > 0
        assert freqs['s6'] < 0  # s6 should be negative
