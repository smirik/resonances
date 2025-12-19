import pytest
from resonances.resonance.factory import create_secular_resonance
from resonances.resonance.secular import SecularResonance


class TestFactoryIntegration:
    """Test that the factory method works with the new secular matrix functionality."""

    def test_create_known_resonances(self):
        """Test creating known resonances through factory."""
        # Test nu5
        res = create_secular_resonance('nu5')
        assert isinstance(res, SecularResonance)
        assert res.to_short() == 'g-g5'

        # Test nu6
        res = create_secular_resonance('nu6')
        assert isinstance(res, SecularResonance)
        assert res.to_short() == 'g-g6'

        # Test nu16
        res = create_secular_resonance('nu16')
        assert isinstance(res, SecularResonance)
        assert res.to_short() == 's-s6'

    def test_create_from_formulas(self):
        """Test creating resonances from mathematical formulas."""
        # Test simple formulas
        res = create_secular_resonance('g-g5')
        assert isinstance(res, SecularResonance)
        assert res.to_short() == 'g-g5'

        res = create_secular_resonance('g-g6')
        assert isinstance(res, SecularResonance)
        assert res.to_short() == 'g-g6'

        res = create_secular_resonance('s-s6')
        assert isinstance(res, SecularResonance)
        assert res.to_short() == 's-s6'

        # Test complex formulas
        res = create_secular_resonance('2g-g5-g6')
        assert isinstance(res, SecularResonance)
        assert res.to_short() == '2g-g5-g6'

        res = create_secular_resonance('g+s-s7-g5')
        assert isinstance(res, SecularResonance)
        assert res.to_short() == 'g+s-s7-g5'

    def test_create_from_list(self):
        """Test creating resonances from list of formulas."""
        formulas = ['g-g5', 'g-g6', '2g-g5-g6']
        resonances = create_secular_resonance(formulas)

        assert len(resonances) == 3
        assert [r.to_short() for r in resonances] == formulas

    def test_error_handling(self):
        """Test error handling for invalid formulas."""
        with pytest.raises(ValueError) as excinfo:
            create_secular_resonance('completely_invalid_formula')

        assert 'Invalid secular formula' in str(excinfo.value)

    def test_existing_resonance_passthrough(self):
        """Test that existing resonance objects pass through unchanged."""
        original_res = SecularResonance('g-g5')
        result = create_secular_resonance(original_res)

        assert result is original_res
