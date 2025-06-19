from resonances.resonance.plot import round_to_nice_value


class TestRoundToNiceValue:
    """Test suite for the round_to_nice_value function."""

    def test_round_to_nice_value_basic_cases(self):
        """Test round_to_nice_value with the provided test cases."""
        assert round_to_nice_value(20000) == 20000
        assert round_to_nice_value(50000) == 50000
        assert round_to_nice_value(200000) == 200000

    def test_round_to_nice_value_edge_cases(self):
        """Test edge cases for round_to_nice_value function."""
        # Test zero and negative values
        assert round_to_nice_value(0) == 0
        assert round_to_nice_value(-10) == 0

        # Test small positive values
        assert round_to_nice_value(1) == 1
        assert round_to_nice_value(5) == 5
        assert round_to_nice_value(15) == 20
        assert round_to_nice_value(25) == 20  # rounds to nearest 10^1 multiple

        # Test values that need rounding
        assert round_to_nice_value(123) == 100
        assert round_to_nice_value(1234) == 1000
        assert round_to_nice_value(12345) == 10000

    def test_major_minor_tick_calculation(self):
        """Test the complete major/minor tick calculation for the provided scenarios."""
        # Test case 1: tmax_yrs = 100,000
        tmax_years = 100000
        major_tick = round_to_nice_value(tmax_years / 5)
        minor_tick = major_tick // 5
        assert major_tick == 20000
        assert minor_tick == 4000

        # Test case 2: tmax_yrs = 250,000
        tmax_years = 250000
        major_tick = round_to_nice_value(tmax_years / 5)
        minor_tick = major_tick // 5
        assert major_tick == 50000
        assert minor_tick == 10000

        # Test case 3: tmax_yrs = 1,000,000
        tmax_years = 1000000
        major_tick = round_to_nice_value(tmax_years / 5)
        minor_tick = major_tick // 5
        assert major_tick == 200000
        assert minor_tick == 40000
