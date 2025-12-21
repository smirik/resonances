"""Tests for CLI commands - focusing on parameter parsing and passing."""

import pytest
from unittest.mock import Mock, patch
from click.testing import CliRunner

from resonances.cli.main import cli


class TestFindCommand:
    """Test suite for 'find' command parameter handling."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation object."""
        sim = Mock()
        sim.config.save_path = 'cache/tests/test_simulation'
        sim.run = Mock()
        return sim

    def test_find_missing_asteroids_parameter(self, runner):
        """Test that find command requires --asteroids parameter."""
        result = runner.invoke(cli, ['find'])

        assert result.exit_code == 1
        assert 'Error: --asteroids is required' in result.output

    def test_find_single_asteroid(self, runner, mock_simulation):
        """Test find with single asteroid number."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(cli, ['find', '--asteroids=463'])

            assert result.exit_code == 0
            mock_find.assert_called_once()

            # Check that asteroids parameter is an integer
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs['asteroids'] == 463

    def test_find_multiple_asteroids_comma_separated(self, runner, mock_simulation):
        """Test find with comma-separated asteroid list."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(cli, ['find', '--asteroids=463,490,588'])

            assert result.exit_code == 0
            mock_find.assert_called_once()

            # Check that asteroids parameter is a list
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs['asteroids'] == [463, 490, 588]

    def test_find_with_planets_parameter(self, runner, mock_simulation):
        """Test find with planets parameter."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(cli, ['find', '--asteroids=463', '--planets=Jupiter,Saturn,Uranus'])

            assert result.exit_code == 0
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs['planets'] == ['Jupiter', 'Saturn', 'Uranus']

    def test_find_with_integration_years(self, runner, mock_simulation):
        """Test find with integration-years parameter."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(cli, ['find', '--asteroids=463', '--integration-years=50000'])

            assert result.exit_code == 0
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs['integration_years'] == 50000
            assert isinstance(call_kwargs['integration_years'], int)

    def test_find_with_type_parameter(self, runner, mock_simulation):
        """Test find with type parameter (resonance types)."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(cli, ['find', '--asteroids=463', '--type=mmr,secular'])

            assert result.exit_code == 0
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs['type'] == ['mmr', 'secular']

    def test_find_with_save_plot_parameters(self, runner, mock_simulation):
        """Test find with save and plot parameters."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(cli, ['find', '--asteroids=463', '--save=all', '--plot=resonant'])

            assert result.exit_code == 0
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs['save'] == 'all'
            assert call_kwargs['plot'] == 'resonant'

    def test_find_with_float_parameter(self, runner, mock_simulation):
        """Test find with float parameter (dt)."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(cli, ['find', '--asteroids=463', '--dt=0.01'])

            assert result.exit_code == 0
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs['dt'] == 0.01
            assert isinstance(call_kwargs['dt'], float)

    def test_find_with_boolean_parameter(self, runner, mock_simulation):
        """Test find with boolean parameter."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(cli, ['find', '--asteroids=463', '--safe-mode=false'])

            assert result.exit_code == 0
            call_kwargs = mock_find.call_args[1]
            assert call_kwargs['safe_mode'] is False

    def test_find_kebab_case_to_snake_case(self, runner, mock_simulation):
        """Test that kebab-case parameters are converted to snake_case."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(
                cli, ['find', '--asteroids=463', '--save-path=./results', '--plot-path=./plots', '--integration-years=50000']
            )

            assert result.exit_code == 0
            call_kwargs = mock_find.call_args[1]

            # Check snake_case conversion
            assert 'save_path' in call_kwargs
            assert 'plot_path' in call_kwargs
            assert 'integration_years' in call_kwargs

            # Check kebab-case NOT in kwargs
            assert 'save-path' not in call_kwargs
            assert 'plot-path' not in call_kwargs

    def test_find_multiple_parameter_types(self, runner, mock_simulation):
        """Test find with multiple different parameter types."""
        with patch('resonances.cli.main.resonances.find', return_value=mock_simulation) as mock_find:
            result = runner.invoke(
                cli,
                [
                    'find',
                    '--asteroids=463,490',
                    '--planets=Mars,Jupiter,Saturn',
                    '--integration-years=100000',
                    '--type=mmr,secular',
                    '--dt=0.01',
                    '--save=all',
                    '--plot=resonant',
                    '--name=test-simulation',
                ],
            )

            assert result.exit_code == 0
            call_kwargs = mock_find.call_args[1]

            # Verify all parameters
            assert call_kwargs['asteroids'] == [463, 490]
            assert call_kwargs['planets'] == ['Mars', 'Jupiter', 'Saturn']
            assert call_kwargs['integration_years'] == 100000
            assert call_kwargs['type'] == ['mmr', 'secular']
            assert call_kwargs['dt'] == 0.01
            assert call_kwargs['save'] == 'all'
            assert call_kwargs['plot'] == 'resonant'
            assert call_kwargs['name'] == 'test-simulation'


class TestCheckCommand:
    """Test suite for 'check' command parameter handling."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_simulation(self):
        """Create mock simulation object."""
        sim = Mock()
        sim.config.save_path = 'cache/tests/test_simulation'
        sim.run = Mock()
        return sim

    def test_check_missing_asteroids_parameter(self, runner):
        """Test that check command requires --asteroids parameter."""
        result = runner.invoke(cli, ['check'])

        assert result.exit_code == 1
        assert 'Error: --asteroids is required' in result.output

    def test_check_missing_resonances_parameter(self, runner):
        """Test that check command requires --resonances parameter."""
        result = runner.invoke(cli, ['check', '--asteroids=463'])

        assert result.exit_code == 1
        assert 'Error: --resonances is required' in result.output

    def test_check_single_asteroid_single_resonance(self, runner, mock_simulation):
        """Test check with single asteroid and single resonance."""
        with patch('resonances.cli.main.resonances.check', return_value=mock_simulation) as mock_check:
            result = runner.invoke(cli, ['check', '--asteroids=463', '--resonances=4J-2S-1'])

            assert result.exit_code == 0
            mock_check.assert_called_once()

            call_kwargs = mock_check.call_args[1]
            assert call_kwargs['asteroids'] == 463
            assert call_kwargs['resonances'] == '4J-2S-1'

    def test_check_multiple_asteroids_comma_separated(self, runner, mock_simulation):
        """Test check with comma-separated asteroids."""
        with patch('resonances.cli.main.resonances.check', return_value=mock_simulation) as mock_check:
            result = runner.invoke(cli, ['check', '--asteroids=463,490', '--resonances=4J-2S-1'])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs['asteroids'] == [463, 490]

    def test_check_multiple_resonances_comma_separated(self, runner, mock_simulation):
        """Test check with comma-separated resonances."""
        with patch('resonances.cli.main.resonances.check', return_value=mock_simulation) as mock_check:
            result = runner.invoke(cli, ['check', '--asteroids=463', '--resonances=4J-2S-1,3J-1S-1'])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs['resonances'] == ['4J-2S-1', '3J-1S-1']

    def test_check_with_planets_parameter(self, runner, mock_simulation):
        """Test check with planets parameter."""
        with patch('resonances.cli.main.resonances.check', return_value=mock_simulation) as mock_check:
            result = runner.invoke(cli, ['check', '--asteroids=463', '--resonances=4J-2S-1', '--planets=Jupiter,Saturn,Uranus'])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs['planets'] == ['Jupiter', 'Saturn', 'Uranus']

    def test_check_with_integration_years(self, runner, mock_simulation):
        """Test check with integration-years parameter."""
        with patch('resonances.cli.main.resonances.check', return_value=mock_simulation) as mock_check:
            result = runner.invoke(cli, ['check', '--asteroids=463', '--resonances=4J-2S-1', '--integration-years=50000'])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs['integration_years'] == 50000

    def test_check_with_save_plot_parameters(self, runner, mock_simulation):
        """Test check with save and plot parameters."""
        with patch('resonances.cli.main.resonances.check', return_value=mock_simulation) as mock_check:
            result = runner.invoke(cli, ['check', '--asteroids=463', '--resonances=4J-2S-1', '--save=all', '--plot=all'])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs['save'] == 'all'
            assert call_kwargs['plot'] == 'all'

    def test_check_with_advanced_parameters(self, runner, mock_simulation):
        """Test check with advanced parameters (dt, integrator)."""
        with patch('resonances.cli.main.resonances.check', return_value=mock_simulation) as mock_check:
            result = runner.invoke(cli, ['check', '--asteroids=463', '--resonances=4J-2S-1', '--dt=0.01', '--integrator=SABA'])

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]
            assert call_kwargs['dt'] == 0.01
            assert call_kwargs['integrator'] == 'SABA'

    def test_check_multiple_parameter_types(self, runner, mock_simulation):
        """Test check with multiple different parameter types."""
        with patch('resonances.cli.main.resonances.check', return_value=mock_simulation) as mock_check:
            result = runner.invoke(
                cli,
                [
                    'check',
                    '--asteroids=463,490',
                    '--resonances=4J-2S-1,3J-1S-1',
                    '--planets=Jupiter,Saturn,Uranus',
                    '--integration-years=100000',
                    '--dt=0.01',
                    '--save=all',
                    '--plot=all',
                    '--name=test-check',
                ],
            )

            assert result.exit_code == 0
            call_kwargs = mock_check.call_args[1]

            # Verify all parameters
            assert call_kwargs['asteroids'] == [463, 490]
            assert call_kwargs['resonances'] == ['4J-2S-1', '3J-1S-1']
            assert call_kwargs['planets'] == ['Jupiter', 'Saturn', 'Uranus']
            assert call_kwargs['integration_years'] == 100000
            assert call_kwargs['dt'] == 0.01
            assert call_kwargs['save'] == 'all'
            assert call_kwargs['plot'] == 'all'
            assert call_kwargs['name'] == 'test-check'


class TestParameterParsing:
    """Test suite for parameter parsing functions."""

    def test_parse_single_int(self):
        """Test parsing single integer."""
        from resonances.cli.main import _parse_param

        assert _parse_param('463') == 463
        assert isinstance(_parse_param('463'), int)

    def test_parse_comma_separated_ints(self):
        """Test parsing comma-separated integers."""
        from resonances.cli.main import _parse_param

        result = _parse_param('463,490,588')
        assert result == [463, 490, 588]
        assert all(isinstance(x, int) for x in result)

    def test_parse_single_float(self):
        """Test parsing single float."""
        from resonances.cli.main import _parse_param

        assert _parse_param('0.01') == 0.01
        assert isinstance(_parse_param('0.01'), float)

    def test_parse_comma_separated_floats(self):
        """Test parsing comma-separated floats."""
        from resonances.cli.main import _parse_param

        result = _parse_param('0.01,0.02,0.03')
        assert result == [0.01, 0.02, 0.03]
        assert all(isinstance(x, float) for x in result)

    def test_parse_single_string(self):
        """Test parsing single string."""
        from resonances.cli.main import _parse_param

        assert _parse_param('Jupiter') == 'Jupiter'
        assert isinstance(_parse_param('Jupiter'), str)

    def test_parse_comma_separated_strings(self):
        """Test parsing comma-separated strings."""
        from resonances.cli.main import _parse_param

        result = _parse_param('Jupiter,Saturn,Uranus')
        assert result == ['Jupiter', 'Saturn', 'Uranus']
        assert all(isinstance(x, str) for x in result)

    def test_parse_mixed_comma_separated(self):
        """Test parsing comma-separated mixed types."""
        from resonances.cli.main import _parse_param

        # Resonance names (strings)
        result = _parse_param('4J-2S-1,3J-1S-1')
        assert result == ['4J-2S-1', '3J-1S-1']

    def test_parse_boolean_true(self):
        """Test parsing boolean true values."""
        from resonances.cli.main import _parse_param

        assert _parse_param('true') is True
        assert _parse_param('True') is True
        assert _parse_param('yes') is True
        # Note: '1' is parsed as int, not bool
        assert _parse_param('1') == 1

    def test_parse_boolean_false(self):
        """Test parsing boolean false values."""
        from resonances.cli.main import _parse_param

        assert _parse_param('false') is False
        assert _parse_param('False') is False
        assert _parse_param('no') is False
        # Note: '0' is parsed as int, not bool
        assert _parse_param('0') == 0

    def test_kebab_to_snake_case(self):
        """Test kebab-case to snake_case conversion."""
        from resonances.cli.main import _parse_extra_args

        args = ['--save-path=./results', '--integration-years=50000']
        kwargs = _parse_extra_args(args)

        assert 'save_path' in kwargs
        assert 'integration_years' in kwargs
        assert kwargs['save_path'] == './results'
        assert kwargs['integration_years'] == 50000
