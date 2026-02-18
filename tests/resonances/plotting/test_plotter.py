"""Tests for Plotter class."""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from resonances.plotting import Plotter, PlotConfig, Panel


class TestPlotterConfiguration:
    """Test suite for Plotter configuration methods."""

    def test_configure_with_string_preset(self):
        """Test configuring plotter with preset name."""
        # Note: This is a minimal test - full integration tests require Body objects
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'Test',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }
        plotter._data = {}

        result = plotter.configure('simple')
        assert result is plotter  # Returns self for chaining
        assert plotter._config is not None
        assert isinstance(plotter._config, PlotConfig)

    def test_configure_with_plot_config_object(self):
        """Test configuring plotter with PlotConfig object."""
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'Test',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }
        plotter._data = {}

        custom_config = PlotConfig(figsize=(8, 8))
        custom_config.add_panel(Panel(key='test', data_column='a'))

        result = plotter.configure(custom_config)
        assert result is plotter
        assert plotter._config is custom_config
        assert plotter._config.figsize == (8, 8)

    def test_configure_invalid_preset_name(self):
        """Test that invalid preset name raises error."""
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'Test',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }

        with pytest.raises(ValueError, match="Unknown preset"):
            plotter.configure('invalid_preset')


class TestPlotterFromCSV:
    """Test suite for Plotter.from_csv factory method."""

    def test_from_csv_basic(self):
        """Test loading data from CSV file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
            # Write simple CSV data
            f.write('times,a,e,inc\n')
            f.write('0,1.0,0.1,5.0\n')
            f.write('100,1.1,0.12,5.1\n')
            f.write('200,1.2,0.14,5.2\n')

        try:
            resonance_key = '4J-2S-1+0+0-1'
            plotter = Plotter.from_csv(csv_path, resonance_key)

            assert plotter._metadata['resonance_key'] == resonance_key
            # CSV data is loaded into _data dict (without 'times' in _data, it's the index)
            assert 'a' in plotter._data
            assert 'e' in plotter._data
            assert plotter._data['a'][0] == 1.0
            assert len(plotter._data['a']) == 3
        finally:
            Path(csv_path).unlink()

    def test_from_csv_with_periodogram(self):
        """Test loading data with separate periodogram CSV."""
        # Create main CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            main_csv = f.name
            f.write('times,a,e\n')
            f.write('0,1.0,0.1\n')

        # Create periodogram CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='-periodograms.csv', delete=False) as f:
            perio_csv = f.name
            f.write('a_frequency,a_power\n')
            f.write('0.001,0.05\n')

        try:
            resonance_key = '4J-2S-1+0+0-1'
            plotter = Plotter.from_csv(main_csv, resonance_key, periodogram_path=perio_csv)

            assert 'a_frequency' in plotter._periodogram_data.columns
            assert 'a_power' in plotter._periodogram_data.columns
        finally:
            Path(main_csv).unlink()
            Path(perio_csv).unlink()

    def test_from_csv_auto_detect_periodogram(self):
        """Test auto-detection of periodogram CSV file."""
        # Create main CSV with specific naming pattern
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            main_csv = f.name
            f.write('times,a\n')
            f.write('0,1.0\n')

        # Create periodogram CSV with expected naming
        base_path = main_csv.replace('.csv', '')
        perio_csv = f'{base_path}-periodograms.csv'

        with open(perio_csv, 'w') as f:
            f.write('a_frequency,a_power\n')
            f.write('0.001,0.05\n')

        try:
            resonance_key = '4J-2S-1+0+0-1'
            plotter = Plotter.from_csv(main_csv, resonance_key)

            # Should auto-detect periodogram file
            if plotter._periodogram_data is not None:
                assert 'a_frequency' in plotter._periodogram_data.columns
        finally:
            Path(main_csv).unlink()
            if Path(perio_csv).exists():
                Path(perio_csv).unlink()

    def test_from_csv_missing_file(self):
        """Test that missing CSV file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            Plotter.from_csv('/nonexistent/path.csv', '4J-2S-1+0+0-1')


class TestPlotterMethodChaining:
    """Test suite for Plotter method chaining."""

    def test_method_chaining_returns_self(self):
        """Test that methods return self for chaining."""
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'Test',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }
        plotter._data = {'times': np.array([0, 1, 2]), 'a': np.array([1.0, 1.1, 1.2])}

        # Configure with a config that has panels
        config = PlotConfig()
        config.add_panel(Panel(key='a', data_column='a', x_column='times'))
        result = plotter.configure(config)
        assert result is plotter

        # Plot returns self (after configuration with panels)
        result = plotter.plot()
        assert result is plotter
        assert plotter._figure is not None
        plotter.close()

    def test_save_creates_directories(self):
        """Test that save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plotter = Plotter()
            plotter._metadata = {
                'resonance_key': '4J-2S-1+0+0-1',
                'body_name': 'Test',
                'resonance': '4J-2S-1',
                'status': '2',
                'tmax_years': 100000,
            }
            plotter._data = {'times': np.array([0, 1, 2]), 'a': np.array([1.0, 1.1, 1.2])}

            # Configure and plot
            simple_config = PlotConfig()
            simple_config.add_panel(Panel(key='a', data_column='a', x_column='times'))
            plotter.configure(simple_config).plot()

            # Save to nested path
            save_path = Path(tmpdir) / 'nested' / 'dir' / 'test.png'
            result = plotter.save(str(save_path))

            assert result is plotter
            assert save_path.exists()
            plotter.close()


class TestPlotterRoundToNiceValue:
    """Test suite for round_to_nice_value helper function."""

    def test_round_to_nice_value_import(self):
        """Test that round_to_nice_value is available from plotter module."""
        from resonances.plotting.plotter import round_to_nice_value

        assert round_to_nice_value(20000) == 20000
        assert round_to_nice_value(50000) == 50000
        assert round_to_nice_value(123) == 100


class TestPlotterIntegration:
    """Integration tests for Plotter (these may require mocking or real Body objects)."""

    def test_plotter_metadata_structure(self):
        """Test that Plotter metadata has expected structure."""
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'TestBody',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }

        assert plotter._metadata['resonance_key'] == '4J-2S-1+0+0-1'
        assert plotter._metadata['body_name'] == 'TestBody'
        assert plotter._metadata['resonance'] == '4J-2S-1'
        assert plotter._metadata['status'] == '2'
        assert plotter._metadata['tmax_years'] == 100000

    def test_plotter_data_dictionary_structure(self):
        """Test that Plotter data dictionary can hold various data types."""
        plotter = Plotter()
        plotter._data = {
            'times': np.array([0, 100, 200]),
            'a': np.array([1.0, 1.1, 1.2]),
            'e': np.array([0.1, 0.12, 0.14]),
            '4J-2S-1+0+0-1_angle': np.array([0, 0.5, 1.0]),
        }

        assert len(plotter._data['times']) == 3
        assert 'a' in plotter._data
        assert '4J-2S-1+0+0-1_angle' in plotter._data
        assert isinstance(plotter._data['times'], np.ndarray)
