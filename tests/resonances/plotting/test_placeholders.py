"""Tests for dynamic column placeholder support."""

import numpy as np
from resonances.plotting import Plotter, PlotConfig, Panel


class TestPlaceholders:
    """Test suite for placeholder replacement in column names."""

    def test_replace_placeholders_resonance_key(self):
        """Test that {resonance_key} placeholder is replaced."""
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'TestBody',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }

        # Test resonance_key replacement
        result = plotter._replace_placeholders('{resonance_key}_angle')
        assert result == '4J-2S-1+0+0-1_angle'

        result = plotter._replace_placeholders('{resonance_key}_angle_filtered')
        assert result == '4J-2S-1+0+0-1_angle_filtered'

    def test_replace_placeholders_body_name(self):
        """Test that {body_name} placeholder is replaced."""
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'Asteroid463',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }

        result = plotter._replace_placeholders('{body_name}_data')
        assert result == 'Asteroid463_data'

    def test_replace_placeholders_multiple(self):
        """Test multiple placeholder replacements in one string."""
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '3J-1S-1+0+0-1',
            'body_name': 'TestBody',
            'resonance': '3J-1S-1',
            'status': '1',
            'tmax_years': 50000,
        }

        result = plotter._replace_placeholders('{body_name}_{resonance_key}_angle')
        assert result == 'TestBody_3J-1S-1+0+0-1_angle'

    def test_replace_placeholders_no_metadata(self):
        """Test that placeholders are preserved if no metadata."""
        plotter = Plotter()
        plotter._metadata = None

        result = plotter._replace_placeholders('{resonance_key}_angle')
        assert result == '{resonance_key}_angle'

    def test_get_column_data_with_placeholder(self):
        """Test that _get_column_data works with placeholders."""
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'TestBody',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }
        plotter._data = {'4J-2S-1+0+0-1_angle': np.array([0, 0.5, 1.0]), 'a': np.array([1.0, 1.1, 1.2])}
        plotter._periodogram_data = None

        # Get data using placeholder
        data = plotter._get_column_data('{resonance_key}_angle')
        assert data is not None
        assert len(data) == 3
        assert data[0] == 0

    def test_get_column_data_with_fallback_placeholder(self):
        """Test that fallback column also supports placeholders."""
        plotter = Plotter()
        plotter._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'TestBody',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }
        plotter._data = {'4J-2S-1+0+0-1_angle': np.array([0, 0.5, 1.0]), 'a': np.array([1.0, 1.1, 1.2])}
        plotter._periodogram_data = None

        # Primary column doesn't exist, fallback uses placeholder
        data = plotter._get_column_data('{resonance_key}_angle_filtered', fallback='{resonance_key}_angle')
        assert data is not None
        assert len(data) == 3

    def test_universal_config_multiple_resonances(self):
        """Test that a single config with placeholders works for multiple resonances."""
        # Create a universal config using placeholders
        config = PlotConfig(figsize=(10, 8))
        config.add_panel(Panel(key='angle', data_column='{resonance_key}_angle', x_column='times'))

        # Test with first resonance
        plotter1 = Plotter()
        plotter1._metadata = {
            'resonance_key': '4J-2S-1+0+0-1',
            'body_name': 'Body1',
            'resonance': '4J-2S-1',
            'status': '2',
            'tmax_years': 100000,
        }
        plotter1._data = {'times': np.array([0, 100, 200]), '4J-2S-1+0+0-1_angle': np.array([0, 0.5, 1.0])}
        plotter1._periodogram_data = None
        plotter1._config = config

        data1 = plotter1._get_column_data('{resonance_key}_angle')
        assert data1 is not None
        assert data1[0] == 0

        # Test with second resonance using same config
        plotter2 = Plotter()
        plotter2._metadata = {
            'resonance_key': '3J-1S-1+0+0-1',
            'body_name': 'Body2',
            'resonance': '3J-1S-1',
            'status': '1',
            'tmax_years': 100000,
        }
        plotter2._data = {'times': np.array([0, 100, 200]), '3J-1S-1+0+0-1_angle': np.array([1, 1.5, 2.0])}
        plotter2._periodogram_data = None
        plotter2._config = config

        data2 = plotter2._get_column_data('{resonance_key}_angle')
        assert data2 is not None
        assert data2[0] == 1
