"""Tests for plotting presets."""

import pytest
from resonances.plotting.presets import create_simple_preset, create_full_preset, get_preset
from resonances.plotting.config import PlotConfig


class TestSimplePreset:
    """Test suite for simple preset configuration."""

    def test_simple_preset_structure(self):
        """Test basic structure of simple preset."""
        resonance_key = '4J-2S-1+0+0-1'
        config = create_simple_preset(resonance_key)

        assert isinstance(config, PlotConfig)
        assert config.figsize == (10, 4)
        assert len(config.panels) == 1

    def test_simple_preset_panel(self):
        """Test panel configuration in simple preset."""
        resonance_key = '4J-2S-1+0+0-1'
        config = create_simple_preset(resonance_key)

        panel = config.panels[0]
        assert panel.key == 'angle'
        assert panel.data_column == f'{resonance_key}_angle'
        assert panel.x_column == 'times'
        assert panel.required is True

    def test_simple_preset_styling(self):
        """Test styling configuration in simple preset."""
        config = create_simple_preset('4J-2S-1+0+0-1')
        panel = config.panels[0]

        assert panel.style.ylabel == r"$\sigma$ (rad)"
        assert panel.style.xlabel == "Time (years)"
        assert panel.style.title == "Resonant angle"
        assert panel.style.color == 'black'
        assert panel.style.marker == ','
        assert panel.style.linestyle == ''

    def test_simple_preset_title_template(self):
        """Test title template in simple preset."""
        config = create_simple_preset('4J-2S-1+0+0-1')
        assert config.plot_title == '{body_name}, resonance = {resonance}, status = {status}'


class TestFullPreset:
    """Test suite for full preset configuration."""

    def test_full_preset_structure(self):
        """Test basic structure of full preset."""
        resonance_key = '4J-2S-1+0+0-1'
        config = create_full_preset(resonance_key)

        assert isinstance(config, PlotConfig)
        assert config.figsize == (10, 12)
        assert len(config.panels) == 8  # 5 time-series + 3 periodograms

    def test_full_preset_panel_keys(self):
        """Test that all expected panels are present."""
        config = create_full_preset('4J-2S-1+0+0-1')
        panel_keys = [p.key for p in config.panels]

        expected_keys = [
            'angle',
            'angle_unwrapped',
            'axis',
            'eccentricity',
            'inclination',
            'periodogram_angle',
            'periodogram_axis',
            'periodogram_ecc',
        ]
        assert panel_keys == expected_keys

    def test_full_preset_angle_panel(self):
        """Test resonant angle panel configuration."""
        resonance_key = '4J-2S-1+0+0-1'
        config = create_full_preset(resonance_key)
        panel = config.panels[0]

        assert panel.key == 'angle'
        assert panel.data_column == f'{resonance_key}_angle'
        assert panel.x_column == 'times'
        assert panel.required is True
        assert panel.style.ylabel == r"$\sigma$ (rad)"
        assert panel.style.title == "Resonant angle"

    def test_full_preset_unwrapped_angle_panel(self):
        """Test unwrapped resonant angle panel."""
        resonance_key = '4J-2S-1+0+0-1'
        config = create_full_preset(resonance_key)
        panel = config.panels[1]

        assert panel.key == 'angle_unwrapped'
        assert panel.data_column == f'{resonance_key}_angle_filtered_unwrapped'
        assert panel.required is False

    def test_full_preset_axis_panel(self):
        """Test semi-major axis panel with fallback."""
        config = create_full_preset('4J-2S-1+0+0-1')
        panel = config.panels[2]

        assert panel.key == 'axis'
        assert panel.data_column == 'a_filtered'
        assert panel.fallback_column == 'a'
        assert panel.required is True

    def test_full_preset_eccentricity_panel(self):
        """Test eccentricity panel."""
        config = create_full_preset('4J-2S-1+0+0-1')
        panel = config.panels[3]

        assert panel.key == 'eccentricity'
        assert panel.data_column == 'e'
        assert panel.x_column == 'times'
        assert panel.required is True

    def test_full_preset_periodogram_panels(self):
        """Test periodogram panel configurations."""
        resonance_key = '4J-2S-1+0+0-1'
        config = create_full_preset(resonance_key)

        # Angle periodogram
        angle_perio = config.panels[5]
        assert angle_perio.key == 'periodogram_angle'
        assert angle_perio.data_column == f'{resonance_key}_power'
        assert angle_perio.x_column == f'{resonance_key}_frequency'
        assert angle_perio.required is False

        # Axis periodogram
        axis_perio = config.panels[6]
        assert axis_perio.key == 'periodogram_axis'
        assert axis_perio.data_column == 'a_power'
        assert axis_perio.x_column == 'a_frequency'
        assert axis_perio.required is False

        # Eccentricity periodogram
        ecc_perio = config.panels[7]
        assert ecc_perio.key == 'periodogram_ecc'
        assert ecc_perio.data_column == 'e_power'
        assert ecc_perio.x_column == 'e_frequency'
        assert ecc_perio.required is False

    def test_full_preset_reference_lines(self):
        """Test reference lines in periodogram panels."""
        config = create_full_preset('4J-2S-1+0+0-1')

        for panel in config.panels[5:8]:  # All periodogram panels
            assert len(panel.style.reference_lines) == 2
            # Red line at 0.05
            assert panel.style.reference_lines[0]['y'] == 0.05
            assert panel.style.reference_lines[0]['color'] == 'r'
            # Green line at 0.1
            assert panel.style.reference_lines[1]['y'] == 0.1
            assert panel.style.reference_lines[1]['color'] == 'g'

    def test_full_preset_periodogram_styling(self):
        """Test periodogram panel styling."""
        config = create_full_preset('4J-2S-1+0+0-1')
        perio_panel = config.panels[5]

        assert perio_panel.style.linestyle == '-'
        assert perio_panel.style.marker == ''
        assert perio_panel.style.color == 'black'

    def test_full_preset_timeseries_styling(self):
        """Test time-series panel styling."""
        config = create_full_preset('4J-2S-1+0+0-1')
        ts_panel = config.panels[0]

        assert ts_panel.style.linestyle == ''
        assert ts_panel.style.marker == ','
        assert ts_panel.style.color == 'black'


class TestGetPreset:
    """Test suite for get_preset function."""

    def test_get_simple_preset(self):
        """Test retrieving simple preset by name."""
        resonance_key = '4J-2S-1+0+0-1'
        config = get_preset('simple', resonance_key)

        assert isinstance(config, PlotConfig)
        assert len(config.panels) == 1
        assert config.panels[0].key == 'angle'

    def test_get_full_preset(self):
        """Test retrieving full preset by name."""
        resonance_key = '4J-2S-1+0+0-1'
        config = get_preset('full', resonance_key)

        assert isinstance(config, PlotConfig)
        assert len(config.panels) == 8

    def test_get_unknown_preset(self):
        """Test that unknown preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset 'unknown'"):
            get_preset('unknown', '4J-2S-1+0+0-1')

    def test_get_preset_with_different_resonances(self):
        """Test that preset works with different resonance keys."""
        resonances = ['4J-2S-1+0+0-1', '3J-1S-1+0+0-1', '2J-1S-1+0+0-1']

        for res_key in resonances:
            simple = get_preset('simple', res_key)
            assert simple.panels[0].data_column == f'{res_key}_angle'

            full = get_preset('full', res_key)
            assert full.panels[0].data_column == f'{res_key}_angle'
            assert full.panels[5].data_column == f'{res_key}_power'
