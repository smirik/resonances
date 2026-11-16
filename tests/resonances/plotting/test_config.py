"""Tests for plotting configuration classes."""

from resonances.plotting.config import PlotConfig, Panel, StyleConfig


class TestStyleConfig:
    """Test suite for StyleConfig dataclass."""

    def test_style_config_defaults(self):
        """Test default values for StyleConfig."""
        style = StyleConfig()
        assert style.color == 'black'
        assert style.marker == ','
        assert style.linestyle == ''
        assert style.linewidth == 1.0
        assert style.markersize is None
        assert style.ylabel == ''
        assert style.xlabel == ''
        assert style.title == ''
        assert style.xlim is None
        assert style.ylim is None
        assert style.grid is False
        assert style.reference_lines == []
        assert style.custom_params == {}

    def test_style_config_custom_values(self):
        """Test StyleConfig with custom values."""
        style = StyleConfig(
            color='red',
            marker='o',
            linestyle='-',
            ylabel='Test Y',
            xlabel='Test X',
            title='Test Title',
            xlim=(0, 10),
            ylim=(0, 1),
            grid=True,
        )
        assert style.color == 'red'
        assert style.marker == 'o'
        assert style.linestyle == '-'
        assert style.ylabel == 'Test Y'
        assert style.xlabel == 'Test X'
        assert style.title == 'Test Title'
        assert style.xlim == (0, 10)
        assert style.ylim == (0, 1)
        assert style.grid is True

    def test_style_config_merge(self):
        """Test merging StyleConfig objects."""
        base = StyleConfig(color='black', marker=',', ylabel='Base Y', title='Base Title')
        override = StyleConfig(color='red', ylabel='Override Y')
        merged = base.merge_with(override)

        assert merged.color == 'red'  # Override wins
        assert merged.marker == ','  # Base default preserved
        assert merged.ylabel == 'Override Y'  # Override wins
        assert merged.title == 'Base Title'  # Base preserved

    def test_style_config_reference_lines(self):
        """Test reference lines configuration."""
        style = StyleConfig(
            reference_lines=[
                {'type': 'axhline', 'y': 0.05, 'color': 'r', 'linestyle': '--'},
                {'type': 'axhline', 'y': 0.1, 'color': 'g', 'linestyle': '--'},
            ]
        )
        assert len(style.reference_lines) == 2
        assert style.reference_lines[0]['y'] == 0.05
        assert style.reference_lines[1]['y'] == 0.1


class TestPanel:
    """Test suite for Panel dataclass."""

    def test_panel_minimal(self):
        """Test Panel with minimal required parameters."""
        panel = Panel(key='test_panel', data_column='a')
        assert panel.key == 'test_panel'
        assert panel.data_column == 'a'
        assert panel.x_column == 'times'  # Default
        assert panel.fallback_column is None
        assert panel.transform is None
        assert isinstance(panel.style, StyleConfig)
        assert panel.required is True
        assert panel.enabled is True

    def test_panel_with_fallback(self):
        """Test Panel with fallback column."""
        panel = Panel(key='axis', data_column='a_filtered', fallback_column='a')
        assert panel.data_column == 'a_filtered'
        assert panel.fallback_column == 'a'

    def test_panel_with_custom_style(self):
        """Test Panel with custom style."""
        style = StyleConfig(color='blue', ylabel='Custom Y')
        panel = Panel(key='test', data_column='e', style=style)
        assert panel.style.color == 'blue'
        assert panel.style.ylabel == 'Custom Y'

    def test_panel_dict_style_initialization(self):
        """Test Panel with dict-style style initialization."""
        panel = Panel(key='test', data_column='e', style={'color': 'red', 'ylabel': 'Eccentricity'})
        assert isinstance(panel.style, StyleConfig)
        assert panel.style.color == 'red'
        assert panel.style.ylabel == 'Eccentricity'

    def test_panel_optional(self):
        """Test optional panel configuration."""
        panel = Panel(key='optional', data_column='missing_data', required=False)
        assert panel.required is False

    def test_panel_disabled(self):
        """Test disabled panel."""
        panel = Panel(key='disabled', data_column='data', enabled=False)
        assert panel.enabled is False


class TestPlotConfig:
    """Test suite for PlotConfig dataclass."""

    def test_plot_config_defaults(self):
        """Test default values for PlotConfig."""
        config = PlotConfig()
        assert config.figsize == (10, 12)
        assert config.dpi == 100
        assert config.panels == []
        assert isinstance(config.global_style, StyleConfig)
        assert config.title_fontsize == 14
        assert config.label_fontsize == 14
        assert config.plot_title == '{body_name}, resonance = {resonance}, status = {status}'

    def test_plot_config_custom_values(self):
        """Test PlotConfig with custom values."""
        config = PlotConfig(figsize=(8, 10), dpi=150, title_fontsize=16, plot_title='Custom Title')
        assert config.figsize == (8, 10)
        assert config.dpi == 150
        assert config.title_fontsize == 16
        assert config.plot_title == 'Custom Title'

    def test_add_panel(self):
        """Test adding panels to configuration."""
        config = PlotConfig()
        panel1 = Panel(key='panel1', data_column='a')
        panel2 = Panel(key='panel2', data_column='e')

        config.add_panel(panel1)
        assert len(config.panels) == 1
        assert config.panels[0].key == 'panel1'

        config.add_panel(panel2)
        assert len(config.panels) == 2
        assert config.panels[1].key == 'panel2'

    def test_add_panel_chaining(self):
        """Test method chaining when adding panels."""
        config = PlotConfig()
        result = config.add_panel(Panel(key='p1', data_column='a'))
        assert result is config  # Returns self for chaining

        config.add_panel(Panel(key='p2', data_column='e')).add_panel(Panel(key='p3', data_column='inc'))
        assert len(config.panels) == 3

    def test_get_enabled_panels(self):
        """Test getting only enabled panels."""
        config = PlotConfig()
        config.add_panel(Panel(key='enabled1', data_column='a', enabled=True))
        config.add_panel(Panel(key='disabled', data_column='e', enabled=False))
        config.add_panel(Panel(key='enabled2', data_column='inc', enabled=True))

        enabled = config.get_enabled_panels()
        assert len(enabled) == 2
        assert enabled[0].key == 'enabled1'
        assert enabled[1].key == 'enabled2'

    def test_merge_panel_styles(self):
        """Test merging global and panel-specific styles."""
        global_style = StyleConfig(color='black', marker=',', ylabel='Global Y')
        config = PlotConfig(global_style=global_style)

        panel_style = StyleConfig(color='red', ylabel='Panel Y')
        panel = Panel(key='test', data_column='a', style=panel_style)

        merged = config.merge_panel_styles(panel)
        assert merged.color == 'red'  # Panel override
        assert merged.marker == ','  # Global default
        assert merged.ylabel == 'Panel Y'  # Panel override

    def test_format_title(self):
        """Test title formatting with variables."""
        config = PlotConfig(plot_title='{body_name}, resonance = {resonance}, status = {status}')
        formatted = config.format_title(body_name='Test Body', resonance='4J-2S-1', status='2')
        assert formatted == 'Test Body, resonance = 4J-2S-1, status = 2'

    def test_format_title_missing_keys(self):
        """Test title formatting with missing keys."""
        config = PlotConfig(plot_title='{body_name} - {missing_key}')
        formatted = config.format_title(body_name='Test Body')
        # Should return unformatted string if key is missing
        assert formatted == '{body_name} - {missing_key}'

    def test_format_title_fixed_string(self):
        """Test title formatting with fixed string (no variables)."""
        config = PlotConfig(plot_title='Fixed Title')
        formatted = config.format_title(body_name='Test')
        assert formatted == 'Fixed Title'
