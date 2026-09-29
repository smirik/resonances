"""Plot kinds and plot options in SimulationConfig."""

import pytest

from resonances.config import config
from resonances.simulation.config import PLOT_KINDS, SimulationConfig


def make(**kwargs):
    return SimulationConfig(save=None, save_path='cache/tests/plot-config', plot_path='cache/tests/plot-config', **kwargs)


def test_plots_default_and_forms():
    assert make().plots == ['evolution']
    assert make(plots='combined').plots == ['combined']
    assert make(plots=('fair', 'portrait')).plots == ['fair', 'portrait']
    assert make(plots=list(PLOT_KINDS)).plots == list(PLOT_KINDS)


def test_unknown_plot_kind_is_an_error():
    with pytest.raises(ValueError, match='phase_portrait'):
        make(plots=['evolution', 'phase_portrait'])


def test_plot_options_kwarg():
    assert make(plot_options={'style': 'paper'}).plot_options['style'] == 'paper'
    with pytest.raises(ValueError):
        make(plot_options={'bogus': 1})


def test_plot_options_from_env_json():
    original = config.get('PLOT_OPTIONS', '')
    config.set('PLOT_OPTIONS', '{"raw": "none", "portrait": {"y": "axis"}}')
    try:
        options = make().plot_options
    finally:
        config.set('PLOT_OPTIONS', original)
    assert options['raw'] == 'none'
    assert options['portrait']['y'] == 'axis'
    assert options['style'] == 'screen'
