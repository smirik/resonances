"""The clones figure, panel (b): the line drawn is the catalogue resonant axis a_res (`resonance.resonant_axis`), labelled
so, at a_res minus the reference (the nominal's initial a); a_res +- the half-width dashed. By hand: nominal a0 =
2.5020 au, a_res = 2.5000 au, half-width 0.0030 au -> lines at -0.0020, -0.0050 and +0.0010 au."""

import matplotlib

matplotlib.use('Agg')

import pytest  # noqa: E402

from resonances.plotting.clone_plots import EnsemblePlotter  # noqa: E402


def test_panel_b_draws_the_catalogue_a_res():
    members = [{'status': 'resonant', 'a0': 2.5019, 'series': None}, {'status': 'non_resonant', 'a0': 2.5023, 'series': None}]
    plotter = EnsemblePlotter().plot(
        parent='p',
        key='3J-1+0+0-2',
        members=members,
        row={'n_clones': 2},
        nominal_a0=2.5020,
        nominal_status='resonant',
        a_star=2.5000,
        half_width=0.0030,
    )
    ax = plotter._figure.axes[1]
    labels = ax.get_legend_handles_labels()[1]
    assert 'a_res (catalogue)' in labels and "a_res +- half-width (clones' episodes)" in labels
    assert not any('a*' in label or 'exact commensurability' in label for label in labels)
    vertical = sorted(line.get_xdata()[0] for line in ax.lines if len(line.get_xdata()) == 2 and line.get_xdata()[0] == line.get_xdata()[1])
    assert vertical == pytest.approx([-0.0050, -0.0020, 0.0010], abs=1e-12)
    plotter.close()
