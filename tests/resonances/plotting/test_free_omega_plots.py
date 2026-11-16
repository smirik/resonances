"""Tests for the free-omega figures."""

import matplotlib

matplotlib.use('Agg')

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.body import Body  # noqa: E402
from resonances.data.const import PLANETARY_FREQUENCIES  # noqa: E402
from resonances.plotting import EccentricityVectorPlotter, FreeOmegaPlotter  # noqa: E402
from resonances.plotting.base import time_unit  # noqa: E402
from resonances.plotting.style import time_cmap  # noqa: E402
from resonances.secular.free_elements import RAD_PER_ARCSEC, free_elements  # noqa: E402


def _wave(times, amplitude, arcsec_per_year):
    return amplitude * np.exp(1j * arcsec_per_year * RAD_PER_ARCSEC * times)


def _body(free_rate_i: float, name: str = 'synthetic') -> Body:
    """A body whose free pericentre turns at -12 arcsec/yr and node at `free_rate_i`.

    Equal rates make the free omega librate, different ones make it circulate.
    """
    times = np.arange(0.0, 4e6, 500.0)
    z_e = _wave(times, 0.10, PLANETARY_FREQUENCIES['g5']) + _wave(times, 0.20, -12.0)
    z_i = _wave(times, 0.05, PLANETARY_FREQUENCIES['s6']) + _wave(times, 0.30, free_rate_i)

    body = Body()
    body.name = name
    body.times = times * 2 * np.pi
    body.ecc, body.inc = np.abs(z_e), 2.0 * np.arcsin(np.abs(z_i))
    body.Omega = np.angle(z_i)
    body.omega = np.angle(z_e) - body.Omega
    return body


class TestFreeOmegaPlotter:
    @pytest.mark.parametrize('method', ['plot_drift', 'plot_portrait', 'plot_vector'])
    def test_each_figure_is_a_single_panel_and_saves(self, method, tmp_path):
        body = _body(free_rate_i=-20.0)
        body.build_free_elements(sampling_years=None)

        plotter = getattr(FreeOmegaPlotter.from_body(body), method)()

        assert len(plotter._axes) == 1
        path = tmp_path / f'{method}.png'
        plotter.save(str(path))
        assert path.exists()
        plotter.close()

    def test_title_carries_the_body_name_and_nothing_else(self):
        # Rates and verdicts belong in summary.csv, not on the figure.
        body = _body(free_rate_i=-20.0, name='797204')
        body.build_free_elements(sampling_years=None)

        title = FreeOmegaPlotter.from_body(body).plot_drift()._axes[0].get_title()

        assert title == '797204 - free $\\omega$'

    def test_drift_starts_at_the_origin_and_ends_at_the_integration_end(self):
        body = _body(free_rate_i=-20.0)
        elements = body.build_free_elements(sampling_years=None)
        grid = elements.omega_times
        span = grid[-1] - grid[0]

        ax = FreeOmegaPlotter.from_body(body).plot_drift()._axes[0]

        divisor, _ = time_unit(span)
        assert ax.get_xlim() == (0.0, span / divisor)
        # Both curves start at zero, and this one only accumulates upwards.
        assert ax.get_ylim()[0] == 0.0

    def test_osculating_curve_is_drawn_on_the_free_omega_grid(self):
        # The free elements are resampled and then thinned again by the origin masks, so
        # the comparison curve has to be interpolated onto that grid rather than assumed
        # to be the same length as either the integration or the resampled series.
        body = _body(free_rate_i=-20.0)
        body.build_free_elements(sampling_years=2000.0)

        plotter = FreeOmegaPlotter.from_body(body)

        assert len(plotter._omega_osculating) == len(body.free_elements.omega_times)
        assert len(plotter._omega_osculating) < len(body.times)

    def test_a_body_without_free_elements_renders_nothing(self):
        body = Body()
        body.name = 'empty'

        assert FreeOmegaPlotter.from_body(body).plot_drift()._figure is None
        assert FreeOmegaPlotter.from_body(body).plot_portrait()._figure is None
        assert FreeOmegaPlotter.from_body(body).plot_vector()._figure is None

    def test_portrait_covers_the_whole_angle_range_from_zero_eccentricity(self):
        body = _body(free_rate_i=-20.0)
        body.build_free_elements(sampling_years=None)

        ax = FreeOmegaPlotter.from_body(body).plot_portrait()._axes[0]

        assert ax.get_xlim() == (0.0, 360.0)
        assert ax.get_ylim()[0] == 0.0

    @pytest.mark.parametrize('draw', ['ecc_vector', 'free_vector', 'free_portrait'])
    def test_time_colour_bar_matches_the_mmr_plots(self, draw):
        # 4 Myr of data: the bar is in Myr, with the MMR plots' colour map and wording.
        body = _body(free_rate_i=-20.0)
        body.build_free_elements(sampling_years=None)
        if draw == 'ecc_vector':
            plotter = EccentricityVectorPlotter.from_body(body).plot()
        else:
            plotter = getattr(FreeOmegaPlotter.from_body(body), 'plot_' + draw.split('_')[1])()

        scatter = plotter._axes[0].collections[0]
        assert scatter.get_cmap() is time_cmap()
        assert scatter.colorbar.ax.get_ylabel() == 'Time (Myr)'
        plotter.close()

    def test_vector_honours_the_osculating_window(self):
        # The two portraits are only comparable if they are drawn on the same scale, so an
        # explicit limit must win over the free vector's own extent.
        body = _body(free_rate_i=-20.0)
        body.build_free_elements(sampling_years=None)
        limit = EccentricityVectorPlotter.from_body(body).extent()

        ax = FreeOmegaPlotter.from_body(body).plot_vector(limit=limit)._axes[0]

        assert ax.get_xlim() == pytest.approx((-limit, limit))
        assert ax.get_ylim() == pytest.approx((-limit, limit))

    def test_a_clustered_baseline_still_draws(self):
        # A 1 Myr baseline merges g8 into its neighbours, so the basis is reduced. The
        # figures carry no verdict either way — they must simply draw.
        times = np.arange(0.0, 1e6, 500.0)
        z_e = _wave(times, 0.10, PLANETARY_FREQUENCIES['g5'])
        z_i = _wave(times, 0.30, -20.0)
        elements = free_elements(
            times,
            np.abs(z_e),
            2.0 * np.arcsin(np.abs(z_i)),
            np.angle(z_i),
            np.angle(z_e) - np.angle(z_i),
            sampling_years=None,
        )

        plotter = FreeOmegaPlotter.from_elements(elements, body_name='short').plot_drift()

        assert plotter._axes[0].get_title() == 'short - free $\\omega$'
        plotter.close()
