"""MMR diagnostic plots: options, diagnostics geometry, figures and a real short run."""

import matplotlib

matplotlib.use('Agg')

import re  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import resonances  # noqa: E402
import tests.tools as tools  # noqa: E402
from resonances.plotting import MMRDiagnostics, MMRPlotter  # noqa: E402
from resonances.plotting.mmr_plots import (  # noqa: E402
    PlanetSeries,
    centre_line,
    portrait_coordinates,
    envelope_bins,
    recurrence_points,
)
from resonances.resonance.oscillations import Cycle  # noqa: E402
from resonances.plotting.style import DEFAULT_PLOT_OPTIONS, PlotStyle, PortraitY, SeriesMode, resolve_plot_options  # noqa: E402
from resonances.simulation.config import MMR_PLOT_KINDS  # noqa: E402


def diagnostics(centre=np.pi, turns=3, n=4000, years=4000.0, planet=True, focus=None):
    """A libration about `centre` with period 400 yr, unwrapped `turns` whole turns away."""
    t = np.linspace(0, years, n)
    sigma = centre + 2 * np.pi * turns + 0.6 * np.sin(2 * np.pi * t / 400)
    longitude = 2 * np.pi * t / 3.2776**1.5
    return MMRDiagnostics(
        times=t,
        sigma=np.mod(sigma, 2 * np.pi),
        sigma_raw=sigma,
        sigma_filtered=sigma,
        axis=3.2776 + 0.002 * np.cos(2 * np.pi * t / 400),
        axis_filtered=3.2776 + 0.002 * np.cos(2 * np.pi * t / 400),
        body_name='synthetic',
        resonance='2J-1+0-1',
        label='2J-1',
        n_planets=1,
        mean_anomaly=np.mod(longitude, 2 * np.pi),
        longitude=longitude,
        planet=PlanetSeries('Jupiter', (sigma + longitude) / 2, np.full(n, 5.2)) if planet else None,
        focus=focus,
    )


def letters(figure):
    return sorted(t.get_text() for ax in figure.axes for t in ax.texts if re.fullmatch(r'\([a-h]\)', t.get_text()))


class TestOptions:
    def test_defaults(self):
        assert resolve_plot_options() == DEFAULT_PLOT_OPTIONS

    def test_strings_become_enum_members(self):
        # Strings from kwargs, .env JSON or a restored simulation.json compare equal either way.
        options = resolve_plot_options({'style': 'paper', 'portrait': {'y': 'axis'}})
        assert options['style'] is PlotStyle.PAPER and options['style'] == 'paper'
        assert options['portrait']['y'] is PortraitY.AXIS

    def test_nested_merge_keeps_other_defaults(self):
        options = resolve_plot_options({'portrait': {'y': 'axis'}, 'style': 'paper'})
        assert options['portrait'] == {'y': 'axis'}
        assert options['style'] == 'paper'
        assert options['recurrence'] == DEFAULT_PLOT_OPTIONS['recurrence']

    @pytest.mark.parametrize(
        'options, message',
        [
            ({'colour': 'red'}, "Unknown plot option 'colour'"),
            ({'recurrence': {'points': 10}}, "Unknown plot option 'recurrence.points'"),
            ({'style': 'poster'}, "'style' must be one of"),
            ({'portrait': {'y': 'e'}}, "'portrait.y' must be one of"),
            ({'raw': 'dots'}, "'raw' must be one of"),
            ({'recurrence': 700}, 'must be a dict'),
        ],
    )
    def test_invalid(self, options, message):
        with pytest.raises(ValueError, match=re.escape(message)):
            resolve_plot_options(options)


class TestDiagnostics:
    def test_whole_record_is_the_default_focus(self):
        d = diagnostics()
        assert d.mask.all() and not d.has_focus
        # 4000 yr of a 400-yr libration: maxima at 100, 500, ..., 3700 -> 9 full cycles.
        assert len(d.cycles) == 9

    def test_focus_restricts_cycles(self):
        d = diagnostics(focus=(0, 2000))
        assert d.has_focus
        # Maxima at 100, 500, 900, 1300, 1700 inside [0, 2000] -> 4 cycles.
        assert len(d.cycles) == 4

    def test_fair_availability(self):
        assert diagnostics().fair_unavailable_reason() is None
        assert 'save_planets' in diagnostics(planet=False).fair_unavailable_reason()
        three_body = diagnostics()
        three_body.n_planets = 2
        assert 'two-body' in three_body.fair_unavailable_reason()

    def test_envelope_bins(self):
        # 400-yr cycles over 4000 yr = 10 cycles. On 900 px with a 2.5-px line a cycle gets
        # 90 px >= 4 x 2.5: lines. On 30 px it gets 3 px: bands of two cycles, 10 / 2 = 5.
        d = diagnostics()
        assert envelope_bins(d, SeriesMode.AUTO, 900, 2.5) == 0
        assert envelope_bins(d, SeriesMode.AUTO, 30, 2.5) == 5
        assert envelope_bins(d, SeriesMode.ENVELOPE, 900, 2.5) == 5
        assert envelope_bins(d, SeriesMode.LINES, 30, 2.5) == 0

    def test_envelope_contains_every_sample(self):
        from resonances.plotting.mmr_plots import _envelope

        fig, ax = plt.subplots()
        t = np.linspace(0, 100, 1001)
        y = np.sin(2 * np.pi * t / 3.0)
        _envelope(ax, t, y, 10)
        band = ax.collections[0].get_paths()[0].vertices
        plt.close(fig)
        assert band[:, 1].min() == pytest.approx(y.min()) and band[:, 1].max() == pytest.approx(y.max())

    @pytest.mark.parametrize('centre, degrees', [(np.pi, 180.0), (0.0, 0.0)])
    def test_portrait_keeps_the_libration_centre(self, centre, degrees):
        # The unwrapped angle sits 3 turns away; only whole turns are removed, so the
        # libration is drawn about its true centre (not about zero), with its 2 x 0.6 rad width.
        x, _, x_raw, _ = portrait_coordinates(diagnostics(centre=centre, turns=3))
        assert np.median(x) == pytest.approx(degrees, abs=1.0)
        assert np.ptp(x) == pytest.approx(2 * np.degrees(0.6), rel=0.01)
        assert np.array_equal(x_raw, x)  # raw equals filtered in this synthetic record

    def test_portrait_rate_is_in_degrees_per_display_unit(self):
        # sigma_dot = 0.6 * 2 pi / 400 rad/yr at its maximum. 4000 yr stays in yr; over
        # 40 kyr the display unit is kyr and the same rate reads 1000 times larger.
        d = diagnostics()
        _, y, _, _ = portrait_coordinates(d, PortraitY.RATE)
        assert d.unit == 'yr'
        assert np.max(y) == pytest.approx(np.degrees(0.6 * 2 * np.pi / 400), rel=1e-3)
        long = diagnostics(n=40000, years=40000.0)
        _, y_long, _, _ = portrait_coordinates(long, PortraitY.RATE)
        assert long.unit == 'kyr'
        assert np.max(y_long) == pytest.approx(np.degrees(0.6 * 2 * np.pi / 400) * 1000, rel=1e-3)
        _, y_axis, _, _ = portrait_coordinates(d, PortraitY.AXIS)
        assert np.ptp(y_axis) == pytest.approx(4.0, rel=1e-3)  # a = a0 + 0.002 cos -> 4e-3 au

    def test_portrait_drops_the_filter_edges(self):
        # edge = 500 yr of 4000: samples within 500 yr of either end leave the portrait
        # (3000 of 4000 years remain); cycles and time series keep the whole record.
        d = diagnostics()
        d.edge = 500.0
        kept = d.times[d.portrait_mask]
        assert kept.min() >= 500 and kept.max() <= 3500
        assert d.portrait_mask.sum() == pytest.approx(0.75 * len(d.times), rel=0.01)
        assert len(portrait_coordinates(d)[0]) == d.portrait_mask.sum()
        d.edge = 1500.0  # would leave a quarter of the record: the portrait keeps it all
        assert d.portrait_mask.all()

    def test_recurrence_points(self):
        # 400-yr cycles over 4000 yr: 8 per cycle = 80, raised to the 700 floor.
        d = diagnostics()
        assert recurrence_points(d) == 700
        assert recurrence_points(d, 300) == 300
        # 50 kyr of 150-yr cycles: 8 * 50000 / 150 = 2667, capped at 2500.
        t = np.linspace(0, 50000, 50000)
        sigma = 0.3 * np.sin(2 * np.pi * t / 150)
        dense = MMRDiagnostics(
            times=t,
            sigma=np.mod(sigma, 2 * np.pi),
            sigma_raw=sigma,
            sigma_filtered=sigma,
            axis=np.ones_like(t),
            axis_filtered=np.ones_like(t),
        )
        assert recurrence_points(dense) == 2500
        # 400-yr cycles over the same span: 8 * 50000 / 400 = 1000.
        sigma = 0.3 * np.sin(2 * np.pi * t / 400)
        medium = MMRDiagnostics(
            times=t,
            sigma=np.mod(sigma, 2 * np.pi),
            sigma_raw=sigma,
            sigma_filtered=sigma,
            axis=np.ones_like(t),
            axis_filtered=np.ones_like(t),
        )
        assert recurrence_points(medium) == pytest.approx(1000, abs=3)

    def test_backward_integration_is_put_in_time_order(self):
        # Times from 0 down to -4000 yr: the diagnostics see increasing time, so the same
        # 9 cycles are found with positive periods, and the rate keeps its physical sign.
        forward = diagnostics()
        t = -forward.times
        sigma = np.pi + 0.6 * np.sin(2 * np.pi * t / 400)
        backward = MMRDiagnostics(
            times=t,
            sigma=np.mod(sigma, 2 * np.pi),
            sigma_raw=sigma,
            sigma_filtered=sigma,
            axis=np.full_like(t, 2.0),
            axis_filtered=np.full_like(t, 2.0),
        )
        assert backward.times[0] == -4000 and backward.times[-1] == 0
        assert len(backward.cycles) == 9 and all(c.period == pytest.approx(400, rel=1e-2) for c in backward.cycles)
        expected_rate = 0.6 * 2 * np.pi / 400 * np.cos(2 * np.pi * backward.times / 400)
        assert backward.rate[100:-100] == pytest.approx(expected_rate[100:-100], abs=1e-6)
        assert MMRPlotter(backward).plot('combined')


class TestCentreLine:
    def cycle(self, start, end):
        return Cycle(start=start, minimum=(start + end) / 2, end=end, diameter=1.0, centre=0.0, closure_error=0.0)

    def test_neighbours_are_joined(self):
        cycles = [self.cycle(0, 1), self.cycle(1, 2), self.cycle(2, 3)]
        (mid,) = centre_line(cycles, [0.5, 1.5, 2.5])
        assert mid.tolist() == [0.5, 1.5, 2.5]

    def test_gap_breaks_the_line(self):
        # The third cycle starts 2 periods after the second ends.
        cycles = [self.cycle(0, 1), self.cycle(1, 2), self.cycle(4, 5)]
        (mid,) = centre_line(cycles, [0.5, 1.5, 4.5])
        assert np.isnan(mid).tolist() == [False, False, True, False]

    def test_bridging_cycle_stands_alone(self):
        # Median period 1; the 10-yr cycle bridges circulation: breaks on both sides.
        cycles = [self.cycle(0, 1), self.cycle(1, 2), self.cycle(2, 12), self.cycle(12, 13)]
        mid, values = centre_line(cycles, [0.5, 1.5, 7.0, 12.5], [1, 2, 3, 4])
        assert np.isnan(mid).tolist() == [False, False, True, False, True, False]
        assert values[~np.isnan(values)].tolist() == [1, 2, 3, 4]


class TestFigures:
    def test_combined_two_body_has_eight_panels(self):
        plotter = MMRPlotter(diagnostics()).plot_combined()
        assert letters(plotter.figure) == [f'({c})' for c in 'abcdefgh']
        assert plotter.figure._suptitle.get_text() == 'Asteroid synthetic, 2J-1'
        plotter.close()

    def test_combined_without_fair_has_seven_panels(self):
        with pytest.warns(UserWarning):
            plotter = MMRPlotter(diagnostics(planet=False)).plot_combined()
        assert letters(plotter.figure) == [f'({c})' for c in 'abcdefg']
        plotter.close()

    def test_paper_style(self, tmp_path):
        plotter = MMRPlotter(diagnostics(), {'style': 'paper'}).plot_combined()
        assert plotter.figure.get_size_inches()[0] == pytest.approx(7.09)
        assert plotter.figure._suptitle is None  # the caption carries the title
        plotter.save(tmp_path / 'f.pdf')
        plotter.close()

    @pytest.mark.parametrize('style', ['screen', 'paper'])
    @pytest.mark.parametrize('planet', [True, False])
    def test_combined_geometry(self, style, planet, recwarn):
        """Top row: the three time series together are as tall as the square recurrence
        (and FAIR) panels. Bottom row: three equal squares. Measured on the drawn axes."""
        plotter = MMRPlotter(diagnostics(planet=planet), {'style': style}).plot_combined()
        fig = plotter.figure
        w, h = fig.get_size_inches()

        def inches(ax):
            box = ax.get_position()
            return box.x0 * w, box.y0 * h, box.width * w, box.height * h

        panels = [ax for ax in fig.axes if any(re.fullmatch(r'\([a-h]\)', t.get_text()) for t in ax.texts)]
        series, squares = panels[:3], panels[3:]
        top = max(inches(ax)[1] + inches(ax)[3] for ax in series) - min(inches(ax)[1] for ax in series)
        upper, lower = squares[: len(squares) - 3], squares[-3:]
        for ax in upper:
            x, y, width, height = inches(ax)
            assert width == pytest.approx(height) == pytest.approx(top)
            assert y == pytest.approx(inches(series[2])[1])  # bottoms aligned
        sides = [inches(ax)[2:] for ax in lower]
        for width, height in sides:
            assert width == pytest.approx(height) == pytest.approx(sides[0][0])
        assert len(upper) == (2 if planet else 1)
        plotter.close()

    def test_kinds_match_the_config(self):
        assert set(MMRPlotter.KINDS) == set(MMR_PLOT_KINDS)

    @pytest.mark.parametrize('kind', MMR_PLOT_KINDS)
    def test_every_kind_draws(self, kind):
        plotter = MMRPlotter(diagnostics())
        assert plotter.plot(kind) is True
        assert plotter.figure is not None
        plotter.close()

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match='phase_portrait'):
            MMRPlotter(diagnostics()).plot('phase_portrait')

    def test_fair_unavailable_draws_nothing(self):
        with pytest.warns(UserWarning, match='save_planets'):
            assert MMRPlotter(diagnostics(planet=False)).plot('fair') is False

    def test_no_cycles_message(self):
        t = np.linspace(0, 1000, 1000)
        d = MMRDiagnostics(
            times=t, sigma=np.mod(t, 2 * np.pi), sigma_raw=t, sigma_filtered=t, axis=np.full(1000, 2.0), axis_filtered=np.full(1000, 2.0)
        )
        plotter = MMRPlotter(d).plot_cycles()
        texts = [t.get_text() for ax in plotter.figure.axes for t in ax.texts]
        assert texts.count('No complete cycles') == 2
        plotter.close()


@pytest.mark.slow
class TestRealRun:
    """A short integration through Simulation.run(), as a user would call it."""

    def test_all_mmr_kinds(self):
        sim = resonances.Simulation(
            name='test_mmr_plots',
            integrator='SABA(10,6,4)',
            tmax=10000,
            save='all',
            save_planets=True,
            plot='all',
            plots=['evolution', 'combined', 'recurrence', 'fair', 'portrait', 'cycles'],
        )
        sim.create_solar_system()
        sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'), name='three')
        sim.add_body(tools.get_2body_elements_sample(), resonances.TwoBody('1J-1'), name='two')
        with pytest.warns(UserWarning):  # FAIR asked for the three-body argument
            sim.run()

        plot_path = Path(sim.config.plot_path)
        for kind in ('combined', 'recurrence', 'portrait', 'cycles'):
            assert (plot_path / f'three-4J-2S-1+0+0-1-{kind}.png').exists(), kind
            assert (plot_path / f'two-1J-1+0+0-{kind}.png').exists(), kind
        assert (plot_path / 'two-1J-1+0+0-fair.png').exists()
        assert not (plot_path / 'three-4J-2S-1+0+0-1-fair.png').exists()
        assert (plot_path / 'three-4J-2S-1+0+0-1.png').exists()  # evolution is unchanged
