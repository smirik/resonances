"""MMR diagnostic plots: options, diagnostics geometry, figures and a real short run."""

import matplotlib

matplotlib.use('Agg')

import re  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import astdys.util  # noqa: E402
import resonances  # noqa: E402
import tests.tools as tools  # noqa: E402
from resonances.plotting import MMRDiagnostics, MMRPlotter  # noqa: E402
from resonances.plotting.mmr_plots import (  # noqa: E402
    PlanetSeries,
    portrait_coordinates,
    recurrence_points,
)
from resonances.plotting.style import DEFAULT_PLOT_OPTIONS, PlotStyle, PortraitY, resolve_plot_options  # noqa: E402


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
        retrograde = diagnostics()
        retrograde.retrograde = True
        assert 'prograde resonances only' in retrograde.fair_unavailable_reason()

    def test_a_retrograde_resonance_has_no_fair(self):
        body = resonances.Body()
        n = 400
        body.times = np.linspace(0, 4000 * 2 * np.pi, n)
        series = np.linspace(0, 1, n)
        body.axis, body.M, body.longitude = np.full(n, 10.8), series, series
        key = '1J-3r0'
        body.angles[key] = body.angles_unwrapped[key] = body.angles_filtered_unwrapped[key] = series
        d = MMRDiagnostics.from_body(body, resonances.create_mmr(key))
        assert d.retrograde is True
        assert d.label == key
        assert 'prograde resonances only' in d.fair_unavailable_reason()

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
        assert MMRPlotter(backward).plot('cycles')


class TestCyclesAxis:
    """1921 2J-1 kind of record: a librates with sigma at 400 yr, 2 milli-au; the body's a low-passed at the run's
    2000-yr cutoff has lost it (here a constant, what that low-pass leaves of a 400-yr wave). The cycles panel folds
    the analysis' own a_f: by hand a - median over a whole cycle of 0.002 cos spans 4 milli-au (+-2)."""

    N, YEARS = 4000, 4000.0

    def body(self, analyses=True):
        from types import SimpleNamespace

        t = np.linspace(0, self.YEARS, self.N)
        sigma = np.pi + 0.6 * np.sin(2 * np.pi * t / 400)
        a_f = 3.2776 + 0.002 * np.cos(2 * np.pi * t / 400)
        raw = a_f + 1e-4 * np.cos(2 * np.pi * t / 7.3)  # the short-period terms of the osculating a
        prep = SimpleNamespace(a_f=a_f, angle_f=sigma)
        return SimpleNamespace(
            name='synthetic',
            times=t * 2 * np.pi,
            angles={'2J-1+0-1': np.mod(sigma, 2 * np.pi)},
            angles_unwrapped={'2J-1+0-1': sigma},
            angles_filtered_unwrapped={'2J-1+0-1': sigma},
            axis=raw,
            axis_filtered=np.full(self.N, 3.2776),
            M=None,
            longitude=None,
            mmr_analyses={'2J-1+0-1': SimpleNamespace(prep=prep)} if analyses else {},
        )

    RESONANCE = type('R', (), {'to_s': lambda self: '2J-1+0-1', 'to_short': lambda self: '2J-1', 'planets_names': ['Jupiter']})()

    def folded_range(self, d):
        fig, (ax_sigma, ax_axis) = plt.subplots(1, 2)
        from resonances.plotting.mmr_plots import draw_cycles
        from resonances.plotting.style import STYLES

        draw_cycles(ax_sigma, ax_axis, d, STYLES[PlotStyle.SCREEN], d.time_norm())
        ys = np.concatenate([segment[:, 1] for segment in ax_axis.collections[0].get_segments()])
        plt.close(fig)
        return float(ys.min()), float(ys.max())

    def test_cycles_fold_the_analysis_a(self):
        body = self.body()
        d = MMRDiagnostics.from_body(body, self.RESONANCE)
        assert np.array_equal(d.axis_cycles, body.mmr_analyses['2J-1+0-1'].prep.a_f)
        low, high = self.folded_range(d)  # each sampled cycle's median is within ~0.02 of 0
        assert high - low == pytest.approx(4.0, abs=0.05) and abs(high + low) < 0.05

    def test_without_the_analysis_arrays_the_raw_a(self):
        d = MMRDiagnostics.from_body(self.body(analyses=False), self.RESONANCE)
        assert np.array_equal(d.axis_cycles, d.axis)
        low, high = self.folded_range(d)  # the 400-yr wave plus up to +-0.1 milli-au of short-period terms
        assert 4.1 < high - low < 4.3

    def test_the_body_low_pass_would_be_flat(self):
        """The defect: folded, `axis_filtered` (the 2000-yr low-pass) is a flat line."""
        d = MMRDiagnostics.from_body(self.body(), self.RESONANCE)
        d.axis_cycles = d.axis_filtered
        assert self.folded_range(d) == (0.0, 0.0)


class TestFigures:
    """The standalone MMR panels; the combined MMR figure is the verdict figure (plotting/mmr_combined.py)."""

    def test_combined_is_not_an_mmr_panel(self):
        assert 'combined' not in MMRPlotter.KINDS
        with pytest.raises(ValueError, match='Unknown MMR plot kind'):
            MMRPlotter(diagnostics()).plot('combined')

    def test_retrograde_fair_is_refused(self):
        d = diagnostics()
        d.retrograde = True
        with pytest.warns(UserWarning, match='prograde resonances only'):
            assert MMRPlotter(d).plot('fair') is False

    def test_paper_style(self, tmp_path):
        plotter = MMRPlotter(diagnostics(), {'style': 'paper'})
        plotter.plot('portrait')
        assert plotter.figure._suptitle is None  # the caption carries the title
        plotter.save(tmp_path / 'f.pdf')
        assert (tmp_path / 'f.pdf').stat().st_size > 0
        plotter.close()

    def test_recurrence_is_built_once_for_all_figures(self):
        d = diagnostics()
        plotter = MMRPlotter(d)
        plotter.plot('recurrence')
        first = d.recurrence(recurrence_points(d), 2)
        plotter.plot('recurrence')
        assert d.recurrence(recurrence_points(d), 2) is first
        plotter.close()

    @pytest.mark.parametrize('kind', MMRPlotter.KINDS)
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

    def test_all_mmr_kinds(self, tmp_path):
        sim = resonances.Simulation(
            name='test_mmr_plots',
            date=astdys.util.convert_mjd_to_datetime(60000),  # the epoch of the sample elements
            integrator='SABA(10,6,4)',
            tmax=10000,
            save='all',
            save_planets=True,
            save_path=str(tmp_path),
            plot_path=str(tmp_path),
            plot='all',
            plots=['evolution', 'combined', 'recurrence', 'fair', 'portrait', 'cycles'],
            plot_layout='flat',  # the statuses of this short run are not known in advance (layouts: test_output_layout)
            plot_combined_layout='flat',
        )
        sim.create_solar_system()
        sim.add_body(tools.get_3body_elements_sample(), resonances.ThreeBody('4J-2S-1'), name='three')
        sim.add_body(tools.get_2body_elements_sample(), resonances.TwoBody('1J-1'), name='two')
        with pytest.warns(UserWarning):  # FAIR asked for the three-body argument
            sim.run()

        plot_path = Path(sim.config.plot_path)
        figures = plot_path / 'figures'  # the individual figures' folder (PLOT_FIGURES_FOLDER)
        for kind in ('recurrence', 'portrait', 'cycles'):
            assert (figures / f'three-4J-2S-1+0+0-1-{kind}.png').exists(), kind
            assert (figures / f'two-1J-1+0+0-{kind}.png').exists(), kind
        # Combined figures have their own folder.
        assert (plot_path / 'combined' / 'three-4J-2S-1+0+0-1-combined.png').exists()
        assert (plot_path / 'combined' / 'two-1J-1+0+0-combined.png').exists()
        assert (figures / 'two-1J-1+0+0-fair.png').exists()
        assert not list(plot_path.rglob('three-4J-2S-1+0+0-1-fair.png'))
        assert (figures / 'three-4J-2S-1+0+0-1.png').exists()  # evolution is unchanged
