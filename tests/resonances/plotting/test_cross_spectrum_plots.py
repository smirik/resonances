"""Tests for the eccentricity-vector and cross-spectrum plotters."""

import matplotlib

matplotlib.use('Agg')

from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from resonances.body import Body  # noqa: E402
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance  # noqa: E402
from resonances.plotting import CrossSpectrumPlotter, EccentricityVectorPlotter  # noqa: E402
from resonances.plotting.base import format_period, time_unit  # noqa: E402
from resonances.resonance import coherence_analysis as ca  # noqa: E402
from resonances.simulation.config import SimulationConfig  # noqa: E402


def _fake_level(period_min, period_max, coherent: bool):
    """A minimal stand-in for a CrossSpectrum, on a 1e-6..1e-4 /yr grid (10 kyr..1 Myr)."""
    frequency = np.linspace(1e-6, 1e-4, 100)
    return SimpleNamespace(
        period_min=period_min,
        period_max=period_max,
        frequency=frequency,
        coherence=np.full(frequency.size, 1.0 if coherent else 0.0),
        in_band=np.ones(frequency.size, dtype=bool),
        gamma2_crit=0.5,
    )


def _covers(runs, period):
    return any(low <= period <= high for low, high in runs)


@pytest.fixture
def body_and_resonance():
    n = 20_000
    t_years = np.arange(n, dtype=float)
    common = np.sin(2 * np.pi * t_years / 500.0)

    body = Body()
    body.name = 'plot_body'
    body.times = t_years * 2 * np.pi
    body.axis = np.full(n, 2.5)
    body.ecc = 0.2 + 0.05 * common
    body.inc = 0.5 - 0.05 * common
    body.omega = np.mod(np.linspace(0.0, 40 * np.pi, n), 2 * np.pi)
    resonance = LidovKozaiResonance()
    body.lidov_kozai_resonances = [resonance]
    body.angles_unwrapped[resonance.to_s()] = np.unwrap(body.omega)
    body.statuses[resonance.to_s()] = 2
    return body, resonance


class TestEccentricityVectorPlotter:
    def test_plots_the_plane(self, body_and_resonance, tmp_path):
        body, resonance = body_and_resonance

        plotter = EccentricityVectorPlotter.from_body(body, resonance).plot()

        assert plotter._figure is not None
        assert len(plotter._axes) == 1
        path = tmp_path / 'nested' / 'ecc-vector.png'
        plotter.save(str(path))
        assert path.exists()
        plotter.close()
        assert plotter._figure is None

    def test_phase_plane_holds_the_k_h_values(self, body_and_resonance):
        body, resonance = body_and_resonance
        expected_k = body.ecc * np.cos(body.omega)

        plotter = EccentricityVectorPlotter.from_body(body, resonance).plot()
        offsets = plotter._axes[0].collections[0].get_offsets()

        np.testing.assert_allclose(offsets[:, 0], expected_k)
        np.testing.assert_allclose(offsets[:, 1], body.ecc * np.sin(body.omega))
        plotter.close()

    def test_from_data(self):
        times = np.linspace(0.0, 1000.0, 500)
        ecc = np.full(500, 0.1)
        omega = np.linspace(0.0, 4 * np.pi, 500)

        plotter = EccentricityVectorPlotter.from_data(times, ecc, omega, body_name='x', resonance_key='LK').plot()

        assert plotter._figure is not None
        plotter.close()

    def test_missing_data_does_not_raise(self):
        body = Body()
        body.name = 'empty'
        plotter = EccentricityVectorPlotter.from_body(body)

        assert plotter.plot()._figure is None


class TestDisplayUnits:
    # The same figure has to read correctly from a thousand-year synthetic to a 50 Myr
    # integration, so the unit is picked from the magnitude rather than hard-coded.
    @pytest.mark.parametrize(
        'span, expected',
        [(500.0, (1.0, 'yr')), (9_999.0, (1.0, 'yr')), (10_000.0, (1e3, 'kyr')), (999_999.0, (1e3, 'kyr')), (5e6, (1e6, 'Myr'))],
    )
    def test_unit_follows_magnitude(self, span, expected):
        assert time_unit(span) == expected

    @pytest.mark.parametrize('period, expected', [(850.0, '850 yr'), (52_083.0, '52.1 kyr'), (1.4e6, '1.4 Myr'), (304_465.0, '304 kyr')])
    def test_period_is_written_in_its_own_unit(self, period, expected):
        assert format_period(period) == expected

    def test_non_finite_span_does_not_raise(self):
        assert time_unit(float('nan')) == (1.0, 'yr')


class TestCrossSpectrumPlotter:
    def test_plots_three_panels(self, body_and_resonance, tmp_path):
        body, resonance = body_and_resonance
        config = SimulationConfig(save=None, plot=None, save_summary=False, coherence_period_min=100.0)
        ca.analyse_resonance(body, resonance, config)  # noqa: F841 - fills body.coherence

        plotter = CrossSpectrumPlotter.from_analysis(
            ca.analyse_resonance(body, resonance, config)['e-i'], body_name=body.name, resonance_key='LK'
        ).plot()

        assert plotter._figure is not None
        assert len(plotter._axes) == 3
        path = tmp_path / 'coherence.png'
        plotter.save(str(path))
        assert path.exists()
        plotter.close()

    def test_from_body_feeds_the_time_domain_panel(self, body_and_resonance, tmp_path):
        body, resonance = body_and_resonance
        config = SimulationConfig(save=None, plot=None, save_summary=False, coherence_period_min=100.0)
        body.coherence[resonance.to_s()] = ca.analyse_resonance(body, resonance, config)

        plotter = CrossSpectrumPlotter.from_body(body, resonance, 'e-i')

        assert plotter._x is body.ecc
        assert plotter._y is body.inc
        plotter.plot()
        # Two lines on the time panel plus its twin: the pair band-passed at the line.
        assert len(plotter._axes[2].lines) == 1
        plotter.close()

    def test_levels_that_cannot_see_a_period_abstain(self):
        # A level whose validity band stops short of a period must not block agreement
        # there: it cannot see the line, so it does not get a vote.
        wide = _fake_level(1e4, 1e6, coherent=True)
        narrow = _fake_level(1e4, 2e4, coherent=True)
        blind = _fake_level(1e4, 1e6, coherent=False)

        # 40-100 kyr is inside `wide` only; `narrow` abstains, so the two that can see it
        # agree, while a level that can see it and disagrees kills the run.
        assert _covers(CrossSpectrumPlotter._agreement_runs([wide, wide, narrow], wide), 5e4)
        assert not _covers(CrossSpectrumPlotter._agreement_runs([wide, blind], wide), 5e4)

    def test_agreement_needs_more_than_one_level(self):
        # A single level agreeing with itself is not a consensus and must not be shaded.
        wide = _fake_level(1e4, 1e6, coherent=True)

        assert CrossSpectrumPlotter._agreement_runs([wide], wide) == []

    def test_only_the_strongest_lines_are_highlighted(self):
        # Agreement is broad whenever the Kozai integral holds, so the highlight is
        # narrowed to the intervals that carry one of the top lines by cross-amplitude.
        agreed = [(1e4, 2e4), (5e4, 6e4), (2e5, 3e5)]
        decisive = SimpleNamespace(lines=[SimpleNamespace(period=p) for p in (5.5e4, 1.5e4, 9e9, 2.5e5)])

        highlighted = CrossSpectrumPlotter._highlighted(agreed, decisive)

        # Ranked 4th, the 250 kyr interval falls outside the top three and stays plain.
        assert highlighted == [(1e4, 2e4), (5e4, 6e4)]

    def test_unavailable_analysis_renders_a_placeholder(self, tmp_path):
        body = Body()
        body.name = 'short'
        n = 800
        body.times = np.arange(n, dtype=float) * 2 * np.pi
        body.ecc = np.full(n, 0.2)
        body.inc = np.full(n, 0.5)
        body.axis = np.full(n, 2.5)
        body.omega = np.zeros(n)
        resonance = LidovKozaiResonance()
        body.lidov_kozai_resonances = [resonance]
        body.angles_unwrapped[resonance.to_s()] = np.zeros(n)
        config = SimulationConfig(save=None, plot=None, save_summary=False)

        analysis = ca.analyse_resonance(body, resonance, config)['e-i']
        plotter = CrossSpectrumPlotter.from_analysis(analysis, body_name='short', resonance_key='LK').plot()

        assert analysis.check == ca.CoherenceCheck.SHORT_BASELINE
        assert plotter._figure is not None
        path = tmp_path / 'placeholder.png'
        plotter.save(str(path))
        assert path.exists()
        plotter.close()
