"""Panel (e) of the combined MMR figure draws the force law of rule R's witness (W28 g11-7, answer A).

`MMRAnalysis.response` is the response the status tree read (`response.deciding_response`): with a witness episode its
windows, law and cutoff, else the pooled ones. Panel (e) draws that response on the series it was measured on.

Truths computed here, independently of the drawing code:
- the pooled windows (phi0 = 2.0 rad, K = 5e-6 au/yr) and the witness's (phi0 = 0.4 rad, K = 1e-6 au/yr) are different
  sines; each drawn curve is K sin(psi - phi0) + C of the witness's window, by numpy on sigma low-passed at the
  witness's cutoff, over the window's central half (the first and last to the edges), never a pooled window;
- the witness's cutoff (3 x the figure's response cutoff) is the one the drawn a_dot and sigma were low-passed at: the
  black a_dot is `smoothed_series` at that cutoff up to the panel's power-of-ten scale, not `prep.adot_response`;
- with the same cutoff as the figure's response series the black a_dot is `prep.adot_response` itself.
"""

import dataclasses

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402

from resonances.plotting import mmr_combined as mc  # noqa: E402
from resonances.plotting import style  # noqa: E402
from resonances.resonance.analysis.mmr import response as response_module  # noqa: E402
from resonances.resonance.analysis.mmr.analyse import analyse_mmr  # noqa: E402
from resonances.resonance.analysis.mmr.models import (  # noqa: E402
    CheckValue,
    EpisodeResponse,
    ForceWindow,
    MMRSeries,
    Regime,
    ResonanceSpec,
    ResponseSource,
)
from resonances.resonance.analysis.settings import settings_from_config  # noqa: E402
from tests.resonances.analysis.builders import make  # noqa: E402

LAYOUT = mc.LAYOUT_V1
S = settings_from_config()
POOLED = [ForceWindow(0.0, 1000.0, 0.31, 5e-6, 2.0, 0.0), ForceWindow(500.0, 2000.0, 0.33, 5e-6, 2.0, 0.0)]
WITNESS = [ForceWindow(0.0, 800.0, 0.93, 1e-6, 0.4, 2e-8), ForceWindow(400.0, 1200.0, 0.95, 1e-6, 0.4, 2e-8)]


def teardown_function():
    plt.close('all')


def _pendulum():
    """sigma = 1 rad x cos(2 pi t / 500) about 0 near 2:1 with Jupiter, 2 kyr at 1 yr, a answering it."""
    n = 2001
    t = np.arange(n, dtype=float)
    n_jupiter = np.radians(30.3496)
    a_star = (np.pi / n_jupiter) ** (2 / 3)
    c = 3 * np.pi * a_star**-2.5
    omega = 2 * np.pi / 500.0
    sigma = np.cos(omega * t)
    a = a_star - omega * np.sin(omega * t) / c
    zero = np.zeros(n)
    return MMRSeries(
        t, a, np.full(n, 0.1), zero, zero, zero, zero, sigma % (2 * np.pi), ResonanceSpec.from_key('2J-1'), Regime(1.0, t[-1], n)
    )


@pytest.fixture(scope='module')
def record():
    series = _pendulum()
    return analyse_mmr(series, S, scan_subresonances=False, scan_owner=False), series


def _witnessed(analysis, cutoff_yr: float):
    """The record's response with the pooled windows POOLED and one witness episode (L4 full pass, L3 and B1 pass)
    whose force law is WITNESS measured at `cutoff_yr`, then rule R's response (what the analysis stores)."""
    witness = make(
        EpisodeResponse,
        episode_id=7,
        before_jump=False,
        n_tp=6,
        t_start_yr=0.0,
        t_end_yr=1200.0,
        period_yr=500.0,
        windows=list(WITNESS),
        force_law='single',
        a_response=CheckValue.PASS,
        identity=CheckValue.PASS,
        a_amplitude=CheckValue.PASS,
        force_cutoff_yr=cutoff_yr,
    )
    pooled = dataclasses.replace(
        analysis.response,
        windows=list(POOLED),
        force_law='single',
        multiplet=None,
        coorbital=None,
        force_harmonic=None,
        force_line_binding_yr=float('nan'),
        episode_responses=[witness],
    )
    deciding = response_module.deciding_response(pooled, False)
    assert deciding.force_source == ResponseSource.WITNESS and deciding.witness_episode_id == 7
    return dataclasses.replace(analysis, response=deciding)


def _draw(analysis, series):
    frame = mc._Frame(analysis, series, LAYOUT, settings=S)
    fig = plt.figure(figsize=LAYOUT.size, dpi=LAYOUT.dpi)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.2))
    mc.draw_adot_panel(ax, frame, LAYOUT)
    return ax, frame


def _model_lines(ax):
    return [line for line in ax.lines if to_hex(line.get_color()) == style.MODEL]


def _ink(ax):
    return np.asarray(
        next(line for line in ax.lines if to_hex(line.get_color()) == to_hex(style.INK) and line.get_linestyle() == '-').get_ydata(),
        float,
    )


def test_panel_e_draws_the_witness_windows_on_the_series_at_its_cutoff(record):
    analysis, series = record
    cutoff = 3 * analysis.prep.response_cutoff_yr
    analysis = _witnessed(analysis, cutoff)
    ax, frame = _draw(analysis, series)
    t = np.asarray(series.t_yr, float)
    sigma, _, _, adot, _ = response_module.smoothed_series(analysis.prep, cutoff, S)
    sigma, adot = np.asarray(sigma, float), np.asarray(adot, float)

    black = _ink(ax)
    shown = np.isfinite(black)
    ratio = black[shown] / adot[shown]
    scale = float(np.median(ratio))
    assert ratio == pytest.approx(np.full(shown.sum(), scale), rel=1e-9)  # a_dot at the witness's cutoff
    base = np.asarray(analysis.prep.adot_response, float)
    assert not np.allclose(base[shown] * scale, black[shown])  # not the figure's response series

    lines = _model_lines(ax)
    assert len(lines) == len(WITNESS)  # one curve per witness window, no pooled one
    halves = [(0.0, 600.0), (600.0, 1200.0)]  # windows overlap by half: the first from 0, the last to its end
    for line, window, (start, end) in zip(lines, WITNESS, halves):
        mask = (t >= start) & (t < end) & frame.inside
        assert np.asarray(line.get_xdata(), float) == pytest.approx(frame.x[mask])
        expected = window.amplitude_au_yr * np.sin(sigma[mask] - window.phi0_rad) + window.offset_au_yr
        assert np.asarray(line.get_ydata(), float) == pytest.approx(expected * scale, rel=1e-9, abs=1e-12)
        pooled = 5e-6 * np.sin(sigma[mask] - 2.0)
        assert np.max(np.abs(np.asarray(line.get_ydata(), float) / scale - pooled)) > 1e-6  # not the pooled law [au/yr]
    texts = [text.get_text() for text in ax.texts]
    assert not any(text.startswith('cutoff') for text in texts)  # no line rule: nothing raised, nothing printed
    assert {'0.93', '0.95'} <= set(texts) and not {'0.31', '0.33'} & set(texts)  # the witness windows' R²


def test_panel_e_keeps_the_figure_series_when_the_witness_read_them(record):
    analysis, series = record
    analysis = _witnessed(analysis, analysis.prep.response_cutoff_yr)
    ax, _ = _draw(analysis, series)
    black = _ink(ax)
    base = np.asarray(analysis.prep.adot_response, float)
    shown = np.isfinite(black)
    scale = 10.0 ** -mc._power_of_ten(np.where(shown, base, np.nan))  # the panel's power-of-ten unit
    assert black[shown] == pytest.approx(base[shown] * scale, rel=1e-12, abs=1e-15)
