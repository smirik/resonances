"""The rules of the secular/ZLK analysis, each on a case whose answer is computed independently.

- the forced waves of a secular angle (4305 g-3g6+2g5: g = 76.22"/yr, waves at g - g6 and g - g5) and the cutoff
  rule P_c = max(2 x 937 yr, 20 dt, 2 x P_w, P_min / 2);
- the free-omega gate's outcome on the verdict: -2 (kinematic) and -9 (no verdict) stay apart, only a libration takes
  every outcome, a transient or a drift only a kinematic one with rho_max < 1;
- the centre scale: a slow libration under faster waves is a libration (518 z1), and waves faster than the shortest
  libration make no episode (2008 UZ220);
- the e response is a column: it changes no status;
- every setting of the package is documented; the one-page figure prints the verdict from the CSV row.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.secular import AngleTerm, GateResult, LongPeriodSeries, analyse_secular
from resonances.resonance.analysis.secular import decide, preprocess
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import AngleStatus, CheckValue, Reason, ResonanceType, Status

SETTINGS = settings_from_config()
T = np.linspace(0.0, 1e7, 20001)  # 10 Myr at 500 yr


def secular_series(angle, e=None, terms=()):
    n = len(T)
    return LongPeriodSeries(
        ResonanceType.SECULAR,
        'test',
        'synthetic',
        T,
        np.asarray(angle, dtype=float),
        np.full(n, 2.5),
        np.full(n, 0.1) if e is None else np.asarray(e, dtype=float),
        np.full(n, 0.1),
        terms=terms,
    )


# --- forced waves and the cutoff ----------------------------------------------------------------------------


def test_forced_waves_of_a_multi_mode_angle():
    # 4305 g-3g6+2g5 with g = 76.22"/yr: 1296000 / (76.22 - 28.2455) = 27014 yr, / (76.22 - 4.2575) = 18010 yr
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -3), AngleTerm('g', 5, 2))
    waves = preprocess.forced_waves(terms, 76.22, -72.34)
    assert waves == pytest.approx({'g6': 27014.0, 'g5': 18010.0}, rel=1e-3)


def test_a_single_mode_angle_has_no_forced_wave():
    # nu6 = g - g6: the forced g6 term is the resonance (Tina: 2 pi / |g - g6| is its 594-kyr libration)
    assert preprocess.forced_waves((AngleTerm('g', None, 1), AngleTerm('g', 6, -1)), 30.43, -55.0) == {}


def test_the_cutoff_of_omega_is_twice_the_great_inequality():
    series = LongPeriodSeries(
        ResonanceType.LIDOV_KOZAI, 'LK', 'x', np.arange(0.0, 1e5, 10.0), np.zeros(10000), np.ones(10000), np.ones(10000), np.ones(10000)
    )
    cutoff = preprocess.choose_cutoff(series, SETTINGS)
    assert cutoff.binding == 'great_inequality' and cutoff.cutoff_yr == pytest.approx(2 * 937.0)


def test_the_cutoff_of_a_secular_angle_keeps_the_shortest_libration_out_of_the_filter():
    # no forced wave known: P_c = max(2 x 937, 20 x 500 yr, 200 kyr / 2) = 100 kyr
    cutoff = preprocess.choose_cutoff(secular_series(np.zeros_like(T)), SETTINGS)
    assert cutoff.binding == 'shortest_libration' and cutoff.cutoff_yr == pytest.approx(1e5)


# --- the gate and the verdict -------------------------------------------------------------------------------


def gate(outcome, rho_max=0.5, cycles=10.0):
    return GateResult(outcome, n_cycles=cycles, min_cycles=3.0, rho_max=rho_max)


def zlk_status(angle_status, result):
    return decide.decide_status(ResonanceType.LIDOV_KOZAI, decide.AngleDecision(angle_status), result, Ledger.from_settings(SETTINGS))


@pytest.mark.parametrize(
    'outcome, status, reason, check, code',
    [
        ('confirmed', Status.RESONANT, Reason.DYNAMIC_LIBRATION, CheckValue.PASS, 2),
        ('kinematic', Status.NON_RESONANT, Reason.KINEMATIC_LIBRATION, CheckValue.FAIL, -2),
        ('planetary_residual', Status.UNCERTAIN, Reason.GATE_UNAVAILABLE, CheckValue.NOT_MEASURED, -9),
        ('short_window', Status.UNCERTAIN, Reason.GATE_UNAVAILABLE, CheckValue.NOT_MEASURED, -9),
        ('unavailable', Status.UNCERTAIN, Reason.GATE_UNAVAILABLE, CheckValue.NOT_MEASURED, -9),
    ],
)
def test_a_libration_takes_every_outcome_of_the_gate(outcome, status, reason, check, code):
    from resonances.resonance.analysis.verdict import status_code

    decision = zlk_status(AngleStatus.LIBRATION, gate(outcome))
    assert (decision.status, decision.reason, decision.checks['check_zlk_free_omega']) == (status, reason, check)
    assert status_code(ResonanceType.LIDOV_KOZAI, decision.status, AngleStatus.LIBRATION, decision.reason) == code


@pytest.mark.parametrize('angle_status', [AngleStatus.EPISODIC, AngleStatus.DRIFT, AngleStatus.INSUFFICIENT_TIME])
def test_without_a_libration_only_a_pinned_kinematic_omega_changes_the_status(angle_status):
    geometry_only = decide._status_by_geometry(angle_status).status
    assert zlk_status(angle_status, gate('kinematic', rho_max=0.6)).status == Status.NON_RESONANT  # 555527
    assert zlk_status(angle_status, gate('kinematic', rho_max=3.4)).status == geometry_only  # 1866: varpi turns freely
    assert zlk_status(angle_status, gate('planetary_residual')).status == geometry_only  # 3752: a lobe switch
    assert zlk_status(angle_status, gate('confirmed')).status == geometry_only


def test_a_circulating_omega_is_not_gated():
    decision = zlk_status(AngleStatus.CIRCULATION, gate('not_applicable'))
    assert decision.status == Status.NON_RESONANT and decision.checks['check_zlk_free_omega'] == CheckValue.NOT_APPLICABLE


# --- scales, floors and the response column ------------------------------------------------------------------


def test_a_slow_libration_under_faster_waves_is_a_libration():
    # 518 z1-like: 3.3 Myr libration of 80 deg carrying 12-deg waves of 430 kyr that fragment the turning points
    angle = np.radians(80.0 * np.sin(2 * np.pi * T / 3.3e6) + 12.0 * np.sin(2 * np.pi * T / 4.3e5))
    analysis = analyse_secular(secular_series(angle), SETTINGS)
    assert (analysis.verdict.status, analysis.verdict.angle_status) == (Status.RESONANT, AngleStatus.LIBRATION)
    assert analysis.libration_period_yr == pytest.approx(3.3e6, rel=0.1)


def test_waves_faster_than_the_shortest_libration_make_no_episode():
    # 2008 UZ220-like: one slow arc of 100 deg over the record with 3-deg waves of 150 kyr (< 200 kyr)
    angle = np.radians(100.0 * np.sin(np.pi * T / 1e7) + 3.0 * np.sin(2 * np.pi * T / 1.5e5))
    analysis = analyse_secular(secular_series(angle), SETTINGS)
    assert analysis.verdict.status == Status.UNCERTAIN
    assert analysis.numbers_row()['sec_n_librating_episodes'] == 0


def test_the_response_of_e_is_a_column_only():
    # Lagrange: e_dot = K sin(sigma) for a libration sigma = 60 deg sin(2 pi t / 2 Myr); a constant e does not answer
    sigma = np.radians(60.0 * np.sin(2 * np.pi * T / 2e6))
    answering = 0.1 + 1e-9 * np.cumsum(np.sin(sigma)) * (T[1] - T[0])
    terms = (AngleTerm('g', None, 1), AngleTerm('g', 6, -1))
    responding = analyse_secular(secular_series(sigma, e=answering, terms=terms), SETTINGS)
    silent = analyse_secular(secular_series(sigma, e=np.full_like(T, 0.1) + 1e-6 * np.sin(2 * np.pi * T / 7.7e5), terms=terms), SETTINGS)
    assert responding.response.r2_e > 0.95 and responding.verdict.checks['check_secular_e_response'] == CheckValue.PASS
    assert silent.verdict.checks['check_secular_e_response'] == CheckValue.FAIL
    assert responding.verdict.status == silent.verdict.status == Status.RESONANT
    assert responding.verdict.confidence == silent.verdict.confidence


def test_the_centre_columns_say_which_scale_decided():
    analysis = analyse_secular(secular_series(np.radians(80.0 * np.sin(2 * np.pi * T / 4e6))), SETTINGS)
    numbers = analysis.numbers_row()
    assert numbers['sec_scale'] == 'fine' and numbers['geo_n_tp'] == 5


# --- settings, figure ---------------------------------------------------------------------------------------


def test_every_secular_and_zlk_setting_is_documented():
    from dataclasses import fields
    from pathlib import Path

    import resonances

    root = Path(resonances.__file__).parent
    env = (root / '.env.dist').read_text()
    docs = (root.parent / 'docs' / 'config.md').read_text()
    names = [f.metadata['env'] for f in fields(type(SETTINGS)) if f.name.startswith(('secular_', 'zlk_'))]
    assert len(names) >= 10
    for name in names:
        assert f'\n{name}=' in env and f'`{name}`' in docs, name


def test_the_figure_prints_the_verdict_of_the_row(tmp_path):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from resonances.plotting import combined_numbers as numbers
    from resonances.plotting.secular_plots import LONG_PERIOD_BLOCK_ORDER, draw_long_period_combined, long_period_blocks
    from resonances.resonance.analysis.zlk import analyse_zlk

    t = np.linspace(0.0, 1e5, 5001)
    series = LongPeriodSeries(
        ResonanceType.LIDOV_KOZAI,
        'LK',
        'synthetic',
        t,
        np.pi + 0.5 * np.sin(2 * np.pi * t / 1e4),
        np.full(5001, 2.5),
        np.full(5001, 0.2),
        np.full(5001, 0.6),
    )
    analysis = analyse_zlk(series, SETTINGS, gate_config=SimpleNamespace(free_gate_enabled=False, free_gate_min_cycles=3.0))
    row = analysis.summary_row()
    blocks = long_period_blocks(row, analysis.verdict.decision_path, analysis.episode_rows())
    printed = numbers.printed_columns(blocks, LONG_PERIOD_BLOCK_ORDER)
    assert printed['status'].value == 'resonant' and printed['geo_period_cycle_yr'].column == 'geo_period_cycle_yr'
    for entry in analysis.verdict.decision_path:  # every number the decision compared is on the figure
        assert not entry.column or entry.column in printed, entry.column
    fig = draw_long_period_combined(analysis)
    assert fig.text_stats['overflow'] == 0
    fig.savefig(tmp_path / 'figure.png', dpi=30)
    plt.close(fig)
