"""W3 guards, G8 (every number of the decision reaches mmr_numbers.csv and the figure) and G9 (no pending defaults).

Codex astra (PHYSICS-MATH §8): G1c compared `geo_final_centre_travel_rad` in the ledger, but the column was missing
from the CSV row and the panel printed a dash. End to end: for every ledger entry of the decision path the column is
in the CSV row (numbers_row + summary_row) with the value the ledger compared, and the right-hand text of the
combined figure prints it - as its own line, as one of the columns a merged line stands for (`NumberLine.also`) or in
the decisive episode's row of the episode table.
"""

import math

import numpy as np
import pytest

from resonances.plotting import combined_numbers as cn
from resonances.plotting import mmr_verdict_panel as panel
from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings, setting_fields, settings_from_config

S = settings_from_config()


def missing_numbers(analysis) -> list:
    """(problem, ledger entry, column) for every ledger number that does not reach the CSV row or the panel."""
    summary, numbers = analysis.summary_row(), analysis.numbers_row()
    episodes = analysis.episode_rows()
    constants = analysis.context.constants_rows() if analysis.context is not None else []
    blocks = panel.text_blocks(summary, numbers, analysis.verdict.decision_path, episodes, constants)
    printed = cn.printed_columns(blocks, panel.TEXT_BLOCKS)
    _, in_episode_table = panel._episode_lines(episodes, numbers.get('ep_episode_id'))
    row = {**numbers, **summary}
    problems = []
    for entry in analysis.verdict.decision_path:
        if not entry.column:
            problems.append(('no_column', entry.name, ''))
        elif entry.column not in row:
            problems.append(('not_in_csv', entry.name, entry.column))
        elif entry.column not in printed and entry.column not in in_episode_table:
            problems.append(('not_printed', entry.name, entry.column))
        elif (
            entry.value is not None and isinstance(row[entry.column], (int, float, np.floating)) and not isinstance(row[entry.column], bool)
        ):
            if not math.isclose(float(row[entry.column]), entry.value, rel_tol=1e-9, abs_tol=1e-12):
                problems.append(('value_differs', entry.name, entry.column))
    return problems


def libration_then_circulation() -> MMRSeries:
    """2J-1: a 1-rad libration of 1000 yr until 12 kyr, then sigma runs 2 pi per 2 kyr (a answers both by the
    identity) - the last regime is a circulation after librating episodes (rule G1c, the case of Codex astra)."""
    n = 20001
    t = np.arange(n, dtype=float)
    n_jupiter = np.radians(30.3496)
    a_star = (np.pi / n_jupiter) ** (2 / 3)
    c = 3 * np.pi * a_star**-2.5
    sigma = np.where(t < 12000.0, np.sin(2 * np.pi * t / 1000.0), np.sin(24 * np.pi) + 2 * np.pi * (t - 12000.0) / 2000.0)
    zero = np.zeros(n)
    a = a_star + np.gradient(sigma, t) / c
    return MMRSeries(
        t, a, np.full(n, 0.1), zero, zero, zero, zero, sigma % (2 * np.pi), ResonanceSpec.from_key('2J-1'), Regime(1.0, t[-1], n)
    )


def test_g1c_numbers_reach_the_csv_and_the_figure():
    analysis = analyse_mmr(libration_then_circulation(), S, scan_owner=False, scan_subresonances=False)
    assert analysis.extra_columns['rec_rule'] != 'G2'
    entry = next(e for e in analysis.verdict.decision_path if e.column == 'geo_final_centre_travel_rad')
    assert entry.value > 2 * np.pi  # 8 turns after the last turning point
    assert analysis.numbers_row()['geo_final_centre_travel_rad'] == pytest.approx(entry.value)
    assert missing_numbers(analysis) == []


def test_numbers_of_a_refused_record_reach_the_csv_and_the_figure():
    series = libration_then_circulation()
    refused = analyse_mmr(MMRSeries(**{**series.__dict__, 'e': np.full(series.t_yr.size, 1.1)}), S)
    assert refused.verdict.status == 'invalid' and missing_numbers(refused) == []


@pytest.mark.benchmark
@pytest.mark.parametrize('example', ['mmr3__1221__1V-2E+1+0+0+0', 'mmr3__52356__5J-2S-2+0+0-1', 'mmr3__117306__1E+5S-4+0+0-2'])
def test_numbers_of_benchmark_decisions_reach_the_csv_and_the_figure(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    assert missing_numbers(analyse_mmr(io.load_series(example), S)) == []


def test_pending_defaults_are_settings_now():
    """zone.py and owner.py read thresholds through `pending_setting` while AnalysisSettings lacks them: every one of
    them is a setting (settings_hash, .env.dist, docs/config.md), so the pending default is never used (G9)."""
    from resonances.resonance.analysis.mmr import owner, zone

    names = {f.name for f in setting_fields(AnalysisSettings)}
    for module in (zone, owner):
        assert set(getattr(module, 'PENDING_SETTINGS', {})) <= names
    for name in ('mmr_planet_unwrap_max_step_deg', 'separatrix_layer_adiabaticity_min', 'mmr_coorbital_period_floor_factor'):
        assert name in names
