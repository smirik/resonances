"""The contract of the right column (`verdict_column.problems`) on the benchmark, through the package export (the very
column the figure draws: `verdict_column_mmr.figure_column`, `verdict_column_long.figure_column`).

- Secular and Lidov-Kozai: every saved example (89 on 2026-11-04, ~16 s of analysis in all).
- MMR and co-orbital: one example per kind of verdict of the last `make test-benchmark` run (benchmark/data/
  test-benchmark/summary.csv): the cheapest (run.json timings) of each combination of family, status, level, reason,
  origin, origin detail, angle subtype and the policy's decider (113 of 726 on 2026-11-04, ~3 min), with their
  derived subresonance rows. Re-analyzing all 726 takes ~40 min of CPU; the strata cover every reason, status, level,
  decider and subtype that occurs.

Each example is analyzed as `benchmark/tools/reclassify*.py` does; the column's header must carry the analysis' own
status and level (the verdict object, not the column's code), and the text export must hold every block. Failures are
collected per example, so one run lists all of them. Marked `benchmark` (local-only data).
"""

import gc
import json
import warnings

import matplotlib

matplotlib.use('Agg')

import pytest  # noqa: E402

from resonances.plotting import verdict_column as vc  # noqa: E402

pytestmark = pytest.mark.benchmark

STRATA = (
    'family',
    'status',
    'confidence',
    'reason',
    'libration_origin',
    'origin_detail',
    'angle_subtype',
    'pol_decided_by',
)
RUN = 'test-benchmark'


def _contract(column: vc.Column, status: str, level: str) -> list:
    out = list(vc.problems(column))
    if (column.header.status, column.header.level) != (status, level):
        out.append(f'header {column.header.status}/{column.header.level} != verdict {status}/{level}')
    text = vc.column_text(column)
    if column.header.word not in text:
        out.append('the text export lacks the status word')
    if not column.checks:
        out.append('no checks')  # problems() asks for exactly one ◆ only when there are steps
    return out


def mmr_sample(run_dir) -> list:
    """One example id per stratum (`STRATA`) of the run's top-level rows, the cheapest by the run's timings."""
    import pandas as pd

    summary = pd.read_csv(run_dir / 'summary.csv', low_memory=False, dtype=str).fillna('')
    summary = summary[summary['parent'] == ''].copy()
    summary['family'] = summary['example_id'].str.split('__').str[0]
    timings = {t['example_id']: t.get('seconds', 0.0) for t in json.loads((run_dir / 'run.json').read_text()).get('timings', [])}
    summary['seconds'] = summary['example_id'].map(timings).fillna(1e9)
    columns = [column for column in STRATA if column in summary.columns]
    picked = summary.sort_values(['seconds', 'example_id']).drop_duplicates(columns)
    return sorted(picked['example_id'])


def test_every_kind_of_mmr_verdict_keeps_the_contract():
    import bench_io as bio
    from resonances.plotting.verdict_column_mmr import figure_column
    from resonances.resonance.analysis.mmr import io
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr
    from resonances.resonance.analysis.settings import settings_from_config

    run_dir = bio.DATA / RUN
    if not (run_dir / 'summary.csv').exists():
        pytest.skip(f'no {RUN} run in {bio.DATA} (make test-benchmark writes it)')
    sample = mmr_sample(run_dir)
    assert len(sample) >= 60, sample
    settings = settings_from_config()
    failures, built = {}, 0
    warnings.simplefilter('ignore')
    for example in sample:
        try:
            analysis = analyse_mmr(io.load_series(example), settings)
        except Exception as exc:  # noqa: BLE001 - the column is under test, not the analysis
            failures[example] = [f'analysis: {exc!r}']
            continue
        for index, one in enumerate([analysis] + list(analysis.derived)):
            name = example if index == 0 else f'{example} / {one.resonance.key}'
            v = one.verdict
            try:
                found = _contract(figure_column(one, None, settings), str(v.status), str(v.confidence or ''))
            except Exception as exc:  # noqa: BLE001 - a builder that raises is a failure of this example
                found = [f'builder: {exc!r}']
            built += 1
            if found:
                failures[name] = found
        del analysis
        gc.collect()
    assert built >= len(sample)
    assert failures == {}


def test_every_secular_and_zlk_example_keeps_the_contract():
    import bench_io as bio
    import reclassify_secular_zlk as reclassify
    from resonances.plotting.verdict_column_long import figure_column
    from resonances.plotting.mmr_verdict_panel import config_settings

    examples = reclassify.select()
    if not examples:
        pytest.skip(f'no saved secular/ZLK examples in {bio.DATA}')
    settings = config_settings(None)
    failures = {}
    warnings.simplefilter('ignore')
    for example in examples:
        try:
            sim, body, resonance = reclassify.build_simulation(bio.load_example(example, with_planets=False))
            sim.identify_librations()
            analysis = sim.long_period_analysis(body, resonance)
            row = reclassify._extra_columns(reclassify.summary_row(example, body, resonance))
        except Exception as exc:  # noqa: BLE001
            failures[example] = [f'analysis: {exc!r}']
            continue
        elements = body.free_elements if resonance.type == 'lidov_kozai' else None
        v = analysis.verdict
        try:
            found = _contract(figure_column(analysis, row, elements, settings), str(v.status), str(v.confidence or ''))
        except Exception as exc:  # noqa: BLE001
            found = [f'builder: {exc!r}']
        if found:
            failures[example] = found
    assert len(examples) >= 80
    assert failures == {}
