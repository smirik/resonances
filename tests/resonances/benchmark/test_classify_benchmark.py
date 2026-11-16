"""Classification benchmark (`make test-benchmark`): an exact gate on the agreement with the author's labels.

The v2 analysis classifies every saved MMR and co-orbital example of benchmark/ (`benchmark/tools/reclassify.py` into
benchmark/data/test-benchmark; nothing is integrated, the test is skipped without the saved series in benchmark/data or
$RESONANCES_BENCHMARK_DATA). Its disagreements with the author (labels made by Claude never count) are then compared,
row by row, with benchmark/KNOWN_DISAGREEMENTS.csv (`benchmark/tools/known_disagreements.py`):

- the status, angle_status and confidence of the author's labels (author_labels_*.csv, then author_label_decisions_*.csv),
  each where the author gave a value;
- the status against every unambiguous `author_verdict` of candidates.csv;
- the secular and Lidov-Kozai examples (`reclassify_secular_zlk.py`, the session fixture `secular_zlk_run`): the same
  author labels, and the status against the author's verdicts `reclassify_secular_zlk.AUTHOR_VERDICTS` (14 rows, all
  agree on 2026-10-27).

The test fails, with a table, on a disagreement that is not listed (new), on a listed one that now agrees (resolved:
update the list) and on a listed one whose package or author value changed. A change of the code that moves any author
row must therefore say so in the list (`known_disagreements.py test-benchmark --write` keeps the reasons; the category
and reason of a new row are written by hand). The agreement numbers are printed (v2-w21a, 2026-10-23: status 81/90,
angle_status 76/81, confidence 42/59, author_verdict 222/230).

The reclassification is the session fixture `reclassified_run` (conftest.py), shared with the episode gate
(test_episode_expectations.py). Workers: BENCHMARK_WORKERS, else auto = min(CPUs - 2, available RAM / 3 GB, 8)
(`bench_io.auto_workers`). The CSVs are read through a binary copy under benchmark/data/cache/columns
(`column_cache.py`), written on the first run.
2026-10-23, 7 workers: 249 s (270 s on the first run).
"""

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import known_disagreements as kd  # noqa: E402


@pytest.mark.benchmark
def test_disagreements_with_the_author_are_the_known_ones(reclassified_run, secular_zlk_run):
    problems = kd.check(reclassified_run, secular_zlk=secular_zlk_run)
    assert problems.empty, kd.format_problems(problems)
