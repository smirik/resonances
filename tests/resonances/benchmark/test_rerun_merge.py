"""bench_io.merge_run_record, which rerun.py uses for data/runs/<run_id>/run.json: a rerun adds its body to the
run's lists instead of replacing them (R6, 2026-10-19: four run.json files had lost their other bodies)."""

import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[3] / 'benchmark' / 'tools'
sys.path.insert(0, str(TOOLS))

import bench_io as bio  # noqa: E402


def record(bodies, sources, commit):
    return {'run_id': 'R-ias15-1kyr', 'package_commit': commit, 'bodies': bodies, 'elements_sources': sources}


def test_rerun_keeps_the_bodies_already_in_the_run(tmp_path):
    path = tmp_path / 'R-ias15-1kyr' / 'run.json'
    bio.write_json(path, record(['A', 'B'], ['astdys', 'first row of x'], 'old'))

    bio.merge_run_record(path, record(['B', 'C'], ['mpc'], 'new'))

    got = json.loads(path.read_text())
    assert got['bodies'] == ['A', 'B', 'C']
    assert got['elements_sources'] == ['astdys', 'first row of x', 'mpc']
    assert got['package_commit'] == 'new'  # every other field is the rerun's


def test_first_record_is_written_as_given(tmp_path):
    path = tmp_path / 'R-ias15-1kyr' / 'run.json'

    bio.merge_run_record(path, record(['C'], ['mpc'], 'new'))

    assert json.loads(path.read_text()) == record(['C'], ['mpc'], 'new')
