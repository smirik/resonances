"""W5 mmr_core: the resample step of an aliased record is finer than the record's own step (it was 10 yr for records at
0.2, 0.25 and 5 yr: 99942, 88710, 1864), and an alias no finer step cures goes to a person (review)."""

import math
from types import SimpleNamespace

import pytest

from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.mmr.analyse import _resample_action, analyse_mmr
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Action, Reason, Status

S = settings_from_config()


def test_resample_action():
    sampling = SimpleNamespace(resample_step_yr=0.2)
    assert _resample_action(Action.RESAMPLE, sampling) == (Action.RESAMPLE, 0.2)
    assert _resample_action(Action.REVIEW, sampling) == (Action.REVIEW, None)
    assert _resample_action(Action.NONE, sampling) == (Action.NONE, None)
    # nothing finer than the record's step fits: a person looks
    assert _resample_action(Action.RESAMPLE, SimpleNamespace(resample_step_yr=None)) == (Action.REVIEW, None)


def _alias(example):
    if not (io.example_path(example) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(example), S)
    assert (analysis.verdict.status, analysis.verdict.reason) == (Status.INVALID, Reason.ALIAS)
    return analysis


def test_1864_resamples_at_the_step_its_rate_allows():
    # 51M-50 at 5 yr, nu ~ -428 deg/yr over an ambiguous a range: |nu| dt <= 90 deg needs dt <= 0.21 yr -> 0.2 yr
    analysis = _alias('mmr2__1864__51M-50+0-1')
    sampling = analysis.sampling
    assert math.degrees(sampling.step_rotation_max_rad) / sampling.step_yr == pytest.approx(428, rel=0.05)  # max |nu|
    assert analysis.verdict.action == Action.RESAMPLE
    assert analysis.verdict.resample_step_yr == 0.2 < sampling.step_yr


@pytest.mark.benchmark
def test_99942_resamples_finer_than_its_step():
    # 7E-6 at 0.2 yr (ambiguous); before W5 the verdict asked for 10 yr
    analysis = _alias('mmr2__99942__7E-6+0-1')
    assert analysis.verdict.action == Action.RESAMPLE
    assert analysis.verdict.resample_step_yr == 0.05 < analysis.sampling.step_yr
