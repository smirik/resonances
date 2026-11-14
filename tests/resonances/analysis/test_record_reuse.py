"""Reuse inside one record analysis (perf cleanup 2026-11-14): what a second pass or a later call takes from an earlier
one must equal what it would compute itself.

- `filtering.memoised` keeps `entries` results per name (`band_pass` BAND_PASS_MEMO_ENTRIES);
- `filtering.value_key` keys by values (nan matches nan, dtype and shape count);
- `response.analyse_episodes(measured=...)` reuses a run's `measure_pieces` only when its inputs are unchanged;
- `zone.build_zone` integrates one model once inside a record memo.
"""

from dataclasses import replace

import numpy as np

from resonances.resonance.analysis import filtering
from resonances.resonance.analysis.mmr import response as r
from resonances.resonance.analysis.mmr import zone
from resonances.resonance.analysis.settings import settings_from_config

from tests.resonances.analysis.impl_response_pendulum_synthetic import episode_from, pendulum_record

S = settings_from_config()


def _same(left, right) -> bool:
    """Two episode responses with equal columns (nan equal to nan)."""
    a, b = left.to_columns(), right.to_columns()
    return a.keys() == b.keys() and all(
        (isinstance(a[k], float) and isinstance(b[k], float) and np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in a
    )


def test_band_pass_keeps_its_own_number_of_results():
    t = np.arange(0.0, 20000.0, 5.0)
    a = 2.5 + 1e-3 * np.sin(2 * np.pi * t / 300.0) + 5e-4 * np.sin(2 * np.pi * t / 2000.0)
    periods = [200.0 + 50.0 * k for k in range(filtering.BAND_PASS_MEMO_ENTRIES)]
    with filtering.record_memo():
        first = [filtering.band_pass(t, a, p, 2.0, S.filter_order) for p in periods]
        again = [filtering.band_pass(t, a, p, 2.0, S.filter_order) for p in periods]
        assert all(x is y for x, y in zip(first, again))  # all kept
        filtering.band_pass(t, a, 9000.0, 2.0, S.filter_order)  # one more: the least recently used goes
        assert filtering.band_pass(t, a, periods[0], 2.0, S.filter_order) is not first[0]
    # the band is the difference of the two low-passes (computed here independently)
    low = filtering.lowpass(t, a, 200.0 / 2.0, S.filter_order) - filtering.lowpass(t, a, 400.0, S.filter_order)
    np.testing.assert_array_equal(first[0], low)


def test_the_memo_entries_are_per_call():
    calls = []

    def compute(value):
        calls.append(value)
        return np.full(2, value)

    key = object()
    with filtering.record_memo():
        for value in (1.0, 2.0, 3.0, 1.0):  # four entries hold all three
            filtering.memoised('test4', (key,), (value,), lambda v=value: compute(v), lambda res: (res,), entries=4)
    assert calls == [1.0, 2.0, 3.0]


def test_value_key_reads_values_not_objects():
    x = np.array([1.0, np.nan, 3.0])
    assert filtering.value_key(x, float('nan')) == filtering.value_key(x.copy(), float('nan'))
    assert filtering.value_key(x) != filtering.value_key(np.array([1.0, np.nan, 3.5]))
    assert filtering.value_key(np.zeros(4)) != filtering.value_key(np.zeros((2, 2)))  # shape
    assert filtering.value_key(np.zeros(4)) != filtering.value_key(np.zeros(4, dtype=np.float32))  # dtype
    assert filtering.value_key(0.5) == filtering.value_key(np.float64(0.5))


def test_a_second_pass_reuses_only_unchanged_runs():
    rec = pendulum_record(span=6000.0)
    tp = rec.geometry.turning_points

    def subset(keep):
        return type(tp)(tp.indices[keep], tp.times_yr[keep], tp.values_rad[keep], tp.kinds[keep], tp.prominence_rad)

    runs = [
        episode_from(subset(tp.times_yr < 2900.0), episode_id=0),
        episode_from(subset(tp.times_yr > 3100.0), episode_id=1),
    ]
    rec.geometry = replace(rec.geometry, runs=runs)
    args = (rec.prep, rec.geometry, rec.context, None, S)
    kwargs = dict(spec=rec.spec, sampling=rec.sampling, series=rec.series, judged_ids=[0, 1])
    fresh = r.analyse_episodes(*args, **kwargs)
    measured = {}
    first = r.analyse_episodes(*args, **kwargs, measured=measured)
    assert len(measured) == 2
    assert all(_same(a.response, b.response) for a, b in zip(first, fresh))  # the same numbers as without the dict
    # the second pass: run 0 unchanged (reused), run 1 no longer judged (its inputs changed: measured again)
    second = r.analyse_episodes(*args, **{**kwargs, 'judged_ids': [0]}, measured=measured)
    assert second[0].response is first[0].response
    assert second[1].response is not first[1].response and second[1].response.a_response is None
    alone = r.analyse_episodes(*args, **{**kwargs, 'judged_ids': [0]})
    assert all(_same(a.response, b.response) for a, b in zip(second, alone))


def test_build_zone_integrates_one_model_once_in_a_record(monkeypatch):
    t = np.linspace(0.0, 4000.0, 2001)
    omega_bar = 2 * np.pi / 300.0
    omega2 = (omega_bar * (1 + 0.2 * np.cos(2 * np.pi * t / 1000.0))) ** 2
    well = np.zeros_like(t)
    calls = []
    original = zone.numeric_zone

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(zone, 'numeric_zone', counted)
    with filtering.record_memo():
        once = zone.build_zone(t, omega2, well, np.nan, S, 1000.0, omega_bar)
        twice = zone.build_zone(t.copy(), omega2.copy(), well.copy(), np.nan, S, 1000.0, omega_bar)  # equal values
        other = zone.build_zone(t, omega2 * 1.01, well, np.nan, S, 1000.0, omega_bar)
    assert len(calls) == 2 and once == twice
    outside = zone.build_zone(t, omega2, well, np.nan, S, 1000.0, omega_bar)  # no memo: computed, the same zone
    assert len(calls) == 3 and outside == once and other != once
