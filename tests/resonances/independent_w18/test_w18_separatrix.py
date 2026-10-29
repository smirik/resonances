"""Independent tests of W18 items 4 and 5 (docs/physics/geometry.md "The separatrix zone", crossings;
docs/physics/confidence.md caps; docs/output.md `flags`, `sep_`):

- the near_separatrix cap: a near_separatrix of the geometry (G3) whose zone is not measured is at most medium;
- the counts of exits and captures and the mark exit_and_capture on a pendulum with one slip and a separate exit;
- the flag/cap separatrix_chaos of old files read back as episodes_in_separatrix_zone.

The pendulum records are integrated here; the reference counts come from the integrated angle itself (its passages
over the top, sigma = pi mod 2 pi), not from the package's turning points.
"""

import dataclasses
import functools

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import solve_ivp

from resonances.resonance.analysis import confidence
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.context import table_mean_motion
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import AnalysisSettings
from resonances.resonance.analysis.verdict import Confidence, Verdict
from resonances.simulation.serializer import SimulationSerializer

SETTINGS = AnalysisSettings()
P0_YR = 400.0  # small-amplitude period of the 3J-1 pendulum
W0 = 2 * np.pi / P0_YR


def _series(t, sigma, sigma_dot, key='3J-1'):
    """3J-1 body whose lambda and a follow sigma exactly; Jupiter on its mean line."""
    spec = ResonanceSpec.from_key(key)
    ones = np.ones_like(t)
    n_j = table_mean_motion('Jupiter')
    lam_j = n_j * t + 0.3
    k_j, k, p, varpi = spec.planet_coefficients[0], spec.body_coefficient, spec.body_varpi_coefficient, 1.0
    lam = (sigma - k_j * lam_j - p * varpi) / k
    n = (sigma_dot - k_j * n_j) / k
    jupiter = PlanetSeries('Jupiter', t, 5.2026 * ones, 0.048 * ones, 0.0228 * ones, 1.753 * ones, 0.257 * ones, np.mod(lam_j, 2 * np.pi))
    return MMRSeries(
        t_yr=t,
        a=(2 * np.pi / n) ** (2.0 / 3.0),
        e=0.15 * ones,
        inc=0.05 * ones,
        Omega=0.4 * ones,
        omega=(varpi - 0.4) * ones,
        varpi=varpi * ones,
        angle_wrapped=np.mod(k_j * lam_j + k * lam + p * varpi, 2 * np.pi),
        resonance=spec,
        regime=Regime(float(t[1] - t[0]), float(t[-1] - t[0]), len(t)),
        planets={'Jupiter': jupiter},
        mean_longitude=np.mod(lam, 2 * np.pi),
        name='synthetic',
    )


def _analyse(series):
    return analyse_mmr(series, SETTINGS, scan_subresonances=False, scan_owner=False)


def _energy(sigma, sigma_dot):
    """h = (sigma_dot^2 / 2 + w0^2 (1 - cos sigma)) / (2 w0^2): 1 on the separatrix."""
    return (sigma_dot**2 / 2 + W0**2 * (1 - np.cos(sigma))) / (2 * W0**2)


# --------------------------------------------------------------------------------------------------------------------
# near_separatrix without a measured zone: <= medium


@functools.lru_cache(maxsize=None)
def _free_pendulum(h, span_yr=20000.0):
    """sigma_ddot = -w0^2 sin sigma from sigma = 0 at energy h (h > 1: slow revolutions stalling at the top)."""
    t = np.arange(0.0, span_yr + 0.5, 1.0)
    sol = solve_ivp(
        lambda _, y: [y[1], -(W0**2) * np.sin(y[0])],
        (t[0], t[-1]),
        [0.0, 2 * W0 * np.sqrt(h)],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        method='DOP853',
    )
    return _analyse(_series(t, sol.y[0], sol.y[1]))


@pytest.mark.parametrize('h', [1.01, 1.03])
def test_near_separatrix_without_zone_is_capped_at_medium(h):
    """Just above the separatrix every revolution stalls at the top (the rate there is 2 w0 sqrt(h - 1), 10-17 % of
    the rate at the bottom): pattern B. A constant pendulum has no modulation, so no zone is measured; the raw level is
    above medium and the cap near_separatrix_without_zone sets medium (564160 3N-4)."""
    verdict = _free_pendulum(h).verdict
    assert (str(verdict.status), str(verdict.angle_status), str(verdict.angle_subtype)) == (
        'near_separatrix',
        'near_separatrix',
        'pattern_b',
    )
    assert str(verdict.checks['check_mmr_separatrix_zone']) != 'pass'
    assert 'near_separatrix_without_zone' in verdict.caps
    assert str(verdict.confidence_raw) in ('high', 'certain')
    assert str(verdict.confidence) == 'medium'


def test_libration_of_the_same_pendulum_takes_no_near_separatrix_cap():
    """h = 0.5 (half-amplitude 90 deg) in the same pendulum: the zone is not measured either, but the cap belongs to
    G3 only."""
    verdict = _free_pendulum(0.5).verdict
    assert str(verdict.status) == 'resonant'
    assert 'near_separatrix_without_zone' not in verdict.caps


@pytest.mark.parametrize('level, expected', [('certain', 'medium'), ('high', 'medium'), ('medium', 'medium'), ('low', 'low')])
def test_the_cap_ceiling_is_medium(level, expected):
    assert str(confidence.apply_caps(Confidence(level), ['near_separatrix_without_zone'])) == expected


# --------------------------------------------------------------------------------------------------------------------
# exit_and_capture: a slip (exit + capture at one moment) and a separate exit


# target energy h(t): libration h = 0.6, a short pulse over the top at 6.1-6.3 kyr, libration h = 0.5, then from
# 13-13.6 kyr a rise to h = 1.12 kept to the end (slow circulation)
SCHEDULE = ((0.0, 0.6), (6000.0, 0.6), (6100.0, 1.6), (6300.0, 0.5), (13000.0, 0.5), (13600.0, 1.12), (20000.0, 1.12))


@functools.lru_cache(maxsize=None)
def _driven_pendulum(step_yr=1.0):
    """sigma_ddot = -w0^2 sin sigma + g sigma_dot, g chosen so that h relaxes to the scheduled h(t) in 150 yr
    (dh/dt = g sigma_dot^2 / (2 w0^2))."""
    times, levels = zip(*SCHEDULE)

    def rhs(x, y):
        sigma, rate = y
        target = np.interp(x, times, levels)
        gain = (target - _energy(sigma, rate)) / 150.0 * 2 * W0**2 / (rate**2 + 0.05 * W0**2)
        return [rate, -(W0**2) * np.sin(sigma) + gain * rate]

    t = np.arange(0.0, 20000.0 + step_yr / 2, step_yr)
    sol = solve_ivp(rhs, (0.0, 20000.0), [2 * np.arcsin(np.sqrt(0.6)), 0.0], t_eval=t, rtol=1e-9, atol=1e-11, method='DOP853', max_step=2.0)
    return t, sol.y[0], sol.y[1]


def _top_passages(t, sigma):
    """Times at which sigma passes a top (pi mod 2 pi)."""
    branch = np.floor((sigma + np.pi) / (2 * np.pi))
    return t[1:][np.diff(branch) != 0]


def _reference_transitions(t, sigma):
    """(exits, captures, distinct times, slip times, first passage of the final circulation) from the passages over
    the top: passages closer than 3 P0 belong to one excursion; an excursion of one passage between librations is a
    slip (an exit and a capture at one moment), one that lasts to the record's end is an exit."""
    passages = _top_passages(t, sigma)
    groups = np.split(passages, np.nonzero(np.diff(passages) > 3 * P0_YR)[0] + 1)
    exits = captures = distinct = 0
    slips, final = [], np.nan
    for group in groups:
        reaches_end = t[-1] - group[-1] < 3 * P0_YR
        if len(group) == 1 and not reaches_end:
            exits, captures, distinct = exits + 1, captures + 1, distinct + 1
            slips.append(group[0])
        elif reaches_end:
            exits, distinct, final = exits + 1, distinct + 1, group[0]
        else:
            exits, captures, distinct = exits + 1, captures + 1, distinct + 2
    return exits, captures, distinct, slips, final


def test_reference_record_has_one_slip_and_one_exit():
    t, sigma, _ = _driven_pendulum()
    exits, captures, distinct, slips, final = _reference_transitions(t, sigma)
    assert (exits, captures, distinct) == (2, 1, 2)
    assert 6100.0 < slips[0] < 6400.0
    assert 13000.0 < final < 14500.0


def test_slip_counts_as_exit_and_capture_at_one_time():
    t, sigma, rate = _driven_pendulum()
    analysis = _analyse(_series(t, sigma, rate))
    row = analysis.numbers_row()
    exits, captures, distinct, slips, final = _reference_transitions(t, sigma)
    assert (row['sep_n_exits'], row['sep_n_captures'], row['sep_n_crossings']) == (exits, captures, distinct)
    times = sorted(float(x) for x in str(row['sep_crossing_times_yr']).split(';'))
    assert times[0] == pytest.approx(slips[0], abs=P0_YR / 2)  # the slip's moment is the passage over the top
    assert times[1] == pytest.approx(final, abs=2 * P0_YR)  # midway between the last half-cycle and the first turn


def test_exit_and_capture_marks_the_slip_record():
    """Episodic, not co-orbital, one exit and one capture at least, the force law measured and passing (a swing of
    ~100 deg), step 1 yr (below P_orb / 2 = 1.98 yr), no jump of a: every documented condition holds."""
    t, sigma, rate = _driven_pendulum()
    verdict = _analyse(_series(t, sigma, rate)).verdict
    assert str(verdict.status) == 'transient'
    assert str(verdict.angle_status) == 'episodic'
    assert str(verdict.checks['check_mmr_a_response']) == 'pass'
    assert 'exit_and_capture' in verdict.flags
    assert 'exit_and_capture' not in verdict.caps  # a mark: no ceiling of its own


@pytest.mark.slow
def test_exit_without_capture_is_not_marked():
    """The same record without the pulse: one exit, no capture."""
    times, levels = zip(*[(x, h) for x, h in SCHEDULE if not 6050.0 < x < 6350.0])

    def rhs(x, y):
        sigma, rate = y
        gain = (np.interp(x, times, levels) - _energy(sigma, rate)) / 150.0 * 2 * W0**2 / (rate**2 + 0.05 * W0**2)
        return [rate, -(W0**2) * np.sin(sigma) + gain * rate]

    t = np.arange(0.0, 20000.5, 1.0)
    sol = solve_ivp(rhs, (0.0, 20000.0), [2 * np.arcsin(np.sqrt(0.6)), 0.0], t_eval=t, rtol=1e-9, atol=1e-11, method='DOP853', max_step=2.0)
    exits, captures, distinct, _, _ = _reference_transitions(t, sol.y[0])
    assert (exits, captures, distinct) == (1, 0, 1)
    analysis = _analyse(_series(t, sol.y[0], sol.y[1]))
    row = analysis.numbers_row()
    assert (row['sep_n_exits'], row['sep_n_captures'], row['sep_n_crossings']) == (1, 0, 1)
    assert 'exit_and_capture' not in analysis.verdict.flags


def test_aliased_step_withholds_the_mark_but_not_the_counts():
    """Every second sample of the slip record: step 2 yr >= P_orb / 2 = 1.98 yr folds the short periods
    (short_period_aliased), one of the documented exclusions; the angle's events are the same."""
    t, sigma, rate = _driven_pendulum()
    analysis = _analyse(_series(t[::2], sigma[::2], rate[::2]))
    assert 'short_period_aliased' in analysis.verdict.flags
    row = analysis.numbers_row()
    assert (row['sep_n_exits'], row['sep_n_captures']) == (2, 1)
    assert 'exit_and_capture' not in analysis.verdict.flags


# --------------------------------------------------------------------------------------------------------------------
# separatrix_chaos in old files


# the verdict columns of 52356 5J-2S-2+0+0-1 in benchmark/data/v2-w16r/summary.csv (written before the rename), with
# the angle_status of files before 2026-10-17 ('transient')
OLD_ROW = {
    'name': '52356',
    'resonance': '5J-2S-2+0+0-1',
    'resonance_type': 'mmr',
    'status': 'transient',
    'angle_status': 'transient',
    'angle_subtype': 'recurrent',
    'confidence': 'medium',
    'confidence_raw': 'high',
    'reason': 'dynamic_episode',
    'flags': 'separatrix_chaos;zone_eq_only',
    'caps': 'separatrix_chaos',
    'libration_origin': 'dynamic',
    'status_code': '1',
    'thin_margins': 'L2;L4',
    'failed_support': '',
    'check_mmr_separatrix_zone': 'pass',
}


def test_old_summary_row_is_read_under_the_new_names(tmp_path):
    path = tmp_path / 'summary.csv'
    pd.DataFrame([OLD_ROW]).to_csv(path, index=False)
    frame = SimulationSerializer._summary_frame(path)
    row = frame.to_dict('records')[0]
    verdict = SimulationSerializer._verdict_of_row(row, row['resonance'], True)
    assert verdict.flags == ['episodes_in_separatrix_zone', 'zone_eq_only']
    assert verdict.caps == ['episodes_in_separatrix_zone']
    assert str(verdict.angle_status) == 'episodic'
    # the cap keeps its medium ceiling (author 2026-10-16, 52356)
    assert str(confidence.apply_caps(verdict.confidence_raw, verdict.caps)) == 'medium'
    # and is written back under the new name only
    rewritten = verdict.to_flat_dict()
    assert 'separatrix_chaos' not in rewritten['flags'] + rewritten['caps']
    assert Verdict.from_flat_dict(rewritten).caps == verdict.caps


def test_new_records_never_write_the_old_name():
    """A pendulum whose strength is modulated by e (+-10 % over 8 kyr) crosses the separatrix back and forth: the
    zone flag is raised, under its new name, and the sep_ column is renamed too. Jupiter's elements are not saved:
    with them the cycles would be classed in Jupiter's averaged 3J-1 well (W20 CS6), whose depth has nothing to do
    with this synthetic pendulum's w0; without them the zone is the pendulum's, the model this test is about."""
    t = np.arange(0.0, 40000.5, 1.0)
    e = 0.15 * (1 + 0.1 * np.sin(2 * np.pi * t / 8000.0))
    modulation = lambda x: (1 + 0.1 * np.sin(2 * np.pi * x / 8000.0)) ** 2  # noqa: E731  K ~ e^2 for 3J-1
    sol = solve_ivp(
        lambda x, y: [y[1], -(W0**2) * modulation(x) * np.sin(y[0])],
        (0.0, 40000.0),
        [0.0, 2 * W0 * np.sqrt(1.02)],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        method='DOP853',
    )
    series = _series(t, sol.y[0], sol.y[1])
    series = dataclasses.replace(series, e=e, planets={})
    analysis = _analyse(series)
    assert analysis.numbers_row()['sep_zone_source'] == 'pendulum'
    flat = analysis.verdict.to_flat_dict()
    row = analysis.numbers_row()
    assert 'episodes_in_separatrix_zone' in analysis.verdict.flags
    assert 'separatrix_chaos' not in flat['flags'] + flat['caps']
    assert 'sep_separatrix_chaos' not in row
    assert bool(row['sep_episodes_in_separatrix_zone'])
