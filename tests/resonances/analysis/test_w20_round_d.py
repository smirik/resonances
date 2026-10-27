"""W20 round D: the cap short_period_aliased without a lift by the record, the harmonic trigger below the fundamental
step, the phase return at the record's edge, the one 1:1 predicate, the forced-wave word, R0 of a 1:1 angle from
the law the tree reads, and the header of the figure.

Independent numbers:
- folded lines (by hand): a line of frequency m/P sampled at dt folds to |m/P - j/dt|, j = round(m dt / P). P_orb
  12 yr at dt 5.999 yr: the second harmonic (m dt / P = 0.99983, j = 1) folds to 1/|1/6 - 1/5.999| = 6 x 5.999 / 0.001
  = 35 994 yr; at dt 4.002 yr the third (1.0005, j = 1) to 4 x 4.002 / 0.002 = 8004 yr and the second
  (0.667, j = 1) to 1/|1/6 - 1/4.002| = 12.02 yr; P = dt = 5 yr folds onto zero frequency (inf);
- the scatter ratios of the benchmark rows were measured independently on fine reruns of the same orbits (MEASURE,
  dec/clean at 0.1 yr): 2000 GB2 3J-1 0.05, 2013 AS27 31E-36 1.21 (smp_da_short_period_au / ep_a_excursion_meas_au);
- the edge cosine: sigma = cos(2 pi t / 400) on [-104, 304] yr, turning points at 0 and 200 yr: the mid-level 0 is
  crossed upwards at -100 and 300 yr (inside), the level 0.45 of the swing (-0.1) at -100 - 400 asin(0.1) / (2 pi) =
  -106.4 yr and 0.55 (+0.1) at 306.4 yr, both beyond the record;
- 624 1J+1+0-2 (sigma = lambda_J + lambda - 2 varpi) has |k| = |k_J| = 1 but k_J = +k: no co-orbital argument.
"""

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis import geometry as g
from resonances.resonance.analysis.confidence import Cap
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr.analyse import analyse_mmr, short_period_caps
from resonances.resonance.analysis.mmr.models import ResonanceSpec
from resonances.resonance.analysis.mmr.sampling import folded_slowest_yr
from resonances.resonance.analysis.settings import settings_from_config, validate_settings
from resonances.resonance.analysis.verdict import Flag

sys.path.insert(0, str(Path(__file__).parent))
from test_w20_fix2 import C, ONE_DEG, filtered  # noqa: E402

from tests.resonances.analysis.builders import make_sampling  # noqa: E402

S = settings_from_config()
LIFT = replace(S, mmr_short_period_cap_ratio=0.1)  # the W20 F5 lift, now opt-in
CAP = str(Cap.SHORT_PERIOD_ALIASED)
FLAGGED = str(Flag.SHORT_PERIOD_ALIASED)


def _sampling(flags=(), folded=float('nan'), da=float('nan'), excursion=float('nan')):
    """The sampling the cap reads; the scatter ratio as the analysis sets it (`decide.a_scatter_ratio`)."""
    sampling = make_sampling(flags=list(flags), folded_slowest_yr=folded, short_period_da_au=da)
    sampling.a_scatter_ratio = decide.a_scatter_ratio(sampling, excursion)
    return sampling


def _analysis(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    return analyse_mmr(io.load_series(example), S)


# --------------------------------------------------------------------------------------------- 1. the cap


@pytest.mark.parametrize(
    'orbit_yr, step_yr, harmonics, expected',
    [(12.0, 5.999, 2, 35994.0), (12.0, 4.002, 3, 8004.0), (12.0, 4.002, 2, 1 / abs(1 / 6 - 1 / 4.002)), (5.0, 5.0, 2, math.inf)],
)
def test_folded_lines_by_hand(orbit_yr, step_yr, harmonics, expected):
    assert folded_slowest_yr(orbit_yr, {}, step_yr, harmonics) == pytest.approx(expected, rel=1e-6)
    assert math.isnan(folded_slowest_yr(orbit_yr, {}, step_yr, 0))


@pytest.mark.parametrize(
    'step_yr, harmonics, capped',
    [
        (5.999, 2, True),
        # a known miss, kept on purpose: only m <= MMR_SHORT_PERIOD_HARMONICS (2) is searched, and the record bounds no
        # harmonic's amplitude, so the count is a convention; this case is caught only with 3 (next row)
        (4.002, 2, False),
        (4.002, 3, True),
        (5.999, 0, False),
    ],
)
def test_a_harmonic_folded_into_the_band_caps_below_the_fundamental_step(step_yr, harmonics, capped):
    """Review C #3: dt below P_orb / 2 = 6 yr sets no flag, but a harmonic folded slower than the 150-yr floor sets the
    flag and the cap (Codex's 12 yr at 4.002 yr needs the third harmonic)."""
    settings = replace(S, mmr_short_period_harmonics=harmonics)
    sampling = _sampling(folded=folded_slowest_yr(12.0, {}, step_yr, harmonics))
    caps = short_period_caps(sampling, math.radians(1.0), math.radians(34.0), settings, Ledger.from_settings(settings), 150.0)
    assert (caps == [CAP]) is capped
    assert (FLAGGED in sampling.flags) is capped


@pytest.mark.parametrize('fast_deg, ratio', [(1.0, 0.05), (1.0, 0.4), (10.0, 0.05)])
def test_the_record_never_lifts_the_cap_by_default(fast_deg, ratio):
    """MMR_SHORT_PERIOD_CAP_RATIO = 0: a flagged record keeps the cap whatever its fast residual and scatter of a; the
    scatter is compared (a warning) without a convention, so it makes no thin margin."""
    sampling = _sampling([FLAGGED], folded=50.0, da=ratio * 1e-3, excursion=1e-3)
    ledger = Ledger.from_settings(S)
    assert short_period_caps(sampling, math.radians(fast_deg), math.radians(34.0), S, ledger, 150.0) == [CAP]
    (entry,) = [e for e in ledger.entries if e.column == 'smp_da_sp_over_a_excursion']
    assert entry.value == pytest.approx(ratio) and entry.setting == '' and not ledger.thin_margins()
    assert str(Flag.A_SHORT_PERIOD_SCATTER) not in sampling.flags


@pytest.mark.parametrize(
    'fast_deg, ratio, excursion, capped, warned',
    [
        (1.0, 0.12, 1e-3, False, False),  # 1221 3E-8 type: both small, the opt-in lift
        (1.0, 0.79, 1e-3, True, True),  # 433 47M-44: the folded lines reach a
        (10.0, 0.12, 1e-3, True, False),  # the fast spread of sigma_f
        (1.0, 0.12, float('nan'), True, False),  # no a excursion measured
    ],
)
def test_the_opt_in_lift_needs_both_sigma_and_a(fast_deg, ratio, excursion, capped, warned):
    sampling = _sampling([FLAGGED], folded=50.0, da=ratio * 1e-3, excursion=excursion)
    caps = short_period_caps(sampling, math.radians(fast_deg), math.radians(34.0), LIFT, Ledger.from_settings(LIFT), 150.0)
    assert (caps == [CAP]) is capped
    assert (str(Flag.A_SHORT_PERIOD_SCATTER) in sampling.flags) is warned


def test_the_scatter_warning_on_a_flagged_record():
    sampling = _sampling([FLAGGED], folded=50.0, da=1.21e-3, excursion=1e-3)
    assert short_period_caps(sampling, math.radians(1.0), math.radians(34.0), S, Ledger.from_settings(S), 150.0) == [CAP]
    assert str(Flag.A_SHORT_PERIOD_SCATTER) in sampling.flags


@pytest.mark.parametrize(
    'example, warned, level',
    [('mmr2__2000_GB2__3J-1+0-2', False, 'medium'), ('mmr2__2013_AS27__31E-36+0+5', True, 'medium')],
)
def test_benchmark_rows_keep_the_cap_and_warn_by_their_measured_scatter(example, warned, level):
    """GB2 (scatter 0.05 of the a swing) and AS27 (1.21) keep the cap (medium); only AS27 carries the warning."""
    result = _analysis(example)
    v = result.verdict
    assert CAP in v.caps and str(v.confidence) == level
    assert (str(Flag.A_SHORT_PERIOD_SCATTER) in v.flags) is warned
    ratio = result.numbers_row()['smp_da_sp_over_a_excursion']
    assert (ratio > S.mmr_short_period_a_ratio) is warned


# --------------------------------------------------------------------------------------------- 2. the phase return


def _edge_cosine(lo=-104.0, hi=304.0, cutoff=5.0):
    t = np.arange(lo, hi + 0.125, 0.25)
    sigma = np.cos(2 * np.pi * t / 400)
    angle = filtered(t, sigma, cutoff, np.gradient(sigma, t) / C)
    return angle, g.measure_geometry(angle, ONE_DEG, S).turning_points


@pytest.mark.parametrize('cutoff', [5.0, 2.0])
def test_a_mid_level_without_observable_neighbours_closes_alone(cutoff):
    """Review C #1: the levels 0.45 and 0.55 cross beyond the record (-106.4 and 306.4 yr): unobservable, they do
    not break the mid-level's band, which closes the 400-yr cycle alone."""
    angle, tp = _edge_cosine(cutoff=cutoff)
    assert np.asarray(tp.times_yr) == pytest.approx([0.0, 200.0], abs=1.0)
    pair = np.array([0, 1])
    t = np.asarray(angle.t_yr)
    i1, i2 = (int(i) for i in np.asarray(tp.indices))
    samples = g.return_samples(t, (i1, i2), None, S.libration_return_period_tol)
    clipped = g.return_clipped(t, (i1, i2), None, S.libration_return_period_tol)
    assert clipped == (True, True)
    assert [g.return_observable(angle, tp, pair, f, samples, clipped) for f in (0.45, 0.5, 0.55)] == [False, True, False]
    cycle = g.phase_return(angle, tp, pair, S)
    assert cycle is not None and cycle.level_fraction == 0.5
    assert (cycle.t_a_yr, cycle.t_b_yr) == (pytest.approx(-100.0, abs=1.0), pytest.approx(300.0, abs=1.0))
    assert g.phase_return(angle, tp, pair, replace(S, libration_return_levels=1)).level_fraction == 0.5


def test_return_level_counts_observable_levels_only():
    fractions = [0.4, 0.45, 0.5, 0.55, 0.6]
    alone = [False, False, True, False, False]
    nan = [float('nan')] * 5
    assert g.return_level(fractions, alone, nan, False, 2) is None  # neighbours observed and failing
    assert g.return_level(fractions, alone, nan, False, 2, [False, False, True, False, False]) == 2
    assert g.return_level(fractions, alone, nan, False, 2, [False, True, True, False, False]) is None  # 0.45 seen, failed
    assert g.return_level(fractions, alone, nan, False, 2, None, 1) == 2  # LIBRATION_RETURN_MID_BAND_LEVELS = 1
    assert g.return_level([0.5], [True], [0.1], True, 2) == 0  # LIBRATION_RETURN_LEVELS = 1: the mid-level only


def test_the_mid_band_must_be_at_least_one():
    with pytest.raises(ValueError, match='at least 1'):
        validate_settings(replace(S, libration_return_mid_band_levels=0))


# --------------------------------------------------------------------------------------------- 2. one 1:1 predicate


def test_one_predicate_for_a_one_to_one_angle():
    from resonances.resonance.analysis.mmr.preprocess import law_planet
    from resonances.resonance.analysis.mmr.strength import direct_applicable, one_to_one

    for key, planet in (('1E-1+0+0', 'Earth'), ('1J-1r0', 'Jupiter'), ('1J+1+0-2', None), ('3J-1+0-2', None)):
        spec = ResonanceSpec.from_key(key)
        assert law_planet(spec) == planet and one_to_one(spec) is (planet is not None)
    assert direct_applicable(ResonanceSpec.from_key('1J+1+0-2'))


def test_624_is_judged_by_its_strength():
    v = _analysis('mmr2__624__1J+1+0-2').verdict
    assert str(v.checks['check_mmr_strength']) != 'not_applicable'


# --------------------------------------------------------------------------------------------- 2. the forced-wave word


def test_129772_keeps_its_word_and_385266_its_calibration_cell():
    """Review C #5: the rewrite touches circulation and drift words only (129772 4N-7 stays insufficient_time/arc under
    G6); review C #9: the confidence reads the tree's word, so 385266's cell is the circulation's, not blank."""
    v = _analysis('mmr2__129772__4N-7+0+3').verdict
    assert (str(v.angle_status), str(v.angle_subtype), str(v.reason)) == ('insufficient_time', 'arc', 'baseline_needed')
    w = _analysis('mmr2__385266__16N-27+0+11').verdict
    assert str(w.angle_status) == 'episodic' and str(w.reason) == 'circulation'
    assert w.cell_n_examples is not None and w.cell_n_examples > 0
    assert 'forced wave about' in w.orbit_type and 'libration' not in w.orbit_type


# --------------------------------------------------------------------------------------------- 3. R0 of a 1:1 angle


def test_wj_reads_r0_from_the_witness_law():
    """2024 WJ: the witness episode's law (energy R^2 0.043, the force law passes) gives R0 not_measured, the number
    printed; before, R0 passed on the pooled law (0.82) while the figure printed 0.043."""
    from resonances.plotting import verdict_text as vt

    result = _analysis('coorbital__2024_WJ__1E-1+0+0')
    law = result.energy.potential_law
    assert law is result.response.coorbital or law.energy_r2 == pytest.approx(result.response.coorbital.energy_r2)
    assert law.energy_r2 < S.mmr_force_r2_fail and str(result.verdict.checks['check_mmr_potential']) == 'not_measured'
    # the tree's entry is the co-orbital one (not the harmonic V_E ~ V_F), not measured as R0 is
    (entry,) = [e for e in result.verdict.decision_path if e.name == '5 R0 co-orbital energy integral R2']
    assert entry.value is None
    row = {**result.numbers_row(), **result.summary_row()}
    facts = vt.MMRFacts(result, S, row, result.episode_rows())
    assert vt._potential_detail(facts)[0].startswith('1:1 energy integral R² 0.0426 <')
    assert 'E is not conserved' in vt._skip_potential(facts)[1]  # not the harmonic V_E reason


def test_a_one_to_one_circulation_prints_the_r0_that_decided():
    """2023 QS7: a circulation keeps the harmonic R0 (no episode for the energy integral), and the figure says so."""
    from resonances.plotting import verdict_text as vt

    result = _analysis('coorbital__2023_QS7__1V-1+0+0')
    assert result.energy.potential_law is None
    row = {**result.numbers_row(), **result.summary_row()}
    assert vt._potential_detail(vt.MMRFacts(result, S, row, result.episode_rows()))[0].startswith('V_E ≈ V_F')


# --------------------------------------------------------------------------------------------- 4. the header


def test_a_moving_centre_names_its_mean():
    v = _analysis('mmr2__2013_AS27__31E-36+0+5').verdict
    assert 'libration about a moving centre (mean ' in v.orbit_type and 'symmetric' not in v.orbit_type


def test_the_owner_rejection_prints_the_bar_that_rejected():
    """Review C #4: owner.py rejects on MMR_OWNER_R2_MIN (0.7), so an R² of 0.57 (105247 29M-52) prints '0.57 < 0.7',
    not '0.57 ≥ 0.4' (the W margin's bar)."""
    from resonances.plotting import verdict_text as vt

    candidate = SimpleNamespace(r2_median=0.57, reason='r2_low')
    assert vt._owner_rejection(SimpleNamespace(s=S), candidate) == 'R² 0.57 < 0.7'


def test_a_long_group_header_wraps_and_stays_with_its_lines():
    """The EPISODES header of a forced-wave row (55 characters) in a 2.2-in column at 8 pt: the header is drawn at
    8.3 pt, 2.2 x 72 / (0.602 x 8.3) = 31.7 -> 31 characters a line, so it takes two rows, kept with its first lines."""
    from resonances.plotting import combined_numbers as cn

    group = 'EPISODES (2 accepted (2 forced waves), 3 rejected runs)'
    lines = [cn.NumberLine('numbers', '', f'#{i} 56.6 kyr-68.3 kyr 7 TP', group=group) for i in range(3)]
    rows = cn._number_rows(lines, 2.2, 8.0)
    assert [kind for kind, *_ in rows[:3]] == ['group', 'group_cont', 'line']
    assert all(len(text) <= 31 for kind, text, _ in rows if kind.startswith('group'))
    assert ' '.join(text.strip() for kind, text, _ in rows if kind.startswith('group')) == group
    first = cn._items(rows)[0]
    assert [kind for kind, *_ in first] == ['group', 'group_cont', 'line', 'line']


def test_symlog_decade_labels_do_not_overlap():
    """142694 41J-14: +-10^8 on the symlog axis of a 0.6-in panel. Labels of 7 pt need 1.2 x 7 = 8.4 pt between
    ticks. 8.4 / (0.6 x 72) = 0.194 of the axis."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from resonances.plotting.mmr_combined import symlog_ticks

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes((0.1, 0.4, 0.8, 0.1))  # 0.6 in high
    ax.set_yscale('symlog', linthresh=1.0, linscale=1.0)
    ax.set_ylim(-3e8, 3e8)
    ticks = symlog_ticks(ax, -3e8, 3e8, 0.1, 7.0)
    to_axes = ax.transData + ax.transAxes.inverted()
    positions = sorted(to_axes.transform((0.0, v))[1] for v in ticks)
    assert 0.0 in ticks and len(ticks) >= 3
    assert min(np.diff(positions)) >= 1.2 * 7.0 / (0.6 * 72) - 1e-9
    plt.close(fig)
