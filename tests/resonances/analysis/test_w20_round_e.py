"""W20 round E: one predicate of "the step folds short periods" for the analysis and the dry-run, the 1:1 predicate in
the header, the R0 text of a 1:1 angle, the stale R0 of a pooled 1:1 law, the cap's title and short reason, and the
round-D branches that had no test.

Independent numbers (by hand):
- a line of frequency m/P sampled at dt folds to |m/P - j/dt|, j = round(m dt / P). 1J-1 with P_orb = 12 yr
  (a = 12^(2/3) au, P = a^(3/2)) at dt = 5.999 yr: dt < P_orb / 2 = 6 yr (no Nyquist flag), the second harmonic
  (2 dt / P = 0.99983, j = 1) folds to 1 / |1/6 - 1/5.999| = 6 x 5.999 / 0.001 = 35 994 yr; P_syn with Jupiter
  (11.862 yr) is ~1 kyr, its harmonics are not folded (2 dt / P_syn = 0.012). The co-orbital period floor is
  0.5 x P_J / sqrt(27 mu_J / 4) = 0.5 x 147.75 = 73.9 yr: 35 994 > 73.9, both sides flag it;
- the 2J-1 pendulum record (n = 2 n_J, P_orb = pi / n_J = 5.931 yr, P_syn,J = 11.86 yr) sampled at dt = 0.995 P_orb / 2:
  the second harmonic of P_orb folds to 1 / |2/P - 1/dt| ~ dt / 0.005 ~ 590 yr (561.6 yr with the record's own mean a),
  above the 150-yr floor;
- a co-orbital table holds 3 arrays of N doubles: 16 points 384 bytes, 32 points 768 bytes.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import resonances
from resonances.data import const
from resonances.plotting import verdict_text as vt
from resonances.resonance.analysis import orbit_type as ot
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import coorbital
from resonances.resonance.analysis.mmr import context as context_module
from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.mmr import preprocess, response as response_module, sampling
from resonances.resonance.analysis.mmr.analyse import analyse_mmr, short_period_folds
from resonances.resonance.analysis.mmr.coorbital import Orbit
from resonances.resonance.analysis.mmr.models import CheckValue, ForcedWave, MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.pendulum import EnergyAnalysis
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import Flag
from resonances.simulation import dry_run as dr
from tests.resonances.analysis.builders import make, make_episode, make_geometry

S = settings_from_config()
FLAGGED = str(Flag.SHORT_PERIOD_ALIASED)
N_J = math.radians(const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'])


# --------------------------------------------------------------------------------------------- 1. one predicate


@pytest.mark.parametrize(
    'step, short_step, folded, floor, expected',
    [
        (6.0, 6.0, 0.0, 150.0, sampling.SHORT_PERIOD_NYQUIST),  # on the Nyquist step: already flagged
        (5.999, 6.0, 35994.0, 150.0, sampling.SHORT_PERIOD_HARMONIC),
        (5.999, 6.0, 120.0, 150.0, ''),  # folded faster than the floor: filtered
        (5.999, 6.0, float('nan'), 150.0, ''),  # MMR_SHORT_PERIOD_HARMONICS = 0
        (5.999, 6.0, 35994.0, 0.0, ''),  # no floor
        (5.999, 6.0, math.inf, 150.0, sampling.SHORT_PERIOD_HARMONIC),  # folded onto zero frequency
    ],
)
def test_the_short_period_predicate(step, short_step, folded, floor, expected):
    assert sampling.short_period_trigger(step, short_step, folded, floor) == expected


def _one_to_one_series(step_yr, span_yr=3000.0):
    """1J-1 with P_orb = 12 yr: a fixed, lambda = lambda_J - 60 deg (sigma = lambda_J - lambda librates nowhere)."""
    t = np.arange(0.0, span_yr, step_yr)
    a = 12.0 ** (2 / 3) * np.ones_like(t)
    lam = 0.3 + N_J * t - math.radians(60.0)
    one = np.ones_like(t)
    return MMRSeries(
        t_yr=t,
        a=a,
        e=0.05 * one,
        inc=0.02 * one,
        Omega=1.1 * one,
        omega=0.4 * one,
        varpi=1.5 * one,
        angle_wrapped=np.mod(0.3 + N_J * t - lam, 2 * np.pi),
        resonance=ResonanceSpec.from_key('1J-1+0+0'),
        regime=Regime(step_yr=step_yr, span_yr=float(t[-1]), n_rows=len(t), integrator='ias15'),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - 1.5, 2 * np.pi),
        name='round-e',
    )


def test_dry_run_and_analysis_agree_below_the_nyquist_step():
    """Review D #3: 1J-1, P_orb 12 yr, dt 5.999 yr. The dry-run row and the analysis read the same predicate: both
    flag short_period_aliased through the folded second harmonic (35 994 yr > 73.9 yr), neither through the Nyquist
    step."""
    spec = ResonanceSpec.from_key('1J-1+0+0')
    a = 12.0 ** (2 / 3)
    inputs = dr.BodyInputs('round-e', a, 0.05, a - 0.01, a + 0.01, 1e-4, 1e-4, None, None)
    sim = resonances.Simulation(name='round-e', date='2026-09-14', save=None, plot=None, save_summary=False)
    settings, dry = sim.config.analysis_settings, sim.config.dry_run_settings
    tables = dr._planet_tables()
    limits = dr._row_limits(spec, inputs, settings, tables)
    row = dr._row_at_step(spec, inputs, limits, 5.999, 3000.0, settings, dry, tables)
    assert row['step_short_period_yr'] == pytest.approx(6.0, rel=1e-9)
    assert row['folded_slowest_yr'] == pytest.approx(35994.0, rel=1e-6)
    assert row['libration_floor_yr'] == pytest.approx(147.75 / 2, rel=2e-3)
    assert row['short_period_trigger'] == sampling.SHORT_PERIOD_HARMONIC
    assert row['expected_flags'] == FLAGGED and 'a harmonic (m <= 2) folds to 3.599e+04 yr' in row['warnings']

    series = _one_to_one_series(5.999)
    context = context_module.build_context(series, S)
    analysed = sampling.analyse_sampling(series, context, S)
    assert FLAGGED not in analysed.flags  # dt < 6 yr: no Nyquist flag
    assert analysed.folded_slowest_yr == pytest.approx(35994.0, rel=1e-6)
    floor = preprocess.libration_period_floor(spec, context, S)[0]
    assert floor == pytest.approx(row['libration_floor_yr'], rel=1e-9)
    assert short_period_folds(analysed, S, Ledger.from_settings(S), floor) is True
    assert FLAGGED in analysed.flags


def test_the_dry_run_flag_at_a_step_that_folds_nothing():
    """The same body at 5 yr: 2 x 5 / 12 = 0.833 folds the second harmonic to 1 / |1/6 - 1/5| = 30 yr, faster than the
    floor: no flag on either side."""
    spec = ResonanceSpec.from_key('1J-1+0+0')
    a = 12.0 ** (2 / 3)
    inputs = dr.BodyInputs('round-e', a, 0.05, a - 0.01, a + 0.01, 1e-4, 1e-4, None, None)
    sim = resonances.Simulation(name='round-e', date='2026-09-14', save=None, plot=None, save_summary=False)
    tables = dr._planet_tables()
    limits = dr._row_limits(spec, inputs, sim.config.analysis_settings, tables)
    row = dr._row_at_step(spec, inputs, limits, 5.0, 3000.0, sim.config.analysis_settings, sim.config.dry_run_settings, tables)
    assert row['folded_slowest_yr'] == pytest.approx(30.0, rel=1e-6)
    assert row['short_period_trigger'] == '' and row['expected_flags'] == ''


def _pendulum_2j1(step_yr, span_yr=8000.0):
    """The 2J-1 pendulum (w11's construction: sigma = 2 lambda_J - lambda - varpi, 600-yr small-swing period, 90 deg)."""
    w0 = 2 * np.pi / 600.0
    t = np.arange(0.0, span_yr, step_yr)
    sol = solve_ivp(
        lambda _, y: [y[1], -(w0**2) * np.sin(y[0] - np.pi)],
        (t[0], t[-1]),
        [np.pi, w0 * 2 * math.sin(math.radians(45.0))],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        max_step=0.5,
    )
    sigma, rate = sol.y
    varpi = 0.7
    lam = 2 * (0.3 + N_J * t) - varpi - sigma
    one = np.ones_like(t)
    return MMRSeries(
        t_yr=t,
        a=(4 * np.pi**2 / (2 * N_J - rate) ** 2) ** (1.0 / 3.0),
        e=0.15 * one,
        inc=0.05 * one,
        Omega=1.1 * one,
        omega=(varpi - 1.1) * one,
        varpi=varpi * one,
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=ResonanceSpec.from_key('2J-1'),
        regime=Regime(step_yr=step_yr, span_yr=float(t[-1]), n_rows=len(t), integrator='ias15'),
        mean_longitude=np.mod(lam, 2 * np.pi),
        mean_anomaly=np.mod(lam - varpi, 2 * np.pi),
        name='round-e-2J-1',
    )


def test_the_harmonic_trigger_reaches_the_verdict_end_to_end():
    """Review D #6: a full analyse_mmr below the Nyquist step (dt = 0.995 P_orb / 2) flags short_period_aliased by the
    folded harmonic, caps the level at medium and writes the flag into summary.csv's row."""
    step = 0.995 * (math.pi / N_J) / 2
    result = analyse_mmr(_pendulum_2j1(step), S)
    smp = result.sampling
    assert step < smp.short_period_step_yr  # below min(P_orb, P_syn) / 2
    orbit = 2 * smp.short_period_step_yr  # the record's own P_orb (P_syn,J = 2 P_orb is longer)
    assert smp.folded_slowest_yr == pytest.approx(1 / abs(2 / orbit - 1 / step), rel=1e-9)
    assert 500.0 < smp.folded_slowest_yr < 600.0
    v = result.verdict
    assert FLAGGED in v.flags and FLAGGED in v.caps and str(v.confidence) in ('medium', 'low')
    assert FLAGGED in str(result.summary_row()['flags']).split(';')
    (entry,) = [e for e in v.decision_path if e.name.startswith('short_period_aliased: a harmonic')]
    assert entry.result is True and entry.threshold == pytest.approx(S.mmr_libration_period_min_yr)


# --------------------------------------------------------------------------------------------- 2. the header


def _run(episode_id, centre_deg, start=0.0, end=1000.0):
    tp = np.radians([centre_deg - 20, centre_deg + 20, centre_deg - 20])
    return make_episode(
        episode_id=episode_id,
        tp_times_yr=np.array([start, (start + end) / 2, end]),
        tp_values_rad=tp,
        centre_mod_2pi_rad=np.radians(centre_deg % 360),
        centre_shift_over_amp=0.1,
    )


def test_a_one_to_one_by_magnitudes_only_is_no_coorbital_in_the_header():
    """Review D #6: 1J+1+0-2 (|k_J| = |k| = 1, k_J = +k) is not a 1:1 angle (`law_planet` None): an accepted run about
    180 deg is a symmetric libration, not a tadpole or horseshoe; 1J-1 about 60 deg stays a tadpole."""
    spec = ResonanceSpec.from_key('1J+1+0-2')
    assert preprocess.law_planet(spec) is None
    geometry = make_geometry(runs=[_run(0, 180)])
    assert ot.mmr_orbit_type(spec, geometry, 0, None, S, 'libration') == 'symmetric libration about 180°'
    trojan = make_geometry(runs=[_run(0, 300)])  # sigma = lambda_J - lambda about 300 deg: theta = -sigma about 60
    assert ot.mmr_orbit_type(ResonanceSpec.from_key('1J-1+0+0'), trojan, 0, None, S) == 'tadpole L4'


def test_the_header_describes_the_real_episode_next_to_a_forced_wave():
    """Round D: a forced wave (id 1) is described only without a real episode (id 0)."""
    spec = ResonanceSpec.from_key('4M-7+0+3')
    real, wave = _run(0, 100, end=5000.0), _run(1, 214, start=6000.0, end=6500.0)
    both = make_geometry(runs=[real, wave], forced_waves=(make(ForcedWave, episode_id=1),))
    assert ot.mmr_orbit_type(spec, both, 1, None, S, 'episodic') == 'exterior, asymmetric libration about 100°'
    alone = make_geometry(runs=[wave], forced_waves=(make(ForcedWave, episode_id=1),))
    assert ot.mmr_orbit_type(spec, alone, 1, None, S, 'episodic') == 'exterior, asymmetric forced wave about 214°'


# --------------------------------------------------------------------------------------------- 3. R0 of a 1:1 angle


def _law(potential, energy, force, energy_r2):
    return SimpleNamespace(potential_check=potential, energy_check=energy, force_check=force, energy_r2=energy_r2)


def _facts(law):
    energy = EnergyAnalysis(None, None, 1.0, 1.0, potential_law=law)
    return SimpleNamespace(s=S, a=SimpleNamespace(energy=energy, geometry=None), amplitude_max_deg=lambda: None)


def test_r0_text_says_not_measured_when_the_energy_integral_was_not_measured():
    """Review D #7: energy R^2 nan (energy_check not_measured): no claim that E is not conserved; '?' (missing), not
    the pale '–' of a test skipped by design, as the confidence counts it as not measured."""
    nan = float('nan')
    state, text = vt._skip_potential(_facts(_law('not_measured', 'not_measured', 'pass', nan)))
    assert (state, text) == (vt.CheckState.MISSING, 'the 1:1 energy integral was not measured')
    assert vt.ICONS[state] == '?'
    state, text = vt._skip_potential(_facts(_law('not_measured', 'fail', 'pass', 0.0426)))
    assert state == vt.CheckState.MISSING and text.startswith('the 1:1 energy integral fails (R² 0.043)')
    assert 'E is not conserved' in text


@pytest.mark.benchmark
def test_wj_r0_is_drawn_as_not_measured():
    """2024 WJ (energy R^2 0.043, the force law passes): its R0 row is '?' and counted as not measured."""
    example = 'coorbital__2024_WJ__1E-1+0+0'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    result = analyse_mmr(io.load_series(example), S)
    row = {**result.numbers_row(), **result.summary_row()}
    text = vt.explain_mmr(result, S, row, result.episode_rows())
    (r0,) = [check for check in text.checks if check.key == 'R0']
    assert r0.icon == '?' and r0.counts == vt.COUNTS_NOT_MEASURED and 'E is not conserved' in r0.note


def test_a_stale_pooled_law_gives_r0_back_to_the_harmonic_test():
    """Review D #2 (W23: a pure copy, `with_coorbital_potential`): R0 comes from the law of the response passed (rule
    R's), never from a law applied before; a response without a law (or no episode, or filter_too_slow) keeps the
    harmonic R0 and no law. The measured energy is never changed."""
    pooled = SimpleNamespace(coorbital=SimpleNamespace(potential_check=CheckValue.PASS))
    witness = SimpleNamespace(coorbital=SimpleNamespace(potential_check=CheckValue.NOT_MEASURED))
    episodes = SimpleNamespace(episodes=[object()])
    energy = EnergyAnalysis(None, None, 1.0, 1.0, with_slow=CheckValue.FAIL)  # the harmonic V_E ~ V_F result
    read = response_module.with_coorbital_potential(energy, pooled, episodes, False)
    assert read.with_slow == CheckValue.PASS and read.potential_law is pooled.coorbital
    read = response_module.with_coorbital_potential(energy, witness, episodes, False)
    assert read.with_slow == CheckValue.NOT_MEASURED and read.potential_law is witness.coorbital
    assert response_module.with_coorbital_potential(energy, SimpleNamespace(coorbital=None), episodes, False) is energy
    for response, geometry, too_slow in ((pooled, SimpleNamespace(episodes=[]), False), (pooled, episodes, True)):
        assert response_module.with_coorbital_potential(energy, response, geometry, too_slow) is energy
    assert energy.with_slow == CheckValue.FAIL and energy.potential_law is None  # the measurement itself is untouched


# --------------------------------------------------------------------------------------------- 4. the figure text


def test_the_tree_word_of_a_forced_wave_rewrite():
    assert vt._tree_word(SimpleNamespace(v=SimpleNamespace(reason='kinematic_drift'))) == 'a drift'
    assert vt._tree_word(SimpleNamespace(v=SimpleNamespace(reason='circulation'))) == 'a circulation'


class _Facts:
    """The ledger entries and sampling columns `_short_period_title` reads."""

    def __init__(self, entries, row):
        self.entries, self.row = entries, row

    def entry(self, name):
        return self.entries.get(name)


def _entry(value, threshold, result):
    return SimpleNamespace(value=value, threshold=threshold, result=result, op='>')


@pytest.mark.parametrize(
    'entries, title, short',
    [
        ({}, 'the output step folds short periods into slow ones (Δt 5 yr ≥ 0.85 yr = min(P_orb, P_syn)/2)', ''),
        (
            {vt.SHORT_PERIOD_HARMONIC: _entry(35994.0, 150.0, True)},
            'the output step folds a harmonic of the short periods slower than the period floor',
            'harmonic folded to 36 kyr',
        ),
        (
            {vt.SHORT_PERIOD_A: _entry(1.21, 0.5, True)},
            'the output step folds short periods into slow ones (Δt 5 yr ≥ 0.85 yr = min(P_orb, P_syn)/2)',
            'a scatter 1.21 × half-swing',
        ),
    ],
)
def test_the_cap_names_the_condition_that_fired(entries, title, short):
    """Verify D: the cap's title names only the condition that fired, with ≥; the short reason (compact line,
    confidence sentence) says what holds it (AS27: a scatter 1.21 x the half-swing)."""
    facts = _Facts(entries, {'smp_step_yr': 5.0, 'smp_short_period_step_yr': 0.85})
    assert vt._short_period_title(facts) == (title, short)
    assert '>=' not in vt.CAP_WORDS['short_period_aliased']


# --------------------------------------------------------------------------------------------- 5. memory


def test_record_tables_account_bytes_over_mixed_grids_and_nested_scopes():
    """A 2000-byte store: 32-point (768 B), 16-point (384 B), 32-point (768 B) tables are kept (1920 B); the next
    16-point one (2304 B) is not; an inner scope with another bound uses the outer store."""
    coorbital.cached_table.cache_clear()
    planet = Orbit(0.0167, 0.0, 0.0, 1.8)
    bodies = [Orbit(0.05 + 1e-4 * i, 0.1, 0.3, 0.2) for i in range(4)]
    with coorbital.record_tables(2000):
        for body, points in zip(bodies[:3], (32, 16, 32)):
            coorbital.table_for(body, planet, points, 1e-3)
        assert coorbital.record_tables_kept() == 3 and coorbital._RECORD_BYTES == [1920, 2000]
        with coorbital.record_tables(1e9):
            coorbital.table_for(bodies[3], planet, 16, 1e-3)
            assert coorbital._RECORD_BYTES == [1920, 2000]
        assert coorbital.record_tables_kept() == 3
        assert coorbital.table_bytes(coorbital.table_for(bodies[3], planet, 16, 1e-3)) == 384
    assert coorbital.record_tables_kept() == 0 and coorbital._RECORD_BYTES == [0, 0]
    coorbital.cached_table.cache_clear()
