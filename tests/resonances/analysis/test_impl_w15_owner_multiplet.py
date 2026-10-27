"""W15: the owner's multiplet law, the slow turns the ripple test counts, and the force law contaminated by a forced line.

Sources of the expected numbers:
- the multiplet: a_dot built from three members phi + j theta with the amplitudes measured on 101703 (2.4, 3.3 and
  0.26e-5 au/yr, EXPLANATIONS Q2); the single phase of their sum, fitted per window here with numpy, winds once per
  turn of theta because the j = 1 member outweighs the other two (3.3 > 2.4 + 0.26);
- the lattice line of 41J-14: at 41 n_J = 14 n the argument lambda - 3 lambda_J turns at n_J (41/14 - 3) = -n_J/14,
  a period of 14 P_J (3 x 14 - 41 = 1); the lines of 3J-2 are n_J |3m + 2l| / 2 >= n_J / 2, periods <= 2 P_J;
- 101703 41J-14 (benchmark record): the author's label non_resonant, the owner 3J-1+0-2 (impl/w13/explain/agent0.md:
  constant-coefficient fit R^2 0.90, single phase coherence 0.41), the line at 164.4 yr against P_e 166.2 yr (W12).
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from tests.resonances.analysis.builders import make_energy, make_kinematics, make_response, make_sampling
from resonances.resonance.analysis.mmr import forced_lines, io, owner
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.context import table_mean_motion
from resonances.resonance.analysis.mmr.models import OwnerCandidate, ResonanceSpec
from resonances.resonance.analysis.mmr.response import sliding_windows
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
TWO_PI = 2 * math.pi
NAN = float('nan')

# --- the multiplet law on a synthetic frame ----------------------------------------------------------------

P_WAVE, P_MOD, G_J = 166.0, 25000.0, 1.0e-5  # the forced wave, the period of varpi - varpi_J, g_J [rad/yr]
AMPLITUDES = (2.4e-5, 3.3e-5, 0.26e-5)  # members j = 0, 1, 2 [au/yr]
PHASES = (0.5, 2.0, -1.0)


def _frame(t, adot):
    """An OwnerFrame with what `owner_multiplet` reads (the fit grid is the whole grid)."""
    n = len(t)
    zeros, every = np.zeros(n), np.ones(n, dtype=bool)
    return owner.OwnerFrame(
        t_yr=t,
        mask=every,
        fit_mask=every,
        period_yr=P_WAVE,
        window_yr=S.mmr_force_window_cycles * P_WAVE,
        a_band=zeros,
        adot_band=adot,
        sigma_f=zeros,
        sigma_clean=zeros,
        sigma_band=zeros,
        sigma_windows={},
        sigma_slow=zeros,
        ripple_factor=S.mmr_owner_ripple_factor,
        sigma_theory_au_yr=1.0,
        body_n=1.0,
        body_n_offset=0.0,
        body_varpi_rate=0.0,
        a_mean_au=2.54,
        varpi_fit=(TWO_PI / P_MOD + G_J) * t,  # varpi - g_J t = theta
        planet_varpi_rates={'Jupiter': G_J},
    )


def _synthetic(span_yr=60000.0):
    t = np.arange(0.0, span_yr, 5.0)
    phi = TWO_PI * t / P_WAVE + 0.3
    theta = TWO_PI * t / P_MOD
    adot = sum(a * np.sin(phi + j * theta + p) for j, (a, p) in enumerate(zip(AMPLITUDES, PHASES)))
    window = S.mmr_force_window_cycles * P_WAVE
    starts = tuple(start for start, _ in sliding_windows(t, window, S.mmr_force_window_step))
    regime = owner.RegimeFit(owner.REGIME_FORCED_POSITIVE, starts, 0.99, (0.99,) * len(starts), 0.3, 1e-5, 0.5, 0.0, 1e-5)
    return t, phi, adot, regime


def _single_phase_coherence(t, phi, adot, starts, length):
    """Per window a_dot = A sin phi + B cos phi + C by numpy least squares; |<e^{i phase}>| over the windows."""
    phases = []
    for start in starts:
        inside = (t >= start) & (t <= start + length)
        design = np.column_stack([np.sin(phi[inside]), np.cos(phi[inside]), np.ones(np.count_nonzero(inside))])
        (a, b, _), *_ = np.linalg.lstsq(design, adot[inside], rcond=None)
        phases.append(np.arctan2(b, a))
    return float(np.abs(np.mean(np.exp(1j * np.array(phases)))))


def test_single_phase_of_a_multiplet_drifts_and_the_multiplet_law_fixes_it():
    t, phi, adot, regime = _synthetic()
    frame = _frame(t, adot)
    # the j = 1 member dominates: the single phase follows theta through 2.4 turns, the resultant is small
    assert _single_phase_coherence(t, phi, adot, regime.starts, frame.window_yr) < 0.5
    spec = ResonanceSpec.from_key('3J-1+0-2')  # p = -2: members j = 0, 1, 2
    fitted = owner.owner_multiplet(spec, phi, frame, regime, S)
    # windows of one modulation period stepped by half inside 60 kyr: starts 0, 12.5, 25 kyr
    assert fitted.multiplet_planet == 'Jupiter' and fitted.multiplet_windows == 3
    assert fitted.multiplet_r2 > 0.99 and fitted.multiplet_coherence > 0.99
    assert owner.multiplet_fixed(fitted, S)
    # sigma (zero here) fits nothing on the same long windows: the owner wins the long-window comparison
    assert fitted.sigma_multiplet_r2 < 0.05
    assert not owner.multiplet_fixed(fitted, replace(S, mmr_owner_law='single'))


def test_a_drifting_wrong_candidate_is_not_fixed_by_the_multiplet():
    """A candidate whose phase runs away from the driver by one turn per 7 kyr: no member j (0, 1/25, 2/25 turns per
    kyr) carries that rotation, so constant coefficients cannot fit a_dot."""
    t, phi, adot, regime = _synthetic()
    wrong = phi + TWO_PI * t / 7000.0
    fitted = owner.owner_multiplet(ResonanceSpec.from_key('3J-1+0-2'), wrong, _frame(t, adot), regime, S)
    assert fitted.multiplet_r2 < S.mmr_owner_r2_min and not owner.multiplet_fixed(fitted, S)


def test_the_multiplet_is_not_measured_without_two_modulation_windows_or_members():
    t, phi, adot, regime = _synthetic(span_yr=30000.0)  # one window of 25 kyr fits in 30 kyr
    fitted = owner.owner_multiplet(ResonanceSpec.from_key('3J-1+0-2'), phi, _frame(t, adot), regime, S)
    assert fitted == regime and math.isnan(fitted.multiplet_r2)
    t, phi, adot, regime = _synthetic()
    single_member = owner.owner_spec(('Venus', 'Saturn'), (1, 3), -4)  # p = 0: one member
    assert single_member.body_varpi_coefficient == 0
    assert owner.owner_multiplet(single_member, phi, _frame(t, adot), regime, S) == regime
    # only regimes whose R^2 passes and whose single phase fails are refitted
    assert owner.needs_multiplet(regime, S)
    assert not owner.needs_multiplet(replace(regime, coherence=0.85), S)
    assert not owner.needs_multiplet(replace(regime, r2=0.6), S)


def _candidate(coherence=0.4, multiplet_r2=0.855, multiplet_coherence=0.978, sigma_multiplet_r2=0.085):
    return OwnerCandidate(
        spec=ResonanceSpec.from_key('3J-1+0-2'),
        rate_rad_yr=0.0,
        r2_median=0.99,
        r2_windows=(0.99, 0.99),
        phi0_coherence=coherence,
        psi_share=0.03,
        force_au_yr=4e-5,
        force_theory_au_yr=7.5e-3,
        strength_ratio=0.005,
        rank=1,
        accepted=False,
        reason='',
        multiplet_r2=multiplet_r2,
        multiplet_coherence=multiplet_coherence,
        sigma_multiplet_r2=sigma_multiplet_r2,
    )


def test_accept_with_the_multiplet_law_compares_the_long_windows():
    """101703's numbers (Codex W15): the owner's constant-coefficient R^2 0.855 on the two long windows against sigma's
    best constant law there, 0.085 (its own 28-member multiplet; the single law gives 0.013): gain 0.77 >= 0.2."""
    assert owner.phase_law(_candidate(), S) == owner.LAW_MULTIPLET
    assert owner.accept(_candidate(), 0.708, S) == (True, 'accepted:r2')
    # the short-window R^2 of sigma does not decide: sigma fitting the long windows as well does (0.855 < 0.7 + 0.2)
    assert owner.accept(_candidate(), 0.0, S) == (True, 'accepted:r2')
    assert owner.accept(_candidate(sigma_multiplet_r2=0.7), 0.0, S) == (False, owner.REASON_NOT_BETTER)
    # no phase route: sigma's drifting single phase does not help a candidate without a single phase
    assert owner.accept(_candidate(sigma_multiplet_r2=0.7), 0.0, S, sigma_coherence=0.0) == (False, owner.REASON_NOT_BETTER)
    # a single phase that holds keeps both routes (short windows, as before W15)
    assert owner.phase_law(_candidate(coherence=0.9), S) == owner.LAW_SINGLE
    assert owner.accept(_candidate(coherence=0.9), 0.95, S, sigma_coherence=0.0) == (True, 'accepted:phase')
    # the multiplet must fit with constant coefficients and keep them in phase; 'single' switches it off
    assert owner.accept(_candidate(multiplet_r2=0.6), 0.0, S) == (False, owner.REASON_COHERENCE)
    assert owner.accept(_candidate(multiplet_coherence=0.79), 0.0, S) == (False, owner.REASON_COHERENCE)
    assert owner.accept(_candidate(), 0.0, replace(S, mmr_owner_law='single')) == (False, owner.REASON_COHERENCE)


def test_catalogue_tolerance_reaches_both_sides_of_the_multiplet_threshold():
    """A catalogue planet's rate error turns the coefficients too: within the tolerance of 0.8 either way the candidate
    is left open (catalogue_ambiguous), beyond it the multiplet law decides."""
    regime = owner.RegimeFit(owner.REGIME_FORCED_POSITIVE, (0.0, 1.0), 0.99, (0.99, 0.99), 0.4, 1e-5, 0.5, 0.0, 1e-5)
    ambiguous = lambda c: owner.catalogue_ambiguous(c, regime, 0.05, S)  # noqa: E731
    assert ambiguous(_candidate(multiplet_coherence=0.79))  # below the threshold, within 0.05
    assert ambiguous(_candidate(multiplet_coherence=0.83))  # above it, within 0.05
    assert not ambiguous(_candidate(multiplet_coherence=0.9))
    assert not ambiguous(_candidate(multiplet_coherence=0.7))
    assert ambiguous(_candidate(coherence=float('nan'), multiplet_coherence=0.79))  # no single phase at all
    assert not owner.catalogue_ambiguous(_candidate(multiplet_coherence=0.79), regime, 0.0, S)  # saved planets


def test_resolved_windows_by_hand():
    """2001 PG14: 4E-7+0+3 turns at 0.499 rad/yr; a 50-yr fit step moves it 24.95 rad per sample (> pi, aliased), the
    data's 5-yr step 2.495 rad (resolved). Codex W15: base 0.0627 rad/yr moves 3.135 < pi per 50 yr, but its member
    j = 2 with theta_dot = 1e-4 moves 3.145 > pi: a fit with that member is not resolved."""
    windows = [(0.0, 900.0), (1000.0, 1900.0)]
    coarse, fine = np.arange(0.0, 2000.0, 50.0), np.arange(0.0, 2000.0, 5.0)
    assert owner.resolved_windows(coarse, [0.499 * coarse], windows).tolist() == [False, False]
    assert owner.resolved_windows(fine, [0.499 * fine], windows).tolist() == [True, True]
    assert owner.resolved_windows(coarse, [0.05 * coarse], windows).tolist() == [True, True]
    # a fast stretch in the second window only
    phase = np.where(coarse < 1000.0, 0.05 * coarse, 50.0 + 0.499 * (coarse - 1000.0))
    assert owner.resolved_windows(coarse, [phase], windows).tolist() == [True, False]
    base, theta = 0.0627 * coarse, 1e-4 * coarse
    assert owner.resolved_windows(coarse, [base], windows).tolist() == [True, True]
    assert owner.resolved_windows(coarse, [base + j * theta for j in (0, 1, 2)], windows).tolist() == [False, False]


def test_libration_turns_by_hand():
    """101703's slow sigma (deg): 378 at the window start, turns at 991 and 352, 5796 at the end: swings 613, 639 and
    5444 deg, every one beyond a full turn (a reversal of circulation, not a libration). 129772-like: one turn with
    150 deg to the start and 170 deg to the end (half a slow libration)."""
    rad = np.radians
    assert not owner.libration_turns(rad(378.0), rad(np.array([991.0, 352.0])), rad(5796.0)).any()
    assert owner.libration_turns(rad(0.0), rad(np.array([150.0])), rad(-20.0)).all()
    # one side within a turn is enough: 991 -> 352 -> 500 keeps the second turn (148 deg to the end, a lower bound)
    assert owner.libration_turns(rad(378.0), rad(np.array([991.0, 352.0])), rad(500.0)).tolist() == [False, True]


def test_slow_oscillation_skips_a_reversal_of_circulation():
    """sigma's slow part runs up 4 turns, back 6 turns, up again (a crossing of the resonance twice): its two turning
    points have more than 2 pi on both sides and are not counted; a slow sine of 1 rad keeps both of its turns."""
    t = np.arange(0.0, 30000.0, 10.0)
    region = np.ones_like(t, dtype=bool)
    band = 0.05 * np.sin(TWO_PI * t / 500.0)
    knots = np.array([0.0, 10000.0, 20000.0, 30000.0]), np.array([0.0, 8 * math.pi, -4 * math.pi, 8 * math.pi])
    reversal = np.interp(t, *knots)
    assert owner.slow_oscillation(t, reversal, band, region, 2.0)[0] == 0
    assert owner.slow_oscillation(t, np.sin(TWO_PI * t / 20000.0), band, region, 2.0)[0] == 3


# --- the forced lines of the lattice ---------------------------------------------------------------------


def test_lattice_line_of_41J_14_is_the_3_1_at_14_jupiter_periods():
    n_j = table_mean_motion('Jupiter')
    spec = ResonanceSpec.from_key('41J-14+0-27')
    lines = forced_lines.lattice_lines(spec, {'Jupiter': n_j}, 0.0, S.matrix_2body_order_max, S.matrix_3body_order_max)
    period_14 = 14 * TWO_PI / n_j  # 166.1 yr
    hit = forced_lines.coincident_line(lines, 166.2, S.mmr_force_window_cycles)
    assert hit is not None
    line, offset = hit
    assert line.label == '3J-1' and line.order == 2 and line.period_yr == pytest.approx(period_14, rel=1e-9)
    assert offset == pytest.approx(abs(1 / period_14 - 1 / 166.2) * S.mmr_force_window_cycles * 166.2)
    # every other line is at most half as long (|41 m + 14 l| >= 2): nothing at 300 yr
    assert max(x.period_yr for x in lines) == pytest.approx(period_14)
    assert forced_lines.coincident_line(lines, 300.0, S.mmr_force_window_cycles) is None


def test_lattice_of_3J_2_has_no_line_at_a_hilda_libration_period():
    n_j = table_mean_motion('Jupiter')
    lines = forced_lines.lattice_lines(ResonanceSpec.from_key('3J-2+0-1'), {'Jupiter': n_j}, 0.0, 10, 6)
    assert max(x.period_yr for x in lines) == pytest.approx(2 * TWO_PI / n_j)  # 2 P_J = 23.7 yr
    assert forced_lines.coincident_line(lines, 257.0, S.mmr_force_window_cycles) is None
    retrograde = ResonanceSpec.from_key('1J-3r0')
    assert forced_lines.lattice_lines(retrograde, {'Jupiter': n_j}, 0.0, 10, 6) == ()


# --- 101703 41J-14 (benchmark record) ---------------------------------------------------------------------

EXAMPLE = 'mmr2__101703__41J-14+0-27'
# the record's cutoff of the W15 study, 20 x 0.6 yr: there the 166-yr forced wave of 3J-1 makes an episode whose force
# law the owner and the contamination rule must read. At the physical cutoff (75 yr) the angle shows no episode at all
# (`test_101703_at_the_physical_cutoff`), so the mechanism is tested at the study's cutoff
STUDY = replace(S, mmr_filter_cutoff_yr=12.0)


def _row(settings):
    analysis = analyse_mmr(io.load_series(EXAMPLE), settings)
    owners = [c for c in analysis.candidate_rows() if c.get('kind') == 'owner']
    return {**analysis.numbers_row(), **analysis.summary_row()}, owners


@pytest.fixture(scope='module')
def record():
    if not (io.example_path(EXAMPLE) / 'series.csv.gz').exists():
        pytest.skip('benchmark data not available')
    return EXAMPLE


def test_101703_at_the_physical_cutoff(record):
    # no episode of its own any more: a circulation whose origin is the 3:1 multiplet (the author's non_resonant/induced)
    row, _ = _row(S)
    assert (row['status'], row['angle_status'], row['libration_origin'], row['owner']) == (
        'non_resonant',
        'circulation',
        'induced',
        '3J-1+0-2',
    )
    assert row['flt_cutoff_yr'] == 75.0 and row['geo_n_librating_episodes'] == 0


def test_101703_is_induced_by_the_3_1_multiplet(record):
    row, owners = _row(STUDY)
    assert (row['status'], row['reason'], row['libration_origin']) == ('non_resonant', 'induced_libration', 'induced')
    assert row['owner'] == '3J-1+0-2' and row['rec_rule'] == '2'
    top = owners[0]
    assert top['candidate'] == '3J-1+0-2' and top['accepted'] and top['reason'] == 'accepted:r2'
    assert top['law'] == 'multiplet' and top['phi0_coherence'] < STUDY.mmr_owner_phi0_coh_min
    assert top['multiplet_R2'] >= STUDY.mmr_owner_r2_min and top['multiplet_coherence'] >= STUDY.mmr_owner_phi0_coh_min
    # the 166.2-yr episode sits on the same line (W12: 164.4 yr against P_e 166.2 yr) with a failed identity: since
    # W20 CS7 (F6) it is a forced wave, no episode of the angle (before, its force law was contaminated: leaf 5f)
    assert row['geo_n_forced_wave_episodes'] == 1 and row['geo_forced_wave_line'] == '3J-1'
    assert row['geo_forced_wave_period_yr'] == pytest.approx(166.2, abs=0.5)


def test_101703_without_the_multiplet_is_not_dynamic(record):
    # the single phase alone (multiplet law and potential route off). Before W20 CS7 the contamination rule kept it
    # uncertain (5f); since then the episode on the line 3J-1 is a forced wave (F6), and the remaining episodes need
    # an implausible strength (5s, non_resonant, the author's status). Never dynamic either way.
    single = replace(STUDY, mmr_owner_law='single', mmr_owner_potential='off')
    row, _ = _row(single)
    assert (row['status'], row['reason'], row['rec_rule']) == ('non_resonant', 'implausible_strength', '5s')
    assert row['geo_forced_wave_line'] == '3J-1'
    # neither safeguard (no lattice line: no contamination and no forced wave): the thin force-law pass makes the
    # forced wave a dynamic episode (the W13 verdict)
    row, _ = _row(replace(single, mmr_force_contamination='off'))
    assert (row['status'], row['libration_origin'], row['rec_rule']) == ('transient', 'dynamic', '5d/6')


def test_101703_single_phase_owner_is_accepted_by_its_averaged_potential(record):
    # W20 F3: the averaged disturbing function at fixed phi_o holds every member of the 3:1 multiplet, so the potential
    # route accepts the owner the single phase cannot fix
    row, owners = _row(replace(STUDY, mmr_owner_law='single'))
    assert (row['status'], row['reason'], row['owner'], row['rec_rule']) == ('non_resonant', 'induced_libration', '3J-1+0-2', '2')
    top = owners[0]
    assert top['candidate'] == '3J-1+0-2' and top['reason'] == 'accepted:potential' and top['route'] == 'potential'
    assert top['pot_R2'] >= STUDY.mmr_owner_r2_min
    assert 1 / STUDY.mmr_owner_potential_scale_max <= top['pot_scale_lo'] <= top['pot_scale_hi'] <= STUDY.mmr_owner_potential_scale_max


# --- leaf 5f (stubs as in test_impl_w4_integrate_decide) -----------------------------------------------------


def _contaminated_response(raw):
    from resonances.resonance.analysis.verdict import CheckValue

    return make_response(
        a_response=CheckValue.NOT_MEASURED,
        a_amplitude=CheckValue.PASS,
        slow_libration=CheckValue.PASS,
        b1_ratio=0.5,
        slow_adot_r2=0.1,
        force_r2_h1=0.63,
        force_r2_h2=0.62,
        centre_rate_rad_yr=1e-6,
        centre_rate_err_rad_yr=1e-5,
        exact_commensurability=False,
        force_law='single',
        multiplet=None,
        phi0_max_dev_rad=0.1,
        phi0_tol_rad=0.5,
        force_contaminated=True,
        force_line='3J-1',
        force_line_offset=0.0383,  # 101703: |1/164.40 - 1/166.20| x 3.5 x 166.20
        force_raw_check=str(raw),
    )


@pytest.mark.parametrize('with_a', ['pass', 'fail', 'not_measured'])
def test_a_contaminated_force_law_never_makes_the_libration_dynamic(with_a):
    """Whatever R0 with a says (it reads the same a_dot), the leaf is 5f: uncertain/origin_unknown; a near-separatrix
    angle too (no measured objection either)."""
    from resonances.resonance.analysis.ledger import Ledger
    from resonances.resonance.analysis.mmr import decide
    from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue

    kinematics = make_kinematics(check=CheckValue.PASS, corr_rate=0.1, detail='')
    for angle in (
        decide.AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2'),
        decide.AngleDecision(AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_A, 'G3'),
    ):
        energy = make_energy(with_slow=CheckValue(with_a), sigma_only=CheckValue.PASS, ve_vf_agree_fraction=0.5)
        ledger = Ledger.from_settings(S)
        response = _contaminated_response('pass')
        decision = decide.decide_status(angle, make_sampling(), None, energy, response, kinematics, None, None, None, S, ledger)
        assert (decision.status, decision.reason, decision.origin, decision.rule) == ('uncertain', 'origin_unknown', 'unknown', '5f')
        assert 'force_law_contaminated' in decision.flags


def test_a_failed_sigma_only_potential_stays_an_objection_under_contamination():
    """R0 sigma-only (period law, fixed well) holds no a_dot: its failure keeps leaf 5e (near_separatrix: non_resonant)."""
    from resonances.resonance.analysis.ledger import Ledger
    from resonances.resonance.analysis.mmr import decide
    from resonances.resonance.analysis.verdict import AngleStatus, AngleSubtype, CheckValue

    kinematics = make_kinematics(check=CheckValue.PASS, corr_rate=0.1, detail='')
    energy = make_energy(with_slow=CheckValue.FAIL, sigma_only=CheckValue.FAIL, ve_vf_agree_fraction=0.5)
    angle = decide.AngleDecision(AngleStatus.NEAR_SEPARATRIX, AngleSubtype.PATTERN_A, 'G3')
    response = _contaminated_response('fail')
    decision = decide.decide_status(
        angle, make_sampling(), None, energy, response, kinematics, None, None, None, S, Ledger.from_settings(S)
    )
    assert (decision.status, decision.reason, decision.rule) == ('non_resonant', 'no_resonant_potential', '5e')
