"""The subresonance check (analysis/mmr/subresonances.py, SPEC §6 / SISTERS.md).

Independent truths: member counts of research/scripts/count_sisters.py (SISTERS §5) and a brute-force count here;
member angles computed directly from synthetic longitudes; the saved angles of 31935 9M-16+2+5 and 65366 6M-11+1+4,
which the integration computed itself (benchmark examples); the invariable-plane geometry by vectors; and the
prototype numbers of SISTERS §2 and codex B-review §4 (phi0 coherence 0.94/0.88 against 0.32/0.08 for 65366,
0.91/0.84 against 0.51/0.24 for 31935; period correlation 0.99/0.97, ratio 1.11/1.44)."""

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import io, subresonances as sr
from resonances.resonance.analysis.mmr.models import (
    MMRSeries,
    PlanetSeries,
    Regime,
    ResonanceSpec,
    SubresonanceCandidate,
    SubresonanceTrigger,
)
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, Flag, Reason

S = settings_from_config()
NAN = float('nan')


# --- members ----------------------------------------------------------------------------------------


def _brute_count(n_slow: int, q: int, budget: int, nodes: bool) -> int:
    """Vectors on (varpi.., Omega..) with sum q, an even node sum, sum |c| <= budget (nodes only when allowed)."""
    half = n_slow // 2
    total = 0
    for c in itertools.product(range(-budget, budget + 1), repeat=n_slow):
        node_part = c[half:]
        if sum(c) != q or sum(node_part) % 2 or sum(map(abs, c)) > budget:
            continue
        if any(node_part) and not nodes:
            continue
        total += 1
    return total


@pytest.mark.parametrize(
    'key, extra, expected',
    [
        ('6M-11+0+5', 0, 6 - 1),  # SISTERS §5: q = 5, two-body planar: 6 (the catalogue angle excluded)
        ('6M-11+0+5', 2, 84 - 1),  # q = 5 with nodes, q_max = q + 2: 84
        ('9M-16+0+7', 2, 150 - 1),  # q = 7, q + 2: 150
        ('3J-1S-1+0+0-1', 2, 48 - 1),  # three-body q = 1, q + 2: 48
        ('4J-2S-1+0+0-1', 0, 3 - 1),  # three-body q = 1 planar: 3
    ],
)
def test_member_counts_follow_count_sisters(key, extra, expected):
    spec = ResonanceSpec.from_key(key)
    members = sr.enumerate_subresonances(spec, spec.order + extra, nodal=extra > 0)
    assert len(members) == expected
    n_slow = 2 * (len(spec.planets) + 1)
    assert len(members) == _brute_count(n_slow, spec.order, spec.order + extra, extra > 0) - 1
    assert all(member.key != spec.key for member, _, _ in members)
    assert len({member.key for member, _, _ in members}) == len(members)


def test_members_of_the_31935_multiplet_and_their_notation():
    spec = ResonanceSpec.from_key('9M-16+0+7')
    members = sr.enumerate_subresonances(spec, spec.order)
    assert [m.key for m, _, _ in members] == [f'9M-16+{j}+{7 - j}' for j in range(1, 8)]
    for member, j, nodal in members:
        assert not nodal and j == member.planet_varpi_coefficients[0]
        parsed = ResonanceSpec.from_key(member.key)
        assert parsed.planet_varpi_coefficients == member.planet_varpi_coefficients
        assert parsed.body_varpi_coefficient == member.body_varpi_coefficient
    # the largest degree and the retrograde case
    capped = sr.enumerate_subresonances(spec, 7, nodal=True)  # degree 7 only: the planar and nodal ones of degree 7
    assert [m for m in capped if not m[2]] == members and all(sr.member_degree(m) == 7 for m, _, _ in capped)
    # a largest degree below q: the degree-q members always exist (82831 13M-24 is of order 11 against
    # MATRIX_2BODY_ORDER_MAX 10)
    assert sr.enumerate_subresonances(spec, 5) == members and sr.enumerate_subresonances(spec, 5, nodal=True) == capped
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('1J-3r0'), 99) == []


def test_nodal_members_have_an_even_node_sum_and_a_notation_of_their_own():
    members = sr.enumerate_subresonances(ResonanceSpec.from_key('6M-11+0+5'), 7, nodal=True)
    nodal = [m for m, _, n in members if n]
    assert nodal and all((m.body_node_coefficient + m.planet_node_coefficient(0)) % 2 == 0 for m in nodal)
    assert '6M-11+2+3[N-1,NM+1]' in {m.key for m in nodal}
    # sorted by degree, planar first
    orders = [sr.member_degree(m) for m, _, _ in members]
    assert orders == sorted(orders)


def _synthetic_series(key='9M-16+0+7', n=2001, step=0.5) -> MMRSeries:
    """Linear secular angles and fast mean longitudes; the catalogue angle computed from them directly."""
    spec = ResonanceSpec.from_key(key)
    t = np.arange(n) * step
    varpi = 0.3 + 1.6e-4 * t
    node = 1.0 - 1.0e-4 * t
    lam = 0.2 + 1.7641 * t
    planet = PlanetSeries(
        name='Mars',
        t_yr=t,
        a=np.full(n, 1.5237),
        e=np.full(n, 0.093),
        inc=np.full(n, 0.03),
        Omega=np.mod(0.8 - 0.9e-4 * t, 2 * np.pi),
        varpi=np.mod(2.0 + 0.9e-4 * t, 2 * np.pi),
        mean_longitude=np.mod(1.1 + 3.3407 * t, 2 * np.pi),
    )
    sigma = 9 * (1.1 + 3.3407 * t) - 16 * lam + 7 * varpi
    return MMRSeries(
        t_yr=t,
        a=np.full(n, 2.2359),
        e=np.full(n, 0.159),
        inc=np.full(n, 0.1),
        Omega=np.mod(node, 2 * np.pi),
        omega=np.mod(varpi - node, 2 * np.pi),
        varpi=np.mod(varpi, 2 * np.pi),
        angle_wrapped=np.mod(sigma, 2 * np.pi),
        resonance=spec,
        regime=Regime(step_yr=step, span_yr=t[-1], n_rows=n),
        planets={'Mars': planet},
        mean_longitude=np.mod(lam, 2 * np.pi),
        name='synthetic',
    )


def test_member_angle_is_built_from_the_saved_elements():
    series = _synthetic_series()
    member = ResonanceSpec.from_key('9M-16+2+5')
    t = series.t_yr
    expected = 9 * (1.1 + 3.3407 * t) - 16 * (0.2 + 1.7641 * t) + 2 * (2.0 + 0.9e-4 * t) + 5 * (0.3 + 1.6e-4 * t)
    built = sr.member_series(series, member)
    built.validate()
    assert built.resonance.key == '9M-16+2+5'
    np.testing.assert_allclose(np.angle(np.exp(1j * (built.angle_wrapped - expected))), 0.0, atol=1e-9)
    offset = sr.slow_offset(series.resonance, member, series)  # 2 (varpi_P - varpi), unwrapped
    np.testing.assert_allclose(np.diff(offset), 2 * (0.9e-4 - 1.6e-4) * np.diff(t), atol=1e-12)
    with pytest.raises(ValueError):
        sr.slow_offset(series.resonance, ResonanceSpec.from_key('7M-13+0+6'), series)


def test_without_the_planet_the_offset_uses_the_mode_longitude():
    series = _synthetic_series()
    bare = MMRSeries(**{**series.__dict__, 'planets': {}})
    offset = sr.slow_offset(bare.resonance, ResonanceSpec.from_key('9M-16+1+6'), bare)
    g4 = 17.8603 * math.pi / 180 / 3600  # const.PLANETARY_FREQUENCIES['g4'] [rad/yr]
    slope = np.polyfit(bare.t_yr, offset, 1)[0]
    assert slope == pytest.approx(g4 - 1.6e-4, rel=1e-6)


# --- invariable plane ---------------------------------------------------------------------------------


INC_I, NODE_I = math.radians(1.5787), math.radians(107.5822)


def test_invariable_plane_inclination_by_vectors():
    rng = np.random.default_rng(1)
    inc, node, omega = rng.uniform(0, 0.6, 50), rng.uniform(0, 2 * np.pi, 50), rng.uniform(0, 2 * np.pi, 50)
    new_inc, _, _ = sr.to_invariable_plane(inc, node, omega)
    normal = np.array([math.sin(INC_I) * math.sin(NODE_I), -math.sin(INC_I) * math.cos(NODE_I), math.cos(INC_I)])
    h = np.stack([np.sin(inc) * np.sin(node), -np.sin(inc) * np.cos(node), np.cos(inc)])
    np.testing.assert_allclose(np.cos(new_inc), normal @ h, atol=1e-12)
    # an ecliptic orbit is tilted by the plane's own inclination
    np.testing.assert_allclose(sr.to_invariable_plane(np.zeros(3), np.zeros(3), np.zeros(3))[0], INC_I, atol=1e-12)


def test_an_orbit_in_the_invariable_plane_keeps_the_longitude_of_the_node_line():
    # R = Rz(Omega_I) Rx(-I) Rz(-Omega_I) leaves the node line of the plane fixed: perihelion on it -> varpi' = Omega_I
    inc, _, varpi = sr.to_invariable_plane(np.array([INC_I, INC_I]), np.array([NODE_I, NODE_I]), np.array([0.0, np.pi / 2]))
    np.testing.assert_allclose(inc, 0.0, atol=1e-7)
    np.testing.assert_allclose(np.angle(np.exp(1j * (varpi - [NODE_I, NODE_I + np.pi / 2]))), 0.0, atol=1e-9)


# --- step 1 -------------------------------------------------------------------------------------------


def _grid():
    return np.arange(0.0, 50000.0, 5.0)


def test_screen_keeps_a_stationary_track_and_rejects_range_rate_and_pass():
    t = _grid()
    centre0 = 12.0 * t / t[-1]  # sigma0's centre moves by 12 rad
    wobble = 0.2 * np.sin(2 * np.pi * t / 10000)
    kept, span, rate, largest = sr.screen(1.0 + wobble, centre0, t, S)
    assert kept and span == pytest.approx(0.4, abs=0.01) and abs(rate) < 1e-5 and largest < np.pi
    # 0.6 of sigma0's range > mmr_subresonance_range_ratio (0.5)
    assert not sr.screen(0.6 * centre0, centre0, t, S)[0]
    # a drift of 0.1 rad/kyr: range 5 rad < 6, but faster than the floor 0.02 rad/kyr
    kept, _, rate, _ = sr.screen(1e-4 * t, centre0, t, S)
    assert not kept and rate == pytest.approx(1e-4, rel=1e-6)
    # a slow swing of +-1.7 rad: no drift, range 3.4 < 6, but one monotone pass of 3.4 rad > pi
    swing = 1.7 * np.sin(2 * np.pi * t / 50000 - np.pi / 2)
    kept, span, rate, largest = sr.screen(swing, centre0, t, S)
    assert not kept and largest == pytest.approx(3.4, abs=0.01)


def test_centre_track_of_a_linear_offset_is_exact():
    t = _grid()
    centre0 = np.sin(t / 3000)
    offset = 3.0 - 2e-4 * t
    np.testing.assert_allclose(sr.centre_track(centre0, offset, t, 8000.0, S.filter_order), centre0 + offset, atol=1e-9)


def test_regular_stretch_trims_the_slow_cycles_at_the_edges():
    # cycles of 5 kyr at the start (near the separatrix), then 1.8 kyr, then 6 kyr at the end (31935-like)
    slow_start = [0.0, 2500.0, 5000.0, 7500.0, 10000.0]
    regular = list(10000.0 + 900.0 * np.arange(1, 41))
    slow_end = [regular[-1] + 3000.0, regular[-1] + 6000.0]
    stretch = sr.regular_stretch(np.array(slow_start + regular + slow_end), 2.0)
    assert stretch == (7500.0, regular[-1])
    assert sr.regular_stretch(np.array([0.0, 1.0]), 2.0) is None


# --- step 2 -------------------------------------------------------------------------------------------


def _candidate(key, rng, coherence, centre_deg=180.0, reason='', rank=1, coef=NAN, nodal=False):
    spec = ResonanceSpec.from_key(key) if '[' not in key else ResonanceSpec.from_key(key.split('[')[0])
    if nodal:
        spec = sr.member_spec(spec, (3, 2, -1, 1))
    return SubresonanceCandidate(
        spec=spec,
        j=int(sum(spec.planet_varpi_coefficients)),
        nodal=nodal,
        centre_range_rad=rng,
        centre_rate_rad_yr=0.0,
        centre_max_pass_rad=rng,
        centre_mean_rad=math.radians(centre_deg),
        phi0_coherence=coherence,
        coefficient_rel=coef,
        composite_rms_rad=NAN,
        period_corr=NAN,
        period_ratio=NAN,
        rank=rank,
        accepted=False,
        reason=reason,
    )


def test_decide_force_law_coherence_and_gain():
    good = _candidate('9M-16+2+5', 0.9, 0.91)
    assert sr.decide([good], 0.51, 0.16, 0.1, S)[0] is good  # SISTERS 31935: 0.91 against 0.51
    assert sr.decide([_candidate('9M-16+2+5', 0.9, 0.75)], 0.3, 0.16, 0.1, S)[0] is None  # < 0.8
    assert sr.decide([_candidate('9M-16+2+5', 0.9, 0.85)], 0.7, 0.16, 0.1, S)[0] is None  # gain 0.15 < 0.2
    assert sr.decide([_candidate('9M-16+2+5', 0.9, 0.95, reason='range')], 0.3, 0.16, 0.1, S)[0] is None  # step 1 failed
    assert sr.decide([_candidate('9M-16+2+5', 0.9, NAN)], 0.3, 0.16, 0.1, S)[0] is None


def test_decide_symmetric_centre_and_where_it_does_not_apply():
    off = _candidate('9M-16+2+5', 0.9, 0.95, centre_deg=90.0)
    assert sr.decide([off], 0.3, 0.16, 0.1, S)[0] is None
    assert sr.decide([off], 0.3, 0.35, 0.1, S)[0] is off  # e > 0.3 (Gallardo 2018)
    assert sr.decide([off], 0.3, 0.16, math.radians(25), S)[0] is off  # i > 20 deg
    assert sr.decide([off], 0.3, 0.16, 0.1, S, symmetric_centre=False)[0] is off  # 1:n, co-orbital
    assert sr.decide([_candidate('9M-16+2+5', 0.9, 0.95, centre_deg=20.0)], 0.3, 0.16, 0.1, S)[0] is not None


def test_decide_tie_goes_to_the_lower_order_then_the_larger_coefficient():
    planar = _candidate('6M-11+1+4', 1.55, 0.92, centre_deg=0.0)
    twin = _candidate('6M-11+2+3', 1.43, 0.95, centre_deg=0.0, nodal=True)  # the node twin of SISTERS §1.4
    accepted, by_order, undetermined = sr.decide([twin, planar], 0.25, 0.14, 0.1, S)
    assert accepted is planar and by_order and not undetermined
    a = _candidate('9M-16+2+5', 1.0, 0.9, coef=1.0)
    b = _candidate('9M-16+1+6', 1.1, 0.9, coef=0.8)
    accepted, weak, _ = sr.decide([b, a], 0.3, 0.16, 0.1, S)
    assert accepted is a and not weak
    accepted, weak, _ = sr.decide([_candidate('9M-16+2+5', 1.0, 0.9), _candidate('9M-16+1+6', 1.05, 0.9)], 0.3, 0.16, 0.1, S)
    assert accepted is not None and weak  # no coefficients: the smallest range, but only medium


def test_decide_nodal_winner_beta_librating_and_test_e():
    nodal = _candidate('6M-11+2+3', 0.5, 0.95, centre_deg=0.0, nodal=True)
    assert sr.decide([nodal], 0.25, 0.14, 0.1, S) == (None, False, False)
    assert sr.decide([_candidate('9M-16+2+5', 0.9, 0.95)], 0.3, 0.16, 0.1, S, beta_librates=True) == (None, False, True)
    good = _candidate('9M-16+2+5', 0.9, 0.95)
    # 49030: the strongest predicted T0 is 45 kyr against an observed 0.94 kyr (SISTERS §2)
    slow = sr.TheoryCheck(composite_rms_rad=math.radians(88), t0_ratio=45 / 0.94)
    assert sr.decide([good], 0.3, 0.16, 0.1, S, theory=slow)[0] is None
    # D never rejects: a failed composite check leaves the member accepted and flags the disagreement
    poor = sr.TheoryCheck(composite_rms_rad=math.radians(60), single_rms_rad=math.radians(70), t0_ratio=0.6)
    outcome = sr.evaluate([good], 0.3, 0.16, 0.1, S, theory=poor, ledger=Ledger.from_settings(S))
    assert outcome.winner is good and outcome.theory_confirms is False and sr.FLAG_THEORY_DISAGREES in outcome.flags


# --- triggers -----------------------------------------------------------------------------------------


def _geometry(runs=(), drift=0.0, max_pass=0.5):
    return SimpleNamespace(runs=list(runs), drift_fraction=drift, centre=SimpleNamespace(max_pass_rad=max_pass))


def _run(n_tp, accepted, reject='', shape=''):
    return SimpleNamespace(n_tp=n_tp, accepted=accepted, reject_reason=reject, shape=shape)


def _response(r2, coherence):
    return SimpleNamespace(windows=[SimpleNamespace(r2=v) for v in r2], phi0_coherence=coherence)


def test_triggers():
    quiet = _response([0.1, 0.2], 0.9)
    assert sr.triggers(_geometry([_run(30, True, shape='stationary')]), quiet, '', 0, True, S) == ((), ())
    assert sr.triggers(_geometry([_run(59, False, 'centre_moves')]), quiet, '', 0, True, S)[0] == (SubresonanceTrigger.T1,)
    assert sr.triggers(_geometry([_run(34, True, shape='modulated')]), quiet, '', 0, True, S)[0] == (SubresonanceTrigger.T1,)
    assert sr.triggers(_geometry(drift=0.9), quiet, '', 0, True, S)[0] == (SubresonanceTrigger.T1,)
    # a saw on a slope: a TP-rich run and a centre turning by more than 2 pi
    assert sr.triggers(_geometry([_run(30, True, shape='stationary')], max_pass=7.0), quiet, '', 0, True, S)[0] == (SubresonanceTrigger.T1,)
    # 65366: a responds (R^2 0.87-0.95) with a rotating phase (coherence 0.28)
    assert sr.triggers(_geometry(), _response([0.87, 0.95], 0.28), '', 0, True, S)[0] == (SubresonanceTrigger.T2,)
    for reason in (Reason.KINEMATIC_DRIFT, Reason.KINEMATIC_LIBRATION, Reason.DRIFT_MECHANISM_UNKNOWN):
        assert sr.triggers(_geometry(), quiet, str(reason), 0, True, S)[0] == (SubresonanceTrigger.T5,)
    assert sr.triggers(_geometry(), quiet, '', 2, False, S)[0] == (SubresonanceTrigger.T6,)
    assert sr.triggers(_geometry(), quiet, '', 2, True, S)[0] == ()


# --- real records (benchmark data, no integration) ----------------------------------------------------


def _example(example_id):
    if not io.example_path(example_id).exists():
        pytest.skip('benchmark data not available')
    return io.load_series(example_id)


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'parent, member',
    [('mmr2__31935__9M-16+0+7', 'mmr2__31935__9M-16+2+5'), ('mmr2__65366__6M-11+0+5', 'mmr2__65366__6M-11+1+4')],
)
def test_member_series_equals_the_angle_the_integration_saved(parent, member):
    series, saved = _example(parent), _example(member)
    built = sr.member_series(series, saved.resonance)
    np.testing.assert_allclose(np.angle(np.exp(1j * (built.angle_wrapped - saved.angle_wrapped))), 0.0, atol=1e-5)


def _scan(example_id, triggered=(SubresonanceTrigger.T1,), **overrides):
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = _example(example_id)
    settings = settings_from_config(**overrides) if overrides else S
    analysis = analyse_mmr(series, settings, scan_subresonances=False, scan_owner=False)
    result = sr.run_subresonances(series, analysis.prep, analysis.geometry, analysis.response, analysis.context, triggered, settings)
    return series, analysis, result


@pytest.mark.benchmark
@pytest.mark.parametrize(
    'example_id, member, sigma0_max, member_min, ratio_range, leading',
    [
        # SISTERS §2 / codex B-review §4: 0.94 / 0.88 against 0.32 / 0.08; period ratio 1.11; j = 1
        ('mmr2__65366__6M-11+0+5', '6M-11+1+4', 0.4, 0.85, (0.9, 1.3), 1),
        # 0.91 / 0.84 against 0.51 / 0.24; period ratio 1.44; j = 2
        ('mmr2__31935__9M-16+0+7', '9M-16+2+5', 0.65, 0.85, (1.2, 1.7), 2),
    ],
)
def test_prototypes_accept_the_member_of_sisters(example_id, member, sigma0_max, member_min, ratio_range, leading):
    series, analysis, result = _scan(example_id)
    assert isinstance(result, sr.SubresonanceScan)
    assert result.accepted is not None and result.accepted.spec.key == member
    assert result.check == CheckValue.FAIL and result.accepted.accepted and result.accepted.reason == ''
    assert result.sigma0_phi0_coherence < sigma0_max and result.accepted.phi0_coherence > member_min
    # composite theory: zero-parameter centre track (RMS 10/8 deg in SISTERS), periods follow T0(t) F(A)
    assert result.theory.measured and result.theory.confirms and result.theory_measured
    assert math.degrees(result.composite_rms_rad) < 20 and abs(math.degrees(result.theory.composite_offset_rad)) < 10
    assert result.theory.period_corr > 0.9 and ratio_range[0] < result.theory.period_ratio < ratio_range[1]
    # composite regime: Rouche fails, P(beta) does not wind (SISTERS §1.3)
    assert result.composite and result.rouche is False and result.winding == 0 and str(Flag.SUBRESONANCE_COMPOSITE) in result.flags
    assert result.leading_j == leading  # at the mean e, e_P of the scan window
    # T6 input at the record's median e (0.126 for 65366, lower than in the episode: j = 2 leads there): never j = 0
    assert sr.leading_member(series, analysis.context, S)[0] in (leading, leading + 1)
    # the drift of sigma0's centre is the member's -Delta_dot = m (g - g_P) (handbook: within 1 % over a beta cycle)
    assert result.drift_pred_rad_yr == pytest.approx(result.drift_meas_rad_yr, rel=0.3)
    columns = result.to_columns()
    assert columns['sub_accepted'] == member and columns['sub_theory_confirms'] is True
    assert columns['sub_n_scanned'] == len(result.candidates) and columns['sub_period_corr'] == result.theory.period_corr


def _sin_half_inc_invariable(inc, Omega) -> float:
    """Median sin(i/2) from the invariable plane by the angle between the orbit normals (independent of the
    package's rotation)."""
    from resonances.data import const

    big_i, big_o = np.radians(const.INVARIABLE_PLANE['inc']), np.radians(const.INVARIABLE_PLANE['Omega'])
    normal = np.array([np.sin(big_i) * np.sin(big_o), -np.sin(big_i) * np.cos(big_o), np.cos(big_i)])
    inc, Omega = np.asarray(inc, float), np.asarray(Omega, float)
    h = np.stack([np.sin(inc) * np.sin(Omega), -np.sin(inc) * np.cos(Omega), np.cos(inc)])
    return float(np.median(np.sin(np.arccos(np.clip(normal @ h, -1, 1)) / 2)))


@pytest.mark.benchmark
def test_nodal_twin_ties_and_loses_by_degree_and_e_prime_rejects_it():
    # SISTERS §1.4: the node member 6M-11+2+3[N-1,NM+1] drifts at nearly the rate of 6M-11+1+4 (g - g_P ~ s_P - s).
    # 65366 has i = 4.6 deg, below MMR_SUBRESONANCE_NODAL_INC_MIN_DEG: the nodal members are forced in here.
    series, _, result = _scan('mmr2__65366__6M-11+0+5', mmr_subresonance_nodal_inc_min_deg=0.0, mmr_subresonance_strength_rel_min=0.0)
    assert result.nodal_scanned and result.accepted.spec.key == '6M-11+1+4' and result.tie_by_order
    twin = next(c for c in result.candidates if c.spec.key == '6M-11+2+3[N-1,NM+1]')
    assert twin.nodal and twin.reason == sr.REASON_C
    assert np.isfinite(result.plane_offset_rad) and result.plane_offset_rad < math.radians(5)
    # its d'Alembert factor e_M^2 e^3 sin(i/2) sin(i_M/2) against the strongest term e^5 (e = 0.126 > e_M = 0.097)
    mars = series.planets['Mars']
    e, e_m = float(np.median(series.e)), float(np.mean(mars.e))
    s, s_m = _sin_half_inc_invariable(series.inc, series.Omega), _sin_half_inc_invariable(mars.inc, mars.Omega)
    expected = e_m**2 * e**3 * s * s_m / max(e, e_m) ** 5
    assert twin.strength_rel == pytest.approx(expected, rel=0.05) and expected < 1e-3
    # test E' (default 1e-3) rejects it before step 1 (a weak nodal member is only counted). The other twin
    # 6M-11+0+5[N+1,NM-1] (e^5 sin(i/2) sin(i_M/2), factor 1.03e-3) just passes E', ties and loses by degree
    _, _, default = _scan('mmr2__65366__6M-11+0+5', mmr_subresonance_nodal_inc_min_deg=0.0)
    assert all(c.spec.key != '6M-11+2+3[N-1,NM+1]' for c in default.candidates) and default.n_weak > 0
    assert any("E' " in entry.name and not entry for entry in default.entries)
    other = next(c for c in default.candidates if c.spec.key == '6M-11+0+5[N+1,NM-1]')
    assert other.strength_rel == pytest.approx(e**5 * s * s_m / max(e, e_m) ** 5, rel=0.05) and other.reason == sr.REASON_C
    assert default.accepted.spec.key == '6M-11+1+4' and default.tie_by_order


@pytest.mark.benchmark
def test_without_the_planets_the_check_is_not_measured():
    series = _example('mmr2__65366__6M-11+0+5')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    bare = io.load_series('mmr2__65366__6M-11+0+5', with_planets=False)
    analysis = analyse_mmr(bare, S, scan_subresonances=False, scan_owner=False)
    result = sr.run_subresonances(bare, analysis.prep, analysis.geometry, analysis.response, analysis.context, (SubresonanceTrigger.T2,), S)
    assert result.check == CheckValue.NOT_MEASURED and result.accepted is None
    assert str(Flag.PLANETS_MISSING) in result.flags and not result.theory.measured
    assert series.planets  # the saved record has them
