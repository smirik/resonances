"""Independent checks of the W6 subresonance package (test E', every d'Alembert split up to |q| + extra, the ranked
list). Written from API.md (W6 subres), the public docstrings and docs/libration-analysis.md only.

Expected values are derived here, not read from the module:

- the members of a multiplet: every integer vector of slow coefficients counted by brute force (signed sum q, node
  coefficients summing to an even number, degree sum |c| of the parity of q);
- the d'Alembert factor prod x^|c| of test E': its exponents checked against the harmonic coefficients of the direct
  disturbing function 1/|r - r'| of a planar 3:1 problem, computed by my own Fourier quadrature (own Kepler solver);
- the strongest term: brute force over every degree-|q| one-sign term with an even node count;
- the invariable plane: the angle between the orbit normal and the invariable pole (Souami & Souchay 2012);
- member angles: sigma_member built from synthetic longitudes by hand.

All fast: no integration.
"""

import dataclasses
import itertools
import math

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import subresonances as sr
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec, SubresonanceCandidate
from resonances.resonance.analysis.settings import AnalysisSettings

TWO_PI = 2 * np.pi


# --------------------------------------------------------------------------------------------------------------
# helpers written for the tests
# --------------------------------------------------------------------------------------------------------------


def _coeffs(spec: ResonanceSpec) -> tuple:
    """(p, p_1.., v, v_1..) read from the spec fields directly (not via the module)."""
    nodes = tuple(spec.planet_node_coefficients) or (0,) * len(spec.planets)
    return (spec.body_varpi_coefficient, *spec.planet_varpi_coefficients, spec.body_node_coefficient, *nodes)


def _brute_members(spec: ResonanceSpec, degree_max: int, nodal: bool) -> set:
    """Every slow-coefficient vector (p, p_1.., v, v_1..) with sum q, even node sum, degree <= degree_max."""
    n_planets = len(spec.planets)
    q = -(sum(spec.planet_coefficients) + spec.body_coefficient)
    n_varpi = 1 + n_planets
    n_node = 1 + n_planets
    out = set()
    rng = range(-degree_max, degree_max + 1)
    for varpi in itertools.product(rng, repeat=n_varpi):
        if sum(abs(c) for c in varpi) > degree_max:
            continue
        node_iter = itertools.product(rng, repeat=n_node) if nodal else [(0,) * n_node]
        for node in node_iter:
            vec = varpi + tuple(node)
            if sum(abs(c) for c in vec) > degree_max:
                continue
            if sum(vec) != q or sum(node) % 2 != 0:
                continue
            out.add(vec)
    out.discard(_coeffs(spec))
    return out


def _kepler_E(M, e):
    E = M + e * np.sin(M)
    for _ in range(30):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    return E


def _planar_position(a, e, M, varpi):
    E = _kepler_E(M, e)
    x = a * (np.cos(E) - e)
    y = a * np.sqrt(1 - e * e) * np.sin(E)
    c, s = np.cos(varpi), np.sin(varpi)
    return c * x - s * y, s * x + c * y


def _harmonic_coefficients(e, e_p, n_grid=128, n_delta=16):
    """|A_m| of exp(i(3 lambda' - lambda + m varpi + m' varpi')), m + m' = -2, in the direct part 1/|r - r'| of a planar
    3:1 problem (a = (1/3)^(2/3), a' = 1). varpi' = 0, varpi = Delta: H(Delta) = e^{i Delta} <R e^{i M - 3 i M'}>_{M, M'}
    = sum_m A_m e^{i m Delta}, and A_m = <H e^{-i m Delta}>_Delta."""
    a = (1.0 / 3.0) ** (2.0 / 3.0)
    M = TWO_PI * np.arange(n_grid) / n_grid
    MM, MP = np.meshgrid(M, M, indexing='ij')
    deltas = TWO_PI * np.arange(n_delta) / n_delta
    xp, yp = _planar_position(1.0, e_p, MP, 0.0)
    H = []
    for d in deltas:
        x, y = _planar_position(a, e, MM, d)
        R = 1.0 / np.hypot(x - xp, y - yp)
        H.append(np.exp(1j * d) * np.mean(R * np.exp(1j * MM - 3j * MP)))
    H = np.array(H)
    return {m: abs(np.mean(H * np.exp(-1j * m * deltas))) for m in range(-4, 3)}


def _member(key, centre_range, strength_rel, phi0=0.95, centre_mean=np.pi, coefficient_rel=1.0, reason=''):
    spec = ResonanceSpec.from_key(key)
    return sr.SubresonanceMember(
        spec=spec,
        j=sum(spec.planet_varpi_coefficients),
        nodal=False,
        centre_range_rad=centre_range,
        centre_rate_rad_yr=0.0,
        centre_max_pass_rad=0.1,
        centre_mean_rad=centre_mean,
        phi0_coherence=phi0,
        coefficient_rel=coefficient_rel,
        composite_rms_rad=float('nan'),
        period_corr=float('nan'),
        period_ratio=float('nan'),
        rank=0,
        accepted=False,
        reason=reason,
        degree=sum(abs(c) for c in _coeffs(spec)),
        strength_rel=strength_rel,
        kept=True,
    )


# --------------------------------------------------------------------------------------------------------------
# degree, order, notation
# --------------------------------------------------------------------------------------------------------------


def test_degree_is_not_order_for_mixed_sign_member():
    # 3J-1S-1+2-2-1: planets' varpi +2, -2, body's -1 -> degree 5; order |3 - 1 - 1| = 1; signed q = -(3 - 1 - 1) = -1
    spec = ResonanceSpec.from_key('3J-1S-1+2-2-1')
    assert sr.member_degree(spec) == 5
    assert spec.order == 1
    assert sr.slow_sum(spec) == -1
    assert sr.member_degree(ResonanceSpec.from_key('3J-1S-1+0+0-1')) == 1
    # 13M-24+0+11: q = 11, degree 11
    assert sr.slow_sum(ResonanceSpec.from_key('13M-24+0+11')) == 11
    assert sr.member_degree(ResonanceSpec.from_key('13M-24+0+11')) == 11


def test_member_key_matches_the_package_notation():
    base = ResonanceSpec.from_key('3J-1S-1+0+0-1')
    # order (p, p_J, p_S, v, v_J, v_S)
    assert sr.member_key(base, (-1, 2, -2, 0, 0, 0)) == '3J-1S-1+2-2-1'
    assert sr.member_spec(base, (-1, 2, -2, 0, 0, 0)).key == '3J-1S-1+2-2-1'
    assert _coeffs(sr.member_spec(base, (-1, 2, -2, 0, 0, 0))) == (-1, 2, -2, 0, 0, 0)
    # the nodal member of 12218 quoted in the docstring: 4M-7+2+1[N-1,NM+1] (p = 1, p_M = 2, v = -1, v_M = +1)
    base2 = ResonanceSpec.from_key('4M-7+1+2')
    assert sr.member_key(base2, (1, 2, -1, 1)) == '4M-7+2+1[N-1,NM+1]'


def test_coorbital_flag():
    """The 1:1 angles have no members: the one 1:1 predicate (`preprocess.law_planet`, W23 S6a). An angle with
    |k_P| = |k| but k_P = +k, sigma = lambda_J + lambda - 2 varpi of 624 Hektor (rate ~ n_J + n, not slow), is not
    1:1: q = -2, so it has the members of degree 2 (and 4)."""
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('1E-1+0+0'), 4) == []
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('1J-1r0'), 4) == []
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('3J-1'), 4)
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('4J-2S-1'), 4)
    # degree 2, q = -2, keys (varpi_J, varpi) coefficients: (-2, 0) and (-1, -1); the catalogue angle (0, -2) excluded
    hektor = sr.enumerate_subresonances(ResonanceSpec.from_key('1J+1+0-2'), 2)
    assert sorted(m[0].key for m in hektor) == ['1J+1-1-1', '1J+1-2+0']


# --------------------------------------------------------------------------------------------------------------
# enumeration (C14): every d'Alembert split, counted by brute force
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'key, extra, nodal',
    [
        ('3J-1S-1+0+0-1', 2, False),
        ('3J-1S-1+0+0-1', 4, False),
        ('4M-7+1+2', 2, False),
        ('4M-7+1+2', 2, True),
        ('3J-1', 2, True),
        ('2J-1', 4, False),
    ],
)
def test_enumeration_equals_brute_force(key, extra, nodal):
    spec = ResonanceSpec.from_key(key)
    q = abs(sr.slow_sum(spec))
    expected = _brute_members(spec, q + extra, nodal)
    got = sr.enumerate_subresonances(spec, q + extra, nodal=nodal)
    vectors = [_coeffs(m) for m, _, _ in got]
    assert len(vectors) == len(set(vectors)), 'duplicate members'
    assert set(vectors) == expected
    # j and nodal flags, written from the docstrings
    for m, j, is_nod in got:
        assert is_nod == (m.body_node_coefficient != 0 or any(c != 0 for c in m.planet_node_coefficients))
        assert j == sum(m.planet_varpi_coefficients)
        assert m.planet_coefficients == spec.planet_coefficients and m.body_coefficient == spec.body_coefficient
    # sorted by degree, then planar first
    keys = [(sr.member_degree(m), is_nod) for m, _, is_nod in got]
    assert keys == sorted(keys)
    # member_vectors agrees
    assert set(sr.member_vectors(spec, q + extra, nodal)) == expected


def test_hand_counts_of_three_body_members():
    # 3J-1S-1+0+0-1 planar, q = -1, (p, p_J, p_S) with sum -1:
    # degree 1: a single -1 in one of three places -> 3, minus the catalogue -> 2
    # degree 3: |c| pattern (2, 1, 0) as (-2, +1, 0): 3 x 2 placements = 6; pattern (1, 1, 1) as two -1 and one +1: 3
    #           ((3, 0, 0) cannot sum to -1) -> 9
    spec = ResonanceSpec.from_key('3J-1S-1+0+0-1')
    got = sr.enumerate_subresonances(spec, 3)
    by_degree = {}
    for m, _, _ in got:
        by_degree[sr.member_degree(m)] = by_degree.get(sr.member_degree(m), 0) + 1
    assert by_degree == {1: 2, 3: 9}
    # the author's 3J-1S-1+2-2-1 (degree 5) appears from degree 5 (MATRIX_3BODY_ORDER_MAX 6 in the package)
    assert '3J-1S-1+2-2-1' not in [m.key for m, _, _ in got]
    assert '3J-1S-1+2-2-1' in [m.key for m, _, _ in sr.enumerate_subresonances(spec, 5)]
    assert '3J-1S-1+2-2-1' in [m.key for m, _, _ in sr.enumerate_subresonances(spec, 6)]


def test_a_degree_of_the_other_parity_adds_nothing_and_degree_q_members_always_exist():
    spec = ResonanceSpec.from_key('4M-7+1+2')  # q = 3
    k2 = {m.key for m, _, _ in sr.enumerate_subresonances(spec, 5)}
    k3 = {m.key for m, _, _ in sr.enumerate_subresonances(spec, 6)}
    assert k2 == k3
    # 13M-24+0+11 with the largest degree 10 (< q = 11): the degree-11 members still exist, nothing above
    spec = ResonanceSpec.from_key('13M-24+0+11')
    got = sr.enumerate_subresonances(spec, 10)
    # (p, p_M) >= 0 with sum 11: 12 vectors, minus the catalogue -> 11
    assert len(got) == 11
    assert all(sr.member_degree(m) == 11 for m, _, _ in got)
    assert '13M-24+2+9' in {m.key for m, _, _ in got}


def test_no_members_for_coorbital_and_retrograde():
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('1E-1+0+0'), 100, nodal=True) == []
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('1J-3r0'), 100, nodal=True) == []


# --------------------------------------------------------------------------------------------------------------
# test E' (C3): the d'Alembert factor
# --------------------------------------------------------------------------------------------------------------


def test_term_factor_by_hand():
    assert sr.term_factor((2, 1), (0.1, 0.05)) == pytest.approx(0.1**2 * 0.05)
    assert sr.term_factor((-1, 2, -2), (0.2, 0.048, 0.054)) == pytest.approx(0.2 * 0.048**2 * 0.054**2)
    # a zero coefficient contributes 1 whatever its x (nan included)
    assert sr.term_factor((3, 0), (0.1, float('nan'))) == pytest.approx(1e-3)
    # 2003 UG22 10E-17+7+0: e_E^7 against the strongest e^7 -> (e_E / e)^7 = (0.0167 / 0.3)^7 ~ 1.7e-9 < 1e-3
    member = sr.term_factor((0, 7), (0.3, 0.0167))
    strongest = sr.strongest_factor(7, [0.3, 0.0167], [math.sin(math.radians(3) / 2), 0.0])
    assert member / strongest == pytest.approx((0.0167 / 0.3) ** 7, rel=1e-9)
    assert member / strongest < AnalysisSettings().mmr_subresonance_strength_rel_min


def test_term_factor_exponents_follow_the_disturbing_function():
    """The harmonic coefficients of 1/|r - r'| in a planar 3:1 scale as e^|m| e'^|m'| (also the mixed-sign m = +1,
    m' = -3 of degree 4): the ratio of my quadrature's coefficients at doubled e (or e') equals the ratio of
    term_factor."""
    e, ep = 0.004, 0.004
    base = _harmonic_coefficients(e, ep)
    de = _harmonic_coefficients(2 * e, ep)
    dp = _harmonic_coefficients(e, 2 * ep)
    for m in (-2, -1, 0, 1):
        mp = -2 - m
        coeffs = (m, mp)  # (p, p_P) in the package's order
        pred_e = sr.term_factor(coeffs, (2 * e, ep)) / sr.term_factor(coeffs, (e, ep))
        pred_p = sr.term_factor(coeffs, (e, 2 * ep)) / sr.term_factor(coeffs, (e, ep))
        assert de[m] / base[m] == pytest.approx(pred_e, rel=0.02), m
        assert dp[m] / base[m] == pytest.approx(pred_p, rel=0.02), m


@pytest.mark.parametrize(
    'q, ecc, nodes',
    [
        (3, [0.15, 0.05], [0.04, 0.01]),
        (4, [0.05, 0.048, 0.054], [0.2, 0.01, 0.02]),  # inclination dominates: m = 4 (even)
        (5, [0.05, 0.048], [0.2, 0.01]),  # inclination dominates, odd q: m = 4 at most
        (2, [0.1], []),
    ],
)
def test_strongest_factor_brute_force(q, ecc, nodes):
    xs = list(ecc) + list(nodes)
    best = 0.0
    for vec in itertools.product(range(q + 1), repeat=len(xs)):
        if sum(vec) != q or sum(vec[len(ecc) :]) % 2 != 0:
            continue
        best = max(best, float(np.prod([x**c for x, c in zip(xs, vec)])))
    assert sr.strongest_factor(q, ecc, nodes) == pytest.approx(best, rel=1e-12)


def test_strongest_factor_without_eccentricity_is_nan():
    assert math.isnan(sr.strongest_factor(3, [float('nan')], [0.1]))
    assert sr.strongest_factor(3, [0.1, float('nan')], [0.01]) == pytest.approx(0.1**3)


def test_nodal_threshold_at_ten_degrees():
    s = AnalysisSettings()
    assert not sr.nodal_applies(math.sin(math.radians(5.5) / 2), s)  # 12218
    assert not sr.nodal_applies(math.sin(math.radians(9.9) / 2), s)
    assert sr.nodal_applies(math.sin(math.radians(10.1) / 2), s)
    assert sr.nodal_applies(math.sin(math.radians(25) / 2), s)


def test_e_prime_rejects_a_weak_member_that_would_win_on_range():
    """Two members kept by step 1: the weak one (factor 1e-9 of the strongest) has the smaller centre range and would
    win C; with E' (1e-3) the strong one is accepted and the weak one carries E_strength. With the threshold 0 the weak
    one wins: E' is what made the difference."""
    weak = _member('10E-17+7+0', centre_range=0.1, strength_rel=1e-9)
    strong = _member('10E-17+1+6', centre_range=0.5, strength_rel=0.4)
    settings = AnalysisSettings()
    out = sr.evaluate([weak, strong], 0.3, 0.1, 0.05, settings)
    assert out.winner is not None and out.winner.spec.key == '10E-17+1+6'
    assert out.reasons.get('10E-17+7+0') == sr.REASON_E_STRENGTH
    accepted, _, undetermined = sr.decide([weak, strong], 0.3, 0.1, 0.05, settings)
    assert accepted is not None and accepted.spec.key == '10E-17+1+6' and not undetermined

    loose = dataclasses.replace(settings, mmr_subresonance_strength_rel_min=0.0)
    out0 = sr.evaluate([weak, strong], 0.3, 0.1, 0.05, loose)
    assert out0.winner is not None and out0.winner.spec.key == '10E-17+7+0'


def test_e_prime_alone_rejects_and_bare_candidate_passes():
    weak = _member('10E-17+7+0', centre_range=0.1, strength_rel=1e-9)
    accepted, _, _ = sr.decide([weak], 0.3, 0.1, 0.05, AnalysisSettings())
    assert accepted is None
    # a bare SubresonanceCandidate (no strength_rel) passes E'
    fields = {f.name for f in dataclasses.fields(SubresonanceCandidate)}
    bare = SubresonanceCandidate(**{k: v for k, v in dataclasses.asdict(weak).items() if k in fields and k != 'spec'}, spec=weak.spec)
    accepted, _, _ = sr.decide([bare], 0.3, 0.1, 0.05, AnalysisSettings())
    assert accepted is not None and accepted.spec.key == '10E-17+7+0'


# --------------------------------------------------------------------------------------------------------------
# the ranked list (display only)
# --------------------------------------------------------------------------------------------------------------


def test_rate_residual():
    assert sr.rate_residual(-0.228e-3, 0.0233e-3) == pytest.approx(0.228 / 0.0233)
    assert math.isnan(sr.rate_residual(1.0, 0.0))
    assert math.isnan(sr.rate_residual(float('nan'), 1.0))
    assert math.isnan(sr.rate_residual(1.0, float('nan')))


def test_rank_order_by_explained_share():
    # W20 S2: (name, E, W, factor): the members test E' lets through (factor >= 1e-3) by E desc, ties by W; then the
    # weak ones by E; written out by hand
    items = [
        ('a', 0.50, 1.0, 1e-2),
        ('b', 0.90, 0.6, 1e-1),
        ('c', 0.90, 0.4, 1.0),
        ('d', -1.5, 3.0, 1e-2),
        ('e', 0.99, 0.1, 1e-5),
    ]
    weak = sr.too_weak([x[3] for x in items], AnalysisSettings())
    order = [items[i][0] for i in sr.table_order([x[1] for x in items], [x[2] for x in items], weak)]
    assert order == ['c', 'b', 'a', 'd', 'e']


def test_listed_member_text():
    m = sr.ListedMember(
        key='3J-1S-1+2-2-1',
        degree=5,
        nodal=False,
        rate_rad_yr=-0.228e-3,
        strength_rel=6.0e-6,
        width_rad=1.2,
        width_ratio=2.4,
        explained=-4.7,
        slow_cycles=3.6,
        swing_ratio=0.05,
    )
    parts = m.text().split(':')
    assert parts[0] == '3J-1S-1+2-2-1'
    assert [float(x) for x in parts[1:]] == pytest.approx([2.4, -4.7, 3.6, 0.05], rel=0.05)


# --------------------------------------------------------------------------------------------------------------
# geometry: member angles, the invariable plane, step 1
# --------------------------------------------------------------------------------------------------------------


def _synthetic_series(key, n=2001, span=20000.0):
    t = np.linspace(0.0, span, n)
    spec = ResonanceSpec.from_key(key)
    lam = 0.3 + 1.7 * t  # arbitrary rates, rad/yr
    lam_p = 1.1 + 0.53 * t
    varpi = 0.4 + 2.0e-4 * t
    varpi_p = 2.0 + 7.0e-5 * t
    node = 5.0 - 1.5e-4 * t
    node_p = 1.0 - 3.0e-5 * t
    sigma0 = spec.planet_coefficients[0] * lam_p + spec.body_coefficient * lam + spec.body_varpi_coefficient * varpi
    sigma0 = sigma0 + spec.planet_varpi_coefficients[0] * varpi_p
    planet = PlanetSeries(
        name=spec.planets[0],
        t_yr=t,
        a=np.full(n, 1.52),
        e=np.full(n, 0.09),
        inc=np.full(n, np.radians(1.85)),
        Omega=np.mod(node_p, TWO_PI),
        varpi=np.mod(varpi_p, TWO_PI),
        mean_longitude=np.mod(lam_p, TWO_PI),
    )
    series = MMRSeries(
        t_yr=t,
        a=np.full(n, 2.0),
        e=np.full(n, 0.15),
        inc=np.full(n, np.radians(3.0)),
        Omega=np.mod(node, TWO_PI),
        omega=np.mod(varpi - node, TWO_PI),
        varpi=np.mod(varpi, TWO_PI),
        angle_wrapped=np.mod(sigma0, TWO_PI),
        resonance=spec,
        regime=Regime(step_yr=t[1] - t[0], span_yr=span, n_rows=n, planets_saved=(spec.planets[0],)),
        planets={spec.planets[0]: planet},
        mean_longitude=np.mod(lam, TWO_PI),
    )
    return series, dict(lam=lam, lam_p=lam_p, varpi=varpi, varpi_p=varpi_p)


def _circ_close(a, b, tol=1e-9):
    return np.max(np.abs(np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b)))))) < tol


def test_planar_member_angle_equals_hand_built_angle():
    series, x = _synthetic_series('4M-7+1+2')
    spec = series.resonance
    member = ResonanceSpec.from_key('4M-7+3+0')  # p_M = 3, p = 0
    expected_offset = (0 - 2) * x['varpi'] + (3 - 1) * x['varpi_p']
    assert _circ_close(sr.slow_offset(spec, member, series), expected_offset)
    sigma_m = 4 * x['lam_p'] - 7 * x['lam'] + 3 * x['varpi_p']
    ms = sr.member_series(series, member)
    assert _circ_close(ms.angle_wrapped, sigma_m)
    assert np.all((ms.angle_wrapped >= 0) & (ms.angle_wrapped < TWO_PI + 1e-12))
    # a member of another commensurability is refused
    with pytest.raises(ValueError):
        sr.slow_offset(spec, ResonanceSpec.from_key('3M-5+1+1'), series)


def test_offset_coefficients_by_hand():
    spec = ResonanceSpec.from_key('3J-1S-1+0+0-1')
    member = ResonanceSpec.from_key('3J-1S-1+2-2-1')
    d = sr.offset_coefficients(spec, member)
    assert sorted(d.values()) == [-2, 2]
    assert len(d) == 2


# invariable plane of the Solar system relative to the J2000 ecliptic (Souami & Souchay 2012, A&A 543, A133)
INV_INC = math.radians(1.578694)
INV_NODE = math.radians(107.582222)


def _normal(inc, node):
    return np.array([np.sin(inc) * np.sin(node), -np.sin(inc) * np.cos(node), np.cos(inc)])


@pytest.mark.parametrize('inc_deg, node_deg', [(0.0, 0.0), (1.578694, 107.582222), (12.0, 40.0), (25.0, 250.0), (150.0, 10.0)])
def test_invariable_inclination_is_the_angle_between_normals(inc_deg, node_deg):
    inc, node = math.radians(inc_deg), math.radians(node_deg)
    expected = math.degrees(math.acos(np.clip(np.dot(_normal(inc, node), _normal(INV_INC, INV_NODE)), -1, 1)))
    inc2, _, _ = sr.to_invariable_plane(np.array([inc]), np.array([node]), np.array([0.7]))
    assert math.degrees(float(np.ravel(inc2)[0])) == pytest.approx(expected, abs=0.01)


def test_invariable_plane_keeps_the_perihelion_direction():
    """The perihelion unit vector, rotated into the invariable frame, has longitude varpi' measured from the new node
    line: check varpi' by the angle between perihelion vectors of two orbits (invariant under any rotation)."""
    incs = np.radians([10.0, 30.0])
    nodes = np.radians([40.0, 200.0])
    omegas = np.radians([70.0, 300.0])

    def peri(inc, node, om):
        cO, sO, ci, si, co, so = np.cos(node), np.sin(node), np.cos(inc), np.sin(inc), np.cos(om), np.sin(om)
        return np.array([cO * co - sO * so * ci, sO * co + cO * so * ci, so * si])

    angle_ecl = math.acos(np.dot(peri(incs[0], nodes[0], omegas[0]), peri(incs[1], nodes[1], omegas[1])))
    i2, O2, w2 = sr.to_invariable_plane(incs, nodes, omegas)
    om2 = np.asarray(w2) - np.asarray(O2)
    angle_inv = math.acos(np.dot(peri(i2[0], O2[0], om2[0]), peri(i2[1], O2[1], om2[1])))
    assert angle_inv == pytest.approx(angle_ecl, abs=1e-9)


def test_screen_standing_versus_drifting_track():
    s = AnalysisSettings()
    t = np.linspace(0.0, 20000.0, 2001)
    rng = np.random.default_rng(1)
    centre0 = 0.5 + 4.0e-4 * t  # drifts 8 rad over the window
    standing = np.pi + 0.05 * np.sin(TWO_PI * t / 3000.0) + 0.005 * rng.standard_normal(t.size)
    kept, rng_rad, rate, max_pass = sr.screen(standing, centre0, t, s)
    assert kept
    assert rng_rad < 0.5 * (centre0.max() - centre0.min())
    assert abs(rate) * 1000.0 < s.mmr_subresonance_rate_floor_rad_kyr
    assert max_pass < s.mmr_subresonance_pass_max_rad
    # 0.1 rad/kyr drift (five times the floor): range 2 rad (< 0.5 x 8), but the rate fails
    drifting = np.pi + 1.0e-4 * t
    kept, _, rate, _ = sr.screen(drifting, centre0, t, s)
    assert not kept
    assert rate == pytest.approx(1.0e-4, rel=0.05)
