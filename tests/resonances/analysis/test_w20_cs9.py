"""W20 CS9: the sigma +- 2 m omega members, one frame for the nodal members, and the ranked list (W, W/W0, E, N, A/W).

Independent truths:
- the member keys of sigma +- 2 omega and sigma +- 4 omega of 2003 UG22 10E-17+0+7, written by hand from omega =
  varpi - Omega (10E-17+0+9[N-2,NE+0] is of degree 9 + 2 = 11 > MATRIX_2BODY_ORDER_MAX);
- the invariable-plane elements from position and velocity: Kepler's equation, r and v rotated into a frame built
  from the invariable pole of const.py (its own longitude origin, the ascending node), the classical r, v -> elements
  formulas (Murray & Dermott 1999, sect. 2.8); the mean anomaly does not depend on the plane, and a d'Alembert angle
  does not depend on the longitude origin;
- E of a centre that follows -2 omega, from the variance of a uniform ramp (range^2 / 12) and of the noise;
- the G report of 2026-10-19 (scratchpad answers-1019/G-subresonances/members.py, its own running-mean centres and
  the mixed frame the package used then) and Codex's tie in the invariable plane for 2013 AS27.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from resonances.data import const
from resonances.resonance.analysis.mmr import io, subresonances as sr
from resonances.resonance.analysis.mmr.models import MMRSeries, PlanetSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
TWO_PI = 2 * np.pi
DEG = np.pi / 180


def _wrap(x):
    return np.angle(np.exp(1j * np.asarray(x, dtype=float)))


# --- S1: the omega members ------------------------------------------------------------------------------------------


def test_omega_members_beyond_the_degree_limit_only_with_the_nodes():
    spec = ResonanceSpec.from_key('10E-17+0+7')
    dmax = sr.degree_max(spec, S)
    assert dmax == 10
    plus2, minus2 = '10E-17+0+9[N-2,NE+0]', '10E-17+0+5[N+2,NE+0]'  # degree 11 and 7
    plus4, minus4 = '10E-17+0+11[N-4,NE+0]', '10E-17+0+3[N+4,NE+0]'  # degree 15 and 7
    # i = 15 deg from the invariable plane: the nodal members join, sigma +- 2 omega and +- 4 omega whatever the degree
    assert sr.nodal_applies(math.sin(15 * DEG / 2), S)
    members = sr.enumerate_subresonances(spec, dmax, True, S.mmr_subresonance_omega_harmonics)
    keys = [m.key for m, _, _ in members]
    assert {plus2, minus2, plus4, minus4} <= set(keys)
    assert len(keys) == len(set(keys))  # those within the limit are not listed twice
    degrees = {m.key: sr.member_degree(m) for m, _, _ in members}
    assert degrees[plus2] == 11 and degrees[plus4] == 15 and degrees[minus2] == degrees[minus4] == 7
    assert all(nodal for m, _, nodal in members if m.key in (plus2, minus2, plus4, minus4))
    # without the omega members only those of degree <= 10 are there
    base = {m.key for m, _, _ in sr.enumerate_subresonances(spec, dmax, True)}
    assert set(keys) - base == {plus2, plus4}
    # i = 5 deg: no nodal members at all, so none of them
    assert not sr.nodal_applies(math.sin(5 * DEG / 2), S)
    planar = {m.key for m, _, _ in sr.enumerate_subresonances(spec, dmax, False, S.mmr_subresonance_omega_harmonics)}
    assert not planar & {plus2, minus2, plus4, minus4}
    # every omega member keeps the d'Alembert rules: the slow sum q and an even node sum
    for vector in sr.omega_vectors(spec, 2):
        assert sum(vector) == sr.slow_sum(spec) and sum(vector[2:]) % 2 == 0


def test_omega_members_of_a_three_body_angle():
    # 4E+9S-5+0+0-8 (2013 AS27): omega is the body's, the planets' coefficients do not change
    spec = ResonanceSpec.from_key('4E+9S-5+0+0-8')
    keys = {sr.member_key(spec, v) for v in sr.omega_vectors(spec, 1)}
    assert keys == {'4E+9S-5+0+0-6[N-2,NE+0,NS+0]', '4E+9S-5+0+0-10[N+2,NE+0,NS+0]'}


# --- S1: one frame ---------------------------------------------------------------------------------------------------


def _rotation(inc, node, peri):
    cn, sn, ci, si, cw, sw = np.cos(node), np.sin(node), np.cos(inc), np.sin(inc), np.cos(peri), np.sin(peri)
    rz_node = np.array([[cn, -sn, 0], [sn, cn, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, ci, -si], [0, si, ci]])
    rz_peri = np.array([[cw, -sw, 0], [sw, cw, 0], [0, 0, 1]])
    return rz_node @ rx @ rz_peri


def _state(a, e, inc, node, peri, mean_anomaly):
    """Position and velocity (mu = 1) from the elements: Kepler's equation by Newton's method."""
    E = mean_anomaly
    for _ in range(50):
        E = E - (E - e * np.sin(E) - mean_anomaly) / (1 - e * np.cos(E))
    n = a**-1.5
    r = np.array([a * (np.cos(E) - e), a * np.sqrt(1 - e * e) * np.sin(E), 0.0])
    v = np.array([-np.sin(E), np.sqrt(1 - e * e) * np.cos(E), 0.0]) * n * a / (1 - e * np.cos(E))
    rot = _rotation(inc, node, peri)
    return rot @ r, rot @ v


def _elements(r, v):
    """(inc, node, varpi, lambda) from r, v (mu = 1), the classical formulas."""
    h = np.cross(r, v)
    inc = np.arccos(h[2] / np.linalg.norm(h))
    node_vec = np.cross([0, 0, 1], h)
    node = np.arctan2(node_vec[1], node_vec[0])
    e_vec = np.cross(v, h) - r / np.linalg.norm(r)
    e = np.linalg.norm(e_vec)
    peri = np.arccos(np.dot(node_vec, e_vec) / (np.linalg.norm(node_vec) * e))
    peri = peri if e_vec[2] >= 0 else TWO_PI - peri
    nu = np.arccos(np.clip(np.dot(e_vec, r) / (e * np.linalg.norm(r)), -1, 1))
    nu = nu if np.dot(r, v) >= 0 else TWO_PI - nu
    E = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    mean_anomaly = E - e * np.sin(E)
    return inc, node, node + peri, node + peri + mean_anomaly


def _invariable_basis():
    """A frame of the invariable plane: z along its pole, x along its ascending node on the ecliptic (its own origin)."""
    inc, node = const.INVARIABLE_PLANE['inc'] * DEG, const.INVARIABLE_PLANE['Omega'] * DEG
    z = np.array([np.sin(inc) * np.sin(node), -np.sin(inc) * np.cos(node), np.cos(inc)])
    x = np.array([np.cos(node), np.sin(node), 0.0])
    return np.vstack([x, np.cross(z, x), z])


def _record(n=60):
    """10E-17+0+7 with an inclined body (i 20-24 deg) and an Earth inclined 1-2 deg to the ecliptic; slow angles drift."""
    k = np.arange(n, dtype=float)
    t = 50.0 * k
    body = dict(a=1.97, e=0.32, inc=(20 + 4 * k / n) * DEG, node=1.0 + 0.011 * k, peri=2.0 - 0.017 * k, M=0.3 + 0.9 * k)
    earth = dict(a=1.0, e=0.0167, inc=(1 + k / n) * DEG, node=4.0 - 0.006 * k, peri=0.5 + 0.004 * k, M=1.1 + 2.3 * k)
    lam = np.mod(body['node'] + body['peri'] + body['M'], TWO_PI)
    lam_e = np.mod(earth['node'] + earth['peri'] + earth['M'], TWO_PI)
    varpi, varpi_e = np.mod(body['node'] + body['peri'], TWO_PI), np.mod(earth['node'] + earth['peri'], TWO_PI)
    sigma0 = 10 * lam_e - 17 * lam + 7 * varpi
    ones = np.ones(n)
    planet = PlanetSeries('Earth', t, earth['a'] * ones, earth['e'] * ones, earth['inc'], np.mod(earth['node'], TWO_PI), varpi_e, lam_e)
    series = MMRSeries(
        t_yr=t,
        a=body['a'] * ones,
        e=body['e'] * ones,
        inc=body['inc'],
        Omega=np.mod(body['node'], TWO_PI),
        omega=np.mod(body['peri'], TWO_PI),
        varpi=varpi,
        angle_wrapped=np.mod(sigma0, TWO_PI),
        resonance=ResonanceSpec.from_key('10E-17+0+7'),
        regime=Regime(step_yr=50.0, span_yr=t[-1], n_rows=n, planets_saved=('Earth',)),
        planets={'Earth': planet},
        mean_longitude=lam,
        mean_anomaly=np.mod(body['M'], TWO_PI),
        name='synthetic-10E-17',
    )
    return series, body, earth


def _invariable_member(body, earth, coefficients):
    """sum c x (lambda'_E, lambda', varpi', Omega', varpi'_E, Omega'_E) from r, v in the frame of `_invariable_basis`."""
    basis = _invariable_basis()
    out = []
    for i in range(len(body['inc'])):
        rb, vb = _state(body['a'], body['e'], body['inc'][i], body['node'][i], body['peri'][i], body['M'][i])
        re, ve = _state(earth['a'], earth['e'], earth['inc'][i], earth['node'][i], earth['peri'][i], earth['M'][i])
        _, node_b, varpi_b, lam_b = _elements(basis @ rb, basis @ vb)
        _, node_e, varpi_e, lam_e = _elements(basis @ re, basis @ ve)
        values = dict(lam_e=lam_e, lam=lam_b, varpi=varpi_b, node=node_b, varpi_e=varpi_e, node_e=node_e)
        out.append(sum(c * values[name] for name, c in coefficients.items()))
    return np.array(out)


def test_mean_anomaly_and_lambda_in_the_invariable_plane():
    # the rotated r, v give the same mean anomaly; lambda' - varpi' = lambda - varpi to 1e-12
    series, body, earth = _record(8)
    basis = _invariable_basis()
    inc_p, node_p, varpi_p = sr.to_invariable_plane(series.inc, series.Omega, series.omega)
    for i in range(len(series.t_yr)):
        r, v = _state(body['a'], body['e'], body['inc'][i], body['node'][i], body['peri'][i], body['M'][i])
        inc, node, varpi, lam = _elements(basis @ r, basis @ v)
        assert _wrap(lam - varpi - body['M'][i]) == pytest.approx(0.0, abs=1e-12)
        assert inc == pytest.approx(inc_p[i], abs=1e-12)
        # the package's frame counts longitudes from another origin: node and varpi differ by the same constant
        assert _wrap((node_p[i] - node) - (varpi_p[i] - varpi)) == pytest.approx(0.0, abs=1e-12)


def test_zero_tilt_is_the_identity(monkeypatch):
    monkeypatch.setitem(const.INVARIABLE_PLANE, 'inc', 0.0)
    inc, node, omega = np.array([0.3, 0.5]), np.array([1.0, -2.0]), np.array([2.5, 0.1])
    inc_p, node_p, varpi_p = sr.to_invariable_plane(inc, node, omega)
    assert np.allclose(inc_p, inc, atol=1e-12)
    assert np.allclose(_wrap(node_p - node), 0, atol=1e-12) and np.allclose(_wrap(varpi_p - node - omega), 0, atol=1e-12)


def test_a_nodal_member_is_built_in_one_frame():
    # sigma + 2 omega = 10 lambda'_E - 17 lambda' + 9 varpi' - 2 Omega', every element from the invariable plane
    series, body, earth = _record()
    member = sr.member_spec(series.resonance, (9, 0, -2, 0))  # (varpi, varpi_E, Omega, Omega_E)
    assert member.key == '10E-17+0+9[N-2,NE+0]'
    expected = _invariable_member(body, earth, dict(lam_e=10, lam=-17, varpi=9, node=-2))
    built = np.asarray(series.angle_wrapped) + sr.slow_offset(series.resonance, member, series)
    assert np.max(np.abs(_wrap(built - expected))) < 1e-9
    # sigma0 itself in the invariable plane: sigma0 + plane_shift
    sigma0 = _invariable_member(body, earth, dict(lam_e=10, lam=-17, varpi=7))
    assert np.max(np.abs(_wrap(np.asarray(series.angle_wrapped) + sr.plane_shift(series) - sigma0))) < 1e-9
    # the mixed frame (sigma0 from the ecliptic, Delta from the invariable plane) misses exactly that shift
    mixed = np.asarray(series.angle_wrapped) + sr._combine(
        sr.offset_coefficients(series.resonance, member), sr.slow_angles(series, True), len(series.t_yr)
    )
    miss = np.max(np.abs(_wrap(mixed - expected)))
    assert miss > 1e-3 and miss == pytest.approx(sr.plane_offset(series), rel=1e-9)
    # a planar member stays in the ecliptic, exactly the angle of the saved sigma0
    planar = ResonanceSpec.from_key('10E-17+1+6')
    lam_e, lam = series.planets['Earth'].mean_longitude, series.mean_longitude
    direct = 10 * lam_e - 17 * lam + 6 * series.varpi + series.planets['Earth'].varpi
    assert np.max(np.abs(_wrap(np.asarray(series.angle_wrapped) + sr.slow_offset(series.resonance, planar, series) - direct))) < 1e-9


def test_without_the_planet_the_shift_is_unknown():
    series, _, _ = _record(10)
    bare = replace(series, planets={})
    assert sr.plane_shift(bare) is None and math.isnan(sr.plane_offset(bare))


# --- S2: the ranked list -------------------------------------------------------------------------------------------


def test_a_centre_following_minus_two_omega_is_explained_by_sigma_plus_two_omega():
    # omega circulates in 45 kyr; c0 = c_N - 2 omega + noise over a 20-kyr window: sigma + 2 omega stands
    rng = np.random.default_rng(3)
    t = np.linspace(0.0, 20000.0, 4001)
    omega_dot = TWO_PI / 45000.0
    omega = 0.4 + omega_dot * t
    noise = 0.05 * rng.standard_normal(len(t))
    c0 = 3.0 - 2 * omega + noise
    other = 1.0 + 0.2 * TWO_PI / 30000.0 * t  # an unrelated slow angle
    angles = np.vstack([omega, other])
    offsets = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0], [0.0, 0.0]])  # +2 omega, -2 omega, other, sigma0
    rate, explained, slow_rate = sr.centre_statistics(t, c0, angles, offsets)
    # by hand: var(c0) = (2 omega_dot T)^2 / 12 + 0.05^2; var(c_+) = var(noise)
    ramp = (2 * omega_dot * 20000.0) ** 2 / 12
    assert explained[0] == pytest.approx(1 - np.var(noise) / (ramp + np.var(noise)), abs=2e-3)
    assert explained[0] >= 0.95 and explained[3] == pytest.approx(0.0, abs=1e-12)
    assert explained[1] < -2  # sigma - 2 omega doubles the drift: var x 4
    # every E and rate equals the direct computation on the member's own track
    for row, d in enumerate(offsets):
        track = c0 + d @ angles
        assert explained[row] == pytest.approx(1 - np.var(track) / np.var(c0), rel=1e-9, abs=1e-12)
        assert rate[row] == pytest.approx(np.polyfit(t, track, 1)[0], rel=1e-9, abs=1e-15)
    assert slow_rate[0] == pytest.approx(2 * omega_dot, rel=1e-9)
    # W / W0 and N of sigma + 2 omega: the noise's range against the ramp; 20 kyr x 2 omega_dot / 2 pi = 0.889 cycles
    width, width0 = np.ptp(c0 + 2 * omega), np.ptp(c0)
    assert width / width0 <= 0.2
    assert 20000.0 * abs(slow_rate[0]) / TWO_PI == pytest.approx(20000.0 * 2 / 45000.0, rel=1e-9)


def test_table_order_puts_the_weak_members_last_then_e_then_w():
    explained = [0.5, 0.98, 0.98, np.nan, 0.9, 0.99]
    width = [1.0, 0.3, 0.2, 0.1, 0.5, 0.05]
    weak = [False, False, False, False, False, True]
    # strong first: E 0.98 (W 0.2), 0.98 (W 0.3), 0.9, 0.5, nan; then the weak one despite its E 0.99
    assert sr.table_order(explained, width, weak) == [2, 1, 4, 0, 3, 5]
    assert sr.too_weak([1e-4, 2e-3, np.nan], S).tolist() == [True, False, False]


def test_listed_member_text_and_columns():
    m = sr.ListedMember(
        key='31E-36+0+7[N-2,NE+0]',
        degree=9,
        nodal=True,
        rate_rad_yr=-4.1e-5,
        strength_rel=1.7e-3,
        width_rad=np.radians(25.9),
        width_ratio=0.438,
        explained=0.855,
        slow_cycles=0.093,
        swing_ratio=1.86,
    )
    assert m.text() == '31E-36+0+7[N-2,NE+0]:0.44:0.85:0.093:1.9'
    scan = sr.SubresonanceScan(
        triggers=(),
        support=(),
        candidates=[],
        accepted=None,
        check='pass',
        composite=False,
        undetermined=False,
        tie_by_order=False,
        theory_measured=False,
        sigma0_phi0_coherence=np.nan,
        composite_rms_rad=np.nan,
        single_rms_rad=np.nan,
        binomial_mode=0,
        drift_pred_rad_yr=np.nan,
        drift_meas_rad_yr=np.nan,
        plane_offset_rad=np.nan,
        flags=[],
        listed=[m],
        catalogue_row=replace(m, key='31E-36+0+5', width_rad=np.radians(59.1), width_ratio=1.0, explained=0.0, swing_ratio=0.81),
        fast_swing_rad=np.radians(48.0),
        strength_caveat='e;i;crossing',
    )
    columns = scan.to_columns()
    assert columns['sub_rank1_key'] == m.key and columns['sub_rank1_W_deg'] == pytest.approx(25.9)
    assert columns['sub_rank1_E'] == pytest.approx(0.855) and columns['sub_rank1_N'] == pytest.approx(0.093)
    assert columns['sub_rank2_key'] == '' and columns['sub_rank3_W_deg'] is None
    assert columns['sub_cat_W_deg'] == pytest.approx(59.1) and columns['sub_fast_swing_deg'] == pytest.approx(48.0)
    assert columns['sub_strength_caveat'] == 'e;i;crossing' and columns['sub_top'] == m.text()


# --- the benchmark (G report of 2026-10-19) --------------------------------------------------------------------------


_CACHE = {}


def _analysis(example_id, mixed=False, monkeypatch=None):
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    if not io.example_path(example_id).exists():
        pytest.skip('benchmark data not available')
    if (example_id, mixed) not in _CACHE:
        if mixed:
            monkeypatch.setattr(sr, 'plane_shift', lambda series, angles=None: None)
        _CACHE[example_id, mixed] = analyse_mmr(io.load_series(example_id), S)
    return _CACHE[example_id, mixed]


def _listed(analysis):
    return {m.key: (rank, m) for rank, m in enumerate(analysis.subresonances.listed, start=1)}


def test_101703_three_body_member_numbers():
    # G report: 1V+3S-7+0-1+4 W 164 deg, E 0.84 over 9.3-50 kyr (running means); the package's W_c low-pass +-10 %
    analysis = _analysis('mmr3__101703__1V+3S-7+0+0+3')
    rank, member = _listed(analysis)['1V+3S-7+0-1+4']
    assert rank == 1  # -1+1+3 explains more (E 0.92) but its factor 2e-4 is below test E' (G report: 2e-4 too)
    assert np.degrees(member.width_rad) == pytest.approx(164, rel=0.10) and member.explained == pytest.approx(0.84, rel=0.10)
    assert member.slow_cycles == pytest.approx(0.97, rel=0.20)  # varpi - varpi_S: 0.97 of a turn over the window
    assert member.swing_ratio < 0.1  # a 14-deg wave against a 164-deg wander: a drift, not a libration
    assert member.strength_rel == pytest.approx(0.0066, rel=0.10)


def test_ug22_sigma_plus_two_omega_leads_the_list():
    # G report (40-kyr record, libration piece 0.16-20.5 kyr): sigma + 2 omega W 101 deg, E 0.96; the 20-kyr record's
    # scan window is that piece (0.17-19.9 kyr). Degree 11: listed only with MMR_SUBRESONANCE_OMEGA_HARMONICS
    analysis = _analysis('mmr2__2003_UG22__10E-17+0+7')
    sub = analysis.subresonances
    rank, member = _listed(analysis)['10E-17+0+9[N-2,NE+0]']
    assert rank == 1 and member.degree == 11 and member.nodal
    assert np.degrees(member.width_rad) == pytest.approx(101, rel=0.10) and member.explained == pytest.approx(0.96, rel=0.10)
    assert np.degrees(sub.catalogue_row.width_rad) == pytest.approx(250, rel=0.10)
    assert 'crossing' in sub.strength_caveat  # q = 0.90 au crosses Earth's orbit: the d'Alembert factor is no measure
    assert sub.to_columns()['sub_rank1_key'] == '10E-17+0+9[N-2,NE+0]'


def test_as27_frame_changes_the_order_but_not_the_verdict(monkeypatch):
    # mixed frame (the package before W20, and the G report): sigma + 2 omega W 13.2 deg, E 0.975 (+-15 % on W: running
    # means against the W_c low-pass); one frame: sigma + 2 omega and 31E-36+1+2[N+2] tie (Codex: S 0.470 and 0.471)
    one = _analysis('mmr2__2013_AS27__31E-36+0+5')
    mixed = _analysis('mmr2__2013_AS27__31E-36+0+5', mixed=True, monkeypatch=monkeypatch)
    rank, member = _listed(mixed)['31E-36+0+7[N-2,NE+0]']
    assert rank == 1 and np.degrees(member.width_rad) == pytest.approx(13.2, rel=0.15)
    assert member.explained == pytest.approx(0.975, rel=0.10)
    assert 'frame_mixed' in mixed.subresonances.flags and 'frame_mixed' not in one.subresonances.flags
    listed = _listed(one)
    plus2, rival = listed['31E-36+0+7[N-2,NE+0]'][1], listed['31E-36+1+2[N+2,NE+0]'][1]
    ranks = {listed['31E-36+0+7[N-2,NE+0]'][0], listed['31E-36+1+2[N+2,NE+0]'][0]}
    assert max(ranks) - min(ranks) == 1 and abs(plus2.explained - rival.explained) < 0.01
    # W20 round C (review B #11): AS27's factors are unreliable (e, i, crossing), so the list is by E alone: the members
    # test E' calls weak (E 0.98, 0.97, 0.91) now come before the tie instead of after every member it lets through
    assert one.subresonances.strength_caveat == 'e;i;crossing'
    order = [m.explained for m in one.subresonances.listed]
    assert order == sorted(order, reverse=True) and min(ranks) == 1 + sum(e > plus2.explained + 0.01 for e in order)
    # the hybrid put sigma + 2 omega up to the plane offset away from its own angle
    assert np.degrees(one.subresonances.plane_offset_rad) == pytest.approx(13.6, abs=0.5)
    # the verdict does not depend on the frame of a listed member
    assert one.verdict.status == mixed.verdict.status and one.verdict.angle_status == mixed.verdict.angle_status
    # the minimum of the averaged potential explains the centre better than any member (G report: RMS 1.6 against 14 deg)
    assert one.subresonances.potential_explained >= 0.95
    assert one.subresonances.potential_width_rad < 0.2 * one.subresonances.catalogue_row.width_rad
