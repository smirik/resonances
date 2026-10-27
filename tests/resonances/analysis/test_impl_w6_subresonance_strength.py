"""W6 subresonances: test E' (the d'Alembert factor against the strongest term), the enumeration of every split up to
degree MATRIX_*_ORDER_MAX, the nodal rule and the ranked list `sub_top` (author's remarks C3, C14 of 2026-10-16).

Independent truths: the skeptic's numbers of impl/w6/verify-c-ug22-subres (10E-17+7+0 of 2003 UG22 at (e_E/e)^7 =
9.8e-10; 3J-1S-1+2-2-1 of 24820 at e_J^2 e_S^2 = 6.8e-6), a brute-force maximum over itertools vectors, the secular
frequencies g5, g6 of const.py for the member's centre rate, and the orbits' own elements read from the records."""

import itertools

import numpy as np
import pytest

from resonances.data import const
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import io, subresonances as sr
from resonances.resonance.analysis.mmr.models import ResonanceSpec, SubresonanceCandidate, SubresonanceTrigger
from resonances.resonance.analysis.settings import settings_from_config

S = settings_from_config()
NAN = float('nan')
ARCSEC_KYR = np.pi / 180 / 3600 * 1000  # arcsec/yr -> rad/kyr


# --- the factor of a term and the strongest one ---------------------------------------------------------


def test_member_factors_reproduce_the_skeptic_numbers():
    # 2003 UG22 10E-17+0+7 (q = 7): the e_Earth^7 member 10E-17+7+0 against the strongest e^7, e = 0.323, e_E = 0.0167
    factors = [0.323, 0.0167, 0.0, 0.0]  # varpi, varpi_E, Omega, Omega_E
    member = sr.slow_coefficients(ResonanceSpec.from_key('10E-17+7+0'))
    strongest = sr.strongest_factor(7, factors[:2], factors[2:])
    assert strongest == pytest.approx(0.323**7)
    assert sr.term_factor(member, factors) / strongest == pytest.approx(9.8e-10, rel=0.02)
    # 24820 3J-1S-1+0+0-1 (q = 1): 3J-1S-1+2-2-1 is e_J^2 e_S^2 e against e (e = 0.127 > e_J, e_S)
    factors = [0.127, 0.0484, 0.0539, 0.0, 0.0, 0.0]  # varpi, varpi_J, varpi_S, Omega, Omega_J, Omega_S
    member = sr.slow_coefficients(ResonanceSpec.from_key('3J-1S-1+2-2-1'))
    assert sr.term_factor(member, factors) / sr.strongest_factor(1, factors[:3], factors[3:]) == pytest.approx(6.8e-6, rel=0.02)


@pytest.mark.parametrize(
    'q, x_e, x_n, expected',
    [
        (2, 0.05, 0.26, 0.26**2),  # an inclined, nearly circular orbit: sin^2(i/2) beats e^2
        (3, 0.05, 0.26, 0.05 * 0.26**2),  # q odd: one eccentricity power is left
        (3, 0.20, 0.05, 0.20**3),
        (0, 0.10, 0.10, 1.0),  # zero order: the term itself
    ],
)
def test_strongest_factor_by_hand(q, x_e, x_n, expected):
    assert sr.strongest_factor(q, [x_e], [x_n]) == pytest.approx(expected)


@pytest.mark.parametrize('q', [-3, -1, 0, 2, 5])
def test_strongest_factor_is_the_maximum_over_every_term(q):
    # brute force over (u, u_P1, u_P2, v, v_P1, v_P2) with sum q, an even node sum, degree <= |q| + 4
    rng = np.random.default_rng(7 + q)
    factors = list(rng.uniform(0.01, 0.4, 6))
    best = 0.0
    for c in itertools.product(range(-abs(q) - 4, abs(q) + 5), repeat=6):
        if sum(c) == q and sum(c[3:]) % 2 == 0 and sum(map(abs, c)) <= abs(q) + 4:
            best = max(best, sr.term_factor(c, factors))
    assert sr.strongest_factor(q, factors[:3], factors[3:]) == pytest.approx(best)


# --- test E' in step 2 ----------------------------------------------------------------------------------


def _member(key, centre_range_deg, strength_rel):
    return sr.SubresonanceMember(
        spec=ResonanceSpec.from_key(key),
        j=0,
        nodal=False,
        centre_range_rad=np.radians(centre_range_deg),
        centre_rate_rad_yr=0.0,
        centre_max_pass_rad=0.1,
        centre_mean_rad=np.pi,
        phi0_coherence=0.95,
        coefficient_rel=NAN,
        composite_rms_rad=NAN,
        period_corr=NAN,
        period_ratio=NAN,
        rank=1,
        accepted=False,
        reason='',
        strength_rel=strength_rel,
    )


def test_e_prime_rejects_the_weak_member_without_the_theory():
    # the weak member has the flatter centre (it would win test C) but a factor of 1e-10: E' takes it out first
    weak, strong = _member('10E-17+7+0', 20.0, 9.8e-10), _member('10E-17+1+6', 30.0, 0.32)
    ledger = Ledger.from_settings(S)
    outcome = sr.evaluate([weak, strong], 0.1, 0.32, np.radians(5.0), S, theory=None, ledger=ledger)
    assert outcome.winner is strong and outcome.reasons[weak.spec.key] == sr.REASON_E_STRENGTH
    assert any(entry.setting == 'mmr_subresonance_strength_rel_min' for entry in ledger)
    alone = sr.evaluate([weak], 0.1, 0.32, np.radians(5.0), S, theory=None)
    assert alone.winner is None and alone.reasons[weak.spec.key] == sr.REASON_E_STRENGTH
    # switched off (threshold 0) the weak one wins on its centre range
    off = settings_from_config(mmr_subresonance_strength_rel_min=0.0)
    assert sr.evaluate([weak, strong], 0.1, 0.32, np.radians(5.0), off, theory=None).winner is weak
    # a bare SubresonanceCandidate carries no strength: E' is not measured and passes
    bare = SubresonanceCandidate(**{f: getattr(weak, f) for f in SubresonanceCandidate.__dataclass_fields__})
    assert sr.evaluate([bare], 0.1, 0.32, np.radians(5.0), S, theory=None).winner is bare


# --- real records ---------------------------------------------------------------------------------------


def _example(example_id):
    if not io.example_path(example_id).exists():
        pytest.skip('benchmark data not available')
    return io.load_series(example_id)


def _scan(example_id, **overrides):
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = _example(example_id)
    settings = settings_from_config(**overrides) if overrides else S
    analysis = analyse_mmr(series, settings, scan_subresonances=False, scan_owner=False)
    triggered = (SubresonanceTrigger.T1,)
    result = sr.run_subresonances(series, analysis.prep, analysis.geometry, analysis.response, analysis.context, triggered, settings)
    return series, analysis, result


def test_ug22_rejects_the_earth_eccentricity_member():
    # C3: 10E-17+7+0 was accepted because test E is skipped without the theory (the orbits cross); E' always runs
    series, _, result = _scan('mmr2__2003_UG22__10E-17+0+7')
    member = next(c for c in result.candidates if c.spec.key == '10E-17+7+0')
    expected = (float(np.mean(series.planets['Earth'].e)) / float(np.median(series.e))) ** 7
    assert member.strength_rel == pytest.approx(expected, rel=0.02) and member.strength_rel < 1e-9
    assert member.reason == sr.REASON_E_STRENGTH and not member.accepted
    assert result.accepted is None or result.accepted.strength_rel >= S.mmr_subresonance_strength_rel_min
    # i = 22 deg from the invariable plane: the nodal members are scanned
    assert result.nodal_scanned and any(c.nodal for c in result.candidates)
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    analysis = analyse_mmr(series, S)
    assert analysis.derived == [] and str(analysis.verdict.reason) != 'subresonance_librates'


def test_low_inclination_keeps_the_accepted_planar_member():
    # 12218 4M-7 (i = 5.5 deg from the invariable plane): no nodal members, 4M-7+1+2 still accepted (factor 0.99)
    _, _, result = _scan('mmr2__12218__4M-7+0+3')
    assert not result.nodal_scanned and not any(c.nodal for c in result.candidates)
    assert result.accepted is not None and result.accepted.spec.key == '4M-7+1+2'
    # the members go up to MATRIX_2BODY_ORDER_MAX = 10, i.e. degree 9 for q = 3 (parity)
    assert result.accepted.strength_rel > 0.9 and max(sr.member_degree(c.spec) for c in result.candidates) == 9


def test_the_ranked_list_reaches_the_catalogue_order():
    # C14: 3J-1S-1+2-2-1 (degree 5 = q + 4, mixed signs) of 24820 3J-1S-1+0+0-1 is enumerated and ranked: the members
    # go up to MATRIX_3BODY_ORDER_MAX = 6, i.e. degree 5 for q = 1 (parity)
    series, _, result = _scan('mmr3__24820__3J-1S-1+0+0-1', mmr_subresonance_top_n=1000)
    assert result.n_listed == len(result.listed) and max(m.degree for m in result.listed) == 5
    listed = {m.key: m for m in result.listed}
    member = listed['3J-1S-1+2-2-1']
    e, e_j, e_s = float(np.median(series.e)), *(float(np.mean(series.planets[p].e)) for p in ('Jupiter', 'Saturn'))
    assert member.strength_rel == pytest.approx(e_j**2 * e_s**2 * e / max(e, e_j, e_s), rel=0.02)
    # its centre runs at the c0 rate + 2 (g5 - g6): the catalogue angle librates, so about 2 (g5 - g6) itself
    g5, g6 = const.PLANETARY_FREQUENCIES['g5'], const.PLANETARY_FREQUENCIES['g6']
    assert member.rate_rad_yr * 1000 == pytest.approx(2 * (g5 - g6) * ARCSEC_KYR, rel=0.05)
    # the decision members carry the same centre rate in the list as in step 1 (the covariance algebra of
    # `centre_statistics`); those test E' rejects are not measured in step 1 at all
    measured = [c for c in result.candidates if c.reason != sr.REASON_E_STRENGTH]
    assert measured and all(np.isnan(c.centre_rate_rad_yr) for c in result.candidates if c not in measured)
    for c in measured:
        assert listed[c.spec.key].rate_rad_yr == pytest.approx(c.centre_rate_rad_yr, rel=1e-9, abs=1e-15)
        # W of the list is step 1's centre range
        assert listed[c.spec.key].width_rad == pytest.approx(c.centre_range_rad, rel=1e-9)
    # the order of the list is `table_order` (W20 S2); the default keeps the best MMR_SUBRESONANCE_TOP_N in sub_top
    weak = sr.too_weak([m.strength_rel for m in result.listed], S)
    order = sr.table_order([m.explained for m in result.listed], [m.width_rad for m in result.listed], weak)
    assert order == list(range(len(result.listed)))
    _, _, default = _scan('mmr3__24820__3J-1S-1+0+0-1')
    assert [m.key for m in default.listed] == [m.key for m in result.listed[: S.mmr_subresonance_top_n]]
    assert default.to_columns()['sub_top'].count(';') == S.mmr_subresonance_top_n - 1


def test_coorbitals_have_no_members():
    # the 1:1 is no expansion in e (the averaged potential of coorbital.py): no multiplet whatever the degree
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('1E-1+0+0'), 99, nodal=True) == []
    assert sr.enumerate_subresonances(ResonanceSpec.from_key('1E-1+0+0'), 1) == []
