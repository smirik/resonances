"""W22 R-3B (author 2026-10-24), part 2: the owner route 'identity' (`owner_identity`). Part 1, the swing cap of the
small-divisor envelope (R-DIV), is tested with the envelope in test_w18_branch_rules.py and
independent_w18/test_w18_branch_rule.py.

Independent numbers:
- the identity fit on a synthetic pair: sigma = r phi + Psi with Psi = 0.3 + 1e-4 t is exact (c / r - 1 = 0, residual
  0); adding 0.05 sin(2 pi t / 137.3) to sigma leaves a residual rms 0.05 / sqrt(2) = 0.0354, over a half-amplitude
  0.5: 0.0707; phi = 1.2 sin(2 pi t / 600) spans 2.4 rad;
- covers by hand: episodes 50–150 and 200–260 yr over the window 100–300 yr cover (50 + 60) / 200 = 0.55;
- 433 1R-5V+7 (step 5 yr and 0.5 yr): the R-3B impact run (an independent monkeypatch, SP/dis/impact/R-3B, identity
  log) found 47M-44+0-3 with c / r - 1 = -6.89e-4 / -5.15e-4, residual 0.0105 / 0.0093 of sigma's half-swing, phi's
  cover 0.9983 / 0.9990, phi's direct-averaging strength ratio 1.355 / 1.361, sigma's ratio 1.27e7; status
  non_resonant/episodic/medium, induced_libration, owner 47M-44+0-3 (the author's non_resonant, DECISIONS 2026-10-17);
- 1580 4M-7J-5: the identity passes for 4E-13+0+9 (c / r - 1 = -8e-3, residual 0.0098), but 4E-13 circulates: no
  owner from the route.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import owner_identity as oi
from resonances.resonance.analysis.mmr import strength as st
from resonances.resonance.analysis.mmr.decide import AngleDecision
from resonances.resonance.analysis.mmr.models import OwnerCandidate, OwnerResult, ResonanceSpec
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import (
    AngleStatus,
    AngleSubtype,
    CheckValue,
    Confidence,
    LibrationOrigin,
    OriginDetail,
    Reason,
    Status,
)

S = settings_from_config()
NAN = float('nan')
SIGMA = ResonanceSpec.from_key('1R-5V+7+0+0-3')
PHI = ResonanceSpec.from_key('47M-44+0-3')
EPISODIC = AngleDecision(AngleStatus.EPISODIC, AngleSubtype.NONE, 'G2', 0)


# --- step 1: the identity numbers -------------------------------------------------------------------------------


def _pair(extra_amplitude=0.0):
    t = np.arange(0.0, 2000.0, 1.0)
    phi = 1.2 * np.sin(2 * np.pi * t / 600.0)
    ratio = -7 / 44
    sigma = ratio * phi + 0.3 + 1e-4 * t + extra_amplitude * np.sin(2 * np.pi * t / 137.3)
    return t, sigma, phi, ratio


def test_identity_fit_of_an_exact_identity():
    t, sigma, phi, ratio = _pair()
    coef_dev, resid, span = oi.identity_fit(t, sigma, phi, ratio, 0.5)
    assert abs(coef_dev) < 1e-9 and resid < 1e-9
    assert span == pytest.approx(2.4, rel=1e-3)


def test_identity_fit_residual_is_sigmas_own_wave_over_its_half_amplitude():
    t, sigma, phi, ratio = _pair(extra_amplitude=0.05)
    coef_dev, resid, _ = oi.identity_fit(t, sigma, phi, ratio, 0.5)
    assert resid == pytest.approx(0.05 / np.sqrt(2) / 0.5, rel=0.02)  # 0.0707
    assert abs(coef_dev) < S.mmr_owner_identity_coef_tol


def _fit(coef=0.0, resid=0.01, span_deg=200.0, unwrap=True, key='47M-44+0-3'):
    return oi.IdentityFit(ResonanceSpec.from_key(key), coef, resid, np.radians(span_deg), unwrap)


@pytest.mark.parametrize(
    'fit, passes',
    [
        (_fit(coef=0.049), True),
        (_fit(coef=-0.049), True),
        (_fit(coef=0.051), False),
        (_fit(resid=0.1), True),  # at the limit (<=)
        (_fit(resid=0.11), False),
        (_fit(span_deg=359.0), True),
        (_fit(span_deg=361.0), False),  # phi runs a full turn: no libration about a fixed centre, not analysed
        (_fit(unwrap=False), False),
    ],
)
def test_passes_identity_limits(fit, passes):
    assert oi.passes_identity(fit, S) is passes


def test_shortlist_takes_the_smallest_spans_up_to_top_n():
    fits = [_fit(span_deg=300.0, key='7J-2+0-5'), _fit(span_deg=100.0, key='47M-44+0-3'), _fit(span_deg=200.0, key='25E-44+0+19')]
    fits.append(_fit(span_deg=50.0, coef=0.2, key='4E-13+0+9'))  # fails the coefficient
    listed = oi.shortlist(fits, settings_from_config(mmr_owner_identity_top_n=2))
    assert [f.spec.key for f in listed] == ['47M-44+0-3', '25E-44+0+19']


# --- steps 2 and 3: phi's own analysis --------------------------------------------------------------------------


def _phi_analysis(angle_status='episodic', spans=((50.0, 150.0), (200.0, 260.0)), method=st.METHOD_DIRECT, check=CheckValue.PASS):
    episodes = [SimpleNamespace(tp_times_yr=np.array([a, (a + b) / 2, b])) for a, b in spans]
    geometry = SimpleNamespace(window=(0.0, 1000.0), episodes=episodes)
    return SimpleNamespace(
        verdict=SimpleNamespace(angle_status=angle_status),
        geometry=geometry,
        strength=SimpleNamespace(method=method, check=check, ratio=1.36),
    )


def test_phi_cover_from_its_episodes():
    phi = oi.phi_ownership(_phi_analysis(), (100.0, 300.0))
    assert phi.cover == pytest.approx(0.55)
    assert phi.intervals == [(50.0, 150.0, 'libration'), (200.0, 260.0, 'libration')]
    assert oi.owns(phi, S)


def test_phi_libration_over_its_record_covers_the_window():
    phi = oi.phi_ownership(_phi_analysis(angle_status='libration', spans=()), (100.0, 300.0))
    assert phi.cover == 1.0 and phi.intervals == [(0.0, 1000.0, 'libration')]


@pytest.mark.parametrize(
    'analysis',
    [
        _phi_analysis(spans=((50.0, 150.0),)),  # cover 0.25 < mmr_owner_cover_min
        _phi_analysis(method=st.METHOD_SCALE),  # no direct averaging: the e^q scale is no measurement
        _phi_analysis(check=CheckValue.FAIL),
        _phi_analysis(angle_status='circulation', spans=()),
    ],
)
def test_phi_does_not_own_sigma(analysis):
    assert not oi.owns(oi.phi_ownership(analysis, (100.0, 300.0)), S)


# --- entry -----------------------------------------------------------------------------------------------------


def _strength(ratio, three_body=True):
    return SimpleNamespace(three_body=three_body, ratio=ratio)


RUN = SimpleNamespace(tp_times_yr=np.array([7616.0, 7920.0, 8226.0]), period_med_yr=610.0, half_amp_med_rad=0.234)


@pytest.mark.parametrize(
    'strength, owner, run, applies',
    [
        (_strength(1000.0), None, RUN, True),  # at the limit (>=)
        (_strength(999.0), None, RUN, False),
        (_strength(1.27e7, three_body=False), None, RUN, False),
        (_strength(NAN), None, RUN, False),
        (_strength(1.27e7), SimpleNamespace(accepted=object()), RUN, False),  # the search accepted an owner already
        (_strength(1.27e7), SimpleNamespace(accepted=None), RUN, True),
        (_strength(1.27e7), None, SimpleNamespace(tp_times_yr=np.array([7616.0])), False),
        (_strength(1.27e7), None, None, False),
    ],
)
def test_route_applies(strength, owner, run, applies):
    assert oi.route_applies(strength, owner, run, S) is applies


# --- the accepted owner and step 2 of the tree ------------------------------------------------------------------


def _candidate(key, r2, rank):
    spec = ResonanceSpec.from_key(key)
    return OwnerCandidate(spec, NAN, r2, (r2,), NAN, NAN, NAN, NAN, NAN, rank, False, 'r2_low')


def _search(candidates):
    nan = dict.fromkeys(
        ('sigma_r2', 'psi_rate_rad_yr', 'psi_slow_corr', 'psi_varpi_corr', 'freq_r2_owner', 'freq_r2_sigma', 'freq_period_yr'), NAN
    )
    return OwnerResult(len(candidates), candidates, None, CheckValue.PASS, flags=['owner_probable'], **nan)


def _accept(original):
    series = SimpleNamespace(resonance=SIGMA, t_yr=np.arange(0.0, 20000.0, 5.0))
    fit = oi.IdentityFit(PHI, -6.89e-4, 0.0105, np.radians(246.0), True)
    phi = oi.PhiOwnership(
        'episodic', [(3431.0, 6612.0, 'libration'), (7617.0, 10618.0, 'libration')], 0.998, st.METHOD_DIRECT, CheckValue.PASS, 1.355
    )
    route = oi.IdentityRoute(PHI.key, fit.coef_dev, fit.resid, fit.span_rad, 'episodic', 0.998, 1.355, 1, 1)
    return oi.identity_owner(original, series, fit, phi, route, RUN)


@pytest.mark.parametrize('original', [None, _search([_candidate('7J-2+0-5', 0.64, 1), _candidate('47M-44+0-3', 0.61, 2)])])
def test_identity_owner_takes_step_two(original):
    owner = _accept(original)
    assert owner.accepted.spec.key == PHI.key and owner.accepted.route == 'identity' and owner.accepted.rank == 1
    assert [c.rank for c in owner.candidates] == list(range(1, len(owner.candidates) + 1))
    assert sum(c.accepted for c in owner.candidates) == 1
    assert owner.check == CheckValue.FAIL and owner.detail == str(OriginDetail.OWNER_LIBRATION)
    assert owner.capture is None and 'owner_probable' not in owner.flags
    assert owner.decisive_window == (7616.0, 8226.0) and owner.cover == pytest.approx(0.998)
    columns = owner.to_columns()
    assert columns['own_route'] == 'identity' and columns['own_name'] == PHI.key
    assert columns['own_id_coef_dev'] == pytest.approx(-6.89e-4) and columns['own_id_phi_str_ratio'] == pytest.approx(1.355)
    if original is not None:  # the candidate of the search keeps its own numbers
        assert owner.accepted.r2_median == pytest.approx(0.61)
    ledger = Ledger.from_settings(S)
    decision = decide._foreign(EPISODIC, None, owner, S, ledger)
    assert (decision.status, decision.reason, decision.origin) == (Status.NON_RESONANT, Reason.INDUCED_LIBRATION, LibrationOrigin.INDUCED)
    assert decision.owner == PHI.key and decision.origin_detail == str(OriginDetail.OWNER_LIBRATION)
    assert 'second_capture_unresolved' in decision.caps  # the route does not test a second capture
    assert not any(entry.name == '2 owner R2' for entry in ledger)  # the route reads nothing on a


def test_empty_identity_columns_without_the_route():
    owner = _search([_candidate('7J-2+0-5', 0.64, 1)])
    columns = owner.to_columns()
    assert columns['own_id_phi_angle_status'] == '' and columns['own_id_coef_dev'] is None


# --- the benchmark ------------------------------------------------------------------------------------------------

# the R-3B impact run's identity log: (c / r - 1, residual, phi cover, phi strength ratio)
IMPACT_433 = {
    'mmr3__433__1R-5V+7+0+0-3': (-6.8946e-4, 0.010517, 0.99834, 1.35483),
    'mmr3__433__1R-5V+7+0+0-3__step0.5yr': (-5.1544e-4, 0.0093330, 0.99900, 1.36080),
}


def _analyse(example):
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(io.load_series(example), S)


@pytest.mark.parametrize('example', sorted(IMPACT_433))
def test_433_follows_47m44_by_the_identity_route(example):
    coef_dev, resid, cover, phi_ratio = IMPACT_433[example]
    analysis = _analyse(example)
    v = analysis.verdict
    assert (v.status, v.angle_status, v.confidence) == (Status.NON_RESONANT, AngleStatus.EPISODIC, Confidence.MEDIUM)
    assert (v.reason, v.libration_origin, v.owner) == (Reason.INDUCED_LIBRATION, LibrationOrigin.INDUCED, '47M-44+0-3')
    assert analysis.strength.ratio == pytest.approx(1.27e7, rel=0.01)
    row = analysis.numbers_row()
    assert row['own_route'] == 'identity'
    assert row['own_id_coef_dev'] == pytest.approx(coef_dev, rel=0.01)
    assert row['own_id_resid'] == pytest.approx(resid, rel=0.01)
    assert row['own_id_phi_cover'] == pytest.approx(cover, rel=1e-3)
    assert row['own_id_phi_str_ratio'] == pytest.approx(phi_ratio, rel=1e-3)
    names = [entry.name for entry in v.decision_path]
    assert '2 owner route (identity)' in names and '2 identity: residual / half-amplitude' in names


def test_433_text_names_the_owner_and_the_route():
    from resonances.plotting import verdict_text as vt

    analysis = _analyse('mmr3__433__1R-5V+7+0+0-3')
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    text = vt.explain_mmr(analysis, S, row, analysis.episode_rows())
    owner_step = next(step for step in text.steps if step.title == '2. owner')
    assert '47M-44+0-3' in owner_step.outcome and 'identity route' in owner_step.outcome
    assert any('47M-44+0-3' in sentence and 'identity route' in sentence for sentence in text.sentences)
    facts = vt.MMRFacts(analysis, S, row, analysis.episode_rows())
    rows = [r for section in vt.physics_sections(facts) for r in section.rows]
    owner_row = next(r for r in rows if r.key == 'owner')
    assert '47M-44+0-3' in owner_row.text and 'by its identity route' in owner_row.text
    assert any(r.key == 'identity_fit' for r in rows) and any(r.key == 'identity_strength' for r in rows)


def test_a_circulating_candidate_gives_no_owner():
    analysis = _analyse('mmr3__1580__4M-7J-5+0+0+8')
    assert analysis.numbers_row()['own_route'] != 'identity'
    assert analysis.verdict.owner != '4E-13+0+9'


def test_without_the_body_longitude_the_route_does_not_run():
    """A series without the body's lambda (a data file without 'l': `MMRSeries.mean_longitude` is optional): the owner
    search reports owner_no_longitudes and the identity route does not run (it crashed in lstsq on a NaN phi before);
    the verdict is the one without an owner, as before W22 (uncertain/episodic)."""
    from dataclasses import replace

    example = 'mmr3__433__1R-5V+7+0+0-3'
    if not io.example_path(example).exists():
        pytest.skip('benchmark data not available')
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    series = replace(io.load_series(example), mean_longitude=None)
    series.validate()
    assert not oi.longitudes_saved(series, S)
    analysis = analyse_mmr(series, S)
    v = analysis.verdict
    assert (v.status, v.angle_status) == (Status.UNCERTAIN, AngleStatus.EPISODIC)
    assert 'owner_no_longitudes' in analysis.owner.flags and analysis.numbers_row()['own_route'] != 'identity'
    assert not v.owner
