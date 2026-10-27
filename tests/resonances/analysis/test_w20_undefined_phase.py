"""W20 (R4, 258640 46J-19+0-27): undefined_phase needs varpi itself undefined; the |p|-scaled jitter alone is the
osculating wobble of p varpi, a flag (jitter_low_e) and a cap <= medium.

Synthetic record, worked out by hand (p = -27, k = -19, a = 2.885 au, 3000 yr at a 0.6-yr step):
- e-vector z = 0.045 exp(2 pi i t / 25 kyr) + r exp(2 pi i t / 4.9 yr), r = 0.0025;
- e = |z| ~ 0.045 + r cos(phi): its short-period std is r / sqrt(2) = 1.768e-3, so the jitter is
  |p| (r / sqrt(2)) / 0.045 = 27 x 0.03928 = 1.061 rad > MMR_JITTER_MAX_RAD = pi/4: jitter_low_e;
- varpi wobbles by arcsin(r / 0.045) = 0.0556 rad (first order) - its own scatter s/|z_f| = r / 0.045 = 0.056,
  far below pi/4 everywhere: varpi_undefined_fraction = 0;
- p varpi wobbles by 27 x 0.0556 = 1.50 rad; per 0.6-yr step that changes by up to 2 x 1.50 x sin(pi 0.6 / 4.9)
  = 1.126 rad = 64.5 deg, plus the second-order term 27 (r/e)^2 / 2 x 2 sin(2 pi 0.6 / 4.9) = 3.3 deg: the full
  angle's residual p99 ~ 68 deg > MMR_UNWRAP_MAX_RESID_P99_DEG = 45 deg (it fails the limits);
- the lambda part 46 lambda_J - 19 lambda is smooth: its residual ~ 0 (unwrap_ok).
Before W20 that record was invalid/undefined_phase (jitter_low_e and the full angle failing), as 258640 was.
"""

import math

import numpy as np
import pytest

from resonances.data import const
from resonances.plotting import verdict_text as vt
from resonances.resonance.analysis.mmr import context as ctx
from resonances.resonance.analysis.mmr import decide, io
from resonances.resonance.analysis.mmr import sampling as smp
from resonances.resonance.analysis.mmr.analyse import analyse_mmr
from resonances.resonance.analysis.mmr.models import MMRSeries, Regime, ResonanceSpec
from resonances.resonance.analysis.ledger import Ledger
from resonances.resonance.analysis.settings import settings_from_config

DEG = math.pi / 180
N_JUPITER = const.PLANET_N_DEG_PER_GAUSS_YR['Jupiter'] * DEG
A_BODY = 2.885  # au, at the 46:19 with Jupiter
E_SLOW, LOOP_RADIUS, LOOP_PERIOD_YR = 0.045, 0.0025, 4.9
S = settings_from_config()


def _series(z: np.ndarray, t: np.ndarray, key: str = '46J-19+0-27') -> MMRSeries:
    """A record of the angle `key` with the body's e-vector z and Jupiter's saved nowhere (sigma from varpi only)."""
    spec = ResonanceSpec.from_key(key)
    lam = float(ctx.mean_motion(A_BODY)) * t + 0.3
    varpi = np.angle(z)
    sigma = spec.planet_coefficients[0] * (N_JUPITER * t + 1.1) + spec.body_coefficient * lam + spec.body_varpi_coefficient * varpi
    ones = np.ones_like(t)
    Omega = 0.7 * ones
    a = A_BODY + 1e-5 * np.sin(2 * math.pi * t / 7.3)
    return MMRSeries(
        t, a, np.abs(z), 0.05 * ones, Omega, np.mod(varpi - Omega, 2 * math.pi), np.mod(varpi, 2 * math.pi),
        np.mod(sigma, 2 * math.pi), spec, Regime(float(t[1] - t[0]), float(t[-1] - t[0]), len(t)), {},
        np.mod(lam, 2 * math.pi), name='w20-synthetic',
    )  # fmt: skip


def _wobbling(key: str = '46J-19+0-27') -> MMRSeries:
    t = np.arange(0.0, 3000.0 + 0.3, 0.6)
    z = E_SLOW * np.exp(2j * math.pi * t / 25000.0) + LOOP_RADIUS * np.exp(2j * math.pi * t / LOOP_PERIOD_YR)
    return _series(z, t, key)


def _noise() -> MMRSeries:
    """The e-vector of test_impl_w15_alias_gate: random direction, Rayleigh length (varpi undefined everywhere)."""
    t = np.arange(0.0, 2000.0 + 0.5, 1.0)
    rng = np.random.default_rng(5)
    return _series(0.002 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t))), t)


def _sampling(series: MMRSeries):
    return smp.analyse_sampling(series, ctx.build_context(series, S), S)


def _full_angle_ok(sampling) -> bool:
    return math.degrees(sampling.unwrap_resid_full_p99_rad) <= S.mmr_unwrap_max_resid_p99_deg and (
        math.degrees(sampling.unwrap_resid_full_max_rad) <= S.mmr_unwrap_max_resid_deg
    )


def test_an_osculating_wobble_of_p_varpi_is_no_undefined_phase():
    sampling = _sampling(_wobbling())
    assert sampling.jitter == pytest.approx(27 * LOOP_RADIUS / math.sqrt(2) / E_SLOW, rel=0.02)  # 1.061 rad
    assert sampling.jitter_low_e and 'jitter_low_e' in sampling.flags
    assert math.degrees(sampling.unwrap_resid_full_p99_rad) == pytest.approx(64.5 + 3.3, abs=3.0)
    assert not _full_angle_ok(sampling)
    assert math.degrees(sampling.unwrap_resid_max_rad) < 1.0
    assert sampling.varpi_undefined_fraction == 0.0
    assert not sampling.undefined_phase and sampling.unwrap_ok
    # the |p|-scaled local proxy is over pi/4 everywhere (27 x 0.056 = 1.5): a column, not a verdict
    assert sampling.phase_undefined_fraction > 0.9
    assert sampling.to_columns()['smp_varpi_undefined_fraction'] == 0.0


def test_varpi_undefined_everywhere_stays_undefined_phase():
    sampling = _sampling(_noise())
    assert sampling.jitter_low_e and not _full_angle_ok(sampling)
    assert sampling.varpi_undefined_fraction > S.mmr_varpi_undefined_max_share
    assert sampling.undefined_phase and not sampling.unwrap_ok


def test_varpi_not_measured_keeps_the_old_rule():
    # 46J-19-1-26: a varpi_J term, Jupiter not saved -> the slow part (and varpi's own scatter) is not measured; the
    # whole angle is judged as before W15. |p| = 26: jitter 26 x 0.03928 = 1.02 rad > pi/4
    sampling = _sampling(_wobbling('46J-19-1-26'))
    assert math.isnan(sampling.varpi_undefined_fraction)
    assert sampling.jitter == pytest.approx(26 * LOOP_RADIUS / math.sqrt(2) / E_SLOW, rel=0.02)
    assert sampling.undefined_phase == (sampling.jitter_low_e and not _full_angle_ok(sampling))
    assert sampling.undefined_phase


def test_the_share_is_a_setting():
    # the noise record's share is ~1: a limit of 1 lets it pass, and the record is no longer undefined_phase
    settings = settings_from_config(mmr_varpi_undefined_max_share=1.0)
    series = _noise()
    sampling = smp.analyse_sampling(series, ctx.build_context(series, settings), settings)
    assert sampling.varpi_undefined_fraction <= 1.0 and not sampling.undefined_phase


def _path(analysis):
    return {entry.name: entry for entry in analysis.verdict.decision_path}


def test_decision_path_explains_the_phase_when_jitter_low_e():
    defined = analyse_mmr(_wobbling(), S, scan_owner=False, scan_subresonances=False)
    path = _path(defined)
    assert str(defined.verdict.angle_status) != 'undefined'
    assert path['G0 angle jitter'].result is False and path['G0 angle jitter'].value == pytest.approx(1.061, rel=0.02)
    assert path['G0 full-angle residual p99'].result is False
    assert path['G0 varpi phase defined'].result is True and path['G0 varpi phase defined'].value == 0.0
    assert 'jitter_low_e' in defined.verdict.caps

    lost = analyse_mmr(_noise(), S, scan_owner=False, scan_subresonances=False)
    assert str(lost.verdict.angle_subtype) == 'undefined_phase'
    assert _path(lost)['G0 varpi phase defined'].result is False


def test_g0_records_nothing_on_the_phase_without_jitter():
    sampling = _sampling(_series(E_SLOW * np.exp(2j * math.pi * np.arange(0.0, 3000.3, 0.6) / 25000.0), np.arange(0.0, 3000.3, 0.6)))
    assert not sampling.jitter_low_e
    ledger = Ledger.from_settings(S)
    decide.decide_angle(sampling, False, None, None, S, ledger)
    assert 'G0 varpi phase defined' not in {entry.name for entry in ledger}


EXAMPLE_258640 = 'mmr2__258640__46J-19+0-27'


def test_258640_46J_19_is_a_circulation():
    """R4: varpi winds 1.96 rev at every band from 10 to 5000 yr, its own scatter never above pi/4 (max 0.555 rad);
    jitter 1.03, full-angle p99 101 deg. Author and Codex: non_resonant; the cap jitter_low_e keeps it at medium."""
    if not io.example_path(EXAMPLE_258640).exists():
        pytest.skip('benchmark data not available')
    analysis = analyse_mmr(io.load_series(EXAMPLE_258640), S)
    v = analysis.verdict
    assert (str(v.status), str(v.angle_status), str(v.confidence)) == ('non_resonant', 'circulation', 'medium')
    assert 'jitter_low_e' in v.caps
    assert analysis.sampling.jitter_low_e and analysis.sampling.varpi_undefined_fraction == 0.0


def _explained(series: MMRSeries):
    analysis = analyse_mmr(series, S, scan_owner=False, scan_subresonances=False)
    row = {**analysis.numbers_row(), **analysis.summary_row()}
    return analysis, vt.explain_mmr(analysis, S, row, analysis.episode_rows())


def test_the_figure_says_why_the_phase_is_undefined():
    # noise: varpi's own scatter above pi/4 on every measured sample (share 1 > 0.5)
    analysis, text = _explained(_noise())
    assert str(analysis.verdict.angle_subtype) == 'undefined_phase'
    assert text.steps[-1].outcome.startswith('ϖ itself undefined on 100% > 50% of the record (e down to ')
    # not measured (Jupiter of the varpi_J term not saved): the old wording, no share
    _, text = _explained(_wobbling('46J-19-1-26'))
    assert text.steps[-1].outcome == 'ϖ undefined at small e (e down to 0.0425; its own scatter not measured)'  # 0.045 - r
    # the wobble: no leaf on the phase, the cap is named
    analysis, text = _explained(_wobbling())
    assert str(analysis.verdict.confidence) == 'medium'
    assert any('the p·ϖ term wobbles by more than 45° at short periods (osculating)' in s for s in text.sentences)
