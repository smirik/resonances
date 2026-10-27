"""W11: a witness of rule R needs an identity that does not fail (tmp/classifier-v2-2026-10-15/impl/w11/witness).

On 1 Myr, 37087 3J+8S-3 has short episodes of ~167 yr whose a_dot follows the multiplet force law strictly while
sigma_dot runs against c (a - a*) (slope about -1.4 c). The expected values come from outside the classifier:

- the period of the circulating near-2:1 Jupiter argument psi = lambda - 2 lambda_J + varpi, read from the saved
  longitudes (Codex, impl/w11/witness/ANSWER.md: 166.4-167.4 yr on the seven episodes, a_dot answers sin psi with
  R^2 0.66-0.71): the short episodes run at that clock, so a and sigma follow another term;
- the author's verdict (benchmark labels, 2026-10-16): non_resonant, a kinematic libration, as on 100 kyr.
"""

import numpy as np
import pytest

from resonances.resonance.analysis.mmr import io
from resonances.resonance.analysis.settings import settings_from_config
from resonances.resonance.analysis.verdict import CheckValue, LibrationOrigin, Reason, Status

S = settings_from_config()
EXAMPLE = 'mmr3__37087__3J+8S-3+0+0-8__1Myr'
PERIOD_TOL = 0.03  # the episodes' periods (166-169 yr) against the clock of psi


def _rate(t_yr, angle):
    return float(np.polyfit(t_yr - t_yr.mean(), np.unwrap(angle), 1)[0])


# The record is loaded once; its two analyses (1M rows each) are never alive together: the cut-line one is made for its
# single test and dropped after it, before the module's own analysis is made (W23 P8: they set the suite's peak).
@pytest.fixture(scope='module')
def series():
    if not io.example_path(EXAMPLE).exists():
        pytest.skip('benchmark data not available')
    return io.load_series(EXAMPLE)


@pytest.fixture(scope='module')
def psi_period_yr(series):
    t = np.asarray(series.t_yr, float)
    jupiter = series.planets['Jupiter']
    psi_rate = _rate(t, series.mean_longitude) - 2 * _rate(t, jupiter.mean_longitude) + _rate(t, series.varpi)
    return 2 * np.pi / abs(psi_rate)


@pytest.fixture(scope='module')
def analysed(series, psi_period_yr):
    from resonances.resonance.analysis.mmr.analyse import analyse_mmr

    return analyse_mmr(series, S), psi_period_yr


@pytest.fixture
def analysed_cut_line(series):
    """The record as the TP threshold of W10-W17 saw it: the libration's own image not taken out
    (`analyse.libration_image` returning nothing), so the 167-yr forced line raised its own threshold to 70.8 deg and fell
    apart into the short episodes the witness rule has to reject."""
    from resonances.resonance.analysis.mmr import analyse as analyse_module

    patch = pytest.MonkeyPatch()
    patch.setattr(analyse_module, 'libration_image', lambda prep, floor, settings: np.zeros(len(prep.t_yr)))
    try:
        return analyse_module.analyse_mmr(series, S)
    finally:
        patch.undo()


def test_37087_1myr_forced_episodes_are_no_witnesses(psi_period_yr, analysed_cut_line):
    assert psi_period_yr == pytest.approx(167.0, rel=PERIOD_TOL)
    analysis = analysed_cut_line
    # W15: these force laws sit on the forced line 2 lambda_J - lambda of the angle's lattice (175 yr at the exact
    # commensurability, within the window's resolution of 166-169 yr): measured strict passes become not_measured
    # W20 CS7 (F6): an episode on that line whose identity fails is a forced wave, no episode of the angle at all
    # (`Geometry.forced_waves`); before, rule R had to keep it from being a witness
    waves = [w for w in analysis.geometry.forced_waves if abs(w.period_yr / psi_period_yr - 1) <= PERIOD_TOL]
    assert len(waves) >= 7  # W10: the seven strict passes 51, 58, 60, 74, 81, 91, 109
    for w in waves:
        assert w.on_line and w.force_line == '2J-1'
        assert w.identity_slope_ratio < 0 and w.identity_corr < 0
    assert not {w.episode_id for w in waves} & {e.episode_id for e in analysis.response.episode_responses}
    assert not any(e.witness for e in analysis.response.episode_responses)
    assert analysis.response.witness_episode_id is None


def test_37087_1myr_the_whole_forced_line_has_no_witness(analysed):
    """W18: the forced line is slower than P_floor, so its runs no longer raise their own threshold (70.8 -> 19.2 deg,
    2286 -> 10294 TPs); the runs fail the centre test and no episode is a witness. The angle alone does not tell a
    forced wave slower than P_floor from a libration: the identity and the force law must."""
    analysis, _ = analysed
    assert len(analysis.geometry.turning_points) > 4 * 2286
    assert not any(e.witness for e in analysis.response.episode_responses)
    assert analysis.response.witness_episode_id is None


def test_37087_1myr_without_its_forced_wave_is_non_resonant(analysed):
    """W20 CS7: the 168-yr episode is a forced wave (line 2J-1, identity -0.35 c); the status and the confidence
    are the author's (non_resonant, medium as this file's earlier test), the angle word is circulation."""
    verdict = analysed[0].verdict
    assert (verdict.status, str(verdict.confidence)) == (Status.NON_RESONANT, 'medium')
    assert analysed[0].geometry.forced_waves and not analysed[0].geometry.episodes


@pytest.mark.xfail(
    strict=True,
    reason=(
        'W20 CS7: the 168-yr forced wave is no episode now (non_resonant/circulation/medium), but the 5.7-kyr '
        'kinematic loops are no episodes at the 75-yr cutoff either: the 168-yr turning points fill the record '
        '(10294 TPs). They show only above the line (cutoff 505 yr: two episodes, P 5.73 kyr), where the force law '
        'passes thin (R2 0.58/0.61) and the verdict would be transient; reason kinematic_libration open (author)'
    ),
)
def test_37087_1myr_is_a_kinematic_libration(analysed):
    verdict = analysed[0].verdict
    assert (verdict.status, verdict.reason, verdict.libration_origin) == (
        Status.NON_RESONANT,
        Reason.KINEMATIC_LIBRATION,
        LibrationOrigin.KINEMATIC,
    )
    assert str(verdict.confidence) == 'medium'
    assert verdict.checks['check_mmr_kinematic'] == CheckValue.FAIL
