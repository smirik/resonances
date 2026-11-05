"""The shared blocks of the right column (`plotting/verdict_column_common.py`): the pills' words for inputs the policy
reads in a way the plain state does not say (a circulation of an alias, an N without an episode), and the chip's
lengths. Hand-built summary rows; the expected texts are written out from the policy's rules."""

from types import SimpleNamespace

from resonances.plotting import verdict_column as vc
from resonances.plotting import verdict_column_common as common
from resonances.plotting.verdict_vocab import MMR_MARGIN_NAMES

CLEAN_CUTOFFS = {'geo_n_tp': 0, 'geo_n_tp_cut_half': 0, 'geo_n_tp_cut_double': 0}


def test_an_alias_circulation_names_c5_not_the_center_turns():
    """An alias row: the policy reads C5 and C2 (`policy._mmr_circulation_clear`); C1 is never measured there
    (`decide._circulation_margins`). C2 full (no turning point at ×½, ×1, ×2) and C5 thin: the level is high because
    'far from a* and one alias branch' is not established, not because of the center's turns."""
    row = {
        'pol_level_inputs': 'circulation:not_clear:high',
        'resonance_type': 'mmr',
        'angle_subtype': 'alias',
        'mrg_C2': 'full',
        'mrg_C5': 'thin',
        **CLEAN_CUTOFFS,
    }
    assert MMR_MARGIN_NAMES['C5'] == 'far from a* and one alias branch'
    (pill,) = common.pills(row, 'high')
    assert (pill.icon, pill.text) == (vc.THIN, 'far from a* and one alias branch: not established')
    assert common.confidence(row, 'high').because == 'not certain because far from a* and one alias branch: not established'
    assert 'whole turns' not in pill.text
    unmeasured = {**row, 'mrg_C5': 'not_measured'}
    assert common.pill_text('circulation', 'not_clear', '', unmeasured) == 'far from a* and one alias branch: not measured'


def test_a_circulation_whose_center_turns_were_not_measured_says_so():
    """A non-alias MMR row with C1 not measured (NaN in the CSV, or 'not_measured'): no 'too few whole turns'; a C1
    thin or failed keeps that sentence; a secular row reads P in its place."""
    base = {'resonance_type': 'mmr', 'angle_subtype': 'stationary', **CLEAN_CUTOFFS}
    for c1 in (float('nan'), 'not_measured'):
        assert common.pill_text('circulation', 'not_clear', '', {**base, 'mrg_C1': c1}) == 'center turns not measured'
    too_few = 'its center passes too few whole turns for a clear circulation'
    assert common.pill_text('circulation', 'not_clear', '', {**base, 'mrg_C1': 'thin'}) == too_few
    secular = {**CLEAN_CUTOFFS, 'resonance_type': 'secular', 'mrg_C1': 'not_measured'}
    assert common.pill_text('circulation', 'not_clear', '', {**secular, 'mrg_P': 'thin'}) == too_few
    assert common.pill_text('circulation', 'not_clear', '', {**secular, 'mrg_P': 'not_measured'}) == 'center turns not measured'


def test_no_episode_prints_no_cycles():
    """N and core of a row without a deciding episode are 'nan' in pol_level_inputs (`policy._number`); the level
    reads them as 0, so N < 2 holds the level at low. The pills say 'no cycles', never 'none cycles'."""
    row = {'pol_level_inputs': 'N:nan<2:low;core:nan<1:high'}
    found = {pill.key: pill for pill in common.pills(row, 'low')}
    assert found['N'].icon == vc.THIN
    assert found['N'].text == 'no cycles counted [[no episode; medium needs ≥ 2]]'
    assert found['core'].text.startswith('no cycles between turning points [[no deciding episode; ')
    assert all('none' not in pill.text for pill in found.values())
    assert common.confidence(row, 'low').because == 'not certain because no cycles counted'
    # a counted N keeps its number
    assert common.pill_text('N', '1', '<2', {}, 'low') == '1 cycle counted [[medium needs ≥ 2]]'


def _verdict(**kwargs):
    return SimpleNamespace(status='transient', confidence='medium', **kwargs)


def test_the_chip_rounds_the_length_before_choosing_the_unit():
    """999.95 yr rounds to 1000 yr = 1 kyr, 999 950 yr to 1 Myr (`verdict_text.years`); never '1e+03 yr'."""
    expected = {999.95: 'extend to 1 kyr', 999_950: 'extend to 1 Myr', 2.0e7: 'extend to 20 Myr', 150_000: 'extend to 150 kyr'}
    for to, text in expected.items():
        assert common.chip(_verdict(action='extend', extend_to_years=to), {}, 'mmr').text == text
    assert common.chip(_verdict(action='extend', extend_to_years=None), {}, 'mmr').text == 'extend the integration'
    assert common.chip(_verdict(action='extend', extend_to_years=True), {}, 'mmr').text == 'extend the integration'
    assert common.chip(_verdict(action='resample', resample_step_yr=0.5), {}, 'mmr').text == 'resample to 0.5 yr'


def test_text_reads_empty_cells():
    assert [common.text(value) for value in (None, float('nan'), ' G6 ', 3)] == ['', '', 'G6', '3']
