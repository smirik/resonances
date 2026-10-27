"""The save and plot modes on a run's files (SAVE, PLOT; docs/config.md, docs/output.md): which bodies get a series
file and which get figures, in the folder layout data/, figures/<layout>/, combined/<layout>/.

Four bodies built by hand in 2J-1, nothing integrated (the bodies of test_impl_w3_output_package):
- librator: a libration with a responding a -> resonant (the analysis' verdict);
- circulator: a circulation -> non_resonant, nothing bounded (the analysis' verdict);
- kinematic: the librator's series with its verdict set by hand to non_resonant with a kinematic libration;
- unsure: the librator's series with its verdict set by hand to uncertain (no level).
The rows each mode selects are written out from the table of docs/config.md, not taken from the package; every mode
but all and none selects some of the four and leaves others. summary.csv always has every body.
"""

import dataclasses

import matplotlib

matplotlib.use('Agg')

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from resonances.resonance.analysis.verdict import Confidence, LibrationOrigin, Status  # noqa: E402
from tests.resonances.simulation.test_impl_w3_output_package import KEY, add_synthetic_body, analysed, synthetic  # noqa: E402

BODIES = ['librator', 'circulator', 'kinematic', 'unsure']
SELECTED = {
    'none': set(),
    'all': set(BODIES),
    'resonant': {'librator'},  # resonant, transient
    'candidates': {'librator', 'unsure'},  # + near_separatrix, uncertain
    'extended': {'librator', 'unsure', 'kinematic'},  # + a non_resonant row with a libration origin
    'nonzero': {'librator', 'unsure', 'kinematic'},  # every row but a non_resonant one with nothing bounded
    'negative': {'unsure'},  # near_separatrix, uncertain, invalid
}
LAYOUT = 'confidence/status'


def _run(tmp_path, **config):
    """The four bodies analysed, their verdicts as in the module docstring, and the run's files written."""
    sim = synthetic(tmp_path, name='librator', plot_layout=LAYOUT, plot_combined_layout=LAYOUT, **config)
    add_synthetic_body(sim, 4000, True, 'circulator')
    add_synthetic_body(sim, 4000, False, 'kinematic')
    add_synthetic_body(sim, 4000, False, 'unsure')
    analysed(sim)
    _set_verdict(sim.bodies[2], status=Status.NON_RESONANT, libration_origin=LibrationOrigin.KINEMATIC)
    _set_verdict(sim.bodies[3], status=Status.UNCERTAIN, confidence=Confidence.NONE, libration_origin=LibrationOrigin.NONE)
    assert {b.name: (b.statuses[KEY], str(b.librations[KEY].libration_origin)) for b in sim.bodies} == {
        'librator': ('resonant', 'dynamic'),
        'circulator': ('non_resonant', ''),
        'kinematic': ('non_resonant', 'kinematic'),
        'unsure': ('uncertain', ''),
    }
    sim.data_manager.save_data(sim.bodies, sim.times, sim)
    return sim


def _set_verdict(body, **fields):
    verdict = dataclasses.replace(body.librations[KEY], **fields)
    body.mmr_analyses[KEY] = dataclasses.replace(body.mmr_analyses[KEY], verdict=verdict)
    body.librations[KEY], body.statuses[KEY] = verdict, str(verdict.status)


def _folder(body):
    """confidence/status of a row ('none' for a verdict without a level), as written out in simulation/layout.py."""
    return f"{str(body.librations[KEY].confidence) or 'none'}/{body.statuses[KEY]}"


def _summary(tmp_path):
    return pd.read_csv(tmp_path / 'summary.csv', dtype=str, keep_default_na=False)


@pytest.mark.parametrize('mode', sorted(SELECTED))
def test_save_mode_writes_the_series_of_the_selected_rows(tmp_path, mode):
    _run(tmp_path, save=mode, plot=None)
    saved = {p.name[len('data-') : -len('.csv')] for p in (tmp_path / 'data').glob('data-*.csv')}
    saved = {name for name in saved if name in BODIES}  # not the periodograms, not the planets
    assert saved == SELECTED[mode]
    assert not list(tmp_path.glob('data-*.csv'))  # series only in the data folder
    assert not (tmp_path / 'figures').exists() and not (tmp_path / 'combined').exists()
    # summary.csv has every body, whatever the mode, with the verdicts of the module docstring
    summary = _summary(tmp_path).set_index('name')
    assert sorted(summary.index) == sorted(BODIES)
    assert summary.loc['kinematic', 'libration_origin'] == 'kinematic' and summary.loc['unsure', 'status'] == 'uncertain'


@pytest.mark.parametrize('mode', sorted(SELECTED))
def test_plot_mode_draws_the_selected_rows(tmp_path, mode):
    sim = _run(tmp_path, save=None, plot=mode, plots=['evolution'])
    expected = {f'figures/{_folder(body)}/{body.name}-{KEY}.png' for body in sim.bodies if body.name in SELECTED[mode]}
    drawn = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*.png')}
    assert drawn == expected
    assert not any((tmp_path / 'data' / f'data-{name}.csv').exists() for name in BODIES)  # save=None
    assert sorted(_summary(tmp_path)['name']) == sorted(BODIES)


def test_the_combined_figures_go_to_their_own_folder(tmp_path):
    sim = _run(tmp_path, save='nonzero', plot='nonzero', plots=['combined'])
    expected = {f'combined/{_folder(body)}/{body.name}-{KEY}-combined.png' for body in sim.bodies if body.name in SELECTED['nonzero']}
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob('*.png')} == expected
    assert {'combined/none/uncertain/unsure-2J-1+0-1-combined.png'} <= expected  # an uncertain row has no level
