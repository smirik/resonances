"""The folder layout of a run's output (simulation/layout.py, docs/output.md).

The tables stay at the top of save_path, the series CSVs go to the data folder, the individual and the combined figures
to their own folders, each split by a layout preset; every name can be overridden, and a restore still finds the
series (new layout, flat layout, and a run saved before the data folder existed). The bodies are the hand-built ones
of test_impl_w3_output_package (a 2J-1 libration -> resonant, a circulation -> non_resonant) and of
test_restore_and_resume; the expected paths are written out here, not taken from the package.
"""

import json
import shutil

import matplotlib

matplotlib.use('Agg')

import pytest  # noqa: E402

from resonances.config import config as package_config  # noqa: E402
from resonances.simulation import layout  # noqa: E402
from resonances.simulation.batch_manager import BatchManager  # noqa: E402
from resonances.simulation.config import SimulationConfig  # noqa: E402
from resonances.simulation.serializer import SimulationSerializer  # noqa: E402
from tests.resonances.simulation.test_impl_w3_output_package import KEY, add_synthetic_body, analysed, synthetic  # noqa: E402
from tests.resonances.simulation.test_restore_and_resume import LIB, _saved_run  # noqa: E402

LEVELS = ('certain', 'high', 'medium', 'low')


# --------------------------------------------------------------------------- the layout function


@pytest.mark.parametrize(
    'text,levels',
    [
        ('', ()),
        (None, ()),
        ('flat', ()),
        ('status', ('status',)),
        ('confidence', ('confidence',)),
        ('confidence/status', ('confidence', 'status')),
        ('status/confidence', ('status', 'confidence')),
        (' Type / Status ', ('type', 'status')),
        ('type,confidence,status', ('type', 'confidence', 'status')),
        (['resonance', 'body'], ('resonance', 'body')),
    ],
)
def test_parse_layout(text, levels):
    assert layout.parse_layout(text) == levels
    assert layout.layout_name(levels) == ('/'.join(levels) or 'flat')


@pytest.mark.parametrize('text', ['colour', 'status/status', 'status/kind'])
def test_unknown_or_repeated_level_is_an_error(text):
    with pytest.raises(ValueError):
        layout.parse_layout(text)
    with pytest.raises(ValueError):
        SimulationConfig(plot_layout=text)


def test_every_preset_parses():
    for preset in layout.PRESETS:
        assert layout.layout_name(layout.parse_layout(preset)) == preset


def test_parts_of_summary_rows():
    """Rows as summary.csv has them: an empty confidence (uncertain, invalid) is 'none', an unsafe name is cleaned."""
    uncertain = {'name': '2003 UG22', 'resonance': '4J-2S-1+0+0-1', 'resonance_type': 'mmr', 'status': 'uncertain', 'confidence': ''}
    zlk = {'name': '10563', 'resonance': 'LK', 'resonance_type': 'lidov_kozai', 'status': 'resonant', 'confidence': 'high'}
    assert layout.layout_parts(('confidence', 'status'), uncertain) == ['none', 'uncertain']
    assert layout.layout_parts(('status', 'confidence'), zlk) == ['resonant', 'high']
    assert layout.layout_parts(('type', 'resonance', 'body'), uncertain) == ['mmr', '4J-2S-1+0+0-1', '2003_UG22']
    assert layout.layout_parts(('status',), {'status': None}) == ['unclassified']
    assert layout.layout_parts(('confidence',), {'confidence': float('nan')}) == ['none']
    assert str(layout.layout_path('/run', 'combined', ('status',), zlk)) == '/run/combined/resonant'
    assert str(layout.layout_path('/run', '', (), zlk)) == '/run'


# --------------------------------------------------------------------------- config keys and the old name


def test_defaults_of_the_package():
    """The folders, the layout of the benchmark's tree and the one-page figure (author 2026-10-27, W24 Q6 and Q8)."""
    config = SimulationConfig()
    assert (config.data_folder, config.plot_figures_folder, config.plot_combined_folder) == ('data', 'figures', 'combined')
    assert (config.plot_layout, config.plot_combined_layout) == ('confidence/status', 'confidence/status')
    assert config.plots == ['combined']
    assert config.data_path == f'{config.save_path}/data'


def test_the_old_env_key_is_read_when_no_layout_is_set(monkeypatch):
    monkeypatch.setitem(package_config.config, 'PLOT_LAYOUT', '')
    monkeypatch.setitem(package_config.config, 'PLOT_SUBFOLDER_STRATEGY', '')
    assert SimulationConfig().plot_layout == 'confidence/status'  # neither key: the default
    monkeypatch.setitem(package_config.config, 'PLOT_SUBFOLDER_STRATEGY', 'flat')
    assert SimulationConfig().plot_layout == 'flat'
    monkeypatch.setitem(package_config.config, 'PLOT_SUBFOLDER_STRATEGY', 'status')
    assert SimulationConfig().plot_layout == 'status'
    monkeypatch.setitem(package_config.config, 'PLOT_LAYOUT', 'confidence/status')
    config = SimulationConfig()
    assert (config.plot_layout, config.plot_combined_layout) == ('confidence/status', 'confidence/status')
    monkeypatch.setitem(package_config.config, 'PLOT_COMBINED_LAYOUT', 'type')
    assert SimulationConfig().plot_combined_layout == 'type'
    # an explicit argument wins over the environment
    assert SimulationConfig(plot_subfolder_strategy='').plot_layout == 'flat'


# --------------------------------------------------------------------------- figures of a run, every preset


@pytest.fixture(scope='module')
def two_rows(tmp_path_factory):
    """A librating body (resonant) and a circulating one (non_resonant), analysed, nothing saved yet."""
    tmp_path = tmp_path_factory.mktemp('layout_rows')
    sim = synthetic(tmp_path)
    add_synthetic_body(sim, 4000, True, 'circulator')
    analysed(sim)
    librator, circulator = sim.bodies
    assert (librator.statuses[KEY], circulator.statuses[KEY]) == ('resonant', 'non_resonant')
    return sim


def _confidence(body):
    level = str(body.librations[KEY].confidence)
    return level or 'none'


@pytest.mark.parametrize(
    'preset,librator,circulator',
    [
        ('flat', '', ''),
        ('status', 'resonant', 'non_resonant'),
        ('confidence', '{lib}', '{circ}'),
        ('confidence/status', '{lib}/resonant', '{circ}/non_resonant'),
        ('status/confidence', 'resonant/{lib}', 'non_resonant/{circ}'),
        ('type/status', 'mmr/resonant', 'mmr/non_resonant'),
        ('type/confidence/status', 'mmr/{lib}/resonant', 'mmr/{circ}/non_resonant'),
        ('resonance', KEY, KEY),
        ('body', 'synthetic', 'circulator'),
    ],
)
def test_figure_folders_of_every_preset(two_rows, preset, librator, circulator):
    sim = two_rows
    lib_body, circ_body = sim.bodies
    sim.config.plot_layout = sim.config.plot_combined_layout = preset
    for body, expected in ((lib_body, librator), (circ_body, circulator)):
        expected = expected.format(lib=_confidence(lib_body), circ=_confidence(circ_body))
        tail = f'/{expected}' if expected else ''
        resonance = body.resonances()[0]
        assert sim.data_manager.figure_folder(body, resonance) == f'{sim.config.plot_path}/figures{tail}'
        assert sim.data_manager.figure_folder(body, resonance, combined=True) == f'{sim.config.plot_path}/combined{tail}'
    assert _confidence(lib_body) in LEVELS  # a resonant verdict has a level


def _files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not p.name.endswith('.lock'))


def test_a_run_lands_where_the_layout_says(tmp_path):
    """Default folders, 'confidence/status' for the individual figures and 'type/status' for the combined ones."""
    sim = synthetic(
        tmp_path, plot='all', plots=['combined', 'portrait'], plot_layout='confidence/status', plot_combined_layout='type/status'
    )
    analysed(sim)
    body = sim.bodies[0]
    level = _confidence(body)
    sim.data_manager.save_data(sim.bodies, sim.times, sim)

    files = _files(tmp_path)
    assert f'figures/{level}/resonant/synthetic-{KEY}-portrait.png' in files
    assert f'combined/mmr/resonant/synthetic-{KEY}-combined.png' in files
    assert 'data/data-synthetic.csv' in files and 'data/data-synthetic-periodograms.csv' in files
    assert 'data/data-planet-Jupiter.csv' in files
    for table in ('summary.csv', 'mmr_numbers.csv', 'episodes.csv', 'constants.csv', 'simulation.json'):
        assert table in files, table
    # nothing else at the top but the tables and simulation.json; no CSV among the figures
    top = [f for f in files if '/' not in f]
    assert all(f.endswith(('.csv', '.json')) and not f.startswith('data-') for f in top), top
    assert not [f for f in files if f.startswith(('figures/', 'combined/')) and not f.endswith('.png')]


def test_every_name_can_be_overridden(tmp_path):
    """Other folder names, nested ones, figures outside the run folder, and '' for the top itself."""
    plots = tmp_path / 'elsewhere'
    sim = synthetic(
        tmp_path / 'run',
        plot='all',
        plots=['combined', 'portrait'],
        plot_path=str(plots),
        data_folder='series/raw',
        plot_figures_folder='',
        plot_combined_folder='verdicts',
        plot_layout='body',
        plot_combined_layout='flat',
    )
    analysed(sim)
    sim.data_manager.save_data(sim.bodies, sim.times, sim)

    assert (tmp_path / 'run' / 'series' / 'raw' / 'data-synthetic.csv').exists()
    assert (tmp_path / 'run' / 'summary.csv').exists()
    assert _files(plots) == [f'synthetic/synthetic-{KEY}-portrait.png', f'verdicts/synthetic-{KEY}-combined.png']
    manifest = json.loads((tmp_path / 'run' / 'simulation.json').read_text())['simulation']['data_files']
    assert manifest['bodies'] == ['series/raw/data-synthetic.csv']

    restored = SimulationSerializer.restore(str(tmp_path / 'run' / 'simulation.json'), recompute_librations=True)
    assert restored.config.data_folder == 'series/raw' and restored.config.plot_layout == 'body'
    assert restored.bodies[0].statuses[KEY] == 'resonant'
    assert KEY in restored.bodies[0].angles


# --------------------------------------------------------------------------- restore finds the series


@pytest.mark.parametrize('data_folder', ['data', ''])
def test_restore_finds_the_series(tmp_path, data_folder):
    _saved_run(tmp_path, data_folder=data_folder)
    folder = tmp_path / data_folder if data_folder else tmp_path
    assert (folder / 'data-librator.csv').exists()

    sim = SimulationSerializer.restore(str(tmp_path / 'simulation.json'), recompute_librations=True)
    librator = sim.bodies[0]
    assert librator.statuses[LIB] == 'resonant'
    assert LIB in librator.angles and LIB in librator.periodogram_frequency


def test_restore_of_a_run_saved_before_the_data_folder(tmp_path):
    """A flat run whose simulation.json has no data_folder (the config then defaults to 'data') and, as the oldest
    files, no manifest of the bodies: the series are found at the top."""
    _saved_run(tmp_path, data_folder='')
    path = tmp_path / 'simulation.json'
    saved = json.loads(path.read_text())
    saved['config'].pop('data_folder')
    for key in ('plot_layout', 'plot_combined_layout', 'plot_figures_folder', 'plot_combined_folder'):
        saved['config'].pop(key)
    saved['config']['plot_subfolder_strategy'] = 'status'
    for files in ({k: v for k, v in saved['simulation']['data_files'].items() if k != 'bodies'}, saved['simulation']['data_files']):
        saved['simulation']['data_files'] = files
        path.write_text(json.dumps(saved))
        sim = SimulationSerializer.restore(str(path), recompute_librations=True)
        assert sim.config.data_folder == 'data' and sim.config.plot_layout == 'status'
        assert sim.bodies[0].statuses[LIB] == 'resonant'
        assert LIB in sim.bodies[0].angles and LIB in sim.bodies[0].periodogram_frequency


# --------------------------------------------------------------------------- edge cases (Codex review, 2026-10-26)


def test_dot_names_never_leave_their_folder():
    assert layout.layout_parts(('body',), {'name': '..'}) == ['__']
    assert layout.layout_parts(('body',), {'name': '.'}) == ['_']
    assert layout.layout_parts(('body',), {'name': '...'}) == ['...']


def test_an_absolute_data_folder_is_listed_relative_to_the_run(tmp_path):
    """The manifest names the series relative to the run folder, so a moved run restores from its new place."""
    run = tmp_path / 'run'
    sim = synthetic(run, data_folder=str(run / 'series'))
    analysed(sim)
    sim.data_manager.save_data(sim.bodies, sim.times, sim)
    manifest = json.loads((run / 'simulation.json').read_text())['simulation']['data_files']
    assert manifest['bodies'] == ['series/data-synthetic.csv']

    moved = tmp_path / 'moved'
    shutil.move(str(run), str(moved))
    restored = SimulationSerializer.restore(str(moved / 'simulation.json'), recompute_librations=True)
    assert KEY in restored.bodies[0].angles and restored.bodies[0].statuses[KEY] == 'resonant'


@pytest.mark.parametrize('data_folder,in_hash', [('data', False), ('', False), ('series', True)])
def test_resume_hash_and_the_data_folder(tmp_path, data_folder, in_hash):
    """'' and 'data' may change on resume (the manifest looks in both); any other folder is part of the hash."""
    config = SimulationConfig(save_path=str(tmp_path), plot_path=str(tmp_path), data_folder=data_folder)
    assert ('data_folder' in BatchManager(config)._get_config_dict()) is in_hash


def test_a_run_resumed_across_the_data_folder_lists_both_places(tmp_path):
    """Batches written flat before the change and into data/ after it: the manifest has both, restore reads both."""
    sim = synthetic(tmp_path, data_folder='')
    add_synthetic_body(sim, 4000, True, 'circulator')
    analysed(sim)
    first, second = sim.bodies
    sim.data_manager.save_body(first, sim.times)
    sim.config.data_folder = 'data'
    sim.data_manager.save_body(second, sim.times)
    sim.data_manager.save_simulation_summary(sim.bodies)
    sim.data_manager.save_configuration_details(sim.bodies, sim)
    manifest = json.loads((tmp_path / 'simulation.json').read_text())['simulation']['data_files']
    assert manifest['bodies'] == ['data-synthetic.csv', 'data/data-circulator.csv']
    restored = SimulationSerializer.restore(str(tmp_path / 'simulation.json'))
    assert all(KEY in body.angles for body in restored.bodies)
