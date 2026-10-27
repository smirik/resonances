"""Real MMR runs through `find`, at a fixed Horizons date (the slow tests drift with the epoch otherwise).

The expected statuses are the author's (benchmark/author_label_decisions_2026-10-15.csv, 2026-10-27: 463 4J-2S-1
resonant, 463 5J-4S-1 non_resonant, 490 5J-2S-2 transient, the Trojans 624/588/617 1J-1 resonant); the saved 100-kyr
records of the same rows are gated by `make test-benchmark`.
"""

import numpy as np
import pytest
import resonances
from resonances.resonance.analysis.verdict import Status
from tests.tools import accepted_episodes

DATE = '2026-09-29'  # fixed, as the backward Lola test


def _status(summary, name, key):
    return summary.loc[(summary['name'] == name) & (summary['resonance'] == key), 'status'].iloc[0]


@pytest.mark.slow
def test_find(tmp_path):
    sim = resonances.find(
        [463, 490],
        ['Jupiter', 'Saturn'],
        name='test_real_mmrs_find',
        source='nasa',
        date=DATE,
        save='all',
        plot=None,  # the figures of a real run: tests/resonances/plotting/test_mmr_plots.py::TestRealRun
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        integration_years=40000,
        type=['mmr'],
    )
    assert isinstance(sim, resonances.Simulation)
    assert 2 == len(sim.bodies)
    # 5J-4S-1 lies 0.09 au from Lola, outside the finder's three-body width: a clone of 463 (same elements, same date)
    # carries it, as test_3body did with the AstDyS axis
    lola_elements = next(b for b in sim.bodies if b.name == '463').initial_data
    sim.add_body(lola_elements, '5J-4S-1', name='463, resonance=5J-4S-1')
    sim.run()

    summary = sim.data_manager.get_simulation_summary(sim.bodies)
    assert _status(summary, '463', '4J-2S-1+0+0-1') == Status.RESONANT
    assert _status(summary, '490', '5J-2S-2+0+0-1') == Status.TRANSIENT
    assert _status(summary, '463, resonance=5J-4S-1', '5J-4S-1+0+0+0') == Status.NON_RESONANT

    # 463 in 4J-2S-1 is a known librator: its libration period and centre are measured
    row_463 = summary.loc[(summary['name'] == '463') & (summary['resonance'] == '4J-2S-1+0+0-1')].iloc[0]
    # Period should be in a reasonable range (roughly 5000-30000 yr for 40 Kyr integration)
    assert 1000 < row_463['geo_period_cycle_yr'] < 30000
    lola = next(b for b in sim.bodies if b.name == '463')
    episodes = accepted_episodes(lola, '4J-2S-1+0+0-1')
    assert episodes
    # Center should be finite and in [0, 360) deg
    assert all(0 <= e['centre_mod_2pi_deg'] < 360 for e in episodes)

    # 490 (transient) has a libration episode too
    episodes_490 = accepted_episodes(next(b for b in sim.bodies if b.name == '490'), '5J-2S-2+0+0-1')
    assert episodes_490 and np.isfinite(episodes_490[0]['centre_mod_2pi_deg'])


@pytest.mark.slow
def test_trojans(tmp_path):
    asteroids = [624, 588, 617]

    sim = resonances.find(
        asteroids,
        ['Jupiter'],
        name='test_real_mmrs_trojans',
        source='nasa',
        date=DATE,
        save=None,
        plot=None,  # find gives each Trojan ~150 high-order Jupiter lines; drawing them took 90 of the 166 s
        save_path=str(tmp_path),
        plot_path=str(tmp_path),
        type="mmr",
    )
    sim.run()
    summary = sim.data_manager.get_simulation_summary(sim.bodies)

    for asteroid in asteroids:
        assert _status(summary, str(asteroid), '1J-1+0+0') == Status.RESONANT, asteroid
    # 1J+1+0-2, the retrograde twin of 1J-1 at the same axis, is no longer in the catalog (TwoBodyMatrix v2)
    assert '1J+1+0-2' not in set(summary['resonance'])
