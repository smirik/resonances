"""The asteroids' elements and the planets must start at the same instant (config.date)."""

import datetime

import astdys
import pytest
from click.testing import CliRunner

import resonances
import resonances.finder.finder as finder_module
import resonances.horizons
from resonances.cli.main import cli
from resonances.data.util import mjd_from_datetime
from resonances.simulation.simulation import Simulation

# MJD 60000 is 2023-02-25 00:00 (JD 2460000.5)
DATE_60000 = datetime.datetime(2023, 2, 25)


def _catalog_epoch() -> datetime.datetime:
    """The AstDyS epoch read off the catalog row, converted by hand (MJD 0 = 1858-11-17)."""
    return datetime.datetime(1858, 11, 17) + datetime.timedelta(days=float(astdys.search(1)['epoch']))


def _fake_horizons(calls):
    """Stands in for the Horizons query: records the date and returns elements tagged with it."""

    def fetch(s, date):
        calls.append((s, date))
        days = (date - DATE_60000).total_seconds() / 86400.0
        return {'a': 2.398, 'e': 0.22, 'inc': 0.236, 'Omega': 0.637, 'omega': 5.75, 'M': 1.0 + 1e-3 * days, 'epoch': 60000.0 + days}

    return fetch


@pytest.mark.parametrize(
    'date, mjd',
    [
        (datetime.datetime(1858, 11, 17), 0.0),
        (datetime.datetime(2000, 1, 1, 12), 51544.5),  # J2000
        ('2023-02-25', 60000.0),
        ('2026-06-09 06:00', 61200.25),
        (datetime.datetime(2000, 1, 1, 14, tzinfo=datetime.timezone(datetime.timedelta(hours=2))), 51544.5),
    ],
)
def test_mjd_from_datetime(date, mjd):
    assert mjd_from_datetime(date) == pytest.approx(mjd, abs=1e-9)


def test_data_source_from_config_sets_catalog_epoch(monkeypatch):
    """DATA_SOURCE=astdys from .env or the environment (no source argument) dates the run at the catalog epoch."""
    monkeypatch.setitem(resonances.config.config, 'DATA_SOURCE', 'astdys')
    expected = _catalog_epoch()
    assert expected != datetime.datetime.combine(datetime.date.today(), datetime.time.min)

    sim = Simulation()
    assert sim.config.source == 'astdys'
    assert sim.config.date == expected


def test_unknown_data_source_raises():
    with pytest.raises(ValueError, match='Unknown data source'):
        Simulation(source='minor-planet-center')


def test_astdys_body_at_catalog_epoch_is_added():
    sim = Simulation(source='astdys')
    sim.add_body(463, '4J-2S-1', name='463')
    assert sim.bodies[0].initial_data['epoch'] == pytest.approx(mjd_from_datetime(_catalog_epoch()), abs=1e-9)


def test_astdys_body_at_another_date_raises():
    sim = Simulation(source='astdys', date='2025-01-01')
    with pytest.raises(ValueError, match='epoch'):
        sim.add_body(463, '4J-2S-1', name='463')
    assert sim.bodies == []


def test_astdys_unknown_body_raises():
    sim = Simulation(source='astdys')
    with pytest.raises(ValueError, match='not in the AstDyS catalog'):
        sim.add_body('no-such-asteroid', '4J-2S-1')


@pytest.mark.parametrize('epoch', [60000.0, 60000, '2023-02-25', DATE_60000, 60000.0 + 5e-5])
def test_element_dict_at_simulation_date_is_added(epoch):
    sim = Simulation(date=DATE_60000)
    sim.add_body({'a': 2.4, 'e': 0.2, 'inc': 0.2, 'Omega': 0.6, 'omega': 5.7, 'M': 6.1, 'epoch': epoch}, '4J-2S-1')
    assert len(sim.bodies) == 1


@pytest.mark.parametrize('epoch', [60001.0, 59999.0, 60000.0 + 2e-4, '2023-02-25 12:00', datetime.datetime(2026, 6, 9)])
def test_element_dict_at_another_epoch_raises(epoch):
    sim = Simulation(date=DATE_60000)
    with pytest.raises(ValueError, match='epoch'):
        sim.add_body({'a': 2.4, 'e': 0.2, 'inc': 0.2, 'Omega': 0.6, 'omega': 5.7, 'M': 6.1, 'epoch': epoch}, '4J-2S-1')


def test_element_dict_without_epoch_is_not_checked():
    sim = Simulation(date=DATE_60000)
    sim.add_body({'a': 2.4, 'e': 0.2, 'inc': 0.2, 'Omega': 0.6, 'omega': 5.7, 'M': 6.1}, '4J-2S-1')
    assert len(sim.bodies) == 1


def test_find_and_check_take_elements_at_simulation_date(monkeypatch):
    """find() asks Horizons for the elements at config.date, as check() does, and both get the same elements."""
    calls = []
    monkeypatch.setattr(finder_module, 'get_body_keplerian_elements', _fake_horizons(calls))
    monkeypatch.setattr(resonances.horizons, 'get_body_keplerian_elements', _fake_horizons(calls))

    date = datetime.datetime(2023, 3, 7)
    found = resonances.find(463, planets=['Jupiter', 'Saturn'], type='mmr', date='2023-03-07', source='nasa')
    checked = resonances.check(463, '4J-2S-1', date='2023-03-07', source='nasa')

    assert [d for _, d in calls] == [date, date]
    assert found.config.date == date
    assert found.bodies[0].initial_data == checked.bodies[0].initial_data
    assert found.bodies[0].initial_data['epoch'] == 60010.0


def test_cli_find_takes_elements_at_date(monkeypatch):
    """`resonances find --date=...` reaches the Horizons query with that date."""
    calls, runs = [], []
    monkeypatch.setattr(finder_module, 'get_body_keplerian_elements', _fake_horizons(calls))
    monkeypatch.setattr(Simulation, 'run', lambda self, progress=False: runs.append(self))

    result = CliRunner().invoke(
        cli, ['find', '--asteroids=401', '--date=2023-02-25', '--type=mmr', '--planets=Jupiter,Saturn', '--save=none', '--plot=none']
    )

    assert result.exit_code == 0, result.output
    assert calls == [(401, DATE_60000)]
    assert runs[0].config.date == DATE_60000
    assert runs[0].bodies[0].initial_data['epoch'] == 60000.0
