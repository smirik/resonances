"""`resonances dry-run` and the static check of find/check before integrating.

The simulation is built from explicit elements (no catalogue, no Horizons): the patched find/check receive the
arguments the CLI derived, and return a real Simulation, so the report is the package's own.
"""

import json
import math
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import resonances
from resonances.cli import cli

UQ = dict(a=1.0940602, e=0.0155969, inc=0.4331147985225309, Omega=3.687738465129981, omega=0.13104333449701386, M=6.077459508522661)


def build(calls):
    def factory(**kwargs):
        calls.append(kwargs)
        keep = {
            key: kwargs[key]
            for key in ('integration_years', 'output_step', 'Nout', 'integrator', 'dryrun_fair', 'save', 'plot')
            if key in kwargs
        }
        sim = resonances.Simulation(name='cli-dryrun', date='2026-09-14', save_summary=False, **keep)
        sim.add_body(UQ, ['7E+2M-9+0+0+0'], name='1999 UQ')
        return sim

    return factory


def test_dry_run_command_check_arguments_and_files(tmp_path):
    calls = []
    out = tmp_path / 'dry'
    with patch('resonances.cli.main.resonances.check', side_effect=build(calls)):
        result = CliRunner().invoke(
            cli,
            [
                'dry-run',
                '--asteroids=1',
                '--resonances=7E+2M-9+0+0+0',
                '--years=20000',
                '--Nout=4000',
                '--integrator=ias15',
                f'--out={out}',
            ],
        )
    assert result.exit_code == 0, result.output
    assert calls[0]['resonance'] == '7E+2M-9+0+0+0' and 'resonances' not in calls[0]
    assert calls[0]['integration_years'] == 20000 and calls[0]['save'] is None
    assert 'aliased_unambiguous' in result.output
    run = json.loads((out / 'dryrun_run.json').read_text())
    assert run['step_configured_yr'] == pytest.approx(5.0012, abs=1e-4)


def test_dry_run_command_auto_step_and_fair_through_find():
    calls = []
    with patch('resonances.cli.main.resonances.find', side_effect=build(calls)):
        result = CliRunner().invoke(cli, ['dry-run', '--asteroids=1', '--years=20000', '--step=auto', '--fair'])
    assert result.exit_code == 0, result.output
    assert calls[0]['output_step'] == 'auto' and calls[0]['dryrun_fair'] == 'required'
    # FAIR at P = 1.144 yr: 0.238 P = 0.272 yr -> 0.25 yr
    assert 'auto 0.25 yr (fair)' in result.output


def test_check_writes_the_static_warnings_and_applies_auto_before_running():
    calls, sims, received = [], [], []

    def factory(**kwargs):
        received.append(kwargs)
        sim = build(calls)(output_step='auto', integration_years=20000)
        # the integration itself is stubbed: Simulation.run starts with the static dry-run, then integrates
        sim._run_single = lambda progress=False: sims.append((sim.config.Nout, sim.config.output_step))
        return sim

    with patch('resonances.cli.main.resonances.check', side_effect=factory), patch('resonances.simulation.dry_run.logger.warning') as warn:
        result = CliRunner().invoke(cli, ['check', '--asteroids=1', '--resonances=7E+2M-9+0+0+0'])
    assert result.exit_code == 0, result.output
    assert received[0]['resonance'] == '7E+2M-9+0+0+0' and 'resonances' not in received[0]  # finder.check's parameter
    # 0.5 yr over 20 kyr, set before the integration: below P_orb / 2 = 1.0940602^1.5 / 2 = 0.572 yr (short periods)
    assert sims == [(40001, 0.5)]
    assert any('dry-run' in str(call.args[0]) for call in warn.call_args_list)
    assert math.isclose(0.5, sims[0][1])
