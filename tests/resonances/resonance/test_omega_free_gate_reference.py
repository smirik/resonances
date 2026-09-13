"""Regression of the free-omega gate against four real objects.

Fixtures are the first 500 kyr of mercurius integrations, thinned to a 100 yr cadence and
gzipped (`tests/fixtures/free-omega/`). They are committed because the integrations they
come from live outside the repository, and without them nothing here is reproducible.

Three of the four are Lidov-Kozai librators whose free omega stays confined. The fourth,
591986, is the case the whole gate exists for: a Jupiter Trojan whose *osculating* omega is
locked for 30 Myr while the free one leaves confinement after 6 kyr, because what bounds
the angle is the forced eccentricity vector, not a resonance. Vinogradova (2024) studied
60 L4/L5 Trojans and found none with a librating omega, which is the external check on
this verdict.

For a co-orbital the g5..g8 basis does not strictly apply at all — the forced term is
Jupiter's own vector rotated by 60 degrees (Milani 1993; Vinogradova 2015) — so 591986 is
here as a false-positive guard, not as a measurement of its rates. `free_rho` and the free
fraction are what show the decomposition is on thin ice for it.
"""

import gzip
import pathlib

import numpy as np
import pandas as pd
import pytest

from resonances.body import Body
from resonances.lidov_kozai.lidov_kozai_resonance import LidovKozaiResonance
from resonances.resonance.classify import classify_resonance
from resonances.resonance.classify.models import ResonanceStatus
from resonances.resonance.filtering import filter_angle, wrap
from resonances.resonance.omega_free_gate import GateOutcome, apply_gate
from resonances.simulation.config import SimulationConfig

FIXTURES = pathlib.Path(__file__).parents[2] / 'fixtures' / 'free-omega'
BASELINE = 5e5


def _load(name):
    """Rebuild a Body and its Lidov-Kozai resonance from a fixture."""
    config = SimulationConfig(save=None, plot=None, save_summary=False, integration_years=BASELINE)
    with gzip.open(FIXTURES / f'{name}.csv.gz', 'rt') as handle:
        data = pd.read_csv(handle)

    body = Body()
    body.name = name
    body.times = data['times'].values * 2 * np.pi
    body.axis, body.ecc, body.inc = data['a'].values, data['e'].values, data['inc'].values
    body.Omega, body.omega = data['Omega'].values, data['omega'].values
    body.axis_filtered = filter_angle(body.axis, config)

    resonance = LidovKozaiResonance()
    body.lidov_kozai_resonances = [resonance]
    key = resonance.to_s()
    unwrapped = np.unwrap(data['LK_angle_unwrapped'].values)
    body.angles_unwrapped[key] = unwrapped
    body.angles[key] = wrap(unwrapped)
    body.angles_filtered_unwrapped[key] = filter_angle(unwrapped, config)
    body.angles_filtered[key] = wrap(body.angles_filtered_unwrapped[key])
    return body, resonance, config


def _gate(name):
    body, resonance, config = _load(name)
    result = classify_resonance(body, resonance=resonance, config=config)['result']
    elements = body.build_free_elements(config.free_elements_sampling_years, config.free_gate_mask_quantile)
    status, gate = apply_gate(result.status, elements, result.metrics.libration_period_1, config)
    return result.status, status, gate, elements


@pytest.mark.parametrize('name', ['4257', '3040', '162474'])
def test_a_real_librator_is_confirmed(name):
    classified, status, gate, elements = _gate(name)

    assert classified == ResonanceStatus.LIBRATION
    assert status == ResonanceStatus.LIBRATION
    assert gate.outcome is GateOutcome.CONFIRMED
    # varpi and Omega precess together, which is what a Lidov-Kozai lock is.
    assert elements.b_omega == pytest.approx(0.0, abs=0.1)
    assert np.degrees(elements.omega_range) < 90.0
    assert elements.rho > 1.0


def test_the_trojan_is_demoted():
    classified, status, gate, elements = _gate('591986')

    assert classified == ResonanceStatus.LIBRATION  # the osculating angle does librate
    assert status == ResonanceStatus.LIBRATION_UNCERTAIN
    assert gate.outcome is GateOutcome.KINEMATIC
    assert gate.matches == []
    # The free omega winds at about 220 arcsec/yr, nowhere near any planetary combination.
    assert elements.b_omega == pytest.approx(220.0, abs=10.0)
    # And the free circle does not enclose the origin: the angle is bounded by geometry.
    assert elements.rho < 1.0


def test_every_object_uses_the_hand_validated_basis():
    # 500 kyr merges the slow modes, and the priority convention resolves each cluster to
    # the basis that was checked independently against 10 Myr integrations.
    for name in ('4257', '3040', '162474', '591986'):
        elements = _gate(name)[3]

        assert list(elements.basis_e) == ['g5', 'g6']
        assert list(elements.basis_i) == ['s6', 'const']


@pytest.mark.parametrize('name', ['4257', '3040', '162474', '591986'])
def test_no_representative_choice_turns_a_non_librator_into_a_libration(name):
    # The cluster representative is a modelling assumption, not a free convention: fitting
    # a different member of the same cluster measurably changes the verdict on some objects.
    # What must hold is that it can only ever move the answer towards withholding, never
    # towards a false confirmation — so the priority order is a safe default, not a guess
    # that could invent a resonance.
    from itertools import product

    from resonances.secular import free_elements as fe

    body, resonance, config = _load(name)
    result = classify_resonance(body, resonance=resonance, config=config)['result']
    times = np.asarray(body.times) / (2 * np.pi)
    expected = ResonanceStatus.LIBRATION if name != '591986' else ResonanceStatus.LIBRATION_UNCERTAIN

    clusters = fe.cluster(fe.candidate_frequencies('e'), BASELINE) + fe.cluster(fe.candidate_frequencies('i'), BASELINE)
    for choice in product(*clusters):
        elements = fe.free_elements(
            times,
            body.ecc,
            body.inc,
            body.Omega,
            body.omega,
            sampling_years=config.free_elements_sampling_years,
            mask_quantile=config.free_gate_mask_quantile,
            representatives=choice,
        )
        status = apply_gate(result.status, elements, result.metrics.libration_period_1, config)[0]

        assert status in (expected, ResonanceStatus.UNCERTAIN), f'{name} fitted {choice} -> {status}'
