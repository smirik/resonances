import pandas as pd
import numpy as np
import pytest
import resonances


def test_pure():
    df = pd.read_csv('tests/resonance/fixtures/pure.csv')
    flag = resonances.libration.has_pure_libration(df['angle'])
    assert flag is True
    flag = resonances.libration.pure(df['angle'])
    assert flag is True

    df = pd.read_csv('tests/resonance/fixtures/apocentric.csv')
    flag = resonances.libration.pure(df['angle'])
    assert flag is False
    flag = resonances.libration.has_pure_libration(df['angle'])
    assert flag is True

    # test has_pure_libration and rewrite it after new normal example


def test_shift():
    data = [2 * np.pi - 0.01, 0.01, 4.0]
    shift = resonances.libration.shift(data)
    assert shift[0] == pytest.approx((data[0] - 2 * np.pi))
    assert shift[1] == pytest.approx(0.01)
    assert shift[2] == pytest.approx(data[2] - 2 * np.pi)


def test_libration():
    df = pd.read_csv('tests/resonance/fixtures/pure.csv')
    data = resonances.libration.find(
        df['times'],
        df['angle'],
        len(df),
        resonances.config.get('libration.start'),
        resonances.config.get('libration.stop'),
        resonances.config.get('libration.num_freqs'),
        resonances.config.get('libration.periodogram.critical'),
        resonances.config.get('libration.density.critical'),
    )
    assert data['flag'] is True
    assert data['status'] is 2

    df = pd.read_csv('tests/resonance/fixtures/transient.csv')
    data = resonances.libration.find(
        df['times'],
        df['angle'],
        len(df),
        resonances.config.get('libration.start'),
        resonances.config.get('libration.stop'),
        resonances.config.get('libration.num_freqs'),
        resonances.config.get('libration.periodogram.critical'),
        resonances.config.get('libration.density.critical'),
    )
    assert data['flag'] is True
    assert data['status'] is 1

    df = pd.read_csv('tests/resonance/fixtures/fp.csv')
    data = resonances.libration.find(
        df['times'],
        df['angle'],
        len(df),
        resonances.config.get('libration.start'),
        resonances.config.get('libration.stop'),
        resonances.config.get('libration.num_freqs'),
        resonances.config.get('libration.periodogram.critical'),
        resonances.config.get('libration.density.critical'),
    )
    assert data['flag'] is False
    assert data['status'] is 0


def test_resolve():
    assert 2 == resonances.libration.resolve(True, 100, 200, 100, 200)
    assert 2 == resonances.libration.resolve(True, 200, 100, 200, 100)
    assert 0 == resonances.libration.resolve(False, 100, 200, 300, 200)
    assert 0 == resonances.libration.resolve(False, 200, 100, 200, 300)
    assert 1 == resonances.libration.resolve(False, 200, 100, 300, 200)
