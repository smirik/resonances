from resonances.resonance.libration import libration
from resonances.config import config
import pandas as pd
import numpy as np
import pytest


def test_pure():
    df = pd.read_csv('tests/resonance/fixtures/pure.csv')
    flag = libration.has_pure_libration(df['angle'])
    assert flag is True
    flag = libration.pure(df['angle'])
    assert flag is True

    df = pd.read_csv('tests/resonance/fixtures/apocentric.csv')
    flag = libration.pure(df['angle'])
    assert flag is False
    flag = libration.has_pure_libration(df['angle'])
    assert flag is True

    # test has_pure_libration and rewrite it after new normal example


def test_shift():
    data = [2 * np.pi - 0.01, 0.01, 4.0]
    shift = libration.shift(data)
    assert shift[0] == pytest.approx((data[0] - 2 * np.pi))
    assert shift[1] == pytest.approx(0.01)
    assert shift[2] == pytest.approx(data[2] - 2 * np.pi)


def test_libration():
    df = pd.read_csv('tests/resonance/fixtures/pure.csv')
    data = libration.libration(
        df['times'], df['angle'], len(df), config.get('libration.start'), config.get('libration.stop'), config.get('libration.Nfreq')
    )
    assert data['flag'] is True
    assert libration.has(df['times'], df['angle'], len(df)) is True
    assert 2 == libration.status(df['times'], df['angle'], len(df))

    df = pd.read_csv('tests/resonance/fixtures/transient.csv')
    data = libration.libration(
        df['times'], df['angle'], len(df), config.get('libration.start'), config.get('libration.stop'), config.get('libration.Nfreq')
    )
    assert data['flag'] is True
    assert libration.has(df['times'], df['angle'], len(df)) is True
    assert 1 == libration.status(df['times'], df['angle'], len(df))

    df = pd.read_csv('tests/resonance/fixtures/fp.csv')
    data = libration.libration(
        df['times'], df['angle'], len(df), config.get('libration.start'), config.get('libration.stop'), config.get('libration.Nfreq')
    )
    # assert data['flag'] is False
    # assert libration.has(df['times'], df['angle'], len(df)) is False
    # assert 0 == libration.status(df['times'], df['angle'], len(df))
