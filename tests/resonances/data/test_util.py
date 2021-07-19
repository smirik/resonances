import resonances.data.util as util
import pytest


def test_axis_from_mean_motion_and_back():
    axis = 1.0
    assert axis == pytest.approx(util.axis_from_mean_motion(util.mean_motion_from_axis(axis)))
    assert 1 == 1
