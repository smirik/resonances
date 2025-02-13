import pytest
import resonances


def test_axis_from_mean_motion_and_back():
    axis = 1.0
    assert axis == pytest.approx(resonances.data.util.axis_from_mean_motion(resonances.data.util.mean_motion_from_axis(axis)))
    assert 1 == 1
