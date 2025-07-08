import pytest
import resonances
import datetime
from resonances.data.util import convert_input_to_list, datetime_from_string


def test_convert_input_to_list():
    asteroids = 1
    expected_output = [1]
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = '2'
    expected_output = ['2']
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = [1, 2, 3]
    expected_output = [1, 2, 3]
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = ['4', '5', '6']
    expected_output = ['4', '5', '6']
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = [7, '8', 9, '10']
    expected_output = [7, '8', 9, '10']
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = []
    expected_output = []
    assert convert_input_to_list(asteroids) == expected_output

    asteroids = None
    expected_output = []
    assert convert_input_to_list(asteroids) == expected_output


def test_axis_from_mean_motion_and_back():
    axis = 1.0
    assert axis == pytest.approx(resonances.data.util.axis_from_mean_motion(resonances.data.util.mean_motion_from_axis(axis)))


def test_datetime_from_string_date_only():
    date_str = "2023-01-01"
    expected = datetime.datetime(2023, 1, 1)
    assert datetime_from_string(date_str) == expected


def test_datetime_from_string_date_time():
    date_str = "2023-01-01 13:45"
    expected = datetime.datetime(2023, 1, 1, 13, 45)
    assert datetime_from_string(date_str) == expected


def test_datetime_from_string_date_time_seconds():
    date_str = "2023-01-01 13:45:30"
    expected = datetime.datetime(2023, 1, 1, 13, 45, 30)
    assert datetime_from_string(date_str) == expected


def test_datetime_from_string_invalid_format():
    with pytest.raises(AttributeError):
        datetime_from_string("01/01/2023")


def test_datetime_from_string_datetime_input():
    dt = datetime.datetime(2023, 1, 1)
    assert datetime_from_string(dt) == dt


def test_datetime_from_string_empty():
    with pytest.raises(AttributeError):
        datetime_from_string("")
