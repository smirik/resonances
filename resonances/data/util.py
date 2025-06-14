from typing import Union, List
from resonances.data import const
import datetime


def convert_input_to_list(asteroids: Union[int, str, List[Union[int, str]]]) -> List[str]:
    if isinstance(asteroids, str) or isinstance(asteroids, int):
        asteroids = [asteroids]
    elif asteroids is None:
        asteroids = []
    return asteroids


def axis_from_mean_motion(mean_motion):
    return (const.K / mean_motion) ** (2.0 / 3)


def mean_motion_from_axis(a):
    return const.K / a ** (3.0 / 2)


def datetime_from_string(date: Union[str, datetime.datetime]) -> datetime.datetime:
    """
    Convert string to datetime object.
    This function is based on the REBOUND package date conversion utilities.
    It converts a date string to a datetime object using various format patterns.
    Args:
        date (Union[str, datetime.datetime]): Input date either as string or datetime object.
        Accepted string formats are:
        - "YYYY-MM-DD"
        - "YYYY-MM-DD HH:MM"
        - "YYYY-MM-DD HH:MM:SS"
    Returns:
        datetime.datetime: Converted datetime object
    Raises:
        AttributeError: If the input string format doesn't match any of the accepted formats
    Example:
        >>> datetime_from_string("2023-01-01")
        datetime.datetime(2023, 1, 1, 0, 0)
        >>> datetime_from_string("2023-01-01 13:45")
        datetime.datetime(2023, 1, 1, 13, 45)
        >>> datetime_from_string("2023-01-01 13:45:05")
        datetime.datetime(2023, 1, 1, 13, 45, 05)
    """

    if isinstance(date, datetime.datetime):
        return date
    elif isinstance(date, str):
        formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"]
        for f in formats:
            try:
                tmp = datetime.datetime.strptime(date, f)
                return tmp
            except ValueError:
                continue
        raise AttributeError("An error occured while calculating the date. Use one of ".join(formats))
