from typing import Union, List
from resonances.data import const
import datetime
import numpy as np


def static_init(cls):
    if getattr(cls, "static_init", None):
        cls.static_init()
    return cls


def convert_input_to_list(asteroids) -> List[Union[int, str]]:
    """The asteroids of find/check as a list: one identifier, a list, or a selection (DataFrame) of
    `resonances.mpcorb.query` (its numbers, else designations) or of astdys (its 'name' column)."""
    if isinstance(asteroids, str) or isinstance(asteroids, int):
        asteroids = [asteroids]
    elif asteroids is None:
        asteroids = []
    elif hasattr(asteroids, 'columns'):
        if 'packed' in asteroids.columns:
            from resonances.mpcorb import identifiers

            return identifiers(asteroids)
        if 'name' in asteroids.columns:
            return asteroids['name'].tolist()
        raise ValueError("A selection of asteroids is a DataFrame of resonances.mpcorb.query or of astdys (a 'name' column)")
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
        A naive datetime is returned as it is; a timezone-aware one as its naive UTC value (the package's dates
        are naive, and REBOUND asks Horizons for the calendar value without its offset).
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
        if date.tzinfo is not None:  # one instant everywhere: the planets' query, the clones' transport, the MJD
            return date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
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


MJD_ZERO = datetime.datetime(1858, 11, 17)


def mjd_from_datetime(date: Union[str, datetime.datetime]) -> float:
    """Modified Julian Date of `date` (naive datetimes are taken as they are, aware ones in UTC)."""
    date = datetime_from_string(date)
    if date.tzinfo is not None:
        date = date.astimezone(datetime.timezone.utc).replace(tzinfo=None)
    return (date - MJD_ZERO).total_seconds() / 86400.0


def time_ordered(times, *series):
    """`times` and `series` in increasing time, as a tuple.

    A backward integration (tmax < 0) samples decreasing times; spans, sampling steps, sliding windows
    and spectra are then computed on the reversed views, so both directions give the same numbers.
    Arrays in time order come back as they are (no copy); None passes through.
    """
    times = np.asarray(times)
    if len(times) < 2 or times[-1] >= times[0]:
        return (times, *(None if s is None else np.asarray(s) for s in series))
    return (times[::-1], *(None if s is None else np.asarray(s)[::-1] for s in series))
