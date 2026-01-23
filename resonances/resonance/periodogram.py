from typing import Dict, List, Optional, Tuple, Any
from astropy.timeseries import LombScargle
from scipy import signal
import numpy as np

from resonances.logger import logger


class Periodogram:

    @classmethod
    def periodogram(
        cls,
        t,
        y,
        integration_time_yrs,
        Nout,
        label="body_label_unspecified",
        libration_period_min=500,
        minimum_frequency=0.00001,
        maximum_frequency=0.002,
        threshold=0.05,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[Dict[str, Any]],
    ]:
        """Calculates Lomb-Scargle periodogram for a time series.

        Parameters
        ----------
        t : list
            list of times
        y : list
            list of values (angles, axis e.t.c.)
        x_max : float
            maximum value of x-axis (e.g. integration time in years)
        Nout : int
            number of output points
        label : str
            label for the body (default is "body_label_unspecified"); used for logging.
        libration_period_min : float
            minimum libration period in years (default is 500 years)
        minimum_frequency : float
            minimum frequency to look for peaks (default is 0.00001)
        maximum_frequency : float
            maximum frequency to look for peaks (default is 0.002)
        threshold : float
            threshold for peaks (default is 0.05 - soft threshold)
        Returns
        -------
        frequency : np.ndarray
            frequency of the peaks
        power : np.ndarray
            power of the peaks
        peaks_data : dict
            dictionary with peaks data
            keys:
                position : list of tuples with left and right positions of the peaks
                peaks : list of indices of the peaks
        """
        integration_time = abs(round(integration_time_yrs))  # abs for backward integration
        fs = Nout / integration_time  # sample rate, Hz || Nout/time, i.e. 10000/100000
        """
        Do not take into account first N and last N points because of the filter applied.
        There is no previous (or following) data for them. Thus, they mess the periodogram.
        """
        points_to_cut = round(libration_period_min * fs)

        try:
            (frequency, power) = cls.lomb_scargle(
                t[points_to_cut : len(y) - points_to_cut] / (2 * np.pi),
                y[points_to_cut : len(y) - points_to_cut],
                minimum_frequency=minimum_frequency,
                maximum_frequency=maximum_frequency,
            )
            peaks_data = cls.find_peaks_with_position(frequency, power, height=threshold)
        except Exception as e:  # pragma: no cover
            logger.error(f"Error in periodogram of semi-major axis for {label}: {e}")
            logger.info(f"Configs: {minimum_frequency}, {maximum_frequency}")
            frequency, power, peaks_data = None, None, None
        return frequency, power, peaks_data

    @classmethod
    def lomb_scargle(cls, x, y, minimum_frequency=0.00001, maximum_frequency=0.002, nyquist_factor=5):
        """Calculates Lomb-Scargle periodogram for a time series.

        Parameters
        ----------
        x : list
            list of times
        y : list
            list of values (angles, axis e.t.c.)
        minimum_frequency : float, optional
            the minimum frequency to look for peaks, by default 0.00001
        maximum_frequency : float, optional
            the maximum frequency to look for peaks, by default 0.002
        nyquist_factor : int, optional
            the parameter from lomg-scargle method, by default 5

        Returns
        -------
        (frequence, power)
            Return list of frequencies among with related power.
        """
        frequency, power = LombScargle(x, y).autopower(
            nyquist_factor=nyquist_factor, minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency
        )
        return (frequency, power)

    @classmethod
    def find_peaks_with_position(cls, frequency, power, height=0.05, distance=10):
        peaks, props = signal.find_peaks(power, height=height, distance=distance, width=(None, None))
        peaks_right, peaks_left = (
            1.0 / frequency[np.rint(props['left_ips']).astype(int)],
            1.0 / frequency[np.rint(props['right_ips']).astype(int)],
        )
        peaks_position = list(zip(peaks_left, peaks_right))
        peaks_position = sorted(peaks_position, key=lambda tup: tup[0])
        return {'position': peaks_position, 'peaks': peaks}

    @classmethod
    def overlap(cls, a: Tuple[float, float], b: Tuple[float, float], delta: float = 0) -> float:
        """Check if two intervals overlap.

        Parameters
        ----------
        a : tuple
            First interval (start, end)
        b : tuple
            Second interval (start, end)
        delta : float
            Tolerance to extend intervals (default: 0)

        Returns
        -------
        float
            The amount of overlap (0 if no overlap)
        """
        return max(0, min(a[1] + delta, b[1] + delta) - max(a[0] - delta, b[0] - delta))

    @classmethod
    def overlap_list(
        cls, a_list: List[Tuple[float, float]], b_list: List[Tuple[float, float]], delta: float = 0
    ) -> List[Tuple[float, float]]:
        """Find all intervals from a_list that overlap with any interval in b_list.

        Parameters
        ----------
        a_list : list or None
            List of intervals (tuples of start, end)
        b_list : list or None
            List of intervals (tuples of start, end)
        delta : float
            Tolerance to extend intervals (default: 0)

        Returns
        -------
        list
            List of intervals from a_list that overlap with at least one interval in b_list
        """
        if not a_list or not b_list:
            return []

        arr = []
        for a_elem in a_list:
            for b_elem in b_list:
                if cls.overlap(a_elem, b_elem, delta=delta):
                    arr.append(a_elem)
                    break
        return arr
