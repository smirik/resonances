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

    # ============================================================================
    # Angle analysis utilities (formerly in libration class)
    # ============================================================================

    @classmethod
    def shift(cls, angle):
        """Shift angles > π by subtracting 2π."""
        tmp = np.array(angle, copy=True, dtype=float)
        tmp[tmp > np.pi] -= 2 * np.pi
        return tmp

    @classmethod
    def is_pure(cls, y):
        """Check if consecutive angle differences never exceed π."""
        prev = y[0]
        for elem in y:
            if abs(elem - prev) > np.pi:
                return False
            prev = elem
        return True

    @classmethod
    def is_pure_apocentric(cls, y):
        """
        Enhanced pure libration detection for apocentric libration around 0/2π.
        Checks if the angle stays within bounds when accounting for
        the 2π periodicity of angles.
        """
        if len(y) <= 1:
            return True

        y_normalized = np.array(y) % (2 * np.pi)
        min_angle = np.min(y_normalized)
        max_angle = np.max(y_normalized)

        has_near_zero = np.any(y_normalized <= np.pi / 2)
        has_near_2pi = np.any(y_normalized >= 3 * np.pi / 2)

        if has_near_zero and has_near_2pi:
            # Wrapping case: find the largest gap to compute the true span
            sorted_angles = np.sort(y_normalized)
            gaps = np.diff(sorted_angles)
            wraparound_gap = (sorted_angles[0] + 2 * np.pi) - sorted_angles[-1]
            all_gaps = np.append(gaps, wraparound_gap)

            largest_gap = np.max(all_gaps)
            total_span = 2 * np.pi - largest_gap
            return total_span <= np.pi
        else:
            angle_range = max_angle - min_angle
            return angle_range <= np.pi

    @classmethod
    def is_apocentric_libration(cls, y, threshold=1.5):
        """
        Detect apocentric libration by checking if most values are near 0 or 2π.
        """
        if len(y) == 0:
            return False

        y_normalized = np.array(y) % (2 * np.pi)
        near_zero = np.sum(y_normalized <= threshold)
        near_2pi = np.sum(y_normalized >= (2 * np.pi - threshold))

        total_near_boundary = near_zero + near_2pi
        boundary_fraction = total_near_boundary / len(y)
        dominant_boundary = max(near_zero, near_2pi) / len(y) > 0.4

        return boundary_fraction > 0.6 or dominant_boundary

    @classmethod
    def pure(cls, y):
        """Check if an angle series represents pure libration."""
        if cls.is_pure(y):
            return True
        if cls.is_pure(cls.shift(y)):
            return True
        if cls.is_apocentric_libration(y):
            if cls.is_pure_apocentric(y):
                return True
        return False

    @classmethod
    def monotony_estimation(cls, data, crit=np.pi) -> float:
        """Estimate the fraction of "decreasing" points, ignoring jumps larger than crit."""
        if len(data) <= 1:
            return 0.0
        num = 0
        prev = data[0]
        for elem in data:
            if prev - elem > crit:
                prev = elem
                continue
            if elem - prev > crit:
                num += 1
                prev = elem
                continue
            if elem < prev:
                num += 1
            prev = elem
        return num / (len(data) - 1)

    @classmethod
    def find_breaks(cls, x, y, break_value=np.pi):
        """
        Find continuous breaks in a dataset.

        Parameters
        ----------
        x : list
            Time array
        y : list
            Data array
        break_value : float
            Threshold for detecting a break (default: π)

        Returns
        -------
        list
            [break_times, directions, prev_values, curr_values]
        """
        prev = y[0]
        res = [[], [], [], []]
        for i, elem in enumerate(y):
            if abs(elem - prev) > break_value:
                res[0].append(x[i])
                direction = 1 if elem > prev else -1
                res[1].append(direction)
                res[2].append(prev)
                res[3].append(elem)
            prev = elem
        return res

    @classmethod
    def circulation(cls, x, y):
        """
        Find libration periods by detecting circulation breaks.

        Parameters
        ----------
        x : list
            Time array
        y : list
            Angle data

        Returns
        -------
        list
            [starts, stops, lengths] of libration periods
        """
        breaks = cls.find_breaks(x, y)

        if 0 == len(breaks[1]):  # full interval is a libration
            return [[x[0]], [x[len(y) - 1]], [x[-1] - x[0]]]

        librations = [[], [], []]  # start, stop, length

        breaks_diff = np.diff(breaks[0])

        libration_start = x[0]
        libration_length = breaks[0][0] - x[0]
        prev_direction = breaks[1][0]

        if 1 == len(breaks[1]):  # pragma: no cover
            librations[0].append(breaks[0][0])
            librations[1].append(x[-1])
            librations[2].append(libration_length + (x[-1] - breaks[0][0]))
            return librations

        for i in range(1, len(breaks[0])):
            curr_direction = breaks[1][i]
            libration_length += breaks_diff[i - 1]
            if curr_direction == prev_direction:  # circulation found
                librations[0].append(libration_start)
                librations[1].append(breaks[0][i])
                librations[2].append(libration_length)

                libration_start = breaks[0][i]
                libration_length = 0.0
            if i == (len(breaks[0]) - 1):
                if breaks[0][i] == x[-1]:
                    if curr_direction != prev_direction:
                        librations[0].append(libration_start)
                        librations[1].append(x[-1])
                        librations[2].append(libration_length + (x[-1] - breaks[0][i]))
                else:
                    librations[0].append(breaks[0][i])
                    librations[1].append(x[-1])
                    librations[2].append(libration_length + (x[-1] - breaks[0][i]))

            prev_direction = curr_direction
        return librations

    @classmethod
    def circulation_metrics(cls, librations):
        """Compute max libration length and number of periods from circulation results."""
        max_libration_length = max(librations[2])
        num_libration_periods = len(librations[0])
        return {'num_libration_periods': num_libration_periods, 'max_libration_length': max_libration_length}

    @classmethod
    def resolve(cls, pure, overlapping, max_libration_length, libration_period_critical, monotony, libration_monotony_critical):
        """Resolve resonance status based on periodogram overlap and libration metrics.

        Returns
        -------
        int
            Status code: 2 (pure libration), 1 (transient), 0 (non-resonant),
            -1 (uncertain transient), -2 (uncertain libration)
        """
        if pure and (len(overlapping) > 0):
            return 2  # pure libration
        elif pure:
            # seems to be pure but libration periods of axis and resonant angle are different
            return -2
        elif (len(overlapping) > 0) and (max_libration_length > libration_period_critical):
            return 1  # transient resonance
        elif (
            (max_libration_length > libration_period_critical)
            and (monotony >= libration_monotony_critical[0])
            and (monotony <= libration_monotony_critical[1])
        ):
            # Long stable period and acceptable monotony, but no overlapping peaks
            return -1
        return 0
