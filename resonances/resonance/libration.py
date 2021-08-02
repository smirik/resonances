import numpy as np
from scipy import signal
from astropy.timeseries import LombScargle

import resonances.config


class libration:
    @classmethod
    def shift(cls, angle, distance=0):
        tmp = np.array(angle, copy=True)
        for i, elem in enumerate(tmp):
            if elem > np.pi:
                tmp[i] = tmp[i] - 2 * np.pi
        return tmp

    @classmethod
    def is_pure(cls, y):
        prev = y[0]
        num = 0
        for elem in y:
            num += 1
            if abs(elem - prev) > np.pi:
                return False
            prev = elem
        return True

    @classmethod
    def pure(cls, y):  # not working for librations that are not around 0 or +/- np.pi
        flag1 = cls.is_pure(y)
        if flag1:
            return True
        flag2 = cls.is_pure(cls.shift(y))
        if flag2:
            return True
        return False

    @classmethod
    def monotony_estimation(cls, data, crit=np.pi):
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
                num += 1  # num of "decreasing" points
            prev = elem
        return num / (len(data) - 1)

    @classmethod
    def find_breaks(cls, x, y, break_value=np.pi):
        """
        This function finds continuos breaks in a given dataset and return these breaks among with additional data.

        Parameters
        ----------

        x : list
            one-dimensional array of time
        y : list
            one-dimensional array of data where to find breaks
        break_value : float
            Used to determine a break. If the absolute diff between the previous one and
            the current is more than break_value, then there is a break. (Default: np.pi)

        Returns
        -------
        list
            Returns a multidimensional array:
            res[0] - the time when the break happened,
            res[1] - the direction of the break (1 if it intersects the top line and thus elem>prev, -1 otherwise),
            res[2] - values of the elements prior to the breaks,
            res[3] - the values in the break points.
        """
        prev = y[0]
        res = [[], [], [], []]  # break point, direction (1 or -1), prev, curr
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
        Find and returns libration periods with some additional data.

        Parameters
        ----------

        x : list
            one-dimensional array of time
        y : list
            Input data to find breaks.

        Returns
        -------

        list
            Returns a multidimensional list.
            res[0] - the start of a libration period,
            res[1] - the end of a libration period,
            res[2] - the length of the given libration period.

        """
        breaks = cls.find_breaks(x, y)

        if 0 == len(breaks[1]):  # full interval is a libration
            return [[x[0]], [x[len(y) - 1]], [x[-1] - x[0]]]

        breaks_diff = np.diff(breaks[0])

        librations = [[], [], []]  # start, stop, length

        libration_start = x[0]
        libration_length = breaks[0][0] - x[0]
        prev_direction = breaks[1][0]

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
                # If the last break has happened before the end, then add one more interval of libration.
                if breaks[0][i] == x[-1]:
                    # flush data for libration period. If there is a circulation, it is already flushed.
                    if curr_direction != prev_direction:
                        librations[0].append(libration_start)
                        librations[1].append(x[-1])
                        librations[2].append(libration_length + (x[-1] - breaks[0][i]))
                else:
                    # append last libration (because break is not in the last point)
                    librations[0].append(breaks[0][i])
                    librations[1].append(x[-1])
                    librations[2].append(libration_length + (x[-1] - breaks[0][i]))

            prev_direction = curr_direction
        return librations

    @classmethod
    def circulation_metrics(cls, librations):
        max_libration_length = max(librations[2])
        num_libration_periods = len(librations[0])

        return {'num_libration_periods': num_libration_periods, 'max_libration_length': max_libration_length}

    @classmethod
    def overlap(cls, a, b, delta=0):
        return max(0, min(a[1] + delta, b[1] + delta) - max(a[0] - delta, b[0] - delta))

    @classmethod
    def overlap_list(cls, a_list, b_list, delta=0):
        arr = []
        for a_elem in a_list:
            for b_elem in b_list:
                if cls.overlap(a_elem, b_elem, delta=delta):
                    arr.append(a_elem)
                    break
        return arr

    @classmethod
    def periodogram(cls, x, y, minimum_frequency=0.00001, maximum_frequency=0.002, nyquist_factor=5):
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
    def butter_lowpass_filter(cls, data, cutoff, fs, order, nyq):
        normal_cutoff = cutoff / nyq
        # Get the filter coefficients
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.filtfilt(b, a, data, method="gust")
        return y

    @classmethod
    def body(cls, sim, body: resonances.Body):
        pure = resonances.libration.pure(body.angle)

        librations = resonances.libration.circulation(sim.times / (2 * np.pi), body.angle)
        libration_metrics = resonances.libration.circulation_metrics(librations)
        monotony = resonances.libration.monotony_estimation(body.angle)

        integration_time = round(sim.tmax / (2 * np.pi))
        fs = sim.Nout / integration_time  # sample rate, Hz || Nout/time, i.e. 10000/100000
        cutoff = sim.oscillations_cutoff  # should be a little bit more than needed
        nyq = 0.5 * fs  # Nyquist Frequency
        order = sim.oscillations_filter_order  # polynom order
        """
        Do not take into account first N and last N points because of the filter applied.
        There is no previous (or following) data for them. Thus, they mess the periodogram.
        """
        points_to_cut = round(sim.libration_period_min * fs)

        angle_filtered = cls.butter_lowpass_filter(body.angle, cutoff, fs, order, nyq)
        (frequency, power) = resonances.libration.periodogram(
            sim.times[points_to_cut : len(angle_filtered) - points_to_cut] / (2 * np.pi),
            angle_filtered[points_to_cut : len(angle_filtered) - points_to_cut],
            minimum_frequency=sim.periodogram_frequency_min,
            maximum_frequency=sim.periodogram_frequency_max,
        )
        axis_filtered = cls.butter_lowpass_filter(body.axis, cutoff, fs, order, nyq)
        (axis_frequency, axis_power) = resonances.libration.periodogram(
            sim.times[points_to_cut : len(axis_filtered) - points_to_cut] / (2 * np.pi),
            axis_filtered[points_to_cut : len(axis_filtered) - points_to_cut],
            minimum_frequency=sim.periodogram_frequency_min,
            maximum_frequency=sim.periodogram_frequency_max,
        )

        angle_peaks_data = cls.find_peaks_with_position(frequency, power, height=sim.periodogram_soft)
        axis_peaks_data = cls.find_peaks_with_position(axis_frequency, axis_power, height=sim.periodogram_soft)
        overlapping = cls.overlap_list(angle_peaks_data['position'], axis_peaks_data['position'], delta=0)

        body.status = cls.resolve(
            pure,
            overlapping,
            libration_metrics['max_libration_length'],
            sim.libration_period_critical,
            monotony,
            sim.libration_monotony_critical,
        )

        if sim.shall_save_body(body):
            body.librations = librations
            body.libration_metrics = libration_metrics
            body.libration_pure = pure

            body.periodogram_frequency = frequency
            body.periodogram_power = power
            body.periodogram_peaks = angle_peaks_data

            body.angle_filtered = angle_filtered
            body.axis_filtered = axis_filtered
            body.axis_periodogram_frequency = axis_frequency
            body.axis_periodogram_power = axis_power
            body.axis_periodogram_peaks = axis_peaks_data

            body.periodogram_peaks_overlapping = overlapping

            body.monotony = monotony

        return body.status

    @classmethod
    def resolve(cls, pure, overlapping, max_libration_length, libration_period_critical, monotony, libration_monotony_critical):
        status = 0
        if pure and (len(overlapping) > 0):
            status = 2  # pure libration
        elif pure:
            # seems to be pure but libration periods of axis and resonant angle are different
            # need manual check
            status = -2
        elif (len(overlapping) > 0) and (max_libration_length > libration_period_critical):
            status = 1  # transient resonance
        elif (
            (max_libration_length > libration_period_critical)
            and (monotony >= libration_monotony_critical[0])
            and (monotony <= libration_monotony_critical[1])
        ):
            # Looks like chaotic but has long stable period and acceptable monotony
            # need to verify manually
            status = -1
        # No resonance
        return status
