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
    def is_pure_apocentric(cls, y):
        """
        Enhanced pure libration detection for apocentric libration around 0/2π.
        This method checks if the angle stays within bounds when accounting for
        the 2π periodicity of angles.
        """
        if len(y) <= 1:
            return True

        # For apocentric libration, we need to handle the wrapping more carefully
        # Convert angles to a consistent range and track the cumulative movement
        y_normalized = np.array(y) % (2 * np.pi)

        # Check if the excursion pattern is consistent with libration
        # Calculate the range of motion, accounting for wrapping
        min_angle = np.min(y_normalized)
        max_angle = np.max(y_normalized)

        # For apocentric libration around 2π/0, we expect either:
        # 1. All values clustered around 0 (small range near 0)
        # 2. All values clustered around 2π (small range near 2π)
        # 3. Values spanning the 0/2π boundary (wrapping case)

        # Case 3: Check for wrapping (values both near 0 and near 2π)
        has_near_zero = np.any(y_normalized <= np.pi / 2)  # within π/2 of 0
        has_near_2pi = np.any(y_normalized >= 3 * np.pi / 2)  # within π/2 of 2π

        if has_near_zero and has_near_2pi:
            # This is the wrapping case - need to check if total span is < π
            # when accounting for wrapping

            # Compute wrapped range by finding the largest gap
            sorted_angles = np.sort(y_normalized)
            gaps = np.diff(sorted_angles)
            # Add the wraparound gap
            wraparound_gap = (sorted_angles[0] + 2 * np.pi) - sorted_angles[-1]
            all_gaps = np.append(gaps, wraparound_gap)

            largest_gap = np.max(all_gaps)
            total_span = 2 * np.pi - largest_gap

            return total_span <= np.pi  # Libration if span <= π
        else:
            # Non-wrapping case: check if range is reasonable for libration
            angle_range = max_angle - min_angle
            return angle_range <= np.pi

    @classmethod
    def is_apocentric_libration(cls, y, threshold=1.5):
        """
        Detect apocentric libration by checking if most values are near 0 or 2π.
        This specifically handles cases where the libration center is around the
        0/2π boundary.
        """
        if len(y) == 0:
            return False

        # Convert to [0, 2π] range
        y_normalized = np.array(y) % (2 * np.pi)

        # Count points near 0 or 2π (within threshold of the boundaries)
        near_zero = np.sum(y_normalized <= threshold)
        near_2pi = np.sum(y_normalized >= (2 * np.pi - threshold))

        # Check if most points (>60%) are near the boundaries
        # Also check for concentration pattern - most points should be near one boundary
        total_near_boundary = near_zero + near_2pi
        boundary_fraction = total_near_boundary / len(y)

        # Additional check: if there's a clear concentration around one boundary
        dominant_boundary = max(near_zero, near_2pi) / len(y) > 0.4

        return boundary_fraction > 0.6 or dominant_boundary

    @classmethod
    def pure(cls, y):
        flag1 = cls.is_pure(y)
        if flag1:
            return True

        flag2 = cls.is_pure(cls.shift(y))
        if flag2:
            return True

        if cls.is_apocentric_libration(y):
            # For apocentric libration, use the enhanced pure detection
            flag3 = cls.is_pure_apocentric(y)
            if flag3:
                return True

        return False

    @classmethod
    def monotony_estimation(cls, data, crit=np.pi) -> float:
        if len(data) <= 1:  # it can be the case for testing purposes mostly
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
        # Ensure normalized cutoff frequency is valid for Butterworth filter (0 < Wn < 1)
        if normal_cutoff >= 1.0:
            # For secular resonances with very long integration times, adjust cutoff
            normal_cutoff = 0.99  # Use maximum allowable value
            resonances.logger.warning(
                f"Cutoff frequency ({cutoff}) >= Nyquist frequency ({nyq}). "
                f"Adjusting normalized cutoff to {normal_cutoff} for filter stability."
            )
        elif normal_cutoff <= 0.0:
            normal_cutoff = 0.01  # Use minimum allowable value
            resonances.logger.warning(
                f"Cutoff frequency ({cutoff}) <= 0. " f"Adjusting normalized cutoff to {normal_cutoff} for filter stability."
            )

        # Get the filter coefficients
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
        y = signal.filtfilt(b, a, data, method="gust")
        return y

    @classmethod
    def body(cls, sim, body: resonances.Body):
        integration_time = abs(round(sim.config.tmax / (2 * np.pi)))  # abs for backward integration
        fs = sim.config.Nout / integration_time  # sample rate, Hz || Nout/time, i.e. 10000/100000
        cutoff = sim.config.oscillations_cutoff  # should be a little bit more than needed
        nyq = 0.5 * fs  # Nyquist Frequency
        order = sim.config.oscillations_filter_order  # polynom order
        """
        Do not take into account first N and last N points because of the filter applied.
        There is no previous (or following) data for them. Thus, they mess the periodogram.
        """
        points_to_cut = round(sim.config.libration_period_min * fs)

        axis_filtered = cls.butter_lowpass_filter(body.axis, cutoff, fs, order, nyq)
        try:
            (axis_frequency, axis_power) = resonances.libration.periodogram(
                sim.times[points_to_cut : len(axis_filtered) - points_to_cut] / (2 * np.pi),
                axis_filtered[points_to_cut : len(axis_filtered) - points_to_cut],
                minimum_frequency=sim.config.periodogram_frequency_min,
                maximum_frequency=sim.config.periodogram_frequency_max,
            )
            axis_peaks_data = cls.find_peaks_with_position(axis_frequency, axis_power, height=sim.config.periodogram_soft)
        except Exception as e:  # pragma: no cover
            resonances.logger.error(f"Error in periodogram of semi-major axis for {body.name}: {e}")
            axis_frequency, axis_power, axis_peaks_data = None, None, None

        body.axis_filtered = axis_filtered
        body.axis_periodogram_frequency = axis_frequency
        body.axis_periodogram_power = axis_power
        body.axis_periodogram_peaks = axis_peaks_data

        try:
            (eccentricity_frequency, eccentricity_power) = resonances.libration.periodogram(
                sim.times[points_to_cut : len(body.ecc) - points_to_cut] / (2 * np.pi),
                body.ecc[points_to_cut : len(body.ecc) - points_to_cut],
                minimum_frequency=sim.config.periodogram_frequency_min,
                maximum_frequency=sim.config.periodogram_frequency_max,
            )
            eccentricity_peaks_data = cls.find_peaks_with_position(
                eccentricity_frequency, eccentricity_power, height=sim.config.periodogram_soft
            )
        except Exception as e:  # pragma: no cover
            resonances.logger.error(f"Error in periodogram of eccentricity for {body.name}: {e}")
            eccentricity_frequency, eccentricity_power, eccentricity_peaks_data = None, None, None

        body.eccentricity_periodogram_frequency = eccentricity_frequency
        body.eccentricity_periodogram_power = eccentricity_power
        body.eccentricity_periodogram_peaks = eccentricity_peaks_data

        all_resonances = body.mmrs + body.secular_resonances
        # for mmr in body.mmrs:
        for resonance in all_resonances:
            pure = resonances.libration.pure(body.angle(resonance))

            librations = resonances.libration.circulation(sim.times / (2 * np.pi), body.angle(resonance))
            libration_metrics = resonances.libration.circulation_metrics(librations)
            monotony = resonances.libration.monotony_estimation(body.angle(resonance))

            try:
                angle_filtered = cls.butter_lowpass_filter(body.angle(resonance), cutoff, fs, order, nyq)
                (frequency, power) = resonances.libration.periodogram(
                    sim.times[points_to_cut : len(angle_filtered) - points_to_cut] / (2 * np.pi),
                    angle_filtered[points_to_cut : len(angle_filtered) - points_to_cut],
                    minimum_frequency=sim.config.periodogram_frequency_min,
                    maximum_frequency=sim.config.periodogram_frequency_max,
                )

                angle_peaks_data = cls.find_peaks_with_position(frequency, power, height=sim.config.periodogram_soft)
                overlapping = cls.overlap_list(angle_peaks_data['position'], axis_peaks_data['position'], delta=0)
            except Exception as e:  # pragma: no cover
                resonances.logger.error(f"Error in periodogram for {body.name} and {resonance.to_s()}: {e}")
                frequency, power, angle_peaks_data, angle_filtered, overlapping = None, None, None, None, []

            body.statuses[resonance.to_s()] = cls.resolve(
                resonance,
                pure,
                overlapping,
                libration_metrics['max_libration_length'],
                sim.config.libration_period_critical,
                monotony,
                sim.config.libration_monotony_critical,
            )

            body.librations[resonance.to_s()] = librations
            body.libration_metrics[resonance.to_s()] = libration_metrics
            body.libration_pure[resonance.to_s()] = pure

            body.periodogram_frequency[resonance.to_s()] = frequency
            body.periodogram_power[resonance.to_s()] = power
            body.periodogram_peaks[resonance.to_s()] = angle_peaks_data

            body.angles_filtered[resonance.to_s()] = angle_filtered
            body.periodogram_peaks_overlapping[resonance.to_s()] = overlapping

            body.monotony[resonance.to_s()] = monotony

        return body.statuses

    @classmethod
    def resolve(cls, resonance, pure, overlapping, max_libration_length, libration_period_critical, monotony, libration_monotony_critical):
        from resonances.resonance.secular import SecularResonance

        status = 0

        # Special logic for SecularResonance
        if isinstance(resonance, SecularResonance):
            if pure:
                status = 2
            elif max_libration_length > libration_period_critical:
                # For secular resonances, we need more nuanced classification
                # Check if this is wide/chaotic libration that should still be considered trapped

                # Very long stability periods (>5x critical) suggest strong trapping
                # even if not perfectly pure
                stability_factor = max_libration_length / libration_period_critical

                # Monotony in acceptable range suggests quasi-periodic behavior
                mono_ok = monotony >= libration_monotony_critical[0] and monotony <= libration_monotony_critical[1]

                if stability_factor >= 4.99 and mono_ok:
                    # This suggests wide libration or chaotic libration
                    # that's still effectively trapped
                    status = 2  # Classify as pure (trapped)
                else:
                    status = 1  # Transient
            else:
                status = 0
        else:
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
