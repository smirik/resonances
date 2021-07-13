import numpy as np
from scipy import signal
from scipy import stats
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
    def pure(cls, y):
        prev = y[0]
        num = 0
        for elem in y:
            num += 1
            if abs(elem - prev) > np.pi:
                return False
            prev = elem
        return True

    @classmethod
    def has_pure_libration(cls, y):  # not working for librations that are not around 0 or +/- np.pi
        flag1 = cls.pure(y)
        if flag1:
            return True
        flag2 = cls.pure(cls.shift(y))
        if flag2:
            return True
        return False

    @classmethod
    def monotony_estimation(data):
        num = 0
        prev = data[0]
        for elem in data:
            if prev - elem > np.pi:
                prev = elem
                continue
            if elem - prev > np.pi:
                num += 1
                prev = elem
                continue
            if elem < prev:
                num += 1  # num of "decreasing" points
            prev = elem
        return num / (len(data) - 1)

    @classmethod
    def find_breaks(cls, data, coeff=1.0, break_value=np.pi):
        """
        This function finds continuos breaks in a given dataset and return these breaks among with additional data.

        Parameters
        ----------

        data : list
            one-dimensional array of data where to find breaks
        coeff : float
            Used as a multiplicator to convert indexes of the output array (i*coeff). Useful for plotting. (Default: 10)
        break_value : float
            Used to determine a break. If the absolute diff between the previous one and the current is more than break_value, then there is a break. (Default: np.pi)

        Returns
        -------
        list
            Returns a multidimensional array: res[0] - the indexes (multiplied by coeff) when the break happened, res[1] - the direction of the break (1 if it intersects the top line and thus elem>prev, -1 otherwise), res[2] - values of the elements prior to the breaks, res[3] - the values in the break points.
        """
        prev = data[0]
        res = [[], [], [], []]  # break point, direction (1 or -1), prev, curr
        for i, elem in enumerate(data):
            if abs(elem - prev) > break_value:
                res[0].append(i * coeff)
                direction = 1 if elem > prev else -1
                res[1].append(direction)
                res[2].append(prev)
                res[3].append(elem)
            prev = elem
        return res

    @classmethod
    def circulation(cls, data, coeff=1.0):
        """
        Find and returns libration periods with some additional data.

        Parameters
        ----------

        data : list
            Input data to find breaks.

        Returns
        -------

        list
            Returns a multidimensional list. res[0] - the start of a libration period, res[1] - the end of a libration period, res[2] - the length of the given libration period.

        """
        breaks = cls.find_breaks(data, coeff=coeff)

        if 0 == len(breaks[1]):  # full interval is a libration
            return [[0], [(len(data) - 1) * coeff], [(len(data) - 1) * coeff]]

        breaks_diff = np.diff(breaks[0])

        librations = [[], [], []]  # start, stop, length

        # we set the first libration to be started at 0
        # librations[0].append(0.0)
        # librations[1].append(breaks[0][0])
        # librations[2].append(breaks[0][0])

        libration_start = 0
        libration_length = breaks[0][0]
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
                librations[0].append(libration_start)
                librations[1].append(len(data) * coeff)
                librations[2].append(libration_length + (len(data) * coeff - breaks[0][i]))

            prev_direction = curr_direction
        return librations

    @classmethod
    def periodogram_lomb(cls, x, y, minimum_frequency=0.00001, maximum_frequency=0.002, kernel=None, nyquist_factor=5):
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
        kernel : int, optional
            if specificied, then it applies scipy.signal.medfilt to y, by default None
        nyquist_factor : int, optional
            the parameter from lomg-scargle method, by default 5

        Returns
        -------
        (frequence, power)
            Return list of frequencies among with related power.
        """
        if kernel is not None:
            frequency, power = LombScargle(x, signal.medfilt(y, kernel)).autopower(
                nyquist_factor=nyquist_factor, minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency
            )
        else:
            frequency, power = LombScargle(x, y).autopower(
                nyquist_factor=nyquist_factor, minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency
            )
        return (frequency, power)
        # frequency, power = LombScargle(x, signal.medfilt(y, kernel)).autopower(nyquist_factor=nyquist_factor, minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency)
        # max_power = power.max()

    @classmethod
    def periodogram(cls, x, y, start, stop, num_freqs):
        ps = np.linspace(start, stop, num_freqs)
        ws = np.asarray([2 * np.pi / P for P in ps])
        periodogram = signal.lombscargle(x, y, ws)
        return {'ps': ps, 'ws': ws, 'periodogram': periodogram}

    @classmethod
    def density(cls, y, num_freqs):
        ps = np.linspace(0, 2 * np.pi, num_freqs)
        adjust = resonances.config.get('libration.density.adjust')
        kernel = stats.gaussian_kde(y)
        kernel.set_bandwidth(kernel.factor * adjust)
        kdes = kernel(ps)
        max_value = max(kdes)
        return {'max': max_value, 'ps': ps, 'kdes': kdes}

    @classmethod
    def body(cls, sim, body: resonances.Body):
        data = cls.find(
            x=sim.times,
            y=body.angle,
            Nout=sim.Nout,
            start=sim.libration_start,
            stop=sim.libration_stop,
            num_freqs=sim.libration_num_freqs,
            pcrit=resonances.config.get('libration.periodogram.critical'),
            dcrit=resonances.config.get('libration.density.critical'),
        )
        body.status = data['status']
        body.libration_data = data
        return data

    @classmethod
    def find(cls, x, y, Nout, start, stop, num_freqs, pcrit, dcrit):
        data = cls.periodogram(x, y, start, stop, num_freqs)
        pmax = np.sqrt(4 * data['periodogram'].max() / Nout)
        density = cls.density(y, num_freqs)
        dmax = density['max']
        pure = cls.has_pure_libration(y)

        status = cls.resolve(pure, pmax, pcrit, dmax, dcrit)
        flag = False
        if status > 0:
            flag = True

        return {
            'periodogram': np.sqrt(4 * data['periodogram'] / Nout),
            'ps': data['ps'],
            'ws': data['ws'],
            'flag': flag,
            'status': status,
            'pure': pure,
            'pmax': pmax,
            'density': density,
            'dmax': dmax,
        }

    @classmethod
    def resolve(cls, pure, pmax, pcrit, dmax, dcrit):
        if pure:
            status = 2
        elif pmax < pcrit:
            status = 0
        elif dmax > dcrit:
            status = 1
        elif dmax <= dcrit:
            status = 0
        return status
