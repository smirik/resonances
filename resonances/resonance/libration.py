import numpy as np
from scipy import signal
from scipy import stats
from resonances.config import config


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
    def periodogram(cls, x, y, start, stop, num_freqs):
        ps = np.linspace(start, stop, num_freqs)
        ws = np.asarray([2 * np.pi / P for P in ps])
        periodogram = signal.lombscargle(x, y, ws)
        return {'ps': ps, 'ws': ws, 'periodogram': periodogram}

    @classmethod
    def density(cls, y, num_freqs):
        ps = np.linspace(0, 2 * np.pi, num_freqs)
        # plt.figure(figsize=(20, 10))
        adjust = config.get('libration.density.adjust')
        kernel = stats.gaussian_kde(y)
        kernel.set_bandwidth(kernel.factor * adjust)
        kdes = kernel(ps)
        max_value = max(kdes)
        #    plt.plot(ps, kdes, label='A{}, i={}'.format(resonant_asteroids[i], i))
        return {'max': max_value, 'ps': ps, 'kdes': kdes}

    @classmethod
    def libration(cls, x, y, Nout, start=1000, stop=20000, num_freqs=1000):
        data = cls.periodogram(x, y, start, stop, num_freqs)
        pmax = np.sqrt(4 * data['periodogram'].max() / Nout)
        density = cls.density(y, num_freqs)
        dmax = density['max']
        pure = cls.has_pure_libration(y)

        if pure:
            flag = True
            status = 2
        elif pmax < config.get('libration.periodogram.critical'):
            flag = False
            status = 0
        elif dmax > config.get('libration.density.critical'):
            flag = True
            status = 1
        elif dmax <= config.get('libration.density.critical'):
            flag = False
            status = 0

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
    def has(cls, x, y, Nout, start=500, stop=20000, num_freqs=1000):
        res = cls.libration(x, y, Nout, start, stop, num_freqs)
        return res['flag']

    @classmethod
    def status(cls, x, y, Nout, start=500, stop=20000, num_freqs=1000):
        res = cls.libration(x, y, Nout, start, stop, num_freqs)
        return res['status']
