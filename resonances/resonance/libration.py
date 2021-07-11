import numpy as np
from scipy import signal


class libration:
    @classmethod
    def get_periodogram(cls, x, y, Nout, start, stop, num_freqs):
        ps = np.linspace(start, stop, num_freqs)
        ws = np.asarray([2 * np.pi / P for P in ps])
        periodogram = signal.lombscargle(x, y, ws)
        return {'ps': ps, 'ws': ws, 'periodogram': periodogram}

    @classmethod
    def has(cls, x, y, Nout, start=500, stop=20000, num_freqs=1000):
        tmp = cls.libration(x, y, Nout, start, stop, num_freqs)
        if tmp['flag']:
            return True
        return False

    @classmethod
    def status(cls, x, y, Nout, start=500, stop=20000, num_freqs=1000):
        res = cls.libration(x, y, Nout, start, stop, num_freqs)
        if res['pure']:
            return 2
        if res['flag']:
            return 1
        return 0

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
    def libration(cls, x, y, Nout, start=1000, stop=20000, num_freqs=1000):
        data = cls.get_periodogram(x, y, Nout, start, stop, num_freqs)
        pmax = np.sqrt(4 * data['periodogram'].max() / Nout)

        flag = False
        if pmax > 0.5:  # magic number
            flag = True

        pure = cls.has_pure_libration(y)

        return {
            'periodogram': np.sqrt(4 * data['periodogram'] / Nout),
            'ps': data['ps'],
            'ws': data['ws'],
            'flag': flag,
            'pure': pure,
            'pmax': pmax,
        }
