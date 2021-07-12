import numpy as np
from scipy import signal
from scipy import stats

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
