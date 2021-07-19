from resonances.data import const


def axis_from_mean_motion(mean_motion):
    return (const.K / mean_motion) ** (2.0 / 3)


def mean_motion_from_axis(a):
    return const.K / a ** (3.0 / 2)
