import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

def angle(times, angle, outputFile='cache/angle.png'):
    fig = plt.figure(figsize=(12, 5))
    plt.ylim((0, 2 * np.pi))
    plt.plot(
        times / (2 * np.pi),
        angle,
        label='Angle',
        linestyle='',
        marker=',',
    )
    plt.savefig(outputFile)


def axis(times, axis, outputFile='cache/axis.png'):
    plt.figure(figsize=(12, 5))
    plt.ylim((2.0, 4.0))
    plt.plot(times, axis, label='semiaxis', linestyle='', marker=',')
    plt.savefig(outputFile)

def plot(times, data, outputFile='cache/plot.png'):
    plt.figure(figsize=(12, 5))
    plt.plot(times, data, linestyle='', marker=',')
    plt.savefig(outputFile)


def periodogram(times, angle, outputFile='cache/periodogram.png'):
    Npts = 1000

    periods = np.linspace(3000 / (2 * np.pi), 200000 / (2 * np.pi), Npts)
    ang_freqs = 2 * np.pi / periods
    power = signal.lombscargle(times / (2 * np.pi), angle, ang_freqs)
    fig = plt.figure(figsize=(12, 5))
    plt.plot(periods, np.sqrt(4 * power / Npts))
    plt.savefig(outputFile)
