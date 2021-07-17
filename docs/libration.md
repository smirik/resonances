# Libration

There is no obvious algorithm how to identify whether or not the asteroid is in the resonance. While there are simple cases (such as 463 Lola in `4J-2S-1`, see [Example](quick-start.ipynb)), there are many asteroids whose resonant angles are librating but not the whole time. Moreover, it might be the case that the resonant angle librates, but the frequency of these oscillations differs from the frequency of the librations of the values of the semi-major axis. Therefore, a complex solution is needed.

## Metrics

The module `resonances.libration` uses several metrics to resolve whether or not an asteroid is resonant:

1. `pure`: this flag shows is there at least one period of circulation. Technically, it means that the resonant angle (limited by 2&pi; filter) should never break the limits of `0` and `2&pi` or be shifted by a constant.
1. `overlapping`: the module calculates the frequencies of oscillations of the resonant angle and semi-major axis (see [Config](config.md) for the limits used by default). If there is a shared frequency within limits, then it signals that there might be a resonance.
1. `max_libration_length`: it represents the longest period of libration of the resonant angle. Or, in other words, how long there was no circulation. For all resonant asteroids, this value should be high enough.
1. `monotony`: a very simple metric that calculates the percentage of the points when the value of the resonant angle for the current time is less than the same value for the previous step. If the resonant angle librates, it means that this value should be close to `0.5`.

All these values are available for `Body` objects after the integration. They are available in the output file `summary.csv`. The time series (if the appropriate flag is set) are also stored in CSV files in the output directory.

You may invoke or recalculate them in the following way:

```python
sim = resonances.Simulation()
# ...
body = sim.bodies[0]
body.librations
body.libration_metrics
body.libration_pure
body.periodogram_frequency
body.periodogram_power
body.periodogram_peaks
body.axis_filtered
body.axis_periodogram_frequency
body.axis_periodogram_power
body.axis_periodogram_peaks
```

For the detailed algorithm, see `resonances/resonance/libration.py` file.

## The workflow of identification

How the app distinguishes different types of asteroids?

1. If there is a pure libration around 0 or &pi; **and** the frequency of libration (based on [the Lomb-Scargle periodogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.lombscargle.html)) is close to the frequency of oscillations of semi-major axis (more precisely: if the peaks taken with their width overlap each other), then the function ends with the status `2`.
2. If there is no shared frequency for the oscillations of the resonant angle and the semi-major axis, but the libration is pure, then the app considers this case _difficult_ and set the status `-2`, which means that manual verification is needed.
3. If there is a shared frequency of oscillations and the libration period is _long enough_ (set in [Config](config.md)), then the function ends with the status `1`, which means that the asteroid is in a transient resonance.
4. However, if there is no shared frequency but still the libration period is _long enough_, the app checks the monotony metrics. If it is acceptable, then the function ends with the status `-1`, which means that there is something strange and thus should be verified manually.
5. For all other cases, the function ends with the status `0`, which means that the asteroid is not in the resonance.

## Possible statuses

| Status | Description                                         |
| :----: | --------------------------------------------------- |
|   2    | There is a pure libration around 0 or Pi.           |
|   1    | There is a libration, but it does not look pure.    |
|   0    | There is no libration (circulation or similar).     |
|   -1   | Requires manual verification (transient libration). |
|   -2   | Requires manual verification (pure libration).      |

### A few comments to the negative statuses

1. `-1`: It looks like that there is no libration because there is no shared frequency of oscillation between resonant angle and semi-major axis. However, the behaviour of the resonant angle is not chaotic and thus needs to be manually verified.
1. `-2`: There is a pure libration of the resonant angle. However, the frequency of oscillations does not correspond to the frequency of the oscillations of the semi-major axis. Thus, it requires manual verification.
