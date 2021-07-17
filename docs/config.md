# Config

All default config values are stored in `config.json` file in the source code. It is not recommended to change anything there.

If you need to change a specific config value, you can do it in the following way:

```python
import resonances
resonances.config.set(KEY, VALUE)
resonances.config.set('integration.dt', 0.01)
```

The service has the method `has`, which verifies is there a config with the given key, and the method `get`, which returns the value by the given key. Note that if there is no config with the given key, `get` method will raise `Exception`.

```python
import resonances
if resonances.config.has('integration.dt'):
    print('There is a config with the key = integration.dt')
    print(resonances.config.get('integration.dt'))
```

## Saving options

- `save` (bool): if `true`, the results of the integration (including the plots) will be saved. If it is `false`, nothing will be saved. All options below work only if `save` is true.
- `save.path` (str): path to save CSV files (for all CSV files).
- `plot.path` (str): path to save plots.
- `save.summary` (bool): if `true`, then the app will create a file called `result.csv` where the results of the whole simulation (with some additional data) will be stored. In other words, it is a summary of the simulation and its result.
- `save.additional.data` (bool): if `true`, then for each object, the app will add additional information: filtered values of semi-major axis and periodograms data.
- `save.only.undetermined` (bool): if `true`, then the data will be saved only for those objects whose statuses are lower than `0` (possible pure or transient librators). This option requires `save` to be `true` and takes into account the value of `save.additional.data`. It does not affect `save.summary`.
- `plot` (bool): if `true`, then for each object, the app will create a plot containing angle, filtered axis, periodograms, and eccentricity. It works even if `save` is `false`. If `save.only.undetermined` is `true`, then the app will create figures **only** for the bodies with the negative statuses.

## Libration options

See [Libration section](libration.md) for explanation!

- `libration.oscillation.filter.cutoff` (float): used to cutoff the frequencies that are higher that this value. By default, it is set to `0.001`, which means that all oscillations with periods lower than `1000` (effectively, `500`) will be removed by the filter.
- `libration.oscillation.filter.order` (int): used for `butter` filter. By default, `2`,
- `libration.periodogram.frequency.min` (float): the minimal frequency of the librations that will be taken into account. By default, `0.00001`. Corresponds to the period of oscillations equal to `100000` years.
- `libration.periodogram.frequency.max` (float): the maximum frequency of the librations that will be taken into account. By default, `0.002`. Corresponds to the period of oscillations equal to `500` years.
- `libration.periodogram.critical` (float): the critical value of the peaks on the periodograms that is counted as significant. By default, `0.1`.
- `libration.periodogram.soft` (float): the _soft_ critical value of the peaks on the periodograms that is counted as significant. Used when you _really_ want to find some librations. By default, `0.05`.
- `libration.period.critical` (int): the critical value of the maximum libration period used to identify is there libration or not. By default, `20000` years.
- `libration.monotony.critical` (list): critical values for the metric `monotony`. By default, `[0.4, 0.6]`.

## Integration options

See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html) for explanation of the integrator's settings.

- `integration.tmax` (int): the end of the integration. Please note that `rebound` does not use years as time steps but years divided into 2&pi;. Therefore, if you want to integrate for 1000 years, the value `integration.tmax` should be set to `1000*2&pi;` ~ `6283`. By default, `628319` (approx. 100000 years).
- `integration.dt` (float): the step of the integration (or the initial step for some integrators). By default, `0.1`.
- `integration.integrator` (string): the default integrator from rebound. By default, `SABA(10,6,4)`. See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html).
- `integration.integrator.safe_mode` (int): the parameter of the integration. By default, `0`. See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html)
- `integration.integrator.corrector` (int): the parameter of the corrector for symplectic integrators. By default, `17`. See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html)
- `solar_system_file` (str): the name of the cache file used to store the initial data of the Sun, planets, and Pluto. It is used to speed up the creation of the simulation. By default, `cache/solar.bin`.

## AstDyS options

- `astdys.catalog.url` (str): the URl of the AstDyS catalogue used to download at the first run.
- `astdys.catalog` (str): if AstDyS catalogue is already downloaded, you may specify its location. By default, `cache/allnum.cat`.
- `astdys.date` (str): the date of the data in AstDyS catalogue converted from MJD to the datetime format. By default, `2020-12-17 00:00`.

## Access to the parameters of `rebound`

Note that while `resonances` provides a wrapper for rebound integrators, you may invoke directly some properties or methods in rebound. You can access the simulation object of rebound directly as an attribute `sim.sim`:

```python
sim = resonances.Simulation()
# ...
# set rebound value for whfast corrector
sim.sim.N_active = 10
sim.sim.ri_whfast.corrector = 17
sim.sim.dt = 0.01
```

`resonances` will not override these values. It sets it only once through initialisation from config.
