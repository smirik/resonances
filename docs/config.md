# Config

All default config values are stored in a file called .env.dist in the source code. It is not recommended to change anything in that file directly.

If you need to change a specific config value, you can do it in the following way:

If you need to change a specific config value, you have several options:

1. Create a `.env` file in your working directory to override selected variables.
2. Set environment variables directly in your system or through your shell.
3. Modify some of the parameters when creating a Simulation (use IDE hint to see all available options):

```python
import resonances

sim = resonances.Simulation(name='test', integrator='whfast', save='all', save_path='output/my_lovely_folder')
```

A resonances.Simulation object has several properties that mirror these config values. Note that in Python code I usually use underscore notation (e.g., plot_type). This could be done after instantiating the object:

```python
import resonances

sim = resonances.Simulation(...)
# for example
sim.config.save_path = 'output/another_lovely_folder'
sim.config.plot = 'nonzero'
```

4. Set or update them at runtime in Python with:

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

Under the hood, resonances reads configuration in the following priority:

1. Parameters when instantiating the Simulation object.
1. `.env` file.
1. Environment variables.
1. The defaults in .env.dist.

## Saving options

Below is the list of options. When lowercase is used, it refers to the arguments of the constructor of Simulation. When uppercase is used, it refers to the `.env.dist`.

- `save`/`SAVE` (string or None): whether or not save the result of the simulation. Options: `all`, `resonant`, `candidates`, `extended`, `nonzero`, `negative`, `None`.
  - `resonant` — only confirmed resonant objects (`status > 0`: transient, libration).
  - `candidates` — resonant, uncertain/near-separatrix, and slow-circulation candidates
    (`status` in {2, 1, -1, -2, -3, -4}).
  - `extended` — candidates + uncertain/slow (`status > 0` or `status` in {-3, -4, -5, -9}).
  - `nonzero` — everything except non-resonant (`status != 0`).
  - `negative` — only negative statuses requiring manual verification (`status < 0`).
  - `all` — save everything.
- `save_path`/`SAVE_PATH`: directory where to save the output CSV files (data only). If you do not specify `save_path` when creating Simulation object, it will use `SAVE_PATH` with a sub-directory based on the current timestamp. In other words, unless explicitly specified, the app will create a subdirectory in `SAVE_PATH` to differentiate multiple runs.
- `plot`/`PLOT` : the same as for `sim.save` but for graphs.
- `save_summary`/`SAVE_SUMMARY` (bool): save summary of the simulation as a dataframe (available through `get_simulation_summary()` method)
- `plot_path`/`PLOT_PATH` (str): the same as `save_path`.
- `plot_type`/`PLOT_TYPE` (str): determines what to do with graphs. `save` - only save graphs as files (default), `show` - just show (if false), `both` - both options. Valid only for plots specified by `plot`. In other words, if you set `plot` as `None`, no graphs will be plotted.
- `plot_subfolder_strategy`/`PLOT_SUBFOLDER_STRATEGY` (str or None): controls how plots are organized within `plot_path`. By default, `None` (empty), meaning all plots are saved directly in `plot_path`. If set to `status`, creates subfolders based on resonance classification: `resonant/` (status=2), `transient/` (status=1), `non-resonant/` (status=0), `controversial-transient/` (status=-1), and `controversial-libration/` (status=-2). The combined MMR figures always live in `combined/` under `plot_path`, with the same status subfolders inside.
- `plots`/`PLOTS` (list or str): plot kinds to generate: `evolution`, `ecc_vector`, `cross_spectrum`, `free_omega` (Lidov-Kozai only), and for MMRs `combined`, `recurrence`, `fair`, `portrait`, `cycles`. Default: `['evolution']`. An unknown name is an error. See [Plots](plots.md).
- `plot_always`/`PLOT_ALWAYS` (list or comma-separated str): resonance types (`mmr`, `secular`, `lidov_kozai`) whose figures are drawn whatever their status, e.g. `plot_always=['lidov_kozai']` to always see the ZLK angle in a survey that otherwise plots `nonzero` only. Has no effect when `plot` is `None`. Default: empty.
- `plot_options`/`PLOT_OPTIONS` (dict, or JSON in `.env`): options of the MMR diagnostic plots — `style` (`screen`/`paper`), `raw`, and per-kind settings. Merged over the defaults and validated. See [Plots](plots.md#options).

## Cross-spectral coherence options

These control the σ–a / σ–e / σ–i / e–i cross spectra and the Lidov-Kozai coherence gate. See [Cross-Spectral Coherence](coherence.md) for what they mean physically.

- `coherence_enabled`/`COHERENCE_ENABLED` (bool): master switch. Default: `True`.
- `coherence_skip_non_resonant`/`COHERENCE_SKIP_NON_RESONANT` (bool): skip bodies classified as non-resonant (`status = 0`). Default: `True`.
- `coherence_alpha`/`COHERENCE_ALPHA` (float): per-bin false-alarm probability behind the threshold `γ²_crit = 1 - α^(1/(n_seg-1))`. Default: `0.01`.
- `coherence_n_segments`/`COHERENCE_N_SEGMENTS` (list or comma-separated str): Welch segmentation levels. The coarsest level with a non-empty validity band decides. Default: `[4, 8, 16]`.
- `coherence_window`/`COHERENCE_WINDOW` (str): Welch window. Default: `hann`.
- `coherence_max_lines`/`COHERENCE_MAX_LINES` (int): how many strongest lines per pair per level reach `coherence.csv`. Default: `10`.
- `coherence_period_min`/`COHERENCE_PERIOD_MIN`, `coherence_period_max`/`COHERENCE_PERIOD_MAX` (float or empty): extra narrowing of the search band, in years. Empty means the per-resonance-type preset (10000 years for Lidov-Kozai and secular). The intrinsic limits `4·dt` and `T/(5·n_seg)` always apply on top.
- `coherence_pairs` (dict, constructor only): which pairs to analyse per resonance type, e.g. `{'mmr': ['sigma-a', 'sigma-e']}`. Replaces the preset for the named types only.
- `coherence_phase_targets` (dict, constructor only): `(target_deg, tolerance_deg)` per resonance type and pair, e.g. `{'lidov_kozai': {'e-i': (180.0, 30.0)}}`.
- `zlk_coherence_check`/`ZLK_COHERENCE_CHECK` (bool): whether to measure the `confirmed_ei` flag — whether e and i really trade at the strongest coherent line. It no longer changes any status; that moved to the free-omega gate below. Default: `True`.
- `zlk_min_anticorrelation`/`ZLK_MIN_ANTICORRELATION` (float): minimum |r| of band-passed e vs i required for the antiphase line to count as a real exchange. Default: `0.5`, **provisional** — calibrated on two objects, see [Cross-Spectral Coherence](coherence.md).

## Free element options

These decide the Lidov-Kozai status after classification: a librating angle is confirmed at `2`, demoted to `-2` when the libration turns out kinematic, or set to `-9` when no verdict can be reached. See [Free elements](free-elements.md).

- `free_elements_enabled`/`FREE_ELEMENTS_ENABLED` (bool): split the eccentricity and inclination vectors into forced and free parts, for bodies carrying a Lidov-Kozai resonance. Off means no split and no gate. Default: `True`.
- `free_elements_sampling_years`/`FREE_ELEMENTS_SAMPLING_YEARS` (float or empty): decimate the series to about this step, with an anti-aliasing filter, before the split. Secular work needs nothing finer, and short-period terms folded into the secular band would otherwise read as free motion. Empty keeps the integration grid. Default: `500`.
- `free_gate_enabled`/`FREE_GATE_ENABLED` (bool): whether the gate may change a status. Off computes and reports everything but leaves classification alone. Default: `True`.
- `free_gate_min_cycles`/`FREE_GATE_MIN_CYCLES` (float): how many libration periods the baseline must hold before the gate will judge at all; below it the status becomes `-9`. Default: `3`.
- `free_gate_mask_quantile`/`FREE_GATE_MASK_QUANTILE` (float): samples where |z_free| falls below this fraction of its median are dropped before the argument is unwrapped, since near the origin the argument is undefined. Default: `0.15`.

The planetary fundamentals used for the fit are the same `g5…g8` / `s5…s8` values as the secular module uses, from `.env` (see below).

## Libration options

- `LIBRATION_FILTER_CUTOFF` (float): used to cutoff the frequencies that are higher that this value. By default, it is set to `0.0005`, which means that all oscillations with periods lower than `500` will be removed by the filter.
- `LIBRATION_FILTER_ORDER` (int): used for `butter` filter. By default, `2`,
- `LIBRATION_FREQ_MIN` (float): the minimal frequency of the librations that will be taken into account. By default, `0.00001`. Corresponds to the period of oscillations equal to `100000` years.
- `LIBRATION_FREQ_MAX` (float): the maximum frequency of the librations that will be taken into account. By default, `0.002`. Corresponds to the period of oscillations equal to `500` years.
- `LIBRATION_CRITICAL` (float): the critical value of the peaks on the periodograms that is counted as significant. By default, `0.1`.
- `LIBRATION_SOFT` (float): the _soft_ critical value of the peaks on the periodograms that is counted as significant. Used when you _really_ want to find some librations. By default, `0.05`.
- `LIBRATION_PERIOD_MIN` (int): the number of years to remove from the beginning and end. If you want to disable the cut of these points, just set it to `0`.
- `LIBRATION_PERIOD_CRITICAL` (int): the critical value of the maximum libration period used to identify is there libration or not. By default, `20000` years.
- `LIBRATION_MONOTONY_CRITICAL` (list): critical values for the metric `monotony`. By default, `[0.4, 0.6]`.

## Integration options

See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html) for explanation of the integrator's settings.

- `tmax`/`INTEGRATION_TMAX` (int): the end of the integration. Please note that `rebound` does not use years as time steps but years divided into 2&pi;. Therefore, if you want to integrate for 1000 years, the value `INTEGRATION_TMAX` should be set to `1000*2&pi;` ~ `6283`. By default, `628319` (approx. 100000 years). Note that you can integrate backwards. In this case, `INTEGRATION_TMAX` can be set to `-628319`. The defaults derived from the span (`libration_period_min`, `libration_period_critical`, `periodogram_critical`, `periodogram_frequency_min`) use the absolute span, so a backward run gets the same positive values as the forward run of the same length.
- `dt`/`INTEGRATION_DT` (float): the step of the integration (or the initial step for some integrators). By default, `0.1`.
- `integrator`/`INTEGRATION_INTEGRATOR` (string): the default integrator from rebound. By default, `SABA(10,6,4)`. See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html).
- `integration_safe_mode`/`INTEGRATION_SAFE_MODE` (int): the parameter of the integration. By default, `0`. See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html)
- `integration_corrector`/`INTEGRATION_CORRECTOR` (int): the parameter of the corrector for symplectic integrators. By default, `17`. See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html)
- `SOLAR_SYSTEM_FILE` (str): the name of the cache file used to store the initial data of the Sun, planets, and Pluto. It is used to speed up the creation of the simulation. By default, `cache/solar.bin`. Note that in order to avoid issues with initial date&time, the app will automatically add postfix equals to the timestamp of the simulation date, i.e., `cache/solar-12345.bin`, and `cache/solar-moon-12345.bin` for the system with the Moon.
- `solar_system_moon`/`SOLAR_SYSTEM_MOON` (bool): the Moon as a separate body. Default: `False`. See below.

### Solar system model

By default the massive bodies are the Sun, Mercury–Neptune and Pluto, fetched from NASA Horizons by name. `Earth` is then Horizons body 3, the Earth–Moon barycentre, with the Earth+Moon mass. That is fine for asteroids that stay far from the Earth.

With `solar_system_moon=True` (`IntegrationEngine.create_solar_system_with_moon`) the Earth is the geocentre (Horizons 399) and the Moon (Horizons 301) is an eleventh body, appended after Pluto. The planets keep their particle indices, so every resonant angle is built as before, with the Earth's angles now taken against the geocentre. The Moon is never a planet of an angle and is not in the saved planet series. Earth and Moon together have exactly the mass, position and velocity of the barycentre. The price: the geocentre's heliocentric osculating elements carry the Moon's monthly reflex (about 12.6 m/s). Measured against the barycentre, a differs by up to 1e-3 AU, e by 9e-4, λ by 0.05° and ϖ by 3°. The saved Earth series and every Earth angle (the 1:1 angle by 0.05°, any secular angle with g3 or s3 much more) contain this term, sampled at the output step. Use it for objects with close Earth encounters (Earth co-orbitals, NEAs): within a few lunar distances the barycentre is a poor stand-in for the two bodies.

Only `ias15` is accepted with the Moon; any other integrator raises `ValueError` when the simulation is created or the integrator is set up. A fixed-step symplectic split around the Sun cannot follow a 27-day orbit with steps of days to months. Measured over 100 yr (2023-02-25 start, against ias15): WHFast, SABA(10,6,4), MERCURIUS and TRACE with dt = 0.05 (3 days) all put the Earth 0.03 AU off along its orbit; MERCURIUS and TRACE still 0.0015 AU with dt = 0.01, at 3 and 0.3 times the CPU of ias15. The hybrid integrators gave the same error as WHFast, although the Moon never leaves the Earth's encounter radius.

`N_active` is set to the number of massive bodies (10, or 11 with the Moon); the asteroids are test particles.

### Horizons and TLS

The planets are fetched by rebound, which uses Python's `urllib`. Python's default CA bundle may lack the root certificate JPL uses (macOS `/etc/ssl/cert.pem` had no Sectigo Public Server Authentication Root R46 in October 2026). While it fetches the planets, the package therefore verifies against [certifi](https://pypi.org/project/certifi/)'s bundle, and restores rebound's `urlopen` afterwards. Verification is never switched off. If `SSL_CERT_FILE` is set, it is used instead. Setting `SSL_CERT_FILE` from inside a running Python process is not enough once any HTTPS connection has been verified there: OpenSSL keeps the store it loaded first.

### Access to the parameters of `rebound`

While resonances wraps many Rebound integrator settings, you can still manipulate Rebound directly. The rebound simulation is `sim.integration_engine.sim`; it exists after `create_solar_system()`:

```python
sim = resonances.Simulation()
sim.create_solar_system()
sim.integration_engine.sim.ri_ias15.epsilon = 1e-10
```

`run_integration()` calls `setup_integrator()`, which sets `integrator`, `dt` and `N_active` (the number of massive bodies, 10 or 11 with the Moon), the WHFast corrector when `INTEGRATION_CORRECTOR` is set and the SABA safe mode. Change those through the config (`sim.config.dt = 0.01`), not on the rebound object: direct values are overwritten.

## AstDyS and Catalogue options

- `source`/`DATA_SOURCE` (str): which source to use for the initial data. By default, it is set to `nasa`. Could be `astdys`. Note that the planet's data will be always taken from NASA Horizon.
- `ASTDYS_URL` (str): the URl of the AstDyS catalogue used to download at the first run.
- `ASTDYS_CATALOG` (str): if AstDyS catalogue is already downloaded, you may specify its location. By default, `cache/allnum.cat`.
- `CATALOG_PATH` (str): the path of the file used to store the converted AstDyS catalogue (in csv).

## Matrices

The mean motion resonance represents a commensurability between the frequencies of several bodies and an asteroid, which implies the oscillations of the resonant angle. The resonant angle has integer coefficients for every variable included. While there are almost no limitations on the values of these integers, the number of possible resonances is infinite. Thus, it should be somehow limited to avoid an infinite loop. These limitations are in `MATRIX_` section of the config.

For more details, please see [Matrix page](matrix.md) and the following papers (note that they have different approaches to setting up the limits):

1. Smirnov, E. A., Dovgalev, I. S. & Popova, E. A. Asteroids in three-body mean motion resonances with planets. Icarus (2017) doi:10.1016/j.icarus.2017.09.032.
1. Smirnov, E. A. & Shevchenko, I. I. Massive identification of asteroids in three-body resonances. Icarus 222, 220–228 (2013).
1. Nesvorný, D. & Morbidelli, A. Three-Body Mean Motion Resonances and the Chaotic Structure of the Asteroid Belt. The Astronomical Journal 116, 3029–3037 (1998).

**Important:** If you have change any of the values below, **DELETE** manually the generated csv-files (e.g., `cache/mmr-3body.csv`) to force the app to regenerate the matrices. Otherwise, the app will simply use old generated files.

- `MATRIX_3BODY_PRIMARY_MAX` (int): the maximum value of the integer for the first planet for three-body resonances. Should be always positive. Default is `8`.
- `MATRIX_3BODY_COEF_MAX` (int): the maximum absolute value of the integer for the other planet and the asteroid in three-body resonance. Should be always positive.
- `MATRIX_3BODY_ORDER_MAX` (int): the maximum value of the order of the three-body mean motion resonance. The order is the absolute value of the sum of the integers for the mean longitudes. Should be always positive.
- `MATRIX_3BODY_FILE` (str): the path to the file where the app dumps the calculated values of the corresponding semi-major axis. You can review the generated CSV file and make modifications if needed. The default option is `cache/mmr-3body.csv`.
- `MATRIX_2BODY_*` (mixed): the same but for two-body resonances.

## Other

- `LOG_FILE` (str): the path to the file where the app stores log data.
- `LOG_LEVEL` (str): the default level what the app should store in the log file (options: `debug`, `info`, `warning`, `error`, `critical`). The default value is `info`.
