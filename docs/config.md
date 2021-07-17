## Available config options

### Saving data

- `save` (bool): if `true`, the results of the integration (including the plots) will be saved. If it is `false`, nothing will be saved. All options below work only if `save` is true.
- `save.summary` (bool): if `true`, then the app will create a file called `result.csv` where the results of the whole simulation (with some additional data) will be stored. In other words, it is a summary of the simulation and its result.
- `save.additional.data` (bool): if `true`, then for each object, the app will add additional information: filtered values of semi-major axis, periodograms data, etc.
- `save.only.undetermined` (bool): if `true`, then the data will be saved only for those objects whose statuses are lower than `0` (possible pure or transient librators). This option requires `save` to be `true` and takes into account the value of `save.additional.data`. It does not affect `save.summary`.
- `plot` (bool): if `true`, then for each object, the app will create a plot containing angle, filtered axis, periodograms, and eccentricity. It works even if `save` is `false`.

- `save.path` (str): path to save CSV files.
- `plot.path` (str): path to save plots.
