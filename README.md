# Mean-Motion Resonances

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Resonances](https://github.com/smirik/resonances/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/smirik/resonances/actions/workflows/ci.yml)

`resonances` is an open-source package dedicated to the identification of mean-motion resonances of small bodies. Many examples are for the Solar system; however, you might use the package for any possible planetary system, including exoplanets.

For more information, [read the documentation](https://smirik.github.io/resonances/).

**Note:** while this app has many functional and integration tests built in, it is still in the dev stage. Hence, it might include some inconsistencies. So, any community help is appreciated!

## What's new

### June 2025

1. **Clean Architecture Implementation**: The `Simulation` class has been completely refactored into a clean, component-based architecture without backward compatibility concerns. The main `simulation.py` file has been reduced by **67%** (from 613 to 204 lines) by removing unnecessary setters/getters and moving specialized functionality into dedicated components: `SimulationConfig`, `BodyManager`, `IntegrationEngine`, and `DataManager`.
1. **Modern Configuration Syntax**: Configuration parameters are now accessed using the clean `sim.config.property` syntax (e.g., `sim.config.dt = 0.1`, `sim.config.save = 'all'`). Essential properties needed by external modules (like libration analysis) are still available directly on the simulation object.
1. **Comprehensive Test Refactoring**: All tests have been updated to use the new clean architecture, with 46 simulation component tests passing, ensuring the refactoring maintains full functionality while improving code quality.

### February 2025

1. Full support for `nasa` Horizon source of the initial data.
1. `find` and `check` methods for quick identification of the resonant status of objects.
1. `create_mmr` method now supports variaty of options: string, a list of strings, an object, a list of objects.
1. `Simulation` constructor got many new parameters allowing to change the settings directly when instantiating.
1. Instead of `config.json`, `.env.dist` is now used. Furthermore, a developer can specify `.env` in the directory, which will overwrite the default parameters or just use environment variables.
1. MMRs now have `order` function.
1. Added full support for backward integration (`dt=-1.0`, `tmax=-600000`).
1. Minor updates to graphs.

### October 2024

1. The `resonances.find` method now accepts extra parameters: `name`, `sigma2`, and `sigma3`, [see the documentation](https://smirik.github.io/resonances/).
2. Fixed bug with wrong titles on the plots (periodograms for the resonant angle and semi-major axis).
3. Fixed bug when adding an asteroid that has no relevant MMRs (previously it caused exception).

### July 2024

1. Now you can choose the type of the output image: it could be either ` pdf`` or  `png`.

```python
sim = resonances.Simulation()
...
sim.image_type = 'pdf'
```

2. You can plot only resonant asteroids:

```python
sim = resonances.Simulation()
...
sim.plot_only_identified = True
```

If it is `true`, then the app will plot every resonant asteroid even if `plot` is `False`. If `plot` is `True`, this option is ignored.

3. New finder module (easy resonance identification):

```python
import resonances

sim = resonances.find([463, 490], planets=['Mars', 'Jupiter', 'Saturn'])
sim.run()
```

## Features

The package:

- can automatically identify two-body and three-body mean-motion resonance in the Solar system,
- accurately differentiates different types of resonances (pure, transient, uncertain),
- provides an interface for mass tasks (i.e. find resonant areas in a planetary system),
- can plot time series and periodograms,
- and, yeah, it is well tested ;)

It actively uses [REBOUND integrator](https://rebound.readthedocs.io) maintained by Hanno Rein and others.

## Installation

To install resonances on your system, follow the instructions on the appropriate [installation guide](https://smirik.github.io/resonances/install/)

## Mean-motion resonances

For those who are not familiar with the mean-motion resonances, here is the list of papers used to develop this package:

### Papers about the automatic identification of resonant asteroids

1. Smirnov, E. A. & Dovgalev, I. S. Identification of Asteroids in Two-Body Resonances. Solar System Research 52, 347–354 (2018).
2. Smirnov, E. A. (2023). A new python package for identifying celestial bodies trapped in mean-motion resonances. Astronomy and Computing, 100707. https://doi.org/10.1016/j.ascom.2023.100707
3. Smirnov, E. A. & Shevchenko, I. I. Massive identification of asteroids in three-body resonances. Icarus 222, 220–228 (2013).
4. Smirnov, E. A., Dovgalev, I. S. & Popova, E. A. Asteroids in three-body mean motion resonances with planets. Icarus (2017) doi:10.1016/j.icarus.2017.09.032.
5. Nesvorný, D. & Morbidelli, A. Three-Body Mean Motion Resonances and the Chaotic Structure of the Asteroid Belt. The Astronomical Journal 116, 3029–3037 (1998).

### Papers about mean-motion resonances

1. Chirikov, B. V. A universal instability of many-dimensional oscillator systems. Physics reports 52, 263–379 (1979).
1. Gallardo, T. Strength, stability and three dimensional structure of mean motion resonances in the solar system. Icarus 317, 121–134 (2019).
1. Gallardo, T. Atlas of the mean motion resonances in the Solar System. Icarus 184, 29–38 (2006).
1. Gallardo, T., Coito, L. & Badano, L. Planetary and satellite three body mean motion resonances. Icarus 274, 83–98 (2016).
1. Milani, A., Cellino, A., Knezevic, Z., Novaković, B. & Spoto, F. Asteroid families classification: Exploiting very large datasets. Icarus 239, 46–73 (2014).
1. Murray, N. & Holman, M. Diffusive chaos in the outer asteroid belt. The Astronomical Journal 114, 1246 (1997).
1. Murray, N., Holman, M. & Potter, M. On the Origin of Chaos in the Asteroid Belt. The Astronomical Journal 116, 2583–2589 (1998).
1. Shevchenko, I. I. On the Lyapunov exponents of the asteroidal motion subject to resonances and encounters. Proc. IAU 2, 15–30 (2006).

### Books

1. Murray, C. D. & Dermott, S. F. Solar system dynamics. (Cambridge Univ. Press, 2012).
1. Morbidelli, A. Modern celestial mechanics: aspects of solar system dynamics. (2002).

## References

Whenever you use this package, we are kindly asking you to refer to one of the following papers (please choose the appropriate):

1. **The package itself**:

- Smirnov, E. A. (2023). A new python package for identifying celestial bodies trapped in mean-motion resonances. Astronomy and Computing. https://doi.org/10.1016/j.ascom.2023.100707

```tex
@article{Smirnov2023,
  title    = {A new python package for identifying celestial bodies trapped in mean-motion resonances},
  journal  = {Astronomy and Computing},
  year     = {2023},
  issn     = {2213-1337},
  doi      = {https://doi.org/10.1016/j.ascom.2023.100707},
  url      = {https://www.sciencedirect.com/science/article/pii/S2213133723000227},
  author   = {E.A. Smirnov},
  keywords = {Mean-motion resonances, Python, Identification, Asteroids},
  abstract = {In this paper, a new open-source package ‘resonances’ written in python is introduced. It allows to find, analyse, and plot two-body and three-body mean-motion eccentricity-type resonances in the Solar and other planetary systems. The package has a better accuracy of the automatic identification procedure for resonant objects compared to previous studies. Furthermore, it has built-in integrations with AstDyS and NASA JPL catalogues. The code is extensively documented and tested with automatic tests. The package is available on GitHub under MIT Licence.}
}
```

2. **The Libration module and automatic identification of librations**:

- Smirnov, E. A. (2023). A new python package for identifying celestial bodies trapped in mean-motion resonances. Astronomy and Computing, 100707. https://doi.org/10.1016/j.ascom.2023.100707

3. **Mass identification of mean-motion resonances:**

- Smirnov, E. A., & Dovgalev, I. S. (2018). Identification of Asteroids in Two-Body Resonances. Solar System Research, 52(4), 347–354. https://doi.org/10.1134/S0038094618040056
- Smirnov, E. A., Dovgalev, I. S. & Popova, E. A. Asteroids in three-body mean motion resonances with planets. Icarus (2017) doi:10.1016/j.icarus.2017.09.032.

## Authors

The authors of the package:

- [Evgeny Smirnov](https://github.com/smirik) ([FB](https://facebook.com/smirik), [Telegram](https://t.me/smirik))

## Acknowledgement

- Many thanks to the co-authors of the papers (prof. I. I. Shevchenko, I. Dovgalev, and Dr. E. Popova).
- The creators of [REBOUND integrator](https://rebound.readthedocs.io).
- The creators of [Astropy](http://astropy.org).
- The creators of `numpy`, `scipy`, `pandas`, and `matplotlib`.

## Contributing

Feel free to contribute to the code by sending pull requests [to the repository](https://github.com/smirik/resonances).

## License

MIT
