# Mean-Motion Resonances

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Resonances](https://github.com/smirik/resonances/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/smirik/resonances/actions/workflows/ci.yml)

`resonances` is an open-source package dedicated to the identification of mean-motion resonances of small bodies. Many examples are for the Solar system; however, you might use the package for any possible planetary system, including exoplanets.

For more information, [read the documentation](https://smirik.github.io/resonances/).

**Note:** while this app has many functional and integration tests built in, it is still in the dev stage. Hence, it might include some inconsistencies. So, any community help is appreciated!

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

1. Smirnov, E. A. & Shevchenko, I. I. Massive identification of asteroids in three-body resonances. Icarus 222, 220–228 (2013).
1. Smirnov, E. A., Dovgalev, I. S. & Popova, E. A. Asteroids in three-body mean motion resonances with planets. Icarus (2017) doi:10.1016/j.icarus.2017.09.032.
1. Smirnov, E. A. & Dovgalev, I. S. Identification of Asteroids in Two-Body Resonances. Solar System Research 52, 347–354 (2018).
1. Nesvorný, D. & Morbidelli, A. Three-Body Mean Motion Resonances and the Chaotic Structure of the Asteroid Belt. The Astronomical Journal 116, 3029–3037 (1998).

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

1. **The package itself**: to be published
1. **The Libration module and automatic identification of librations**: to be published
1. **Mass identification of mean-motion resonances:** Smirnov, E. A., Dovgalev, I. S. & Popova, E. A. Asteroids in three-body mean motion resonances with planets. Icarus (2017) doi:10.1016/j.icarus.2017.09.032.

## Authors

The authors of the package:

- [Evgeny Smirnov](https://github.com/smirik) ([FB](https://facebook.com/smirik), [Telegram](https://t.me/smirik))

## Acknowledgement

- Many thanks to the co-authors of the papers (prof. I. I. Shevchenko, I. Dovgalev Dr. E. Popova).
- The creators of [REBOUND integrator](https://rebound.readthedocs.io).
- The creators of [Astropy](http://astropy.org).
- The creators of `numpy`, `scipy`, `pandas`, and `matplotlib`.

## Contributing

Feel free to contribute to the code by sending pull requests [to the repository](https://github.com/smirik/resonances).

## License

MIT
