# Mean-Motion Resonances on steroids

`resonances` is an open-source package dedicated to the identification of mean-motion and secularresonances of small bodies. Many examples are for the Solar system; however, you might use the package for any possible planetary system, including exoplanets. For now, the package supports only eccentricity-type resonances. However, it will be improved in the future.

## Features

The package:

- can automatically identify two-body and three-body mean-motion resonance in the Solar system,
- can identify secular resonances (linear and non-linear),
- accurately differentiates different types of resonances (pure, transient, uncertain),
- provides an interface for mass tasks (i.e. find resonant areas in a planetary system),
- has integration with NASA Horizon (through astroquery package) and AstDyS catalog,
- can plot time series and periodograms,
- and, yeah, it is well tested ;)

It actively uses [REBOUND integrator](https://rebound.readthedocs.io) maintained by Hanno Rein and others.

## Mean-motion resonances

For those who are not familiar with the mean-motion resonances, here is the list of papers used to develop this package:

### Papers about the automatic identification of resonant asteroids

1. Smirnov, E. A. (2023). A new python package for identifying celestial bodies trapped in mean-motion resonances. Astronomy and Computing, 100707. [https://doi.org/10.1016/j.ascom.2023.100707](https://doi.org/10.1016/j.ascom.2023.100707)
1. Smirnov, E. A. & Shevchenko, I. I. Massive identification of asteroids in three-body resonances. Icarus 222, 220–228 (2013).
1. Smirnov, E. A., Dovgalev, I. S. & Popova, E. A. Asteroids in three-body mean motion resonances with planets. Icarus (2017) doi:10.1016/j.icarus.2017.09.032.
1. Smirnov, E. A. & Dovgalev, I. S. Identification of Asteroids in Two-Body Resonances. Solar System Research 52, 347–354 (2018).
1. Nesvorný, D. & Morbidelli, A. Three-Body Mean Motion Resonances and the Chaotic Structure of the Asteroid Belt. The Astronomical Journal 116, 3029–3037 (1998).

### Papers about mean-motion resonances

1. Smirnov, E. A. & Shevchenko, I. I. Massive Identification of Asteroids in Three-Body Resonances. Icarus 222, 220–228 (2013).
1. Smirnov, E. A., Dovgalev, I. S. & Popova, E. A. Asteroids in three-body mean motion resonances with planets. Icarus (2017) doi:10.1016/j.icarus.2017.09.032.
1. Nesvorný, D. & Morbidelli, A. Three-Body Mean Motion Resonances and the Chaotic Structure of the Asteroid Belt. The Astronomical Journal 116, 3029–3037 (1998).
1. Murray, N. & Holman, M. Diffusive chaos in the outer asteroid belt. The Astronomical Journal 114, 1246 (1997).
1. Murray, N., Holman, M. & Potter, M. On the Origin of Chaos in the Asteroid Belt. The Astronomical Journal 116, 2583–2589 (1998).
1. Smirnov, E. A. & Dovgalev, I. S. Identification of Asteroids in Two-Body Resonances. Solar System Research 52, 347–354 (2018).
1. Gallardo, T. Atlas of the mean motion resonances in the Solar System. Icarus 184, 29–38 (2006).
1. Gallardo, T. Strength, stability and three dimensional structure of mean motion resonances in the solar system. Icarus 317, 121–134 (2019).
1. Chirikov, B. V. A universal instability of many-dimensional oscillator systems. Physics reports 52, 263–379 (1979).
1. Gallardo, T., Coito, L. & Badano, L. Planetary and satellite three body mean motion resonances. Icarus 274, 83–98 (2016).
1. Shevchenko, I. I. On the Lyapunov exponents of the asteroidal motion subject to resonances and encounters. Proc. IAU 2, 15–30 (2006).
1. Milani, A., Cellino, A., Knezevic, Z., Novaković, B. & Spoto, F. Asteroid families classification: Exploiting very large datasets. Icarus 239, 46–73 (2014).
1. Smirnov, E. A Highly Resonant Neptunian Region: A Systematic Search for Two-Body and Three-Body Mean-Motion Resonances. Icarus 436, 116584 (2025).

### Books

1. Valerio Carruba, Evgeny Smirnov, Dagmara Oszkiewicz. Machine Learning for Small Bodies in the Solar System. (Elsevier, 2024). https://doi.org/10.1016/C2023-0-51021-3
1. Murray, C. D. & Dermott, S. F. Solar system dynamics. (Cambridge Univ. Press, 2012).
1. Morbidelli, A. Modern celestial mechanics: aspects of solar system dynamics. (2002).

## References

Whenever you use this package, we are kindly asking you to refer to one of the following papers (please choose the appropriate):

**The package itself**: Smirnov, E. A. (2023). A new python package for identifying celestial bodies trapped in mean-motion resonances. Astronomy and Computing. https://doi.org/10.1016/j.ascom.2023.100707

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

**The Libration module and automatic identification of librations**:

- Smirnov, E. A. (2023). A new python package for identifying celestial bodies trapped in mean-motion resonances. Astronomy and Computing, 100707. https://doi.org/10.1016/j.ascom.2023.100707

**Mass identification of mean-motion resonances:**

- Smirnov, E. A., & Dovgalev, I. S. (2018). Identification of Asteroids in Two-Body Resonances. Solar System Research, 52(4), 347–354. https://doi.org/10.1134/S0038094618040056
- Smirnov, E. A., Dovgalev, I. S. & Popova, E. A. Asteroids in three-body mean motion resonances with planets. Icarus (2017) doi:10.1016/j.icarus.2017.09.032.

## Authors

The authors of the package:

- [Evgeny Smirnov](https://github.com/smirik) ([FB](https://facebook.com/smirik), [Telegram](https://t.me/smirik))

## Acknowledgement

- Many thanks to the co-authors of the papers (prof. I. I. Shevchenko, I. Dovgalev Dr. E. Popova).
- The creators of [REBOUND integrator](https://rebound.readthedocs.io).
- The creators of [Astropy](http://astropy.org).
- The creators of `numpy`, `scipy`, `pandas`, and `matplotlib`.
