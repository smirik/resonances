## Overview

This package can identify mean-motion resonances (MMRs), secular resonances and Lidov-Kozai resonances in the Solar system. It loads the data from AstDyS catalog or NASA Horizon, set up a simulation, perform numerical integration using rebound python package (integrator), build resonant angles and periodograms for some times series, classify the results, depending on the configuration, save the data and plots to files.

Core modules in /resonances directory:

- cli - support for command line interface
- data - utility functions for data handling, constants, etc.
- finder - high-level interface to different resonance finders (MMR, secular, Lidov-Kozai)
- lidov_kozai - Lidov-Kozai resonance module; provides class for Lidov-Kozai resonance, methods to compute the Lidov-Kozai invariants, etc.
- matrix - (abstract by logic) base class for MMRs catalogs (2-body and 3-body); his children are TwoBodyMatrix and ThreeBodyMatrix. They contain a list of all considered relevant resonances and their resonant semi-major axes.
- mmr - MMR module; provides abstract class for MMR and TwoBody and ThreeBody classes implementing the MMR interface; provides Matrix classes for them; provides finder functions for finding MMRs
- plotting - provides classes for plotting time series and periodograms and their flexible configurator
- resonance - core components of the package; classify - classify the resonant angle time series into libration, circulation or chaotic behavior; factory - factory for creating different types of resonances; filtering - filter time series using scipy filters to remove noise; libration - old class, not used now; periodogram - provides class for building and analyzing Lomb-Scargle periodograms through astropy python package; planets_mappings - utility class to map a letter to the planet name and vice versa; resolver - resolver for the final status of MMRs only (need overlapping of the periodogram peaks of semi-major axis and resonant angle); resonance - abstract class Resonance for all types of resonances;
- secular - secular resonance module; provides class for secular resonance, secular formula parser, finder for secular resonances and functions to build the proper angle from the osculating elements (where for the planets a linear model is used instead of the real Keplerian elements);
- simulation - simulation module with core simulation classes and functions; batch_manager - manages multiple batches (use multiple cores if possible), body_manager - manages bodies in the simulation, data_manager - manages data saving for simulation itself and its bodies / results of identification; integration - performs numerical integration using rebound python package and invokes data storage during the simulation, serializer - serializes and restores simulations that were interrupted, state_manager - manages simulation state for batches and interrupted simulations;

Files in resonances/:

- body.py - class Body used in the simulation; contains structure of the data and other important vars
- config.py - load config from .env.dist and .env files if presented; allows to rewrite config values at runtime.
- horizons.py - get Keplerian elements from NASA Horizon for a given body.
- logger.py - logger for the package; allows to log messages with different levels. (info used for something useful during the development, error - for critical errors (e.g., asteroid's initial data cannot be retrieved), warning - something that does not affect but worth mentioning, e.g., when eccentricity becomes too large)
- .env.dist - default config values for the package.

## Important notes

- Note that this is a scientific project. The methods and functions must be very accurately tested and validated and stay reliable.

## Documenting

- In /docs there is actual documentation of the project. It's written for humans. Keep it up to date. Keep it short and concise but clear for understanding. Don't write too much.
- In functions or classes, you may add comments for the files and functions if this is not obvious from the code or the names. E.g., the function sum(a,b) does not require any comments, whereas the function resolve_mmr_status requires a comment explaining what are the statuses and how they are resolved. But really keep it simple.
- If you change a function or a class, update the documentation inside the code accordingly.
- Update overall documentation at the end of the task.

## How to run

Everything should be run through `uv run` (which automatically manages the virtual environment). Use `uv sync` to install dependencies.

## Testing

Any functionality except temporary should be tested. The package has the following conventions:

- black for code formatting and flake8 for code quality. Run every time before saying "done".
- `make test-fast` to run all fast tests that do not require numerical integration, which takes time.
- `make test-slow` to run all slow tests that require numerical integration, which takes enough time. Run them at the end of the task when all other fast tests and formatting are done and passed.
- `make test` to run all tests - fast and slow.
- never change real (slow) tests unless you are absolutely sure that you know what you do; ask me if you are not sure.

For tests:

- Prefer functional tests over unit tests and mocks.
- When you create a new test, make sure that it tests a case obtained from different sources. For example, if you need to test a sum(a,b) function, calculate first a few examples independently (e.g., 2+3=5, 4+9=13) and then use these examples to test the function in a functional way.
