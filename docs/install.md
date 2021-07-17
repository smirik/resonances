# Installation

## Poetry

The best way to install `resonances` package is through [poetry](https://python-poetry.org), which is the dependency manager for python. In this case, you only need to write:

```bash
poetry add resonances
```

It will install the package itself, as well as all dependencies.

## Through `pip`

However, if you prefer manual installation, you need to perform the following steps:

```bash
pip install numpy, scipy, matplotlib, pandas, astropy, rebound
pip install resonances
```

Note that while `numpy`, `scipy`, `pandas`, and `mathplotlib` are quite common, the package also requires [REBOUND integrator](https://rebound.readthedocs.io/en/latest/) and [Astropy](https://www.astropy.org).

## Manual installation

If for some reason, you want to launch the project from the source code, you need to download it from `github` and install dependencies through `poetry`:

```bash
git clone https://github.com/smirik/resonances.git
cd resonances
poetry install
```

You may verify the installation by running tests:

```bash
poetry run pytest -v tests
```

Note that you have to install prior to these steps `python` (preferably, through `pyenv`) and set up a virtual environment for poetry.

## Updating

Use the same manager (`poetry` or `pip`) and its standard workflow to update.
