# Installation

The fastest way:

```bash
pip install resonances
```

## uv (recommended)

The _best_ way to install and manage `resonances` is through [uv](https://docs.astral.sh/uv/), which is an extremely fast Python package and project manager. In this case, you only need to write:

```bash
uv add resonances
```

It will install the package itself, as well as all dependencies.

You might face some difficulties through the installation for a new project. Below is a detailed guide.

### Guide

**A.** Follow the instructions on the official website of [uv](https://docs.astral.sh/uv/) and install it.

**B.** Create a new project:

```bash
uv init example
cd example
```

**C.** Add `resonances`:

```bash
uv add resonances
```

**D.** Now you can run your first simulation. Create `test.py` in the root directory of your project and place the following code there:

```python
import resonances

sim = resonances.check(463, resonance='4J-2S-1')
sim.run()
```

Note that the first run might take a while because the app needs to download AstDyS catalogue and initial data for the Solar system. You can see the progress in `cache/resonances.log` file.

**E.** Now you can see the results in `cache/%current_datetime%` folder.

## PIP

However, if you prefer manual installation, you need to perform the following steps:

```bash
pip install resonances
```

Note that while `numpy`, `scipy`, `pandas`, and `matplotlib` are quite common, the package also requires [REBOUND integrator](https://rebound.readthedocs.io/en/latest/) and [Astropy](https://www.astropy.org). While it should be handled automatically, if something goes wrong, please verify that these packages are installed.

It is highly recommended to use [virtual environments](https://docs.python.org/3/tutorial/venv.html) to avoid possible conflicts.

After installation, you can create the file `test.py` with the same content as described in the previous section and run it:

```python
python test.py
```

## Jupyter Notebooks

Generally, it should work out of the box, just follow [any guide](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/) that explains how to add a package to [Jupyter Notebooks](https://jupyter.org) and the instructions above.

If you use virtual environment and [VSCode](https://code.visualstudio.com), do not forget to choose the right Python Kernel. Also, it is required to install `ipykernel` from the console (related to your virtual environment):

```bash
uv add ipykernel
```

or with pip:

```bash
pip install ipykernel
```

## Manual installation

If for some reason, you want to launch the project from the source code, you need to download it from `github` and install dependencies through `uv`:

```bash
git clone https://github.com/smirik/resonances.git
cd resonances
uv sync
```

You may verify the installation by running tests:

```bash
uv run pytest -v tests
```

or simply `make test`

## Updating

Use the same manager (`uv` or `pip`) and its standard workflow to update.

```bash
uv lock --upgrade-package resonances
uv sync
```

or with pip:

```bash
pip install --upgrade resonances
```

## Common issues

An error when retrieving data from JPL Horizons (SSL certificate issue):

```bash
pip install --upgrade certifi
```

It will install the latest version of `certifi` package, which is required by `astroquery` to work properly.
