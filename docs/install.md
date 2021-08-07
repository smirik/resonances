# Installation

## Poetry

The best way to install `resonances` package is through [poetry](https://python-poetry.org), which is the dependency manager for python. In this case, you only need to write:

```bash
poetry add resonances
```

It will install the package itself, as well as all dependencies.

You might face some difficulties through the installation for a new project. Below is a detailed guide.

### Guide

**A.** Follow the instructions on the official website of [poetry](https://python-poetry.org) and install it. Of course, python, pip, virtual environments and everything related should be already installed.

**B.** Create a new project:

```bash
poetry new example
cd example
```

**C.** If you now run `poetry add resonances`, you might face the issue that `resonances` is not compatible with the default `python` version that poetry has created. It happens because scipy has a requirement for python `<3.10`, whereas poetry has less explicit requirement `<4`. To fix that, open `pyproject.toml` file and replace the version of python to `>=3.7.1,<3.10`:

```
[tool.poetry.dependencies]
python = ">=3.7.1,<3.10"
```

**D.** Now you can add `resonances`:

```bash
poetry add resonances
```

**E.** Now you can run your first simulation. Create `test.py` in the root directory of your project and place the following code there:

```python
import resonances

mmr = resonances.create_mmr('1J-1')
sim = resonances.Simulation()
sim.create_solar_system()
sim.add_body(588, mmr, name='Asteroid 558')
sim.dt=1.0
sim.run()
```

Note that the first run might take a while because the app needs to download AstDyS catalogue and initial data for the Solar system. You can see the progress in `cache/resonances.log` file.

**F.** Now you can see the results in `cache` folder.

## PIP

However, if you prefer manual installation, you need to perform the following steps:

```bash
pip install resonances
```

Note that while `numpy`, `scipy`, `pandas`, and `mathplotlib` are quite common, the package also requires [REBOUND integrator](https://rebound.readthedocs.io/en/latest/) and [Astropy](https://www.astropy.org). While it should be handled automatically, if something goes wrong, please verify that these packages are installed.

It is highly recommended to use [virtual environments](https://docs.python.org/3/tutorial/venv.html) to avoid possible conflicts.

After installation, you can create the file `test.py` with the same content as described in the previous section and run it:

```python
python test.py
```

## Jupyter Notebooks

Generally, it should work out of the box, just follow [any guide](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/) that explains how to add a package to [Jupyter Notebooks](https://jupyter.org) and the instructions above.

If you use virtual environment and [VSCode](https://code.visualstudio.com), do not forget to choose the right Python Kernel. Also, it is required to install `ipykernel` from the console (related to your virtual environment):

```bash
pip install ipykernel
```

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

```bash
poetry update
```
