# Installation

`resonances` needs Python 3.11 or newer. It depends on [REBOUND](https://rebound.readthedocs.io) for the
integration and on [Astropy](https://www.astropy.org)/astroquery for NASA Horizons; both are installed
automatically.

## With uv (recommended)

[uv](https://docs.astral.sh/uv/) manages the Python version, the virtual environment and the dependencies:

```bash
uv init my-study        # a new project
cd my-study
uv add resonances
```

## With pip

In a [virtual environment](https://docs.python.org/3/tutorial/venv.html):

```bash
python -m venv .venv
source .venv/bin/activate
pip install resonances
```

## First run

Create `test.py`:

```python
--8<-- "getting_started.py:install"
```

and run it with `uv run python test.py` (or `python test.py`). The [Quick start](quick-start.md) explains the
output.

What happens on the first run:

- The elements of the asteroid and of the planets are downloaded from **NASA Horizons** at the start date (today,
  by default; the default data source is `DATA_SOURCE=nasa`). The planets are cached in `cache/solar-*.bin`.
  AstDyS and the MPC catalogue are optional sources ([Configuration](config.md), [Examples](examples.md)).
- The catalogues of resonances are built once and cached in `cache/mmr-2body.csv`, `cache/mmr-3body.csv`
  ([Resonance matrices](matrix.md)).
- The default integration is 100 000 years with `SABA(10,6,4)` and a saved point every
  15.9 years.
- The results go to `sim.config.save_path`: `cache/resonance_find` for `check` and `find` (their default name), or
  `cache/<name>` for `Simulation(name=...)`. If that folder is not empty, a time stamp is appended, so a new run
  never overwrites an old one. The log is `cache/resonances.log`.

All paths are relative to the working directory. Settings can be changed per run (`Simulation(...)` keywords) or in
a `.env` file in the working directory ([Configuration](config.md)).

If the download from Horizons fails with a certificate error, see [Horizons and TLS](config.md#horizons-and-tls).

## Jupyter

Add `ipykernel` to the same environment (`uv add ipykernel` or `pip install ipykernel`) and choose that
environment's kernel. On macOS and Windows, code that starts worker processes (batches of more than
`BATCH_THRESHOLD` bodies, clones) must run in a script with an `if __name__ == '__main__':` guard, not in a notebook
cell ([Examples](examples.md#clones)).

## From the source

```bash
git clone https://github.com/smirik/resonances.git
cd resonances
uv sync          # the package, the test tools and the documentation tools
make test-fast   # every test of at most 1 s, about 3 minutes
make test-slow   # integrations and longer tests, about 10 minutes (Horizons over the network)
make docs        # build this site (mkdocs build --strict)
```

Both need nothing beyond the clone. `make test-benchmark` needs the benchmark's saved series, which are not in git.

## Agent skill

For a coding agent (Claude Code and others that read Agent Skills), install the package's skill into your
project: `resonances skill install` (see [Using with AI agents](agents.md)). After upgrading the package, run
`resonances skill install --force` to refresh it.

## Updating

```bash
uv lock --upgrade-package resonances && uv sync     # uv
pip install --upgrade resonances                    # pip
```
