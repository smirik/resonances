# Command line

Installing the package adds the `resonances` command (with uv: `uv run resonances ...`). `resonances --help` lists
the commands and `resonances <command> --help` shows the options of one.

| command | what it does |
|---|---|
| [`check`](#check) | integrate the given bodies and classify the given resonances |
| [`find`](#find) | look for candidate resonances of the given bodies, integrate them and classify every angle |
| [`dry-run`](#dry-run) | check a planned `find` or `check` run (output step, aliasing, integrator, cost) without integrating it |
| [`mpc-search`](#mpc-search) | select objects of the MPC catalogue by their elements and flags |
| [`mpc-update`](#mpc-update) | download the MPC catalogue again |
| [`secular-candidates`](#secular-candidates) | list the asteroids near a secular resonance in the AstDyS synthetic proper elements |
| [`skill install`](#skill-install) | install the agent skill into a project (or for the user) |

## How `check`, `find` and `dry-run` read their options

These three commands take any keyword of `Simulation` as an option: `--integration-years=50000` is
`integration_years=50000` ([Simulations, bodies, resonances](core.md#the-simulation); every key in
[Configuration](config.md)). Dashes become underscores. A value is read as an integer, a float, a boolean (`true`,
`false`, `yes`, `no`) or a string; a comma makes a list (`--asteroids=463,490`, `--planets=Jupiter,Saturn`), with no
shell quoting. `--key=value` and `--key value` both work; an option without a value is `True`. Unset options take the
defaults of `.env` and `.env.dist`, as in Python.

The options used most:

| option | default | meaning |
|---|---|---|
| `--asteroids` | (required) | numbers or designations, comma-separated |
| `--source` | `nasa` | elements from `nasa` (Horizons), `astdys` or `mpc` |
| `--date` | today (`nasa`), the catalog epoch (`astdys`, `mpc`) | start of the integration |
| `--integration-years` (or `--tmax`) | 100000 | the span in years (`tmax` in REBOUND units, 2π per year; negative = backwards) |
| `--integrator`, `--dt` | `SABA(10,6,4)`, `1.0` | the REBOUND integrator and its step (REBOUND units) |
| `--output-step` (or `--Nout`) | \|tmax\| / 100 points | years between saved points; `auto` lets the dry-run choose |
| `--save`, `--plot` | `review` | which rows get series and figures, by status ([Configuration](config.md#saving-and-plotting)) |
| `--save-path`, `--plot-path`, `--name` | `cache/resonance_find` | where the output goes ([Output data](output.md)) |
| `--clones`, `--clones-source`, `--clones-seed` | `0`, `auto`, `0` | virtual asteroids from the orbit's covariance ([Clones](clones.md)) |

`check` and `find` print the parameters they read and, at the end, the folder with the results.

## check

Checks the given resonances for the given bodies (`resonances.check`). Required: `--asteroids`, `--resonances` (MMR
keys such as `4J-2S-1`, secular formulas such as `g-g6`, or `LK` for Lidov-Kozai;
[the resonance keys](core.md#resonances)).

```bash
resonances check --asteroids=463 --resonances=4J-2S-1
resonances check --asteroids=463,490 --resonances=4J-2S-1,3J-1S-1 --integration-years=50000 --save=all --plot=all
resonances check --asteroids=463 --resonances=4J-2S-1 --source=astdys --clones=100
```

## find

Finds the candidate resonances of the given bodies and checks them all (`resonances.find`). Required:
`--asteroids`. Its own options:

| option | default | meaning |
|---|---|---|
| `--type` | all three | `mmr`, `secular`, `lidov_kozai` (comma-separated) |
| `--planets` | the package's default list | planets of the MMR search |
| `--sigma2`, `--sigma3`, `--sigma-retro` | `0.1`, `0.02`, `0.5` | how far (au) the resonant semi-major axis may lie from the body's: two-body, three-body, retrograde |
| `--formulas` | from the proper elements | secular resonances to check instead of the search |

```bash
resonances find --asteroids=463
resonances find --asteroids=463 --planets=Mars,Jupiter,Saturn --integration-years=100000 --type=mmr,secular --plot=resonant
```

## dry-run

Checks a run before integrating it ([Dry-run](dry-run.md)): the aliasing of every angle at the output step, the
`auto` step, the integrator and its step, the cost. It takes the options of `find` (or of `check` when
`--resonances` is given) and its own:

| option | default | meaning |
|---|---|---|
| `--years` | the run's span | the span of the run (`--integration-years`) |
| `--step` | the run's step | `auto` or a step in years (`--output-step`) |
| `--pilot` | off | a short fine integration per body (seconds each) instead of the elements alone |
| `--fair` | off | the step must also resolve the FAIR plot (`DRYRUN_FAIR=required`) |
| `--out` | none | write `dryrun_rows.csv`, `dryrun_bodies.csv` and `dryrun_run.json` into this folder |

```bash
resonances dry-run --asteroids=463 --years=100000 --step=auto
resonances dry-run --asteroids=1865 --resonances=10E-11+0+1 --years=20000 --integrator=ias15 --Nout=4000 --pilot
```

## mpc-search

Selects objects of the MPC catalogue (`resonances.mpcorb.query`; [Minor Planet Center](config.md#minor-planet-center-sourcempc)).
A range is `MIN:MAX`, inclusive; either end may be empty (`:0.3`, `2.5:`); angles may wrap (`350:10`).

| option | meaning |
|---|---|
| `--a`, `--e`, `--i`, `--q`, `--Q`, `--H` | semi-major axis (au), eccentricity, inclination (deg), perihelion and aphelion distances (au), absolute magnitude |
| `--node`, `--peri`, `--M` | longitude of the node, argument of perihelion, mean anomaly (deg) |
| `--n-obs`, `--n-opp`, `--arc-years`, `--U` | observations, oppositions, observed arc (years), uncertainty parameter 0–9 |
| `--epoch` | `any` (default) or only the catalogue's `standard` epoch |
| `--numbered` / `--provisional` | only numbered objects, or only provisional designations |
| `--orbit-type` | MPC orbit types, comma-separated: `atira`, `aten`, `apollo`, `amor`, `hungaria`, `hilda`, … |
| `--neo`, `--km-neo`, `--pha`, `--one-opp-earlier`, `--critical-list` | MPC's flags (each with a `--not-…` form) |
| `--catalog` | `mpcorb`, `nea`, … or a URL (default `MPC_CATALOG`) |
| `--limit` | rows printed, default 20 (the CSV gets every row) |
| `--output` | write every matching row to this CSV |
| `--ids` | print the identifiers, comma-separated, for `find --asteroids=...` |

```bash
resonances mpc-search --a 0:1 --e :0.3 --limit 20
resonances mpc-search --orbit-type=apollo --pha --H :18 --ids
```

## mpc-update

Downloads the MPC catalogue again and prints its snapshot (rows, standard epoch, checksum). This is the only way the
cached catalogue is refreshed.

| option | meaning |
|---|---|
| `--catalog` | `mpcorb`, `nea`, `neam00`, `distant`, `unusual` or a URL (default `MPC_CATALOG`) |
| `--force` | download even when the server reports no change |

```bash
resonances mpc-update --catalog=nea --force
```

## secular-candidates

Prints the names of the asteroids whose proper frequencies (AstDyS synthetic proper elements) put them near the given
secular resonance ([Secular resonances](secular.md)). The argument is the resonance: `z1`, `g-g6`, ….

| option | default | meaning |
|---|---|---|
| `--threshold` | `0.3` | the largest \|divisor\| of the resonance (the combination of frequencies), ″/yr |
| `--threshold_negative`, `--threshold_positive` | none | bounds of the divisor below and above zero, given together; they replace `--threshold` |
| `--numbered` | off | numbered asteroids only |
| `--debug` | `false` | print the counts on the way (takes a value: `--debug=true`) |

```bash
resonances secular-candidates z1 --threshold=0.3
resonances secular-candidates g-g6 --threshold_negative=-6 --threshold_positive=0.3 --numbered
```

## skill install

Copies the package's agent skill into `<project>/.claude/skills/resonances` ([Using with AI agents](agents.md)).

| option | meaning |
|---|---|
| `--path DIR` | the project folder (default: the current one) |
| `--user` | install into `~/.claude/skills` instead, for every project of the user |
| `--force` | replace an installed copy, e.g. after upgrading the package |

```bash
resonances skill install
resonances skill install --user --force
```
