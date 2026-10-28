# resonances

`resonances` is an open-source Python package that finds small bodies in resonances and decides, from a
numerical integration, whether they really are in them. It handles

- **mean-motion resonances**: two-body (including retrograde orbits and co-orbitals, 1:1) and three-body;
- **secular resonances**: linear and non-linear, with apsidal and nodal frequencies (ν6, z1, …);
- **Lidov-Kozai (ZLK) resonances**: libration of the argument of perihelion.

For each body and resonance the package integrates the orbit with the planets
([REBOUND](https://rebound.readthedocs.io)), builds the resonant angle, and gives it a verdict: a status, the
geometry of the angle, the physical origin of a libration, a confidence level and the next action. The data
come from NASA Horizons, AstDyS or the Minor Planet Center (MPCORB). The finders, catalogues and simulation workflow
target the Solar System.

## The statuses

| status | what it means physically |
|---|---|
| `resonant` | the resonant angle librates over the whole integration, with the evidence its type requires (MMR: the pendulum checks; secular: the angle's geometry; ZLK: the free-ω gate). For an MMR the dynamical origin may still be unknown: `reason` and `libration_origin` say whether it was measured |
| `transient` | the angle librates in episodes (a capture, an exit, or both) that do not cover the record as one sustained libration; circulation, a drift or a change of regime may separate them |
| `near_separatrix` | the angle stalls or slips at the top of its own potential well: the orbit sits on the border of the resonance |
| `non_resonant` | the angle circulates, or it is bounded by something else: the perihelion precession, another resonance, a too weak resonant term |
| `uncertain` | the record cannot decide: too short, or the physics contradicts the geometry |
| `invalid` | the record cannot be measured: the output step aliases the angle, or the orbit is unphysical |

How each status is reached, with the formulas and thresholds, is explained in
[How the status is determined](physics/overview.md); the [Gallery](physics/gallery.md) shows one example of
each.

## Where to start

- [Installation](install.md) and the [Quick start](quick-start.md): check one asteroid in three lines.
- [Simulations, bodies, resonances](core.md) and [Examples](examples.md): the main objects and practical
  recipes (catalogues, clones, many bodies, restore).
- [Configuration](config.md): every option; [Output data](output.md): every column.
- [How to read the figures](reading-figures.md): every panel of the verdict figures.
- [Rule index](libration-analysis.md): every rule of the classifier, with its settings.
- [Using with AI agents](agents.md): `resonances skill install` gives a coding agent (Claude Code and others)
  the package's skill: how to set up runs and how to read the verdicts.

```python
--8<-- "getting_started.py:index"
```

## Citing

If you use the package, please cite the paper listed in [About](about.md).
