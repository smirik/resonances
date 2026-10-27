<!-- Generated from docs/quick-start.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Quick start

Three checks: a mean-motion resonance, a secular resonance and a Lidov-Kozai resonance. Each integrates the orbit
of an asteroid with the planets, builds the resonant angle and gives it a verdict
([How the status is determined](physics/overview.md)).

## A mean-motion resonance

(463) Lola is in the three-body resonance `4J-2S-1`, with the angle
$\sigma = 4\lambda_J - 2\lambda_S - \lambda - \varpi$ ([Resonance matrices](matrix.md) explain the notation).

```python
import resonances

# (463) Lola in the three-body resonance 4J-2S-1 (4 lambda_J - 2 lambda_S - lambda + ...)
sim = resonances.check(463, '4J-2S-1', integration_years=2_000)  # a real verdict: integration_years=100_000, output_step=1.0
sim.run()

body = sim.bodies[0]
for key, verdict in body.librations.items():  # key: the full angle, 4J-2S-1+0+0-1
    print(key, verdict.status, verdict.angle_status, verdict.confidence, verdict.reason, verdict.action)
print(sim.config.save_path)  # summary.csv, the series and the figures are written here
```

`check` downloads the elements from NASA Horizons at today's date, integrates, analyses the angle and saves the
results. The code above runs only 2 000 years, so the verdict is

```text
4J-2S-1+0+0-1 uncertain insufficient_time  baseline_needed extend
```

The libration period of Lola is about 10 000 years: 2 000 years hold no full cycle, the status is `uncertain`, the
reason `baseline_needed`, and the suggested action `extend` (`verdict.extend_to_years` says how long to integrate).
With `integration_years=100_000, output_step=1.0` the verdict is (from the [Gallery](physics/gallery.md#quick-start-463)
run: 100 kyr, step 1 yr, SABA(10,6,4); not re-run by the script above)

| status | angle_status | reason | confidence | action |
|---|---|---|---|---|
| `resonant` | `libration` | `dynamic_libration` | `high` | `none` |

The output step matters: without `output_step` the default step of a 100 kyr run is about 15.9 yr, longer than half
of Lola's 3.7-yr orbit, so the record folds short periods and the confidence is capped at `medium`
(`short_period_aliased`, [Dry run](dry-run.md)).

The fields:

- `status`: the answer, one of `resonant`, `transient`, `near_separatrix`, `non_resonant`, `uncertain`, `invalid`;
- `angle_status`: what the angle does over the whole record (`libration`, `episodic`, `circulation`, ...);
- `reason` and `libration_origin`: why (`dynamic_libration`: the angle librates and the semi-major axis follows it
  as the pendulum of this resonance requires);
- `confidence`: how well the evidence supports the status (`certain`, `high`, `medium`, `low`);
- `action`: what to do next (`none`, `spot_check`, `review`, `extend`, `resample`).

The [overview](physics/overview.md) defines every value. All of them are also in `summary.csv`:

```python
summary = sim.data_manager.get_simulation_summary(sim.bodies)
print(summary[['name', 'resonance', 'status', 'angle_status', 'confidence', 'reason', 'action']])
```

## The figure

For every row selected by the `plot` mode (by default `nonzero`: everything except a `non_resonant` angle that
nothing bounds, such as a plain circulation) a one-page figure is
saved in `<save_path>/combined/<confidence>/<status>/`: the angle, the semi-major axis and the pendulum's energy
in time on the left, and the verdict with every number behind it on the right. This is Lola over 100 000 years at a
1-year output step:

[![463 Lola in 4J-2S-1](https://smirik.github.io/resonances/img/gallery/mmr3__463__4J-2S-1+0+0-1.png)](https://smirik.github.io/resonances/img/gallery/mmr3__463__4J-2S-1+0+0-1.png)

The [Gallery](physics/gallery.md#quick-start-463) explains how to read it and shows one example of every status.

## A secular resonance

(1222) Tina is in the linear secular resonance $\nu_6$, `g-g6`: the angle is $\varpi - (g_6 t + \phi_6)$, the
body's longitude of perihelion against the phase of the planetary secular mode $g_6$ (by default the body's forced
terms at the other planetary frequencies are removed first; [Secular and Lidov-Kozai](physics/secular-zlk.md)).

```python
# (1222) Tina in the linear secular resonance nu6 (angle varpi - varpi of Saturn's mode g6)
sim = resonances.check(1222, 'g-g6', integration_years=5_000)  # 2-10 Myr for a real verdict
sim.run()
print(sim.bodies[0].statuses)  # {'g-g6': 'uncertain'} on this short run
```

Secular angles librate with periods of hundreds of thousands of years, so a real check needs 2–10 Myr
(`integration_years=2_000_000` or more); over 10 Myr Tina is `resonant` with `certain` confidence
([Gallery](physics/gallery.md#secular-nu6)). See [Secular resonances](secular.md).

## A Lidov-Kozai resonance

The Lidov-Kozai angle is the argument of perihelion $\omega$. (3040) Kozai librates about 270°:

```python
# (3040) Kozai: the Lidov-Kozai angle is the argument of perihelion omega
sim = resonances.check(3040, resonances.LidovKozaiResonance(), integration_years=5_000)  # 500 kyr for a real verdict
sim.run()
print(sim.bodies[0].statuses)  # {'LK': ...}
```

Over 500 000 years the verdict is `resonant` with `certain` confidence ([Gallery](physics/gallery.md#zlk-resonant)).
See [Lidov-Kozai resonances](lidov-kozai.md).

## Next

- [Simulations, bodies, resonances](core.md): build a simulation yourself, several bodies and resonances, the
  settings.
- [Examples](examples.md): finding resonances, other data sources, clones, output layouts, restoring a run.
- [How the status is determined](physics/overview.md): the physics behind each verdict.
- [Using with AI agents](https://smirik.github.io/resonances/agents/): working through a coding agent? `resonances skill install` gives it the
  package's skill.
