# Secular Resonances

A secular resonance is a commensurability between the slow precession of the body's perihelion
(`g`) or node (`s`) and the fundamental frequencies of the planets (`g1…g8`, `s1…s8`, Mercury…Neptune).

## Creating resonances

```python
import resonances

nu6 = resonances.create_secular_resonance('nu6')              # g-g6
nu14 = resonances.create_secular_resonance('nu14')            # s-s4
z1 = resonances.create_secular_resonance('g-g6+s-s6')
z2 = resonances.create_secular_resonance('2(g-g6)+(s-s6)')
other = resonances.create_secular_resonance('-g+s+g5-s7')
```

Aliases: `nu1`…`nu8` are `g-g1`…`g-g8`, `nu11`…`nu18` are `s-s1`…`s-s8` (no `nu15`: `s5` is
identically zero), `z1`…`z4`, `2nu6-nu5`, `3nu6-2nu5`. A formula needs at least one term of the body
(`g` or `s`); planet indices run from 1 to 8. A string is recognised as secular by
`detect_resonance_type` if it is an alias, starts with `g`/`s`, or parses as a formula (capital
letters are MMR notation: `4J-2S-1`).

## How the angle is built

With `secular_angle_mode='proper'` (the default), the angle that is classified, plotted and saved is

    sigma = p*varpi + q*Omega + sum_j (p_j*(g_j*t + phase_j) + q_j*(s_j*t + phase_j))

- the planets enter through their **mode longitudes**: `g_j*t + phase_j` with the frequencies
  `PLANETARY_FREQUENCIES` and the phases at J2000 `PLANETARY_PHASES` (`resonances/data/const.py`),
  shifted to the simulation date;
- the body's `Omega` is measured from the **invariable plane** (`INVARIABLE_PLANE`): its constant
  vector is subtracted from `(q, p) = sin(i/2)·(cos Ω, sin Ω)`.

Both matter. Measured from the ecliptic, the node of every planet is dominated by the constant
1.58° tilt of the invariable plane and does not circulate at all (Jupiter's and Saturn's mean rate
is 0, not `s6`); the osculating perihelia of Venus, Earth, Mars, Uranus and Neptune do not turn at
their own mode either (Uranus' turns at ~5.8″/yr, not `g7`). Only Mercury's, Jupiter's and Saturn's
perihelia follow their mode. So an angle built from the planets' osculating elements is wrong for
every term but `g1`, `g5`, `g6`: `z1 = g+s-g6-s6` of (847) Agnia turns at −26″/yr (exactly −s6)
instead of librating. The same holds for a body of low free inclination, whose ecliptic node
librates around the invariable node instead of turning at `s`.

**Reduced angle** (`secular_remove_forced=True`, the default). The body's e and i vectors carry
forced terms at every planetary fundamental. Those at frequencies that are *not* in the angle (g5 for
z1 or ν6, s7 for z2, …) only shake it, and when the free e or i is not much larger than the forced one
they decide whether the angle turns at all: (363) Padua's z1 angle from the osculating elements
circulates for 6 of 10 Myr while the resonance holds. The literature removes them by low-pass
filtering (Carruba et al. 2016 filter the z2 angle at 700 kyr "to eliminate all short-period
perturbations such as the g5 frequency"); here they are subtracted by least squares at the known
frequencies (`proper_angle.reduced_vectors`, with the fitting of `free_elements.remove_forced`), for the
fundamentals of the free-element basis absent from the angle. The terms at the angle's own frequencies
(g6, s6 for z1; g6 for ν6) are the resonance and stay — one rule for linear and nonlinear resonances.
Removing all of them would give the free-phase angle of Vokrouhlický et al. (2006, Agnia), which for a
linear resonance removes the resonance itself.

The angle from the osculating elements (same mode longitudes, nothing removed) is saved as
`<resonance>_angle_osculating`; `secular_angle_mode='osculating'` classifies the integration's angle
instead. A LIBRATION whose centre only drifts is lowered to −4 by the centre check (see
[Plots](plots.md), "Secular diagnostics", and [Config](config.md)).

## Planetary frequencies and phases

`g5…g8`, `s6…s8` are the AstDyS values (Knežević & Milani 2000). `g1…g4`, `s1…s4` and all the phases
are measured on the package's own model (Sun, eight planets, Pluto, Newtonian, Earth = Earth–Moon
barycentre) by a frequency analysis over 10 Myr: `experiments/2026-10-planetary-frequencies/measure.py`.
Against Laskar (1990, Icarus 88, 266, Tables II–III) the outer planets agree to 0.01–0.02″/yr and
1–2° in phase, `g3, g4, s3, s4` to 0.01–0.06″/yr; `g1` is lower by 0.38″/yr, `g2` by 0.12, `s1` higher
by 0.11 — the model has no general relativity. The inner frequencies drift by ~0.01″/yr over 10 Myr,
and their phases are uncertain by tens of degrees (this shifts a libration centre, not a status).
`s2` is split into several lines in this model and is the least certain of all.

## Which resonances to check

`SECULAR_FORMULAS` is the catalogue: the linear `ν_j`, `ν_1j` of all planets, the classical degree-4
set, `z1…z4` and the g-type terms (75 distinct angles). For a complete search around one body use
`resonances.secular.combinations.near_resonant(g, s, max_order, thresholds, modes)`: it builds every
angle allowed by the d'Alembert rules (coefficients summing to zero, an even number of node terms) up
to the given order and keeps those whose divisor `p g + q s + Σ…` is inside the window for its
order. Order 6 with all fifteen planetary modes gives ~900 angles within 1″/yr of a main-belt
body — the window and the modes have to be chosen physically (see `experiments/2026-09-binary-asteroids/scan_secular.py`).

`SecularResonanceFinder` evaluates the catalogue's divisors on AstDyS synthetic proper elements
(`resonances secular-candidates` on the command line).

## Example: (847) Agnia in z1

```python
import resonances
from resonances.secular.secular_resonance import SecularResonance

sim = resonances.Simulation(date='2026-09-29', integrator='SABA(10,6,4)', dt=0.5,
                            integration_years=2_000_000, Nout=20001)
sim.create_solar_system()
sim.add_body(847, [SecularResonance('g-g6+s-s6'), SecularResonance('s-s6')], name='Agnia')
sim.run()
print(sim.bodies[0].statuses)   # z1 librates (2), nu16 does not
```

## References

1. Laskar, J. (1990). The chaotic motion of the solar system. Icarus 88, 266.
2. Knežević, Z. & Milani, A. (2000). Synthetic proper elements for outer main belt asteroids. Celest. Mech. Dyn. Astron. 78, 17.
3. Milani, A. & Knežević, Z. (1990). Secular perturbation theory and computation of asteroid proper elements. Celest. Mech. Dyn. Astron. 49, 347.
4. Souami, D. & Souchay, J. (2012). The solar system's invariable plane. A&A 543, A133.
5. Vokrouhlický, D., Brož, M., Bottke, W. F., Nesvorný, D. & Morbidelli, A. (2006). The peculiar case of the Agnia asteroid family. Icarus 183, 349.
6. Carruba, V. (2009). The (not so) peculiar case of the Padua family. MNRAS 395, 358.
