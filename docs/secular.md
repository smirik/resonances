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
instead. A LIBRATION whose smoothed centre moves but turns back fewer than
`secular_min_turning_points` times (3: a cycle and a half) is lowered to −6 by the centre check (see
[Plots](plots.md), "Secular diagnostics", and [Config](config.md)).

**Free divisor** (summary.csv `free_divisor`, ″/yr; a diagnostic, no status depends on it):
`p g + q s + Σ…` with the body's own `g` and `s` — the rates of its free vectors, every fundamental
removed (`proper_angle.free_rates`). Near 0 the free vector itself keeps pace with the planets' modes
(16712 and 48717 in ν6, the z1 and z2 librators). A libration whose period is `2π/|free divisor|` is
one around the forced equilibrium: the angle's own forced term is larger than the free amplitude and
the free vector turns at its own rate — Tina in ν6 (2.18″/yr, 592 kyr), the anti-aligned ν6 libration
of Carruba & Morbidelli (2011). The two cannot be told apart by this number alone, so it changes nothing.

## Planetary frequencies and phases

`g5…g8`, `s6…s8` are the AstDyS values (Knežević & Milani 2000). `g1…g4`, `s1…s4` and all the phases
are measured on the package's own model (Sun, eight planets, Pluto, Newtonian, Earth = Earth–Moon
barycentre) by a frequency analysis over 10 Myr: `experiments/2026-10-planetary-frequencies/measure.py`.
Against Laskar (1990, Icarus 88, 266, Tables II–III) the outer planets agree to 0.01–0.02″/yr and
1–2° in phase, `g3, g4, s3, s4` to 0.01–0.06″/yr; `g1` is lower by 0.38″/yr, `g2` by 0.12, `s1` higher
by 0.11 — the model has no general relativity. The inner frequencies drift by ~0.01″/yr over 10 Myr,
and their phases are uncertain by tens of degrees. `s2` is split into several lines in this model and
is the least certain of all.

**Measured phases** (`secular_planet_phases='measured'`, needs `save_planets=True`). Over tens of
Myr the linear law drifts away from the integrated planets: in a 20 Myr run of the package's model the
phase of s4 is off by 80° after 17 Myr, s3 by 23°, g3 and s2 by 20°, and even s6 by 16° (the AstDyS
`s6` is 0.003″/yr off the model's). `resonances/secular/planet_modes.py` fits the phase of mode `j`
in planet `j`'s own vector (`e·exp(iϖ)` for `g_j`, `sin(i/2)·exp(iΩ)` for `s_j`) by least squares
on all eight modes of that kind, in windows of `secular_phase_window` (5 Myr: g3 and g4 are 0.54″/yr
apart), a tenth of a window apart, and builds the longitude as `g_j t + φ_j(t)`. Within half a window
of either end the phase goes on along its slope. A baseline shorter than the window keeps the linear
law: a shorter fit cannot separate g3 from g4 (over 200 kyr it puts g3 51° off).

## Which resonances to check

`SECULAR_FORMULAS` is the catalogue: the linear `ν_j`, `ν_1j` of all planets, the classical degree-4
set, `z1…z4` and the g-type terms (75 distinct angles). For a complete search around one body use
`resonances.secular.combinations.near_resonant(g, s, max_order, thresholds, modes)`: it builds every
angle allowed by the d'Alembert rules (coefficients summing to zero, an even number of node terms) up
to the given order and keeps those whose divisor `p g + q s + Σ…` is inside the window for its
order. Order 6 with all fifteen planetary modes gives ~900 angles within 1″/yr of a main-belt
body — the window and the modes have to be chosen physically (see `experiments/2026-09-binary-asteroids/scan_secular.py`).

The catalogue is wider than what the literature reports. Librations are shown for ν5, ν6, ν16, ν4
(one Hungaria), z1, z2, g+g5−2g6, g+2g5−3g6, 2g−3g5+g6, g−g5+s−s7, 2g−g5−g6+s−s6 and, among the
Hungarias, g−s−g5+s6; s−g5+g8−s6, s+g5−g8−s3, 2s−s4−s6 and 2g−g5−g6 are located there but shown only as
temporary interactions (Warner et al. 2009; Milani et al. 2010; Knežević 2022). Nothing is reported for
combinations with `g1, g2, s1, s2` or mixed degree-4 terms of the inner planets with Uranus or Neptune;
their divisors in this model are also off by 0.1–0.3″/yr (no general relativity). A degree-4 resonance
(half-width ~0.3″/yr) librates in ~9 Myr, a degree-6 one (~0.1″/yr) in ~26 Myr: the baseline has to
hold at least a cycle and a half (`experiments/2026-09-binary-asteroids/sim_secular_v5.py`, 50 Myr).

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
7. Carruba, V. & Morbidelli, A. (2011). On the first ν6 anti-aligned librating asteroid family of Tina. MNRAS 412, 2040.
8. Warner, B. D., Harris, A. W., Vokrouhlický, D., Nesvorný, D. & Bottke, W. F. (2009). Analysis of the Hungaria asteroid population. Icarus 204, 172.
9. Milani, A., Knežević, Z., Novaković, B. & Cellino, A. (2010). Dynamics of the Hungaria asteroids. Icarus 207, 769.
10. Knežević, Z. (2022). Survey of secular resonances in the asteroid belt. Serb. Astron. J. 204, 1.
