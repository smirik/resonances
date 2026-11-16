<!-- Generated from docs/physics/gallery.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Gallery

One benchmark example for every status, every way of reaching it and every resonance type, each with its one-page
verdict figure. The figures are drawn by the package itself from the saved integrations; the verdict line under
each caption is the package's verdict on that record (`status` · `angle_status` · `reason` · `confidence`), so a
caption and its figure always agree. Click a figure to open it at full size.

A resonance key lists integer coefficients: first those of the mean longitudes of the planets (J Jupiter, S Saturn,
E Earth, M Mars, V Venus, N Neptune) and of the body, then those of the longitudes of perihelion in the same order
(and of the nodes, when present): `4J-2S-1+0+0-1` is $4\lambda_J - 2\lambda_S - \lambda - \varpi$. See
[Resonance matrices](../matrix.md). Secular keys name the frequencies ($g - g_6 + s - s_6$ is z1), `LK` is the
Lidov-Kozai angle $\omega$.

## How to read the figure

The left half of each figure holds the time series and the phase portraits, the right half the verdict: the plaque
with the status, the verdict in words, the steps of the [status tree](status-tree.md), the evidence the
[confidence](confidence.md) level rests on and the numbers behind them, each a column of the [output files](../output.md). What
every panel, line and icon means: [How to read the figures](../reading-figures.md).

## Resonant

The angle librates over the whole record and the libration belongs to this resonance.

### A two-body resonance: the textbook pendulum {#resonant-two-body}

$\sigma$ librates about a fixed center over the whole record, $a$ oscillates in step with it, and every test of the force law and of the pendulum's energy passes.

**1921**, `2J-1+0-1`: `resonant` · `libration` · `dynamic_libration` · `certain`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr2__1921__2J-1+0-1](https://smirik.github.io/resonances/img/gallery/mmr2__1921__2J-1+0-1.png)](https://smirik.github.io/resonances/img/gallery/mmr2__1921__2J-1+0-1.png)

### A three-body resonance with Jupiter and Saturn {#resonant-three-body}

The same reading for a three-body angle: $\sigma$ librates over the whole record and $a$ follows it as the pendulum of this resonance requires.

**2487**, `4J-2S-1+0+0-1`: `resonant` · `libration` · `dynamic_libration` · `certain`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr3__2487__4J-2S-1+0+0-1](https://smirik.github.io/resonances/img/gallery/mmr3__2487__4J-2S-1+0+0-1.png)](https://smirik.github.io/resonances/img/gallery/mmr3__2487__4J-2S-1+0+0-1.png)

### The body of the quick start {#quick-start-463}

The resonance checked in the [Quick start](../quick-start.md). The libration is as clear as above: about nine cycles over the whole record, more than the three that certain needs, the amplitude clear of the separatrix and nothing objecting, so the confidence is certain.

**463**, `4J-2S-1+0+0-1`: `resonant` · `libration` · `dynamic_libration` · `certain`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr3__463__4J-2S-1+0+0-1](https://smirik.github.io/resonances/img/gallery/mmr3__463__4J-2S-1+0+0-1.png)](https://smirik.github.io/resonances/img/gallery/mmr3__463__4J-2S-1+0+0-1.png)

### An Earth co-orbital {#resonant-coorbital}

The angle $\lambda - \lambda_E$ goes from horseshoe motion to a compound orbit (horseshoe and quasi-satellite in turn) and back to horseshoe, and never circulates: one episode in three sub-regimes. A 1:1 angle is judged by its own force law, from the averaged direct and indirect potential of the planet.

**2002 AA29**, `1E-1+0+0`: `resonant` · `libration` · `dynamic_libration` · `certain`. Record: 1 kyr, step 0.05 yr, ias15 + Moon.

[![coorbital__2002_AA29__1E-1+0+0](https://smirik.github.io/resonances/img/gallery/coorbital__2002_AA29__1E-1+0+0.png)](https://smirik.github.io/resonances/img/gallery/coorbital__2002_AA29__1E-1+0+0.png)

### Two co-orbital regimes in turn {#coorbital-regime-change}

Quasi-satellite motion, a compound orbit, then horseshoe motion: regimes of the same 1:1 resonance follow each other without a circulation in between. A 1:1 episode ends only at a separatrix, so this is one episode with changes of well (the mark well_change in episodes.csv; subtype regime_change). The header names the wells in time order (`orbit: …`) and (a) brackets each sub-regime with its well.

**164207**, `1E-1+0+0`: `resonant` · `libration` · `dynamic_libration` · `certain`. Record: 1 kyr, step 0.05 yr, ias15 + Moon.

[![coorbital__164207__1E-1+0+0](https://smirik.github.io/resonances/img/gallery/coorbital__164207__1E-1+0+0.png)](https://smirik.github.io/resonances/img/gallery/coorbital__164207__1E-1+0+0.png)

### Libration about a moving center {#moving-center}

The center of libration follows the slowly moving minimum of the averaged potential; the libration is tested in $\psi = \sigma - \sigma_c(t)$. The output step folds short periods: the flag short_period_aliased warns but lowers nothing. 131 cycles would give certain, but another resonance comes close to driving $a$ (best $R^2$ 0.63, margin W thin), so the objections count as only partly measured and the confidence is high.

**2013 AS27**, `31E-36+0+5`: `resonant` · `libration` · `dynamic_libration` · `high`. Record: 20 kyr, step 5 yr, ias15.

[![mmr2__2013_AS27__31E-36+0+5](https://smirik.github.io/resonances/img/gallery/mmr2__2013_AS27__31E-36+0+5.png)](https://smirik.github.io/resonances/img/gallery/mmr2__2013_AS27__31E-36+0+5.png)

### A slow libration on a long record {#center-scale}

A high-order resonance with Neptune whose libration period is long (64 kyr): over this record the angle closes 4.6 cycles and is resonant. The other resonances are tested on two windows fitted to this short record: none of 14 drives $a$, and the confidence is certain, with no action asked (the force law, which needs two windows of 3.5 periods, is not tested on this record). The same body on a shorter record is [uncertain](#insufficient-time).

**385458**, `9N-17+0+8`: `resonant` · `libration` · `dynamic_libration` · `certain`. Record: 300 kyr, step 1 yr, SABA(10,6,4).

[![mmr2__385458__9N-17+0+8__300kyr](https://smirik.github.io/resonances/img/gallery/mmr2__385458__9N-17+0+8__300kyr.png)](https://smirik.github.io/resonances/img/gallery/mmr2__385458__9N-17+0+8__300kyr.png)

### Libration decided by the geometry alone {#geometric-libration}

$\sigma$ librates over the whole record, but neither the force law nor the energy of the pendulum could be measured (a half-amplitude of 21°: $\sin\sigma \approx \sigma$). The geometry decided (reason geometric_libration, origin unknown). The resonance's strength agrees with the period of the cycles to an order of magnitude (margin F), so the level is not held at the low of a geometry-only row; nothing objects, and 14 clean cycles give certain. The missing measurements are a flag (no_full_physics_support).

**2348**, `4J-2S-1+0+0-1`: `resonant` · `libration` · `geometric_libration` · `certain`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr3__2348__4J-2S-1+0+0-1](https://smirik.github.io/resonances/img/gallery/mmr3__2348__4J-2S-1+0+0-1.png)](https://smirik.github.io/resonances/img/gallery/mmr3__2348__4J-2S-1+0+0-1.png)

### A libration the physics questions {#measured-objection}

$\sigma$ librates over the whole record (its center moves, but the bands of the cycles overlap), but the energy test of the pendulum fails. The objection is measured but does not decide (flag measured_objection): the tree's leaf said uncertain, the minimal policy keeps the geometry's status, resonant, at low (a measured objection, and the angle's own force is not shown), with action extend.

**2003 TX9**, `3E+2J-10+0+0+5`: `resonant` · `libration` · `geometric_libration` · `low`. Record: 20 kyr, step 5 yr, ias15.

[![mmr3__2003_TX9__3E+2J-10+0+0+5](https://smirik.github.io/resonances/img/gallery/mmr3__2003_TX9__3E+2J-10+0+0+5.png)](https://smirik.github.io/resonances/img/gallery/mmr3__2003_TX9__3E+2J-10+0+0+5.png)

## Transient

Librating episodes in part of the record, circulation or a drift in the rest.

### Repeated capture and release {#transient-recurrent}

Librating episodes alternate with circulation (subtype recurrent). The deciding episode closes 3.5 cycles, enough for certain, but the episodes reach the separatrix zone (flag episodes_in_separatrix_zone: the amplitude margin is thin), so the confidence is high.

**52356**, `5J-2S-2+0+0-1`: `transient` · `episodic` · `dynamic_episode` · `high`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr3__52356__5J-2S-2+0+0-1](https://smirik.github.io/resonances/img/gallery/mmr3__52356__5J-2S-2+0+0-1.png)](https://smirik.github.io/resonances/img/gallery/mmr3__52356__5J-2S-2+0+0-1.png)

### A co-orbital that leaves the 1:1 resonance {#transient-exit}

A horseshoe libration over most of the record. The exit is seen in the orbit-averaged angle: after it $\lambda - \lambda_E$ circulates to the end of the record (final_regime circulation).

**2017 UM52**, `1E-1+0+0`: `transient` · `episodic` · `ends: circulation` · `dynamic_episode` · `high`. Record: 1 kyr, step 0.05 yr, ias15 + Moon.

[![coorbital__2017_UM52__1E-1+0+0](https://smirik.github.io/resonances/img/gallery/coorbital__2017_UM52__1E-1+0+0.png)](https://smirik.github.io/resonances/img/gallery/coorbital__2017_UM52__1E-1+0+0.png)

### An episode, then circulation {#transient-capture-exit}

A libration episode followed by circulation to the end of the record. The episode closes only one cycle, fewer than the 1.5 that medium needs, so the confidence is low and the action asks for a review.

**115978**, `11M-19+0+8`: `transient` · `episodic` · `ends: circulation` · `dynamic_episode` · `low`. Record: 50 kyr, step 0.6 yr, ias15.

[![mmr2__115978__11M-19+0+8](https://smirik.github.io/resonances/img/gallery/mmr2__115978__11M-19+0+8.png)](https://smirik.github.io/resonances/img/gallery/mmr2__115978__11M-19+0+8.png)

### An episode ended by a jump of the semi-major axis {#transient-before-jump}

Two episodes, both before the jumps of $a$ (hatched; flag episode_before_jump): the long first one closes 87 cycles. From about 30 kyr $a$ jumps (kicks, as at close approaches to a planet) and the angle circulates to the end. Its amplitude, up to 138°, is near the separatrix, so the confidence is high, not certain.

**2003 UG22**, `10E-17+0+7`: `transient` · `episodic` · `ends: circulation` · `dynamic_episode` · `high`. Record: 40 kyr, step 0.25 yr, ias15.

[![mmr2__2003_UG22__10E-17+0+7__40kyr](https://smirik.github.io/resonances/img/gallery/mmr2__2003_UG22__10E-17+0+7__40kyr.png)](https://smirik.github.io/resonances/img/gallery/mmr2__2003_UG22__10E-17+0+7__40kyr.png)

### Episodes made by a member of the multiplet {#transient-subresonance}

The catalog angle librates in episodes about a center that moves with $2(\varpi_M - \varpi)$, while its member $\sigma + 2(\varpi_M - \varpi)$ (13M-24+2+9) keeps a fixed center: the episodes belong to that member (reason subresonance_librates), and this row takes the member's status and its 36 cycles. Its own figure draws the episodes faint (`cand`, not counted here); the second figure is the member's own verdict, with them counted.

**82831**, `13M-24+0+11`: `transient` · `episodic` · `ends: circulation` · `subresonance_librates` · `high`. Record: 100 kyr, step 0.5 yr, SABA(10,6,4).

[![mmr2__82831__13M-24+0+11](https://smirik.github.io/resonances/img/gallery/mmr2__82831__13M-24+0+11.png)](https://smirik.github.io/resonances/img/gallery/mmr2__82831__13M-24+0+11.png)

**82831**, `13M-24+2+9`: `transient` · `episodic` · `ends: circulation` · `dynamic_episode` · `high`. Record: 100 kyr, step 0.5 yr, SABA(10,6,4).

[![mmr2__82831__13M-24+0+11__13M-24+2+9](https://smirik.github.io/resonances/img/gallery/mmr2__82831__13M-24+0+11__13M-24+2+9.png)](https://smirik.github.io/resonances/img/gallery/mmr2__82831__13M-24+0+11__13M-24+2+9.png)

### An episode decided by the geometry alone {#geometric-episode}

A librating episode in part of the record. Neither the force law nor the potential could be measured, so the geometry decided (reason geometric_episode, origin unknown); the resonance's strength agrees with the cycles to an order of magnitude (margin F), so the level is not capped: high.

**24820**, `3J-1S-1+0+0-1`: `transient` · `episodic` · `ends: circulation` · `geometric_episode` · `high`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr3__24820__3J-1S-1+0+0-1](https://smirik.github.io/resonances/img/gallery/mmr3__24820__3J-1S-1+0+0-1.png)](https://smirik.github.io/resonances/img/gallery/mmr3__24820__3J-1S-1+0+0-1.png)

## Near the separatrix

No closed libration cycle, but the angle stalls at the top of its own well.

### Stalls at the top of the well {#near-separatrix}

No closed libration cycle, but $\sigma$ stalls near the top of its own potential well in each revolution while the energy fits the potential built from $a$ (pattern B): motion close to the separatrix. A measured objection ($\dot\sigma$ does not follow $a - a_*$ closely: the identity check fails) makes the confidence low, and the action asks for a review. This status is rare.

**523899**, `3N-5+0+2`: `near_separatrix` · `near_separatrix` · `separatrix_zone` · `low`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr2__523899__3N-5+0+2](https://smirik.github.io/resonances/img/gallery/mmr2__523899__3N-5+0+2.png)](https://smirik.github.io/resonances/img/gallery/mmr2__523899__3N-5+0+2.png)

## Non-resonant

Circulation, or a bounded angle that this resonance does not make.

### Circulation {#circulation}

$\sigma$ circulates over the whole record: no libration anywhere.

**42355**, `3N-4+0+1`: `non_resonant` · `circulation` · `circulation` · `certain`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr2__42355__3N-4+0+1](https://smirik.github.io/resonances/img/gallery/mmr2__42355__3N-4+0+1.png)](https://smirik.github.io/resonances/img/gallery/mmr2__42355__3N-4+0+1.png)

### A stroboscope: the output step aliases a fast angle {#alias}

The output step is close to a multiple of the fast period of $\sigma$ (it turns 365° per step), so the sampled angle seems to move slowly. The analysis finds a single alias branch and $a$ far from $a_*$, and reads the true motion as a fast circulation (reason alias).

**1999 UQ**, `7E+2M-9+0+0+0`: `non_resonant` · `circulation` · `alias` · `certain`. Record: 20 kyr, step 5 yr, ias15.

[![mmr3__1999_UQ__7E+2M-9+0+0+0](https://smirik.github.io/resonances/img/gallery/mmr3__1999_UQ__7E+2M-9+0+0+0.png)](https://smirik.github.io/resonances/img/gallery/mmr3__1999_UQ__7E+2M-9+0+0+0.png)

### The same body at a fine step {#alias-fine-step}

Sampled finely, the angle circulates fast, as the alias analysis of the coarse record predicted.

**1999 UQ**, `7E+2M-9+0+0+0`: `non_resonant` · `circulation` · `circulation` · `certain`. Record: 20 kyr, step 1 yr, ias15.

[![mmr3__1999_UQ__7E+2M-9+0+0+0__step1yr](https://smirik.github.io/resonances/img/gallery/mmr3__1999_UQ__7E+2M-9+0+0+0__step1yr.png)](https://smirik.github.io/resonances/img/gallery/mmr3__1999_UQ__7E+2M-9+0+0+0__step1yr.png)

### A kinematic libration {#kinematic-libration}

$\sigma$ librates over the whole record, but $a$ does not follow it: the swing is $\sigma$'s $-4\varpi$ term at the 18-kyr cycle of $2\omega$, a secular cycle of the elements, not the resonant force (origin kinematic). The veto is measured, so the confidence is high.

**285637**, `3M+7J-6+0+0-4`: `non_resonant` · `libration` · `kinematic_libration` · `high`. Record: 50 kyr, step 0.6 yr, ias15.

[![mmr3__285637__3M+7J-6+0+0-4](https://smirik.github.io/resonances/img/gallery/mmr3__285637__3M+7J-6+0+0-4.png)](https://smirik.github.io/resonances/img/gallery/mmr3__285637__3M+7J-6+0+0-4.png)

### A kinematic drift {#kinematic-drift}

$\sigma$ librates briefly about a center that drifts with its perihelion term ($4\varpi$, the 20-kyr cycle of $e$), while $a$ carries almost none of the swing: a kinematic drift, no resonant force at work.

**62243**, `1M-4J-1+0+0+4`: `non_resonant` · `drift` · `kinematic_drift` · `high`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr3__62243__1M-4J-1+0+0+4](https://smirik.github.io/resonances/img/gallery/mmr3__62243__1M-4J-1+0+0+4.png)](https://smirik.github.io/resonances/img/gallery/mmr3__62243__1M-4J-1+0+0+4.png)

### Episodes induced by another resonance {#induced-forced-wave}

$a$ oscillates because of another resonance, the driver 11M-20+0+9: it circulates with the period of that oscillation (about 2 kyr) and drives $a$ by its forced wave. $\sigma$ follows it only through the identity $\sigma = (7/20)\varphi + \Psi/20$ (origin induced, a forced wave).

**117306**, `1V+5J-7+0+0+1`: `non_resonant` · `episodic` · `ends: circulation` · `induced_libration` · `medium`. Record: 50 kyr, step 0.6 yr, ias15.

[![mmr3__117306__1V+5J-7+0+0+1](https://smirik.github.io/resonances/img/gallery/mmr3__117306__1V+5J-7+0+0+1.png)](https://smirik.github.io/resonances/img/gallery/mmr3__117306__1V+5J-7+0+0+1.png)

### The owner librates {#induced-owner-librates}

Here the owner, the two-body resonance 10E-17+0+7 with Earth, librates itself and drives $a$; the episodes of the three-body angle are its consequence (origin induced).

**1997 GK3**, `3M+5S-3+0+0-5`: `non_resonant` · `episodic` · `ends: libration` · `induced_libration` · `medium`. Record: 20 kyr, step 5 yr, ias15.

[![mmr3__1997_GK3__3M+5S-3+0+0-5](https://smirik.github.io/resonances/img/gallery/mmr3__1997_GK3__3M+5S-3+0+0-5.png)](https://smirik.github.io/resonances/img/gallery/mmr3__1997_GK3__3M+5S-3+0+0-5.png)

### An implausible strength {#implausible-strength}

A resonance of order 31 with Saturn. $\sigma$ circulates, with two short accepted episodes early in the record; they would need a resonant force at least $10^{35}$ times stronger than this resonance's harmonic provides, so they are not this resonance (a measured veto: high). The status does not count them: (a) leaves them unfilled and (l) labels them `forced (strength)`.

**2897**, `35S-4+0-31`: `non_resonant` · `circulation` · `implausible_strength` · `high`. Record: 50 kyr, step 0.6 yr, ias15.

[![mmr2__2897__35S-4+0-31](https://smirik.github.io/resonances/img/gallery/mmr2__2897__35S-4+0-31.png)](https://smirik.github.io/resonances/img/gallery/mmr2__2897__35S-4+0-31.png)

## Uncertain

The record cannot decide.

### The same body, too short a record {#insufficient-time}

The body of [the slow libration](#center-scale) on a shorter record. The slow center of the angle turns back only twice: the record holds half of one slow swing in this resonance's well (rule G5w), too short to show the libration whole. Uncertain, insufficient_time, and the action asks for a longer integration.

**385458**, `9N-17+0+8`: `uncertain` · `insufficient_time` · `baseline_needed`. Record: 100 kyr, step 1 yr, SABA(10,6,4).

[![mmr2__385458__9N-17+0+8](https://smirik.github.io/resonances/img/gallery/mmr2__385458__9N-17+0+8.png)](https://smirik.github.io/resonances/img/gallery/mmr2__385458__9N-17+0+8.png)

### An unexplained drift {#uncertain-drift}

$\sigma$ drifts, and neither a kinematic nor an induced explanation passes: uncertain, extend the integration.

**142694**, `1V+3S-7+0+0+3`: `uncertain` · `drift` · `drift_mechanism_unknown`. Record: 50 kyr, step 0.6 yr, ias15.

[![mmr3__142694__1V+3S-7+0+0+3](https://smirik.github.io/resonances/img/gallery/mmr3__142694__1V+3S-7+0+0+3.png)](https://smirik.github.io/resonances/img/gallery/mmr3__142694__1V+3S-7+0+0+3.png)

## Invalid

The record cannot be measured at these settings.

### An ambiguous alias {#invalid-alias}

The output step folds the angle so that its alias branches cannot be told apart: the angle cannot be measured at this step, and the action asks for a resample at the step given in the plaque (0.2 yr).

**1864**, `51M-50+0-1`: `invalid` · `circulation` · `alias`. Record: 20 kyr, step 5 yr, ias15.

[![mmr2__1864__51M-50+0-1](https://smirik.github.io/resonances/img/gallery/mmr2__1864__51M-50+0-1.png)](https://smirik.github.io/resonances/img/gallery/mmr2__1864__51M-50+0-1.png)

## Secular resonances

Angles of the proper perihelion and node against the planets' secular modes.

### The ν6 resonance {#secular-nu6}

The angle of the proper perihelion against the phase of Saturn's g6 mode librates over the whole record.

**1222**, `g-g6`: `resonant` · `libration` · `geometric_libration` · `certain`. Record: 10 Myr, step 500 yr (one point in 5 of a 100 yr output), SABA(10,6,4).

[![secular__1222__g-g6](https://smirik.github.io/resonances/img/gallery/secular__1222__g-g6.png)](https://smirik.github.io/resonances/img/gallery/secular__1222__g-g6.png)

### The z1 resonance {#secular-z1-resonant}

A slow libration of g − g6 + s − s6: only about one and a half cycles fit in the record, so the confidence is medium and the action asks for a longer record.

**1020**, `g-g6+s-s6`: `resonant` · `libration` · `geometric_libration` · `medium`. Record: 10 Myr, step 500 yr (one point in 5 of a 100 yr output), SABA(10,6,4).

[![secular__1020__g-g6+s-s6](https://smirik.github.io/resonances/img/gallery/secular__1020__g-g6+s-s6.png)](https://smirik.github.io/resonances/img/gallery/secular__1020__g-g6+s-s6.png)

### z1: one unfinished slow swing {#secular-z1-unfinished-swing}

The center of the z1 angle moves, then turns back after less than one revolution: the record holds part of one slow swing, not a drift, so it is too short to decide (insufficient_time, extend).

**2306**, `g-g6+s-s6`: `uncertain` · `insufficient_time` · `baseline_needed`. Record: 10 Myr, step 500 yr (one point in 5 of a 100 yr output), SABA(10,6,4).

[![secular__2306__g-g6+s-s6](https://smirik.github.io/resonances/img/gallery/secular__2306__g-g6+s-s6.png)](https://smirik.github.io/resonances/img/gallery/secular__2306__g-g6+s-s6.png)

### Just outside z1 {#secular-z1-circulation}

The z1 angle circulates slowly over the whole record: the body is close to the resonance but not in it.

**847**, `g-g6+s-s6`: `non_resonant` · `circulation` · `circulation` · `high`. Record: 10 Myr, step 500 yr (one point in 5 of a 100 yr output), SABA(10,6,4).

[![secular__847__g-g6+s-s6](https://smirik.github.io/resonances/img/gallery/secular__847__g-g6+s-s6.png)](https://smirik.github.io/resonances/img/gallery/secular__847__g-g6+s-s6.png)

### Too few cycles {#secular-insufficient-time}

The angle has only two turning points in 10 Myr, fewer than one cycle needs: the shape cannot tell a slow libration from a slow circulation. Too short to decide, extend the integration.

**363**, `g-g6+s-s6`: `uncertain` · `insufficient_time` · `baseline_needed`. Record: 10 Myr, step 500 yr (one point in 5 of a 100 yr output), SABA(10,6,4).

[![secular__363__g-g6+s-s6](https://smirik.github.io/resonances/img/gallery/secular__363__g-g6+s-s6.png)](https://smirik.github.io/resonances/img/gallery/secular__363__g-g6+s-s6.png)

## Lidov-Kozai

The argument of perihelion $\omega$, judged together with its free part (the free-omega gate).

### A Lidov-Kozai libration {#zlk-resonant}

$\omega$ librates and its free part does too, so the free-omega gate confirms a Lidov-Kozai libration.

**3040**, `LK`: `resonant` · `libration` · `dynamic_libration` · `certain`. Record: 500 kyr, step 100 yr, mercurius.

[![zlk__3040__LK](https://smirik.github.io/resonances/img/gallery/zlk__3040__LK.png)](https://smirik.github.io/resonances/img/gallery/zlk__3040__LK.png)

### A kinematic libration of the argument of perihelion {#zlk-kinematic}

The osculating $\omega$ librates, but the free $\omega$ (the forced terms removed) winds: the forced eccentricity holds the osculating $\omega$, and the libration is kinematic.

**591986**, `LK`: `non_resonant` · `libration` · `kinematic_libration` · `high`. Record: 500 kyr, step 100 yr, mercurius.

[![zlk__591986__LK](https://smirik.github.io/resonances/img/gallery/zlk__591986__LK.png)](https://smirik.github.io/resonances/img/gallery/zlk__591986__LK.png)

### Lidov-Kozai episodes {#zlk-transient}

Two long episodes of libration of $\omega$, the first about 270°, the second about 90°, with a short switch between them. The free-omega gate confirms the libration, and the deciding episode closes 13 cycles: certain.

**162474**, `LK`: `transient` · `episodic` · `dynamic_episode` · `certain`. Record: 1.5 Myr, step 10 yr (one point in 5 of a 2 yr output), SABA(10,6,4).

[![zlk__162474__LK__1500kyr](https://smirik.github.io/resonances/img/gallery/zlk__162474__LK__1500kyr.png)](https://smirik.github.io/resonances/img/gallery/zlk__162474__LK__1500kyr.png)

### No verdict from the gate {#zlk-gate-unavailable}

$\omega$ librates, but the free-omega gate gives no verdict on this record (the free node is not yet separated from $s_6$). The angle decides (resonant, flag gate_unavailable). The forced terms are small ($\rho_{\max} \ge 3$) and the $e$–$i$ integral holds in the libration, so the forcing cannot hold $\omega$ (flag kinematic_excluded): nothing caps the level, and 19 cycles give certain.

**1981**, `LK`: `resonant` · `libration` · `geometric_libration` · `certain`. Record: 500 kyr, step 10 yr, ias15.

[![zlk__1981__LK](https://smirik.github.io/resonances/img/gallery/zlk__1981__LK.png)](https://smirik.github.io/resonances/img/gallery/zlk__1981__LK.png)

### Circulation of the argument of perihelion {#zlk-circulation}

$\omega$ circulates over the whole record.

**15527**, `LK`: `non_resonant` · `circulation` · `circulation` · `certain`. Record: 500 kyr, step 10 yr, SABA(10,6,4).

[![zlk__15527__LK](https://smirik.github.io/resonances/img/gallery/zlk__15527__LK.png)](https://smirik.github.io/resonances/img/gallery/zlk__15527__LK.png)

### An unphysical orbit {#zlk-invalid}

The body is ejected: $e$ grows above 1 and $a$ becomes negative, so the record cannot be classified (reason unphysical).

**2010 HY22**, `LK`: `invalid` · `undefined` · `unphysical`. Record: 500 kyr, step 10 yr, ias15.

[![zlk__2010_HY22__LK](https://smirik.github.io/resonances/img/gallery/zlk__2010_HY22__LK.png)](https://smirik.github.io/resonances/img/gallery/zlk__2010_HY22__LK.png)

## Regenerating the figures

Every figure of this documentation (this gallery, the schematic drawings of the physics pages and the standalone
figures of [How to read the figures](../reading-figures.md)) is made by one command from the repository root:

```bash
make docs-figures                                  # everything (a few minutes, no integration)
make docs-figures ONLY=alias,pendulum_portrait     # some figures, by anchor, schematic or set name
uv run python docs/tools/build_figures.py --list   # the names
```

The script (`docs/tools/build_figures.py`) reads the saved series of the benchmark examples (`benchmark/data`, or
the folder in `RESONANCES_BENCHMARK_DATA`), analyses each again with the installed package and draws it with the
package's own figure code; nothing is integrated. It rewrites this page from the fresh verdicts and prints any
difference from the last saved benchmark run. Without the benchmark data the gallery and standalone figures are kept
and only the schematics are redrawn. Running it twice changes nothing: a file whose content is the same is not rewritten.
