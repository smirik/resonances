<!-- Generated from docs/physics/geometry.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# What the angle does

This page describes how the filtered angle $\sigma_f$ ([From the integration to the angle](signal.md)) is read as
geometry: where it turns back, whether its swings close into cycles about a centre, where such a libration starts
and ends, and how the whole record is summarised in `angle_status`. Nothing here asks whether $a$ responds to the
angle; that is the subject of [Is it this resonance?](response.md). All rules below are those of a mean-motion
resonance; secular and Lidov-Kozai angles use the same turning points, cycles and episodes
([Secular and Lidov-Kozai](secular-zlk.md)).

## Turning points

**Question: where does the angle turn back?** A turning point (TP) is a local maximum or minimum of $\sigma_f$ whose
prominence (the height of the extremum above the higher of its two bases, the lowest points between it and the
nearest more extreme value on either side, as in `scipy.signal.find_peaks`) is at least

$$
p_\text{TP} = \max\!\left(\texttt{MMR\_TP\_PROMINENCE\_FLOOR\_DEG},\ 4\,r\right),
$$

with the floor 1° and the factor `MMR_TP_RIPPLE_FACTOR` = 4. Here $r$ is the half-amplitude of what $\sigma_f$
still holds faster than the shortest libration $P_\text{floor}$ (the 95th percentile of that fast residual, or the
planetary ripple that could not be removed). A wave of half-amplitude $r$ makes extrema of prominence up to $2r$, so
the factor 4 keeps it from making turning points. The libration itself is subtracted before $r$ is measured, and
each run of turning points (below) gets its own $p_\text{TP}$ from its own samples, so a noisy stretch elsewhere in
the record does not erase the swings of a quiet libration. Turning points alternate: of two maxima in a row the
higher is kept.

## Cycles and their closure

Three alternating turning points (max–min–max or min–max–min) make **one full cycle**, one libration period:

$$
P_i = t_{i+2} - t_i .
$$

A half-cycle joins two neighbouring turning points; its half-amplitude is $A_i = \tfrac12\lvert\sigma_{i+1} - \sigma_i\rvert$
and its midpoint $m_i = \tfrac12(\sigma_{i+1} + \sigma_i)$. The full-cycle centre and half-amplitude are the means of
two neighbouring half-cycles, which cancels the alternation.

**Question: does the angle come back, or does it only pause on its way round?** A cycle is **closed** when its net
advance is smaller than its swing:

$$
Q_i = \frac{\lvert\sigma_{i+2} - \sigma_i\rvert}{\tfrac12\left(\lvert\sigma_{i+1} - \sigma_i\rvert + \lvert\sigma_{i+2} - \sigma_{i+1}\rvert\right)} < 1 .
$$

This is the same as each of the three turning points lying on its own side of the cycle centre, or the two
half-swings differing less than threefold; there is no setting. A circulation that stalls and turns back a little
closes no cycle. An episode needs at least one closed cycle, not all: the entry and exit half-cycles of a capture,
or a tadpole that becomes a horseshoe, often leave an edge cycle open.

**Closure by phase return.** A cycle can start between two turns, so two opposite turning points
($\sigma_1$ at $t_1$, $\sigma_2$ at $t_2$) are also one closed cycle when the angle crosses a level between them before
$t_1$ (at $t_a$) and after $t_2$ (at $t_b$) in the same direction and is back in the same state:

1. $\lvert a(t_b) - a(t_a)\rvert \le 0.05\times$ the swing of $a$ on $[t_a, t_b]$, and $\dot\sigma(t_b)/\dot\sigma(t_a)$
   within $1 \pm 0.05$ (`LIBRATION_RETURN_TOL` = 0.05);
2. $(t_b - t_a)/\big(2(t_2 - t_1)\big)$ within $1 \pm 0.25$ (`LIBRATION_RETURN_PERIOD_TOL`): a libration spends as
   long going from the level to a turn as coming back, while a circulation creeping back gives about 0.5;
3. $a$ carries the swing: $\operatorname{corr}\big(\dot\sigma,\ \operatorname{sign}(c)(a - \bar a)\big) \ge 0.9$
   (`LIBRATION_RETURN_IDENTITY_CORR_MIN`);
4. the swing is symmetric: between the turning points the angle crosses the level back as fast as at $t_a$,
   $\lvert\dot\sigma(t_m)/\dot\sigma(t_a) + 1\rvert \le 0.05$.

The mid-level $(\sigma_1 + \sigma_2)/2$ is a convention, so the levels at $k/20$ of the swing
(`LIBRATION_RETURN_LEVELS` = 19) are tried as well, and a level counts only inside a band of at least two
neighbouring levels that also close (`LIBRATION_RETURN_BAND_LEVELS`, `LIBRATION_RETURN_MID_BAND_LEVELS`): a real
return holds over a band of sections, a match at one isolated level is a coincidence. Levels whose crossing may lie
beyond the record neither close nor break a band, and the crossings are sought inside the run's own regime between
jumps of $a$. Such a run is an accepted stationary episode of one cycle of period $t_b - t_a$. A verdict resting on
it has its confidence capped at `medium` (cap `one_return_cycle`, `LIBRATION_RETURN_CYCLE_CAPS_MEDIUM`).

![Turning points, a closed and an open cycle, and a cycle closed by its phase return](https://smirik.github.io/resonances/img/physics/turning_points.png)

## Runs and episodes

Turning points are grouped into **runs**. A run breaks where a gap exceeds
`LIBRATION_EPISODE_GAP_FACTOR` = 3 times its median half-period, where a half-amplitude reaches $\pi$ (the angle
goes over the top), or where two consecutive midpoints differ by more than $\pi$ (a slip).

**Question: is this run a libration about a centre?** A run is an accepted **episode** when it has at least
`LIBRATION_MIN_TURNING_POINTS` = 3 turning points (one full cycle; or two closed by their phase return), at least
one closed cycle, and a centre of one of three kinds:

- **stationary**: the bands $[m_k - A_k,\ m_k + A_k]$ of all its full cycles share a point. With fewer than
  `LIBRATION_CENTRE_MIN_ESTIMATES` = 2 independent cycles the episode is accepted but its centre is not measured
  (cap `three_tp_episode`, at most `high`);
- **modulated**: the cycle centres turn back at least `LIBRATION_CENTRE_RETURNS_MODULATED` = 3 times within a range
  below `LIBRATION_MODULATED_RANGE_MAX_RAD` = $\pi$. Unequal members of a resonance multiplet, or a slowly modulated
  resonant term, make the libration's amplitude and centre oscillate (Morbidelli 2002, §9.4, pp. 224–226); the
  rule tests the observed returns, not a model of them;
- **moving centre**: the bands share a point in the frame $\psi = \sigma - \sigma_c(t)$, and every band holds
  $\psi = 0$. Here $\sigma_c(t)$ is the minimum of the averaged disturbing function of the angle's planet, recomputed
  along the record from the slowly varying elements: the stable centre of a resonance sits at that minimum
  (Gallardo 2006, §3, p. 30), and when the well changes much more slowly than one libration the angle librates
  about the moving minimum (the adiabatic condition: Wisdom 1985, pp. 274, 279). The track is accepted only when
  the observed cycle centres follow it better (in RMS) than the best fixed centre and than the same track run
  backwards in time. This is done for two-body angles (1:1 and retrograde included) with the planet saved; this
  implementation has no directly averaged potential for three-body angles, which stay in $\sigma$.

??? info "The centre track in detail"
    The track is built only for a run that fails in $\sigma$ and is not modulated. $\sigma_c(t)$ is the minimum of the
    full averaged potential (`averaged_potential`) at the elements averaged over $P_c$, at nodes the larger of $P_c$
    and the span / `MMR_CENTRE_TRACK_NODES` (400) apart, followed by descent from the previous node. A node without a
    well, without a converged quadrature or within `MMR_STRENGTH_MIN_DISTANCE_HILL` (3) Hill radii of the planet
    leaves the run in $\sigma$. $\psi$ is read at the turning points of $\sigma$, and the force law of such an episode
    is fitted in $\psi$.

An accepted episode counts as a libration of this angle only when its median period is not shorter than the
shortest libration $P_\text{floor}$ (a faster one is a forced wave the cutoff let through) and its centre does not
run a full revolution inside it. A run whose centre moves is tried again on its regular part (full periods within
`LIBRATION_REGULAR_PERIOD_RATIO` = 2 of the median, or without one slow edge half-cycle), and may be cut where its
regime changes: at a step of its centre between two measured librations, or at a stall of a slowly moving centre
(`LIBRATION_REGIME_CUTS`, `LIBRATION_STALL_MIN_CYCLES` = 3). Rejected runs keep their reason (`too_few_tp`,
`centre_moves`, `no_closed_cycle`, `over_the_top`, `edge`) in `episodes.csv`.

??? info "Regime cuts in detail"
    A run still rejected for its moving centre is cut where its regime changes, and the cut is kept only when the
    regimes it claims are accepted. (1) A *centre step*: a half-cycle with ≥ 3 turning points and a closed cycle on
    each side, each side passing the band test while the two together do not, each side with a measured centre, and
    the cycle centres differing across the step by more than within either side. Failing that, (2) a *stall of the
    slow centre*: the longest stretch of ≥ `LIBRATION_STALL_MIN_CYCLES` independent cycles whose midpoints move less
    per period than the cycles swing, at the run's own period, cut out as its own run. A steady drift has no stall
    and is not cut.

??? info "Forced waves (rule F6)"
    Once each accepted episode's own response is measured ([Is it this resonance?](response.md)), an episode is a
    forced wave of another argument, not an episode of the angle, when its identity fails with the wrong sign
    (slope/$c < 0$ at $\lvert\text{corr}\rvert \ge$ `MMR_IDENTITY_CORR_MIN`), or when its period sits on a forced line
    of the angle's lattice and its identity fails (the line alone suffices only with `MMR_FORCED_WAVE_LINE_ALONE`).
    Forced waves are kept apart (`episodes.csv` `forced_wave`), and the response, the energy, the exit and the entry
    are read again without them. Status, reason and confidence are those without them; only the reported word may
    keep their loops ([The reported angle word](status-tree.md#the-reported-angle-word)).

The example of a moving centre is 2013 AS27 in 31E−36 ([Gallery](gallery.md#moving-centre)): its 260 turning
points (period 150 yr, half-amplitude 24°) fail the band test in $\sigma$ by 8.8°, because the Earth's well moves by
about 57° in 20 kyr, and pass in $\psi$ with a margin of 35.5°; the cycle centres lie 1.6° (RMS) from the track,
against 14.4° from a fixed centre and 28.2° from the reversed track.

![Stationary, modulated and moving-centre episodes](https://smirik.github.io/resonances/img/physics/centre_types.png)

## The centre and circulation

The **slow centre** $c(t)$ is $\sigma_f$ smoothed over $W_c$ = `LIBRATION_CENTRE_CYCLES` = 3 libration periods (3 $P_c$
without cycles). Its own turns need a prominence $p_c = \max(10°,\ 2p_\text{TP})$
(`LIBRATION_CENTRE_TP_PROMINENCE_DEG`, `LIBRATION_CENTRE_TP_FACTOR`). A **monotone pass** is the stretch of $c(t)$
between two of its turns, measured by its maximum minus its minimum.

**Question: does the angle go round?** The angle circulates when the largest monotone pass is at least $2\pi$
(rule G4): an arc longer than one revolution is never a piece of a libration. A staircase with small step-backs
circulates as well when the centre advances by at least `LIBRATION_CIRCULATION_NET_REV` = 2 revolutions net over
the window (G4b). On an aliased step such a circulation has the subtype `alias`.

**Drift.** When rejected runs whose centre moves cover at least `LIBRATION_DRIFT_MIN_WINDOW_FRACTION` = 0.5 of the
window and no revolution is completed, the angle drifts (G5). If the slow centre turns back exactly once inside the
record, farther than $W_c$ from the window's edges, and its range stays below one revolution, the record holds only
part of one slow swing and cannot tell a drift from a slow libration: `insufficient_time` / `arc` (G5a).

**Too few turns.** With turning points but no episode the angle is `insufficient_time` / `few_cycles`; with none it
is `arc` (the range of $\sigma_f$ is at least $\pi$) or `stall` (below $\pi$), rule G6. The record then sets only a
lower bound on the libration period, which gives the length to integrate ([Confidence](confidence.md#actions)).

??? info "The lower bound of an arc or a stall"
    A libration about a centre 0 or $\pi$ moves $a$ at $\lvert\dot a\rvert = K\lvert\sin\sigma\rvert$. Over the
    analysis window of length $\tau$, with the range $\delta a = \max a_f - \min a_f$ and
    $s_{\min} = \min\lvert\sin\sigma_f\rvert$ there,
    $K_{\max} = \delta a/(\tau\,s_{\min})$ and

    $$
    P_\text{lib,min} = \frac{2\pi}{\sqrt{\lvert c\rvert K_{\max}}} = 2\pi\sqrt{\frac{\tau\,s_{\min}}{\lvert c\rvert\,\delta a}} .
    $$

    The least $\lvert\sin\rvert$ gives the most cautious bound; $s_{\min} = 0$ gives no positive bound.

## Coverage, exits and entries

**Question: does the libration last the whole record?** An episode covers a record edge when it starts (ends)
within `LIBRATION_COVERAGE_PERIODS` = 1 of its own periods of that edge, or when no turning point is missing between
the edge of the analysis window and the episode and the stretch the filter trims off that edge is itself within one
period. An episode covering both edges makes the angle a `libration`; otherwise it is `episodic` (`recurrent` with two
or more librating episodes).

**Exit and entry.** After the last turning point the angle may leave its centre:

- **suspected**: $\sigma_f$ leaves the episode's centre by more than $\pi$ with less than one period left. The label
  stays, the flag `exit_suspected` caps the confidence at `medium`;
- **observed**: the raw angle averaged over one orbital period runs at least `LIBRATION_EDGE_ARC_REV` = 1 turn away
  from the last turning point to the end of the record and moves monotonically there. The edge is then not covered,
  there is no `exit_suspected`, and `final_regime` is a circulation from the exit. The same holds for an entry at the
  start of the record. The centre-scale pass (G2c) does not look for a slower libration behind such an arc.

2017 UM52 in 1E−1 ([Gallery](gallery.md#transient-exit)) is a horseshoe that exits at 949 yr: its averaged angle then
runs 1.97 turns monotonically, so the exit is observed and the verdict is `transient`.

**Union of episodes.** Consecutive librating episodes join into one libration over the record when nothing between
them goes round: in each gap the range of the turning points and the range of the orbit-averaged raw angle are both
below $2\pi$, the gap is at most `LIBRATION_UNION_GAP_PERIODS` = 1 times $(P_\text{left} + P_\text{right})$, nothing
goes round over the whole chain, and the outer episodes reach the record's edges. The angle is then `libration` /
`regime_change`. For a 1:1 angle this is rule G2: a quasi-satellite becoming a horseshoe without circulating is one
libration ([Gallery](gallery.md#coorbital-regime-change)). For any other angle it is rule G2u: the reported word is
`libration` / `regime_change`, but the status tree reads the chain as episodic and promotes it to `resonant` only from
a dynamic transient ([The status tree](status-tree.md)).

![A union of episodes against an observed exit](https://smirik.github.io/resonances/img/physics/episodes.png)

## Jumps of $a$ and the last regime

Close encounters and passages through a chaotic layer change the orbit abruptly. **Question: has the orbit jumped to
another regime?** A candidate is a change of $a_f$ over $\tau = \max(2P_c,\ 5\Delta t)$
(`LIBRATION_JUMP_TAU_CUTOFFS`, `LIBRATION_JUMP_TAU_STEPS`) above `LIBRATION_JUMP_NOISE_FACTOR` = 8 robust standard
deviations of such changes, and accumulated in less than one libration period $P_\text{cyc}$ (a kick, not an
adiabatic drift). It is a confirmed **jump** when the mean dimensionless energy $h$ of
`LIBRATION_JUMP_CYCLES` = 2 cycles on each side differs by more than $\max(2\sigma_h,\ 0.1)$
(`LIBRATION_JUMP_DH_MIN`); with fewer cycles, when the step of $a$ exceeds the resonance half-width (flag
`jump_by_a_only`). A step without an energy change is a kick and cuts nothing; so does a candidate within
$\max(\tau, P_\text{cyc})$ of an edge (flag `jump_at_edge`).

After a confirmed jump the geometry is measured again on the **last regime** (flag `last_regime_after_jump`; each
regime has its own centre and turning-point prominence, and no cycle crosses a jump). The last regime decides the
angle word, with two exceptions that keep the earlier episodes visible:

- the last regime holds no episode but librating episodes precede the jump: the angle is reported `episodic`, the
  status reads those episodes (cap `episode_before_jump`, at most `medium`), and `final_regime` is the last regime's
  word (G1e). Example: 2003 UG22 in 10E−17 over 40 kyr ([Gallery](gallery.md#transient-before-jump)), two episodes,
  then circulation between jumps of $a$ after 31 kyr;
- the last regime librates but nothing librated before the jump (a capture): reported `episodic`,
  `final_regime = libration` (G1l).

### A step of the libration action

A step of the libration amplitude can happen without a visible step of the mean $a$, for instance when an encounter
changes the swing of $a$ but not its mean. Away from the separatrix the action of a libration is an adiabatic
invariant; an amplitude that changes by a large factor within one half-cycle is a candidate impulse onto another
energy level of the same potential. On every accepted run in the $\sigma$ frame (not 1:1) that no jump already cuts,
the half-amplitudes are split at the half-cycle that separates them best, and the record is cut there, as at a
jump, when all of these hold:

1. each side has at least `LIBRATION_ACTION_STEP_MIN_HALF_CYCLES` = 2 half-cycles;
2. the sides are separated: min(larger side) / max(smaller side) $\ge$ `LIBRATION_ACTION_STEP_SEPARATION` = 2;
3. their medians differ by a factor $\ge$ `LIBRATION_ACTION_STEP_RATIO` = 3;
4. the swing of $a$ (its standard deviation per side) changes the same way by a factor $\ge$
   `LIBRATION_ACTION_STEP_A_RATIO` = 1.5, since $\delta a \propto \dot\sigma$: a step of $\sigma$ alone is no step of
   the libration;
5. the transitional half-cycle lasts at most `LIBRATION_ACTION_STEP_MAX_PERIODS` = 1 period of the shorter side;
6. each side has at least `LIBRATION_ACTION_STEP_CYCLES` = 1 cycle of measured energy, and $h$ changes the way the
   amplitude does by more than $\max(2\sigma_h,\ 0.1)$: beats at a constant energy do not;
7. both regimes librate in one potential, each at its own energy (column `geo_action_step_failed = regime_periods`
   when this fails): the median periods of the two sides differ by more than
   `LIBRATION_ACTION_STEP_PERIOD_TOL` = 0.2 (in a nearly harmonic well every loop has the same period, so a forced
   wave whose envelope steps would pass otherwise); and at every smoothing $C = f\times$ the shorter period,
   $f$ in `LIBRATION_ACTION_STEP_CUTOFFS` = 0.075, 0.1, 0.15, the potential fitted to the cycles of the larger swings
   explains them, covers the smaller swings, and holds every disjoint cycle of both sides in a loop of its own
   energy, with the observed period within 20 % of the period predicted in the well that holds the cycle and the
   turning points within `LIBRATION_ACTION_STEP_TP_TOL` = 0.2 of the loop's; the energies of each side stay put
   (`LIBRATION_ACTION_STEP_ENERGY_DRIFT` = 0.5), and the number of wells is the same at every smoothing.

With fewer than two energy cycles on a side, or a single cycle of the larger swings, the cut is provisional (cap
`action_step_provisional`, at most `medium`). The record is also analysed whole: if the cut leaves a measurement that
failed on the whole record (identity, force law, R0, zone, strength) unmeasured and changes the status or the
confidence, the cut is withheld and the whole record's analysis is kept, because geometry does not override a
failed measurement (`geo_action_step_withheld_checks`). A cut that stands is a confirmed jump (`jump_action_step` = 1)
at the middle of the transitional half-cycle (rule G2a); columns `geo_action_step_*`, and panel (a) of the figure
marks it.

2003 TX9 in 3E+2J−10 shows a candidate at 6.96 kyr: the half-amplitude falls from 146° to 12° (separation 8.3, the
swing of $a$ ÷ 7.1, $\Delta h = -0.54$), so tests 1–6 pass; test 7 fails (the period law of the smaller swings in the
potential of the larger ones is 1.61), and the record is not cut ([Gallery](gallery.md#uncertain-origin)).

## The separatrix zone

The separatrix zone is the band swept by the separatrix of a **measured, time-dependent pendulum**, not a fixed
tolerance about $h = 1$. The force of an MMR is that of its multiplet (members
$\sigma_j = \sigma + j(\varpi - \varpi_P)$, [Is it this resonance?](response.md#subresonances)), which is one harmonic
of $\sigma$ with a slowly turning complex amplitude (Morbidelli 2002, §9.4, pp. 224–225, eqs. 9.20–9.22):

$$
W(t) = \sum_j (A_j + iB_j)\,e^{ij(\varpi - \varpi_P)}, \qquad
\ddot\sigma = -\Omega(t)^2\sin\big(\sigma - \sigma_s(t)\big), \qquad
\Omega^2 = \lvert c\,W\rvert, \quad \sigma_s = -\arg W\ (+\pi \text{ for } c > 0).
$$

Without resolved members the strength follows the bounded oscillatory part of the eccentricity,
$\Omega^2 \propto (e_\text{osc}/\bar e)^{\lvert p\rvert}$ ($p$ the coefficient of $\varpi$), with a fixed centre (flag `zone_eq_only`). The scale $\bar\Omega$ comes from
the potential fitted to $\sigma$, and $m(t) = \Omega(t)^2/\bar\Omega^2$. Trial librations and rotations of this model are
integrated from a grid of starts to the end of the record: initial actions $J_0/\langle J_s\rangle$ from
`MMR_ZONE_J_MIN` = 0.1 to 1 and rotation energies $h_0$ from 1 to `MMR_ZONE_H_MAX` = 2, both in steps of
`MMR_ZONE_J_STEP` = 0.025, each at `MMR_ZONE_PHASES` = 16 start times over one modulation period (tolerance
`MMR_ZONE_RTOL`). Then

- $J_\text{lo}$ is the **smallest** starting action that ever goes over the top,
- $h_\text{hi}$ is the **largest** starting rotation energy that is ever trapped (its relative rate changes sign).

With `MMR_ZONE_METHOD` = `adiabatic` the zone is instead $J_\text{lo} = \Omega_\text{min}/\bar\Omega$,
$h_\text{hi} = (\Omega_\text{max}/\bar\Omega)^2$; for a slow modulation (adiabaticity
$\lambda = \nu_\text{mod}/\bar\Omega <$ `MMR_ZONE_ADIABATIC_LAMBDA_MAX` = 0.1) a numeric $J_\text{lo}$ that differs from
this by more than `MMR_ZONE_ADIABATIC_TOL` = 0.05 sets the flag `zone_numeric_mismatch`. There is no zone without a
finite $\bar\Omega$, or when $2\pi/\bar\Omega$ is shorter than the smoothing of the angle.

**Energy scatter.** The uncertainty of a cycle's energy is the robust scatter of the energies $h_j$ of neighbouring
cycles,

$$
\sigma_h = \max\!\Big(\sigma_\text{floor},\ \frac{1.4826\,\mathrm{MAD}(h_{j+1} - h_j)}{\sqrt2}\Big),
$$

with $\sigma_\text{floor}$ = `SEPARATRIX_SIGMA_FLOOR` = 0.02; with fewer than two finite energies $\sigma_h$ is not
measured and the floor is used.

**Classes.** With $z$ = `SEPARATRIX_SIGMA_MARGIN` = 2 and the cycle's mean strength $m$:

- a libration is **deep** if $J/\langle J_s\rangle + z\,\sigma_J < J_\text{lo}$, where
  $J/\langle J_s\rangle = (J/J_s)\sqrt m$ and $\sigma_J = \dfrac{d(J/J_s)}{dh}\sqrt m\,\sigma_h$, with
  $d(J/J_s)/dh = (V_\text{max} - V_\text{min})\,P/(2\pi J_s)$ ($\mathsf K(h)/2$ for one harmonic);
- a rotation is **above** if $h\,m - z\,\sigma_h > h_\text{hi}$;
- a libration above the lower saddle of its well in a potential of several wells is **not measured**
  (`not_measured`): its loop encloses both wells and has no single-well $J/J_s$ (W25,
  [The resonant pendulum](pendulum.md#the-pendulum)); it is neither deep nor in the zone;
- every other cycle is in the **zone**; a cycle within one $\sigma$ of an edge sets `zone_edge_cycle`.

$J/J_s$ is the action of the cycle's own loop over the separatrix action of the well it occupies (the basin of its
fastest sample; [The resonant pendulum](pendulum.md#the-pendulum)). For
a two-body episode whose planet was saved, the directly averaged well replaces the fitted class. Each turning point's
action is computed in that profile itself, at its own time: $J/J_s$ of the loop of energy $\bar D(\sigma_\text{TP})$
about the bottom of the well that holds the swing's middle, over the action of that well's separatrix loop (at its
barrier, the lower of its two tops), both by quadrature of $\sqrt{2(E - V)}$ on the profile. $J_\text{lo}$ is in units
of the record's mean separatrix action, so the ratio is carried there by the averaged well itself:

$$
\frac{J}{\langle J_s\rangle} = \frac{J}{J_s(t)}\,\frac{J_s(t)}{\langle J_s\rangle},
$$

with $J_s(t)$ in absolute units ($V = 3k^2\mu_P n^2 a\,\bar D$, so the profile's action times the square root of that
scale) and $\langle J_s\rangle$ its time mean over the record, from `MMR_SEPARATRIX_AVGPOT_TP_MAX` even times (the well
that holds the swing middle of the nearest turning point). The action at a fixed $h$ grows as the square root of the
well's strength: where the well is at a quarter of its mean strength, $J/J_s(t) = 0.6$ is
$J/\langle J_s\rangle = 0.3$. An episode cycle is in the zone if any measured turning point has
$J/\langle J_s\rangle \ge J_\text{lo}$, deep if all are measured and below it, and otherwise keeps its fitted class;
a turning point above the barrier of a profile of several wells is not measured, and none is when a profile on the
record's grid fails (the mean is then unknown). This uses no $\sigma_h$. Actions are compared, not energies:
$J/J_s = 0.5$ lies at $h = 0.581$ for $V = -\cos\sigma$ and at $h = 0.628$ for $V = -\cos\sigma - 0.2\cos2\sigma$,
so an energy mapped from $J_\text{lo}$ through the fitted potential misreads a profile of another shape (before W25).
$h_\text{TP}$ is still reported (`sep_h_avgpot_*`, beside `sep_J_avgpot_*`). The zone and the classes are diagnostics
of the fitted model, not a measurement of N-body chaos. The same $\sigma_h$ enters the jump and action-step tests above.

The zone gives the flags `libration_in_separatrix_zone`, `episodes_in_separatrix_zone` (crossings, and every cycle in
the zone: a cycle not measured breaks it), `near_separatrix_energy` and `zone_edge_cycle`, the margins L2 (thin with a
cycle in the zone, not measured with a cycle not measured and none in the zone) and Z, and the cap
`episodes_in_separatrix_zone`. The classes are not columns only: they move the confidence through these, and one status
path: when the episodes lie in the zone, O1 is not read as an objection ([The status tree](status-tree.md)), so the
unmeasured-force branch does not end at uncertain on it.

**Crossings.** They are read from $\sigma$ (`pendulum.regime_transitions`): an exit is a libration half-cycle
followed by a slip or a turn, a capture the reverse. The crossings get one mechanism, `sep_mechanism`, the first
that applies:

| `sep_mechanism` | condition |
|---|---|
| `kick` | an energy-changing step of $a$ within one cycle period of a crossing |
| `trend` | the slope of $J/\langle J_s\rangle$ over the cycles differs from 0 by more than `SEPARATRIX_TREND_SIGMA` = 3 standard errors, and its total change exceeds $1 - J_\text{lo}$ |
| `pulsation` | the crossings favour the weak phase of $m(t)$: cyclic-shift p-value ≤ `SEPARATRIX_PERMUTATION_P_MAX` = 0.05 |
| `layer` | $\lambda = \nu_\text{mod}/\bar\Omega >$ `SEPARATRIX_LAYER_ADIABATICITY_MIN` = 0.5: a fast modulation, a chaotic layer |
| `unexplained` | none of these |
| (empty) | no crossing |

The mechanism names a reading of the crossing; it sets no status. An episodic angle whose mechanism is `unexplained`
and whose episodes are not in the zone gets the cap `transient_unexplained` (not for a 1:1 angle,
`MMR_COORBITAL_LIFTS_UNEXPLAINED`). The mark `exit_and_capture` (an episodic, non-co-orbital angle with at least one
exit and one capture, no kick, its own force measured, no filter or sampling caveat) lifts that cap
(`MMR_EXIT_AND_CAPTURE_LIFTS_UNEXPLAINED`).

## Near the separatrix

Without any episode, an angle can still show that it lives at the top of its own potential well. These patterns
need the potential fitted to $\sigma$ alone to hold (the pendulum test R0 on $\sigma$,
[Is it this resonance?](response.md)):

- **pattern A, slips**: the oscillation midpoint steps by a whole, non-zero number $r$ of turns,
  $\lvert\Delta m/2\pi - r\rvert \le$ `LIBRATION_SLIP_TOLERANCE` = 0.2 (a slip can span several turns). A slip shorter
  than `LIBRATION_SLIP_MIN_DURATION_T0` = 0.5 small-amplitude periods $T_0$ (when $T_0$ is measured) is rejected: an
  unwrap error smeared by the filter is that fast, and a pendulum passage over the top with $h \lesssim 1.6$ is not.
  This is a convention aimed at near-separatrix passages, not a bound on a fast circulation. At least
  `LIBRATION_PATTERN_A_MIN_SLIPS` = 2 slips of pieces with a median half-amplitude $\ge$
  `LIBRATION_PATTERN_A_MIN_AMP_DEG` = 60° are needed;
- **pattern B, stalls**: every revolution is within reach of the separatrix,
  $h_\text{rev} \le h_\text{hi} + z\,\sigma_h$, and in at least
  `LIBRATION_STALL_REV_FRACTION` = 0.75 of them the rate is slowest within `LIBRATION_STALL_PHASE_TOL_DEG` = 30° of the
  top of the potential.

The angle is then `near_separatrix` (`pattern_a` or `pattern_b`; rule G3). Example:
523899 in 3N−5 ([Gallery](gallery.md#near-separatrix)).

## The centre scale

A slow libration can carry faster waves whose turning points fill the record: at $P_c$ no episode covers it, but the
centre $c(t)$ itself librates. **Question: is there a libration at the scale of the centre?** When the fine scale
shows no libration (and no observed exit or entry), the angle is filtered at $W_c = 3P_\text{cyc}$, or, failing that,
at half the period of the centre's own turns. If that angle holds an episode covering the record, the whole analysis
is repeated at that cutoff, with its turning points taken at the centre-turn prominence $p_c$
(`LIBRATION_CENTRE_SCALE_TP_AT_CENTRE`), and its verdict is kept when it is a whole-record libration, or an episode
that would cover both edges of the record but for an exit or entry observed at that scale (rule G2c; column
`geo_centre_scale_cutoff_yr`). If the libration found there is too fast for that cutoff, it is analysed once more at
half its period. 385458 in 9N−17 over 300 kyr is a libration at $W_c$ = 4.6 kyr
([Gallery](gallery.md#centre-scale)); over 100 kyr the same body is `insufficient_time`
([Gallery](gallery.md#insufficient-time)).

## From geometry to `angle_status`

The first rule that applies gives the angle word:

| rule | condition | `angle_status` (`angle_subtype`) |
|---|---|---|
| G0 | unphysical orbit; the unwrap residual is over its limits; $\varpi$ undefined ([signal](signal.md)); $P_c \le 2\Delta t$ | `undefined` (`unphysical`, `alias`, `undefined_phase`); `invalid` / `undersampled` |
| G1 | a confirmed jump of $a$, or a step of the action (G2a) | the rules below on the last regime |
| G1c | after the last turning point the centre completes a revolution, for longer than one period of the last episode | `episodic` (`recurrent` with ≥ 2 episodes), `final_regime = circulation` from the last turning point |
| G2 | a librating episode covers the record | `libration` (`stationary`, `modulated`, `moving_centre`) |
| G2, G2u | librating episodes without circulation between them cover the record | `libration` (`regime_change`) |
| G2 | librating episodes that do not cover it | `episodic` (`recurrent` with ≥ 2) |
| G1l | a libration of the last regime after a jump, no librating episode from the record's start to the jump | `episodic`, `final_regime = libration` from the start of the last episode (a capture); a libration on both sides of the jump stays `libration` |
| G1q | a prograde 1:1 angle without a libration whose orbit-averaged angle stays in the quasi-satellite well long enough | analysed again at the quasi-satellite floor ([pendulum](pendulum.md#co-orbitals-11)) |
| G2c | no libration at $P_c$ (and no observed exit or entry), but the centre librates | analysed again at the centre scale (below); the rejected analysis is in `geo_cs_rej_*` |
| G3 | no episode, the $\sigma$-only potential holds, slips or stalls at the top of the well | `near_separatrix` (`pattern_a`, `pattern_b`) |
| G4, G4b | the centre makes a revolution, or ≥ 2 net | `circulation` (`alias` on an aliased step) |
| G5 | runs with a moving centre cover half the window | `drift` |
| G5a | the slow centre turns back once, away from the edges, range < $2\pi$ | `insufficient_time` (`arc`) |
| G6 | turning points but no episode / none: range ≥ $\pi$ / < $\pi$ | `insufficient_time` (`few_cycles` / `arc` / `stall`) |
| G1e | the last regime holds no episode, librating episodes precede the jump | `episodic`, `final_regime` = the last regime's word; the status reads the earlier episodes with their own response and no energy (cap `episode_before_jump`) |

**`final_regime`** says how the record ends: a circulation from the last turning point (G1c) or from an observed
exit; the last regime's word after a jump (G1e); `libration` from the start of the last librating episode when that
episode reaches the record's end, which includes every whole-record libration and a capture (G1l). It is empty when
none of these holds.

**The reported word.** The word in the output is not always the raw geometric one: G1c, G1e and G1l are reported as
`episodic`, a G2u union as `libration` / `regime_change`, a kinematic drift as the centre's word, and forced waves
may keep their loops ([The reported angle word](status-tree.md#the-reported-angle-word)).

So `angle_status` describes what $\sigma$ does, and `status` says whether that is this resonance. A body can be
`non_resonant` with `angle_status = libration` ([Gallery](gallery.md#kinematic-libration)), or `uncertain` with
`angle_status = libration` ([Gallery](gallery.md#uncertain-origin)).
