# Is it this resonance?

A bounded resonant angle is not yet a resonance. In a real resonance the resonant term moves the semi-major axis,
and $\sigma$ librates *because* $a$ swings about $a_*$ ([The resonant pendulum](pendulum.md)). An angle can also look
bounded when the perihelion precession carries it, when another resonance drives $a$ and $\sigma$ follows through the
identity, or when a forced wave of a nearby term happens to have the right period. The tests on this page ask, one
by one, whether $a$ does what the pendulum of *this* resonance requires, and if not, what else makes the cycles.

The outcome is `libration_origin`:

| origin | meaning | example |
|---|---|---|
| `dynamic` | a measured force of this resonance (force law, energy integral or strength) explains the libration | [two-body](gallery.md#resonant-two-body), [co-orbital](gallery.md#resonant-coorbital) |
| `kinematic` | $\sigma$ is bounded by geometry (perihelion term, a secular cycle, a planetary ripple); $a$ does not respond | [kinematic libration](gallery.md#kinematic-libration), [kinematic drift](gallery.md#kinematic-drift) |
| `induced` | another resonance (the *owner*) drives $a$; $\sigma$ follows by the identity | [forced wave](gallery.md#induced-forced-wave), [owner librates](gallery.md#induced-owner-librates) |
| `subresonance` | another member of the same multiplet librates about a fixed center | [82831](gallery.md#transient-subresonance) |
| `unknown` | bounded, mechanism not established | [2003 TX9](gallery.md#measured-objection), [geometric libration](gallery.md#geometric-libration) |

How the origins combine into a status is on [The status tree](status-tree.md). Every check has four outcomes:
`pass` supports the resonant reading, `fail` refutes it, `not_measured` means the record could not say, and
`not_applicable` means the test does not exist for this angle. Each outcome is a column of `summary.csv`
([Output data](../output.md)).

## Where the response is measured

The derivative tests read $\sigma$ and $a$ smoothed once more at
$\max(P_c,\ 0.1\,P_\text{cyc})$ (`MMR_RESPONSE_CUTOFF_CYCLES` = 0.1, $P_\text{cyc}$ the median cycle period): the
record's cutoff $P_c$ is chosen for $\sigma$, and a time derivative amplifies whatever leaks through it
([From the integration to the angle](signal.md)). The tests are made on the union of the librating episodes (else
the whole record) and again **on every librating episode** with its own period. An episode is a *witness* when its
force law passes with full margin while the amplitude test does not fail and the identity is not measured to fail
(or the force law passes thinly with both of those passing). With a witness, the status tree reads the identity, the
force law and the amplitude from the witness with most turning points: one episode decides, nothing is counted over
windows (rule R; columns `rsp_*_source`). Without a witness it reads the pooled measurement (after a jump with no
episode in the last regime, G1e: pooled over the episodes before the jump). A witness next to an episode whose force
law fails gives the cap `episode_force_law_mixed` (at most `medium`). Windows that overlap a confirmed jump of $a$
([What the angle does](geometry.md)) are left out.

## The identity

*Question.* Does the rate of the angle follow the semi-major axis as $\dot\sigma = c\,(a - a_*)$?

*Measurement.* Regression of the filtered $\dot\sigma$ on the filtered $a$:

$$
\dot\sigma_f = s\,a_f + \text{const}, \qquad \text{pass if }\ \left\lvert \frac{s}{c} - 1\right\rvert \le 0.2
\ \text{ and }\ \operatorname{corr}(a_f, \dot\sigma_f)\,\operatorname{sign} c \ge 0.9 ,
$$

with `MMR_IDENTITY_SLOPE_TOL` = 0.2 and `MMR_IDENTITY_CORR_MIN` = 0.9.

*Meaning.* The identity holds for almost any angle that contains $k\lambda$, so a pass says little. A **fail** is
informative: the oscillation of $\sigma$ sits in the perihelion part, the output step aliases the angle, or $a$ and
$\sigma$ follow another clock. An episode whose identity fails with the wrong sign ($s/c<0$ at
$\lvert\text{corr}\rvert \ge 0.9$) cannot be a pendulum of this angle at all and is treated as a forced wave (below).
*In the output:* `check_mmr_identity`.

## The force law

*Question.* Is $\dot a$ the force of a pendulum in $\sigma$?

*Measurement.* From Lagrange's equation, the cosine pendulum has $\dot a \propto \sin(\sigma - \sigma_c)$. In sliding
windows of `MMR_FORCE_WINDOW_CYCLES` = 3.5 cycles (long enough to define the phase, short enough for the force to stay
constant), the package fits by least squares

$$
\dot a_f = R\,\sin(\psi - \phi_0) + C ,
$$

with $\psi$ the angle (or $\sigma - \sigma_c(t)$ in a moving well) and $C$ a free offset (a slow drift of $a_*$, not
part of the potential). Its $R^2$ is the smaller of the median $R^2$ in the two halves of the record:

- pass with $R^2 \ge$ `MMR_FORCE_R2_FULL` = 0.7 (a *thin* pass for $0.4 \le R^2 < 0.7$);
- fail with $R^2 <$ `MMR_FORCE_R2_FAIL` = 0.4 under every law tried;
- not measured below an amplitude of `MMR_A_RESPONSE_MIN_AMP_DEG` = 45° (there $\sin\psi$ and $\psi$ differ by
  less than 10 %, and any oscillation carried by $a$ fits), or with fewer than `MMR_FORCE_MIN_WINDOWS` = 2 windows.

The fitted phase tells where the center is, but its sign depends on $c$: $\sigma$ is stable at $\phi_0$ when $c<0$ and
at $\phi_0 + \pi$ when $c>0$ (from $\ddot\sigma = c\,\dot a$). The phase spread over the windows is recorded but not
tested: a modulated pendulum's phase legitimately moves (Morbidelli 2002, §9.3).

**Multiplet law.** A two-body commensurability has several members $\sigma_j = \sigma + j(\varpi - \varpi_P)$ with
the same fast part (Morbidelli 2002, §9.3–9.4). Their sum

$$
\dot a_f = \sum_j\big[A_j\sin\sigma_j + B_j\cos\sigma_j\big] + C
$$

is taken when its $R^2$ beats the single law by `MMR_FORCE_MULTI_GAIN` = 0.1, and only if its leading coefficients
stay coherent over the windows (`MMR_FORCE_COEF_COH_MIN` = 0.8; otherwise the law fails).

**Several harmonics of the angle.** A well that is not a cosine (a double well, a flat bottom) has the force law of
its own potential, $\ddot\sigma = c\,\dot a = -V'(\sigma)$ with a few harmonics:

$$
\dot a_f = \sum_{k=1}^{N}\big[A_k\sin k\psi + B_k\cos k\psi\big] + C .
$$

When the one-harmonic law fails or passes thinly (never for a 1:1 angle, never inside a null test), $N$ grows from 2
(the one-harmonic law is $N = 1$) while $R^2$ gains at least `MMR_FORCE_MULTI_GAIN` = 0.1, up to
`MMR_FORCE_HARMONICS_MAX` = 2. The law is taken only if it passes, in order: a shifted null test exists; its
coefficients are coherent (0.8); its $R^2$ beats the null by `MMR_FORCE_NULL_GAIN` = 0.2; $R^2 \ge 0.4$; and the law
fitted on the earlier half (in time) of the windows' samples predicts the later half with $R^2 \ge 0.4$, and the
reverse. It can only lift the check, never lower it. A purely kinematic linear swing wider than about $\pm150°$ can also fit two harmonics, so the force
law is necessary, not sufficient: the kinematic tests, the energy and the strength decide the origin.

**Guards.** Three effects can fake or spoil the force law, and each has a guard:

- *A line of $\dot a$ near the cutoff.* A short-period term just above the response cutoff, with an amplitude of at
  least `MMR_RESPONSE_LINE_AMP_FRAC` = 0.5 of the fitted force, raises the cutoff of that measurement to
  `MMR_RESPONSE_LINE_MARGIN` = 1.5 × its period (at most $P/4$). The law is then refitted and must beat a **null
  test**: the same fit with $\dot a$ shifted in time by $(m+\tfrac14)P$ and $(m+\tfrac34)P$
  (`MMR_FORCE_NULL_SHIFTS` = 8 lags), by at least 0.2 in $R^2$.
- *A forced line at the libration period.* Inside a libration $n$ sits at the commensurate value $n_c$, and every
  argument $m\lambda + \sum_j l_j\lambda_j$ of the angle's own planets turns at $m n_c + \sum_j l_j n_j$. Such a
  term moves $a$ at its own frequency, and through the identity a wave $a - \bar a = A\cos\nu t$ gives
  $\dot a = -(\nu^2/c)(\sigma - \sigma_c)$: a pendulum-looking law with no resonant force. A window of 3.5 periods
  resolves frequencies only to $1/(3.5P)$, so when such a line lies within that resolution of the measured period the
  force law is `not_measured` (`MMR_FORCE_CONTAMINATION` = lattice). If the episode's identity also fails, the
  episode is a **forced wave**, not a libration of this angle: it is removed from the episodes.
- *Folded short periods in $a$.* When the output step folds orbital or synodic periods into slow lines
  ([From the integration to the angle](signal.md)), these lines reach $a$ far more than $\sigma_f$. A failed force
  law and a failed identity are then read as `not_measured` when the strength by direct averaging passes and the
  short-period scatter of $a$ exceeds `MMR_SHORT_PERIOD_A_RATIO` = 0.5 of the libration's half-excursion in $a$. A
  fail never becomes a pass.

??? info "The guards in detail"
    - *Lines.* Each measurement fits the law at its base cutoff, takes the amplitude $K$, and reads the spectrum of
      $\dot a$ from `MMR_RESPONSE_LINE_BAND_CUTOFFS` = 0.5 × that cutoff to $P/2$, leaving out the libration's
      harmonics. The raised cutoff is at most $P$ / `MMR_RESPONSE_MAX_FRACTION` = 4. The null shifts reach up to half the
      record; without a null (half the record shorter than $1.75P$) the force law is not measured, and below the null
      gain it fails.
    - *Jumps.* Windows within $\tau/2$ + `FILTER_EDGE_TRIM_CUTOFFS` × the response cutoff of a confirmed jump are
      dropped; kicks do not count.
    - *Forced lines.* The lattice holds $1 \le m \le \lvert k\rvert$, $\lvert l_j\rvert \le \lvert k_j\rvert$; a line is
      "at the period" when $\lvert 1/T - 1/P\rvert < 1/(WP)$, $W$ = `MMR_FORCE_WINDOW_CYCLES`. A contaminated L4
      (`rsp_force_contaminated`) never makes a libration dynamic (leaf 5f of [the status tree](status-tree.md)).
    - *Folded short periods* (`MMR_SHORT_PERIOD_A_UNMEASURED`): the check columns keep the measured fail
      (`rsp_short_period_unmeasured`), and the cap `short_period_aliased` stays.

*Meaning.* A pass is support for a physical mechanism: $a$ is pushed back and forth by a force in phase with the
angle. A fail with a bounded angle points to a kinematic or induced origin. *In the output:* `check_mmr_a_response`;
the law taken in `rsp_force_law` (`mmr_numbers.csv`), the harmonic law in `rsp_force_harm_*`.

### The direct averaged-force law

A line of $\dot a$ between the response cutoff and the period can swamp the fitted sine and also add small turns that
shorten the period the turning points give. 59358 8N-13 shows both: the conjunction cycle $13P_N \approx 2.1$ kyr in
$\dot a$, a period of 4.4 kyr from the turns against a swing of 8.8 kyr, and L4 halves of 0.16. Where the harmonic
L4 fails or passes thinly on a two-body angle that is not 1:1 and has a direct strength, the
resonance's own force is then tested directly. Lagrange's equation with the directly averaged disturbing function of
the term, the planet's **real mass** and each window's elements,

$$
\dot a_\text{th}(t) = \frac{2k}{na}\,G m_P\,\frac{d\langle D\rangle}{d\sigma}\big(\sigma(t)\big),
$$

is evaluated along the recorded $\sigma(t)$, and $\dot a_\text{th}$ and the observed $\dot a$ go through the **same**
low-pass at $f\,P_\text{large}$, $f$ = `MMR_DIRECT_FORCE_CUTOFF_CYCLES` = 0.35 (the filter of the force is compared
with the filter of $\dot a$, never $\sin$ of a filtered angle). Only one scale $s$ per window is free: shape and phase
are the theory's. $P_\text{large}$ is the period of the turns that survive a change of the cutoff: the turns of
$\sigma$ low-passed at $f P$ count when their number stays within `MMR_FILTER_GUARD_TP_REL` with the cutoff doubled;
otherwise the doubled cutoff's period is tried (a clean libration keeps its own period at once).

The law is taken when two windows share no sample (one stretch counted twice is not a confirmation), a shifted null
exists, the smaller half median $R^2 \ge 0.4$ (thin below 0.7), every $s$ lies in $[1/\sqrt{10},\ \sqrt{10}]$
(`MMR_STRENGTH_CHECK_RATIO`) and $R^2$ beats the null by 0.2. It lifts L4 (`rsp_force_law` = `direct_force`, rule
`direct_force_law`), never lowers it, never makes an episode, and the level is at most medium (cap
`force_law_theory_only`: a computed first-order average with a fitted scale, not the measured shape). F then reads the
strength the large turns need. It is not consulted where the step folds short periods into $a$ (the guard above):
it reads $a$ too. 59358: $R^2$ 0.75, $s$ 1.14–1.29, null 0.02 → transient, medium; 2005 GQ107 40M-57 stays
uncertain (its episode at 31.7–38.8 kyr holds two overlapping windows only). *In the output:* `rsp_dforce_*`
(`rsp_dforce_reason`: `taken` or the guard that refused it), `geo_period_large_yr`; per episode `force_dforce_*`.

### Co-orbital force law

A 1:1 angle has its own law with no free shape ([The resonant pendulum](pendulum.md#co-orbitals-11)). The model
$\dot a = 2\mu n a\,\tilde F(\theta)$ is evaluated on the *unfiltered* angle and then passed through the data's own
filters (smoothing $\theta$ first would flatten the sharp peak of the force at the planet's passage):

$$
\dot a_\text{data} = K\,\dot a_\text{model} + C ,
\qquad \text{pass if } R^2 \ge 0.4 \text{ in both halves and } K/K_\text{theory} \in [1/3,\ 3],
$$

overall and in both halves; the pass is thin while the smaller half's $R^2 < 0.7$ (`MMR_FORCE_R2_FAIL` = 0.4,
`MMR_FORCE_R2_FULL` = 0.7, `MMR_COORBITAL_SCALE_RATIO_MAX` = 3). $K/K_\text{theory} = 1$ means the libration period is the one the planet's mass
gives, which a kinematic oscillation does not reproduce, so the 45° amplitude floor does not apply and even a small
tadpole is measured. The owner search does not apply to 1:1 angles: every candidate is a multiple of $\sigma$ plus a
combination of the planets alone. The pendulum test (below) is the energy integral of this law, fitted per cycle.

??? info "The retrograde 1/−1"
    The retrograde co-orbital `1J-1r<k>` ($i > 90°$) is judged by the same law, with the body placed at
    $M = \sigma + \lambda' - ((2k-1)\omega + \Omega)$: the angle of Li et al. (2019), built from $\Omega$, $\omega$ and
    $M$ only.

### Local force and energy on a single leg

A co-orbital body is often in the 1:1 well for less than a cycle: it is reversed once or twice by the averaged
potential between two passages, then leaves. Such a stretch closes no cycle, so no episode is accepted, but it may
still be a capture. The single-leg branch (status tree, leaf G4c) reads the *leg*: the
orbit-averaged angle (`MMR_COORBITAL_EDGE_TURN_ORBITS` = 3 orbits) between two reversals of at least
`MMR_COORBITAL_EDGE_TURN_MIN_DEG` = 20°, outside every accepted run, period $2L$ at least the floor less
`CONFIDENCE_THIN_BAND`. Four questions, each from the averaged problem of
[the co-orbital pendulum](pendulum.md#co-orbitals-11):

- **Topology.** Which of the planet ($\theta = 0$) and L3 ($\pi$) the leg crosses: none (tadpole), L3 (horseshoe),
  the planet (quasi-satellite), both (a lap past the planet, no swing of one well).
- **Local law.** The 1:1 law above on the leg widened by `MMR_COORBITAL_LEG_WINDOW_FACTOR` × $L$ on each side, at the
  leg's own cutoff $\max(2P_\text{orb},\ 0.1\cdot 2L)$ (`MMR_COORBITAL_LEG_CUTOFF_ORBITS`,
  `MMR_COORBITAL_LEG_CUTOFF_FRACTION`): the angle's $P_c$ is as long as the shortest legs. Its force check with
  $K > 0$ and its energy integral must pass. On the leg alone the energy is too short to fit; ±0.25 L keeps 85 % of
  the legs of confirmed cycles passing and separates them 2.0× from the single legs of non-resonant rows.
- **Turn conservation.** The averaged problem is time-reversible about a turn, so a reflection mirrors
  $|a - a_P|$: out/in $= |a - a_P|_\text{after} / |a - a_P|_\text{before}$, medians over
  `MMR_COORBITAL_TURN_SIDE_WINDOW` × the shorter adjacent leg on each side, is 1. A close approach (a kick) is not
  reversible: $|\ln(\text{out}/\text{in})| >$ `MMR_COORBITAL_TURN_ENERGY_TOL` = ln 1.4, the 90th percentile of the
  turns of confirmed cycles.
- **Zone.** A level is reversed only below the highest $\tilde R$ before the planet or L3 ahead of the turn: the
  zone $|a - a_P| / \Delta a_\text{max} \le$ `MMR_COORBITAL_TURN_ZONE_MAX` = 1.1 (97.6 % of the turns of confirmed
  cycles; $\Delta a_\text{max} = a_P\sqrt{(8/3)\mu(\tilde R_\text{top} - \tilde R(\theta))}$).

All four hold → a transient of one leg (≤ medium); one fails → uncertain; both turns fail → no capture, but a near
miss (the near-episode cap). Example: 1991 VG's horseshoe leg 552–613 yr follows the Earth's law (K 2.1, energy
passes), but its turns read out/in 0.61 and 1.59: two kicks by Earth approaches, not reflections, so the verdict is
`non_resonant / medium` (near-episode); 2023 OE5's leg 586–665 yr (out/in 0.84 and
1.20, K 1.2, energy passes) is a transient.

## Energy and potential

*Question.* Does $\sigma$ move in a fixed potential well, with the period that well predicts?

*Measurement.* Within each cycle the energy is constant, so a potential $V_E$ (up to `SEPARATRIX_HARMONICS_MAX` = 2
harmonics) is fitted from

$$
\dot\sigma^2 = 2E_w - 2V(\sigma) \qquad \text{(one energy } E_w \text{ per cycle)},
$$

and, independently, $V_F$ from the force, $c\,\dot a = -V'(\sigma)$, with no free constant: a periodic $V$ has no
mean force over a closed cycle, and on a short arc a constant column would take part of the force (137295 1N-2 visits
98° of an asymmetric well; with the constant fitted and dropped, $V_F$'s well moved from 279° to 338°). In a frame
that follows a moving well, $\varphi = \sigma - s(t)$, the frame's own acceleration is taken off instead,
$c\,\dot a - \ddot s = -V'(\varphi)$, so $V_F$ sees the same motion as $V_E$. The test has two parts:

- *$\sigma$ alone*: the **period law** $\lvert\operatorname{median}(T_\text{obs}/T_\text{pred}) - 1\rvert \le$
  `SEPARATRIX_PERIOD_TOL` = 0.2, and the wells fitted in different windows agree (phase coherence $\ge$
  `SEPARATRIX_PHASE_COHERENCE` = 0.9);
- *with $a$*: $V_E$ and $V_F$ correlate at $r \ge$ `SEPARATRIX_VEVF_R_MIN` = 0.9 in at least
  `SEPARATRIX_VEVF_FRACTION` = 2/3 of the eligible windows (one window per cycle or turn). The two are compared on
  the arc of $\varphi$ the window visits (`SEPARATRIX_VEVF_ARC`; outside it both fits are extrapolations), with the
  number of harmonics chosen by held-out prediction (fitted on one half of the cycle, predicted on the other;
  `SEPARATRIX_HOLDOUT_GAIN`). A window is eligible when $\sigma$ swings at least `MMR_A_RESPONSE_MIN_AMP_DEG` = 45°
  in it, or turns: on a shorter arc the shape of $V$ is not constrained (2004 BG41 3J-1: $r$ = 0.64–0.90 at
  ±20–30°, 0.98–1.00 at ±60–108°). No eligible window: not measured, never a pass. `sep_VE_VF_n_eligible` of
  `sep_VE_VF_n_windows` counted, `sep_VE_VF_arc_deg` their median arc.
  A correlation does not see a scale: $V_F = kV_E$ correlates at 1.00 for any $k > 0$. So $V_F$ must also carry the
  observed period (`SEPARATRIX_VF_PERIOD_LAW`): each compared cycle's energy is recomputed in $V_F$ from the observed
  $\varphi$ and $\dot\varphi$ (the same frame and $m(t)$), its period taken in $V_F$'s loop of that energy, and
  $\lvert\operatorname{median}(T_\text{obs}/T_\text{pred}(V_F)) - 1\rvert \le$ `SEPARATRIX_PERIOD_TOL` over the
  eligible windows, else R0 fails. A cycle whose $V_F$ loop is of the other kind (a libration above $V_F$'s top, a
  turn below it) has no law; none left: not measured. For a pendulum at 90°, $V_F = 2V_E$ gives 1.48 and
  $V_E/2$ 0.62. On the benchmark every passing row of this route lies within 0.94–1.14
  (`sep_VF_period_law_med` over `sep_VF_period_law_n` cycles).

$T_\text{pred}$ of a cycle is its period in a potential, through its own turning points. For a two-body angle (not
1:1) with its planet saved, the potential is the **full averaged well**
$V = 3k^2\mu_P n^2 a\,\bar D(\sigma)$ at the cycle's middle time, with nothing fitted (shape and scale are the
theory's), when the swing librates in that well. Otherwise it is $V_E$, in the loop of the cycle's energy around the
minimum whose basin holds the cycle's fastest sample (with two wells, the well the cycle is in, not the deepest one).
A well that moves at a steady rate $\dot\sigma_c$ is followed: the turning points are moved to the well at the cycle's
middle time, with the kinetic energy $\dot\sigma_c^2/2$ they carry. In a frame that follows a moving well the
coherence of the per-cycle wells is reported but not required; the period law carries the test there.

Not measured when $V_E$ explains less than `SEPARATRIX_VE_R2_MIN` = 0.7 of $\dot\sigma^2$ within the cycles, when the
filter is too slow for the cycles, and (the part with $a$) below 45° without a slip over the top. For a 1:1 angle the
energy integral of the co-orbital law replaces this test.

*Per episode* (`SEPARATRIX_R0_EPISODES`). One $V_E$ fitted over every cycle of the record mixes regimes: two episodes
in different wells and the circulation between them (79653 3M+3J-7: $R^2$ 0.25, not measured). When the record's R0 is
not measured, each accepted episode with at least two disjoint full cycles is fitted on its own (its period, its best
frame, the same thresholds); R0 passes when some episode passes and none fails (column `sep_R0_episode_id`), and the
×2 filter guard, which compared another episode, does not withdraw it. 79653's first episode (4.8–32.5 kyr): $V_E$
$R^2$ 0.93, period law 0.96, $V_E \approx V_F$ at $r \ge 0.996$.

??? info "The order of the two parts"
    `zone.r0_checks` decides the part on $\sigma$ alone (R0σ) first. If it fails or is not measured, the part with
    $a$ takes the same value: a failed period law stays an objection even at a small amplitude. Only when R0σ passes is
    the part with $a$ not measured below `MMR_A_RESPONSE_MIN_AMP_DEG` = 45° without a turn (there the identity alone
    makes $V_E \approx V_F$); above it the window fraction decides.

*Meaning.* A pass says the cycles have the period a pendulum of their energy would have; with $a$, that the force on
$a$ is the gradient of the same potential. The full-well period law reads the theory's strength, so it is not
independent of the strength check. *In the output:* `check_mmr_potential`; `sep_period_law_*` in `mmr_numbers.csv`.

## Amplitude of $a$

*Question.* Does $a$ carry the amplitude the libration needs?

*Measurement.* By the identity, each harmonic of $\sigma$ has an amplitude in $a$ of $\omega\,\sigma_\omega/\lvert c\rvert$.
In the band $[P/2,\ 2P]$ around the cycle period (`MMR_B1_BAND_FACTOR` = 2) the package compares

$$
\text{B1} = \frac{\sqrt2\,\operatorname{std}(a_\text{band})}{\sqrt2\,\operatorname{std}(\dot\sigma_\text{band})/\lvert c\rvert}
$$

and fails it below `MMR_B1_RATIO_MIN` = 0.1.

*Meaning.* A ratio below 0.1 puts the oscillation in the perihelion part of the angle: a kinematic libration.
*In the output:* `check_mmr_a_amplitude`.

## Kinematic origins

A libration is **kinematic** when the angle is bounded by geometry while $a$ does not respond. The package looks for
three mechanisms (`origin_detail`):

- **perihelion term** (`varpi_term`): without an episode, the slow part of $\sigma$ follows $p\varpi$ (correlation
  $\ge$ `MMR_KIN_CORR_MIN` = 0.9) while the slow $\dot a$ does not answer $\sin\sigma$ ($R^2 \le$
  `MMR_KIN_SLOW_R2_MAX` = 0.2): the $\lambda$ part cancels $p\dot\varpi$. The amplitude test below 0.1 says the same
  for an episode;
- **secular cycle** (`secular_cycle`): an episode whose period matches the eccentricity cycle $2\pi/\lvert g - g_5\rvert$
  or the $2\omega$ cycle $\pi/\lvert\dot\omega\rvert$ within `MMR_KIN_PERIOD_TOL` = 10 %, while its identity fails and
  its force law stays below 0.4 under every law;
- **planetary ripple** (`planet_ripple`): the period matches a planetary inequality that survived the filter.

**Kinematic drift.** Some angles show episodes that are only a ripple on a drifting center. The angle is a kinematic
drift (rule K2, `rec_rule` 3k2) when the force law, measured on at least `MMR_FORCE_MIN_WINDOWS` = 2 windows, fails under every law, the largest episode swing is below
`MMR_KIN_RIPPLE_SHARE_MAX` = 0.5 of the range of the slow center, and the rate of the slow center correlates
($\ge 0.9$) with the rate of the angle's secular part $s = \sigma - \psi_\lambda$ low-passed at the same $W_c$,
over at least `MMR_KIN_MIN_POINTS` = 3 independent points $(T - 2W_c)/W_c$. Over a fraction of $W_c$ two smooth
curves correlate whatever they are: (485) Genua in 3J-1S-1, a clean libration after 42 kyr of circulation, read
15 kyr of a 42-kyr $W_c$ and one force window, and is transient, not a drift. A
kinematic libration whose center moves by more than `MMR_KIN_DRIFT_SHIFT_RATIO` = 1 amplitude over its episode is
read the same way. *In the output:* `check_mmr_kinematic`, `kin_*`.

**K1 on the episode and the swing shares.** On the decisive episode, $\sigma$ and $p\varpi$ are low-passed at the
episode's response cutoff ($\max(P_c, 0.1\,P_e)$), cut to the episode and detrended; their difference is the
$\lambda$ part $\psi_\lambda$. Three numbers describe who makes the swing:

$$
\text{K1e} = \operatorname{corr}(\sigma, p\varpi), \qquad
\text{swing share} = \frac{\tfrac12\operatorname{range}(p\varpi)}{\tfrac12\operatorname{range}(\sigma)}, \qquad
\lambda\text{ share} = \frac{\tfrac12\operatorname{range}(\psi_\lambda)}{\tfrac12\operatorname{range}(\sigma)} .
$$

A kinematic libration has K1e ≈ 1, a swing share ≈ 1 and a small $\lambda$ share; a pendulum swings $\sigma$ through
its own $a$, so its $\lambda$ share is ≈ 1. The numbers are nan, with the reason in `kin_k1_episode_reason`, for
$p = 0$, a retrograde angle, no librating episode, or an episode shorter than `MMR_KIN_MIN_POINTS` cutoffs.
`kin_check_reason` names why the slow-angle correlation (K1) has no number, e.g. `too_few_points` when the record is
shorter than three center windows. 285637 3M+7J-6: K1e 0.971, swing share 1.13 ($\psi_\lambda$ carries a slow
±7° wave longer than the 18-kyr cycle, $\lambda$ share 0.24). *In the output:* `kin_k1_episode`, `kin_swing_share`,
`kin_lambda_swing_share`.

**When a kinematic libration is measured.** K1e ≥ `MMR_KIN_CORR_MIN`, a flat $a$ (B1 < `MMR_KIN_B1_CLEAR` = 0.01, far
below what a pendulum needs) and a period at a secular one (within `MMR_KIN_PERIOD_TOL`) together measure the
kinematic cause: the margin K1e is full and counts as the verdict's physics, so the verdict may reach `high`
([confidence](confidence.md)). 285637 (B1 6·10⁻⁵) and 481085 7E-10M-6 (3.2·10⁻³) are high; 41588 3J+9S-2 (B1 0.095)
stays medium. No response of $a$ means no resonance: the class stays non_resonant, only the confidence in it rises.

**A slower libration of the center (slow scale).** Whenever the center turns, the package stores the slower
libration at its own scale (`MMR_SLOW_SCALE_REPORT`): the runs G2c finds at the center scale $W_c$, even when they do
not cover the record (episodes.csv rows with `scale` = `slow`), else the turns of the slow center $c(t)$ of a
librating angle. Its period, closed cycles and half-amplitude, its own identity $\dot\sigma_s = c\,(a_s - a_*)$, B1 in
the band of its period, and the nearest secular period ($\pi/\lvert\dot\omega\rvert$, $2\pi/\lvert g - g_j\rvert$)
are columns. A slow swing at a secular period is that cycle seen through the angle's $p\varpi$ part: a modulation, not
a second libration. Its cause is named with a level: *identified* when the period matches a secular period and the
center follows that cycle ($\dot c$ on its harmonics, $R^2 \ge$ `MMR_OWNER_R2_MIN`, or the slow center's rate follows the
secular part of $\sigma$, $\ge$ `MMR_KIN_CORR_MIN`); *possible* with one of the two; *unknown* with neither, and then
it is said as unknown. Where the identity holds on the slow scale the slow libration is a `dynamic_candidate`: it
changes no status. An owner row names its owner's slow $x$ cause first. The swing $p\varpi$ alone makes is
$\lvert p\rvert\arcsin(\min(e_f, e_F)/\max(e_f, e_F))$ (`kin_pred_swing_deg`, next to the measured
`kin_meas_swing_deg`). 37087 3J+8S-3: slow runs 17.3–62.4 and 65.8–96.6 kyr, period 5.69 kyr against
$2\pi/(g - g_5)$ = 5.70 kyr, identity slope/$c$ 2.8 (fails): identified. 2001 HW7 5J+6S-2 (owner row): one slow
cycle, the $2\omega$ cause of its owner's $x$, identified. *In the output:* `slow_*`.

??? info "The perihelion rate of a two-mode eccentricity vector"
    With $z = e\,e^{i\varpi} = e_f e^{igt} + e_F e^{ig_5t}$ (free and forced parts, $\theta$ the angle between them),

    $$
    \dot\varpi = g_5 + (g - g_5)\,\frac{e_f^2 + e_f e_F\cos\theta}{e_f^2 + e_F^2 + 2e_f e_F\cos\theta} ,
    $$

    so $\varpi$, and with it $p\varpi$, can be held by the forced mode while $a$ does nothing.

**A slower libration (B5).** A drifting center could still be a libration slower than the record. B5 excludes it when
the center moves faster than any libration through the observed arc could, given how little $a$ moves; else the cap
`slow_libration_not_excluded` (at most `medium`). In step 3 of [the status tree](status-tree.md), a kinematic
reading whose center rate is consistent with an exact commensurability (within `MMR_B5_EXACT_SIGMA` = 2 standard
errors) becomes `uncertain / baseline_needed` (`rec_rule` 3b5). *In the output:* `rsp_b5_ratio`, `rsp_b5_route`.

??? info "B5: formula"
    On the analysis window of length $T_w$ the center is fitted by a line, $\sigma_c(t) = b + \nu_c t$, with residual
    standard deviation $s_c$; then

    $$
    \delta\nu_c = \frac{2\sqrt3\,s_c}{T_w},\qquad S = \lvert\nu_c\rvert\,T_w,\qquad
    K_{\max} = \frac{\lvert\nu_c\rvert\,\delta a}{1 - \cos(\min(S, 2\pi)/2)},\qquad \omega_{0,\max} = \sqrt{\lvert c\rvert K_{\max}},
    $$

    $\delta a$ the range of the slow $a$ on the window: $K_{\max}$ is the largest force a libration through the
    observed arc could have without moving $a$ by more than observed (worst case: the arc centered on the stable
    point). The slow libration is excluded when $\lvert\nu_c\rvert - \delta\nu_c \ge 2\omega_{0,\max}$ (full at the ratio
    `MMR_B5_RATIO_FULL` = 2), when the center completes a revolution at every smoothing over an interval that overlaps
    the deciding episode, or by an owner.

## Another resonance owns the oscillation

Two angles of the same body, $\sigma = k\lambda + s$ and $\phi_o = k_o\lambda + s_o$, satisfy an exact identity with
$d = \gcd(\lvert k\rvert, \lvert k_o\rvert)$, $u = k_o/d$, $v = k/d$:

$$
\Psi = u\,\sigma - v\,\phi_o \quad (\text{free of } \lambda), \qquad \sigma = \frac{v}{u}\,\phi_o + \frac{\Psi}{u}.
$$

So when resonance B (the angle $\phi_o$) drives $a$, $\sigma$ oscillates with it: an *induced* libration. The identity
holds whoever drives $a$; a shared oscillation is necessary, not proof. Resonance B is accepted as the **owner** on an
interval when its own force explains $\dot a$ there, by one of four routes.

The first three routes are tried only on candidates that pass common admissibility checks on the same windows
(the force law of B is fitted to $\dot a$ band-passed to $[P/f,\ fP]$, $f$ = `MMR_OWNER_BAND_FACTOR` = 2, $P$ the
reference period of the $\dot a$ oscillation):

- B explains $\dot a$ with $R^2 \ge$ `MMR_OWNER_R2_MIN` = 0.7;
- the $\lambda$-free remainder carries little of the oscillation:
  $\operatorname{std}[(\Psi/u)_\text{band}]/\operatorname{std}(\sigma_\text{band}) \le$ `MMR_OWNER_PSI_SHARE_MAX` = 0.1
  (an amplitude ratio, not a power ratio);
- the apparent turns are not a planetary ripple;
- for the $R^2$-gain and phase routes, the phase $\phi_0$ is fixed (coherence $\ge$ `MMR_OWNER_PHI0_COH_MIN` = 0.8, or
  coherent coefficients of B's multiplet);
- a circulating owner (a forced wave) turns at a period inside the band of $a$.

The windows are `MMR_FORCE_WINDOW_CYCLES` = 3.5 periods long at a half-window step, and two of them must fit
between the band-pass edges ($2P$ from either end): about $9.25P$ of record. A shorter record is searched again with
windows of `MMR_OWNER_SHORT_WINDOW_CYCLES` = 2 periods, about $7P$ of record (flag `owner_short_windows`; a multiplet
keeps its modulation-period windows; 385533 7N-12 over $8.1P$ finds no owner among 17 candidates).
When even those do not fit, a third stage trims less: the band-pass edge error falls to its interior level $0.75P$
from an end (measured on the order-2 band $[P/2, 2P]$; `MMR_OWNER_FITTED_EDGE_TRIM` = 0.375 $fP$), and two windows are
fitted to the span left, each between `MMR_OWNER_FITTED_MIN_CYCLES` = 1.5 and 2 periods: about $3.75P$ of record
(flag `owner_fitted_windows`; 385458 9N-17 over 300 kyr, $4.6P$: none of 14 candidates drives $a$). The first two
stages keep the $2P$ trim: on 2001 HW7 over 150 kyr, windows nearer the ends split the owner 5E-16's regimes
differently, and its cover of σ's episode falls from 1 to 0.

The potential route keeps the $R^2$, remainder and ripple checks but needs no fixed phase. The identity route has its
own conditions (in the table).

| route | condition | settings |
|---|---|---|
| $R^2$ gain | $\dot a = R\sin(\phi_o - \phi_0) + C$ fits with $R^2 \ge 0.7$, at least 0.2 above $\sigma$'s own law | `MMR_OWNER_R2_MIN`, `MMR_OWNER_R2_GAIN` |
| phase | the phase $\phi_0$ is fixed (coherence $C_B \ge 0.8$) and exceeds what it could be if $\sigma$'s own term drove $a$: $C_B \ge L_B + 0.2$, $L_B = \lvert\langle e^{i\Psi/v}\rangle\rvert$, while $\sigma$'s coherence stays below $L_\sigma + 0.2$ | `MMR_OWNER_PHI0_COH_MIN`, `MMR_OWNER_PHI0_COH_GAIN` |
| potential | for a two-body B with its planet saved: the directly averaged force $\dot a_B = (2k_o/na)\,\partial\bar R_B/\partial\phi_o$ along the recorded $\phi_o$ predicts $\dot a$ with $R^2 \ge 0.7$ and a scale within $[1/2,\ 2]$ | `MMR_OWNER_POTENTIAL_SCALE_MAX` |
| identity | for a three-body $\sigma$ whose period needs $\ge 10^3\,\mu_i\mu_j e^q$: a two-body $\phi$ follows $\sigma$ by the identity, librates over $\sigma$'s episode and passes its own strength check | `MMR_OWNER_IDENTITY_RATIO_MIN` |

An owner may itself librate (`owner_libration`) or circulate at the right period, a forced wave (`owner_forced_wave`).
It is credited only with its measured regimes (libration or admissible forced wave; a gap between two of its libration
pieces is not its time), and it takes an episode of $\sigma$ only when these intervals cover at least
`MMR_OWNER_COVER_MIN` = 0.5 of it, checked on every accepted episode (rule F7). Near the ends of the record its force
is not measured at all: its windows keep clear of the band-pass edges, so the record's first and last stretch of
about $2P$ is in no window, and neither is the tail the windows' step leaves. There the owner is neither present nor
absent. A window where it circulates too fast to fit is a measurement: the owner is absent there. With covered fraction $c$ and an unmeasured fraction $u$ of the episode, the episode is induced when
$c \ge 0.5$, the row's own when $c + u < 0.5$, and otherwise not measured: it neither counts for the owner nor
decides against it (2001 HW7 without planets: an episode at 19.5 kyr, after the owner's last window ends at 19.0 kyr).
A cover equal to the threshold counts as covering; round-off cannot push it below. An owner in its libration regime
must also be slow enough to make $\sigma$'s cycles (P2): $\Psi/u$ turns less than
`MMR_OWNER_PSI_TURNS_PER_CYCLE_MAX` = 1 per cycle, $\lvert\dot\Psi\rvert P/(2\pi u) < 1$; a closed cycle of $\sigma$
cannot hold more than one turn of the remainder. The episodes it covers and passes P2 for are induced
(`libration_origin` = `induced` in `episodes.csv`); the others are the row's own and decide the status: every episode
induced gives non_resonant / induced_libration, an own episode next to induced ones is read without them (2000 SF8
8M-9: 17E-36 librates before and after its episode, so it is a transient). The owner of a
three-body angle can be a two-body resonance, and the multiplet law above applies to an owner too.

**Two librations are not two resonances.** Membership is judged by the term that moves $a$, not by a librating
angle (Nesvorný & Morbidelli 1998, §2, pp. 3030–3031: a three-body resonance is identified by the correlation of the
filtered $a$ with $\sigma$, a rival one rejected because $a$ does not follow its angles; Gallardo, Coito & Badano
2016, p. 89: at small $e$ a high-order two-body and a zero-order three-body angle both librate, only the second moves
$a$). Two angles sharing $\lambda$ are tied by $\Psi$: while the owner holds $a$, both librate as one oscillation, and
the body is in both resonances only when $x = \Psi/u$ has a well of its own (*confirmed* below). An owner explains the
measured forcing but does not exclude an extra restoring term, so an unresolved test caps the confidence (2001 HW7);
a residual after the owner is no second capture either. A slow swing of the center at $\pi/\lvert\dot\omega\rvert$
or $2\pi/\lvert g - g_j\rvert$ is the secular cycle when $\ddot x$ is a function of $2\omega$ or $\varpi - g_jt$, not
of $x$: a modulation, not a second libration. With no term holding $\lambda$ the same beat is a
[kinematic libration](#kinematic-origins) (285637, 481085): $a$ stays flat, so non_resonant. Perihelion dominance is no
veto by itself, and an apocentric center is a branch of a resonant model, not a synonym for non-resonant.

*Two resonances at once?* Both angles can librate together only if $\Psi$ is itself bounded, and a $\Psi$ held by
$\sigma$'s own term is a pendulum in $x = \Psi/u$. A second capture is excluded by either of two routes:

- *sweep and energy*: $x$ sweeps at least `MMR_OWNER_CAPTURE_SWEEP_DEG` = 180° and $\dot x^2$ is not a function of
  $x$ ($R^2 < 0.7$ on its first harmonics);
- *secular forcing* (F3): $\ddot x$ follows the secular cycle, $R^2 \ge 0.7$ on the harmonics of $2\omega$ or of
  $\varpi - g_jt$ ($j = 5..8$), and not $x$ itself, $R^2 < $ `MMR_OWNER_CAPTURE_SELF_R2_MAX` = 0.2 on
  $[1, \sin x, \cos x]$: the remainder is pushed by the perihelion's cycle, not held in a well of its own (78085
  5J-4S-1: 0.98 against 0.001, while $x$ moved only 155°).

*Confirmed* (the third outcome): "both angles librate" is not "in both resonances"; $\Psi$ is, when $x$
has a well of its own. All of these on the test window: $x$ is bounded (sweep < 180°), $\dot x^2$ is a function of
$x$ and $\ddot x$ of $\sin x$ ($R^2 \ge 0.7$ both), its period $P_x$ (the median full period of its turns at the
center-turn prominence `LIBRATION_CENTRE_TP_PROMINENCE_DEG` = 10°) is no secular one ($|P_x/(P_\text{sec}/k) - 1| >$
`MMR_KIN_PERIOD_TOL` for $\pi/|\dot\omega|$, every $2\pi/|\dot\varpi - g_j|$ and $k = 1, 2$: a small forced
sinusoid passes both fits like a pendulum, only its period tells it), and $x$ closes at least
`LIBRATION_CENTRE_MIN_ESTIMATES` = 2 cycles (a monotonic $x$ fits the energy integral trivially: 2013 TW135, $R^2$
1.00, no turn). The row then keeps the tree's own reading of $\sigma$ with origin `joint` (detail `joint_capture`,
`owner` = the other resonance, role `joint`; rule `both_resonances_confirmed`), when that reading claims the
libration; another reading stands as it is. With one cycle the row is `uncertain / baseline_needed`, origin `joint`.
The owner's own row is untouched. No benchmark row is confirmed (45 owner rows; the bounded $x$ of nine
of them has no turn but HW7's one).

Otherwise it stays unresolved and caps the confidence at medium (2001 HW7: $\ddot x$ follows $2\omega$ but also
$\sin x$, $R^2$ 0.63). A slow wobble of the center on an owner row is described, not judged: the period of $x$ and its
secular cause, `identified` when its period matches $\pi/|\dot\omega|$ or $2\pi/|g - g_j|$ (or a harmonic) within
`MMR_KIN_PERIOD_TOL` and $\dot x$ follows that term with $R^2 \ge 0.7$, `possible` when one of the two holds (HW7:
6.6 kyr against $\pi/|\dot\omega|$ = 7.2 kyr, identified). A secular
or Lidov-Kozai angle never demotes an MMR verdict: it has no $\lambda$, so no identity ties them. *In the output:*
`check_mmr_owner`, the role columns below, `own_*`; the candidates are in `candidates.csv`.

??? info "Owner routes and the second capture in detail"
    - *Identity route.* The two-body $\varphi$ follows $\sigma$ on the decisive window as $\sigma_f = c\,\varphi_f + a + bt$
      with $\lvert c/(k_\sigma/k_\varphi) - 1\rvert \le$ `MMR_OWNER_IDENTITY_COEF_TOL` and a residual at most
      `MMR_OWNER_IDENTITY_RESID_MAX` × the half-amplitude; $\varphi$ librates there by its own analysis, and its directly
      averaged strength passes. Nothing is measured on $a$.
    - *Multiplet owner* (`MMR_OWNER_LAW` = `multiplet`). An owner whose single phase drifts is fitted with all its
      members $\varphi_j = \varphi_o + j(\varpi - \varpi_P)$, and must then beat $\sigma$'s own constant-coefficient $R^2$
      by 0.2.
    - *Outside the episode.* An owner whose intervals cover less than `MMR_OWNER_COVER_MIN` of an own episode (rule F7)
      sets the flag `owner_outside_episode`; one that covers it but fails P2 sets `owner_psi_too_fast`. Its key goes to
      `neighbours` with where it acts (before, during or after the row's own episodes), the origin is the tree's.
    - *Second capture window.* The test runs on the hull of the owner's intervals that meet $\sigma$'s episodes: a test
      window only, never owner time.
    - *Second capture.* $x = \Psi/u$ is low-passed at `MMR_OWNER_CAPTURE_LOWPASS_PERIODS` = 3 periods before any
      derivative; $\dot x^2$ is fitted as a function of $x$ with `MMR_OWNER_CAPTURE_HARMONICS` = 2 harmonics, and F3
      fits $\ddot x$ on each secular phase alone with the same number of harmonics. A secular term or a period of $x$
      is read only when `MMR_OWNER_X_MIN_CYCLES` = 2 of its cycles fit in the test window (over less, the cosine of a
      nearly uniform phase is a low polynomial in $t$ and fits any trend). The phases are $2\omega$ and the body's
      $\varpi$ against the giant planets' modes $g_jt$, not their osculating $\varpi_j$ (the fit's cosine and sine
      take up the phase). Unresolved: the cap `second_capture_unresolved`; an exclusion by F3 fires the rule
      `second_capture_f3`.
    - *Well selection (F2), a column only.* Over $\sigma$'s episodes in the test window, the Rayleigh length of the
      owner's centers against $\sigma$'s (78085: 0.97 against 0.24: the owner returns to its own well, $\sigma$ does
      not). Not a criterion: the episodes are few and an owner that re-captures is what rule 2 already says.
    - *Catalog longitudes.* A planet whose $\lambda$ was not saved enters the owner search (and nothing else) as the
      line $\lambda_{j0} + n_jt$. Candidates whose decision lies within the phase error this causes are
      neither accepted nor rejected (flag `owner_catalogue_ambiguous`), and every such search carries the cap `catalogue_longitudes` (at most `medium`).

### Roles of the other resonance {#roles}

The verdict names the other term by what it does; only an owner, a driver or the partner of a confirmed joint capture
goes into `owner`, so the column suggests two resonances at once only when they were measured (role `joint`):

| `cause_role` | when | column | example |
|---|---|---|---|
| owner | another commensurability librates and makes every deciding episode (F7, P2; `owner_libration`) | `owner` = `cause_term` = its key | 2001 HW7 5J+6S-2: 5E-16 |
| driver | it circulates and its forced wave makes the cycles (`owner_forced_wave`) | `owner` = `cause_term` = its key | 385266 16N-27: 3N-5+0+2 |
| neighbor | an accepted owner that does not make the row's own episodes (`owner_outside_episode`, `owner_psi_too_fast`): the body is in it before or after, not at the same time | `neighbours` = `KEY t0–t1 before\|during\|after;…` | 1999 GL4 1V-5: 4J-1 after |
| empty | rule C: the librating member of the same multiplet; one resonance, its status is the row's | `resonant_argument` | 31935 9M-16+0+7: 9M-16+2+5 |
| kinematic | the secular part of $\sigma$ makes the cycles, $a$ does not respond | `cause_term` = the term | 285637 3M+7J-6: `-4varpi` |
| joint | 'both resonances at once?' confirmed: the other commensurability librates and $\Psi$ has a well of its own (origin `joint`) | `owner` = `cause_term` = its key | — (none on the benchmark) |

`commensurability` (`9M-16`) is the key a statistic counts once per body. The place of a neighbor's block is read by
its midpoint against the span of the row's own episodes.

## Subresonances

The members of one multiplet, $\sigma_s = \sigma + \Delta_s$, share the fast part and differ by a slow angle
$\Delta_s = m(\varpi_P - \varpi)$ plus node terms, including the members $\sigma \pm 2m\omega$ (`MMR_SUBRESONANCE_OMEGA_HARMONICS` = 2)
through which an inclined orbit's center moves with $\omega$ (Gallardo 2006, §3). If member $s$ librates about a fixed
center $c_s$, the catalog angle's center drifts as $c_s - \Delta_s(t)$. Nodal members join when the inclination to
the invariable plane is at least `MMR_SUBRESONANCE_NODAL_INC_MIN_DEG` = 10°. Nodal members are evaluated in the
invariable-plane frame and **listed for diagnosis only**: a nodal winner is never accepted. The check is then
`not_measured`, with the flag and cap `nodal_subresonance_candidate` (confidence at most `medium`), and nothing is
passed to the catalog row. What follows applies to accepted planar members.

A member that stands still is not enough: in a libration about a moving well every member's frame looks steady. A
member is accepted when its center range is at most `MMR_SUBRESONANCE_RANGE_RATIO` = 0.5 of the catalog angle's and
does not drift or revolve; its d'Alembert factor is at least `MMR_SUBRESONANCE_STRENGTH_REL_MIN` = $10^{-3}$ of the
strongest term of the commensurability (a weaker member would librate over 30 times slower and cannot set the shared
oscillation); it librates itself in the window; its force-law phase is coherent ($\ge 0.8$, and 0.2 above the
catalog angle's) where the swing exceeds 45°; its center lies near 0 or $\pi$ (within 30°, at low $e$ and $i$); it is
unique; and its theoretical small-amplitude period is at most `MMR_SUBRESONANCE_T0_RATIO_MAX` = 2 times the observed
period (it is a resonance at all). Every one of these tests is a veto applied before the choice, so a member that fails
one hands the choice to the next. The choice itself is physical, not a score: for a two-body angle whose orbits do not
cross, the theory chooses the member whose theoretical center $c_j(t) = \pi - \arg P(\beta(t)) + j\beta(t)$ is the
flattest over the episode, provided the measured center follows the theory (RMS ≤ `MMR_SUBRESONANCE_THEORY_RMS_DEG` =
20°); the measured center range $W$ breaks ties (within `MMR_SUBRESONANCE_TIE_FRACTION` = 20 %) and is the fallback
(three-body, crossing orbits, a theory that does not follow). Each accepted episode is scanned in its own window; the
row's member is that of the episode its status rests on, and episodes.csv names each episode's own. When nothing is
admissible, the theory follows, the members are told apart ($\beta$ moves by more than
`MMR_SUBRESONANCE_CENTRE_MOVE_RATIO` × the half-amplitude) and the two flattest theoretical centers tie (within the
tie fraction: two members of equal weight do, since $C_a = C_b$ gives
$P = 2C\cos\frac{(a-b)\beta}{2}\,e^{i(a+b)\beta/2}$ and both centers move by $|a-b|/2$ of $\beta$'s range), the
regime is **composite**: the oscillation is about the moving center of $P(\beta)$, the catalog angle's own verdict
stands, with `origin_detail` `composite_multiplet` and `resonant_argument` `composite`. An episode whose own window
cannot decide (members only the unmeasured test A could tell apart) caps the row as the decisive one would
(`subresonance_undetermined`). The theory names the same member as before on the five two-body accepted rows of the
benchmark (12218, 27011, 31935, 65366, 67495; 82831's orbits cross Mars: W). An accepted member gets a row of its own
(`added_by = subresonance_check`) and the catalog angle gets origin `subresonance`
([Gallery](gallery.md#transient-subresonance)). *In the output:* `check_mmr_subresonances`, `sub_*`.

??? info "The subresonance scan in detail"
    The scan runs on a trigger: a drift; a moving, modulated or unclosed center; $a$ responding with a rotating
    $\phi_0$; a kinematic or unexplained-drift verdict; another member leading by the averaged coefficients; the
    theoretical center of the catalog member $\pi - \arg P + j\beta$ moving over an accepted episode by more than
    `MMR_SUBRESONANCE_CENTRE_MOVE_RATIO` = 1 × its half-amplitude (T7, two-body). A center that moves by more than that
    while its bands still overlap, or an `origin_unknown` verdict (2003 ND, 2003 EH16), runs the scan as a
    **diagnostic** (T8): its numbers go to `sub_diag_*`, it never accepts a member and the check stays
    `not_applicable`. Members are
    every d'Alembert-valid split of the slow coefficients up to the catalog's largest order
    (`MATRIX_2BODY_ORDER_MAX`, `MATRIX_3BODY_ORDER_MAX`); nodal members are built wholly in the invariable plane.

    The tests, as the columns name them: E′ (the d'Alembert factor), L (the member librates in the scan window), A
    (a coherent force-law phase), B (center near 0 or $\pi$), C (unique; a tie goes to the lower degree), E (a
    resonance at all: the period ratio). Test D compares the center with the composite multiplet theory
    $c_0(\beta) = \pi - \arg\sum_j C_j e^{ij\beta}$ (Morbidelli 2002, §9.3–9.4) and only raises the confidence. When
    several members pass and only A could tell them apart, no member is accepted (cap `subresonance_undetermined`);
    the other subresonance caps are `subresonance_tie`, `subresonance_t6_only` and `subresonance_theory_unconfirmed`.

    `sub_top` lists the best `MMR_SUBRESONANCE_TOP_N` = 10 members as `key:W/W0:E:N:A/W`: the range $W$ of the
    member's slow center against $W_0$ of the catalog angle, $E = 1 - \operatorname{var}(c_s)/\operatorname{var}(c_0)$,
    $N$ the number of its slow cycles the window holds, $A/W$ the swing over $W$; `sub_rank1..3_pass_deg` is the
    largest monotone pass of that member's center in the scan window (≈ 360° for a center that turns once there, which
    step 1 rejects). The list decides nothing.

    **Cheap numbers on every row** (prograde, not 1:1; scan or not; nothing reads them for the status). The theory
    (two-body, planet saved, orbits not crossing) gives the planar shares $|C_j|/\sum|C_j|$, Rouché for the catalog
    member, $|P|/\max|P|$ of the compound term $P(\beta) = \sum_j C_j e^{ij\beta}$ inside the gaps between accepted
    episodes and at their edges, and the center travel behind T7. A gap where $|P|$ nearly vanishes is the compound term
    canceling as $\beta = \varpi_P - \varpi$ passes its cancellation phase (12218 4M-7: 9·10⁻⁵ between 48.7 and
    76.2 kyr). The cheap list puts every planar member up to the catalog's order on the existing center of the
    longest accepted episode, with test E′ and step 1: `sub_cheap_n` pass both, `sub_cheap_best` is the first of them,
    else the flattest frame. A flat frame with a negligible factor is a frame, not a force (2003 UG22 40 kyr: +5+2 and
    +6+1, factors 10⁻⁸ and 10⁻¹⁰).

## Onset of an episode (O1)

*Question.* Is the episode a capture, or does it continue an $a$-wave that was already there?

*Measurement.* With the band-passed axis $a_\text{band} = \operatorname{LP}_{P/2}(a_f) - \operatorname{LP}_{2P}(a_f)$
($P$ the episode's libration period),

$$
\mathrm{O1} = \frac{\operatorname{std}(a_\text{band})_\text{before}}{\operatorname{std}(a_\text{band})_\text{episode}} .
$$

The background is the $2P$ before the episode, or before its group: an earlier episode ending inside that stretch with
no observed circulation in the gap joins the group, and an earlier one followed by a circulation cuts the stretch at
its end. The stretch is also cut at confirmed jumps of $a$ and at the filter edges; with less than one period $P$ of
clear background (or of episode), O1 is not measured.

*Applicability.* O1 tests a capture claim: the angle's own libration, a subresonance or a librating owner. It does
not apply to a kinematic libration, a forced wave of a circulating owner, or a non_resonant verdict of unknown origin;
nor when an already librating owner covers at least `MMR_O1_OWNER_COVER_MIN` = 0.5 of the background; nor when the
episodes lie in the separatrix zone ([What the angle does](geometry.md#the-separatrix-zone)), where the wave before
is the same force's near-separatrix circulation.

A large ratio alone does not show a foreign wave. Before a capture from circulation the body circulates near the
separatrix under the same force (Murray & Dermott 1999, §8.12), so $a$ already swings at about $P$; after an encounter
kick the swing before is noise. Three more numbers on the background say which it is:

- O1c, coherence: $R^2_\text{coh}$ of one sinusoid $A\cos(2\pi t/P - \varphi) + C$ fitted to $a_\text{band}$ there;
- O1c, phase: $|\varphi_\text{bg} - \varphi_\text{ep}|$, with $\varphi_\text{ep}$ fitted on the episode's first $2P$
  (both phases are absolute in $t$, so this is the background wave extrapolated into the episode; a capture starts
  its own phase);
- O1l, own law: $R^2$ of $\dot a$ on the background under the episode's own force law,
  $\dot a = K\sin(\sigma - \varphi_0) + c$ with the episode's $K$ and $\varphi_0$ fixed (only $c$ free; for a 1:1
  angle the co-orbital law with the episode's $K/K_\text{th}$), over the same $R^2$ on the episode.

*Meaning.* Full below `MMR_O1_RATIO_THIN` = 0.25, thin from 0.25 to below `MMR_O1_RATIO_FAIL` = 0.5. From 0.5 O1
fails only for a foreign wave: coherent ($R^2_\text{coh} \ge$ `MMR_O1_COHERENCE_R2_MIN` = 0.5), in phase (gap ≤
`MMR_O1_PHASE_GAP_MAX_DEG` = 90°) and not explained by the own law (the ratio of the two $R^2$ below
`MMR_O1_OWN_LAW_R2_FRACTION` = 0.5); otherwise it is thin. The three settings are conventions. A part that cannot be
measured is no evidence of a wave (O1c) and explains nothing (O1l). A failed applicable O1 caps the confidence at
`low` (`onset_failed`). Leaf 5o ([The status tree](status-tree.md)), where nothing else measured the force, still
reads the ratio alone: narrowed, it would move two author labels (2013 VA14 over 60 kyr, the background incoherent
at its period; 2003 TG2 over 100 kyr, coherence 0.23), whose objection is the strong wave itself.
*In the output:* `ep_onset_ratio`, `ep_onset_coherence_r2`, `ep_onset_phase_gap_deg`, `ep_onset_own_law_r2`,
`ep_onset_own_law_r2_episode`, `mrg_O1`.

## Strength

*Question.* Could this resonance produce the observed period at all?

*Measurement.* The required strength $\mu_\text{req}$ from the observed small-amplitude period against the theory
$\mu_\text{eff,th}$ from the averaged potential ([The resonant pendulum](pendulum.md#how-strong-must-the-resonance-be)):
pass when the range of $\mu_\text{req}/\mu_\text{eff,th}$ over the snapshots overlaps $[1/10,\ 10]$
(`MMR_STRENGTH_CHECK_RATIO`). For three-body angles the theory is only the scale $\mu_i\mu_j e^q$, and the check is a
support, not a measurement of the force.

*Meaning.* When neither the force law nor the energy could be measured, the strength is the only physics left. If the
observed cycles would need a force at least `MMR_STRENGTH_IMPLAUSIBLE_RATIO` = 1000 times what the resonance gives (for
a three-body angle, even after the small-divisor envelope), the bounded angle is not this resonance: `non_resonant`,
reason `implausible_strength` ([Gallery](gallery.md#implausible-strength)). The strength check does not apply to 1:1
angles, which have their own law. *In the output:* `check_mmr_strength`, `str_*`.

## What the tests decide

The tests give `libration_origin` as in the table at the top; the reason each combination gives is listed in
[Every reason](status-tree.md#every-reason), and the order in which the tests are applied, and what happens when
they disagree, is on [The status tree](status-tree.md). How much each pass, thin pass or failure weighs is on
[Confidence](confidence.md).
