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
| `subresonance` | another member of the same multiplet librates about a fixed centre | [82831](gallery.md#transient-subresonance) |
| `unknown` | bounded, mechanism not established | [2003 TX9](gallery.md#uncertain-origin), [geometric libration](gallery.md#geometric-libration) |

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

The fitted phase tells where the centre is, but its sign depends on $c$: $\sigma$ is stable at $\phi_0$ when $c<0$ and
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

## Energy and potential

*Question.* Does $\sigma$ move in a fixed potential well, with the period that well predicts?

*Measurement.* Within each cycle the energy is constant, so a potential $V_E$ (up to `SEPARATRIX_HARMONICS_MAX` = 2
harmonics) is fitted from

$$
\dot\sigma^2 = 2E_w - 2V(\sigma) \qquad \text{(one energy } E_w \text{ per cycle)},
$$

and, independently, $V_F$ from the force, $c\,\dot a = -V'(\sigma)$. The test has two parts:

- *$\sigma$ alone*: the **period law** $\lvert\operatorname{median}(T_\text{obs}/T_\text{pred}) - 1\rvert \le$
  `SEPARATRIX_PERIOD_TOL` = 0.2, and the wells fitted in different windows agree (phase coherence $\ge$
  `SEPARATRIX_PHASE_COHERENCE` = 0.9);
- *with $a$*: $V_E$ and $V_F$ correlate at $r \ge$ `SEPARATRIX_VEVF_R_MIN` = 0.9 in at least
  `SEPARATRIX_VEVF_FRACTION` = 2/3 of the windows.

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

**Kinematic drift.** Some angles show episodes that are only a ripple on a drifting centre. The angle is a kinematic
drift (rule K2, `rec_rule` 3k2) when the force law fails under every law, the largest episode swing is below
`MMR_KIN_RIPPLE_SHARE_MAX` = 0.5 of the range of the slow centre, and the rate of the slow centre correlates
($\ge 0.9$) with the rate of the angle's secular part $s = \sigma - \psi_\lambda$ low-passed at the same $W_c$. A
kinematic libration whose centre moves by more than `MMR_KIN_DRIFT_SHIFT_RATIO` = 1 amplitude over its episode is
read the same way. *In the output:* `check_mmr_kinematic`, `kin_*`.

??? info "The perihelion rate of a two-mode eccentricity vector"
    With $z = e\,e^{i\varpi} = e_f e^{igt} + e_F e^{ig_5t}$ (free and forced parts, $\theta$ the angle between them),

    $$
    \dot\varpi = g_5 + (g - g_5)\,\frac{e_f^2 + e_f e_F\cos\theta}{e_f^2 + e_F^2 + 2e_f e_F\cos\theta} ,
    $$

    so $\varpi$, and with it $p\varpi$, can be held by the forced mode while $a$ does nothing.

**A slower libration (B5).** A drifting centre could still be a libration slower than the record. B5 excludes it when
the centre moves faster than any libration through the observed arc could, given how little $a$ moves; else the cap
`slow_libration_not_excluded` (at most `medium`). In step 3 of [the status tree](status-tree.md), a kinematic
reading whose centre rate is consistent with an exact commensurability (within `MMR_B5_EXACT_SIGMA` = 2 standard
errors) becomes `uncertain / baseline_needed` (`rec_rule` 3b5). *In the output:* `rsp_b5_ratio`, `rsp_b5_route`.

??? info "B5: formula"
    On the analysis window of length $T_w$ the centre is fitted by a line, $\sigma_c(t) = b + \nu_c t$, with residual
    standard deviation $s_c$; then

    $$
    \delta\nu_c = \frac{2\sqrt3\,s_c}{T_w},\qquad S = \lvert\nu_c\rvert\,T_w,\qquad
    K_{\max} = \frac{\lvert\nu_c\rvert\,\delta a}{1 - \cos(\min(S, 2\pi)/2)},\qquad \omega_{0,\max} = \sqrt{\lvert c\rvert K_{\max}},
    $$

    $\delta a$ the range of the slow $a$ on the window: $K_{\max}$ is the largest force a libration through the
    observed arc could have without moving $a$ by more than observed (worst case: the arc centred on the stable
    point). The slow libration is excluded when $\lvert\nu_c\rvert - \delta\nu_c \ge 2\omega_{0,\max}$ (full at the ratio
    `MMR_B5_RATIO_FULL` = 2), when the centre completes a revolution at every smoothing over an interval that overlaps
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

The potential route keeps the $R^2$, remainder and ripple checks but needs no fixed phase. The identity route has its
own conditions (in the table).

| route | condition | settings |
|---|---|---|
| $R^2$ gain | $\dot a = R\sin(\phi_o - \phi_0) + C$ fits with $R^2 \ge 0.7$, at least 0.2 above $\sigma$'s own law | `MMR_OWNER_R2_MIN`, `MMR_OWNER_R2_GAIN` |
| phase | the phase $\phi_0$ is fixed (coherence $C_B \ge 0.8$) and exceeds what it could be if $\sigma$'s own term drove $a$: $C_B \ge L_B + 0.2$, $L_B = \lvert\langle e^{i\Psi/v}\rangle\rvert$, while $\sigma$'s coherence stays below $L_\sigma + 0.2$ | `MMR_OWNER_PHI0_COH_MIN`, `MMR_OWNER_PHI0_COH_GAIN` |
| potential | for a two-body B with its planet saved: the directly averaged force $\dot a_B = (2k_o/na)\,\partial\bar R_B/\partial\phi_o$ along the recorded $\phi_o$ predicts $\dot a$ with $R^2 \ge 0.7$ and a scale within $[1/2,\ 2]$ | `MMR_OWNER_POTENTIAL_SCALE_MAX` |
| identity | for a three-body $\sigma$ whose period needs $\ge 10^3\,\mu_i\mu_j e^q$: a two-body $\phi$ follows $\sigma$ by the identity, librates over $\sigma$'s episode and passes its own strength check | `MMR_OWNER_IDENTITY_RATIO_MIN` |

An owner may itself librate (`owner_libration`) or circulate at the right period, a forced wave (`owner_forced_wave`).
It takes $\sigma$'s episode only when its intervals cover at least `MMR_OWNER_COVER_MIN` = 0.5 of it. The owner of a
three-body angle can be a two-body resonance, and the multiplet law above applies to an owner too.

*Two resonances at once?* Both angles can librate together only if $\Psi$ is itself bounded. When $x = \Psi/u$ sweeps
at least `MMR_OWNER_CAPTURE_SWEEP_DEG` = 180° and $\dot x^2$ is not a function of $x$ ($R^2 < 0.7$ on its first
harmonics), a second capture is excluded; otherwise it stays unresolved and caps the confidence at medium. A secular
or Lidov-Kozai angle never demotes an MMR verdict: it has no $\lambda$, so no identity ties them. *In the output:*
`check_mmr_owner`, `owner`, `own_*`; the candidates are in `candidates.csv`.

??? info "Owner routes and the second capture in detail"
    - *Identity route.* The two-body $\varphi$ follows $\sigma$ on the decisive window as $\sigma_f = c\,\varphi_f + a + bt$
      with $\lvert c/(k_\sigma/k_\varphi) - 1\rvert \le$ `MMR_OWNER_IDENTITY_COEF_TOL` and a residual at most
      `MMR_OWNER_IDENTITY_RESID_MAX` × the half-amplitude; $\varphi$ librates there by its own analysis, and its directly
      averaged strength passes. Nothing is measured on $a$.
    - *Multiplet owner* (`MMR_OWNER_LAW` = `multiplet`). An owner whose single phase drifts is fitted with all its
      members $\varphi_j = \varphi_o + j(\varpi - \varpi_P)$, and must then beat $\sigma$'s own constant-coefficient $R^2$
      by 0.2.
    - *Outside the episode.* An owner whose intervals cover less than `MMR_OWNER_COVER_MIN` of the episode (rule F7)
      sets the flag `owner_outside_episode`, and the tree goes on.
    - *Second capture.* $x = \Psi/u$ is low-passed at `MMR_OWNER_CAPTURE_LOWPASS_PERIODS` = 3 periods; $\dot x^2$ is
      fitted as a function of $x$ with `MMR_OWNER_CAPTURE_HARMONICS` = 2 harmonics. Unresolved: the cap
      `second_capture_unresolved`.
    - *Catalogue longitudes.* A planet whose $\lambda$ was not saved enters the owner search (and nothing else) as the
      line $\lambda_{j0} + n_jt$. Candidates whose decision lies within the phase error this causes are
      neither accepted nor rejected (flag `owner_catalogue_ambiguous`), and every such search carries the cap `catalogue_longitudes` (at most `medium`).

## Subresonances

The members of one multiplet, $\sigma_s = \sigma + \Delta_s$, share the fast part and differ by a slow angle
$\Delta_s = m(\varpi_P - \varpi)$ plus node terms, including the members $\sigma \pm 2m\omega$ (`MMR_SUBRESONANCE_OMEGA_HARMONICS` = 2)
through which an inclined orbit's centre moves with $\omega$ (Gallardo 2006, §3). If member $s$ librates about a fixed
centre $c_s$, the catalogue angle's centre drifts as $c_s - \Delta_s(t)$. Nodal members join when the inclination to
the invariable plane is at least `MMR_SUBRESONANCE_NODAL_INC_MIN_DEG` = 10°. Nodal members are evaluated in the
invariable-plane frame and **listed for diagnosis only**: a nodal winner is never accepted. The check is then
`not_measured`, with the flag and cap `nodal_subresonance_candidate` (confidence at most `medium`), and nothing is
passed to the catalogue row. What follows applies to accepted planar members.

A member that stands still is not enough: in a libration about a moving well every member's frame looks steady. A
member is accepted when its centre range is at most `MMR_SUBRESONANCE_RANGE_RATIO` = 0.5 of the catalogue angle's and
does not drift or revolve; its d'Alembert factor is at least `MMR_SUBRESONANCE_STRENGTH_REL_MIN` = $10^{-3}$ of the
strongest term of the commensurability (a weaker member would librate over 30 times slower and cannot set the shared
oscillation); it librates itself in the window; its force-law phase is coherent ($\ge 0.8$, and 0.2 above the
catalogue angle's) where the swing exceeds 45°; its centre lies near 0 or $\pi$ (within 30°, at low $e$ and $i$); it is
unique; and its theoretical small-amplitude period is at most `MMR_SUBRESONANCE_T0_RATIO_MAX` = 2 times the observed
period (it is a resonance at all). An accepted member gets a row of its own
(`added_by = subresonance_check`) and the catalogue angle gets origin `subresonance`
([Gallery](gallery.md#transient-subresonance)). *In the output:* `check_mmr_subresonances`, `sub_*`.

??? info "The subresonance scan in detail"
    The scan runs on a trigger: a drift; a moving, modulated or unclosed centre; $a$ responding with a rotating
    $\phi_0$; a kinematic or unexplained-drift verdict; another member leading by the averaged coefficients. Members are
    every d'Alembert-valid split of the slow coefficients up to the catalogue's largest order
    (`MATRIX_2BODY_ORDER_MAX`, `MATRIX_3BODY_ORDER_MAX`); nodal members are built wholly in the invariable plane.

    The tests, as the columns name them: E′ (the d'Alembert factor), L (the member librates in the scan window), A
    (a coherent force-law phase), B (centre near 0 or $\pi$), C (unique; a tie goes to the lower degree), E (a
    resonance at all: the period ratio). Test D compares the centre with the composite multiplet theory
    $c_0(\beta) = \pi - \arg\sum_j C_j e^{ij\beta}$ (Morbidelli 2002, §9.3–9.4) and only raises the confidence. When
    several members pass and only A could tell them apart, no member is accepted (cap `subresonance_undetermined`);
    the other subresonance caps are `subresonance_tie`, `subresonance_t6_only` and `subresonance_theory_unconfirmed`.

    `sub_top` lists the best `MMR_SUBRESONANCE_TOP_N` = 10 members as `key:W/W0:E:N:A/W`: the range $W$ of the
    member's slow centre against $W_0$ of the catalogue angle, $E = 1 - \operatorname{var}(c_s)/\operatorname{var}(c_0)$,
    $N$ the number of its slow cycles the window holds, $A/W$ the swing over $W$. The list decides nothing.

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

*Meaning.* Full below `MMR_O1_RATIO_THIN` = 0.25, thin from 0.25 to below `MMR_O1_RATIO_FAIL` = 0.5, failed from 0.5.
A failed applicable O1 caps the confidence at `low` (`onset_failed`), and in the branch where nothing else measured
the force it is an objection that makes the status `uncertain` (leaf 5o, [The status tree](status-tree.md)).
*In the output:* `ep_onset_ratio`, `mrg_O1`.

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
