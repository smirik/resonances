# Physics notes the code relies on

Conclusions measured during the programme, with the objects that showed them. The formulas are in
`docs/libration-analysis.md`; here is why the code is built the way it is.

## The a↔σ identity and what it forbids

- σ̇ = Σ k_j n_j + k n(a) + p ϖ̇ ties the angle to a: a resonant-angle swing of half-amplitude A_σ with period P implies
  A_σ = 1.5 |k| n A_a P / (2π a). "σ and a share a period" is therefore automatic and is not evidence; the atlas
  reconstructed σ from a within 1–2 % in all 121 cases, forced ripples included.
- The fits of a's response (V_E) and of the force law (V_F) are linked by the same identity: V_E ≈ V_F is necessary,
  not sufficient. A synthetic kinematic swing passed every window but its period law gave T_obs/T_pend = 0.61,
  against ≈ 1 for real pendulums. The nonlinear period law (R0) is what separates them.
- L3 (the identity's slope) and L4 (the force law) are not independent; a measured L3 failure overrides a good L4
  (37087 3J+8S-3: seven 167-yr episodes pass strict L4 and fail L3 with the opposite slope).
- Below a half-amplitude of 45°, sin σ ≈ σ and the force law cannot be told from the identity: L4 and R0 with a are
  not_measured (2348 4J-2S-1 at 14–21°).

## Kinematic and induced librations

- 285637: a −4ϖ beating looks like a clean ±26° libration, but a responds 3000× too weakly.
- 37087 3J+8S-3: the episode cycle sits on the secular period 2π/|g − g5| (`secular_cycle`); the 1-Myr record is the
  benchmark's reference kinematic libration.
- 101703 41J-14: the apparent 166.2-yr libration is driven by 3J-1 (164.4-yr lattice line); the owner fit R² 0.855
  against σ's own 0.085, coherence 0.978 → non_resonant/induced.
- A geometrically librating multiplet member can be physically negligible: 2003 UG22 10E-17+7+0 has
  (e_E/e)^7 ≈ 6e-11 and must not displace the real member.
- 2012 KL9: accepted 0.5–0.6° loops follow an external 222-yr wave of a that is not on the angle lattice; F6 misses
  them, strength rejects the row.
- A detuned linear forced secular vector (two lines 4.32″/yr apart) produces a bounded 14.5° angle at 300 kyr without
  any trapping: a bounded angle alone is not a mechanism.

## Strength

- Two-body strength is computed: the curvature of the directly averaged potential at the occupied centre.
- Three-body strength is known only to a coefficient of order 1–100 (Gallardo 2006, p. 30): μμe^q is an order
  estimate, not a bound. Hence a three-body libration whose force cannot be measured is capped at medium
  (`strength_scale_only`). 2348 4J-2S-1: required/e^q = 1.59, while its large-swing siblings (force measured) have
  1.30–1.66 and μ_force/μ_req ≈ 0.95–1.03.
- A theoretical three-body strength needs second-order averaging in the masses (Morbidelli 2002, ch. 10; Nesvorný &
  Morbidelli 1998): the harmonic pairs divided by their own frequencies, a refusal near two-body commensurabilities.
  Neither Gallardo's ρ nor a "clean-model force law" supplies it (see [rejected.md](rejected.md)).
- The small-divisor envelope B is capped by the observed swing, B ≤ n·min(P, P_cyc)/2π; it may not diverge near
  commensurability (W22, R-DIV).

## Aliasing and sampling

- Stroboscopic ghosts lie at n(a_m) = n_res + 2πm/(kΔt) with m = round(νΔt/2π) and apparent rate ν − 2πm/Δt (the sign
  was fixed by a test dispute).
- 1999 UQ 7E+2M-9 at a 5.0012-yr step is a ghost: the predictive unwrap gives 3998.84 turns (3998.86 at 1 yr),
  `np.unwrap` gives −0.16; mean a is 4e-6 AU from the ghost at 1.0941756 AU. Its 97-yr wave is real Earth 8:7 forcing
  (present at 1 yr, at 5.0012 yr and in a 0.0127-yr pilot).
- Density rule Δt ≤ P_c/20; FAIR needs Δt ≤ P/4.2. Integrator: ias15 for NEAs (q < 1.3 au) and planet-crossers; SABA
  with dt ≤ min(58 d, P/10, τ_peri/10).
- Folded short-period lines are negligible in σ (≤ 0.024 of the half-amplitude on 47 fine re-integrations); they
  matter through a. Only four rows lost their verdict to aliasing (TG2 4E-1M-3, TK14 31M-34, UC22 16M-27,
  433 47M-44); their 0.5-yr twins are in the benchmark.
- The alias gate reads the λ part of the angle: 17346 had a λ residual of 0.34° but a full-angle residual of 173.5°
  across low-e perihelion flips.

## Multiplets (sisters)

- Sisters share the fast wiggle; only the centre tells them apart. 65366 6M-11 and 31935 9M-16: separations
  6.3e-6 / 3.2e-6 AU against widths ~1e-4 AU (λ = 0.032 / 0.018): one overlapped multiplet, a modulated pendulum.
  The composite centre c0(β) = π − arg P(β) fits to 8–10° RMS against 21–27° for one sister.
- The guiding member is the largest coefficient (Murray & Dermott §8.14.2), and only over an episode: for 65366 and
  31935 Rouché dominance fails and the winding over a full β cycle is zero, so the origin is reported as composite,
  not one member as the global owner; the binomial mode is
  j = floor((q + 1)x/(1 + x)), with j = 0 largest for qx ≤ 1 (the earlier x < 1/(2q − 1) rule was wrong).
- ϖ-type and node-type members can drift at the same rate (g − g_P ≈ −(s − s_P)); rate matching alone cannot choose.
- The catalogue row takes the status of an accepted member of its multiplet (rule C).

## Separatrix and moving centre

- The pulsating-zone half-range is q(e_max − e_min)/(2 mean e) (it had been doubled: 52356 0.609 → 0.304,
  469333 0.139 → 0.070). min/mean speed < 1/4 maps to pendulum h ≈ 1.012, not 1.07. Remove secular trends before
  defining a zone, or a trend-driven capture widens its own zone.
- 145480 1N-4 needs an asymmetric two-harmonic potential (second/first 1.91; energy R² 0.30 → 0.87).
- 2013 AS27 31E-36 librates about a moving centre (literature: Gallardo 2006 p. 30; Wisdom 1985 p. 274): track RMS
  1.55° against 14.4° fixed and 28.2° reversed; near-zero osculating MOID was not an encounter (minimum 0.0714 AU =
  7.1 Hill radii). Its 4E+9S-5 row is induced.
- 2003 UG22 10E-17: the full flat-bottom well predicts the periods within 0.98–1.10 against 0.76–0.80 for a cosine.
- 23240 1M+3J-3: force-plus-wave fits predict both halves (R² 0.72) yet K_J/K_cat crosses 1 between equally good
  fits (0.64–1.34), changing the well topology: uncertain means the well is unidentified, not that there is no
  resonance.
- The 52356 slip-placement significance p < 5e-5 was overstated: 0.0007 for independent placements, 0.017 for cyclic
  shifts that keep the spacing.
- 33128 6N-7: chaos and sustained libration coexist (Lyapunov 4.2–4.4 kyr against P_lib 9.1 kyr); one slip is an exit
  plus a capture.
- In a two-harmonic well every loop of one energy has one period in either well, so R0 reads each cycle in the well
  that holds its fastest sample (`pendulum.cycle_period`).
- 2003 TX9 3E+2J-10: an amplitude step (146° → 12°, a swing ÷7) right after Mars encounters at 6.49–6.87 kyr; the
  regimes' periods do not hold in one shared potential on every cycle (law 1.61 on a late cycle), so no cut.

## Co-orbitals

- Force sign: with R positive, da/dt = +2 ∂⟨R⟩/∂σ / (na); a minus gradient needs U = −R. A signed K catches a sign
  error that R² cannot; time-weighted averaging uses uniform mean anomaly.
- Evaluate the force on the raw or briefly smoothed angle, then filter it like a; smoothing σ first ruined 2002 AA29
  (R² 1.00 → 0.04). Use the full 3D eccentricity and orbit normal.
- R² ≥ 0.8 alone is not enough: the passing orbit 85770 fits with R² 0.883, K/K_th = 1.020.
- Earth-only fits to Earth+Moon integrations carry the Moon's 1.2 % mass (K/K_th ≈ 1.0123).
- The free-element basis does not apply to co-orbitals: their forced vector follows the perturber's own vector.

## Secular

- From the ecliptic no planet's node circulates (the 1.58° invariable-plane tilt dominates), and only Mercury's,
  Jupiter's and Saturn's osculating perihelia follow their modes. The zero inclination mode s5 is the invariable plane
  (physical); the eccentricity basis has no physical zero mode, its constant term only absorbs a slow arc.
- The model has no GR: g1, g2, s1, s2 are off by 0.1–0.3″/yr; with GR La90 is reproduced to 0.02–0.08″/yr. The
  inner frequencies and phases are measured on the package's own model.
- The measured mode longitudes drift from the linear law by tens of degrees over 20 Myr (inner modes, s6).
- The secular cutoff must include the angle's longest forced wave: 4305 (27/18 kyr waves) and 8089 (137/68 kyr) hid
  real librations behind a fixed window.
- A frequency error matters through accumulated phase: residual fraction √(1 − sinc²(δT/2)); 0.1″/yr at 500 kyr
  leaves ~7 %, 0.05″/yr at 5 Myr ~34 %.

## ZLK

- A bounded osculating ω in a Trojan can be forced-eccentricity geometry (591986; Vinogradova 2024 on shifted ZLK
  centres): the free ω winds while the forced e holds the osculating one.
- e–i exchange also occurs with circulating ω (41/186 bodies), so it cannot decide ZLK.
- A planetary-frequency free residual means leakage or a true secular resonance; either makes the ZLK reading
  ambiguous (withhold, do not call it kinematic; 4257).
- Different baselines are different evidence: 162474 is episodic over 1.5 Myr, resonant on its first 0.5 Myr alone.
- SABA cannot follow Earth-crossers (3752, 10563 end at −99); 162474 needs SABA dt = 0.5; 3040's verdict over 500 kyr
  of SABA at dt = 1 changes with a few weeks of start epoch (chaotic).

## Literature

Murray & Dermott (1999); Morbidelli (2002); Nesvorný & Morbidelli (1998, 1999); Gallardo (2006, 2014); Gallardo, Coito
& Badano (2016); Wisdom (1985); Shevchenko (2020); Vinogradova (2024); Knežević & Milani; Laskar (1990); Souami &
Souchay (2012); Li, Huang & Gong (2019); Morais & Namouni (retrograde λ*, ϖ*); Milani, Chesley & Valsecchi (2000) and
the clone literature in `docs/clones.md`.
