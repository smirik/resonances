# Status on 2026-11-01

Replace this page at each programme step. W28 (the author's card answers of 2026-10-30/31) is implemented, not
committed (commit only on the author's command). Latest benchmark run: `test-benchmark` (+ `-secular-zlk`) of
2026-11-01, the frozen W28 code after four post-F7 fixes (the explained run is `v2-w28`; the two differ only in 55390
1J+3S-1, back to non_resonant/high). Against `v2-w26-final`: 49 status moves (co-orbital 34, mmr2 9, secular 1, ZLK 5),
29 level-only moves, 7 new L1 records (unlabelled). `KNOWN_DISAGREEMENTS.csv`: 32 rows (open_question 18, old_verdict
14), all from W28; agreement: status 173/179, angle_status 124/126, confidence 127/133, author_verdict 157/173,
secular/ZLK status 39/40. Episode gate: 146 cases, A 55/55, B 34 + E67 (known_fail), C 3 failing (not gating).
Tests on 2026-11-01: test-fast 5212 passed, 5 xfailed (2.5 min); test-slow 406 passed, 1 xfailed (9.5 min);
test-benchmark 655 passed, 3 xfailed (47 min with the reclassification; 20 min earlier the same day).
`resonances/data/confidence_cells.json` rebuilt on `v2-w28` + secular/ZLK (387 labelled, 332 counted, 2 errors at
medium: 2019 DJ1, 2025 YU11; 55 abstentions). The record of W28: `benchmark/DECISIONS.md` (entries 2026-10-31 and the
two of 2026-11-01), `dev/decisions.md`.

## Open for the author (W28 review page; folder `benchmark/data/figures/review-w28`, 366 rows)

No fix before he has discussed these (his request).

- Decisions d01–d16: d01 2016 JA confidence (q-2 b against his high; E71); d02 33128 6N-7 confidence (q-2 b);
  d03 2019 DJ1 transient from the C4 edge turn; d04 2026 AC4 transient or uncertain (q-3 A, force law unmeasured);
  d05 2007 VU6 edge-turn threshold (E67); d06 the g01-5 rows now uncertain/one_coorbital_leg (C6 against "statuses do
  not change"); d07 2015 WZ12, 2024 AV2, 2024 XP16, 2022 BB2; d08 2009 BE81 20 kyr medium or low; d09 2023 RS2 ZLK −9
  or transient (deciding episode 2.4 < 3 periods); d10 1981 ZLK 1 Myr (free node on s6); d11 10563 ZLK 100 kyr (−9
  review or extend); d12 5026 secular: the mode-phase table (`within_20myr` demotes it, the epoch-anchored one keeps it
  resonant); d13 2004 TK14 1 yr: the lost resample advice; d14 17346 2J-1 low or medium; d15 2003 ND 40 kyr reads
  circulation/high; d16 labels Claude wrote under his name (2020 CB, the card-derived 2306 and 5026 rows).
- Codex visual review (9 serious, 29 minor; the serious ones adjudicated by Claude):
  - real bugs, not fixed: **S1** 42946 43S-6, G4h's edge test (`geometry.half_librations`) ignores σ_f's turns
    between the last turning point and the window edge (8.1 turns there), so a half-libration followed by circulation
    reads as cut by the record end and asks for a futile 100-kyr extend; **S7** 2007 VU6, the fixed 20° edge-turn
    prominence (`MMR_COORBITAL_EDGE_TURN_MIN_DEG`) rejects an 18.8° turn that closes the 839–994 yr cycle (proposed: a
    threshold in units of the averaged angle's own noise, all co-orbitals rerun);
  - text defects (no status): S2 WK15 coverage sentence, S3 2002 BG ripple counted as an episode in the prose, S4
    RS2 prints the record's e–i instead of the episode's, S5 2001 HW7 150 kyr "both resonances at once?" read over the
    owner hull that spans ~93 kyr of circulation, S8 XP16/OE5/YU11 support counted from the circulation checks, S9
    AC4 says "not measured" where the cause is `filter_too_slow`;
  - physics question: S6 2016 JA after 540 yr (bounded tadpole below the turn prominence: transient or resonant;
    proposed: extend the record to ~5 kyr).
- Labels wanted: the 7 L1 records (385533 300 kyr, 1999 GL4 0.25 yr, 2001 HW7 150 kyr ×2, 2003 EH16 30 kyr, 2003 ND
  40 kyr, 1981 ZLK 1 Myr), 2014 BT8 (one QS loop), 42946 43S-6, the single-leg rows where his two answers disagree.

## Next steps (agreed order)

1. The author's W28 review (above); then the fixes he chooses (S1 and S7 first), one test-benchmark, commit on his
   command.
2. A real co-orbital test.

## Separate tasks

- **Computed three-body strength** (2348 4J-2S-1 medium only because of `strength_scale_only`; would also move 68436
  and 24820 3J-1S-1 and the origin of 48789). Paths: second-order averaging (Morbidelli 2002 ch. 10), an off-resonance
  numerical probe (untested idea), or a family calibration (cheap, data-dependent). Task description:
  kept in the local working notes (not in git).
- **2003 TX9 two-regime split (Q1b).** The author accepted uncertain with one episode. A split needs each regime's
  period measured in one shared two-well potential on every cycle; test 7 currently fails on late cycles.
- **Barycentric elements for distant bodies (W28 B6a, cards g07-5/g07-6).** Heliocentric elements osculate about
  GM_sun alone; a distant body moves about the Sun plus the planets inside its orbit (ε ≈ 1.34e-3), so a carries a
  wave at its own P_orb of half-amplitude 2 ε e a (measured on the benchmark series 2026-10-31: 15874 0.126 au vs
  0.128 predicted, 523683 0.0379 vs 0.0380, 145480 0.082 vs 0.079, 469333 0.053 vs 0.049). B6a filters it
  (`MMR_FILTER_ORBIT_FACTOR` = 1.5: P_c ≥ 1.5 P_orb, binds beyond a planet's 1:2: 23 benchmark rows). Barycentric
  elements would remove the wave at its source but change a and every angle; not done without a separate check.

- **Leaf 5o reads the old O1 ratio (W28 B4).** The cap `onset_failed` reads the narrowed O1 (a coherent, in-phase
  background a-wave not explained by the episode's own law); leaf 5o still reads the ratio alone, because narrowed it
  would move the author's uncertain labels of 2013 VA14 (60 kyr: background coherence 0.69, phase gap 135°) and 2003
  TG2 (100 kyr: coherence 0.23). Open: what 5o should require when nothing else measured the force. 17346 2J-1 loses
  `onset_failed` under the narrowed O1 (own law 0.69 of 0.79) against the s-1 control label: in KNOWN_DISAGREEMENTS.

- **GQ107 and the direct averaged-force law (W28 B8, cards g07-2/g07-3).** On 2005 GQ107 40M-57's E01 (31.7–38.8
  kyr) the direct law with the real mass of Mars gives R² 0.70/0.78, s 0.73, null 0.14: it would pass and make the row
  transient. It is refused only because E01 holds two overlapping windows (3.5 × 1.11 kyr each), no two disjoint ones
  (`no_independent_windows`); the author's label (uncertain, 14.10, kept 31.10) stands on that. A longer record of E01
  would pass: ask the author whether the force measurement then outweighs his label. Also open: the direct law for a
  `direct_harmonic` strength (the profile is in extended precision; not measured now).

## Deferred episode mechanisms (episode gate `known_fail`)

- Only E67 (2007 VU6, tier B) is `known_fail` (codex S7, d05). E15 (B2's forced (strength) mark), E17 (rewritten,
  g11-4 A), E21 (rule q), E30 (rule s) and E43 (2021 XO3, one 1:1 episode with marks, C7) pass since W28. Tier C fails,
  not gating: E71 2016 JA (confidence only), C09 1998 QA62, C10 17346 2J-1.
- Episode rule (b) (the claimed regular core must pass by itself) is rejected in its current form (W28 g11-6 B, g08-5
  B; `rejected.md`): on 2003 TX9 3E+2J-10 (5 yr) it gives resonant/low by losing R0's measured objection at the new
  scale, and it shifts 719 55S-8's episode. Return condition: the same candidate's measured objection must survive a
  scale change (strength and energy re-checked on every allowed scale, a conflict = uncertain with a
  scale-sensitivity mark; synthetic test: a split is not accepted when only its remainder passes; TX9 at 5 and 0.5 yr
  stays uncertain). Meanwhile TX9 (both steps) is uncertain with action extend to 30 kyr (B9, A-FW: two force
  windows for the late regime; the early R0 failure stays).

## Known gaps in the code

- Left after W28 (cosmetic, no status): an owner cover that is not measured (NaN: the episode lies outside the owner's
  measured span, FIX-owner-edge) prints as a dash in the roles line and the right column, without its own words; the
  2001 HW7 150-kyr "both resonances at once?" line reads the owner hull, not the deciding episode (codex S5); the
  docs gallery has no secular transient example since 2306 became uncertain/insufficient_time (anchor
  `secular-z1-unfinished-swing`). The q-2 b ceiling fires no Rule (`rules_fired` is empty on 2023 QD2, 33128), and the
  C3 floor and C7 wells remove `filter_sensitive` without recording their rule (DH65, GF1, NL, CD).
- Secular bounded angles get a dynamic origin directly; an episodic ZLK keeps it when the whole-record gate withholds
  and its episodes are not all pinned (W26: median |z_free|/|z_forced| < 1 and >= 0.5 free turns per whole cycle in
  every episode makes it kinematic, 624 Hektor).
- The secular `FilteredAngle` leaves the MMR regime-cut options (split regimes, return closure) at their defaults.
  The 88899 ZLK two-island loss is fixed by the half-turn run break (W28 D3: two episodes at 67-169 kyr about 90 deg and
  191-294 kyr about 270 deg, transient/medium before D10; D01 is tier B on the episodes); its status without a gate
  verdict follows q-1 E (D10: e-i class `none` there, so uncertain -9 is expected).
- **Non-1:1 episode boundaries, q-5 (i): implemented (W28 C6 G4h, W28 C16 `regime_cut_mark`).** C15 audited them on 2026-11-01. It covered all 460
  mmr2/mmr3 examples and 466 angles (data in the local working notes).
  - **Boundaries by cause.** 374 boundaries next to an accepted episode:
    - 146 separatrix crossings;
    - 159 gaps (A7's union);
    - 17 jumps of a;
    - 51 K3 steps, stalls, regular splits and trims;
    - 0 G2a cuts. Every step fails a test; TX9 fails test 7, so it is one episode.
  - **Steps and stalls as marks: "same region" only.** A step or stall becomes a mark only when it lies between two
    accepted pieces whose adjacent cycles share a band m ± A and a period (ratio ≤ LIBRATION_REGULAR_PERIOD_RATIO).
    - This physical reading merges 9 boundaries in 7 angles and moves no status (prototype).
    - It keeps VA14 #4/#5 (bands disjoint, E17).
    - The naive "every step is a mark" breaks E17 and moves 2002 BG 20 kyr.
    - 82831's 28.6/29.8-kyr boundary is a regular split, so (i) keeps it.
  - **Single half-libration → uncertain.** Required: the resonance's own one-harmonic law (R² ≥ MMR_FORCE_R2_FAIL
    in both halves) with restoring signs at both turns. Only 42946 43S-6 qualifies (it ends at the record edge:
    extend); 18 rows would move by the count alone.
  - **Three-body floor.** No three-body cycle can hide below the 150-yr floor: min T0_th is 3.9 kyr, which would
    need 665 × μ e^q.
  - **Done.** C6 implemented the half-libration leaf (G4h, `MMR_SINGLE_HALF_LIBRATION`; 42946 uncertain + extend);
    C16 the marks (`LIBRATION_REGIME_CUT_MARKS`): on one-example checks the 9 boundaries in 7 angles join, no verdict
    moves, VA14 60 kyr keeps two episodes and 82831 its regular split. The author can still overturn the two
    readings. The full run found one wrong C16 union (55390: centre travel −299°), fixed by bounding a union to one
    turn of the centre; 42946's G4h edge reading is a bug (S1 above).
- **Physical evidence behind positive statuses (W28 D9, g09-2, Codex audit).** MMR:
  a positive status needs a measured force (L4, R0 or F). Secular: no physical status condition; the response column
  (`check_secular_e_response`) passes 11 of 27 positive rows (every ν6/ν5/ν16-type row and 518; 1 of 7 z1, 0 of 6
  z2, not 21028, 4305, 155) and 0 of 9 non-resonant rows. Its failure on z-type rows is not a measurement of absence
  (1020: R²_e 0.002, R²_i 0.012: other waves dominate ė and di/dt). Candidate checks, not built: the action response
  on the Poincaré actions Γ = L − G, Z = G − H with the trade ΔΓ:ΔZ = p:q of the angle (q Γ − p Z conserved), and an
  independent averaged secular torque (the analogue of MMR's F). ZLK: resonant needs the gate confirmed (6/6, each
  with e-i zlk); 8 transients need no physical pass (gate without verdict; 1866 kinematic but not pinned).
- **e-i for q-1 E (W28 D9 → D10).** Measure in the deciding episode at its period, not over the whole record: the whole
  record mixes circulating stretches, whose constant-H trade lies near P (88899 500 kyr: in both episodes `none` for
  two blind estimators, H not conserved, η_H 1.1-1.4; over the whole record the package says none, Claude's
  estimator zlk; 2020 PP1 whole record zlk, deciding episode none). Two blind estimators agree 51/51 with the rule
  antiphase, |coherence| ≥ 0.5, η_H = |δH|/|cos i δG| ≤ 0.5 (zlk items ≤ 0.28, others ≥ 0.58), ≥ 3 periods; the
  first-order slope factor 2 fails on large cycles (477492 ratio 2.03 for Codex, 1866 3.0, with H cancelled to 4-5 %).
  Episodes of 1.5-3 periods: the two agree 3/11, so the 3-period minimum stays. Under this rule 373136 500 kyr, 3752
  500 kyr (Codex window-sensitive) and 797204 500 kyr stay transient; 88899 500 kyr, 797204 100 kyr and **2023 RS2**
  (2.4 periods: not measured) become uncertain −9 (D10, implemented). RS2 carries the author's 30.10 blanket label
  transient/certain: a KNOWN_DISAGREEMENTS row (open_question), d09.
- **Record edges, flag extend_time (W28 C14, g01-7 A).** 2022 BB2 1V-1 stays open until it is integrated further: its
  last ~600 yr are an arc after 52-yr revolutions (final_regime arc, flag `extend_time` forward to 2 kyr; status
  uncertain/one_coorbital_leg from C6, q-4 A, not the card's non_resonant/medium). The cap `record_ends_unresolved`
  binds only a non_resonant circulation; on the current tree no co-orbital row takes it (BZ147 and DJ1, the other two
  end slowdowns of a circulation, are transient now). The flag alone marks the edge reflections (2008 EZ84 backward,
  2021 VF, 2022 RA, 2007 VU6 ... forward) and the clean 1:1 edge legs (2018 KS, 2015 TG24); counts in the W28 F7 run.
  The flag's length is round_up(2T) without an estimate; nobody has integrated a flagged row further yet, so whether
  the arc ends in a capture is open. Backward integration is not wired into the run tools.
- The averaged-route J/J_s (W25) ignores the moving-well kinetic term, as h did, and reads the profile's well count on
  its coarse grid (a very small dimple can be missed). A one-well potential above its top still reports J/J_s = 1
  (`separatrix`); a several-well one reports NaN (`not_measured`).
- The low-memory allocator is macOS-only; Linux was not measured.
- Clones: a restore keeps the clones' stored verdicts even when their series were saved (with `SAVE` ≠ all those are a
  status-selected subset; a recompute of all clones would need every series and a clone-sized analysis budget; waits
  for the author). `restore(recompute_librations=True)` rewrites the nominal's clones.csv columns and summary.csv rows
  together (W26). The resume
  hash includes the planets' initial state and the package versions (strict by design). No triage (decided 2026-10-28:
  a small spread in a does not certify the nominal's verdict, and integration is cheap compared with analysis).
- MPC `propagate` is checked over 138 days only; `MPC_LOOKUP=file_then_api` (get-orb fallback) is off by default and
  names are passed to get-orb unmapped (not verified live).
- ZLK (W25): the e–i coupling phase tolerance is 45° (591986's forced beat sits 32° from in-phase); the 591986 slow-line
  reconstruction reaches r = 0.991 only with a quadratic detrend; `sec_centre_*` of modulated secular rows is covered
  by a synthetic test only (no benchmark row); the node lock is a flag since W26 (`zlk_node_lock`: 591986 4 Myr g5,
  500 kyr unresolved); the 4-Myr record is the benchmark example `zlk__591986__LK__4myr` (author's verdict pending).
  The attribution's beat test compares P_cyc with 2π/|b_ϖ − g_j| within the Rayleigh width plus P_cyc's own grid
  precision, 1296000 Δt/P² (W26; 591986 4 Myr: 0.32 + 3.5″/yr).
- The three-body catalogue axis applies the body's precession correction once (`mmr/three_body.py:107`) with the
  small-α g; measured harmless (≤ 4e-7 au iterated, ≤ 2.5e-4 au from the g formula, against a 0.02 au search window;
  the classifier uses its own `exact_axis`).
- Moving-centre geometry has two documented possible false refusals with no real-record example yet: the continued
  potential minimum can jump basins when a minimum disappears, and a time-symmetric track cannot beat its reversed
  null (W18).

- Owner per episode (A1, W28): an episode is attributed whole (cover ≥ `MMR_OWNER_COVER_MIN`) or not at all; Codex's
  alternative, cutting an episode at the owner's boundary, and a per-component attribution (two resonances sharing
  one episode) are open. A real short slip of 1–3 turns between two owner librations is no owner time any more, so a
  production cover may fall below 0.5 (accepted by the author, g04-2); a measured transition could replace it
  later. The mixed path (induced and own episodes, second pass) is not on the benchmark, synthetic tests only.
  719 36J-13 #4 (a slow component) was not touched (g04-3).
- Second capture F3 (A2, W28): the secular phases are 2ω and ϖ − g_jt for j = 5..8 only; the inner modes are left out
  (chaotic, g3 ≈ g4), so an NEA forced by an inner mode gets no F3 exclusion and no `own_x_cause` from it. A joint
  capture modulated at 2ω stays unresolved (the safe side). The third outcome "both confirmed" (card g05-8 B) is A2b.
  HW7's slow x is `identified` as the 2ω cycle (6.6 vs 7.2 kyr, ẋ R² 0.74); on its 150-kyr record (L1, unlabelled)
  5J+6S-2 reads non_resonant/episodic/medium, induced_libration (5E-16 is transient/high); the joint-window text there
  is codex S5.

- Length-limited extend (B9, W28): the cause is read from L4's own window count and leaf 5n only. The kinematic
  slow-angle point count (`MMR_KIN_MIN_POINTS` × W_c, column `kin_check_reason` too_few_points) is not one: that test
  does not judge a librating angle (`decide.kinematic_check`), and it would ask for 5 W_c (105–112 kyr for 2003 ND and
  EH16). EH16 (5g) is extended for its force windows, which the episode-cycle kinematic route needs. The owner's
  `owner_too_few_windows` decides none of the leaves 5e/5g/5x/5n and is not read. `round_up` keeps one significant
  digit, so 79653's 122 kyr asks for 200 kyr.

## Provisional thresholds

- Lidov-Kozai thresholds, split by what they change (W28 D12, g11-9 A: text only; the recalibration of class 2 waits
  for a larger ZLK sample, today they rest on 591986 and 162474 and a handful of benchmark rows). Each class was checked
  against the code that reads it:
  1. column only: `ZLK_MIN_ANTICORRELATION = 0.5` (`zlk_coherent`, set on two objects), `ZLK_COHERENCE_CHECK`, the W25
     e–i conventions `ZLK_EI_*` (`zlk_ei_coupling`, `confirmed_ei`), `ZLK_CENTRE_MIN_CYCLES` (the search band of the
     centre's slow line, `zlk_centre_*`/`sec_centre_*`; the card put all `ZLK_CENTRE_*` in class 2, but this one bounds
     a column only), `ZLK_NODE_LOCK_MIN_CYCLES` (a flag). Since D10 (q-1 E) `ZLK_EPISODIC_NO_VERDICT` with
     `ZLK_EI_MIN_CYCLES`, `ZLK_EI_PHASE_TOL_DEG`, `ZLK_EI_EPISODE_MIN_COHERENCE` and `ZLK_EI_H_RESIDUAL_MAX` are class 2:
     an episodic row without a gate verdict is transient only when its deciding episode's e–i is zlk, else −9.
  2. may change a status: `ZLK_CENTRE_MODEL_R_MIN`, `ZLK_CENTRE_MODEL_AMP_FACTOR` (the centre-scale guard,
     `sec_centre_rejected`), `ZLK_RHO_WINDOW_CYCLES`, `ZLK_COORBITAL_HILL_FACTOR`, `ZLK_EPISODE_FREE_TURNS_PER_CYCLE`
     (kinematic episodes), the period floor and turning-point prominence (`ZLK_LIBRATION_PERIOD_MIN_YR`,
     `ZLK_TP_PROMINENCE_DEG`, `ZLK_TP_RIPPLE_FACTOR`), `ZLK_FILTER_GI_FACTOR`, the gate's `FREE_GATE_MIN_CYCLES` and
     `FREE_GATE_MASK_QUANTILE`; `ZLK_RUN_BREAK_HALF_AMPLITUDE_DEG` (90°) and `ZLK_ANGLE_FRAME` change statuses too but are
     physical / a model choice, not provisional numbers.
  3. budget (action and `extend_to_years` only, `analysis.secular.extend`): `ZLK_EXTEND_RESOLUTION_MARGIN`,
     `ZLK_EXTEND_MAX_YR` (extend or review), and the N8 factors shared with MMR, `MMR_EXTEND_MIN_SPAN_FACTOR` = 2 and
     `MMR_NEXT_CHECK_SPAN_FACTOR` = 4 (and the MMR-only `MMR_EXTEND_FORCE_WINDOWS` = 2 of B9's A-FW).
- Conventions, documented as such: `SECULAR_LIBRATION_PERIOD_MIN_YR` (200 kyr), ZLK period floor 2 kyr, TP prominence
  5°, response R² 0.5, `MMR_SHORT_PERIOD_HARMONICS` = 2, `LIBRATION_CIRCULATION_NET_REV` = 2, the N8 factors 2 and 4, C14's `MMR_END_ARC_SLOWDOWN_FACTOR` = 2 (a
  slowdown, set between XF62 20 kyr 1.005 / TY11 1.10 and BB2 11.5) and `MMR_END_ARC_MIN_FLOOR_PERIODS` = 1 (2D ≥ floor).
- `MMR_OWNER_POTENTIAL_CONFIRM` (direct averaging as the deciding confirmation of a two-body owner) is off by default.
- `MMR_OWNER_CAPTURE_SELF_R2_MAX` = 0.2 (F3; 78085 0.001, HW7 0.63, the nearest unexcluded secular-forced row 2004 PY27
  0.30) and `MMR_OWNER_X_MIN_CYCLES` = 2 are conventions.

## Unresolved science

- 2021 XA1 1V-1 (W28 C5, card g02-8 A, the G1q audit): not a G1q defect. The stay in the quasi-satellite well
  (0-479 yr) triggers the pass, but for Venus at e = 0.41 P_QS (150 yr) is the tadpole period, so the
  quasi-satellite floor (75.2 yr) equals the tadpole floor (75.7) and the pass repeats the first one. What the record
  holds: one loop of ~110 yr over 0-200 yr that follows the 1:1 law (local law R^2 0.84, K/K_theory 1.43, energy
  integral fails), then 50-75 yr wiggles about a drifting centre that do not (200-430 yr: R^2 0.22; Earth crossing,
  q = 0.43 au), swinging 7-10 deg below the local turning-point prominence (13.8 deg). Status unchanged
  (non_resonant/medium). Open: whether the single 0-200 yr loop is an episode is the single-leg question (C6, q-4 A).
- 2014 BT8 1E-1 (W28 C5): the lowered G1q trigger accepts one quasi-satellite loop 794.5-804.2 yr (local law K/K_theory
  1.08, energy passes), the same physics as 2002 PN, so non_resonant/high -> transient/medium. Card g01-5 A put BT8
  among the 13 rows whose status stays (medium only); PN's answer (B, with "technically transient if it converges
  better") is the more specific one for a clean quasi-satellite loop. To confirm with the author.

- A centre that turned back (W28 B6c against D6). MMR: two slow turns read as `insufficient_time` only when the
  half-swing obeys the resonance's own pendulum (G5w: slow identity, a swing and period within √10 of the directly
  averaged well); unmeasured (three-body, 1:1) stays `drift`, because the author's drifts 101703/142694 1V+3S-7 have two
  turns too (they fail the slow identity, −0.27/−0.25). Secular/ZLK (D6, `centre_turned_insufficient`): any turn back is
  `insufficient_time`, with no well test, since no computed secular well exists to test it against. Not yet asked:
  should the secular rule also need a physical match once a secular pendulum is available, or should the MMR one drop
  it for two-body angles? Moves at B6c: 523683 2U-9 (few_cycles 200 kyr → arc 400 kyr, the author's g07-5) and
  385458 9N-17 at 100 kyr (few_cycles 200 kyr → arc 300 kyr: identity 1.00, a 1.01, period 0.97; its 300-kyr record
  is resonant/high, so the slow swing is the libration). The card's other fix, keeping a centre-scale (G2c) 'slow part
  unresolved' outcome in `_decide_at_scale`, was not needed: G5w reads the same centre track at the fine scale.

- 591986: the free-element g is 220″/yr (the 6-kyr free precession, seen directly as the period of the osculating ω
  libration) against 6.65″/yr in the proper-element catalogue (definition not resolved). Its ~54-kyr centre wave is the
  g6 − g5 beat of Jupiter's own forced vector (W25); over 4 Myr its node librates against g5 (1.46 Myr).
- 78085: the captures are clear, yet the unwrapped angle keeps growing (to be analysed later).
- Two resonances at once (W28 A9, g05-7; the rule in words: `docs/physics/response.md`, 'Two librations are not two
  resonances'). Open: 2001 HW7 5J+6S-2's slow component. Its x is the 2ω cycle at the package's cut, but a genuine
  second capture is not excluded on 20 kyr (x drifts +1.5°/kyr, not significant; ≥ 150 kyr would sweep ≥ 180°, L1).
  x also holds a 1.7–2-kyr component of unknown origin, seen in the slow a too (±5e-5 au). What would decide it is a
  direct three-body coefficient or a coupled averaged Hamiltonian (Codex): the owner-force residual bounds σ's own
  period only from below (1.7 kyr), which does not exclude a 7–8-kyr oscillation. Literature: the a-response
  criterion is NM98 §2 and Gallardo, Coito & Badano 2016 p. 89; no paper on the 2ω-modulated ϖ̇ beat of a
  three-body angle was found ('kinematic libration' is the package's term).
- 385266 16N-27 (W28 B1, Codex's question): the period of the joined forced run 56.6-98.7 kyr mixes the 4.4-kyr wave
  of the driver 3N-5 (5:3, circulating) and the ripple: after 68.4 kyr the ripple adds turning points (full cycles of
  0.8-3.6 kyr, median 1.7 kyr), so the episode's median period is not the wave's. Rule s joins it because one cycle
  of the leftover has the stall's period; the episode period printed for it is not a measurement of the wave.
- 3040 ZLK over 500 kyr of SABA at dt = 1 is chaotic in its start epoch (resonant, episodic or circulating within a few
  weeks); the slow test is pinned to 2026-06-10.
- 2024 WJ 1E-1: high, the author leans to medium; a "three-turning-point" rule was left for later.
- 1981 Midas ZLK (W28 D8, g09-7 C): ω librates 19 cycles over 500 kyr, but the free node is 1.85″/yr from s6 inside
  the 2.59″/yr width (`zlk_gate_node_near` s6): uncertain/gate_unavailable, extend to 1 Myr (L1, about 40 min of
  ias15). Open: is it a ν16 lock? Ω_free − λ_s6 in 50-kyr means runs 267, 255, 234, 205, 165, 100, 64, 53, 63, 96° —
  it turns back instead of drifting 257° (what −1.85″/yr gives), so a libration of the ν16 angle with a period
  > 500 kyr is not excluded, and then no length separates the node. `free_rho_i_max` is 1.4 there, not ≫ 1: while d
  is below the width the fitted s6 term (0.30) takes part of the free node (0.25), so the proposed rule "free
  inclination ≫ forced → own frequency" cannot be read from this record. The 1-Myr record (L1, 2026-11-01):
  uncertain/gate_unavailable, extend 4e6; d fell to 0.022″/yr against a resolution of 1.296″/yr, ρ_i,max 1.72, 38.4
  cycles: the free node sits on s6 (ν16), not a resolution artefact, and D8's separation estimate does not hold. A
  node rule, keeping uncertain, or review instead of the futile extend: d10 for the author.
