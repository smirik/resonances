# Decisions (index by topic)

An English map of `benchmark/DECISIONS.md` (the author's log, Russian, append-only) and of the decisions recorded in
the wave reports. Each line: date, the decision, why. "Superseded" marks an entry a later one replaced; the log keeps
both. "(author)" means the author decided; "(measured)" means the decision followed a measurement he accepted.
The exact current behavior is in the code and in `docs/physics/`; the rule IDs are indexed in `docs/libration-analysis.md` and are explained in
`resonances/resonance/analysis/mmr/CLAUDE.md`.

## Status model

- **Oct 13 (author).** `status` ∈ resonant, transient, near_separatrix, non_resonant, uncertain, invalid. `invalid`
  replaces `chaotic`: classification impossible with these parameters (alias, unphysical orbit, undefined phase), the
  cause in `reason`. transient = a resonant object, if only for a while; near_separatrix = not a resonance but
  resembling one, a status of its own.
- **Oct 13 (author).** Separate columns: `angle_status` (geometry only), `confidence` (certain / high / medium / low;
  empty for uncertain/invalid), `checks` per resonance type (pass / fail / not_measured / not_applicable), `reason`,
  `action` (none, spot_check, review, extend, resample). `not_measured` is never `fail`. `slow_circulation` dropped:
  ≥ 1 turn is circulation, < 1 is insufficient_time.
- **Oct 13 (author).** Everything can be recomputed from the saved CSV/JSON without integrating (classification and
  figures). Integer statuses are gone; `status_code` is a comparison column only.
- **Oct 13 (author).** Kinematic libration (the angle bounded by geometry, ϖ̇ or a forced oscillation of a, without
  dynamics) is the same for every type: angle libration, `non_resonant`, `kinematic_libration`; ZLK −2 belongs here.
  A drift with a proven kinematic mechanism → non_resonant; without one → uncertain + extend.
- **Oct 13 (author).** A long libration whose three turning points do not fit the record → uncertain + extend. ZLK
  gate unavailable → uncertain (−9), `gate_unavailable`.
- **Oct 15 (author).** Origin of a libration is its own column `libration_origin` (dynamic, induced, subresonance,
  kinematic, unknown) with `owner` and `origin_detail`. A subresonance member found in a multiplet gets its own
  summary row and figure.
- **Oct 15 evening (author, after an independent review).** Status by observed behavior: accepted libration episodes
  → transient/resonant; near_separatrix only without episodes, with a responding a and energy at the separatrix. The
  separatrix physics is columns and flags and does not change the status (52356 → transient/medium plus a flag).
- **Oct 16 (author).** angle_status describes the whole record: accepted episodes followed by another regime, or a
  capture after circulation, make the angle word episodic; the ending regime goes into `final_regime`. Supersedes
  "82831 angle_status = circulation" of Oct 15.
- **Oct 17 (author).** angle_status `transient` renamed `episodic` (the angle word no longer equals the status word);
  the package no longer reads old files (v1.0); the benchmark tools do (`bench_io.read_summary`).
- **Oct 26 (author, W24 Q3).** Leaf 5u/6 (no force measured, origin unknown) writes `geometric_libration` /
  `geometric_episode` instead of `dynamic_*`; rows with a measured force keep `dynamic_*`. Old files: `dynamic_*`
  with origin unknown is read as `geometric_*` by the benchmark tools only (`bench_io.read_summary`).

## Geometry and episodes

- **Oct 13 (author).** An episode needs ≥ 3 turning points (not 4). "Libration over the whole interval" means the
  episode starts within one libration period of the start and ends within one of the end (not "90 %").
- **Oct 15 (author).** transient and near_separatrix are separated by physics (energy relative to the separatrix,
  regime change), never by a turning-point count. Every convention is a setting with its physical meaning documented.
- **Oct 15 (author).** A visible exit in the last period: libration, ≤ medium, flag `exit_suspected`. An arc > 2π is
  not part of a libration. a's response at amplitude < 45° is not_measured.
- **Oct 16 (W9).** Each regime between jumps of a is measured on its own samples; a capture after a jump (G1l) is
  transient with `final_regime` libration.
- **Oct 17 (author).** A closed cycle is mandatory for an episode (1864 3E-5S-5 is non_resonant). The scales of L4
  and R0 are physical, not tied to P_c.
- **Oct 18 (W18).** Moving center: a run rejected for center motion is tested in ψ = σ − c(t) (c the occupied minimum
  of the averaged potential); L4 in ψ, R0 in the full well. `separatrix_chaos` renamed
  `episodes_in_separatrix_zone` (the model zone does not measure chaos).
- **Oct 19 (author, q5).** A cycle closed only by phase return (2 TP) caps the level at medium (`one_return_cycle`).
- **Oct 20 (W20).** Edge rule E1 (variant a): an exit/entry is observed when the orbit-averaged raw angle runs ≥ 1 turn
  monotonically from the last TP. Regime cuts at a center step or stall (K3 stall cut left to the author, still open).
- **Oct 24 (author, D1).** The union rule extends from 1:1 (G2) to every MMR (G2u): episodes without circulation
  between them cover the record together; gaps ≤ 1 × (P_l + P_r). A G2u union read as episodic is promoted to
  resonant only from a dynamic transient (6u). The gap criterion is superseded Oct 31 (W28 g06-5 A).
- **Oct 24 (author, D14).** G4b: a slow center whose net shift is ≥ 2 turns is circulation.
- **Oct 25 (author).** TX9's episodes were found wrongly; episode-detection tests from the author's history were asked
  for → the episode gate (W23, 79 cases after E16 was removed on Oct 26).
- **Oct 26 (W23).** Rule (a): at the center scale (G2c) turning points are taken at the center-turn prominence p_c that
  qualified the scale. Rules (b) and (c) of the design were not shipped (see [rejected.md](rejected.md)).
- **Oct 27 (W24, author accepted 12:35).** G2a action step: implemented with test 7 (`regime_periods`) after review;
  TX9 fails test 7 and stays one episode, uncertain. The author accepted uncertain.
- **Oct 28 (W25, author approved both; Codex's conservative variant).** Separatrix zone: (1) above the lower saddle of
  its well a cycle's loop encloses several wells and has no single-well J/J_s (nan, topology `merged_wells`, class
  `not_measured`, never "in the zone"; J_loop and T_loop/2π reported); (2) the averaged route compares actions (J/J_s
  of each turning point's loop in the averaged profile) with J_lo, not h with an h_lo mapped through V_E. Measured on
  all 719 MMR examples (frozen trees, settings 59c557f7a08b): no status changes; 45 rows change flags, caps or confidence;
  medium → high for 2015 SO2, 2019 GM1, 2020 SB, 2024 TY11 (20 kyr), 2025 PN7, 439908 (20 kyr) (cap
  `episodes_in_separatrix_zone` gone), high → medium for 82831 13M-24 (both rows: its turning points' J/J_s
  0.106–0.119 ≥ J_lo 0.1 at h 0.105–0.124 below the old edge 0.125). The zone classes are not columns only: L2, Z, caps
  and leaf 5o read them.
- **Oct 28 (W25, Codex final review #3).** The averaged route compared J/J_s(t) (the profile's units at each time) with
  J_lo = J0/<J_s> (the record's mean separatrix action; `numeric_zone`: J/J_s(t0) = J0/√m0). Action at fixed h scales
  as √strength, so at m = 0.25 a local 0.6 is 0.3. The pre-W25 h-route had the same mismatch (local h against h(J_lo)).
  Now each ratio is carried by J_s(t)/<J_s> of the averaged well itself (absolute J_s = √(3k²μn²a) × the profile's,
  <J_s> its time mean on MMR_SEPARATRIX_AVGPOT_TP_MAX even times, the basin of the nearest turning point's swing
  middle); no mean (a failed profile) → the pendulum's classes. Not the zone model's √m: J_lo and the classes would
  then share a model the averaged route exists to replace (2004 TK14: depth ×100). Measured on 719 MMR examples
  against test-benchmark: no status or confidence changes; 23 rows change cycle classes, 4 change flags/thin margins:
  2000 SF8 8M-9 (3 cycles deep → zone, its episode in the well's strong phase, J_s(t)/<J_s> 1.6–3.05: L2 thin),
  2003 UG22 10E-17 and 2004 TK14 31M-34 0.5 yr (zone → deep: flag libration_in_separatrix_zone and thin L2 gone),
  87555 4N-5 (deep → zone: zone_edge_cycle gone); 3233 30M-53 loses the route (a failed profile on the grid). 82831
  13M-24 (both rows, J/<J_s> ≤ 0.47 ≥ J_lo 0.1) and the 1:1 rows (no averaged route) are unchanged.

- **Oct 31 (author, W28 g06-5 A).** Episodes join when σ stays inside the neighbors' libration band through the
  gap (bounded-gap criterion), replacing D1b's 1 × (P_l + P_r); 2001 WK15 and 2002 XA are expected resonant/medium.
  Implemented in A7: every gap sample within `LIBRATION_UNION_BAND_MARGIN_DEG` (10°, a convention) of one of three
  bands — the left neighbor's own c_l ± A_l at its end, the right one's c_r ± A_r at its start (facing full-cycle
  centers, A = half_amp_max), or the band carried linearly from c_l to c_r with ± max(A) (a step of the center).
  Claude's choice, not the WP's low-pass center track: the track follows an excursion slower than its 3-P window (a
  synthetic excursion to π + 200° over 2 P overshoots it by 31°, the bands by more than 100°). The carried band alone split
  the co-orbital 524522 1V-1 (σ swings back into the first band as the second episode starts: 21° beyond the carried
  band, inside the first; author 19.10 wants it joined); the two own bands alone split every step between separated
  bands (q-5 i: steps are annotations) — hence all three; the no-2π tests, the chain span and the edges unchanged;
  `LIBRATION_UNION_GAP_PERIODS` removed (no row needed it: TK14 ×2, TX9 ×2, 88710 9V-16, YG1 100 kyr unchanged).
  Applies to the 1:1 union as well (a glide between two centers joins however long; C7 re-reads the 1:1 boundaries).
- **Oct 31 (author, W28 g03-2 A, g03-8 A).** The MMR exit/entry test reads the local center at that edge, not the
  episode's mean (C9): the edge half-cycle's midpoint, or the moving well's c(t) at the edge turning point in ψ
  (`Episode.edge_centre_rad`, rule `local_centre_entry` when the mean reading differs). Claude's choice over the WP's
  first-full-cycle midpoint, measured: on 2016 JA 1E−1 (horseshoe half-swing about 182°, then tadpole about 282°) the
  full-cycle midpoint still put the first turning point 190° away and kept entry_suspected. Over the 719 MMR rows the
  verdict moves only on 2016 JA (medium → high, expected), 2017 XG62 1M−1 20 kyr and both 27011 4M−7 rows (medium →
  high: entry_suspected gone, the episode's center moves 34–47° from its start) and 2003 UG22 10E−17 (action
  spot_check → resample); 2004 PY27 2E−7 gains entry_suspected (σ runs 333° down to its first turning point, an arc
  0.92 turn: a near-full turn at the record's start), level unchanged. Secular and ZLK edges keep the mean center.
- **Oct 31 (author, W28 g11-4 A).** 2013 VA14 60 kyr: the stall remainder #4 and episode #5 may be two episodes (the
  center moved between them); supersedes the Oct 23 reading "one episode"; the label is unchanged.
- **Oct 31 (author, W28 g03-1 A, g03-9 A, q-5 i).** A 1:1 episode ends only at a separatrix crossing or an owner
  change; well and amplitude changes are marks (2021 XO3 one episode, E07–E09 one episode with a sub-regime),
  superseding the XO3 boundaries of Oct 23/30. For non-1:1 only center steps and stalls become marks, a single
  non-1:1 half-libration between circulations is uncertain, and the VA14, 82831 and TX9 decisions stay.
- **Nov 1 (W28 C16, q-5 i, g03-9, g11-4 A, g11-2 A; Claude's implementation of C15's physical reading).** A K3 center
  step or stall between two ACCEPTED pieces of an angle that is not 1:1 is a mark of one episode (`Episode.marks`
  `centre_step@t0:t1`, the step half-cycle; `stall@t0:t1`, the stall) when the last full cycle before the cut and
  the first after it share a band (m ± A overlap > 0) and a period (ratio ≤ LIBRATION_REGULAR_PERIOD_RATIO): σ stays
  in one libration region, the non-1:1 analogue of C7's same well; no threshold of its own (`cycles_continue`, rule
  `regime_cut_mark`, `LIBRATION_REGIME_CUT_MARKS`). The joined run keeps the band test's shape when it passes; when
  it fails only for its moving center and closes a cycle it is accepted as `regime_change`, L6 not measured (as
  `_accept_in_wells`); any other rejection keeps the cut. An accepted|rejected boundary always stays (385266). Why
  not "every step is a mark": that breaks E17 (VA14 60 kyr, facing bands −4.5° apart) and moves 2002 BG 20 kyr.
  Measured on one-example runs: 9 boundaries in 7 angles join (138537 ×2, 182259, 482824, 719 36J-13, 2002 BG 100 kyr
  ×2, 2004 YG1 100 kyr, 2012 DE61), no status, level or reason moves; 46596 (C15's prototype, a trial pass) is
  unchanged. The single half-libration (q-5 i, second part) was already leaf G4h of C6.
- **Nov 1 (W28 FIX-c16, Claude; F7 found it).** The facing-cycle test alone joined 55390 1J+3S-1 at its 8.3-kyr center
  scale across a "step" of a steady drift (centers 114 → −185° over 75 kyr, half-amplitudes 40–90°; P 16 kyr at a
  secular period, identity failing): G2c then made a non_resonant/high row uncertain + extend 200 kyr. Bound added:
  the turning points of the joined stretch, through the step or the stall, span less than a turn
  (`within_one_turn`; a libration does not take every value; no threshold, 2π is the circle). 55390 returns to
  non_resonant/circulation/high (span 455°); 482824 19N-32 loses its join (572°; back to its two W27 episodes, no
  status move); the other 8 joins span 3.3–159° and stay. Rejected: the facing bound |Δm| ≤ max(A_l, A_r) (the
  orchestrator's first spec). Physics does not require it (a sudden shift of the well keeps the swing through the
  point where σ was, i.e. band overlap, not |Δm| ≤ A), and it does not separate: 55390 is at |Δm|/A = 1.26 while
  the intended joins are at 0.91–1.9 (182259 1.13, 719 1.35, 2002 BG 1.45/1.3, 2012 DE61 1.84, 138537 1.9/1.1), so
  it would have undone 7 of the 9.
- **Oct 31 (author, W28 g11-6 B, g08-5 B).** Rule (b) ("the regular core must pass on its own") deferred with a return
  condition (candidate identity); 2003 TX9 stays uncertain with action extend to ~30 kyr.

## Response, force and strength (MMR)

- **Nov 5 (author: "4–5 librations are enough to test the alternatives", 385458).** Owner search stage 3
  (`owner.fitted_window_settings`): when even the 2-P windows do not fit, the measured trim (the band-pass edge error
  reaches its interior level 0.75 P from an end, `MMR_OWNER_FITTED_EDGE_TRIM` 0.375 fP) and two windows fitted to the
  span, 1.5–2 P each (`MMR_OWNER_FITTED_MIN_CYCLES`, flag `owner_fitted_windows`; ~3.75 P of record). Rejected: the
  0.75-P trim for every row (HW7 150 kyr's owner regimes re-split, cover 1 → 0, non_resonant → transient; 1685 the
  same; TK14 0.5 yr certain → high); a sliding grid from the start (385458's second window missed by 0.3 kyr). v2-w35-fit:
  no status moves, 14 rows high → certain; 137295 against the author's high (listed, asked).
- **Oct 13 (author).** Gallardo's resonance strength is not used for statuses. Confirmed by measurement on Oct 26
  (W24): his normalization is arbitrary for this purpose.
- **Oct 16 (W9).** L4 (force law ȧ = K sin(σ − φ₀)) decided by R² without a φ₀ scatter test; 7343 1M+1J-2 resonant/high.
- **Oct 16 (W9).** Strength F counts only swings inside the well of ⟨R⟩ around the theoretical center.
- **Oct 16 (W10/W11).** Rule R: the response is measured on every librating episode and a witness is chosen; a
  measured L3 failure disqualifies a witness, not_measured does not. The tree and the text read the same response.
- **Oct 17 (W12).** `MMR_RESPONSE_CUTOFF_CYCLES` 0.05 → 0.1 (derivative noise averaged on a physical cycle scale).
- **Oct 17 (author).** The owner search tests a candidate by the multiplet law (coherent coefficients of
  φ_o + j(ϖ − ϖ_P)); L4 contaminated by a forced line of the angle is not_measured and does not lead to dynamic.
- **Oct 17 (author).** The "L4 and R0 not measured" branch splits by physics: support / measured objection / nothing.
- **Oct 20 (W20, author Oct 19).** F6 forced waves; K2 kinematic drift of the slow center; F8 baseline N from the
  deciding scale; subresonance members σ ± 2mω in one invariable frame; `certain` for far circulations.
- **Oct 23 (independent reviews).** A failed O1 = uncertain (a real pendulum capture gives O1 ≈ 0.5); the earlier
  recommendation to the contrary was wrong. Reformulated Oct 31 (W28 g03-4 A).
- **Oct 24 (author, D3–D15).** R-C4 (C4 not_measured when no force law explains ȧ), R-WJ (1:1 on one cycle with a
  failed energy R² ≤ medium), R-L3K/R-FK (L3 n/a and F read differently for kinematic verdicts), R-O1-OWNER,
  R-KDRIFT (K2b), R-ALIAS-A, owner route 'identity', R-DIV (B ≤ n·min(P, P_cyc)/2π), R-F (5u/6 cap
  `strength_scale_only` ≤ medium when the e^q scale passes).
- **Oct 24 (author, D2).** R-IS and R-SP-IS (lift caps for an implausible strength) dropped after he changed the
  2001 PG14 label.
- **Oct 26 (W23).** CS12 multi-harmonic force law: lifts L4, never lowers it; never for 1:1 or a null test.

- **Oct 31 (author, W28 g03-4 A, g08-3 A).** O1 stays a measured check but objects only to a coherent foreign a-wave:
  coherent at P, continuing in phase and not explained by the episode's own force law (one rule for 1:1 and other
  MMRs); the D4 labels of Oct 24 (TG2 100 kyr, VA14 ×2) must not move.
- **Oct 31 (author, W28 g04-0 A, g04-2 A, g04-3 A).** Owner (F7): the owner is credited only with its measured
  regimes, cover is checked in every accepted episode (0.5), the slip gap is not credited, P2 is a guard; with one
  induced and one own episode, the own episodes decide.
- **Nov 1 (W28 FIX-owner-edge, Claude; tfix-bench found it).** F7 at the record edges: the owner's windows lie inside
  the band-pass trim (filter_edge_trim_cutoffs × f P from either end) and stop a step short of its end, so an
  episode there was read as "owner absent" (2001 HW7 without planets: episode 8 at 19.5–19.8 kyr got cover 0, became
  the row's own and made the mixed path: transient/geometric_episode/low instead of induced). Now the cover lies in
  [c, c + u], u = the part outside the hull of the frame's force windows (`owner.measured_span`). Induced when
  c ≥ cover_min, own when c + u < cover_min, else not measured (nan, in neither list of `induced_split`). No new
  setting: nothing is assumed about the edge zone. Covers are rounded to 9 decimals (HW7 episode 0 was
  0.49999999999796 for an exact half). `own_cover` (the decisive episode) keeps c: a nan there would hit
  `decide.owner_in_episode`'s early return and hand the owner the row. A row whose every episode is not measured
  therefore falls back to c (owner refused). Two variants were rejected. (1) Extending the intervals half a window
  past the first and last window centers: the intervals are already the full windows, the unmeasured stretch is the
  trim, and extending into it would credit the owner with time it was never measured on. (2) The hull of the
  candidate's own fitted windows: a window it turns too fast to be fitted in (`resolved_windows`) measures its
  absence. That variant made 1998 UM1, 2001 ND13 and 1862 covers not measured; their verdicts did not change.
  Moves on the benchmark: none (67 owner rows before/after, only the rounding).
- **Oct 31 (author, W28 g05-3 B, g05-4 A).** 285637 3M+7J-6: label medium → high (kinematic libration: K1, B1 < 0.01,
  period equal to the secular one); 481085 7E-10M-6 non_resonant/libration, kinematic. Claude's reading: the term
  (−4ϖ, 9ϖ) goes into `cause_role`/`cause_term` and the figure's owner slot, not the `owner` column.
- **Oct 31 (author, W28 g08-8 A).** 37087 3J+8S-3: reason kinematic_libration (secular cycle); status and level of the
  Oct 23 label stay.
- **Oct 31 (author, W28 q-2 b).** Confidence ceiling: high only when at least one physical support (L4, R0, F, the
  1:1 law) passes with a full margin, else medium. Claude's reading: it replaces the "low" of g03-3 C for 2020 CB.
- **Oct 31 (author, W28 g06-8c A; g06-1 A, g06-2 A).** The subresonance is chosen physically, no scalar score: every
  veto (E′, L, A, B, E) before the choice, one window per episode, the theory's flattest center c_j(t) for two-body
  non-crossing rows (the measured center must follow it, W breaks ties and is the fallback), 'composite' allowed. One
  resonance per (body, commensurability); the member is the row's `resonant_argument`. Claude's reading (A6b,
  Nov 1): 'composite' is not Rouché's failure (it fails on all six accepted rows, whose member the author named) nor
  "the member's center travels more than the half-amplitude" (that flags clean 3J-2/3J-1 rows) nor "the flattest
  moves by half of β's range" (2003 SL215 8J-3); it is "the two flattest theoretical centers tie" (equal weights),
  with the members told apart (range β > MMR_SUBRESONANCE_CENTRE_MOVE_RATIO × half-amplitude) and nothing
  admissible. An episode whose own window cannot decide caps the row (`subresonance_undetermined`) as the old
  single window did (78085 5J-4S-1 keeps medium, the author's s-1 control).

## Co-orbitals (1:1)

- **Oct 16 (W4).** Judged by the averaged direct + indirect force law with the planet's own mass (K/K_th = 1), no owner
  search; W5 added the retrograde 1:1 law.
- **Oct 19 (author, q2).** No `transient_unexplained` cap for 1:1 (a co-orbital capture is a close encounter; the
  pendulum separatrix model does not apply).
- **Oct 20 (W20 CS5).** Quasi-satellite pass G1q: a second pass at the period floor and cutoff taken from the averaged
  potential.
- **Oct 19 (author, q3) → superseded Oct 27.** 2003 TX9 3E+2J-10 labeled resonant/medium (double well, capture into
  the second well after Mars encounters); after CS12 and G2a measured no per-regime physics, the author accepted
  uncertain.
- **Oct 17 (author).** "All Jupiter Trojans uncertain" was a bug; fixed in W16.

- **Oct 31 (author, W28 q-3 A).** The co-orbital check F2 (2026 AC4, g01-6) is replaced by a period floor from the
  resonance's own averaged potential, for all resonances; AC4's outcome follows the measurement.
- **Nov 1 (W28 C3, cards g02-1 A, g01-6 A, g01-8 A, q-3 A; Claude's implementation).** F2 of g01-6 ("a crosses a_P
  at each turn") is not implemented: by the a<->sigma identity it holds for every turn of sigma, so it tests nothing.
  It is replaced by the floor of the resonance's own averaged potential, per run: a 1:1 run whose turns (read on the
  orbit-averaged angle: the filter rounds a planet turn off, YN107 39/319 deg -> 2.0/358.3 deg) all reflect on a top
  has the floor 0.5 x the loop through its turns, if its local law has K/K_th > 0; other runs keep the tadpole floor,
  the QS well G1q's (rule `coorbital_floor_profile`). The fast-residual reference became min(tadpole floor, 0.5 x the
  fastest horseshoe loop turning 3 r_H out), not the WP's T_hs(3 r_H) alone: measured, T_hs(3 r_H) is above the
  tadpole floor on every profile tried (Earth 113 vs 111 yr, Jupiter 235 vs 74), so it would have raised every 1:1
  prominence; the reference is the shortest floor any run can get. The own floor is above the tadpole floor on 53 of
  the 102 reflection runs of the 259 co-orbitals and rejects none of them. Moves (259 co-orbitals, profile floor on vs
  off): 2003 YN107 non_resonant/medium -> transient/episodic/medium (author label); 2026 AC4 non_resonant/high ->
  uncertain/episodic/origin_unknown (the 274-369-yr cycle appears, period 94 yr = 0.90 of its 105-yr loop, K/K_th 1.4;
  the force law is unmeasured under filter_too_slow at P_c 55.5 yr; q-3 A: by measurement, no label); 2023 RX1
  high -> medium (near_episode). The near-episode now needs floor x (1 - CONFIDENCE_THIN_BAND) for every MMR (g01-8
  A): on the 29.10 run this adds 1991 VG and 2020 HO8 20 kyr (high -> medium) and no non-1:1 row. The cutoff is
  unchanged.
- **Nov 1 (W28 C7, cards g03-1 A, g03-9 A, g11-5; Claude's implementation).** A 1:1 episode ends only at a
  separatrix (rule `coorbital_well_cut`, `MMR_COORBITAL_WELL_EPISODES`); changes of well and amplitude are
  `episodes.csv` marks (`well_change@t`, `amplitude_change@t0:t1`, `kick@t`) with the wells sequence (`wells`). Measured
  choices: (1) a run the band test rejects is accepted when every full cycle's swing holds a minimum of the averaged
  potential to within the run's turning-point prominence: the strict test (W2) fails XO3 (transitional cycles 4-8 deg
  off at 11 deg) and the center-range-only test (W0, the WP's "no monotone pass") accepts 2022 BB2, a ripple drifting
  past L5 7-25 deg off at 4 deg, which the author keeps non_resonant (g01-7 A); the center bound is one turn (the
  bound `librating_episodes_with` already applies), not LIBRATION_CIRCULATION_NET_REV = 2 rev of the WP; (2) only
  accepted neighbors join (joining a rejected neighbor extended TY11 20 kyr to 420 yr, 277617 20 kyr, SZ2, PN, BT8,
  AP1: C8's residence, not a measured libration); (3) an amplitude change reuses G2a's amplitude tests on two
  half-cycles each side plus a same-well and an energy-level check, an annotation only; (4) a run of several wells has
  shape regime_change, the floor of the fastest well it holds (`FloorWell.holds`), G1q keeps the outside episodes it
  contains and L5 is not applicable: without them 164207 and 469219 fell to transient/medium. Sweep of the 259
  co-orbitals (scratch impl-C7/sweep1.csv), status/level moves, none with an author label: 2009 BE81 20 kyr
  non_resonant/medium -> transient/episodic/medium (one episode 7.75-16.7 kyr; C8 refines it to two cores, g02-3 A
  low); 2026 SZ2 transient/high -> resonant/libration/medium (the slow first half-cycle the regular-part split cut off is
  part of the episode: 121-897 yr, cap force_law_disagrees); 2015 YA transient high -> medium (cap episodes_in_separatrix_zone);
  2024 JR16 uncertain/drift -> uncertain/libration (review; a tadpole whose amplitude changes). Episodes only: XO3,
  524522, 164207, 469219 one each (status unchanged, E07-E09 and E43 rewritten), 2025 RB7 3 kyr, 524522 20 kyr. No
  non-1:1 row is touched (the model is None there).
- **Nov 1 (W28 C8, cards g03-3 C, g03-8 A, g03-9 A, g02-3 A; Claude's implementation).** A 1:1 episode stores its
  core (the measured TP span; coverage, edges, margins and status read only it), its sub-regimes (the core cut at the
  C7 marks, each with its own 1:1 law when there are two or more) and an inferred residence (rule
  `coorbital_residence`, `MMR_COORBITAL_RESIDENCE`). Measured choices: (1) the residence is bounded only by a leg of the
  3-orbit averaged angle (reversals of 20 deg, the C1 legs) that sweeps `LIBRATION_EDGE_ARC_REV` = 1 turn, or by the
  record's edge; the WP's |a - a_P| <= da_max bound is reported only (`residence_zone_max`): on local profiles the
  accepted core of 2024 TY11 20 kyr starts at a turn of zone 2.2 (its law passes there), so as a bound it would cut the
  residence g03-8 A asks for; (2) the 2-rev `LIBRATION_CIRCULATION_NET_REV` of the WP is replaced by the edge rule's one
  turn (an arc above 2 pi is no piece of libration; consistent with exit_observed and C7's joins); (3) the G1q visit
  is not reused: it is the quasi-satellite well only. Sweep of the 259 co-orbitals with the switch on and off
  (scratch impl-C8/sweep.csv): no verdict field and no episode differs; 128 rows get a residence, 74 reach the record's
  end. 2020 CB: core 443-601, residence 443-1000 (edge: 601 is no exit); TY11 20 kyr: residence 0-8.69 kyr; JA: core
  128-537 (horseshoe 128-339, tadpole L4 339-537), residence 0-1000 (inferred); 2009 BE81 20 kyr: one episode
  7.75-16.7 kyr, residence = core (legs of 1.7 and 1.3 turns on both sides), sub-regime laws (force/energy) tadpole L5
  pass/pass, horseshoe pass/fail (energy R² 0.02), tadpole L4 pass/pass, compound pass/pass: the author's low (g02-3
  A, given for an unverified continuation) is not reproduced by a physical criterion; medium stays, the label is in
  KNOWN_DISAGREEMENTS for F9. The 1:1 orbit type now reads the sub-regimes' wells (summary `geo_orbit_type_sequence`;
  more than 4 shortened in the header): text only, e.g. JA 'compound: horseshoe → tadpole L4', AA29 'compound: horseshoe →
  compound → horseshoe', 522684 'horseshoe' → 'tadpole L4'. Cost: +0.04-0.2 s CPU on most records, +1.2 s on 524522
  20 kyr (15 sub-regime laws).
- **Oct 31 (author, W28 q-4 A, g02-5 B, g02-6 B).** One default rule for a single co-orbital leg (clean leg + local
  1:1 force + energy + conservative turns → transient; one mismatch → uncertain/episodic; cut by the edge →
  uncertain + extend; encounters → non_resonant/medium with a near-episode); where it disagrees with an earlier
  per-object answer, the row is shown in the final review folder.
- **Nov 1 (W28 C6, cards g01-1/4/5/8, g02-4..g02-7, q-4 A, q-5 i; Claude's implementation).** Leaf G4c: a 1:1
  circulation without an episode is judged on the legs of its 3-orbit averaged angle outside every accepted run
  (C1's measurement and thresholds: W2 = leg +- 0.25 L at max(2 P_orb, 0.1 x 2 L), out/in band ln 1.4, zone 1.1).
  Choices: (1) the default is q-4 A literally, a failed 1:1 law is one discrepancy (uncertain); C1's force gate is
  `MMR_COORBITAL_LEG_FORCE_GATE` (off; it moves 138175, 2015 WZ12, 2019 JX1, 2020 CL1, 2021 VH2, 2022 TD); (2) the
  strongest outcome of the legs decides (C1's rule 6), not the longest leg of the WP text; (3) a leg overlapping an
  accepted run is never read (C1's rule 7, extended to an accepted run below the floor: 2022 AP1 stays non_resonant,
  where C1 read uncertain); (4) the scattering outcome keeps the circulation leaf and adds the near-episode cap; (5) the
  structural near-episode adds 1:1 over_the_top 2-TP runs whose half-swings cross at most one of 0 and pi (g01-8 A).
  Leaf G4h (q-5 i): a 2-TP half-swing of another MMR between passages becomes uncertain only with its own one-harmonic
  law (C15's test, R^2 >= MMR_FORCE_R2_FAIL per half, restoring): of C15's 19 candidate records only 42946 43S-6
  moves (non_resonant/high -> uncertain/insufficient_time, extend); without the law 18 slow circulation turns would
  move. Moves on the 259 co-orbitals (scratch impl-C6, base.csv -> new2.csv): to transient OE5, XP16, YU11 (medium,
  cap one_coorbital_leg); to uncertain/one_coorbital_leg 138175, VU6, RZ53, WZ12, YQ1, KS, SS2, XH2, VF, VH2, BB2,
  BY39, RA, TD, AV2, DU7, FP4, GL, SC; to uncertain/extend TG24, JX1, CL1, YF4; high -> medium (near-episode) RQ2
  20 kyr, KS3, 613440 20 kyr (scattering legs) and HM4 (structural over_the_top run). Unchanged: VG, HO8 20 kyr,
  RX1, FN3, 255071 (already medium, near-episode), EZ84 and every negative control (JE1, QS7, GP2, JL1, XC352), CA138,
  XW2, WJ, AP1, 85770, XE133 20 kyr, 872436 20 kyr. Against the author's per-object answers (the rule decides, F9
  shows them on the W28 review page, d06/d07; 17 of 36 differ): the g01-5 rows read
  uncertain where he expected non_resonant/medium (the same legs as his g02-5/g02-7 uncertain rows), FN3 and 255071
  stay non_resonant (scattering), AV2 uncertain (energy fails, entry kick), XP16 transient, WZ12 uncertain (its law
  fails). Cost: +18 % CPU over the 259 records (330 -> 389 s), mostly the local laws of clean legs.

## Sampling and aliasing

- **Oct 13 (author).** An alias whose expected rate from mean a is unambiguously far from the resonance →
  non_resonant, `alias`; otherwise invalid/alias + resample. The finder warns before integrating.
- **Oct 15 (author).** `OUTPUT_STEP=auto` is an option; the dry-run always warns.
- **Oct 17 (W15).** The alias gate reads the λ part of the angle, not the full angle across low-e perihelion flips.
- **Oct 20 (W20 round D).** `short_period_aliased` stays on every record whose step folds short periods; the record
  never lifts it (`MMR_SHORT_PERIOD_CAP_RATIO` = 0). Measured on 47 fine re-integrations: folded lines are negligible
  in σ and matter through a.
- **Oct 23 (author, Q1d).** No cap on a geometric circulation without forced-wave runs. The cap is made actionable:
  `smp_fold_free_step_yr` and action `resample`.
- **Oct 24 (author, D11).** R-ALIAS-A: on a folding step with F direct passing and a's short-period scatter above
  `MMR_SHORT_PERIOD_A_RATIO`, a failed L4/L3 is not_measured.

## Confidence

- **Oct 13 (author).** certain = never needs a manual check; high = all checks pass, at least one thin, ≥ 97–98 %;
  medium ≥ 90 % (spot check); low = no promise.
- **Oct 15 (author).** high allows any number of thin margins and no failure; nothing physical measured → ≤ medium.
- **Oct 15 → superseded Oct 16.** "certain/high need ≥ 100 labels per cell": dropped; the level comes from margins and
  caps and `certain` is shown; calibration cells are reported only (`cell_*`, `resonances/data/confidence_cells.json`).
- **Oct 19 (author, q1).** Every cap down to medium or low names the other reading and what decides it
  (`verdict_text.CAP_ALTERNATIVES`).

## Secular and ZLK

- **August (measured).** The free-omega gate decides the ZLK status; e–i coherence is a flag. −2 (measured kinematic)
  and −9 (no verdict) are never merged. Calibrated at 500 kyr.
- **Oct 12.** Secular angles from the planets' mode longitudes and the invariable plane, reduced by default.
- **Oct 13.** The secular study uses the 21 literature resonances; no broad order-≤4 scans for status (the finder
  still scans the 75 catalog formulas by divisor).
- **Oct 16 (W4).** Secular/ZLK moved to analysis v2; the secular cutoff includes the angle's longest forced wave;
  secular e/i response and e–i coherence are diagnostics only. The secular status is decided in
  `analysis.secular.decide.decide_status` (no hook after the analysis).
- **Oct 17 (author).** 162474 over 1.5 Myr is an ordinary transient (island 270° → 90°); no `island_switch` flag.
- **Oct 27 (author, W24 Q5).** The secular/ZLK author verdicts are gated in `make test-benchmark` (14/14); 189865 and
  477492 lose the "invalid" expectation (it came only from SABA failing on planet-crossers).
- **Oct 28 (author, W25 items 7, 8).** An uncertain secular/ZLK verdict names its next check (`analysis/secular/
  extend.py`, MMR's `next_check` with N by cause: baseline 1.5 P + 2 P_c; gate short_window FREE_GATE_MIN_CYCLES
  P_cyc; planetary_residual 1.25 × 1296000/d_min, review beyond 100 Myr); undefined_phase, unavailable and
  unusable_series now ask for review instead of extend (no length cures them). No status changes. The dry-run has
  advisory secular/ZLK rows (baseline 500 kyr / 9.4 kyr, the ZLK gate's 3 quadrupole times). Panel (f) draws the
  analysis' own response fit (it refitted at the fine P_c while printing the decisive scale's R²).
- **Oct 28 (author, W25 items 3, 4, 5, 10 and the ZLK double-scale studies).** ZLK ρ_max is local: max over the
  record of rollmed(|z_free|, W)/rollmed(|z_forced|, W), W = max(beat window, `ZLK_RHO_WINDOW_CYCLES` P_cyc); the old
  median ratio stays a column (`zlk_gate_rho_global`). Co-orbital guard, variant (a): only for planets whose modes are
  not fitted (Mercury to Mars), outcome unavailable (`coorbital_unfitted_forcing`), never not_applicable; Trojans
  covered. c1, c2, h_quad in the invariable plane. e–i coupling at the decisive period for every ZLK row
  (`zlk_ei_coupling`: zlk / forced_beat / none / not_measured; `confirmed_ei` = zlk). Attribution of a kinematic
  libration (`zlk_forcing_*`), the center's slow line (`zlk_centre_*`, 591986: g6−g5), and a scale guard: a
  center-scale libration at a forced beat with ρ_max < 1 never decides. n_cycles is T/P_cyc of the osculating ω (the
  figure said "free ω"). No status changed on the benchmark. Pending (author): a node-lock flag (Ω − ϖ5, ≥ 3 Myr).
- W26 (author, 2026-10-29: "state the presumed cause of an induced ZLK libration; overlay g5/g6 in one more panel only
  for such rows"): the cause sentence (`verdict_text.zlk_cause`, measured columns only), panel (a′) (ω_f, arg z_forced
  − Ω, ϖ_j − Ω; drawn only for an attributed row or a forced-beat center, others keep seven panels), the flag
  `zlk_node_lock` (approved 2026-10-28; resolved only with a full cycle of Ω − ϖ_j, `ZLK_NODE_LOCK_MIN_CYCLES`),
  the center origin `forced_direction`, and the attribution of an episodic row inside its episodes (scope
  `episodes`, 624). Columns only: no status changes.

- **Oct 31 (author, W28 g09-6 B).** An episodic ZLK row of a co-orbital with unfitted modes
  (`coorbital_unfitted_forcing`) → uncertain −9 (2020 PP1, 10563 500 kyr); supersedes Claude's W25 note "2020 PP1
  stays transient".
- **Oct 31 (author, W28 q-1 E).** An episodic ZLK row without a gate verdict is transient only when the e–i class at
  the decisive period is zlk, else uncertain −9: for these rows e–i becomes a status condition (supersedes "e–i
  changes no status"; rows with a gate verdict keep it). Per-episode classes are rechecked with Codex first (D9).
- **Oct 31 (author, W28 g09-4 B).** 2306 g−g6+s−s6: angle insufficient_time instead of drift when the center turned
  back; the uncertain label stays.
- **Nov 7 (W36 F1, author's items 1-2).** The edge rule E1 applies to secular and ZLK angles on σ_f itself (no orbit to
  average, the episode's mean center): ≥ `LIBRATION_EDGE_ARC_REV` turns monotone after the last TP (before the first)
  is an observed exit (entry), the edge is not covered, the angle is episodic, `final_regime` circulation, no center
  scale behind it. Overrides the Oct 15 "visible exit → libration ≤ medium" for arcs ≥ 1 turn (below a turn it still
  holds). 2012 AX10, 33325 2s−s4−s6: resonant → transient.
- **Nov 7 (W36 F2, author's items 4, 9).** Secular and ZLK turning points: each run of provisional turning points (at
  the type's floor) takes the prominence measured on its own samples, as the MMR does (`preprocess.run_prominence`,
  `geometry.local_thresholds`), capped at the record's: a circulation elsewhere must not set the threshold of a quiet
  libration (19289 76.8° → 31.7°, 278067 83.0° → 23.3°: one episode each, 5.1/6.1 Myr to the end). The cap because a
  run's fast part keeps the libration's own swing where interleaved waves hide its period from the image (2020 PP1
  ZLK 101° vs 52°). Settings `SECULAR_TP_LOCAL_PROMINENCE`, `ZLK_TP_LOCAL_PROMINENCE` (on).
- **Nov 7 (W36 F2 guard, Claude; regression found on pairs wave 1).** The local prominence may add libration, never
  remove it: every librating episode of the record's reading must lie inside one librating episode of the local
  reading, else the stretches it overlaps keep the record's prominence (repeated until none is lost;
  `secular.analyse.measure_fine`). 455327 g−s−g5+s6 (ias15): the run's 13.8° (record 39.0°) took a 32° wiggle of the
  drifting approach, the run became 9.14–12.47 Myr centre_moves and the crescent episode 10.23–12.47 Myr was lost
  (non_resonant); with the guard transient again. No setting (a guarantee, not a threshold). Benchmark: no row changes;
  example `secular__455327__g-s-g5+s6` (c0808, not an author label), case C61 (tier C, Codex).
- **Nov 7 (W36 F3, author's item 5: 42946).** A secular/ZLK run rejected only for its moving centre is accepted as
  `modulated` (stationarity not measured) when it closes a cycle, its TPs span < a turn, its cycle centre turns back
  at least once (instead of `LIBRATION_CENTRE_RETURNS_MODULATED` = 3: a secular libration makes few cycles in a record)
  and one of its cycles swings over every centre (`geometry.accept_turning_centre`; `SECULAR_TURNING_CENTRE`,
  `ZLK_TURNING_CENTRE`, on). Without the last condition 363, 2306 (4–6° waves at P ≈ 210 kyr on a centre travelling
  42–45°) and 518 went uncertain → transient/certain, against the author's 363 label. 42946: the run 1.3–14.6 Myr is the
  early episode; transient stays, confidence high → certain (N 2.0 → 8.8 cycles of the decisive episode).
- **Nov 7 (W36 F4, author's items 6, 7, 8, 10).** A secular episode that is a wave on a slower motion of the angle is no
  episode (`secular.decide.slow_waves`, `Geometry.slow_wave_ids`, reason `slow_wave`; `SECULAR_SLOW_WAVES`): on a slow
  swing (σ_f between the centre turns around it < a turn) when A_max ≤ 0.5 × R/2 (`SECULAR_SLOW_SWING_RIPPLE_RATIO`, the
  MMR W34 ratio); inside a circulation when the slow track beside it moves ≥ A_max in as long as the episode and keeps
  ≥ 0.5 of that rate inside (`SECULAR_STALL_RATE_RATIO`: a capture stops the circulation, a stall slows it). The slow
  scale gets the MMR second scale, the centre's own turns (`SECULAR_CENTRE_TURNS_SCALE`). 101703 → resonant/medium +
  extend, 348018 → non_resonant, 56700 → uncertain + extend; 142694, 165548 unchanged (see benchmark/DECISIONS.md).
  Secular only: on ZLK the swing ratio also caught 591986 (4 Myr) and 162474 (1.5 Myr) librations.
- **Nov 7 (W36 F4a, Codex review P1).** The slow-swing ripple must be carried by the swing: the slow track S (as
  `stall_reading`) moves inside the episode at >= 1/16 of its fastest rate beside it (`SECULAR_SLOW_SWING_CARRY_RATIO`,
  a convention: half of the 1/8 that a parabolic turn centred on the episode gives). Codex's two stationary
  librations joined by a monotone transition (0.013-0.037) stay episodes; the waves the rule is for read 0.12-0.57
  (101703 0.16/0.12, 518 0.12-0.22, 56700 0.51). No benchmark verdict changed. Thin for a very slow transition (a
  10-Myr variant: up to 0.078 at the centre scale). Also `SECULAR_STALL_ARM_MIN_FRACTION` (0.25, was hard-coded) and
  the secular figure accepts an explicit MMR layout (`long_period_layout(base=...)`).
- **Nov 7 (W36 F5, Claude's finding: 143662 s+g5−g8−s3).** The policy turns a secular/ZLK insufficient_time angle into
  non_resonant when σ_f's travel to the end is ≥ 1 turn (`_insufficient`); 143662 got high from 1.09 turns while the
  figure's only decisive text said "Not yet". The travel is a range (the swing 2A in it), so within
  `CONFIDENCE_THIN_BAND` of one turn the level is at most medium (`policy._travel_thin`, input `end_travel`), like a
  thin veto; the status stays (principle 5: a record that ends in circulation is decided now). The figure prints the
  policy's step P (`Is the record long enough?`, MMR's question) with the travel; step 0 keeps "Not yet" and says the
  policy decides at P. No benchmark row changes (no secular/ZLK row is decided this way there).
- **Nov 7 (W36, the author's items 2 and 4: periods and a stronger filter).** No per-body libration period is predicted
  and the angle is not filtered harder everywhere: P₀ = 2π/sqrt(|KA|) needs the torsion K and the resonant coefficient
  A, which one body's proper elements do not give and the package does not measure; no general lower bound on the
  period exists (the ν6 librators repeat at the free/forced beat, 0.24–0.57 Myr). Slower scales are tried only where
  the record shows a slow motion (center scale, center-turns scale, F4's slow waves). The references (Codex,
  page-checked): `benchmark/DECISIONS.md` 2026-11-07 W36; the summary in `docs/physics/secular-zlk.md`.

## Figures

- **Nov 5 (author).** The right column has one place per block on every figure: header → Checks in the left half,
  Evidence at the top of the right half, then Caveats, Numbers, footer (`verdict_column_draw.SECOND_SUBCOLUMN`); Numbers
  groups in one order for every family; the time axis labeled under (b) too. 'a-scales' everywhere ('a-ranges' gone);
  ratio bands at the Numbers' rounding. American English in every printed text and in the docs; identifiers and CSV
  columns keep their names (`snake_words` turns them into American words). The independent text review (124
  findings) was applied file by file: failed checks are named by what was measured, the Why states each ruled-out
  alternative with its number.
- **Oct 15 (author).** One combined figure per row (layout V1, "everything on one screen"); every decision number on
  the right, also in the CSV.
- **Oct 16–17 (author).** σ̇ axis symlog σ̇/2ω₀ with the zero line highlighted; the plaque reads
  `STATUS · angle_status · CONFIDENCE`, "ends: …" last.
- **Oct 17 (author).** 33128 6N-7 kept as the example where FAIR fails.
- **Oct 23 (author).** a_res on top of panel (d); angle labels in [0°, 360°); header `episodes about X° (not this
  resonance)` for induced/kinematic rows.
- **Oct 24–25 (author).** a_res style B (width 1.3, alpha 0.8, halo); a* alpha 0.8 (`AXIS_LEVEL_ALPHA`).
- **Oct 27 (author, W24 Q6/Q8).** `PLOTS=combined` by default; output layout `confidence/status` by default; a command
  to regenerate the documentation figures is required.

- **Oct 31 (author, W28 g10-1b A, g10-1a A).** PLOT and SAVE default to `review` (queue in the CSV, count in the
  log), replacing nonzero; doubtful non_resonant rows keep a physical status. Implemented (E1): `review` = nonzero ∪
  `verdict.needs_review` (level medium/low, or action spot_check/review, or origin unknown); the level decides on its
  own, so a medium row whose action became `resample` stays in it (measured: by the action alone the benchmark gives
  539 rows, 1036 9J-2S-3 missing; by the level 540, the card's count). Columns `review_required`, `review_reasons`;
  save, release and plot share `DataManager._selection_inputs`; the counts are logged after the run, not before the
  figures (a body is analyzed and drawn before the next one).
- **Oct 31 (author, W28 right column: g10-2 A, g10-3 C, g10-4/4b A, g10-5 B, g10-6 B, g10-7 C, g10-8 A, g03-5 A).**
  The explanation is slotted bullets (verdict, σ, decisive, objection, cause, scale, change, orbit), the same for
  secular/ZLK; every rule and check prints `short name [id]` (ids of docs/libration-analysis.md, `RULE_NAMES`); the
  steps are questions with a "Why"; LIMIT ▼ rows come first (a failed cap margin is never ✗) and the arithmetic is two
  lines; a framed block 'Numbers behind this verdict' (DECISIVE, LIMIT, CONFLICT, UNMEASURED) sits over the table;
  the radial overlap with all eight planets' bands is one line; pendulum numbers are labeled in words with their
  source (observed / from observed cycles / theory); the cause line has three kinds (a driver whose regimes cover the
  episode, a candidate, none). Implemented by E2–E4, E6, E8–E11.
- **Oct 31 (author, W28 g03-6 A, g03-7 A, g02-9 A, g04-6 A, g08-8 A).** (l) draws each episode over its fill interval
  of (a); no empty (m) slot, a warning line says why FAIR is missing; an accepted run the status does not count is a
  gray strip `#id < floor` (episodes.csv `eligible`); the header names the other resonance by its role (owner,
  driver, neighbor); a secular cause is drawn as a line (pϖ on (j), slow intervals on (i)).
- **Oct 31 (author, W28 C13: g03-1 A, g03-3 C, g03-8 A, g03-9 A).** One 1:1 episode in parts: sub-regime brackets
  named by their well, a hatched `inferred` residence beyond the measured core (never on (l)), ▼ `kick` on (d), header
  `orbit: A → B`. Detail: resonances/plotting/CLAUDE.md, The author's figure decisions.
- **Nov 1 (E16).** Docs figures regenerated once after the code freeze; gallery 2306 renamed `secular-z1-unfinished-swing`
  (D6 makes it uncertain / insufficient_time), the TX9 caption follows its action extend (B9). The W28 rules have their
  doc ids (`rules.RuleInfo.doc_id`), new rows 2j, 5x, K3s, Pc, extend, ZF, FE, SZ.G, SZ.ei on the rule index.
- **Nov 4 (author, the right column redesign: go A, D1 A … D10 A; page `tmp/verdict_column_spec/review/page.html`,
  answers `answers.json`).** The visual part of the Claude Desktop spec adopted after a check; its science mostly not
  (wrong definitions and renames: `spec_review.json`). The column is ten blocks in one order for every family
  (`plotting/verdict_column.py`; reader's guide docs/reading-figures.md#the-right-column). D1 A: a number is printed
  wherever it is needed, no `↑` pointers (the CS11 pointer rule of Oct 31 is superseded: Why holds 2–3 numbers, the
  thresholds live in Checks). D2 B: the W28 tree's verdict only in the footer, always, as `W28 tree: status/level`,
  no agrees/differs (a red line would fire on 35–55 % of figures, by his own decisions). D3 B: rule ids as small gray
  tags on check rows; W-codes, column stems and flag names only in the footer. D4 A: the spec's status palette
  everywhere (`style.STATUS_COLOURS`; transient had contrast 2.85). D5 A: the `glance at (…)` chip names panels from
  one map input → panels (`verdict_vocab.GLANCE_PANELS`), to be reviewed by the author once. D6 A: the forced-wave
  veto keeps reason kinematic_libration; the text reads `origin_detail` (no new Reason for 4 rows). D7 A: the
  secular e-response stays out of Caveats until Q11 (it fails on 15 of 25 resonant secular rows, 13 combination
  angles). D8 B, with his note «А почему бы просто не увеличить высоту?»: the figure grows downward first
  (`COLUMN_MAX_GROWTH_IN`), then Numbers groups and ✓ sub-checks are trimmed; Evidence is compacted, never dropped.
  D9 A: the package's terms with plain glosses (owner, kinematic, witness/leg/run/episode kept); only 'overruled by X'
  and 'not certain because …' adopted. D10 A: R0 with a also checks the period law in V_F
  (`SEPARATRIX_VF_PERIOD_LAW`, `SEPARATRIX_PERIOD_TOL` 0.2; Codex measured 0 pass→fail, 0 status or level changes,
  `tmp/r0-scale-2026-11-04/answer.md`).
- 2026-11-07 (the author, paper review item 17): a paper variant of the verdict figure goes into the package, plot
  option `combined.figure='panels'` (MMR and secular/ZLK): no title/header, legend or right column, no empty slot
  (an (l) without an accepted episode and an undrawn FAIR are left out, the rest lettered on from (k)), the margin
  labels right of the panels kept (the caption explains them), the page cropped (12.45 × 12.8 in). Default `full`.
- 2026-11-07 (W36, the author's items 1 and 4): the bottom row of the secular/ZLK figure is four squares: (i) the
  osculating portrait, (j) σ_f against its rate, (k) the filtered Charlier portrait (σ_f and e, or e and i for an
  Ω-only angle, low-passed at the decisive cutoff; R = sqrt(2(1 − sqrt(1 − e²))), Carruba, Aljbaae & Winter 2016
  Fig. 3B), (l) the same per accepted episode in the colours of (a). The legend moved to a band under the row; the
  paper figure drops empty squares and re-letters from (i). Both are for reading only: nothing in them feeds a status.

## Benchmark and process

- **Oct 13 (author).** A label belongs to its exact record (integrator, span, step, epoch); labels from other runs are
  hints only.
- **Oct 15 (author).** All configuration in `SimulationConfig`/`AnalysisSettings` (no separate parameter objects, no
  legacy code, readable names); benchmark code in the main repository; tests written by a context-free subagent.
- **Oct 15 (author).** Examples may be added freely, with honest label sources; reruns under 10 min need no approval,
  runs of an hour or more wait for the final stage.
- **Oct 16.** Agreement is measured against all author verdicts, not only the recent labels.
- **Oct 24 (author).** No new iteration until every disagreement is discussed; an iteration starts on his explicit go.
- **Oct 24 (author, D13).** R-AV: an old `author_verdict` in `candidates.csv` is not compared where a newer author
  label exists for the same record.
- **Oct 27 (author, W24 Q4).** Clones: interface B (orbit solution object + ensemble), 100 clones by default, weak
  NEAs start at the fit epoch (`.eq0`), non-gravitational parameters only flagged, the finder takes the cloud's
  candidates.
- **Oct 27 (author, W24 Q3).** MPC source: query by elements as for AstDyS; a separate module that can become its own
  package; a body at another epoch is integrated to the run date by default.
- **Oct 28 (author + main session, W25).** Clones: a run with clones resumes as a whole (one checkpoint
  `clones_state/`, cleared with simulation_state.json only after ordinary part and clones succeed: the old early clear
  re-ran every ordinary batch after a stop in the clones); the identity covers what was run, the starts are kept and
  compared bit for bit, a mismatch refuses. Q1: a finished one-simulation ordinary part is skipped (verdicts from
  summary.csv, coverage checked), never moved into another grouping. Q2: `ensemble-<parent>-results.npz` kept. Q3:
  restore reclassifies the nominal only and rewrites clones.csv's nominal columns; clones keep stored verdicts. No
  triage (docs/clones.md "Limits"). MPC: `MPC_LOOKUP=file_then_api` (default `file`) takes a missing identifier's
  nominal from get-orb without a covariance; names are documented by the MPC API, so no name mapping was added.
- **Oct 29 (Claude, W26 known gaps; not yet seen by the author).** Restore with recompute also rewrites the nominal's
  summary.csv rows (and mmr_numbers/episodes/candidates) in the run's folder, from the same verdict as clones.csv's
  nominal columns (`clone_store._rewrite_nominal_rows`): one writer of the verdict (`Simulation._finalise_verdict`),
  two projections that cannot disagree on disk. Clones are still not reclassified (Q3 kept): with `SAVE` other than
  `all` the saved clone series are a status-selected subset, so a recompute of those alone would bias the shares.
- **Oct 31 (author, W28 s-1 A, s-2 A, s-4 A; log entry by Claude, F5).** Card labels use
  `author:2026-10-31 (W28 <card>)` (the date of the answers: Claude's reading of s-2 A); the 30.10 blanket "certain
  resonant/transient: all correct" gives `author:2026-10-30` labels (Claude's reading); the s-2 rows that the later
  group answers supersede are listed in the log; the final review folder is discussed + changed + same_rule
  (`benchmark/tools/sort_figures.py --review`).
- **Nov 1 (Claude, W28 close; `benchmark/DECISIONS.md` 2026-11-01, second entry).** The card answers are implemented
  and measured: 49 status moves against `v2-w26-final` (co-orbital 34, mostly C6's single-leg rule; mmr2 9; secular 1,
  5026 by D5; ZLK 5 to −9 by D4/D10), 29 level-only moves, 7 L1 records; `KNOWN_DISAGREEMENTS.csv` goes from empty to
  32 rows (all W28, each on the review page); only E67 stays `known_fail`. Four post-F7 fixes changed no author
  decision (C14 speed, C16 union bound, owner edge cover, figure text). Open for the author: decisions d01–d16 and
  Codex's 9 serious findings (S1 42946 G4h edge and S7 VU6 edge-turn threshold are real bugs, not fixed before he
  discusses them).
- **Nov 2 (author; log entry by Claude, `benchmark/DECISIONS.md` 2026-11-02).** W29: the minimal physical policy
  replaces the W28 tree as the default (`CLASSIFIER_POLICY=minimal`; `v2` kept for comparison, pinned in the tests).
  Status from the geometry of σ_f; physics only vetoes, on a measured contradiction; the level from N cycles, the
  amplitude margin, the veto and stability, with no caps; everything else is a flag; extend only for an unfinished
  libration at the end, a deciding period too long for the record, or (Q5) a veto that needs a longer record within
  `MMR_NEXT_CHECK_SPAN_FACTOR` × T (else review); ZLK without a gate verdict keeps the geometry's status at ≤ medium
  with `gate_unavailable`. Why: the author found the tree over-complicated and his own labels noisy (09:57); the
  principles were agreed at 10:17. His answers to the results page: all defaults, plus Q18 (an alias circulation is
  certain on C5 and C2 full) and Q5 (clause C of extend; the action shown as a tag next to the plaque). The W28 review
  page answers became 70 label rows (`author:2026-11-02 (W28 review page…)`) and episode cases E100–E105; where a label
  and an accepted-by-default rule differ, `KNOWN_DISAGREEMENTS.csv` names the rule and the question (category
  `minimal_policy`, 100 rows).
