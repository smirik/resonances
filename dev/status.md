# Status on 2026-11-08

## QS run floor and the 1:1 coverage audit (2026-11-14, worktree `qs-missed-episodes`, uncommitted)

Rule `coorbital_qs_run_floor` (`MMR_COORBITAL_QS_RUN_FLOOR`) and the audit (`MMR_COORBITAL_COVERAGE_AUDIT`): see
`resonance/analysis/mmr/CLAUDE.md`; measurements in
`experiments/2026-11-earth-coorbitals/rosengren/claude/followup/implement/CHANGES.md`. `make test-benchmark` not run
yet (another session's pytest); expected moves there: 2010 JK1 non_resonant/high → transient/low (one 25-yr QS
cycle), 2015 YA gains its 51-121-yr QS episode; 2022 YG and 2025 DQ are not reached (mixed qs/planet turns) and keep
the flag `floor_above_own_loop`. Open for the author:

1. Guard of `qs_loop`: K/K_th > 0 plus the period test P >= 0.7 x the run's own loop (`MMR_COORBITAL_QS_LOOP_FACTOR`,
   after review 2's negative controls). `force_check == pass` would not help: its band [1/3, 3] holds the 3x-force
   loop (K 2.68).
2. Audit flags: review only (implemented, minimal policy) or also a cap on a certain/high non_resonant row?
3. 2022 YG (CS5): its 35-482-yr QS run is still rejected (mixed turns); widen the rule to a run whose QS part has
   its own loop, or keep CS5?
4. Which YORP clones and sweep cases go into the benchmark (c046, c087, 2015 YA, 2025 DQ, 2020 FN3, 2015 SO2)?
5. 2010 JK1 becomes transient/low on one 25-yr QS cycle (P/loop 0.87; floor 0.7 x 28.9 = 20.2 yr, above the pass
   cutoff 16.3 yr), against the author's label non-resonant (candidates c0441): accept and list it, or not? It is
   `filter_too_slow` (G(25.1 yr) = 0.85). A re-run at P_c = own floor / 2 instead (review 1) is a new pass, not done.
6. Clone rows: `PLOT=review` does not see the audit flags of clones (`clone_store.ROW_FIELDS`, schema change).

## Dead-code cleanup (2026-11-14, analysis + simulation; plotting in a parallel pass)

The author's rule: code used only by tests goes, unless a test of live code needs it as an oracle or fixture builder;
then it lives in `tests/resonances/analysis/oracles.py`, and a thin wrapper is dropped for the function the package
calls. Removed: geometry wrappers (`classify_run`, `visible_edge`, `_visible_edge`, `prominent_peaks`,
`_prominent_extrema`, `midpoints`), superseded formulas (`varpi_rate_modulation`, `force_law_fails`,
`half_width_in_potential`, `period_through`, `fixed_law_r2`, `period_in_profile`, `composite_centre`, `count_turns`,
`loop_da`, `owner_angle`, `fit_owner`), `classify.util` (`check_chaos`, `is_unphysical_orbit`, `merge_intervals`),
`butter_lowpass_filter`, the calibration-cell lookup (`assess` takes status, margins, caps, U items, physics), the
back-compat readers (integer statuses, old spellings, legacy resume hashes, removed plot kinds,
`PLOT_SUBFOLDER_STRATEGY`) and parameters no caller passed. Moved to the oracles: `fixed_frame`,
`owner_angle_reference`, `lowest_ceiling`, `record_tables_kept`, `averaged_profile`, `elements_from_invariable_plane`,
`argument_of_pericentre`. Reclassification of 42 MMR/co-orbital + 30 secular/ZLK examples byte-identical. Kept:
`PolicyResult.policy`/`Verdict.policy` (plotting reads them; a column), `load_cells` (the sha column),
`mode_phase_ranges` (the builder of `data/mode_phase_uncertainty.json`).

Second pass (2026-11-14, the author: only the format the current code writes is read): restore reads only the
manifest's paths in the run's data folder (no flat-run, no-manifest or `data/` fallbacks), every verdict column of a
summary.csv row (`Verdict.from_flat_dict` raises on a missing one; no status-only rows), no empty periodogram files;
the data folder is in the resume hash (`RESUME_DATA_FOLDERS` gone; `settings_hash` unchanged); the benchmark loader
has no `lam`/`<Planet>.l` names (none of 828 series and 445 planet files used them); `clones.stats.upgrade_row`
(clones.csv before 2026-11-06) and the figure's g5/g6 fallback without per-mode amplitudes are gone. Kept, current
cases: `verdict_text`'s angle-rule reconstruction (G4c/G4h words keep the circulation's angle rule), the policy-less
(a) (inner passes, hand-built verdicts), `pendulum.fit_potential_from_force(free_constant=)`
(`SEPARATRIX_VF_FREE_CONSTANT` and every window's reference correlation). Runs written before (experiments/nea-mmrs,
2026-09-binary-asteroids: no `multiplet_status`) no longer restore. Reclassification of 40 MMR/co-orbital + all 101
secular/ZLK examples byte-identical.

## W36 (2026-11-07/08, merged into v1.0.0): secular episodes

The author's review of the pairs secular wave 1 (20 Myr, SABA) → fixes F1–F5 + the four-square bottom row
(record: `benchmark/DECISIONS.md` 2026-11-07 and 2026-11-08 W36; `dev/decisions.md` Secular and ZLK, Figures). F1
observed exit/entry on σ_f (≥ 1 turn; overrides the Oct 15 rule for such arcs), F2 local TP prominence (capped at the
record's), F3 a bounded run whose centre turns back once → modulated, F4 secular ripples/stalls yield to the slow
motion + the centre-turns scale, F5 a thin end travel holds non_resonant at medium and the figure prints step P. New
settings (all on, two ratios 0.5 conventions): `SECULAR_TP_LOCAL_PROMINENCE`, `ZLK_TP_LOCAL_PROMINENCE`,
`SECULAR_TURNING_CENTRE`, `ZLK_TURNING_CENTRE`, `SECULAR_SLOW_WAVES`, `SECULAR_SLOW_SWING_RIPPLE_RATIO`,
`SECULAR_STALL_RATE_RATIO`, `SECULAR_CENTRE_TURNS_SCALE`. 11 new examples (author:2026-11-07), E107–E117, C55–C60.
On 2026-11-08 the author accepted the package's readings of 142694 (transient), 165548 (transient/medium, removed from
KNOWN_DISAGREEMENTS) and 455327 s−s3 (episode 9.2–10.8 Myr): secular/zlk status 53/53, confidence 38/38, no known
episode failures. 143662 s+g5−g8−s3: he agrees with non_resonant (the policy keeps `t_last` on accepted runs).
The two MMR tests of `test_impl_w3_simcore_verdicts.py` passed only where `cache/solar-*.bin` exists (catalogue
longitudes); fixture `epoch_solar_cache` (tests/CLAUDE.md) makes them portable.

Open:
- ZLK slow waves: the swing ratio overlaps real librations (591986, 162474); 2020 PP1 (464–491 kyr) and 78201
  (83–101 kyr) carry ripple episodes from F2 that F4 does not touch on ZLK.
- No per-body libration period (needs K and A; `dev/physics-notes.md` Secular). A resonance-specific lower cutoff
  only after measuring them.
- Pairs wave 2 (the extend rows, `experiments/2026-11-pairs-secular-zlk/results/final/extend.csv`) waits for the
  author.

## Before W36 (2026-11-03)

Replace this page at each program step. W29 (the minimal physical policy, `CLASSIFIER_POLICY=minimal`, now the
default; `v2` kept for comparison and pinned in the tests) is implemented and committed (not pushed). Latest benchmark
runs: `v2-w29-final` (+ `-secular-zlk`), 725 + 89 examples, 0 errors; the comparison runs `v2-w29-v2` and the W28
baseline `v2-w28-head` (HEAD 0be0ee5). Acceptance (`benchmark/tools/w29_acceptance.py`): 0 invariant violations on
820 rows, clear sets 100 % with 0 opposite verdicts, 0 hard swaps on the author's certain/high labels, held-out 0 ≤ 0
(three-class agreement 0.912; v2 0.987). `KNOWN_DISAGREEMENTS.csv`: 109 rows (minimal_policy 100: a label against a
rule the author accepted by default on 2026-11-02, each with the rule and the question; old_verdict 5; open_question
4); agreement: status 205/226, angle_status 170/176, confidence 112/160, author_verdict 148/156, secular/ZLK status
34/42. Episode gate: 152 cases; 14 `known_fail` (E67 and 13 from the minimal policy's counting: vetoed episodes are
candidates, a subresonance parent's runs are its member's, Q3, P3, the next-check cap, the ZLK gate flag). Tests:
test-fast 5388 passed, 5 xfailed (1.3 min); test-slow 406 passed, 1 xfailed (7.3 min); test-benchmark
656 passed, 3 xfailed (19.6 min, with the reclassification under `--policy minimal`). The record: `benchmark/DECISIONS.md` 2026-11-02, `dev/decisions.md`.

## Before the merge into main (the author 2026-11-04: «Фикси всё, что осталось до слияния»)

Done: untracked deletions (86 benchmark runs, 599 tmp binaries; reports, pages and answers stay until the release);
591986 (Codex + Claude: no rule change, the g7 split is contaminated; docs/CLAUDE.md wording); `pol_veto_need_yr`
writes inf; `ZLK_EPISODIC_NO_VERDICT` removed (the tree always reads the episode's e-i coupling); the R0 sentence names
its episode (79653). In order (one agent at a time where they share files):
1. figure text: the W28 Codex text defects S2 WK15 coverage, S3 BG ripple, S4 RS2 e-i, S5 HW7 150 kyr joint window,
   S8 XP16 support, S9 AC4 filter_too_slow; words for the former caps under the minimal flags; the docs-audit list;
2. dead output: `caps` (always empty), `confidence_raw` (= confidence), `cell_*` and the calibration cells
   (`CONFIDENCE_CELLS`, `resonances/data/confidence_cells.json`, `benchmark/tools/calibrate.py`);
3. back-compat: readers of pre-W29 files (`verdict_from_status_code`, `angle_status_from_text`, `reason_from_text`,
   the figure's pre-W29 path, old summary columns) — the package readers removed 2026-11-14 (the cleanup below);
4. 2016 JA over 5 kyr (rerun `__5kyr`, regime C), then the author's look;
5. R0's scale (a V_F period check, Codex W34) — measure first;
6. computed three-body strength: a research task (Morbidelli ch. 10 averaging), to plan with the author.

## The right column redesign (2026-11-04, the author's D1–D10 + go)

The right column of both verdict figures is the ten-block model `plotting/verdict_column.py` (builders
`verdict_column_mmr/_long/_why/_common`, drawing `verdict_column_draw`; reader's guide
docs/reading-figures.md#the-right-column; decisions `dev/decisions.md` Figures, Nov 4; log `benchmark/DECISIONS.md`
2026-11-04 (3)). The figure grows downward when the column needs it (D8). Open: the D5 panel map
(`verdict_vocab.GLANCE_PANELS`) awaits the author's review; D10's V_F period law (`sep_VF_period_law_med`) is not yet a
line of the Numbers block. The old column engine (`combined_numbers.draw_explained`, `explained_lines`,
`secular_plots.draw_long_period_explained_v1`) was removed on 2026-11-14, and with it, the same day, the text only it
drew (`verdict_text` sentences, bullets, policy block, arithmetic, plaque, `explain_long_period`) and its tests.

## Cleanup after W34 (2026-11-04)

The v2 policy path is gone (`classifier_policy`, `_setting_from_env`, the conftest pin, `--policy` in the tools);
`v2_verdict` keeps the W28 tree's verdict. Tests migrated by four agents (reports `tmp/.../w34/cleanup/report-*.md`):
tree tests read `v2_verdict` / `builders.tree_and_final`; end-to-end ones take the minimal values. Fixed: zlk_integral
veto no longer fails `check_zlk_free_omega`; `pol_core_cycles` column. Open (author): the 591986 g7 representative
gives resonant/low + gate_unavailable (the minimal rule); the minimal figure shows former caps only as flag codes.
Untracked deletion candidates (await the author): `tmp/.../w34/cleanup/delete-benchmark-runs.txt` (86 run folders,
1.2 GB), `delete-tmp-binaries.txt` (599 npz/pkl/pickle/npy files, 5.9 GB). Still before main: the pre-W29 figure path (plotting),
dead columns under the policy (`caps` always empty, `confidence_raw` = `confidence`, `cell_*` blank, the calibration
cells), `ZLK_EPISODIC_NO_VERDICT` (tree only), and the deferred list of the W34 page.

## W34 (2026-11-04, committed ab39e1f): the author's answers to the W33 page

Answers verbatim: `tmp/classifier-v2-2026-10-15/w34/answers-w33-page.json`; reviews `tmp/.../w34/codex-force/answer.md`
and `w34/claude-force/` (L4/R0 on 79653). Labels: 1865, ND, TG2 100 kyr, 23240, VA14 20 kyr
(`author_label_decisions_2026-11-04.csv`). Done:
- R0 per episode (`zone.episode_r0`, `SEPARATRIX_R0_EPISODES`): a record-pooled R0 not measured is read inside each
  accepted episode with >= 2 disjoint cycles (own period, own best frame, thresholds unchanged), lift-only; the x2
  guard keeps it when it compared another episode (`decide.r0_beside_guard`, `R0_KEPT`); column `sep_R0_episode_id`.
  79653 → transient/high (label). L4/multiplet halves: a missing half is no half (`a_response_check`,
  `lower_half_median`; Codex).
- ZLK gate inside the episodes (`omega_free_gate.episode_confirmation`, `ZLK_GATE_EPISODES`; columns
  `zlk_gate_rho_episodes_min`, `zlk_gate_free_turns_episodes_max`, detail `episodes`): 88899 500 kyr → transient/low
  (the e-i integral an objection; against his W32 label: the one open disagreement), 1866 low → medium.
- TG2's stay named in the text (`Episodes.stay_text`). The slow-swing-as-episode rule (1580) was built and removed
  after the author's bounds B («может, тогда это вообще излишне?»): it fired on 1580 only.
- Episode tool: `pool: geometric` (counted + candidates) for the nine cases; D01, E67 rewritten; known_fail 0.
Runs `v2-w34a` + `v2-w34-secular-zlk`: moved 79653, 88899, 1866 (+ R0 pass on 10 rows already dynamic, levels kept).
KNOWN 2 rows (88899). Acceptance PASS with the 88899 swap reason (`w34/swap-reasons.csv`). Page:
https://claude.ai/artifact/9iBbFEh5YTQLPfMyCV2pUG (cards: force A, L4 short route, ZLK, bounds, E02/E28, the rest,
commit). The old pages (W32, W33) could not be deleted from the session (needs the author's confirmation).
Tests: fast 5447 passed (5 xfailed), slow 406 (1 xfailed), benchmark 656 + 1 failure fixed (1866 control: gate kinematic -> confirmed by the episode reading; rerun alone, 5 passed). Open (on the page): the L4 short route (Claude >= 2 P, Codex >= 3 P with <= medium), R0's scale (V_F period) check.

## W33 (2026-11-03, uncommitted): the author's answers to the W32 disagreements page

Answers verbatim: `tmp/classifier-v2-2026-10-15/w33/ANSWERS-2026-11-03c.json`; Codex review `tmp/.../w33/codex/`.
Labels written for every "package" / A answer (+ 555527 high, 1580 transient/low). Two rules:
- F (`policy.mmr_force_shown`, `CONFIDENCE_GEOMETRY_ONLY_MAX`=low): a resonant/transient MMR row with none of L4, R0,
  F passing is at most low; the identity is no force (the author: "no resonant/transient by geometry alone"; Codex
  and Claude keep the status, the angle is bounded). `confidence_basis` ends `;F=shown|none`.
- Owner short windows (`owner.run_owner` → `search_owner`, `MMR_OWNER_SHORT_WINDOW_CYCLES`=2): the search needs
  ~9.25 P of record at 3.5-P windows; with too few windows it runs again at 2 P (~7 P), flag `owner_short_windows`.
  Under v2 it lifts 41588's all_physics_not_measured (medium → high); the v2-pinned text/kinematics tests read 41588
  without the retry (`BEFORE_W33_OWNER`).
  Stage 3 (Nov 5): the measured 0.75-P trim and two windows fitted to the span (~3.75 P), flag
  `owner_fitted_windows`; 14 rows high → certain (v2-w35-fit), 137295 awaits the author.
- Slow-swing extension (the author's W30 epext B, missed in W31): the minimal policy's next check takes the baseline
  estimate when it is a slow swing seen in part (`mmr.decide.SLOW_SWING_SOURCES`), within 4T: 523683 400 kyr (E76
  passes), 385458 300, BG/YG1/ET4 50, EQ7/QT9 60 kyr. The W30 card's "cap 2T" was wrong (the cap is 4T).
- Episode tool: `single_covers` reads each edge as `geometry.covers_edges` (E60 WK15 passes: start gap 1.0000000004 P).
Run `v2-w33b` (= w33a + the 7 extensions); before it `v2-w33a` (secular/ZLK unchanged `v2-w31b-secular-zlk`): 10 rows moved (385533 300 kyr, 463, 7343 → certain; 1862,
1865, 79653, TU54, TG2 0.5 yr high → low; ND, VA14 20 kyr medium → low). KNOWN 38 → 6 (1865, 79653, ND: labels from the
W30 bulk acceptance against F; VA14 20 kyr, TG2 100 kyr, 23240: status, the open "uncertain without force?"). Status
228/231, confidence 171/174, secular/ZLK all agree. Episode gate: no differences, known_fail 13 → 11. Acceptance PASS
(0 violations on 820 rows). Tests: fast 5425 passed, slow 406, benchmark 651 + the 5 41588 tests fixed (rerun alone).
Page for the author: https://claude.ai/artifact/YMFyWbewNTUTyJb1mTBnZ3 (`tmp/.../w33/page`; Codex on episodes:
`tmp/.../w33/codex-episodes/answer.md`). Not done (questions): Codex's stricter ladder (e^q only <= medium, high needs a full
force margin: 10 more rows, 33128 certain -> medium); 1580's and TG2's episode bounds (the author: 1580 ~2.5-6 kyr is the
slow scale's swing, TG2 longer) go to the episode page with the 13 known_fail cases.

## W32 (2026-11-03, uncommitted): the author's answers to the W31 page

Answers verbatim: `tmp/classifier-v2-2026-10-15/w32/ANSWERS-2026-11-03b.json`. He took the secular rows at high, PG14
33S-2 non_resonant, asked for WK15 high "if cheap", accepted R-PASS/R-WAVE/R-NET/R-HALF on condition of a redundancy
check (Claude subagent + Codex, `tmp/.../w32/redundancy/`), and read 2002 BG as one unfinished slow libration
(insufficient_time + extend). Both reviews rejected all four rules (R-PASS is the slow-center turn and fires on forced
waves and exits too; R-NET exists as G4b; R-WAVE was measured on the raw angle, the great inequality; R-HALF: one
cycle fits the sine law for any clean wave). Done instead: B6b takes a ripple whose identity does not pass (fails or
not measured) → BG, 1992 SZ 1M-10J+1, 2011 ET4 uncertain/insufficient_time extend 40 kyr; an all-forced row whose slow
center passes < 1 rev goes to the drift rule; forced runs join the background of the others (DE61 ep 4);
filter_too_slow below `gain_too_slow` (0.911, the thin band of MMR_L8_GAIN_FULL) → WK15 resonant/high; the x2 guard
compares shapes only at the full gain. Still disagreeing (listed): 1862 high vs low, 501710/VA14/DE61/1580/TG2/23240
transient vs uncertain.
Run `v2-w32b` (secular/ZLK unchanged: `v2-w31b-secular-zlk`): 4 rows moved against `v2-w31b`, KNOWN 43 → 41 (status
223/231, confidence 151/169), episode gate no differences (E60 passes the level, fails only on its 1.00-P start gap),
acceptance PASS (0 violations on 820 rows). Tests: fast 5408 passed, slow 406, benchmark 651 + the 5 updated (B6b and
the thin band in v2-pinned tests: 46596 and TX9 diagnostics read without the ripple rule, TX9 stall mark one step).

## W31 (2026-11-03, uncommitted): the author's answers to the W30 page

Answers verbatim: `tmp/classifier-v2-2026-10-15/w31/ANSWERS-2026-11-03.json`; labels he accepted:
`benchmark/author_label_decisions_2026-11-03.csv`; research reports (Codex astra on confidence, three Claude agents on
episodes, disputed three-body rows, a forced-wave criterion): `tmp/.../w31/research/*/`. Done: the ZLK D1-D6 rules
(e-i integral as objection or veto at rho_max >= `ZLK_RHO_KINEMATIC_EXCLUDED`, no cap with a held integral and small
forcing, omega's revolution 2 pi, median and undefined-phase vetoes only with a broken integral, geometric origin
without a confirmed gate); the common confidence ladder (certain N >= 3 with 3 observed cycles in the decisive
episode, high N >= 2; a measured veto is high); the background forced-wave veto (`mmr/background_wave.py`); the
strength veto's word from the center; the zone and floor thin bands; the 1:1 compound-swing join (439908); the
covered-edge fill; a failed identity under `varpi_fast_low_e` is not measured. Runs `v2-w31b` + `v2-w31a-secular-zlk`:
KNOWN_DISAGREEMENTS 118 → 43 rows; agreement status 222/231, confidence 151/170. Open for the author: WK15's 150-yr
floor (a 135-yr floor moves the cutoff of 369 rows), certain for 2.5-cycle secular rows (Codex: no family discount),
PG14 33S-2 (research disagrees), 1862 (high by the rules), 1580/TG2 (a passage at a*: R-PASS proposed), 501710, VA14,
DE61 (drift: R-WAVE/R-NET proposed), 79653 (L4 on halves: R-HALF proposed), 2002 BG (background unsteady).

## W30 (2026-11-02 evening, uncommitted)

After two independent reviews of W29 (Fable, Codex astra; `tmp/.../w29/final-review/`) the author asked to fix the
MMR defects and to have both research the secular and ZLK criteria in the literature (his point: the Kozai constant
and the quadrupole approximation hold for a limited class only; free-ω looks better). Done in code: the strength
marks a not-upheld 5s veto counts again are Episode rows for N (`policy._countable`: 2012 KL9 N nan → 3); a
subresonance parent's N is `N=…(member)`; N < `CONFIDENCE_MEDIUM_CYCLES` (1.5, new) → low (PRINCIPLES §1, AC4, 1580,
4305, 8089); S needs no `filter_sensitive` (2002 XA certain → high); a non_resonant long-period row with a candidate
is medium (`policy.py` candidate was always False); the header's orbit type is read again when the policy moves a
status (KL9 "not this resonance" on a transient); the figure's EPISODES block, panel (l) and the forced-wave labels
follow the policy's counted/candidate; the ZLK gate is the deciding step only with a verdict (555527); no e-i
requirement and no 39.2° line under the minimal policy; an unknown mode-phase error is V not_measured; a zero free
vector has no phase (`free_elements.mask_origin`); the acceptance gate reads the held-out expectation. The criteria
synthesis and the six disputed decisions (D1 the e-i integral as veto or objection, D2 ρ_max ≫ 1 without a gate
verdict, D3 π or 2π for ω's circulation, D4 median-based vetoes, D5 a per-order secular period floor, D6 dynamic or
geometric origin): `tmp/classifier-v2-2026-10-15/w30/criteria/SYNTHESIS.md`; they wait for the author with the MMR
disagreement page (`tmp/.../w30/disagreements/`).

## Open for the author

- W30 candidates he named: the epoch-anchored mode-phase table for secular rows (Q2; 5026 then resonant), a
  re-measured three-body strength (Q3: today only the e^q scale, a flag), the author's own wish for 1997 GK3 high (Q7,
  default medium), 2016 JA extended to ~5 kyr (c6 A), the noise criterion for the co-orbital edge turn (c7 A, VU6),
  AC4's force law against the filter cutoff (c9 B), "are there captures?" on the d06 single-leg rows (his d06 comment).
- The episode cases that fail under the minimal counting (E02, E21, E28, E31, E46, E47, E64, E77, E79): the author's
  episodes are geometric, the minimal figure counts only unvetoed ones. Either the ops read candidates for such cases
  or the cases are rewritten; a question for him, not a code fix.
- `pol_veto_need_yr` = inf (no record up to `ZLK_EXTEND_MAX_YR` resolves the gate, 797204 500 kyr) is written blank
  like "no length named"; the action (review) tells them apart.

## Next steps (agreed order)

1. The benchmark disagreements (109 KNOWN rows, 14 known_fail episode cases) settled with the author, so the gate
   checks something again (2026-11-02: "они нам нужны работающими"); studies wait until then.
2. The deferred studies the author wants to start once the package is in a good state (2026-11-02).
3. A real co-orbital test.
4. Before preparing the merge into main: remove everything unused (the author, 2026-11-02). No `CLASSIFIER_POLICY=v2`
   path if the minimal policy is the one in use, no dead rules, settings or flags, no backward compatibility.

## Separate tasks

- **Computed three-body strength: closed by the author (Nov 5).** No general cheap method exists (the off-resonance
  probe costs 15–30 s per row; a cached second-order coefficient per resonance, Morbidelli 2002 §10.1, would be a
  per-resonance pilot, planar and low-i only), so `eq_scale` stays. The `strength_scale_only` cap (<= medium) exists
  only in the W28 tree (`v2_verdict`, leaf 5u/6; 4 benchmark rows: 2348, 24820, 48789, 68436); the minimal policy
  shows F without capping the level (2348 certain), so no final verdict depends on this task. Reopen only if a
  general method appears (`rejected.md`).
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

- Under the minimal policy (2026-11-02) 14 cases are `known_fail`: E67 (2007 VU6, codex S7, d05) and 13 with a
  minimal-policy reason in `known_reason` (E02, E15, E21, E28, E31, E46, E47, E60, E64, E76, E77, E79, D01). Under
  v2, E17, E30 and E43 pass since W28 (E15 and E21 too). E100-E105 (the author's episode notes of 2026-11-02) pass.
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
    readings. The full run found one wrong C16 union (55390: center travel −299°), fixed by bounding a union to one
    turn of the center; 42946's G4h edge reading is a bug (S1 above).
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
  first-order slope factor 2 fails on large cycles (477492 ratio 2.03 for Codex, 1866 3.0, with H canceled to 4-5 %).
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
- The three-body catalog axis applies the body's precession correction once (`mmr/three_body.py:107`) with the
  small-α g; measured harmless (≤ 4e-7 au iterated, ≤ 2.5e-4 au from the g formula, against a 0.02 au search window;
  the classifier uses its own `exact_axis`).
- Moving-center geometry has two documented possible false refusals with no real-record example yet: the continued
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
     center's slow line, `zlk_centre_*`/`sec_centre_*`; the card put all `ZLK_CENTRE_*` in class 2, but this one bounds
     a column only), `ZLK_NODE_LOCK_MIN_CYCLES` (a flag). Since D10 (q-1 E) `ZLK_EPISODIC_NO_VERDICT` with
     `ZLK_EI_MIN_CYCLES`, `ZLK_EI_PHASE_TOL_DEG`, `ZLK_EI_EPISODE_MIN_COHERENCE` and `ZLK_EI_H_RESIDUAL_MAX` are class 2:
     an episodic row without a gate verdict is transient only when its deciding episode's e–i is zlk, else −9.
  2. may change a status: `ZLK_CENTRE_MODEL_R_MIN`, `ZLK_CENTRE_MODEL_AMP_FACTOR` (the center-scale guard,
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
  integral fails), then 50-75 yr wiggles about a drifting center that do not (200-430 yr: R^2 0.22; Earth crossing,
  q = 0.43 au), swinging 7-10 deg below the local turning-point prominence (13.8 deg). Status unchanged
  (non_resonant/medium). Open: whether the single 0-200 yr loop is an episode is the single-leg question (C6, q-4 A).
- 2014 BT8 1E-1 (W28 C5): the lowered G1q trigger accepts one quasi-satellite loop 794.5-804.2 yr (local law K/K_theory
  1.08, energy passes), the same physics as 2002 PN, so non_resonant/high -> transient/medium. Card g01-5 A put BT8
  among the 13 rows whose status stays (medium only); PN's answer (B, with "technically transient if it converges
  better") is the more specific one for a clean quasi-satellite loop. To confirm with the author.

- A center that turned back (W28 B6c against D6). MMR: two slow turns read as `insufficient_time` only when the
  half-swing obeys the resonance's own pendulum (G5w: slow identity, a swing and period within √10 of the directly
  averaged well); unmeasured (three-body, 1:1) stays `drift`, because the author's drifts 101703/142694 1V+3S-7 have two
  turns too (they fail the slow identity, −0.27/−0.25). Secular/ZLK (D6, `centre_turned_insufficient`): any turn back is
  `insufficient_time`, with no well test, since no computed secular well exists to test it against. Not yet asked:
  should the secular rule also need a physical match once a secular pendulum is available, or should the MMR one drop
  it for two-body angles? Moves at B6c: 523683 2U-9 (few_cycles 200 kyr → arc 400 kyr, the author's g07-5) and
  385458 9N-17 at 100 kyr (few_cycles 200 kyr → arc 300 kyr: identity 1.00, a 1.01, period 0.97; its 300-kyr record
  is resonant/high, so the slow swing is the libration). The card's other fix, keeping a center-scale (G2c) 'slow part
  unresolved' outcome in `_decide_at_scale`, was not needed: G5w reads the same center track at the fine scale.

- 591986: the free-element g is 220″/yr (the 6-kyr free precession, seen directly as the period of the osculating ω
  libration) against 6.65″/yr in the proper-element catalog (definition not resolved). Its ~54-kyr center wave is the
  g6 − g5 beat of Jupiter's own forced vector (W25); over 4 Myr its node librates against g5 (1.46 Myr).
- 78085: the captures are clear, yet the unwrapped angle keeps growing (to be analyzed later).
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
