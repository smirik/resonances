# benchmark — labels, gates, tools

The benchmark (MMR, co-orbital, secular, ZLK) is tracked here; series CSVs, combined plots (`benchmark/data/`,
overridable by `RESONANCES_BENCHMARK_DATA`) and the runs' package output stay outside git. How runs are made:
`tools/RUN.md` (`bench_io.py` loads an example); file layout: `LAYOUT.md`; sources: `INVENTORY.md`; paper examples: `PAPER-EXAMPLES.md`.

## Files and who owns them

- `candidates.csv`: the working catalogue. Its old `author_verdict` values are overridden by newer author labels for
  the same record (R-AV, 2026-10-24).
- `author_label_decisions_2026-10-15.csv` (later rows win; `label_source=author:<date>`; later files too, e.g.
  `_2026-11-07.csv`: the pairs secular wave 1) and the
  `author_labels_*.csv` exports: the author's labels.
- `tools/import_review_verdicts.py <export.json> --review NAME [--write]`: the author's verdicts on a review folder
  (`sort_figures.py --review NAME`; keys `example_id` or `example_id::resonance`, `verdict` agree|status) as label rows
  for `author_label_decisions_2026-10-15.csv` (`author:<date>`) and proposed episode cases (tier A, empty checks) in
  `review/<NAME>_PROPOSED_CASES.csv`, printed as a diff; `--write` applies, an unknown id rejects the whole export, a
  second run adds nothing. It never writes EPISODE_EXPECTATIONS.csv: a proposed case gets its ops by hand.
- `labels.csv`: label buckets; its proposed/claude buckets are not author labels.
- `DECISIONS.md`: the author's decision log (Russian, append-only; a later entry overrides an earlier one; English
  map in `dev/decisions.md`). Entries written by Claude say so.
- `KNOWN_DISAGREEMENTS.csv`: every current disagreement with the author (MMR, co-orbital, secular, ZLK), as
  (example_id, field) with field in status, angle_status, confidence, author_verdict; its category and reason are
  Claude's reading, not labels. Empty from `v2-w24-tx9b` to `v2-w26-final`; 32 rows after W28; 109 since W29
  (`v2-w29-final`, 2026-11-02: minimal_policy 100, a label against a minimal rule the author accepted by default,
  each reason naming the rule and the RESULTS question; old_verdict 5; open_question 4); empty at W35; W36 test-first: 4 secular rows
  `unreviewed`, each naming the fix expected to clear it; empty again after W37, 2026-11-14; 4 co-orbital rows `unreviewed`
  the same day, test-first: the Earth co-orbital clones' status labels, each naming the pending co-orbital fix; cleared
  by the three co-orbital fixes, which left 2 new `unreviewed` rows: 2022 BB2 status, 2026 SZ2 confidence; both cleared
  by the author's labels of 2026-11-15, the list empty again; 2026-11-20 test-first: 709123 2J-1 confidence `unreviewed`, cleared the same day by the G2c guard; 2026-11-21 test-first: 709123 2J-1 20-kyr record (c0819) status and angle_status `unreviewed`, the false episode break, until the episode-detector fix).
- `EPISODE_EXPECTATIONS.csv`: 177 episode cases (E100-E105: the author's episode notes of 2026-11-02; E107-E117 and C55-C60:
  his review of the pairs secular wave 1, 2026-11-07; C61: Codex's reading of 455327 g-s-g5+s6, the W36 F2 guard; E118: 485 Genua
  3J-1S-1, the author 2026-11-14; E119-E122: Earth co-orbital clones 2003 YN107 c027/c032 and 2024 JR16 c005/c010, the
  author 2026-11-14, `known_fail` test-first, cleared the same day by rules coorbital_well_core and
  coorbital_pair_loop_floor; E123: 2022 RD2 c075, turns inside 1-3 r_H, the author 2026-11-15) from the
  author's history (tier A 79 his words, B 35 a review he accepted, C 63 old verdicts or provisional readings), checks
  as JSON ops of `benchmark/tools/episode_expectations.py`, verbatim quotes with file:line in `benchmark/EPISODE_QUOTES.json`;
  `known_fail`/`known_reason` are Claude's reading (W28: only E67, 2007 VU6, the edge-turn threshold; W29: 13 more
  where the minimal policy counts differently, e.g. a vetoed episode is a candidate, not counted). W28 F3
  added the no-code cases of the author's 2026-10-31 answers (quotes: the answer line of the card and the s-1 approval
  of the case list), rewrote E17 (g11-4 A: two episodes allowed) and replaced C04/C11; a later WP that moves a guard
  (E66 by C5, E72 by C8, E99 by A2) updates it with a reason.
- Ops besides the episode shapes: `period_in` (decisive or any episode's period), `action` (with `min_extend_to`),
  `field` (a summary column equal / list item / at least), `flag` (present or absent) and `episode_mark` (the
  episodes.csv `marks` column, items `name`, `name@t`, `name@t0:t1`; a run without it fails with 'marks not written');
  every op takes `scale` (default main; B10a writes `scale=slow` rows). A WP that writes marks or scale rows uses these
  spellings. W28 C8 (1:1 core and inferred residence, episodes.csv `core_*`/`residence_*`): `core_near` (t0, t1,
  tol), `residence_reaches_end` (the last episode's residence bounded by the record's edge), `residence_starts_before`
  (t); a run without the columns fails ('core / residence not written'); `exit_observed` takes `value` (false: no
  exit). Sub-regime boundaries are checked with `episode_mark` on `well_change` (E70, E71).
- The ops count `accepted` runs ('plain': those without a `forced_wave` route). Since W28 C11 episodes.csv also has
  `eligible`/`ineligible_reason` (an accepted run the status does not count: `below_period_floor`, `centre_turns`,
  `forced_wave`, `induced`, `before_last_jump`); no op reads them yet, so an accepted run below the period floor
  still counts in `n_accepted` (2003 YN107, 2022 AP1). A case about the episodes the status counts needs an op on
  `eligible`.
- `resonances/data/confidence_cells.json`: calibration cells, rebuilt by `tools/calibrate.py` (reported, never a cap;
  last rebuilt on `v2-w28` + secular/ZLK, 2026-11-01).
- Review folder of an iteration (W28 s-4): `data/figures/review-<NAME>/{discussed,changed,same_rule}/` (hard links)
  + `review-<NAME>.csv`, built by `tools/sort_figures.py --review NAME --run <run> --baseline <previous>` (`LAYOUT.md`);
  `review/W28_DISCUSSED.csv` (tracked) lists the records the cards named. `review-w28`: 366 rows (discussed 132,
  changed 57, same_rule 177). The author's verdicts come back through `tools/import_review_verdicts.py`.

## Gates (`make test-benchmark`)

- One reclassification per session (`reclassified_run`, `secular_zlk_run`, `tests/resonances/benchmark/conftest.py`):
  every saved MMR/co-orbital example (`tools/reclassify.py`) and every secular/ZLK one
  (`tools/reclassify_secular_zlk.py`, seconds); no integration; skipped without `benchmark/data`.
- Status gate: fails on any disagreement with the author not in `KNOWN_DISAGREEMENTS.csv`, on a listed one that now
  agrees, and on a listed one whose values changed (`tools/known_disagreements.py <run> [--secular-zlk <run>]
  [--write]`). Secular/ZLK: the same author labels plus the 14 verdicts of `reclassify_secular_zlk.AUTHOR_VERDICTS`.
  On v2-w26-final (2026-10-29): status 111/111, angle_status 94/94, confidence 68/68, author_verdict 198/198,
  secular/ZLK status 6/6, author_verdict 13/13. With the W28 F3 labels (1862, CB, EZ84, 373136 100 kyr, the s-1
  controls and the 44 rows of the 30.10 blanket "certain resonant/transient: всё верно"), read on the same run:
  status 149/149, angle_status 96/96, confidence 115/115, author_verdict 186/186, secular/ZLK status 32/32, confidence
  26/26, author_verdict 8/8 (a newer label replaces an old author_verdict, R-AV). On the W28 final run
  (`test-benchmark`, 2026-11-01): status 173/179, angle_status 124/126, confidence 127/133, author_verdict 157/173,
  secular/ZLK status 39/40, angle_status 9/9, confidence 27/28, author_verdict 8/8; 32 listed disagreements;
  episodes A 55/55, B 34 + 1 known (E67), C 3 failing (not gating).
- Episode gate (W23): a tier A/B case that fails and is not `known_fail` fails the gate; so does a `known_fail` case
  that passes (clear the flag). C/D never gate. Synthetic companions: `tests/resonances/analysis/test_episode_detection.py`.
- Workers: `BENCHMARK_WORKERS`, else min(CPUs − 2, available RAM / 3 GB, 8), or RAM / 1.6 GB with the low-memory
  allocator when memory limits them (`LOW_MEMORY_ALLOCATOR`). Examples run longest first; workers are recycled
  (`--tasks-per-worker`, default 5). A 1-Myr record peaks at ~2.6 GB per worker.
- Series load through `tools/column_cache.py` (binary copy, planets as read-only memory maps, capitals escaped for
  case-insensitive file systems).

## Process

- Whenever a session meets an object worth keeping as an example, propose it to the author (name, resonance,
  integration settings, his verdict, why distinct); add it only after his explicit confirmation, with
  `label_source=author:<date>`. Additions are welcome; reruns under 10 min need no approval.
- The author's visual verdict overrides old labels and algorithmic statuses. Labels produced by Claude or an LLM judge
  are marked as such and never count as the author's.
- A label belongs to its exact record (integrator, span, step, epoch). Variant records (`__20kyr`, `__0.5yr`) are
  separate examples.
- A change that moves an author row updates `KNOWN_DISAGREEMENTS.csv` with a reason; a change that makes an episode
  case pass or fail updates its flag with a reason. Every changed verdict row is explained against the previous run.
- Run names `v2-wNN[-suffix]` in `benchmark/data/`.
- A paper-worthy case is proposed to the author with a one-line formulation and its topic; after his yes it goes
  into `PAPER-EXAMPLES.md` (grouped by paper topic, numbers only from a named run's CSVs, "кандидат" until the
  formulation is settled; no `tmp/` paths).

## Traps

- `tools/reclassify.py --figures` writes `combined_v2.png` into `data/examples/<id>/` even with `--out`: trial
  renders go elsewhere, or back up and restore. Never write into `data/examples` by hand.
- Environment variables do not reach the MMR reclassification's settings (`reclassify.py` calls `settings_from_config()`
  without a source: defaults); `reclassify_secular_zlk.py` builds a `Simulation` and reads them. Pass overrides
  explicitly and check `settings_hash` in `run.json`.
- Byte-identical comparisons exclude `sep_zone_seconds` and key rows by (example_id, resonance).
- `tools/calibrate.py --run <run>` reads `<run>/secular_zlk/summary.csv`, but since W26 the secular/ZLK part is the
  sibling run `<run>-secular-zlk`: link it (`ln -s ../<run>-secular-zlk benchmark/data/<run>/secular_zlk`) or the
  secular/ZLK rows silently drop out of the cells.
- `make test-benchmark` writes the session's run to `data/test-benchmark` (+ `-secular-zlk`): it is the latest run of
  the working tree, overwritten by the next session; copy it to a named `v2-wNN` run to keep it.
- Freeze the code before a paired run: spawned workers import the working tree.
- A `benchmark`-marked test in a module (e.g. `test_episode_expectations.py`) pulls the `reclassified_run` fixture, a
  full reclassification on up to 8 workers: run a module's fast tests with `-m "not benchmark"`.
- 0 errors in 30 rows bounds the error rate at 9.5 % (95 %); agreement with every label is not an accuracy estimate.
- The repository is public: no absolute paths in tracked files. `data_path` is relative to the repository or
  `<external>/...` (mount point in `RESONANCES_EXTERNAL_DATA`, `extract_existing.resolve_data_path`).
