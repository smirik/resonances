# benchmark — labels, gates, tools

The benchmark (MMR, co-orbital, secular, ZLK) is tracked here; series CSVs, combined plots (`benchmark/data/`,
overridable by `RESONANCES_BENCHMARK_DATA`) and the runs' package output stay outside git. How runs are made:
`tools/RUN.md` (`bench_io.py` loads an example); file layout: `LAYOUT.md`; sources: `INVENTORY.md`; paper examples: `PAPER-EXAMPLES.md`.

## Files and who owns them

- `candidates.csv`: the working catalogue. Its old `author_verdict` values are overridden by newer author labels for
  the same record (R-AV, 2026-10-24).
- `author_label_decisions_2026-10-15.csv` (later rows win; `label_source=author:<date>`) and the
  `author_labels_*.csv` exports: the author's labels.
- `labels.csv`: label buckets; its proposed/claude buckets are not author labels.
- `DECISIONS.md`: the author's decision log (Russian, append-only; a later entry overrides an earlier one; English
  map in `dev/decisions.md`). Entries written by Claude say so.
- `KNOWN_DISAGREEMENTS.csv`: every current disagreement with the author (MMR, co-orbital, secular, ZLK), as
  (example_id, field) with field in status, angle_status, confidence, author_verdict; its category and reason are
  Claude's reading, not labels. Empty since `v2-w24-tx9b` (still empty on `v2-w25-final`).
- `EPISODE_EXPECTATIONS.csv`: 79 episode cases from the author's history (tier A 50 his words, B 12 a review he
  accepted, C 16 old verdicts, D 1 analogies), checks as JSON ops of `benchmark/tools/episode_expectations.py`, verbatim
  quotes with file:line in `benchmark/EPISODE_QUOTES.json`; `known_fail`/`known_reason` are Claude's reading (today E15, E17, E21, E30,
  E43).
- `resonances/data/confidence_cells.json`: calibration cells, rebuilt by `tools/calibrate.py` (reported, never a cap).

## Gates (`make test-benchmark`)

- One reclassification per session (`reclassified_run`, `secular_zlk_run`, `tests/resonances/benchmark/conftest.py`):
  every saved MMR/co-orbital example (`tools/reclassify.py`) and every secular/ZLK one
  (`tools/reclassify_secular_zlk.py`, seconds); no integration; skipped without `benchmark/data`.
- Status gate: fails on any disagreement with the author not in `KNOWN_DISAGREEMENTS.csv`, on a listed one that now
  agrees, and on a listed one whose values changed (`tools/known_disagreements.py <run> [--secular-zlk <run>]
  [--write]`). Secular/ZLK: the same author labels plus the 14 verdicts of `reclassify_secular_zlk.AUTHOR_VERDICTS`.
  On v2-w25-final (2026-10-28): status 111/111, angle_status 94/94, confidence 68/68, author_verdict 198/198,
  secular/ZLK status 3/3, author_verdict 13/13.
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

## Traps

- `tools/reclassify.py --figures` writes `combined_v2.png` into `data/examples/<id>/` even with `--out`: trial
  renders go elsewhere, or back up and restore. Never write into `data/examples` by hand.
- Environment variables do not reach the MMR reclassification's settings (`reclassify.py` calls `settings_from_config()`
  without a source: defaults); `reclassify_secular_zlk.py` builds a `Simulation` and reads them. Pass overrides
  explicitly and check `settings_hash` in `run.json`.
- Byte-identical comparisons exclude `sep_zone_seconds` and key rows by (example_id, resonance).
- Freeze the code before a paired run: spawned workers import the working tree.
- 0 errors in 30 rows bounds the error rate at 9.5 % (95 %); agreement with every label is not an accuracy estimate.
- The repository is public: no absolute paths in tracked files. `data_path` is relative to the repository or
  `<external>/...` (mount point in `RESONANCES_EXTERNAL_DATA`, `extract_existing.resolve_data_path`).
