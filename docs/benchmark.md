# Benchmark

The thresholds of the analysis are conventions. They are chosen and tested on a benchmark of real integrations whose
verdicts were given by eye, from the one-page figures; the labels are the visual verdicts of the package's author.
The benchmark lives in `benchmark/`
(tracked: the catalog, the labels, the tools); the series and figures are outside git, in `benchmark/data/`
(`RESONANCES_BENCHMARK_DATA` overrides the folder).

## What it holds

Each example is one body in one resonance, with its saved series (`series.csv.gz`), the run's metadata and, where
saved, the planets. Nothing is integrated to classify it again.

| type | examples |
|---|---|
| two-body MMR (`mmr2`) | 289 |
| three-body MMR (`mmr3`) | 178 |
| co-orbital (`coorbital`) | 264 |
| secular | 64 |
| Lidov-Kozai (`zlk`) | 37 |

The 731 MMR and co-orbital examples cover 600 bodies, from near-Earth space to the trans-Neptunian region.

`benchmark/candidates.csv` is the working catalog (the visual verdict where one was given, the sources),
`benchmark/labels.csv` the labels, `benchmark/DECISIONS.md` the decisions on disputed cases. Labels proposed by a
program or a language model are marked as such and never count as labels. The examples of every status
shown in the [Gallery](physics/gallery.md) are benchmark examples.

## Reclassifying

```bash
uv run python benchmark/tools/reclassify.py --out benchmark/data/<run>                    # MMR and co-orbital
uv run python benchmark/tools/reclassify_secular_zlk.py --out benchmark/data/<run>-secular-zlk
uv run python benchmark/tools/known_disagreements.py <run> --secular-zlk <run>-secular-zlk  # against the labels
uv run python benchmark/tools/episode_expectations.py <run>                                # the episode cases
```

`reclassify.py` writes summary.csv (with the derived subresonance rows), mmr_numbers.csv, episodes.csv,
candidates.csv, constants.csv and agreement.csv into the run folder. Use a folder of your own for `--out`. The option
`--figures` draws the combined figure **into each example's folder** in `benchmark/data/examples`; the documentation's
figures are drawn elsewhere by `make docs-figures` (see [Gallery](physics/gallery.md)). Workers: `--workers auto`
(`BENCHMARK_WORKERS`, else min(CPUs − 2, available RAM / 3 GB, 8)); about 5–7 minutes for the 727 MMR examples with
4 workers, seconds for the secular and ZLK ones.

## The two gates of `make test-benchmark`

`make test-benchmark` reclassifies every saved example into `benchmark/data/test-benchmark` (and
`test-benchmark-secular-zlk`) and runs two checks on that run.

1. **Agreement with the labels.** Every disagreement (example, field) with the labels (status, angle_status,
   confidence where given) and with the unambiguous verdicts of `candidates.csv` must be listed in
   `benchmark/KNOWN_DISAGREEMENTS.csv`. A new disagreement, a listed one that now agrees, and a listed one whose values
   changed all fail the gate. The list's category and reason are explanations, not labels.
2. **Episodes.** `benchmark/EPISODE_EXPECTATIONS.csv` holds 176 cases of episodes named in the visual review, each
   with machine checks ("one accepted episode covers the record", "no episode in 39.5–40.2 kyr") and the reviewer's
   verbatim words in `benchmark/EPISODE_QUOTES.json`. Tiers: A, the reviewer's own words (78 cases); B, a reading the
   reviewer accepted (35); C, old verdicts
   or provisional readings (63); D, analogies (none at present). A failing A or B case that is not flagged `known_fail` fails the gate, and so does a flagged
   case that passes; C and D are only printed. Four cases are flagged `known_fail`: the Earth co-orbital clones added on
   2026-11-14 (2003 YN107 c027 and c032, 2024 JR16 c005 and c010), labeled before the fix they need; every other A and B
   case passes (14 of the 63 C cases fail and are printed only).

## Current agreement

Run `v2-w37-clean` (MMR and co-orbital, 2026-11-14) with `v2-w37-clean-secular-zlk` (secular and ZLK, 2026-11-14),
settings hash `3d1ef11de54f`, checked by `known_disagreements.py` against the labels of 2026-11-14:

| field | agree / labeled |
|---|---|
| status | 232 / 232 |
| angle_status | 190 / 190 |
| confidence | 178 / 178 |
| the verdicts in candidates.csv | 151 / 151 |
| secular and ZLK: status / angle_status / confidence / verdicts | 53 / 53, 22 / 22, 38 / 38, 8 / 8 |

The run predates the four clone labels of 2026-11-14 (2003 YN107 c027 and c032 transient, 2024 JR16 c005 and c010
resonant); they are listed as known disagreements until the co-orbital fix they need. Labeled examples are a minority of the benchmark, and the labels are themselves revised when a figure is read again; the numbers say that the rules
reproduce today's labels, not that they are right on unlabelled bodies.

## Synthetic tests

The benchmark checks the rules on real integrations, where the true cause is a judgement. The test suite also checks
them on synthetic records whose source of the libration is known by construction (`make test-fast`). Examples:

| known source | built in | tested in |
|---|---|---|
| a pendulum (the force law holds exactly) and a kinematic swing (σ moves by its ϖ part, `a` stays flat) | `tests/resonances/analysis/impl_response_pendulum_synthetic.py` | `test_impl_response_pendulum_*.py`, `test_pendulum.py`, `test_kinematics.py` |
| a libration copied from another resonance (the owner librates, the catalog angle follows it) | `tests/resonances/analysis/test_impl_w2_owner_synthetic.py` | the same file |
| aliasing by the output step | `tests/resonances/analysis/test_w22_alias.py`, `test_impl_w15_alias_gate.py` | the same files |
| members of a multiplet | `tests/resonances/analysis/test_impl_w15_owner_multiplet.py` | the same file |

Each test states its expected numbers as worked out by hand in its docstring, not taken from the code.

## Calibration of the confidence

The same labels give, for every cell (status, angle_status, reason) of the verdicts, how many labeled verdicts at
`high` or `certain` exist and how many of them were wrong, with the Wilson 95 % interval of the accuracy
(`benchmark/tools/calibrate.py` → `resonances/data/confidence_cells.json`). summary.csv reports the cell of every verdict
(`cell_n_examples`, `cell_n_errors`, `cell_accuracy_low_95`, `cell_accuracy_high_95`). The counts never change a level
([Confidence](physics/confidence.md)).
