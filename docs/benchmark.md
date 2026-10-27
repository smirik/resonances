# Benchmark

The thresholds of the analysis are conventions. They are chosen and tested on a benchmark of real integrations whose
verdicts the author of the package gave by eye, from the one-page figures. The benchmark lives in `benchmark/`
(tracked: the catalogue, the labels, the tools); the series and figures are outside git, in `benchmark/data/`
(`RESONANCES_BENCHMARK_DATA` overrides the folder).

## What it holds

Each example is one body in one resonance, with its saved series (`series.csv.gz`), the run's metadata and, where
saved, the planets. Nothing is integrated to classify it again.

| type | examples |
|---|---|
| two-body MMR (`mmr2`) | 286 |
| three-body MMR (`mmr3`) | 174 |
| co-orbital (`coorbital`) | 259 |
| secular | 52 |
| Lidov-Kozai (`zlk`) | 35 |

`benchmark/candidates.csv` is the working catalogue (the author's verdict where he gave one, the sources),
`benchmark/labels.csv` the labels, `benchmark/DECISIONS.md` the author's decisions on disputed cases. Labels written
by Claude or by an automatic judge are marked as such and never count as the author's. The examples of every status
shown in the [Gallery](physics/gallery.md) are benchmark examples.

## Reclassifying

```bash
uv run python benchmark/tools/reclassify.py --out benchmark/data/<run>                    # MMR and co-orbital
uv run python benchmark/tools/reclassify_secular_zlk.py --out benchmark/data/<run>-secular-zlk
uv run python benchmark/tools/known_disagreements.py <run> --secular-zlk <run>-secular-zlk  # against the author
uv run python benchmark/tools/episode_expectations.py <run>                                # the episode cases
```

`reclassify.py` writes summary.csv (with the derived subresonance rows), mmr_numbers.csv, episodes.csv,
candidates.csv, constants.csv and agreement.csv into the run folder. Use a folder of your own for `--out`. The option
`--figures` draws the combined figure **into each example's folder** in `benchmark/data/examples`; the documentation's
figures are drawn elsewhere by `make docs-figures` (see [Gallery](physics/gallery.md)). Workers: `--workers auto`
(`BENCHMARK_WORKERS`, else min(CPUs − 2, available RAM / 3 GB, 8)); about 5 minutes for the 719 MMR examples with
4 workers, seconds for the secular and ZLK ones.

## The two gates of `make test-benchmark`

`make test-benchmark` reclassifies every saved example into `benchmark/data/test-benchmark` (and
`test-benchmark-secular-zlk`) and runs two checks on that run.

1. **Agreement with the author.** Every disagreement (example, field) with the author's labels (status,
   angle_status, confidence where he gave them) and with his unambiguous verdicts must be listed in
   `benchmark/KNOWN_DISAGREEMENTS.csv`. A new disagreement, a listed one that now agrees, and a listed one whose values
   changed all fail the gate. The list's category and reason are Claude's reading, not labels.
2. **Episodes.** `benchmark/EPISODE_EXPECTATIONS.csv` holds 79 cases of episodes the author asked for, each with
   machine checks ("one accepted episode covers the record", "no episode in 39.5–40.2 kyr") and his verbatim words in
   `benchmark/EPISODE_QUOTES.json`. Tiers: A, his own words (50 cases); B, a review he accepted (12); C, old verdicts
   (16); D, analogies (1). A failing A or B case that is not flagged `known_fail` fails the gate, and so does a flagged
   case that passes; C and D are only printed. Five cases are flagged `known_fail` in the current file (E15, E17, E21, E30, E43, all tier A).

## Current agreement

Run `v2-w24-tx9b` (MMR and co-orbital, 2026-10-27) with `v2-w23-secular-zlk` (secular and ZLK, 2026-10-26), checked by
`known_disagreements.py` against the labels of 2026-10-27:

| field | agree / labelled |
|---|---|
| status (author's labels) | 104 / 104 |
| angle_status | 88 / 88 |
| confidence | 61 / 61 |
| the author's verdicts in candidates.csv | 202 / 202 |
| secular and ZLK: status / verdicts | 1 / 1, 13 / 13 |

No disagreement is listed. Labelled examples are a minority of the benchmark, and the labels are themselves revised when a figure is read again; the numbers say that the rules
reproduce today's labels, not that they are right on unlabelled bodies.

## Calibration of the confidence

The same labels give, for every cell (status, angle_status, reason) of the verdicts, how many labelled verdicts at
`high` or `certain` exist and how many of them were wrong, with the Wilson 95 % interval of the accuracy
(`benchmark/tools/calibrate.py` → `resonances/data/confidence_cells.json`). summary.csv reports the cell of every verdict
(`cell_n_examples`, `cell_n_errors`, `cell_accuracy_low_95`, `cell_accuracy_high_95`). The counts never change a level
([Confidence](physics/confidence.md)).
