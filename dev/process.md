# How the work is done

Working conventions agreed with the author during the programme, and the resource rules learned the hard way.

## An iteration

1. **Design, then measure.** A proposed rule (from a review, an outside specification, or a question page) is checked
   against the benchmark before it is implemented. Twice an outside specification was largely overturned by
   measurement (the free-omega gate v3 → v4; SPEC v2's four errors). Report which claims survived; a deviation is
   fine when the measurement behind it is stated.
2. **Physics, not counts.** Statuses are separated by physical criteria. An unavoidable convention becomes a setting
   (`AnalysisSettings` field + `.env.dist` key + `docs/config.md` line) whose comment says it is a convention and
   what it means physically.
3. **Literature with a skeptic.** Follow good literature, then let an independent reviewer (a second model or a
   context-free subagent, with access to the papers) try to break the argument. Ask the author only where the skeptic
   disagrees.
4. **Independent tests.** Tests for a wave are written by a subagent without the implementer's context
   (`tests/resonances/independent_wN/`). When one fails, check the code first: on Oct 15, 6 of 9 failures were the
   code's fault. Disputes are recorded with their reason.
5. **Benchmark gate.** `make test-benchmark` must stay exact; every moved row is explained, and a change that moves an
   author row updates `KNOWN_DISAGREEMENTS.csv` with a reason. Diff the new run against the previous one row by row
   (summary.csv keyed by (example_id, resonance)) and explain every changed verdict.
6. **Independent review of the result.** Each wave ended with a review that tried to break it; its findings were
   verified one by one before fixing (several "fixes" in W17 corrected earlier claims of the implementer).
7. **Author page.** Open questions go to the author on one page (see below). No new iteration starts while
   disagreements remain undiscussed, and an iteration starts on his explicit go.

## Working with the author

- His labels are tentative: argue on the merits and disagree openly; agreement with him is not evidence. His visual
  verdict still overrides old labels and algorithmic statuses, and a final label needs his confirmation
  (`label_source=author:<date>`).
- Make answering cheap: one folder of figures numbered by question, in the current figure format; full paths; plain
  words with the mechanism, not internal codes (R0, rule C); a recommended option per question; a one-line reply form
  ("1a, 2b, 3 yes"); no stale questions left on the page.
- Explanatory material (derivations) in LaTeX, plain language, full derivations, literature references with page or
  equation from the reference library. Do not cite the author's own papers.
- Before explaining what a check means, read what it compares (see [pitfalls.md](pitfalls.md), process). Do not put
  words in his mouth: quote, or say it is your reading.
- His verbatim answers stay local (`tmp/`); the repository holds `benchmark/DECISIONS.md` (his log) and paraphrases.
- Benchmark examples may be added freely with honest label sources; reruns under 10 minutes need no approval; runs of
  an hour or more wait for the final stage.

## Resources

- Heavy runs: `nice -n 10`, at most 4 workers, no heavy test suites in parallel agents, never two pytest sessions in
  one working directory.
- Durable scratch in `tmp/<programme>/` (git-ignored), never only in a temporary directory.
- Freeze the code (a copy loaded through `PYTHONPATH`) before a paired comparison of runs.
- After the code changes, a simplification pass with a memory and CPU focus (the package must run on weak machines);
  verify it byte-identical on the benchmark (fingerprint every CSV row and figure before and after).
