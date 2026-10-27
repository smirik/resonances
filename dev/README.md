# Development notes

This folder is for people (and agents) who work on the package from a clone of the repository. It is **not** part of
the documentation site: `mkdocs.yml` reads `docs/` only, and nothing here is linked from there. The user documentation
is in `docs/`; the release procedure is `dev/release.md`.

What is here is the durable part of the classifier v2 programme (October 2026) and of the earlier work it rests on:
what was decided and why, what was tried and rejected, the traps that cost time, and the physics the code relies on.
The working notes it was condensed from (reports of every wave, review rounds, question pages) are local and
git-ignored (`tmp/`); a path into `tmp/` below is provenance, not a link that works in a fresh clone.

| Page | What it holds | Read it before |
|---|---|---|
| [history.md](history.md) | The timeline from July to October 2026: waves W1–W24, runs and agreement numbers | writing a paper section on the method, or asking "when did this change" |
| [decisions.md](decisions.md) | The decisions with dates and reasons, by topic; superseded ones marked | changing a status rule, a cap, a label or a figure convention |
| [rejected.md](rejected.md) | Approaches that were built or measured and rejected: do not retry without new evidence | proposing a "new" criterion |
| [pitfalls.md](pitfalls.md) | Traps in tooling, units, epochs, numerics and process | running the benchmark, comparing runs, touching units or epochs |
| [physics-notes.md](physics-notes.md) | The physical conclusions the code relies on, with the objects that showed them | changing the analysis |
| [verified.md](verified.md) | What was checked against independent sources and found correct | re-checking epochs, frames, constants |
| [process.md](process.md) | How an iteration runs, how to work with the author, resource rules | starting an iteration or a long run |
| [status.md](status.md) | Open questions, deferred work and provisional thresholds as of 2026-10-27 | picking up the next task |

Related files in the repository:

- `CLAUDE.md` (root) and the `CLAUDE.md` of each subpackage hold the rules that change how code is written; they link
  here for the reasons.
- `benchmark/DECISIONS.md` is the author's decision log (Russian, append-only: a later entry overrides an earlier one).
  [decisions.md](decisions.md) is its English map by topic, not a translation.
- `benchmark/tools/RUN.md`, `benchmark/LAYOUT.md`, `benchmark/INVENTORY.md`, `benchmark/PAPER-EXAMPLES.md`: how the
  benchmark is built and run.
- `docs/libration-analysis.md`: every formula and the status tree, for users.

Conventions of these pages: dates are 2026 unless stated; "W12" is a wave of the v2 programme (see
[history.md](history.md)); an object is named by number or designation and resonance (`2003 TX9 3E+2J-10`); a rule
ID (G2u, F6, K2, R-ALIAS-A, ...) is explained in `resonances/resonance/analysis/mmr/CLAUDE.md`.
