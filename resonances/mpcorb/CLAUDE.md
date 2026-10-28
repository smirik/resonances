# mpcorb — the MPC catalogue (`DATA_SOURCE=mpc`)

A self-contained subpackage: it imports nothing of `resonances` (standard library, numpy, pandas; settings through
`Settings`/arguments, logger `mpcorb`) so that it can become its own package (author 2026-10-27). Keep it so: tests
copy it out of the tree and import it alone. The package side lives in `simulation/mpc_source.py` (MPC_* keys →
`mpcorb.Settings`, registered as its defaults; `epoch_refusal`) and `simulation/mpc_getorb.py` (`MPC_LOOKUP`: the
get-orb fallback for identifiers the snapshot lacks). User docs: `docs/config.md`, "Minor Planet Center".

## What it does

- MPCORB (or a subset, `MPC_CATALOG`: nea, neam00, distant, unusual, any http(s)/file URL in the same format) cached
  in `MPC_CACHE_DIR/<catalog>/`: one generation per download, `<sha12>-<file>` + `table-<sha12>-p<parser>.csv`,
  published by snapshot.json (sha256, ETag, Last-Modified, standard epoch). A failure at any step leaves the previous
  snapshot intact. Refreshed only by `update_catalog` / `resonances mpc-update` (a conditional GET on the ETag);
  older than `MPC_MAX_AGE_DAYS` only warns; `MPC_SHA256` pins a snapshot.
- Packed numbers (incl. `~`), provisional designations (incl. extended `_`), packed epochs (MJD TT); names and
  designations in any case; an identifier matching two rows raises.
- `search`, `search_by_axis` (as astdys), `query` (inclusive ranges on a, e, i, q, Q, H, angles, n_obs, n_opp,
  arc_years, U; epoch, numbered, the 4-hex flags: orbit type, NEO, PHA, …; CLI `resonances mpc-search`; `find`/`check`
  take its DataFrame). Bulk lookup is one `isin` pass per column (64k identifiers in 0.5 s).
- Default run date = the catalogue's standard epoch; `config.mpc_snapshot` is the provenance (simulation.json and the
  resume hash). Workers never open the catalogue; `Simulation.run` releases the table first (~0.35 GB).
- `BodyManager.mpc_elements` integrates a row at another epoch to the run date with a warning per body
  (`MPC_EPOCH_POLICY=propagate`, default, `simulation/propagate.py`, IAS15 in the run's planets;
  `MPC_PROPAGATE_MAX_DAYS` an optional limit, empty = none; `find` logs one summary line) or refuses it (`strict`;
  `find` skips such a body with a warning, `add_body`/`check` raise).

## Facts that were measured (do not re-derive)

- Elements: heliocentric osculating, J2000 ecliptic, μ = k² (n = k·a^−1.5 holds on all 1.58 M rows within rounding),
  epoch TT at 0h. The run date is the TT calendar value; Horizons reads it as TDB (difference 0.7 ms).
- The standard epoch equals AstDyS's (MJD 61200 in October 2026) and moves in 200-day steps. About 1.9k rows sit at
  other epochs: the ~1865 unperturbed one-opposition orbits at the end of the file, each at its own epoch, plus a few
  others (e.g. 2008 TC3).
- Propagation was validated over 138 days only (Eros: Δa −5e-8 AU, Δλ 1.4e-6°); 463 from MPC and AstDyS agree over
  2 kyr (4J-2S-1 angle within 0.012°).
- MPC planets: none; planets come from Horizons at the same date, as for AstDyS.
- Quality: rows with U ≥ `MPC_WARN_U`, a letter or blank U, or no perturbers produce a warning. NEA subsets (neam00)
  are wrong for bodies discovered at ~20,000 km Earth passes.

## Traps

- Cached files must follow the umask (staging with `os.open(..., 0o666)`, not `mkstemp`); stale `*.part` files older
  than a day are pruned, newer ones may belong to a running update.
- Packed forms are case-sensitive; only unpacked provisional designations are upper-cased. 1904 OA packs to J04O00A.
- Do not cite `tmp/` reports in code or tests; cite `docs/config.md`.
