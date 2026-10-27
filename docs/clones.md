# Clones (virtual asteroids)

An orbit is known only within its errors. A clone, or virtual asteroid, is one orbit the observations allow; the
nominal orbit is one of them, "without any special significance" (Milani, Chesley & Valsecchi 2000, p. 947). A run
with N clones of an asteroid classifies each clone like any body and reports, per resonance, the share of clones with
each status and its 95 % interval: the probability of a status given the provider's orbit solution, the package's model
and the integration span. It says nothing about the model's or the classifier's own errors.

```python
--8<-- "clones_run.py:clones"
```

The clones run in worker processes, so a script needs the `if __name__ == '__main__':` guard (without it a run on
macOS or Windows stops with an error that names the missing guard). Use 100–1000 clones and the full span for real shares.

Short forms: `sim.add_body(463, '4J-2S-1', clones=100)`, `resonances.check(463, '4J-2S-1', clones=100)`,
`resonances.find(463, clones=100)` (candidates from the whole cloud), `CLONES=100` in `.env`, and on the command line
`resonances check --asteroids=463 --resonances=4J-2S-1 --clones=100`. The body itself is then added from `DATA_SOURCE`
as without clones; its clones come from the orbit solution of `CLONES_SOURCE` (`auto` = the same provider: AstDyS for
`astdys`, MPC for `mpc`, JPL SBDB for `nasa`).

Your own clones: `resonances.Ensemble.from_elements('name', rows, epoch_mjd=..., nominal=...)` (rows with a, e, inc,
Omega, omega, M in radians, one epoch), or an ensemble saved by `Ensemble.save` / read by `Ensemble.load`.

## Where the covariance comes from

An orbit is fitted to the astrometry by least squares. With the residuals linearised around the best fit X₀, the
orbit's density is approximated by the Gaussian

$$
p(X) \propto \exp\!\left(-\tfrac12\,\Delta X^{\mathsf T} C^{-1} \Delta X\right), \qquad \Delta X = X - X_0, \qquad
C = (B^{\mathsf T} W B)^{-1},
$$

B the derivatives of the observations by the six coordinates, W their weights. The clones use the Gaussian
approximation in the published covariance coordinates: C as each provider publishes it, in its coordinates and at one
epoch (whether these are the coordinates the fit was solved in is not stated). A Gaussian is a local approximation and
depends on the coordinates; carrying its draws through the nonlinear map does not repair a poor one. The shares below
are probabilities under this approximation, not the true astrometric posterior:

| source | coordinates X | units | epoch | file |
|---|---|---|---|---|
| `astdys` (numbered) | equinoctial a, h = e sin ϖ, k = e cos ϖ, p = tan(i/2) sin Ω, q = tan(i/2) cos Ω, λ | au, λ in **degrees** (in the covariance too) | `.eq1`: the catalogue's standard epoch (default); `.eq0`: the fit's own | OEF2.0 `newton.spacedys.com/~astdys2/epoch/numbered/<n/1000>/<n>.eq1` |
| `neodys` (NEAs) | as AstDyS | as AstDyS | `.eq0`: the fit's own, near the middle of the arc (default); `.eq1` | `~neodys2/epoch/<designation>.eq0` |
| `sbdb` (JPL) | cometary e, q, t_p, Ω, ω, i | au, JD TDB, degrees | the orbit-solution epoch (not the epoch of the listed elements) | `ssd-api.jpl.nasa.gov/sbdb.api?sstr=..&cov=mat&full-prec=1` |
| `mpc` | Cartesian x, y, z, vx, vy, vz (heliocentric) | au, au/day | the orbit's epoch | `data.minorplanetcenter.net/api/get-orb` |

All in the J2000 ecliptic, heliocentric, as the package; a file stating another frame (OEF `refsys` other than
`ECLM J2000`, MPC `system_data` not ecliptic/ICRF or another obliquity than 84381.448″, SBDB `equinox` not J2000) is
refused, nothing is rotated. An OEF file is read as one record: a second one (a bulk `.ctc`) is refused. The units were checked on the providers' own numbers (tests in
`tests/resonances/clones/`): the OEF `EIG` line is √eig(C) only with λ converted to radians; MPC's `eigenvalues` are
√eig of its CAR covariance; MPC's CAR and COM blocks give its own KEP block; AstDyS and MPC agree on 463 to 3·10⁻⁸ au.

A file is downloaded once into `CLONES_CACHE_DIR/<source>/` with `<file>.meta.json` (URL, download time, sha256);
`orbit_solution(..., refresh=True)` fetches it again (an orbit improves with new observations). The ensemble's JSON
keeps the whole solution.

## Why a multivariate normal in the published coordinates

The coordinates are strongly correlated: in JPL's covariance of 463, |corr(e, q)| = 0.99999. Independent errors per
element, as several papers draw them, put the clones outside the orbit's confidence region; with e and q independent σ_a
comes out 4–900 times too large. Each clone is therefore

$$
X_i = X_0 + L z_i, \qquad C = L L^{\mathsf T}\ \text{(Cholesky)}, \qquad z_i \sim \mathcal N(0, I_6),
$$

in the published coordinates, and converted exactly to Keplerian elements (equinoctial: e = √(h² + k²), ϖ = atan2(h, k),
i = 2 atan √(p² + q²), Ω = atan2(p, q), ω = ϖ − Ω, M = λ − ϖ; cometary: a = q/(1 − e), M = k a^−3/2 (t − t_p);
Cartesian: the two-body map with μ = k², the phase from the argument of latitude, so a circular orbit keeps it). L is
computed from the correlation matrix (a covariance of a good orbit has a
condition number of 10¹³–10¹⁹ in its raw units). A clone that is no bound ellipse (e ≥ 1) is kept as an `invalid` clone,
never redrawn.

Each clone's $d^2 = \Delta X^{\mathsf T} C^{-1} \Delta X = \lvert z\rvert^2$ is stored: it follows $\chi^2$ with 6 degrees of freedom. When float64 finds C singular
(Cholesky fails), the eigen factor keeps its `rank` modes (`Ensemble.rank`, in the JSON as `covariance_rank`) and d² is
the pseudo-inverse distance ΔXᵀ C⁺ ΔX, χ² with `rank` degrees of freedom. In six dimensions the "3σ"
ellipsoid d ≤ 3 holds only 82.6 % of the clones (P(χ²₆ ≤ 9) = 1 − e^−4.5 (1 + 4.5 + 4.5²/2)); 95 % lie within d ≤ 3.55.
The draws are not truncated.

The draws of a body are seeded by (`CLONES_SEED`, crc32(name)) and prefix-stable: the first 100 of 400 clones are the
100 clones.

## Why the covariance's own epoch

`.eq1` is the `.eq0` covariance propagated **linearly**, C₁ = J C₀ Jᵀ. For an NEA observed over a short arc years ago
this no longer describes the orbits: for 2013 RZ53 (a 3-day arc in 2013) σ_a grows 47 times and σ_λ 359 times from
`.eq0` to `.eq1`, through Earth encounters a linear map does not follow, and the `.eq1` matrix is numerically indefinite.
The clones are drawn where the fit was made and each one is integrated to the run's date (IAS15, the run's planets,
`propagate.py`; the nominal alone, the clones together); all of them then start together. Hence the defaults `.eq0` for
NEODyS, the solution epoch for SBDB, and MPC refused for U ≥ 3 (MPC publishes the covariance only at its standard
epoch; U ≥ 3 is a conservative policy, not a physical boundary). For numbered main-belt asteroids `.eq1` is fine and
needs no integration (its epoch is AstDyS's catalogue epoch).

Any gap between the covariance's epoch and the run's date is crossed: up to 10⁻⁴ d (8.6 s) on the two-body ellipses
(5 s already move 463's mean anomaly by 3.8 σ_λ), beyond that by integration. clones.csv records the span
(`propagation_days`) and the way (`propagation`: none, kepler, ias15). The integration was validated against a
provider's own propagation only for MPC elements over 138 days ([Configuration](config.md#minor-planet-center-sourcempc), `MPC_EPOCH_POLICY`); with
`MPC_PROPAGATE_MAX_DAYS` set, a longer span of the clones gets a warning (not a refusal). A timezone-aware run date is
taken as its UTC value, the same instant for the planets and the clones.

The carried cloud is in the package's model: no relativity, no asteroid perturbers, no non-gravitational force. A body
added from another source (the short forms) can then lie off its cloud: `nominal_offset_sigma` in clones.csv is
max(|Δa|/sd(a), |Δλ|/sd(λ)) of the body against its clones at the date, with a warning above 3.

**Non-gravitational parameters** (Yarkovsky A2, comet-like A1–A3) are not modelled: the orbital 6×6 block of the covariance
(the marginal over them) is used, a warning is logged and clones.csv says so (`nongrav`).

**Main belt.** A numbered main-belt asteroid has σ_a ~ 10⁻⁹ au, orders of magnitude below any resonance width: its
clones differ only through chaos, and a regular orbit gives the nominal's verdict to every clone.

## How many clones

A share k/N is a binomial estimate; its 95 % interval is Wilson's score interval (Wilson 1927), with $\hat p = k/N$
and $z = 1.96$:

$$
\text{centre} = \frac{\hat p + z^2/2N}{1 + z^2/N}, \qquad
\text{half-width} = \frac{z}{1 + z^2/N}\sqrt{\frac{\hat p(1 - \hat p)}{N} + \frac{z^2}{4N^2}},
$$

which stays within [0, 1] and is not empty at 0/N or N/N.

| N | half-width at p = 0.5 | 0 of N: upper bound | use |
|---|---|---|---|
| 100 (default) | ±10 % | 3.7 % | resonant, split or not |
| 400 | ±5 % | 0.95 % | the share itself is the result |
| 1000 | ±3 % | 0.38 % | rare outcomes |

Integration is cheap (about 0.04 s per clone per 10 kyr with SABA); the classification is not: 1–2 CPU-min per clone and
resonance for a 1-Myr record. Choose the span and the output step with that in mind.

## How a run treats them

- The nominal is an ordinary body of the run: in its own simulation, as without clones, so its verdict does not change.
- The clones `<name>~c001`, ... are test particles run after the ordinary bodies by the batch workers, `CLONES_BATCH`
  (20) per REBOUND simulation, on `BATCH_N_CORES` processes. Each has its rows in summary.csv and, by `SAVE`, its data
  files; its own figures only with `CLONES_PLOT_EACH=True`.
- In a find (`resonances.find(..., clones=N)` or `add_ensemble(ens)` without resonances), every clone gets the union of
  the MMR candidates of the whole cloud (a clone may sit in a resonance the nominal never visits); the nominal keeps its
  own, so its rows and the `OUTPUT_STEP=auto` step planned from them are those of a run without clones. The clones run
  at that step; a clone-only row it undersamples is flagged by the analysis' own sampling caps.

## Output

- `clones.csv` (top of `save_path`), one row per body and resonance: `n_clones`, `n_classified`, `n_missing`,
  `n_draw_invalid` (e ≥ 1 at the draw, counted as `invalid`); for each status `n_<status>`, `f_<status>`,
  `f_<status>_lo`, `f_<status>_hi` (Wilson 95 %); `modal_status`, `modal_share`; `nominal_status`,
  `nominal_confidence`, `nominal_agreement` (share of clones with the nominal's status, with `_lo`, `_hi`: a diagnostic,
  the nominal has no special weight); `nominal_offset_sigma`; `source`, `product` (eq0, eq1, sbdb, get-orb), `epoch_mjd`
  (the covariance's), `run_date_mjd`, `propagated`, `propagation` (none, kepler, ias15), `propagation_days` (run date
  − epoch), `seed`, `nongrav`. The shares' denominator is `n_classified` (the draw-invalid clones included as
  `invalid`; clones without a verdict are `n_missing`, left out). A Lidov-Kozai libration found kinematic
  (non_resonant) and one without a gate verdict (uncertain) stay in different statuses, never merged.
- `data/ensemble-<name>.csv` and `.json`: every draw (clone index and name, Keplerian elements at the covariance's
  epoch, `valid`, `d2`, the draw in the fit's coordinates `x_<label>`, the elements at the run's date `<element>_start`)
  and the orbit solution with its provenance. `resonances.Ensemble.load` reads it back.
- `figures/<layout>/<name>-<resonance>-clones.png` (plot kind `clones`, also drawn with `combined`; in the folder of the
  nominal's row; drawn when `PLOT` selects the nominal's row or any clone's): (a) the angle of every clone coloured by its
  status, the nominal in black; (b) the status of every clone against its initial a, with a* and a* ± the libration
  half-width the clones' episodes measured; (c) the shares with their Wilson intervals.

## Settings

`CLONES` (0 = off), `CLONES_SOURCE` (`auto`, `astdys`, `neodys`, `sbdb`, `mpc`), `CLONES_EPOCH` (`fit`, `standard`, empty
= the source's default), `CLONES_SEED` (0), `CLONES_BATCH` (20), `CLONES_PLOT_EACH` (False), `CLONES_CACHE_DIR`
(`cache/clones`); see [Configuration](config.md#clone-options).

Not done: the clones' batches are not resumed after an interruption, and `simulation.json`/restore know only the
ordinary bodies (the clones' rows and series are in summary.csv and `data/`); no line-of-variations sampling or
statistical ranging for orbits too weak for a Gaussian; no weighted ensembles.

References: Milani, A., Chesley, S. R. & Valsecchi, G. B. (2000), Planet. Space Sci. 48, 945; Milani, A. et al. (2005),
A&A 431, 729 (multiple solutions, the line of variations); Broucke, R. A. & Cefola, P. J. (1972), Celest. Mech. 5, 303
(equinoctial elements); Wilson, E. B. (1927), JASA 22, 209; Newcombe, R. G. (1998), Stat. Med. 17, 857.
