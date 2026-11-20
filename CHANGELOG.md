# Changelog

All notable changes to `resonances` are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- MMR rule G2c (the scale of the center) no longer displaces a fine libration of dynamic origin by a slower reading
  that a measurement contradicts at its own scale (leaves 5a, 5a*, 5b, 5e, 5f, 5g, 5o): the fine verdict stays, the
  slower motion is reported as the slow scale and in the new `geo_cs_displaced_*` columns (rule
  `centre_scale_displaced`, setting `LIBRATION_CENTRE_SCALE_OBJECTION_GUARD`, on by default). 709123 2J-1 over 50 kyr:
  resonant/low (P 3.26 kyr, the g − g5 wave of the center) → resonant/high (P 458 yr); no other benchmark row moves.
- MMR near_separatrix, pattern B (rule G3) is judged revolution by revolution: a turn counts only when it is slowest
  at the top, slower than a free rotation at its own energy by `LIBRATION_STALL_RATE_TOL`, and its top passage is
  within reach of the separatrix zone (new columns `geo_stall_counted_fraction`, `geo_h_top_med`, `geo_h_top_min`).
  A free rotation just above the separatrix is now a circulation. The separatrix zone reads its strength modulation
  only from motion slower than `LIBRATION_CENTRE_CYCLES` × the angle's own period; a record too short for that has a standing separatrix, zone [1, 1], flag
  `zone_external_unresolved`. The frames the potential is fitted in keep the members' own fit, so the pendulum test
  R0 is unchanged. The ×2 filter guard compares pattern B only on a record without librating episodes (else `stall`
  is in `flt_unchecked_quantities`). Benchmark: 564160 3N-4 (100 kyr) near_separatrix → non_resonant, 145480 1N-4
  (1 Myr) non_resonant → near_separatrix/medium; the author confirmed both verdicts, and 145480 is now the second
  near_separatrix example of the gallery. The standing separatrix of a record too short to resolve external motion
  is a documented convention.
- Pattern B and the separatrix zone after an independent review of the change above. The energy of a turn's top
  passage is read on its own samples, from the observed speed and the potential at the measured phase (it was the
  branch's top speed scaled by the slowest rate, wrong when the turn does not last its branch's period, and signed,
  so a fast backward passage looked below the separatrix); a turn that backs up at the top is read at its turning
  point (new column `geo_stall_reversed_fraction`). The multiplet members modulate the zone when their fit predicts
  the force on blocks it was not fitted on (new column `sep_modulation_held_out_gain`, new flag
  `zone_members_unresolved` when it does so only in sample), not only when their offset turns once: a librating
  secular offset counts. The trend of e is removed before the edge hold, so dK/K and the integrated strength agree
  (a pure trend made a zone [0.975, 1.05]); a librator whose record is too short for its slow window also gets the
  standing separatrix and `zone_external_unresolved`.

### Fixed

- `resonances check --resonances=4E-15` (and any Earth resonance such as `1E-2`) failed: the CLI read the key as the
  number 4e-15. `--resonances` values are now kept as text.

## [1.0.1] - 2026-11-16

### Fixed

- Requires `rebound<5`. REBOUND 5 (released after 1.0.0) renamed the particle `hash` to `name` and does not read the
  archives REBOUND 4 wrote, so with it a fresh install of 1.0.0 failed on the first integration.

### Changed

- Requires Python 3.11.5 or newer (the security fixes of `soupsieve`, which astroquery uses, need it).
- The development dependencies are updated for security fixes (pillow, tornado, cryptography, urllib3, black and
  others); the code is formatted with black 26.

## [1.0.0] - 2026-11-16

The first stable release, and a large one: changes since 0.5.0. One libration analysis now classifies mean-motion,
secular and Lidov-Kozai angles. Each verdict states what the angle does, what causes it and how certain the result
is, and every number the decision used is written to `summary.csv`. The documentation explains the physics behind
each status: [How the status is determined](https://smirik.github.io/resonances/physics/overview/).

### Breaking changes

Code and scripts written for 0.5.x must be updated for these changes:

- **Statuses are strings, not integers.** `body.statuses[key]` and the `status` column of `summary.csv` hold
  `resonant`, `transient`, `near_separatrix`, `non_resonant`, `uncertain` or `invalid`. Comparisons such as
  `status > 0` or `status == 2` no longer work, and the save, plot and layout code refuses an integer status
  (`TypeError`). `body.librations[key]` is the full verdict (a `Verdict` object). The old
  integer survives only in the column `status_code`, for comparison with old runs
  ([legacy codes](https://smirik.github.io/resonances/physics/status-tree/#legacy-integer-codes)).
- **The old classifier is removed.** The module `resonances.libration`, the body attributes `libration_pure`,
  `libration_metrics` and `monotony`, and the `.env` keys `LIBRATION_CRITICAL`, `LIBRATION_PERIOD_CRITICAL`,
  `LIBRATION_MONOTONY_CRITICAL`, `LIBRATION_PERIOD_MIN` and `LIBRATION_FREQ_MIN` no longer exist.
- **`summary.csv` columns renamed or removed.** `type` is now `resonance_type` (`mmr`, `secular`, `lidov_kozai`).
  The initial elements `a`, `e`, `inc`, `Omega`, `omega`, `M` are now `a0`, `e0`, `inc0`, `Omega0`, `omega0`, `M0`.
  `pure`, `num_libration_periods`, `max_libration_length`, `monotony` and `overlapping` are removed.
- **Save and plot settings.** The keys `SAVE_MODE` and `PLOT_MODE` are now `SAVE` and `PLOT` (the arguments are
  still `save` and `plot`). A mode now selects rows by their status string, and the default is `review` instead of
  `nonzero`. `candidates` has a new meaning: `resonant`, `transient`, `near_separatrix` and `uncertain`. The new
  modes are `extended`, `review` and `negative`
  ([Configuration](https://smirik.github.io/resonances/config/#saving-and-plotting)).
- **Integrator arguments renamed.** `integrator_corrector` and `integrator_safe_mode` are now `integration_corrector`
  and `integration_safe_mode` (arguments and `SimulationConfig` attributes). The old names, and the old classifier's
  `libration_period_critical` and `periodogram_critical`, are refused with a `TypeError` that names the replacement.
- **Output layout.**
    - Each saved body has one series file, `data/data-<body>.csv`, with every angle, plus
      `data/data-<body>-periodograms.csv`. Before, the run folder held one file per body and resonance.
    - Figures go to `figures/`, and the one-page verdict figures to `combined/`, both in `confidence/status`
      subfolders.
    - `simulation.json` replaces `simulation.cfg`.

    To get a flat folder, set `data_folder=''` and `plot_layout='flat'`
    ([Output data](https://smirik.github.io/resonances/output/#layout)).
- **`check` and `find`.**
    - `check` returns one `Simulation` for any mix of resonance types. Before, it returned a list of simulations
      when the types were mixed.
    - Without `type`, `find` searches MMRs, secular and Lidov-Kozai resonances. To search only one kind, pass
      `type='mmr'`, `'secular'` or `'lidov_kozai'`.
    - The `order` argument of `find` is removed.
    - `find` no longer defaults to 1 Myr: the span is the configured one (100 kyr unless set).
- **Python API.**
    - Removed from the top level: `find_asteroids_in_mmr`, `secular_check` (use `resonances.check(asteroids,
      'g-g6')`), `Nu6Resonance`, `Nu5Resonance`, `Nu16Resonance` and `GeneralSecularResonance` (use
      `SecularResonance`, or `create_secular_resonance('nu6')`), `Resonance` and `datetime_from_string`.
    - `resonances.resonance.plot` is replaced by `resonances.plotting`.
    - The MMR classes and catalogues moved to `resonances.mmr`. The top-level names `MMR`, `TwoBody`, `ThreeBody`,
      `TwoBodyMatrix`, `ThreeBodyMatrix`, `create_mmr` and `find_mmrs` stay.
- **Removed `.env` keys.** The planetary frequencies `g5`–`g8` and `s5`–`s8` are now constants of the package
  (`resonances/data/const.py`). `ASTDYS_URL`, `ASTDYS_CATALOG` and `CATALOG_PATH` are removed.
- **Packaging.** The project is built with uv and hatchling instead of Poetry. Python 3.11 or newer is still
  required. The new dependencies are `click`, `mpmath` and `gmpy2`.

### Added

- **Libration analysis.** One classifier for every resonance type. It filters the angle at a physical cutoff,
  finds turning points and episodes, and describes the geometry in `angle_status` (`libration`, `episodic`,
  `near_separatrix`, `circulation`, `drift`, `insufficient_time`). It then checks the physics:
    - the identity σ̇ = c (a − a\*);
    - the force law of the semi-major axis;
    - the energy and period law of the potential;
    - the strength from the directly averaged disturbing function;
    - for co-orbitals (1:1), their own force law, with horseshoe, tadpole and quasi-satellite regimes.

    A search for the cause tells a libration of the resonance itself from a kinematic one (driven by the perihelion
    term), from a libration induced by another resonance (owner or driver), from a member of the same multiplet
    (subresonance) and from a joint capture. The verdict columns are `status`, `angle_status`, `reason`,
    `libration_origin` (`dynamic`, `induced`, `subresonance`, `kinematic`, `joint`, `unknown`), `origin_detail`,
    `owner`, `final_regime` and `veto` ([Rule index](https://smirik.github.io/resonances/libration-analysis/)).
- **Confidence and actions.**
    - Each verdict has a confidence level (`certain`, `high`, `medium`, `low`) and an action: `none`,
      `spot_check`, `review`, `extend` (integrate longer, to `extend_to_years`) or `resample` (integrate again at
      `resample_step_yr`).
    - `review_required` marks the rows that need a look, and the `review` save/plot mode keeps them
      ([Confidence](https://smirik.github.io/resonances/physics/confidence/)).
- **The minimal policy.** The status comes from the geometry of the filtered angle. Only a measured contradiction
  (column `veto`: `owner`, `kinematic`, `forced_wave`, `strength`, `zlk_kinematic`, `zlk_free_e`, `zlk_integral`)
  turns a bounded angle `non_resonant`. Every other finding is a flag. The full status tree's own verdict is kept in
  `v2_verdict` ([minimal policy](https://smirik.github.io/resonances/physics/status-tree/#minimal-policy)).
- **Mean-motion resonances.** Retrograde two-body resonances (`1J-3r0`) of bodies with i > 90°, and co-orbitals
  (1:1).
- **Secular resonances, rebuilt.** A formula parser (`g-g6`, `2(g-g6)+(s-s6)`, aliases such as `nu6` and `z1`). Angles are
  built from the planets' proper mode longitudes in the invariable plane (`secular_angle_mode='proper'`, the
  default). `find` looks for candidates in the AstDyS synthetic proper elements
  ([Secular resonances](https://smirik.github.io/resonances/secular/)).
- **Lidov-Kozai resonances.** `LidovKozaiResonance` (key `LK`), Lidov's integrals `c1` and `c2`, and the
  quadrupole Hamiltonian `h_quad`. The free elements split the perihelion into its forced and free parts, and the
  free-ω gate decides whether a libration of ω is the body's own (`check_zlk_free_omega`): confirmed, kinematic or
  no verdict ([Lidov-Kozai resonances](https://smirik.github.io/resonances/lidov-kozai/)).
- **Cross-spectral coherence** of e and i (`coherence.csv`).
- **Clones (virtual asteroids)** drawn from the covariance of the orbit fit (AstDyS, NEODyS, JPL SBDB, MPC):
  `clones=N`. `clones.csv` gives the status shares with Wilson 95 % intervals
  ([Clones](https://smirik.github.io/resonances/clones/)).
- **Minor Planet Center as a data source** (`source='mpc'`, MPCORB), with an epoch policy that integrates rows at
  another epoch to the run's date.
- **Dry-run.** `sim.dry_run()` and `resonances dry-run` check the output step for aliasing, along with the
  integrator and the cost, before a long run. `output_step='auto'` chooses a safe step
  ([Dry-run](https://smirik.github.io/resonances/dry-run/)).
- **Command line.** `resonances check`, `find`, `dry-run`, `mpc-search`, `mpc-update`, `secular-candidates` and
  `skill install` ([Command line](https://smirik.github.io/resonances/cli/)).
- **The combined verdict figure** (`plots=['combined']`, the default). One page per verdict shows the angle, a, e,
  the counted episodes and a column that explains the verdict. A variant for papers is
  `plot_options={'combined': {'figure': 'panels'}}`. The other figure kinds are `evolution`, `ecc_vector`,
  `cross_spectrum`, `free_omega`, `clones`, `recurrence`, `fair`, `portrait`, `cycles`, `angle` and `rate`
  ([Plots](https://smirik.github.io/resonances/plots/),
  [How to read the figures](https://smirik.github.io/resonances/reading-figures/)).
- **New output.**
    - New tables: `mmr_numbers.csv`, `episodes.csv`, `candidates.csv`, `constants.csv`, `coherence.csv` and
      `clones.csv`.
    - New `summary.csv` columns: the verdict, the physical checks (`check_*`), the margins (`mrg_*`), `flags`,
      `rules_fired`, `decision_path`, the policy columns (`n_cycles`, `n_counted_episodes`, `veto`, `v2_verdict`,
      `pol_*`), `settings_hash` and `package_commit`
      ([Output data](https://smirik.github.io/resonances/output/)).
- **Large runs.** Batches on several cores (`BATCH_*`), resume of a batched run in the same folder, and
  `SimulationSerializer.restore` to read a saved run back (`recompute_librations=True` runs the analysis again)
  ([Serialization](https://smirik.github.io/resonances/serialization/)).
- **The Moon as a separate body** (`solar_system_moon`, with IAS15).
- **Agent skill.** `resonances skill install [--user] [--path DIR] [--force]` installs instructions for coding
  agents (how to set up runs and read the verdicts) into `.claude/skills/resonances`
  ([Using with AI agents](https://smirik.github.io/resonances/agents/)).
- **Time in libration.** `summary.csv` columns `lib_time_yr` and `lib_share`: the years of the record inside the
  counted episodes and their share of the record.
- **Short co-orbital episodes.** A quasi-satellite or horseshoe run shorter than the tadpole period floor counts when
  its own loops and its own 1:1 force law support it (`MMR_COORBITAL_QS_LOOP_FACTOR`,
  `MMR_COORBITAL_PAIR_LOOP_FLOOR`, `MMR_COORBITAL_PAIR_LOOP_MIN_HILL`); a 1:1 run whose late cycles leave the well
  keeps its well-holding core and ends only at a separatrix crossing. The record's end is read from the last turning
  point of any run, for mean-motion, secular and Lidov-Kozai angles alike.
- **One run per output folder.** A second run into a folder that a run (or a worker of an earlier run) is still
  writing stops with an error instead of doubling the rows of `summary.csv` (`.run.lock`). The lock is an `flock`:
  on Windows, which has none, it is not taken.
- **Settings for every threshold.** Each threshold of the analysis has an `.env` key and a line in
  [Configuration](https://smirik.github.io/resonances/config/). Thresholds that are conventions are marked as
  such.
- **A benchmark** of 833 real integrations, part of them with visual labels, on which the thresholds are tested
  ([Benchmark](https://smirik.github.io/resonances/benchmark/)).

### Changed

- Secular angles are built from the planets' secular modes (mode longitudes in the invariable plane), not from the
  planets' osculating elements, which do not give the angle of an s_j term correctly.
- The documentation is rewritten. It now includes physics pages for every status, a rule index, a guide to the
  figures, references for configuration and output, and a gallery of benchmark examples.
- Requires `astdys>=1.0.2`: the earlier versions held `pyarrow<18`, which has no wheels for Python 3.13 and later.

## [0.5.0] and earlier

See "Changes before 1.0.0" in the [README](https://github.com/smirik/resonances#changes-before-100).

[1.0.1]: https://github.com/smirik/resonances/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/smirik/resonances/compare/v0.5.0...v1.0.0
[0.5.0]: https://github.com/smirik/resonances/releases/tag/v0.5.0
