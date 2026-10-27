<!-- Generated from docs/config.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Configuration

The defaults of every option are in `resonances/.env.dist`, with a comment each. Do not edit that file. To change a
value, in order of priority (the first wins):

1. an argument of `Simulation` (lower-case name), or an attribute of `sim.config` set before `sim.run()`;
2. a `.env` file in the working directory (upper-case key);
3. an environment variable (upper-case key);
4. the default of `.env.dist`.

```python
import resonances

sim = resonances.Simulation(name='test', integrator='whfast', save='all', save_path='output/my_run')
sim.config.plot = 'nonzero'  # after creation: through sim.config
resonances.config.set('INTEGRATION_DT', '0.5')  # a .env.dist key, as a string; read when a Simulation is created
print(resonances.config.has('INTEGRATION_DT'), resonances.config.get('INTEGRATION_DT'))
```

`config` holds the merged mapping of `.env.dist`, the environment variables and `.env`. `config.set` raises for a
key that is not in that mapping; `config.get` raises for a missing key unless a default (not `None`) is given. The
mapping is not a list of supported options: an environment-only key is accepted. `find()` and `check()` pass their
extra keyword arguments to `Simulation`.

## Saving and plotting

The save and plot filters select rows by their `status` string (`status_selected` in
`resonances/simulation/config.py`; the statuses are explained in [How the status is determined](physics/overview.md)):

| mode | rows selected |
|---|---|
| `all` | every row |
| `resonant` | `resonant`, `transient` |
| `candidates` | `resonant`, `transient`, `near_separatrix`, `uncertain` |
| `extended` | candidates, and `non_resonant` rows with a `libration_origin` (kinematic, induced, subresonance: a bounded angle that is not this resonance's own) |
| `nonzero` | everything except `non_resonant` without a `libration_origin` (default) |
| `negative` | `near_separatrix`, `uncertain`, `invalid` |
| `None` | nothing |

| option / key | default | meaning |
|---|---|---|
| `save` / `SAVE` | `nonzero` | which rows' series are written (mode above) |
| `save_path` / `SAVE_PATH` | `cache` | the run folder. Without the argument it is `SAVE_PATH/<name>`, where `name` defaults to the current timestamp (`find` and `check` use `name='resonance_find'`). The tables (`summary.csv`, `episodes.csv`, …) and `simulation.json` go to its top |
| `data_folder` / `DATA_FOLDER` | `data` | the folder of the series CSVs under `save_path` (`data-<body>.csv`, periodograms, `<body>-omega-free.csv`, `data-planet-<planet>.csv`); empty = `save_path` itself. Restore finds the series either way |
| `save_summary` / `SAVE_SUMMARY` | `True` | write `summary.csv`, `episodes.csv` and `coherence.csv` (every resonance type) and, for MMRs, `mmr_numbers.csv`, `candidates.csv`, `constants.csv`. Writing the same bodies again replaces their rows ([Output data](output.md)) |
| `save_planets` / `SAVE_PLANETS` | `analysis` | which planets are kept and written: `all`, `analysis` (those of the bodies' resonances, plus `MMR_OWNER_PLANETS` when the owner search is on) or `none` (a bool: `True` = `all`). The MMR analysis reads them for FAIR, the removal of the planetary ripple, the directly averaged strength, the moving centre and the subresonances; without them the owner search falls back to catalogue longitudes (confidence ≤ medium). `secular_planet_phases='measured'` needs `all`. A planet costs 64 bytes per output step (about 6 MB for 100 kyr at 1 yr) |
| `plot` / `PLOT` | `nonzero` | which rows get figures (mode above) |
| `plots` / `PLOTS` | `combined` | the figure kinds: `combined` (the one-page verdict figure, MMR, secular, ZLK), `evolution`, `ecc_vector`, `cross_spectrum`, `free_omega` (ZLK), `clones`, MMR `recurrence`, `fair`, `portrait`, `cycles`, secular/ZLK `portrait`, `angle`, `rate`. An unknown name is an error ([Plots](plots.md)) |
| `plot_always` / `PLOT_ALWAYS` | empty | resonance types (`mmr`, `secular`, `lidov_kozai`) plotted whatever their status (not when `plot` is `None`) |
| `plot_type` / `PLOT_TYPE` | `save` | `save` the figures, `show` them, or `both` |
| `image_type` / `PLOT_IMAGE_TYPE` | `png` | the image format, `png` or `pdf` |
| `plot_config` / `PLOT_CONFIG` | `full` | the panel preset of the `evolution` figure: `full` or `simple` |
| `plot_options` / `PLOT_OPTIONS` | empty | options of the diagnostic figures (dict, or JSON in `.env`), merged over the defaults and validated ([Plots](plots.md#options)) |
| `plot_path` / `PLOT_PATH` | `cache` | the figures' folder; without the argument `PLOT_PATH/<name>`, as `save_path` |
| `plot_figures_folder` / `PLOT_FIGURES_FOLDER` | `figures` | the individual figures under `plot_path`; empty = `plot_path` itself |
| `plot_combined_folder` / `PLOT_COMBINED_FOLDER` | `combined` | the combined figures under `plot_path`; empty = `plot_path` itself |
| `plot_layout` / `PLOT_LAYOUT` | `confidence/status` | subfolders inside both figure folders: `flat`, or levels joined by `/` from `status`, `confidence` (`none` for a verdict without a level), `type`, `resonance`, `body`. Presets: `flat`, `status`, `confidence`, `confidence/status`, `status/confidence`, `type/status`, `type/confidence/status`, `resonance`, `body`. A row without a verdict goes to `unclassified` ([Output data](output.md#layout)) |
| `plot_combined_layout` / `PLOT_COMBINED_LAYOUT` | empty | the layout of the combined figures alone; empty = `plot_layout` |
| `plot_subfolder_strategy` / `PLOT_SUBFOLDER_STRATEGY` | empty | old name, still read: `status` = layout `status`, `flat` = `flat`; the key only when `PLOT_LAYOUT` is empty |

## Output of the MMR analysis

| option / key | default | meaning |
|---|---|---|
| `save_episodes` / `SAVE_EPISODES` | `True` | write `episodes.csv` (every run of turning points, every resonance type) |
| `save_candidates` / `SAVE_CANDIDATES` | `top` | rows of `candidates.csv`: the best and the accepted subresonances and owners (`top`), or `all` |
| `save_constants` / `SAVE_CONSTANTS` | `True` | write `constants.csv` (mean motions, frequencies, great inequality), once per run |
| `output_step` / `OUTPUT_STEP` | empty | the output step in years: sets `Nout` = ⌈span / step⌉ + 1 (the step is an upper bound); `auto` lets the dry-run choose a step that folds no short period ([Dry-run](dry-run.md)); empty = `Nout` = \|tmax\| / 100. An explicit `Nout` wins |

## Cross-spectral coherence

The σ–a, σ–e, σ–i and e–i cross spectra ([Cross-spectral coherence](coherence.md)). They change no status.

| option / key | default | meaning |
|---|---|---|
| `coherence_enabled` / `COHERENCE_ENABLED` | `True` | master switch |
| `coherence_skip_non_resonant` / `COHERENCE_SKIP_NON_RESONANT` | `True` | skip rows with status `non_resonant` (kinematic and induced included); they dominate surveys |
| `coherence_alpha` / `COHERENCE_ALPHA` | `0.01` | per-bin false-alarm probability α of the threshold $\gamma^2_{\rm crit} = 1 - \alpha^{1/(n_{\rm seg}-1)}$ |
| `coherence_n_segments` / `COHERENCE_N_SEGMENTS` | `4,8,16` | Welch segmentation levels; the coarsest level with a non-empty validity band decides |
| `coherence_window` / `COHERENCE_WINDOW` | `hann` | Welch window |
| `coherence_max_lines` / `COHERENCE_MAX_LINES` | `10` | strongest lines per pair and level written to `coherence.csv` |
| `coherence_period_min`, `coherence_period_max` / `COHERENCE_PERIOD_MIN`, `COHERENCE_PERIOD_MAX` | empty | narrower search band in years; empty = the preset of the resonance type (10 kyr for secular and ZLK). The limits $4\,\Delta t$ and $T/(5 n_{\rm seg})$ always apply |
| `coherence_pairs` (argument only) | preset | pairs per resonance type, e.g. `{'mmr': ['sigma-a', 'sigma-e']}` |
| `coherence_phase_targets` (argument only) | preset | `(target_deg, tolerance_deg)` per type and pair, e.g. `{'lidov_kozai': {'e-i': (180.0, 30.0)}}` |
| `zlk_coherence_check` / `ZLK_COHERENCE_CHECK` | `True` | measure the `confirmed_ei` flag (do e and i trade at the strongest coherent line); a flag only |
| `zlk_min_anticorrelation` / `ZLK_MIN_ANTICORRELATION` | `0.5` | minimum \|r\| of the band-passed e and i for that flag; provisional, calibrated on two objects |

## Free elements and the free-ω gate

The forced/free split of the e and i vectors and the free-ω gate ([Free elements](free-elements.md)). The gate is a
step of the Lidov-Kozai analysis: its outcome is read by the status decision
([Secular and Lidov-Kozai](physics/secular-zlk.md)). On a librating ω, *confirmed* keeps `resonant`, *kinematic*
gives `non_resonant` (reason `kinematic_libration`), and *no verdict* gives `uncertain` (reason
`gate_unavailable`).

| option / key | default | meaning |
|---|---|---|
| `free_elements_enabled` / `FREE_ELEMENTS_ENABLED` | `True` | split the vectors of bodies with a Lidov-Kozai resonance; off means no split and no gate |
| `free_elements_sampling_years` / `FREE_ELEMENTS_SAMPLING_YEARS` | `500` | decimate to about this step (with an anti-aliasing filter) before the split; empty keeps the integration grid |
| `free_gate_enabled` / `FREE_GATE_ENABLED` | `True` | whether the gate's outcome enters the status; off computes and reports it only |
| `free_gate_min_cycles` / `FREE_GATE_MIN_CYCLES` | `3` | libration periods the record must hold before the gate judges; fewer is *no verdict* |
| `free_gate_mask_quantile` / `FREE_GATE_MASK_QUANTILE` | `0.15` | samples with \|z_free\| below this fraction of its median are dropped before unwrapping (the argument is undefined near the origin) |

The planetary fundamentals of the fit are $g_5 \ldots g_8$ and $s_5 \ldots s_8$ of `PLANETARY_FREQUENCIES`
(`resonances/data/const.py`).

## Secular angles

| option / key | default | meaning |
|---|---|---|
| `secular_angle_mode` / `SECULAR_ANGLE_MODE` | `proper` | `proper`: the angle against the planets' mode longitudes $g_j t + \phi_j$, the body's node from the invariable plane (what the classifier reads); `osculating`: from the planets' osculating elements, right only for $g_1$, $g_5$, $g_6$ terms ([Secular resonances](secular.md)) |
| `secular_remove_forced` / `SECULAR_REMOVE_FORCED` | `True` | reduce the proper angle: subtract the forced terms at the fundamentals that are not in the angle; `False` removes only the invariable plane |
| `secular_planet_phases` / `SECULAR_PLANET_PHASES` | `linear` | the planets' mode phases: the linear law of `const.py`, or `measured` in the integration (needs `save_planets='all'`) |
| `secular_phase_window` / `SECULAR_PHASE_WINDOW` | `5000000` | the sliding window of the measured phases [yr] |

## Body-level filter and periodograms

These feed the `evolution` figure and the periodograms only. The classifier chooses its own cutoff for every angle
([From the integration to the angle](physics/signal.md)).

| option / key | default | meaning |
|---|---|---|
| `oscillations_cutoff` / `LIBRATION_FILTER_CUTOFF` | `0.0005` | cutoff [1/yr] of the Butterworth low-pass of a (`a_filtered`) and of the portrait edge: periods below 2000 yr are damped |
| `oscillations_filter_order` / `LIBRATION_FILTER_ORDER` | `2` | order of that filter |
| `filter` / `FILTER` | `butter` | the body-level filter; only the Butterworth filter (`butter`) exists |
| `periodogram_frequency_max` / `LIBRATION_FREQ_MAX` | `0.002` | highest frequency of the periodograms [1/yr] |
| `periodogram_frequency_min` (argument only) | 1/span | lowest frequency of the periodograms |
| `periodogram_soft` / `LIBRATION_SOFT` | `0.05` | power above which periodogram peaks are reported |
| `libration_period_min` (argument only) | 5 % of the span | cut from each end of a series before its periodogram |

## Libration analysis settings

Every threshold of the libration analysis and the dry-run. Each is a `SimulationConfig` attribute of the same name in
lower case, passed to the analysis as one frozen `AnalysisSettings` (`sim.config.analysis_settings`;
`resonances/resonance/analysis/settings.py`). Shared rules have shared names (`LIBRATION_*`, `FILTER_*`,
`SEPARATRIX_*`, `CONFIDENCE_*`), a type's own rules the prefix `MMR_`, `SECULAR_` or `ZLK_`. *Convention* marks an
agreed threshold, not derived from physics: a decision within `CONFIDENCE_THIN_BAND` of it is a thin margin
([Confidence](physics/confidence.md)). The rule names (L4, R0, G2c, …) are those of the
[full reference](libration-analysis.md). The dry-run keys change no verdict; its step rules reuse the analysis' own
(cutoff, `MMR_STEP_RESOLVED_DEG`, `MMR_RESAMPLE_STEPS_PER_CYCLE`, `MMR_JITTER_MAX_RAD`).

The tables below are generated from the code (`uv run python docs/tools/config_table.py`).

<!-- settings-table:begin (generated by docs/tools/config_table.py, do not edit) -->

### Filter and cutoff

| key | default | meaning |
|---|---|---|
| `FILTER_ORDER` | `2` | Butterworth order N of the sos low-pass; filtfilt gain G(P) = 1/(1 + (tan(pi dt/P)/tan(pi dt/P_c))^(2N)) (-> 1/(1 + (P_c/P)^(2N)) for P_c >> dt) |
| `FILTER_EDGE_TRIM_CUTOFFS` | `1.0` | turning points, passes and portrait use [t0 + x P_c, t1 - x P_c]: filtfilt edges spread ~ +-P_c |
| `FILTER_MIN_LIBRATION_CUTOFFS` | `2.0` | the floor term of the cutoff is the shortest libration / this (MMR: P_floor, secular/ZLK: P_min): where it sets P_c, that libration keeps G = 1/(1 + this^-4) = 0.94; secular/ZLK: a measured libration period below this many P_c sets filter_too_slow (MMR: G(P_cyc) below MMR_L8_GAIN_FULL); shape checks become not_measured |
| `FILTER_GRID_RTOL` | `0.001` | a time grid is uniform for the filter when every step is within this fraction of the median step (REBOUND rounding) |
| `MMR_FILTER_SYNODIC_FACTOR` | `1.5` | P_c >= this x the longest synodic period with the planets of the angle (leak <= 0.5 deg) |
| `MMR_LIBRATION_PERIOD_MIN_YR` | `150.0` | shortest libration period of a non-co-orbital MMR, a convention (not a physical minimum: Hildas ~260 yr; the forced waves of 52356 and 23240 are 115-121 yr): faster 'episodes' are forced waves and the TP prominence covers what sigma_f holds faster than this (0 = off); co-orbitals use MMR_COORBITAL_PERIOD_FLOOR_FACTOR [yr] *Convention.* |
| `MMR_COORBITAL_PERIOD_FLOOR_FACTOR` | `0.5` | a co-orbital angle (1:1 with its planet) librates no faster than this x the linear tadpole period P_orb / sqrt(27 mu / 4) of the planet (Jupiter 148 yr, Earth 222 yr, Mars 1274 yr); wide horseshoes are faster (Earth horseshoes of the benchmark down to 0.65 of it); its P_c is the floor / FILTER_MIN_LIBRATION_CUTOFFS (0 = off) *Convention.* |
| `MMR_COORBITAL_QS_FLOOR` | `True` | rule G1q: a prograde 1:1 angle whose orbit-averaged angle stays inside the quasi-satellite well of the averaged potential for >= LIBRATION_MIN_TURNING_POINTS / 2 of its small-amplitude periods P_QS, and that has no libration over the record, is analysed again with the floor MMR_COORBITAL_PERIOD_FLOOR_FACTOR x P_QS (episodes centred outside the well keep the tadpole floor); kept when that pass finds a libration in the well *Convention.* |
| `MMR_COORBITAL_QS_WELL_SHARE` | `0.5` | rule G1q: an episode counts as centred in the quasi-satellite well (and takes its floor) only where the averaged potential has that well on at least this share of the episode's samples; otherwise it keeps the tadpole floor *Convention.* |
| `MMR_COORBITAL_QS_MAX_BLOCKS` | `4000` | rule G1q (cost): the 1:1 force law of the quasi-satellite pass reads the elements in blocks of P_QS,min (not longer than the angle's own P_c), but in no more than this many over the record (longer blocks then): each block may need its own table of the averaged potential (2010 CG236 over 20 kyr: 3774 blocks of 5.3 yr, 45 s; 1000 blocks of 20 yr fail its energy integral, R^2 0.11) |
| `MMR_FILTER_GUARD_TP_ABS` | `2` | x2 guard: the TP counts at P_c and 2 P_c on their common interval may differ by this many (one TP at each end) or by MMR_FILTER_GUARD_TP_REL of the larger count; more is a change (filter_sensitive) *Convention.* |
| `MMR_FILTER_GUARD_TP_REL` | `0.1` | x2 guard: relative tolerance of the TP counts (see MMR_FILTER_GUARD_TP_ABS) *Convention.* |
| `MMR_FILTER_GUARD_BOUNDARY_PERIODS` | `1.0` | x2 guard: the decisive episode (else any accepted episode) whose start or end moves by more than this many libration periods is a change *Convention.* |
| `MMR_FILTER_GUARD_H_TOL` | `0.1` | x2 guard: the median cycle energy h (separatrix 1) changing by more than this x max(1, the smaller \|h\|), or across 1, is a change (relative far above the separatrix: h ~ 1e4-1e7 of a far circulation moves by < 1 %) *Convention.* |
| `MMR_FILTER_GUARD_AMP_REL` | `0.2` | x2 guard: the largest half-amplitude of the decisive episode against its counterpart at 2 P_c (else of the episode with the most TPs) changing by more than this fraction is a change; energy and shape are compared only when the gain at P_cyc of the 2 P_c filter is >= MMR_L8_GAIN_FULL (else not checked) *Convention.* |
| `MMR_FILTER_CUTOFF_YR` | empty | explicit cutoff period overriding the rule (empty = rule) [yr] |
| `MMR_RIPPLE_REMOVE` | `True` | subtract sum_j k_j dlambda_j (great inequality) from the angle when planets are saved |
| `MMR_RIPPLE_BAND_YR` | `300.0,3000.0` | period band of the diagnostic correlation of the subtracted ripple with the angle (flt_ripple_corr) [yr] |
| `MMR_RIPPLE_RECONSTRUCTION_MAX_DEG` | `1.0` | the ripple is subtracted when the saved angle equals the combination of the saved longitudes within this [deg] |

### MMR sampling: alias branches and the unwrap

| key | default | meaning |
|---|---|---|
| `MMR_UNWRAP` | `predictive` | 'predictive' = unwrap by the rate model from a; 'numpy' = np.unwrap (diagnostic) |
| `MMR_UNWRAP_MAX_RESID_P99_DEG` | `45.0` | 99th percentile of the per-step model error of the lambda part for a stitchable record [deg] |
| `MMR_UNWRAP_MAX_RESID_DEG` | `135.0` | largest single per-step model error of the lambda part (pi/4 below pi) [deg] |
| `MMR_UNWRAP_SUSPECT_RESID_DEG` | `90.0` | a step with \|residual\| above this is unwrap_step_suspect [deg] |
| `MMR_VARPI_REF_WINDOW_YR` | `5000.0` | window of the varpi slope used as the slow term of the rate model [yr] |
| `MMR_JITTER_MAX_RAD` | `0.785` | \|p\| de_sp/e above this (pi/4) = jitter_low_e (the osculating wobble of p varpi: a flag and a cap <= medium); undefined_phase also needs varpi itself undefined (MMR_VARPI_UNDEFINED_MAX_SHARE); locally \|p\| s/\|z_f\| = phase_undefined_local [rad] |
| `MMR_VARPI_UNDEFINED_MAX_SHARE` | `0.5` | undefined_phase needs varpi's own (p = 1) local scatter s/\|z_f\| above MMR_JITTER_MAX_RAD on more than this share of the measured record (not measured: undefined_phase as before, jitter_low_e and the full angle failing) *Convention.* |
| `MMR_JITTER_BAND_YR` | `50.0` | periods shorter than this make the short-period e scatter de_sp; the band of the slow vector z_f = <e exp(i varpi)> [yr] |
| `MMR_STEP_RESOLVED_DEG` | `90.0` | step verdict resolved: \|nu\| dt <= this [deg] |
| `MMR_FAR_FROM_RES_FACTOR` | `10.0` | far_from_resonance: \|<a> - a*\| >= this x the a scale of the record, max(half the a range, \|a(lambda_dot) - <a>\|): a libration about a* needs the slow a to reach a* (no strength enters: mu e^q is not a bound) |
| `MMR_AXIS_RANGE_PERCENTILE` | `1.0` | the a range of the rate model (nu_lo, nu_hi, alias branches, far ratio) is the [x, 100 - x] percentiles of a [%] |
| `MMR_PLANET_UNWRAP_MAX_STEP_DEG` | `90.0` | a saved planet's n_j is the slope of its unwrapped lambda_j only when no step errs from the table rate by more than this; otherwise the table value [deg] |
| `MMR_ALIAS_NUMPY_REV_TOL` | `0.5` | alias_np_unwrap_differs: predictive and np.unwrap revolutions differ by more [rev] |
| `MMR_ALIASED_LINE_HARMONICS` | `3` | harmonics m of synodic waves checked for folding into [P_cyc/2, 2P_cyc] |
| `MMR_ALIASED_LINE_BAND_FACTOR` | `2.0` | a folded synodic line is reported when its period is within [P_cyc/x, x P_cyc] |
| `MMR_RESAMPLE_STEPS_YR` | `0.05,0.1,0.2,0.25,0.5,1.0,2.0,2.5,5.0,10.0` | round values a resample step (and smp_fold_free_step_yr) is chosen from [yr] |
| `MMR_RESAMPLE_STEPS_PER_CYCLE` | `20.0` | recommended step <= P_cyc / this (an undersampled record and the dry-run: the shortest libration P_floor / this) |
| `MMR_EXTEND_MIN_SPAN_FACTOR` | `2.0` | an extension asks for at least this x the record length T (the total length from the start epoch) |
| `MMR_NEXT_CHECK_SPAN_FACTOR` | `4.0` | insufficient_time/extend: the next check is round_up(max(f_min T, min(N, this x T))), N the estimate from the period (base_extend_to_years); a budget, not a sufficiency estimate: the record is analysed again after it *Convention.* |

### Geometry: turning points, centre, episodes, jumps

| key | default | meaning |
|---|---|---|
| `LIBRATION_MIN_TURNING_POINTS` | `3` | turning points of one full cycle: an episode needs this many; a secular moving centre too |
| `LIBRATION_NEAR_EPISODE_TP_DEFICIT` | `1` | a run that would be a librating episode but for its count (half-cycles below pi, no over-the-top neighbour, period not below the shortest libration) and lacks at most this many turning points is a near-episode: a verdict without libration gets flag near_episode, confidence <= medium; 0 = off |
| `LIBRATION_COVERAGE_PERIODS` | `1.0` | an episode covers a record edge when it starts (ends) within this many periods of it, or when no turning point is missing at the edge of the analysis window (that gap not longer than the longest TP-to-TP step of the episode, nor than this many periods) and the stretch the filter trims off that edge is within this many periods |
| `LIBRATION_UNION_GAP_PERIODS` | `1.0` | MMR: two consecutive librating episodes join into one libration over the record (G2, `covering_union`) only when the gap between them is at most this x (P_left + P_right): each side misses at most its own period, the symmetric reading of LIBRATION_COVERAGE_PERIODS at an edge *Convention.* |
| `LIBRATION_EPISODE_GAP_FACTOR` | `3.0` | a gap between turning points above this x the median half-period splits a run |
| `LIBRATION_REGULAR_PERIOD_RATIO` | `2.0` | a run whose centre moves is judged again on its longest stretch of full periods <= this x their median (a slow edge cycle is a change of regime, not part of the stationary libration), and may drop one edge half-cycle slower than this x the median half-period |
| `LIBRATION_EDGE_ARC_REV` | `1.0` | MMR: an exit (entry) is observed, not suspected, when the orbit-averaged raw angle runs at least this many turns away from the centre from the last turning point of the last episode to the record end (from the record start to the first one) and moves monotonically there; that edge is then not covered [rev] *Convention.* |
| `LIBRATION_RETURN_TOL` | `0.05` | MMR: two opposite turning points are one closed cycle when sigma_f crosses their mid-level before the first and after the second in the same direction, a returning to within this x its swing there, the rate to within this fraction, and sigma_f crossing the mid-level between them as fast as at the first crossing, the other way (closure by phase return; 2016 CA138: 0.08 %, 0.2 %, 0.1 %) *Convention.* |
| `LIBRATION_RETURN_PERIOD_TOL` | `0.25` | MMR: the phase return also needs (t_b - t_a) / (2 x the time between the two turning points) within 1 +- this (the quarter-period symmetry of a cycle; a circulation creeping back between two turns gives ~0.5; 2016 CA138: 1.00) *Convention.* |
| `LIBRATION_RETURN_IDENTITY_CORR_MIN` | `0.9` | MMR: the phase return also needs corr(sigma_dot, sign(c) (a - mean a)) >= this between the two crossings (a kinematic back-swing with a flat a is no cycle); the value of MMR_IDENTITY_CORR_MIN |
| `LIBRATION_RETURN_LEVELS` | `19` | MMR: the phase return of two opposite turning points is tried at the mid-level and at the levels k / (n + 1) of their swing, k = 1..n, with the same conditions, a level counting only inside a band of closing levels (LIBRATION_RETURN_MID_BAND_LEVELS for the mid-level, LIBRATION_RETURN_BAND_LEVELS for another); the mid-level is kept when it counts, else the best match (the mid-level is a convention, any level is a section the cycle crosses twice; 164207 never crosses its mid-level before the horseshoe turn); 0 or 1 = the mid-level only, alone *Convention.* |
| `LIBRATION_RETURN_BAND_LEVELS` | `2` | MMR: a level other than the mid-level closes a cycle by its phase return only inside at least this many consecutive closing levels of LIBRATION_RETURN_LEVELS (the mismatches change continuously with the level, a real return holds over a band of sections; 2025 RB7 at 1 kyr closes at 0.20-0.25, 164207 at 0.15-0.20) *Convention.* |
| `LIBRATION_RETURN_MID_BAND_LEVELS` | `2` | MMR: the mid-level closes a cycle by its phase return inside at least this many consecutive closing levels of LIBRATION_RETURN_LEVELS (2: one closing neighbour), counted over the observable levels only: a level whose crossing may lie beyond the record neither closes nor breaks the band, so a mid-level without an observable neighbour closes alone; 1 = the mid-level alone *Convention.* |
| `LIBRATION_RETURN_CYCLE_CAPS_MEDIUM` | `True` | MMR: a verdict that rests on a cycle closed only by its phase return (two turning points) has confidence <= medium *Convention.* |
| `LIBRATION_REGIME_CUTS` | `True` | MMR: a run whose centre moves is cut at a centre step between two measured regimes or at a stall of its slow centre (F2, K3; 524522 1V-1, 385266 16N-27); off: such a run stays one rejected run *Convention.* |
| `LIBRATION_STALL_MIN_CYCLES` | `3` | MMR: a run whose centre moves is cut at a stall of its slow centre: at least this many independent full cycles whose midpoints move less per period than the cycles swing (385266 16N-27, 55-95 kyr); a steady drift has none *Convention.* |
| `LIBRATION_CENTRE_CYCLES` | `3.0` | the centre is the angle smoothed over this many cycles |
| `LIBRATION_CENTRE_TP_PROMINENCE_DEG` | `10.0` | floor of the prominence of a centre turn (4x the centre noise A/82) [deg] |
| `LIBRATION_CENTRE_TP_FACTOR` | `2.0` | a centre turn also needs this x the turning-point prominence |
| `LIBRATION_CENTRE_SCALE_TP_AT_CENTRE` | `True` | MMR G2c: the analysis at the scale of the centre takes its turning points at the centre-turn prominence p_c (MMR_TP_PROMINENCE_FLOOR_DEG raised to p_c there): the turns the precheck that qualified the scale reads: a swing below p_c is no turn of the centre there (residue of the fast component W_c attenuates, or a libration too small for this scale, which the fine scale judges); off: the floor of the fine scale |
| `LIBRATION_CENTRE_RETURNS_MODULATED` | `3` | a bounded centre turning back this many times is modulated, not drift |
| `LIBRATION_CENTRE_MIN_ESTIMATES` | `2` | a stationary centre needs this many independent full cycles (no shared half-cycle; every band overlapping); with fewer (3 TPs = one cycle, one centre) the episode is accepted but its centre is not measured (L6 thin) |
| `LIBRATION_MODULATED_RANGE_MAX_RAD` | `3.141592653589793` | modulated: range of the centre below this [rad] |
| `LIBRATION_SLIP_TOLERANCE` | `0.2` | a slip is a step of 2 pi (1 +- this) of the oscillation midpoint *Convention.* |
| `LIBRATION_SLIP_MIN_DURATION_T0` | `0.5` | a slip faster than this x T0 is an unwrap error (the angle cannot pass the top faster) |
| `LIBRATION_PATTERN_A_MIN_SLIPS` | `2` | pattern A needs at least this many slips |
| `LIBRATION_PATTERN_A_MIN_AMP_DEG` | `60.0` | without a measured zone, pattern-A pieces need this amplitude (J/J_s ~ 0.3) [deg] *Convention.* |
| `LIBRATION_STALL_PHASE_TOL_DEG` | `30.0` | pattern B: rate minimum within this of the potential top [deg] |
| `LIBRATION_STALL_REV_FRACTION` | `0.75` | pattern B: this fraction of revolutions has the rate minimum at the top *Convention.* |
| `LIBRATION_STALL_RATE_TOL` | `0.2` | \|r_min - r_pend(h)\| above this = stall_rate_inconsistent (flag only) |
| `LIBRATION_DRIFT_MIN_WINDOW_FRACTION` | `0.5` | drift: rejected centre_moves runs cover at least this fraction of the window |
| `LIBRATION_CIRCULATION_NET_REV` | `2.0` | G4b: the angle circulates when the slow centre advances by at least this many revolutions net over the window (\|c(t1) - c(t0)\| / 2pi), even when its largest monotone pass stays below one (a staircase with small step-backs); also in the angle word of K2 *Convention.* |
| `LIBRATION_JUMP_NOISE_FACTOR` | `8.0` | a jump candidate exceeds this many robust std of the slow-variable increments |
| `LIBRATION_JUMP_DH_MIN` | `0.1` | a jump changes the cycle energy h by more than max(2 sigma_h, this) |
| `LIBRATION_JUMP_CYCLES` | `2` | cycles compared on each side of a jump candidate |
| `LIBRATION_ACTION_STEP` | `True` | G2a (MMR, not 1:1): an accepted run whose half-amplitudes step between two separated bands, with the swing of a and the cycle energy changing the same way within one half-cycle, is cut there as at a confirmed jump of a (an impulse: a close encounter puts the orbit on another energy level; 2003 TX9 3E+2J-10 at the Mars encounters of 6.5-6.9 kyr); off: such a run stays one episode *Convention.* |
| `LIBRATION_ACTION_STEP_MIN_HALF_CYCLES` | `2` | G2a: half-cycles on each side of the transitional half-cycle (2 = one closed cycle) |
| `LIBRATION_ACTION_STEP_SEPARATION` | `2.0` | G2a: the two bands of half-amplitudes are separated: the smallest on the larger side over the largest on the smaller side is at least this (an adiabatic decay needs an e-folding of 1.5 cycles to reach 2 across one excluded half-cycle) |
| `LIBRATION_ACTION_STEP_RATIO` | `3.0` | G2a: ratio of the median half-amplitudes of the two sides |
| `LIBRATION_ACTION_STEP_A_RATIO` | `1.5` | G2a: the swing of a (std of a_f on each side at the record's cutoff) changes the same way by at least this factor (delta a ~ sigma_dot; 74729 1J+6S-1: a sigma step that a does not show, ratio 0.37) |
| `LIBRATION_ACTION_STEP_CYCLES` | `1` | G2a: cycles of the energy analysis on each side that confirm the step (\|dh\| > max(2 sigma_h, LIBRATION_JUMP_DH_MIN), the sign of the amplitude change); fewer than LIBRATION_JUMP_CYCLES on a side makes the split provisional (cap action_step_provisional, confidence <= medium) |
| `LIBRATION_ACTION_STEP_PERIOD_TOL` | `0.2` | G2a test 7: every disjoint cycle of both sides has T_obs/T_pred within this of 1, T_pred the period of the loop of the cycle's own energy, in the well that holds it, of V(sigma) fitted (as V_E) on the cycles of the larger swings (held out: each of them in V of the others); and the two sides' median periods differ by more than this (else one forcing frequency explains both: in a nearly harmonic V every loop has one period). As SEPARATRIX_PERIOD_TOL for R0 |
| `LIBRATION_ACTION_STEP_TP_TOL` | `0.2` | G2a test 7: the observed extremes of sigma of every cycle lie within this x its half-swing of the turning points of the loop of its energy (a swing off the minimum of V is no libration in V). In the harmonic limit the half-width at a given energy scales as the period, so this equals LIBRATION_ACTION_STEP_PERIOD_TOL |
| `LIBRATION_ACTION_STEP_ENERGY_DRIFT` | `0.5` | G2a test 7: on each side the cycle energies in V spread by at most this x the step of the energy between the sides (between impulses the energy of a libration stays put; at 0.5 the two bands are apart by at least half the step, as the half-amplitude bands of LIBRATION_ACTION_STEP_SEPARATION) |
| `LIBRATION_ACTION_STEP_CUTOFFS` | `0.075,0.1,0.15` | G2a test 7 holds at every shape cutoff C = f x the shorter side period of this list, with the same number of wells (the numbers reported at the middle one): a factor 2, lighter than SEPARATRIX_CUTOFF_CYCLES, since the test reads turning points and a second well, which live in the high harmonics of a swing across a barrier (a double well integrated and kicked: T_obs/T_pred 0.994 at 0.1, 0.911 at 0.2, the second well lost at 0.3); the rate's noise sets the floor (V_E R^2 >= SEPARATRIX_VE_R2_MIN). A well or a law one cutoff alone gives is the filter's |
| `LIBRATION_ACTION_STEP_MAX_PERIODS` | `1.0` | G2a: the transitional half-cycle lasts at most this many of the shorter median full period of the two sides |
| `LIBRATION_JUMP_TAU_CUTOFFS` | `2.0` | jump locator baseline tau >= this x P_c |
| `LIBRATION_JUMP_TAU_STEPS` | `5.0` | jump locator baseline tau >= this x the output step |
| `LIBRATION_JUMP_EDGE_PERIODS` | `1.0` | a jump candidate needs max(tau, this x P_cyc) of the window on each side to be confirmed; closer to an edge it is kept as jump_at_edge and cuts nothing |
| `MMR_TP_PROMINENCE_FLOOR_DEG` | `1.0` | turning-point prominence floor: twice the synodic leak after the filter [deg] |
| `MMR_TP_RIPPLE_FACTOR` | `4.0` | prominence >= this x the unremoved ripple half-amplitude or the part of sigma_f faster than MMR_LIBRATION_PERIOD_MIN_YR (95th percentile, the image of the libration runs through the two filters taken out first; a wave of half-amplitude r gives turns <= 2r) |

### Energy and the separatrix zone

| key | default | meaning |
|---|---|---|
| `SEPARATRIX_HARMONICS_MAX` | `2` | harmonics of V(sigma): asymmetric external 1:N need 2 |
| `SEPARATRIX_HARMONIC_GAIN` | `0.2` | the second harmonic is kept when R^2 of V_E grows by this |
| `SEPARATRIX_PERIOD_TOL` | `0.2` | period-energy law: \|median T_obs/T_pend - 1\| <= this (pendula 0.9-1.1, kinematics 0.61) |
| `SEPARATRIX_PHASE_COHERENCE` | `0.9` | fixed well: coherence of the stable point over windows (potential 0.95-0.99, kinematics 0.71) |
| `SEPARATRIX_VEVF_R_MIN` | `0.9` | R0 with the slow variable: correlation of V_E and V_F in a window |
| `SEPARATRIX_VEVF_FRACTION` | `0.666` | R0 with the slow variable: fraction of windows where V_E ~ V_F |
| `SEPARATRIX_SIGMA_MARGIN` | `2.0` | margin in sigma_h when a cycle is put into a zone class |
| `SEPARATRIX_SIGMA_FLOOR` | `0.02` | floor of the cycle-to-cycle energy scatter sigma_h |
| `SEPARATRIX_TREND_SIGMA` | `3.0` | trend mechanism: slope of J/J_s over cycles significant at this many sigma |
| `SEPARATRIX_PERMUTATION_P_MAX` | `0.05` | pulsation mechanism: cyclic-shift p-value below this *Convention.* |
| `SEPARATRIX_LAYER_ADIABATICITY_MIN` | `0.5` | layer mechanism: crossings at an adiabaticity lambda above this are the chaotic layer's own (no adiabatic invariant) |
| `SEPARATRIX_NEAR_ZONE_WIDTHS` | `1.5` | near_separatrix_energy flag: revolutions within this many zone widths above the zone |
| `MMR_EXIT_AND_CAPTURE_LIFTS_UNEXPLAINED` | `True` | the mark exit_and_capture lifts the cap transient_unexplained: an angle that left its libration and came back is the transient the record shows even without a named mechanism *Convention.* |
| `MMR_COORBITAL_LIFTS_UNEXPLAINED` | `True` | a 1:1 angle gets no cap transient_unexplained: the pendulum separatrix model that names a mechanism does not apply to it, its crossings are close encounters *Convention.* |
| `SEPARATRIX_CUTOFF_CYCLES` | `0.2` | V_E, V_F, the modulation and the zone read sigma and a smoothed at max(P_c, this x P_cyc) (G(P_cyc) 0.998) |
| `SEPARATRIX_VE_R2_MIN` | `0.7` | R0 is measured only when V_E explains this share of sigma_dot^2 within the cycles *Convention.* |
| `MMR_ZONE_METHOD` | `numeric` | zone the separatrix reaches: 'numeric' (modulated pendulum), 'eq_only', 'adiabatic' |
| `MMR_SEPARATRIX_AVGPOT_TP_MAX` | `40` | two-body angle with its planet saved: the energy of each turning point is read in the averaged potential at its own time, h = (<D>(sigma_TP) - <D>_min) / (<D>_barrier - <D>_min); at most this many evenly chosen turning points are evaluated (cost: one profile each, ~0.02 s), the others interpolated in time |
| `MMR_ZONE_J_MIN` | `0.1` | smallest initial action J0/<J_s> of the zone grid |
| `MMR_ZONE_J_STEP` | `0.025` | step of the zone grid in J0/<J_s> and in h0 |
| `MMR_ZONE_H_MAX` | `2.0` | largest initial energy h0 on the rotation side of the zone grid |
| `MMR_ZONE_PHASES` | `16` | start phases per modulation period in the zone grid |
| `MMR_ZONE_RTOL` | `1e-09` | relative tolerance of the DOP853 zone trajectories |
| `MMR_ZONE_ADIABATIC_LAMBDA_MAX` | `0.1` | below this adiabaticity the numeric zone is checked against the adiabatic edge |
| `MMR_ZONE_ADIABATIC_TOL` | `0.05` | numeric vs adiabatic J_lo differing by more = zone_numeric_mismatch |

### MMR physics: response of a, force law, identity, kinematics

| key | default | meaning |
|---|---|---|
| `MMR_A_RESPONSE_MIN_AMP_DEG` | `45.0` | below this amplitude sine and line differ < 10% (A^2/6): harmonic force law not_measured (not the 1:1 law) [deg] *Convention.* |
| `MMR_FORCE_WINDOW_CYCLES` | `3.5` | force-law fit window in cycles (phase defined, force ~ constant) |
| `MMR_FORCE_WINDOW_STEP` | `0.5` | step of the force-law windows as a fraction of the window |
| `MMR_FORCE_MIN_WINDOWS` | `2` | fewer force-law windows = not_measured |
| `MMR_FORCE_R2_FULL` | `0.7` | force law passes with R^2 >= this in both halves *Convention.* |
| `MMR_FORCE_R2_FAIL` | `0.4` | force law fails with R^2 below this under both laws *Convention.* |
| `MMR_FORCE_PHASE_TOL_DEG` | `30.0` | floor of the expected spread of phi0 (the column rsp_phi0_tol_deg only) [deg] |
| `MMR_FORCE_MULTI_GAIN` | `0.1` | the multiplet law is taken when its R^2 exceeds the single one by this |
| `MMR_FORCE_COEF_COH_MIN` | `0.8` | multiplet law: coherence of the leading complex coefficients over windows |
| `MMR_FORCE_POWER_SHARE` | `0.8` | the leading multiplet terms carry this share of the fitted power |
| `MMR_FORCE_CONTAMINATION` | `lattice` | 'lattice' = a measured harmonic force law is not_measured (contamination) when a forced line of the angle's lattice lies within 1/(MMR_FORCE_WINDOW_CYCLES P) of 1/P (the fit cannot tell the two apart; it never makes a libration dynamic); 'off' |
| `MMR_RESPONSE_CUTOFF_CYCLES` | `0.1` | identity, force law, potentials use sigma, a smoothed at max(P_c, this x P_cyc): derivatives amplify what P_c leaks; 0.1 keeps the cycle and its first 5 harmonics (G(2 cutoffs) = 0.94, as FILTER_MIN_LIBRATION_CUTOFFS) and costs an exact pendulum R^2 < 0.01 up to 178 deg; 0.15-0.2 bend the law near the separatrix (F[sin sigma] != sin F[sigma]) |
| `MMR_RESPONSE_LINE_AMP_FRAC` | `0.5` | harmonic force law (L4): a line of a_dot (not a harmonic P_e/k of the libration) with an amplitude of at least this x the force K of the first fit, in the series the law reads, raises the response cutoff of that measurement (2004 TK14 31M-34: 64 yr at 0.9 K; 433 Eros 47M-44: 83 yr at 1.5 K) |
| `MMR_RESPONSE_LINE_MARGIN` | `1.5` | such a line raises the response cutoff to this x its period (G = 0.16 of the line left in a, 2nd-order filtfilt) |
| `MMR_RESPONSE_MAX_FRACTION` | `4.0` | the response cutoff raised by a line stays at or below the period of the measurement / this (P_e/4 keeps G = 0.996 of the cycle; a slower line is left in) |
| `MMR_RESPONSE_LINE_BAND_CUTOFFS` | `0.5` | lines of a_dot are searched from this x the base response cutoff up to P_e/2: a line faster than the cutoff leaks through it (2004 TK14: 64 yr under a 75-yr cutoff keeps 0.9 K); below half the cutoff the filter leaves < 6 % |
| `MMR_FORCE_NULL_SHIFTS` | `8` | null test of a force law whose cutoff a line raised: R^2 of the same law with a_dot shifted in time (not wrapped) by this many lags (m + 1/4) P_e and (m + 3/4) P_e, alternately forward and back, m >= 1 whole periods spread up to half the record; its largest value is the null R^2 (whole or half periods alone keep a steady libration in phase) |
| `MMR_FORCE_NULL_GAIN` | `0.2` | a force law whose cutoff a line raised passes only with R^2 at least this above the null R^2 (433 Eros 47M-44: 0.6-0.8 against 0.15-0.35) *Convention.* |
| `MMR_FORCE_HARMONICS_MAX` | `2` | a failed or thin harmonic force law (L4) is fitted again with up to this many harmonics of the angle, a_dot = sum_k [A_k sin k sigma + B_k cos k sigma] + C, the force of the V(sigma) R0 fits (as SEPARATRIX_HARMONICS_MAX; a double well or a flat bottom): taken when N >= 2 gains MMR_FORCE_MULTI_GAIN, its coefficients are coherent (MMR_FORCE_COEF_COH_MIN), it beats the shifted null by MMR_FORCE_NULL_GAIN and one half of the windows predicts the other with R^2 >= MMR_FORCE_R2_FAIL; it lifts L4, never lowers it; 1 = off |
| `MMR_SHORT_PERIOD_CAP_RATIO` | `0.0` | the record itself lifts the cap short_period_aliased only when the fast residual of sigma_f (p95, flt_fast_residual_deg) is at most this x the decisive episode's median half-amplitude and the short-period scatter of a passes MMR_SHORT_PERIOD_A_RATIO; 0 = never: the record cannot bound a line folded into a, only a finely sampled pilot measures it *Convention.* |
| `MMR_SHORT_PERIOD_CAP_CIRCULATION` | `False` | the cap short_period_aliased also on a circulation (angle word circulation, reason circulation or alias; kinematic_drift, no_resonant_potential and a circulation with forced-wave runs (F6, its failed identity reads a) keep it); off: a circulation reads neither the identity nor the force law, which the folded lines in a spoil, and the folded lines move sigma_f by <= 0.024 of a half-amplitude, so they cannot turn a libration into a circulation; the flag stays *Convention.* |
| `MMR_SHORT_PERIOD_A_RATIO` | `0.5` | flag a_short_period_scatter (a warning, no cap by itself) on a record whose step folds short periods: its short-period scatter of a (smp_da_short_period_au, periods below MMR_JITTER_BAND_YR) above this x the a half-excursion of the decisive episode (ep_a_excursion_meas_au); the folded lines reach the a-based checks (identity, force law) there. A proxy, not a bound: a line folded to centuries is nearly invisible to it. With MMR_SHORT_PERIOD_CAP_RATIO > 0 it also keeps the cap; the same comparison gates MMR_SHORT_PERIOD_A_UNMEASURED *Convention.* |
| `MMR_SHORT_PERIOD_A_UNMEASURED` | `True` | on a record whose step folds short periods (flag short_period_aliased), a failed force law (L4) and identity (L3), which read a, count as not_measured in the status tree and the confidence when the strength by direct averaging passes and the short-period scatter of a exceeds MMR_SHORT_PERIOD_A_RATIO x the decisive a half-excursion; a fail never becomes a pass and the cap stays *Convention.* |
| `MMR_SHORT_PERIOD_HARMONICS` | `2` | harmonics m / P (m = 1..this) of P_orb and of the P_syn of the angle's planets that the step may fold: a line folded slower than the angle's period floor (smp_folded_slowest_yr) sets the flag and the cap short_period_aliased also below the fundamental step min(P) / 2. Only m <= this is searched: P_orb 12 yr at 4.002 yr folds its third harmonic to 8 kyr and stays unflagged at 2; 0 = the fundamental step alone *Convention.* |
| `MMR_COORBITAL_GRID_POINTS` | `720` | co-orbital (1:1) law: points of the theta grid and of the trapezoidal average over the planet's longitude (720 = 0.5 deg; the average converges at 360, the grid interpolates the near-encounter peaks to ~5 % rms) |
| `MMR_COORBITAL_RECORD_TABLES_MB` | `140.0` | co-orbital law (memory): the tables of the averaged potential kept for the analysis of one record, in bytes of their arrays (17.3 kB each at 720 points: 140 MB hold 8101); a table beyond them is computed again when a later pass needs it (2010 CG236 over 20 kyr needs 7909, 137 MB, at 7 ms each). A bound in bytes holds at any MMR_COORBITAL_GRID_POINTS. It bounds this store only, not the process: the shared cache of 512 tables, the figure and the FFT workspaces come on top, and a process keeps the memory of its largest record (the benchmark tools recycle their workers, reclassify --tasks-per-worker) [MB] |
| `MMR_COORBITAL_SOFTENING_HILL` | `0.1` | co-orbital law: the direct term is softened at this many Hill radii a_P (mu/3)^(1/3), so that an exact orbit crossing stays finite (the averaging does not describe the motion inside the Hill sphere; at 1 r_H the fitted K / K_theory of close-approaching horseshoes rises by 10-20 %, at 0.1 it is within 2 % of no softening) |
| `MMR_COORBITAL_RETABULATE_E` | `0.005` | co-orbital law: a new table of the averaged potential when the eccentricity vector e exp(i omega) of the body or the planet moved by more than this |
| `MMR_COORBITAL_RETABULATE_INC_DEG` | `0.5` | co-orbital law: a new table when the inclination vector sin i exp(i Omega) of the body or the planet moved by more than sin of this [deg] |
| `MMR_COORBITAL_SCALE_RATIO_MAX` | `3.0` | co-orbital law passes only with the fitted force K / K_theory in [1/this, this] (K_theory = 1: the model holds the planet's mass; a negative K pushes a the wrong way) *Convention.* |
| `MMR_IDENTITY_SLOPE_TOL` | `0.2` | identity: \|slope/c - 1\| <= this |
| `MMR_IDENTITY_CORR_MIN` | `0.9` | identity: correlation of sigma_dot and a >= this |
| `MMR_FORCED_WAVE_LINE_ALONE` | `False` | an accepted episode whose period sits on a forced line of the lattice is a forced wave, no episode of the angle, even when its identity does not fail (the line alone); off: the line needs the failed identity, since a chance coincidence with a line happens for 6-17 % of the force laws *Convention.* |
| `MMR_B1_RATIO_MIN` | `0.1` | a carries < this of the needed amplitude: the oscillation sits in the varpi part |
| `MMR_B1_BAND_FACTOR` | `2.0` | B1 compares a and sigma_dot / \|c\| in the band [P/f, f P] around the cycle period P (the libration and its wandering period; the identity holds there harmonic by harmonic) *Convention.* |
| `MMR_KIN_CORR_MIN` | `0.9` | kinematic: slow sigma follows p varpi with correlation >= this *Convention.* |
| `MMR_KIN_SLOW_R2_MAX` | `0.2` | kinematic: slow a_dot answers sin sigma_slow with R^2 <= this *Convention.* |
| `MMR_KIN_MIN_POINTS` | `3` | independent points (length / centre window) needed for the kinematic correlation |
| `MMR_KIN_PERIOD_TOL` | `0.1` | origin_detail: the angle period matches the e-cycle, the 2 omega cycle or a planetary ripple within this fraction |
| `MMR_KIN_RIPPLE_SHARE_MAX` | `0.5` | kinematic drift of the slow centre (K2): the largest swing of a librating episode (2 x its largest half-amplitude) stays below this x the range of the slow centre, so the episodes are a ripple on the drift, not the libration (21028 7E-10M-6: 117 deg against 741 deg, 0.16; the Hilda 153 3J-2: 0.95) *Convention.* |
| `MMR_KINEMATIC_DRIFT_WORD` | `True` | a verdict non_resonant/kinematic_drift of K2 reports the angle word of its slow centre, not of the ripple the geometry reads as episodes: drift, or circulation when the centre passes a revolution in one direction (G4); off: the geometry word (episodic) stays *Convention.* |
| `MMR_KIN_DRIFT_SHIFT_RATIO` | `1.0` | a kinematic libration (step 3, B1 or the kinematic cycle) whose decisive episode moves its centre by more than this x its mean amplitude (L6) is decided as K2: non_resonant/kinematic_drift with the slow centre's word, drift or circulation *Convention.* |
| `MMR_B5_RATIO_FULL` | `2.0` | slow libration excluded with a full margin: centre rate >= this x the bound |
| `MMR_B5_EXACT_SIGMA` | `2.0` | exact commensurability: \|nu_c\| < this x its error |
| `MMR_C4_RATIO_FULL` | `1.5` | C4 full: <\|sigma_dot\|> / (2 omega0_K) >= this |

### MMR subresonances and the owner search

| key | default | meaning |
|---|---|---|
| `MMR_SUBRESONANCE_STRENGTH_REL_MIN` | `0.001` | test E': a member's d'Alembert factor prod x^\|c\| (x = e, e_P, sin i/2, sin i_P/2) is at least this x that of the strongest term of the same commensurability (its pendulum is then <= 1/sqrt(this) ~ 30 times slower) |
| `MMR_SUBRESONANCE_NODAL_INC_MIN_DEG` | `10.0` | members with node coefficients are enumerated when the median inclination to the invariable plane reaches this (their factor against the planar one of the same degree is (sin(i/2) / e)^2) [deg] |
| `MMR_SUBRESONANCE_RANGE_RATIO` | `0.5` | candidate: its centre range <= this x the range of the catalogue angle centre |
| `MMR_SUBRESONANCE_RATE_FLOOR_RAD_KYR` | `0.02` | candidate: \|centre rate\| <= max(2 sigma_rate, this) [rad/kyr] |
| `MMR_SUBRESONANCE_PASS_MAX_RAD` | `3.141592653589793` | candidate: largest monotone centre pass below this [rad] |
| `MMR_SUBRESONANCE_PHI0_COH_MIN` | `0.8` | test A: phi0 coherence of the force law relative to the subresonance |
| `MMR_SUBRESONANCE_PHI0_COH_GAIN` | `0.2` | test A: and higher than for the catalogue angle by this |
| `MMR_SUBRESONANCE_CENTRE_TOL_DEG` | `30.0` | test B: mean centre within this of 0 or pi [deg] |
| `MMR_SUBRESONANCE_CENTRE_E_MAX` | `0.3` | test B is skipped above this eccentricity |
| `MMR_SUBRESONANCE_CENTRE_INC_MAX_DEG` | `20.0` | test B is skipped above this inclination [deg] |
| `MMR_SUBRESONANCE_TIE_FRACTION` | `0.2` | test C: centre ranges within this fraction tie |
| `MMR_SUBRESONANCE_THEORY` | `True` | test D: compare with the composite multiplet theory (confidence only: D never rejects a subresonance) |
| `MMR_SUBRESONANCE_THEORY_RMS_DEG` | `20.0` | test D: RMS of the centre about pi - arg P(beta) [deg] |
| `MMR_SUBRESONANCE_THEORY_SINGLE_RATIO` | `0.5` | test D: and <= this x the RMS of the best rival single-member hypothesis (sigma0 in, the accepted member out) |
| `MMR_SUBRESONANCE_THEORY_COEF_FRACTION` | `0.2` | test D: the accepted one has the largest coefficient or within this of it |
| `MMR_SUBRESONANCE_PERIOD_CORR_MIN` | `0.8` | test D: correlation of the period with T0(t) F(A) |
| `MMR_SUBRESONANCE_PERIOD_RATIO_MAX` | `2.0` | test D: median observed/predicted period within this factor |
| `MMR_SUBRESONANCE_T0_RATIO_MAX` | `2.0` | test E: predicted T0 <= this x the observed period (is a resonance at all) |
| `MMR_SUBRESONANCE_T2_R2_MIN` | `0.4` | trigger T2: a responds (R^2 >= this)... |
| `MMR_SUBRESONANCE_T2_COH_MAX` | `0.7` | trigger T2:... but the phi0 coherence is below this |
| `MMR_SUBRESONANCE_T3_SIGMA` | `2.0` | support T3: drift rate matches a secular combination within this sigma |
| `MMR_SUBRESONANCE_T3_RATE_FRAC` | `0.25` | support T3: or within this fraction of the drift rate |
| `MMR_SUBRESONANCE_TOP_N` | `10` | length of the ranked list sub_top; subresonances written to candidates.csv in top mode |
| `MMR_SUBRESONANCE_CYCLE_PERIOD_RATIO` | `2.0` | the scan window is the longest stretch of the run whose full periods stay <= this x their median (the slow cycles at the run edges are near the separatrix of the modulated pendulum, where no centre is stationary) |
| `MMR_SUBRESONANCE_OMEGA_HARMONICS` | `2` | with the nodal members (inclination >= MMR_SUBRESONANCE_NODAL_INC_MIN_DEG) the members sigma +- 2 m omega (omega = varpi - Omega), m = 1..this, are scanned and listed whatever their degree (2003 UG22 10E-17+0+7: sigma + 2 omega has degree 11); 0 = none beyond the degree limit *Convention.* |
| `MMR_SUBRESONANCE_DALEMBERT_E_MAX` | `0.2` | the ranked list marks the d'Alembert factor unreliable above this eccentricity (display only, sub_strength_caveat) *Convention.* |
| `MMR_SUBRESONANCE_DALEMBERT_INC_MAX_DEG` | `20.0` | the ranked list marks the d'Alembert factor unreliable above this inclination to the invariable plane (display only) [deg] *Convention.* |
| `MMR_OWNER_SEARCH` | `auto` | 'auto' = search the owner by the rules (always when the force law is not_measured); 'off' |
| `MMR_OWNER_PLANETS` | `Venus,Earth,Mars,Jupiter,Saturn,Uranus,Neptune` | planets of the owner candidates |
| `MMR_OWNER_COEF_MAX` | `60` | largest planet coefficient of a two-body owner |
| `MMR_OWNER_ORDER_MAX` | `20` | largest order of a two-body owner (owners of 7-17 were seen) |
| `MMR_OWNER_COEF_MAX_3B` | `12` | largest \|coefficient\| of a three-body owner |
| `MMR_OWNER_RATE_FACTOR` | `1.5` | owner candidate: \|rate\| <= this x 2 pi / min(P_a, P_cyc) somewhere in the a range of the record |
| `MMR_OWNER_R2_MIN` | `0.7` | owner accepted: median R^2 of a_dot on its angle in the windows of one regime (0 false owners on 29 measurable own librations at any value 0.5-0.95 with the ripple test; GK3 and 117306 are lost at 0.9, YG4 at 0.95) *Convention.* |
| `MMR_OWNER_PHI0_COH_MIN` | `0.8` | owner accepted: phi0 coherence |
| `MMR_OWNER_R2_GAIN` | `0.2` | owner accepted (route r2): R^2 above that of the angle itself by this; routes phase and potential are the alternatives |
| `MMR_OWNER_PHI0_COH_GAIN` | `0.2` | owner accepted: its phi0 coherence C_B >= L_owner + this, L_owner = \|<exp(i Psi_w / v)>\| the most it could have if sigma's own term drove a, while sigma's C_sigma < L_sigma + this (L_sigma = \|<exp(i Psi_w / u)>\|, corrected for the owner's slips); L_owner > 1 - this: phase_unresolved (at small amplitude both angles fit a_dot by the identity sigma = (v/u) phi_o + Psi/u) |
| `MMR_OWNER_BAND_FACTOR` | `2.0` | owner test: a_dot band [P/f, f P] around the a-wave period P; P is searched within a factor f of P_cyc; force-law windows keep f P (x filter_edge_trim_cutoffs) from the record ends |
| `MMR_OWNER_RIPPLE_FACTOR` | `2.0` | owner rejected as a ripple: sigma below the owner's band turns back in the regime's windows with a prominence of at least this x the band's peak-to-peak, more than a circulating owner can move it |
| `MMR_OWNER_PSI_SHARE_MAX` | `0.1` | owner accepted: fast share of the remainder psi below this |
| `MMR_OWNER_STRENGTH_MARGIN` | `10.0` | order-of-magnitude band of K_o / (2 \|k\| n a prod mu): orders the owner candidates and qualifies owner_probable; never accepts or rejects an owner |
| `MMR_OWNER_TOP_N` | `5` | owner candidates written to candidates.csv |
| `MMR_OWNER_LAW` | `multiplet` | owner phase test: 'multiplet' = where the single phi0 drifts, fit phi_o + j (varpi - varpi_P), j from 0 to -p, with constant coefficients (R^2 >= MMR_OWNER_R2_MIN, coefficient coherence >= MMR_OWNER_PHI0_COH_MIN); 'single' = phi0 only |
| `MMR_OWNER_COVER_MIN` | `0.5` | an accepted owner takes sigma's oscillation (origin induced) only when its intervals (its passing regimes' windows and the slips between them) cover at least this fraction of sigma's decisive episode (for a libration over the record or no episode: the analysis window); else origin none, flag owner_outside_episode *Convention.* |
| `MMR_OWNER_POTENTIAL` | `auto` | owner route 'potential': 'auto' = a two-body candidate with the planet's saved elements may be accepted when its directly averaged force (2 k_o / (n a)) G m_P d<D>/dphi_o, band-passed as a, explains a_dot (R^2 >= MMR_OWNER_R2_MIN, every window scale s in [1/MMR_OWNER_POTENTIAL_SCALE_MAX, MMR_OWNER_POTENTIAL_SCALE_MAX]); 'off' |
| `MMR_OWNER_POTENTIAL_SCALE_MAX` | `2.0` | owner route 'potential': the fitted amplitude scale s of the predicted a_dot lies within [1/x, x] in every window (x = 2: the owner supplies at least half the force; the model error of the frozen average is ~0.1) *Convention.* |
| `MMR_OWNER_POTENTIAL_SCALE_FULL` | `1.5` | margin W of an owner accepted by 'potential': full when its scale range lies within [1/x, x] and R^2 >= MMR_W_R2_FULL, else thin *Convention.* |
| `MMR_OWNER_POTENTIAL_SNAPSHOTS` | `5` | owner route 'potential': the averaged force is evaluated in at most this many of the regime's windows, evenly chosen |
| `MMR_OWNER_POTENTIAL_TOP_N` | `3` | owner route 'potential' (cost): tried on the best this many candidates by R^2 that pass R^2, psi share and the ripple test, when no candidate was accepted by R^2 or phase (the accepted owner gets it too, for its columns) |
| `MMR_OWNER_POTENTIAL_MAX_KERNELS` | `100000000.0` | owner route 'potential' (cost): one profile <D>(phi_o) on L nodes per turn evaluates \|k_o\| L^2 kernels (L values of phi_o, \|k_o\| L nodes each); skipped (reason potential_cost) when the next L would exceed this (about 5 ns per kernel: 1e8 is ~0.5 s per profile, at most MMR_OWNER_POTENTIAL_SNAPSHOTS x MMR_OWNER_POTENTIAL_TOP_N profiles per row) |
| `MMR_OWNER_POTENTIAL_CONFIRM` | `False` | a two-body owner accepted by R^2 or phase must also agree with its directly averaged force where that is measured: every window K/K_direct within [1/MMR_OWNER_POTENTIAL_SCALE_MAX, MMR_OWNER_POTENTIAL_SCALE_MAX] and \|phi0 - phi0_direct\| <= MMR_OWNER_POTENTIAL_PHASE_MAX_DEG, else rejected (potential_unconfirmed); off = columns only *Convention.* |
| `MMR_OWNER_POTENTIAL_PHASE_MAX_DEG` | `20.0` | MMR_OWNER_POTENTIAL_CONFIRM: the largest \|phi0 - phi0_direct\| of the fitted and the directly averaged force [deg] *Convention.* |
| `MMR_OWNER_CAPTURE_LOWPASS_PERIODS` | `3.0` | second capture: Psi = u sigma - v phi_o is low-passed at this many owner-band periods P (its slow part: the owner's wave cancels in Psi, what is left is the separation of the two resonances); the filter edges are trimmed |
| `MMR_OWNER_CAPTURE_SWEEP_DEG` | `180.0` | second capture (sigma held by its own term while the owner pins phi_o) is excluded when x = Psi_slow / u sweeps at least this and x_dot^2 is not a function of x (R^2 on [1, cos x, sin x, cos 2x, sin 2x] < MMR_OWNER_R2_MIN); otherwise unresolved (flag second_capture_unresolved, confidence <= medium) [deg] *Convention.* |
| `MMR_OWNER_CAPTURE_HARMONICS` | `2` | second capture: the energy integral x_dot^2 = 2E - 2V(x) is fitted with the harmonics 1.. this of x (V periodic in x; 2 = [1, cos x, sin x, cos 2x, sin 2x], R5 §1.7) *Convention.* |
| `MMR_OWNER_CAPTURE_SAMPLES_PER_TERM` | `3` | second capture: the energy fit needs at least this many samples per fitted term (after the filter edges are trimmed), else the block is not measured |
| `MMR_OWNER_POTENTIAL_MODEL_ERROR` | `0.1` | display only (the figure's owner lines): the model error of the frozen first-order average; a bound on sigma's own force max \|s - 1\| below it is printed as this value ('model limit') and the own period bound is taken at it *Convention.* |
| `MMR_OWNER_IDENTITY_RATIO_MIN` | `1000.0` | owner route 'identity': tried on a librating three-body angle without an accepted owner whose period needs at least this many times mu_i mu_j e^q (str_ratio, before the small-divisor envelope) *Convention.* |
| `MMR_OWNER_IDENTITY_COEF_TOL` | `0.05` | owner route 'identity': on sigma's decisive window sigma_f = c phi_f + a + b t with \|c / (k_sigma / k_phi) - 1\| at most this (the exact identity passes it for almost any unwrapped candidate: phi librating there and its strength decide) *Convention.* |
| `MMR_OWNER_IDENTITY_RESID_MAX` | `0.1` | owner route 'identity': rms of sigma_f - (k_sigma / k_phi) phi_f about its line on sigma's decisive window, over sigma's median half-amplitude, at most this *Convention.* |
| `MMR_OWNER_IDENTITY_SPAN_MAX_DEG` | `360.0` | owner route 'identity' (cost pre-filter): only a candidate whose low-passed phi_f spans less than this on sigma's decisive window is analysed in full (a phi running a full turn there does not librate about a fixed centre) [deg] |
| `MMR_OWNER_IDENTITY_TOP_N` | `3` | owner route 'identity' (cost): at most this many candidates passing the identity numbers, smallest phi_f span first, get their own full analysis (libration, cover of the window, direct-averaging strength) |
| `MMR_FORCED_WAVE_LINEAR_RATIO` | `2.0` | display only (the figure's owner lines): the linear forced-wave amplitude delta a = K / nu holds when the owner's rate \|nu\| is at least this many times its own small-amplitude frequency omega0 (nu >> omega0) *Convention.* |

### MMR strength

| key | default | meaning |
|---|---|---|
| `MMR_STRENGTH_E_MAX` | `0.3` | the e^q order-of-magnitude strength scale is used only when the median e of the record is below this and the body's radial range crosses no planet of the term; the same median e enters e^q |
| `MMR_STRENGTH_CHECK_RATIO` | `10.0` | check_mmr_strength passes when required/theory is within [1/x, x]: two-body, the range over the snapshots of the directly averaged theory whose observed swing lies in the well overlaps it (none: fail); three-body, required / mu e^q (legal e^q only, a scale, not a bound, support only); 1:1: not_applicable, its scale is judged by MMR_COORBITAL_SCALE_RATIO_MAX |
| `MMR_STRENGTH_IMPLAUSIBLE_RATIO` | `1000.0` | implausible_strength: required/theory >= this or <= 1/this (two-body direct averaging; with the e^q scale of three-body terms, whose f(alpha) reaches ~1e2 (2003 EH16 1M-3J-1: 254), the lower side only with legal e^q); rule 5s: a librating angle for which neither L4 nor R0 passes (a measured failure included) and whose period needs >= this x a computed strength is non_resonant (two-body direct averaging or the harmonic, a centre outside the well excluded unless MMR_STRENGTH_FIRST_OUTSIDE_WELL; three-body only with legal e^q and when required / (mu e^q B^2) of the small-divisor envelope stays >= this too) |
| `MMR_STRENGTH_SNAPSHOTS` | `3` | two-body strength: the averaged disturbing function is evaluated with the simultaneous elements of the body and the planet at this many evenly spaced times inside the decisive episode; the check reads the range over them |
| `MMR_STRENGTH_UNSAVED_PLANET_PHASES` | `4` | two-body strength without the planet's saved elements: its mean J2000 orbit turned about the ecliptic pole by this many evenly spaced angles (its perihelion and node phase is unknown), all in the range |
| `MMR_STRENGTH_NODES_PER_TURN` | `720` | two-body strength: trapezoidal nodes per turn of the planet's longitude in the average at fixed sigma (doubled until the curvature with half of them agrees within MMR_STRENGTH_QUADRATURE_TOL) |
| `MMR_STRENGTH_MAX_NODES_PER_TURN` | `5760` | two-body strength: the largest number of nodes per turn; not converged there = not measured (nearly crossing orbits) |
| `MMR_STRENGTH_SIGMA_GRID` | `72` | two-body strength: points of the coarse sigma grid that finds the wells of <R> |
| `MMR_STRENGTH_FIT_HALF_WIDTH_DEG` | `2.0` | two-body strength: the curvature at the centre is the quadratic fit over +- this (the 1:1 tadpole 9/4 is 0.08 % off at 2 deg, 0.5 % at 5 deg) [deg] |
| `MMR_STRENGTH_QUADRATURE_TOL` | `0.02` | two-body strength: the curvature is converged when the half-node quadrature agrees within this relative difference |
| `MMR_STRENGTH_MIN_DISTANCE_HILL` | `3.0` | two-body strength: at the centre the body must stay this many Hill radii a_P (mu/3)^(1/3) from the planet along the average, else the averaged problem does not hold there (an encounter) and the strength is not measured |
| `MMR_STRENGTH_MP` | `auto` | two-body strength where double precision cannot resolve the well (no stable centre, no converged curvature, or a curvature below MMR_STRENGTH_DOUBLE_FLOOR x \|D0\|): 'auto' = mu_P a 2\|C\| of the resonant harmonic C in extended precision (mpmath; method direct_harmonic); 'off' = not measured, as before |
| `MMR_STRENGTH_MP_DPS` | `60` | two-body strength in extended precision: decimal digits of the first attempt (doubled up to MMR_STRENGTH_MP_DPS_MAX while \|C\| / \|D0\| < 10^(MMR_STRENGTH_MP_GUARD_DIGITS - dps)) |
| `MMR_STRENGTH_MP_DPS_MAX` | `130` | two-body strength in extended precision: the largest number of digits; a harmonic still unresolved there is not measured (precision_exhausted) |
| `MMR_STRENGTH_MP_GUARD_DIGITS` | `15` | two-body strength in extended precision: digits kept beyond the cancellation log10(\|D0\| / \|C\|) of the sum |
| `MMR_STRENGTH_MP_NODES_BODY` | `512` | two-body strength in extended precision: trapezoidal nodes of the body's mean anomaly; the twin grid shifted by half a node must agree within MMR_STRENGTH_QUADRATURE_TOL |
| `MMR_STRENGTH_MP_NODES_PLANET` | `128` | two-body strength in extended precision: trapezoidal nodes of the planet's mean longitude (the same twin check) |
| `MMR_STRENGTH_DOUBLE_FLOOR` | `1e-12` | two-body strength: a double-precision curvature of <D> below this x \|D0\| is rounding, not a well (the quadratic fit over +-2 deg divides the 1e-16 rounding of <D> by ~1e-3 rad^2); the harmonic in extended precision is used instead (numerical, not a convention) |
| `MMR_STRENGTH_FIRST_OUTSIDE_WELL` | `False` | rule 5s (implausible strength) runs before 5a*/5e/5g/5u for a librating angle whose force law and potential with a do not pass; with this off a two-body strength whose observed swings all lie outside the theoretical well (centre_outside_well) keeps the old order and reaches 5s only inside the branch where L4 and R0 are both not measured *Convention.* |
| `MMR_CENTRE_TRACK_NODES` | `400` | moving centre: the most nodes of the track c(t) of the averaged-potential minimum along a record; the node step is max(P_c, span / this). Numerical resolution, not a threshold: the well moves on secular time scales (2013 AS27 31E-36: 56 deg in 20 kyr, 0.2 deg per node), c(t) is interpolated linearly between nodes |

### MMR margins of the confidence

| key | default | meaning |
|---|---|---|
| `MMR_L1_TP_FULL` | `7` | L1 full: turning points at cutoffs x0.5, x1, x2 (three cycles) *Convention.* |
| `MMR_L2_AMP_FULL_DEG` | `120.0` | L2 without a zone: full below this amplitude [deg] *Convention.* |
| `MMR_L2_AMP_THIN_DEG` | `160.0` | L2 without a zone: thin up to this amplitude, above it a medium cap [deg] *Convention.* |
| `MMR_L5_RATIO_FULL` | `1.15` | L5: T0 max/min within this x Omega_max/Omega_min (T0 resolution ~10%) *Convention.* |
| `MMR_L5_RATIO_FAIL` | `2.0` | L5 fails beyond this x the full threshold |
| `MMR_L5_E_CAVEAT` | `0.2` | above this e the period-amplitude law may fail: L5 at most thin |
| `MMR_L6_SHIFT_FULL` | `0.5` | L6 full: centre shift <= this x the mean amplitude |
| `MMR_L7_GAP_FULL` | `0.7` | L7 full: coverage gaps <= this x the period |
| `MMR_L8_STEP_FULL_DEG` | `10.0` | L8 full: rotation per step below this (36 points per turn); not graded for a non_resonant circulation whose slow a stays MMR_FAR_FROM_RES_FACTOR a-scales from a* (the step need not resolve turning points it cannot have) [deg] |
| `MMR_L8_GAIN_FULL` | `0.94` | L8 full: filter gain at the cycle period >= this; below it the libration is filter_too_slow (shape checks not_measured) |
| `MMR_L8_GAIN_FAIL` | `0.5` | L8 fails: filter gain at the cycle period below this |
| `MMR_C1_REV_FULL` | `10.0` | C1 full: largest centre pass of a circulation in revolutions [rev] |
| `MMR_C3_MODULATION_MAX` | `0.3` | C3: rate modulation below this fraction of the mean rate |
| `MMR_C5_RATE_TOL` | `0.2` | C5: \|measured/expected rate - 1\| <= this |
| `MMR_C5_BRANCH_RESID_FACTOR` | `2.0` | C5 alias full: far from a* (MMR_FAR_FROM_RES_FACTOR) and the branch clearance per step [deg] > this x the largest per-step unwrap residual (smp_unwrap_resid_max_deg): no step can have been put on the wrong branch *Convention.* |
| `MMR_C5_FAR_RATIO_MIN` | `3.0` | C5 alias: far ratio below this is not an alias circulation but invalid/alias + resample |
| `MMR_O1_RATIO_THIN` | `0.25` | O1: std of the a-wave before / in the episode thin from this |
| `MMR_O1_RATIO_FAIL` | `0.5` | O1: fails (cap low) from this |
| `MMR_O1_OWNER_COVER_MIN` | `0.5` | O1 not_applicable for an induced libration (origin_detail owner_libration) when the owner's libration intervals cover at least this fraction of O1's background window (the band before sigma's decisive episode): the a-wave there is the owner's libration, not a regime before a capture *Convention.* |
| `MMR_K1_FULL` | `0.97` | K1 full: correlation of slow sigma_dot with p varpi_dot |
| `MMR_W_R2_NONE` | `0.4` | W full (no owner): best candidate R^2 below this |
| `MMR_W_R2_FULL` | `0.85` | W full (induced): owner R^2 at least this |
| `MMR_S_CLOSE_RATIO` | `0.8` | S full (none accepted): the closest member's centre range > this x that of sigma0 in the same scan window |

### Secular and Lidov-Kozai angles

| key | default | meaning |
|---|---|---|
| `SECULAR_FILTER_GI_FACTOR` | `2.0` | P_c of a secular angle >= this x the great inequality period (937 yr), the longest short-period term of an asteroid's elements: the filter passes 1/(1 + x^(2N)) of it (6% for x = 2, N = 2) |
| `SECULAR_FILTER_FORCED_FACTOR` | `2.0` | P_c >= this x the longest forced wave 2 pi/\|f - f_j\| of the angle's own planetary modes (body f = g or s); a single-mode angle has none (its forced response is the resonance itself) |
| `SECULAR_TP_PROMINENCE_DEG` | `5.0` | turning-point prominence floor of a secular angle (see SECULAR_TP_RIPPLE_FACTOR) [deg] *Convention.* |
| `SECULAR_LIBRATION_PERIOD_MIN_YR` | `200000.0` | shortest libration period of a secular angle, a convention: a pendulum librates no faster than half its rate width, and the widest linear resonances are a few arcsec/yr wide (nu6: +-1.2"/yr, Carruba et al. 2018), so periods are >~ 1 Myr; the fastest of the benchmark is Tina around its forced equilibrium (590 kyr). Faster 'episodes' are forced waves, and the TP prominence covers what sigma_f holds faster than this [yr] *Convention.* |
| `SECULAR_TP_RIPPLE_FACTOR` | `4.0` | TP prominence >= this x the 95th percentile of what sigma_f holds faster than SECULAR_LIBRATION_PERIOD_MIN_YR (a wave of half-amplitude r turns by up to 2 r) |
| `SECULAR_RESPONSE_R2_MIN` | `0.5` | check_secular_e_response passes when e_dot (and d sin(i/2)/dt for an s-angle) follows A sin(sigma) + B cos(sigma) + C with R^2 >= this; a column only, it changes no status and no confidence *Convention.* |
| `ZLK_FILTER_GI_FACTOR` | `2.0` | P_c of the Lidov-Kozai angle omega >= this x the great inequality period (937 yr) (6% passes for x = 2, N = 2) |
| `ZLK_LIBRATION_PERIOD_MIN_YR` | `2000.0` | shortest libration period of omega, a convention: the quadrupole ZLK time (2/3pi)(M_sun/m_J) P_J^2/P (1 - e_J^2)^(3/2) (Antognini 2015, the coefficient of eq. 27) is 3.9 kyr at 4 au and 31 kyr at 1 au, and a libration lasts several of it; faster 'episodes' are forced waves [yr] *Convention.* |
| `ZLK_TP_RIPPLE_FACTOR` | `4.0` | TP prominence of omega >= this x the 95th percentile of what omega_f holds faster than ZLK_LIBRATION_PERIOD_MIN_YR |
| `ZLK_TP_PROMINENCE_DEG` | `5.0` | turning-point prominence floor of omega (see ZLK_TP_RIPPLE_FACTOR) [deg] *Convention.* |

### Confidence

| key | default | meaning |
|---|---|---|
| `CONFIDENCE_THIN_BAND` | `0.1` | margin U: a decision within this relative distance of a convention |
| `CONFIDENCE_CELLS` | empty | calibration cells with their author-label counts, written by benchmark/tools/calibrate.py and reported in the cell_* columns, never changing a level (empty = resonances/data/confidence_cells.json) |

### Dry-run

| key | default | meaning |
|---|---|---|
| `DRYRUN_FAIR` | `optional` | 'required' = the step resolves FAIR; 'optional' |
| `DRYRUN_FAIR_FRACTION` | `0.238` | FAIR needs a step <= this x the orbital period (1/4.2) |
| `DRYRUN_DA_STATIC` | `0.005,0.001,0.002` | spread of a without a pilot: belt, NEA, planet-crossing (provisional) [au] |
| `DRYRUN_DA_SP_STATIC` | `0.003,0.0003,0.001` | short-period a scatter without a pilot: belt, NEA, crossing [au] |
| `DRYRUN_STEPS_PER_ORBIT` | `10.0` | SABA steps per orbit (measured) |
| `DRYRUN_STEPS_PER_PERI` | `10.0` | SABA steps per perihelion passage (provisional) |
| `DRYRUN_NEA_Q` | `1.3` | ias15 below this perihelion distance [au] |
| `DRYRUN_HILL_MARGIN` | `1.0` | crossing band around a planet orbit in Hill radii |
| `DRYRUN_DROP_ALIASED` | `False` | drop guaranteed aliased circulations before integrating |
| `DRYRUN_DROP_GHOST_FACTOR` | `20.0` | drop only beyond this many Nyquist half-widths from a_res |
| `DRYRUN_PILOT_STEP_YR` | `0.05` | output step of the pilot integration [yr] |
| `DRYRUN_PILOT_MIN_YR` | `600.0` | shortest pilot [yr] |
| `DRYRUN_PILOT_MAX_YR` | `3000.0` | longest pilot [yr] |
| `DRYRUN_PILOT_CUTOFFS` | `10.0` | pilot length >= this x the predicted cutoff |
| `DRYRUN_AUTO_STEPS_YR` | `0.05,0.1,0.2,0.25,0.5,1.0,2.0,2.5,5.0,10.0,20.0,50.0,100.0,200.0,500.0` | round values OUTPUT_STEP='auto' is chosen from [yr] |
| `DRYRUN_MAX_STEP_YR` | empty | the user's cap on OUTPUT_STEP='auto' (empty = none) [yr] |
| `DRYRUN_FORCED_TERM_MIN_DEG` | `2.0` | pilot: report a line of sigma the filter passes (period >= P_c) whose filtered amplitude reaches this [deg] |
| `DRYRUN_FORCED_TERM_MAX_COEFFICIENT` | `12` | pilot: name such a line \|p n_j - q n\| with p, q up to this |

<!-- settings-table:end -->

## Integration options

See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html) for explanation of the integrator's settings.

- `tmax`/`INTEGRATION_TMAX` (int): the end of the integration in REBOUND units, where 2π is one year: 1000 years is `1000*2π` ≈ `6283`. Default `628319` (100 kyr). A negative value integrates backwards; every quantity computed on the series puts it in time order first, and plots show the actual negative times. `integration_years` (argument) sets the span in years and wins over `tmax`.
- `dt`/`INTEGRATION_DT` (float): the step of the integration in rebound units (2π = 1 yr), or the initial step of an adaptive integrator. By default, `1.0` = 58 days. See [Step size](#step-size).
- `integrator`/`INTEGRATION_INTEGRATOR` (string): the default integrator from rebound. By default, `SABA(10,6,4)`. See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html).
- `integration_safe_mode`/`INTEGRATION_SAFE_MODE` (int or empty): `safe_mode` of WHFast and SABA. Empty (default) keeps the integrator's own default: `0` for WHFast (with `1` its corrector would run at every step) and `1` for SABA, which is what earlier versions used, so their runs are reproduced bit for bit. `0` merges the drifts of consecutive steps and is faster; rebound still synchronises at every output time, but the orbits differ from `1` at round-off level (SABA, `dt=1`, a = 2.2 AU: |Δa| ≈ 3·10⁻⁹ AU after 20 kyr) and chaotic or planet-crossing orbits diverge, so a run with `0` does not reproduce one with `1`. See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html).
- `integration_corrector`/`INTEGRATION_CORRECTOR` (int): the parameter of the corrector for symplectic integrators. By default, `17`. See [rebound documentation](https://rebound.readthedocs.io/en/latest/integrators.html)
- `SOLAR_SYSTEM_FILE` (str): the name of the cache file used to store the initial data of the Sun, planets, and Pluto. It is used to speed up the creation of the simulation. By default, `cache/solar.bin`. Note that in order to avoid issues with initial date&time, the app will automatically add postfix equals to the timestamp of the simulation date, i.e., `cache/solar-12345.bin`, and `cache/solar-moon-12345.bin` for the system with the Moon.
- `solar_system_moon`/`SOLAR_SYSTEM_MOON` (bool): the Moon as a separate body. Default: `False`. See below.

### Step size

The default `dt = 1.0` (58 days) is shorter than the orbit of every planet but Mercury (88 days, 1.5 steps per orbit). With SABA(10,6,4) this is enough for the planets' secular dynamics and for the main belt. Measured against ias15 and SABA with dt = 0.5 (`2026-10-13`): over 2 Myr the planetary frequencies g1, g2, g5, g6, s1, s6 agree with dt = 0.5 to better than 2e-4″/yr; over 2 kyr Mercury's mean longitude is 1.8° off ias15 (0.04° with dt = 0.5), while a main-belt orbit (a = 1.8–3 AU) is within 3e-4° and Jupiter within 3e-6°. WHFast at the same steps is not: Mercury's longitude was 180° off, an asteroid at 2.5 AU 0.3°.

A fixed step needs about ten steps per orbit of the asteroid itself: the run logs a warning below `MIN_STEPS_PER_ORBIT = 10` (a < 1.36 AU at dt = 1). Close encounters are not resolved by a fixed step, whatever the period. Before integrating, the [dry-run](dry-run.md) flags bodies that cross a planet's band (within `DRYRUN_HILL_MARGIN` Hill radii) or have q < `DRYRUN_NEA_Q`, and recommends ias15 for them; it screens the orbits, it does not predict each encounter. With dt = 1, 162474 was ejected (an unphysical orbit: status `invalid`); with dt = 0.5 it was not.

### Solar system model

By default the massive bodies are the Sun, Mercury–Neptune and Pluto, fetched from NASA Horizons by name. `Earth` is then Horizons body 3, the Earth–Moon barycentre, with the Earth+Moon mass. That is fine for asteroids that stay far from the Earth.

With `solar_system_moon=True` (`IntegrationEngine.create_solar_system_with_moon`) the Earth is the geocentre (Horizons 399) and the Moon (Horizons 301) is an eleventh body, appended after Pluto. The planets keep their particle indices, so every resonant angle is built as before, with the Earth's angles now taken against the geocentre. The Moon is never a planet of an angle and is not in the saved planet series. Earth and Moon together have exactly the mass, position and velocity of the barycentre. The price: the geocentre's heliocentric osculating elements carry the Moon's monthly reflex (about 12.6 m/s). Measured against the barycentre, a differs by up to 1e-3 AU, e by 9e-4, λ by 0.05° and ϖ by 3°. The saved Earth series and every Earth angle (the 1:1 angle by 0.05°, any secular angle with g3 or s3 much more) contain this term, sampled at the output step. Use it for objects with close Earth encounters (Earth co-orbitals, NEAs): within a few lunar distances the barycentre is a poor stand-in for the two bodies.

Only `ias15` is accepted with the Moon; any other integrator raises `ValueError` when the simulation is created or the integrator is set up. A fixed-step symplectic split around the Sun cannot follow a 27-day orbit with steps of days to months. Measured over 100 yr (2023-02-25 start, against ias15): WHFast, SABA(10,6,4), MERCURIUS and TRACE with dt = 0.05 (3 days) all put the Earth 0.03 AU off along its orbit; MERCURIUS and TRACE still 0.0015 AU with dt = 0.01, at 3 and 0.3 times the CPU of ias15. The hybrid integrators gave the same error as WHFast, although the Moon never leaves the Earth's encounter radius.

`N_active` is set to the number of massive bodies (10, or 11 with the Moon); the asteroids are test particles.

### Horizons and TLS

The planets are fetched by rebound, which uses Python's `urllib`. Python's default CA bundle may lack the root certificate JPL uses (macOS `/etc/ssl/cert.pem` had no Sectigo Public Server Authentication Root R46 in October 2026). While it fetches the planets, the package therefore verifies against [certifi](https://pypi.org/project/certifi/)'s bundle, and restores rebound's `urlopen` afterwards. Verification is never switched off. If `SSL_CERT_FILE` is set, it is used instead. Setting `SSL_CERT_FILE` from inside a running Python process is not enough once any HTTPS connection has been verified there: OpenSSL keeps the store it loaded first.

### Access to the parameters of `rebound`

While resonances wraps many Rebound integrator settings, you can still manipulate Rebound directly. The rebound simulation is `sim.integration_engine.sim`; it exists after `create_solar_system()`:

```python
sim = resonances.Simulation()
sim.create_solar_system()
sim.integration_engine.sim.ri_ias15.epsilon = 1e-10
```

`run_integration()` calls `setup_integrator()`, which sets `integrator`, `dt` and `N_active` (the number of massive bodies, 10 or 11 with the Moon), the WHFast corrector when `INTEGRATION_CORRECTOR` is set and the SABA safe mode. Change those through the config (`sim.config.dt = 0.01`), not on the rebound object: direct values are overwritten.

## AstDyS and Catalogue options

- `source`/`DATA_SOURCE` (str): where the asteroids' elements come from, `nasa` (default), `astdys` or `mpc` (Minor Planet Center, below). The planets always come from NASA Horizons, at `date`.
- `date` (str or datetime): the start of the integration. Default: the catalogue epoch for `astdys` and `mpc` (whether set by argument, `.env` or environment), today's midnight for `nasa`. A timezone-aware datetime is taken as its naive UTC value (one instant for the planets, the bodies and the clones); a naive one as it is. `find()` and `check()` take the Horizons elements at this date. Elements that carry an `epoch` (AstDyS and MPC rows, Horizons, or your own dict, as MJD or a date) must be at `date`: `add_body` raises `ValueError` otherwise, since REBOUND would start the body at the wrong place on its orbit relative to the planets (MPC rows at another epoch are propagated to `date` by default, below).
- Numbers passed to Horizons (`401` or `'401'`) are always asteroid numbers: a bare `'401'` would be Phobos.
- The AstDyS catalogue is downloaded and cached by the [astdys](https://pypi.org/project/astdys/) package.

### Minor Planet Center (`source='mpc'`)

The elements come from [MPCORB.DAT](https://minorplanetcenter.net/iau/MPCORB.html) (`resonances/mpcorb/`, a self-contained subpackage; the package's side, MPC_* keys and the epoch policy, is `resonances/simulation/mpc_source.py`): heliocentric osculating elements in the J2000 ecliptic with μ = k² (n = k a^−1.5 holds on every row of the file to its rounding), at an epoch in TT. Most rows share the standard epoch (MJD 61200 = 2026-06-09 on the 2026-10-25 file, the epoch of AstDyS too; it moves in steps of 200 days), which is the default `date`. The date is the TT calendar value, and Horizons reads it as TDB for the planets (TT − TDB < 2 ms), as for AstDyS. The planets come from Horizons (DE441), the MPC's fit used DE431 and the AST17 asteroid perturbers: integrated by the package for 138 days, the MPC elements of 433, 719 and 887 reproduce the MPC's own propagation to the file's rounding (5·10⁻⁸ AU, 5·10⁻⁶ deg).

- `MPC_CATALOG`: `mpcorb` (default, `MPCORB.DAT.gz`, 94 MB, 1.58 M orbits), `nea`, `neam00` (NEAs at today's epoch), `distant`, `unusual`, or the URL (`https://` or `file://`) of a file in the same format.
- The file is downloaded on first use into `MPC_CACHE_DIR/<catalog>/` (default `cache/mpc`, next to the AstDyS catalogue), parsed into a table (the file's own digits) and described in `snapshot.json`: URL, ETag, Last-Modified, the sha256 of the downloaded bytes, row counts, the standard epoch. Each download is a generation of its own files (`<sha12>-MPCORB.DAT.gz`, `table-<sha12>-p<parser>.csv`), published by replacing `snapshot.json`; the previous generation is kept, older ones are deleted, and so are staging files (`*.part`) older than a day. A simulation records these in `simulation.json` (`mpc_snapshot`), and a batch run resumes only on the same snapshot. Workers get the elements from their parent and never open the catalogue.
- The catalogue is refreshed only by `resonances mpc-update [--catalog=...] [--force]` (or `resonances.mpcorb.update_catalog()`): a conditional GET on the ETag, then the download, its Content-Length, and a parse that must find MPCORB records with μ = k² before the old snapshot is replaced. A failed update keeps the old one. Older than `MPC_MAX_AGE_DAYS` (30; 0 = never) days since its Last-Modified, it only warns. `MPC_SHA256` refuses any other snapshot.
- A body is found by number (`463`, `'(463)'`, packed `'00463'`, `'A1429'` = 101429, `'~02pM'` = 630872), by provisional designation, packed or not (`'2001 ND13'`, `'2001ND13'`, `'K01N13D'`, the extended `'_PO00fh'` = 2025 OL723), the principal designation of a numbered object without a name (`'1998 VF31'`), or a name; designations and names in any case. An empty identifier, or one that matches two rows, raises `ValueError`. A list is looked up in one pass over the table (64,000 numbers in 0.5 s).
- `MPC_EPOCH_POLICY`: `propagate` (default) integrates every row whose epoch is not `date`, as test particles with IAS15 in the run's planets taken at their own epoch, to the common `date`, and the run then integrates all bodies together. Each such body gets a warning (designation, gap in days, U, arc and oppositions), and `find` adds one line, "N bodies propagated from other epochs, max gap X d"; the element dict has `propagated_from_epoch`. In the 2026-10-25 file they are 1,891 of 1.58 M rows: the 1,865 unperturbed one-opposition orbits of its last section, each at its own epoch, and 26 others (impactors such as 2008 TC3, orbits left at the previous standard epoch). The propagation was validated against the MPC's own propagation of 433, 719 and 887 to Δa ≤ 5·10⁻⁸ AU over 138 days; asteroid perturbers, relativity and non-gravitational forces are not in the model. A one-opposition orbit with an old epoch is poorly determined whatever the propagation: the integration carries its elements, not its uncertainty. `MPC_PROPAGATE_MAX_DAYS` (empty by default: no limit) is an optional safety limit on the span; a longer one is refused (for the clones of [Clones](clones.md), only a warning). `strict` refuses every such row (`add_body`/`check` raise; `find` leaves the body out with a warning, as one not in the catalogue).
- `MPC_WARN_U` (6): a warning for orbits with the uncertainty parameter U ≥ this, a letter (E: eccentricity assumed) or none, and for unperturbed orbits.
- Each element dict (`body.initial_data`, `simulation.json`) carries `source`, `packed`, `number`, `designation`, `mpc_name`, `U`, `perturbers` and `catalog_sha256`.
- Only the elements of the run come from MPCORB. `find()` without `type` (or with `type='secular'` and no `formulas`) also searches secular resonances, and that search reads the AstDyS synthetic proper frequencies g and s of each body whatever `source` is (`SecularResonanceFinder`): a body AstDyS does not list gets no secular candidates (a warning), one listed without g or s stops the search with `ValueError`. Pass `type='mmr'`, or `formulas=[...]`, to stay off AstDyS. The Lidov-Kozai angle is added to every body without a catalogue search. The same holds for `source='nasa'`.
- MPC's per-object orbit API (get-orb) gives the covariance: the clones read it (`CLONES_SOURCE=mpc`, [Clones](clones.md)); the elements of a run still come from MPCORB.


#### Selecting objects of the catalogue

`resonances.mpcorb` searches the cached catalogue (the one `MPC_CATALOG` names), as the `astdys` package does AstDyS:

```python
from resonances import mpcorb  # noqa: E402

mpcorb.search(463)  # the elements of (463) Lola (radians, epoch MJD TT); None if absent
mpcorb.search([1, '2001 ND13'])  # {identifier: elements}
mpcorb.search_by_axis(2.4, sigma=0.01)  # DataFrame of the rows with 2.39 <= a <= 2.41 AU
mpcorb.query(a=(0, 1))  # orbits inside 1 AU
mpcorb.query(q=(None, 1.3), H=(None, 18), U=(0, 3), numbered=True)
mpcorb.query(orbit_type=['apollo', 'aten'], pha=True, epoch='standard')
mpcorb.query(node=(350, 10))  # angle ranges may wrap through 0
```

`query` takes inclusive ranges `(min, max)` (None for an open end) on `a`, `e`, `i`, `q` = a(1−e), `Q` = a(1+e), `H`, `node`, `peri`, `M` (degrees), `n_obs`, `n_opp`, `arc_years` (last minus first year of observation; days/365.25 for a one-opposition orbit) and `U` (0–9; a row with a letter or no U fails a U range), plus `epoch` (`'any'` or `'standard'`), `numbered` (True/False) and the MPC flags of columns 162–165 ([MPOrbitFormat](https://minorplanetcenter.net/iau/info/MPOrbitFormat.html)): `orbit_type` (`atira`, `aten`, `apollo`, `amor`, `q_lt_1665` = q < 1.665 AU, `hungaria`, `mpc_internal` (value 7, "internal MPC use", 16,289 rows near a = q = 2 AU), `hilda`, `jupiter_trojan`, `distant`) and `neo`, `km_neo`, `pha`, `one_opp_earlier`, `critical_list` (True/False). The MPC says its classification "is not 100% reliable". The result has the elements, q, Q, H, U, n_obs, n_opp, arc_years, flags, orbit_type and the flag columns. `find` and `check` take it as the list of bodies:

```python
sim = resonances.find(mpcorb.query(a=(2.38, 2.42), numbered=True, U=(0, 2)), planets=['Jupiter', 'Saturn'], type='mmr', source='mpc')
```

From the shell: `resonances mpc-search --a 0:1 --e :0.3 --limit 20` prints a table, `--output sel.csv` writes every row, `--ids` prints the identifiers for `find --asteroids=...` (`resonances mpc-search --help` lists every option).

`resonances/mpcorb/` imports nothing else of the package (settings come as `mpcorb.Settings` or arguments, messages go to the `mpcorb` logger), so it can be moved out as a package of its own; inside `resonances` its defaults are the MPC_* keys.

## Matrices

The catalogues of mean-motion resonances are bounded by the `MATRIX_` keys ([Resonance matrices](matrix.md)). Each
cached file carries a build record (`<file>.meta.json`); a file built with other limits or planets logs a warning, so
**delete the cached CSV after changing a limit** to rebuild it. A file written by an older version of the package
(first line not the current catalogue stamp) is rebuilt automatically.

| key | default | meaning |
|---|---|---|
| `MATRIX_2BODY_PRIMARY_MAX` | `12` | the planet's coefficient runs from 1 to this − 1 |
| `MATRIX_2BODY_COEF_MAX` | `11` | the asteroid's coefficient runs from −this to −1 |
| `MATRIX_2BODY_ORDER_MAX` | `10` | largest order (\|sum of the coefficients of the mean longitudes\|); also bounds the degree of the subresonances the MMR analysis scans |
| `MATRIX_2BODY_FILE` | `cache/mmr-2body.csv` | the cached two-body catalogue (resonant semi-major axes) |
| `MATRIX_3BODY_PRIMARY_MAX` | `8` | the first planet's coefficient runs from 1 to this − 1 |
| `MATRIX_3BODY_COEF_MAX` | `7` | the coefficients of the second planet and the asteroid run from −this to this − 1 |
| `MATRIX_3BODY_ORDER_MAX` | `6` | largest order; also bounds the subresonances scanned |
| `MATRIX_3BODY_FILE` | `cache/mmr-3body.csv` | the cached three-body catalogue |
| `MATRIX_2BODY_RETRO_COEF_MAX` | `15` | largest p and q of a retrograde p/−q resonance (`1J-3r0`) |
| `MATRIX_2BODY_RETRO_ORDER_MAX` | `30` | largest order p + q of a retrograde resonance (with the coefficient limit: the 143 resonances per planet of Li et al. 2019) |
| `MATRIX_2BODY_RETRO_FILE` | `cache/mmr-2body-retro.csv` | the cached retrograde catalogue, every k for every pair |

## Batch options

- `batch_enabled`/`BATCH_ENABLED` (bool): run more than `batch_threshold` bodies in batches, each in its own process. Default: `True`.
- `batch_threshold`/`BATCH_THRESHOLD` (int): the number of bodies above which batching starts. Default: `100`.
- `batch_size`/`BATCH_SIZE` (int or empty): bodies per batch. Empty: 50, or fewer so that every core gets two batches. Default: `100`.
- `n_cores`/`BATCH_N_CORES` (int): worker processes. Default: `3`.
- `LOW_MEMORY_ALLOCATOR` (`auto`, `on`, `off`): the allocator of spawned worker processes (batch runs, benchmark
  reclassify). On macOS `on` starts them with `MallocLargeCache=0`, so freed large blocks return to the system instead of
  staying in the worker's footprint: benchmark workers peak at 1.4 GB instead of 2.3 GB, with the same results and more
  CPU time. `auto` turns it on where memory, not the CPUs, limits the number of workers (the benchmark's `--workers auto`
  then sizes them at 1.6 GB instead of 3 GB); a batch run's `n_cores` is set by the user, so `auto` leaves it off there.
  Other systems: no policy (not measured). `make test*` run the tests with it (`LOW_MEMORY_ALLOCATOR=off make ...` does
  not). Default: `auto`.

A batched run can be resumed in the same `save_path`; see [Serialization](serialization.md).

## Clone options

Clones (virtual asteroids) drawn from an orbit fit's covariance; see [Clones](clones.md).

- `clones`/`CLONES` (int): clones of every body added by its number or designation (`add_body`, `check`, `find`, CLI
  `--clones=N`); 0 = none. 100 gives a status share to ±10 % (95 %, Wilson), 400 to ±5 %, 1000 to ±3 %. Default: `0`.
- `clones_source`/`CLONES_SOURCE`: `auto` (the provider of `DATA_SOURCE`: `astdys` → astdys, `mpc` → mpc, `nasa` →
  sbdb), `astdys` (numbered asteroids), `neodys` (NEAs), `sbdb` (JPL) or `mpc` (refused for U ≥ 3). Default: `auto`.
- `clones_epoch`/`CLONES_EPOCH`: for AstDyS/NEODyS, `fit` (`.eq0`, the fit's own epoch) or `standard` (`.eq1`, the
  catalogue's epoch, a linear propagation of `.eq0`); empty = `standard` for astdys, `fit` for neodys. A covariance at
  another epoch than `date` is sampled there and every clone integrated to `date`.
- `clones_seed`/`CLONES_SEED` (int): the seed; each body's generator is seeded by (seed, crc32(name)). Default: `0`.
- `clones_batch`/`CLONES_BATCH` (int): clones per REBOUND simulation; the batches run on `n_cores` processes after the
  ordinary bodies. Default: `20`.
- `clones_plot_each`/`CLONES_PLOT_EACH` (bool): the figures of every clone as of an ordinary body; otherwise only the
  ensemble figure (plot kind `clones`, also drawn with `combined`). Default: `False`.
- `CLONES_CACHE_DIR` (str): the downloaded orbit solutions, `<dir>/<source>/<file>` and `<file>.meta.json`. Default:
  `cache/clones`.

## Other

- `LOG_FILE` (str): the path to the file where the app stores log data.
- `LOG_LEVEL` (str): the default level what the app should store in the log file (options: `debug`, `info`, `warning`, `error`, `critical`). The default value is `info`.
