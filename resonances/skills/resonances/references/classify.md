<!-- Generated from docs/classify.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Secular and ZLK classification details

This page lists the rules specific to secular and Lidov-Kozai (ZLK) angles. The physical explanation is in
[Secular and Lidov-Kozai](physics/secular-zlk.md); the shared geometry, ledger and confidence rule are in the
[full reference](libration-analysis.md).

`analysis.secular.analyse_secular(series, settings)` and `analysis.zlk.analyse_zlk(series, settings, free_elements=...)`
read a `LongPeriodSeries`: the classified angle (the reduced proper angle of a secular resonance, $\omega$ for ZLK) with
$a$, $e$, $i$, $\varpi$, $\Omega$ on a uniform grid of years. Both are called by `Simulation.identify_librations` on
every path (a run, a batch worker, `restore(recompute_librations=True)`).

## Steps

| step | what |
|---|---|
| 0 input | an unphysical orbit ($e > 1.3$, $a < 0$, non-finite $a$ or $e$) → `invalid/undefined` (`unphysical`); fewer than three samples or non-finite times or angles → `uncertain/undefined` (`unusable_series`) |
| 1 cutoff | $P_c$ and the turning-point prominence (below) |
| 2 filter | $\sigma_f$ at $P_c$, $P_c/2$ and $2P_c$; the window $[t_0 + P_c,\ t_1 - P_c]$. When $\sigma$ cannot be filtered at $P_c$ (a grid not uniform and increasing within `FILTER_GRID_RTOL`, or $P_c \le 2\Delta t$) → `uncertain/undefined` (`unusable_series`), no geometry (an MMR gives `invalid/undersampled` there) |
| 3 geometry | turning points, runs, episodes, the centre and its passes, exit and entry (shared with MMRs) |
| 4 angle | `angle_status` from the geometry (below); without a libration at $P_c$, the same at the scale of the centre |
| 5 physics | secular: the $e$/$i$ response (a column); ZLK: the free-omega gate |
| 6 status | `analysis.secular.decide.decide_status` |
| 7 confidence | margins N, S, F, P, G and the caps |

## Cutoff and the shortest libration

$$
P_c = \max\left(f_{GI}\,P_{GI},\ f_w \max_j \frac{2\pi}{|f - f_j|},\ \frac{P_{\min}}{f_{\min}}\right)
$$

- $P_{GI}$ = 937 yr, the Jupiter–Saturn great inequality; $f_{GI}$ = `SECULAR_FILTER_GI_FACTOR` or
  `ZLK_FILTER_GI_FACTOR` (2).
- The forced waves (secular angles with two or more planetary modes only): the forced term at the mode $f_j$ of the
  angle turns against the free one at $f - f_j$, where $f$ is the body's own $g$ or $s$; $f_w$ =
  `SECULAR_FILTER_FORCED_FACTOR` (2). A single-mode angle such as $\nu_6 = g - g_6$ has none: there the forced term is
  the resonance itself.
- $P_{\min}$ = `SECULAR_LIBRATION_PERIOD_MIN_YR` (200 kyr) or `ZLK_LIBRATION_PERIOD_MIN_YR` (2 kyr);
  $f_{\min}$ = `FILTER_MIN_LIBRATION_CUTOFFS` (2).

The output step does not enter. An episode is a libration of the angle only when its period is at least
$\max(P_{\min},\ f_{\min}P_c)$; a faster one is a forced wave. The turning-point prominence is
$\max(p_{\text{floor}},\ f_r r)$ with $p_{\text{floor}}$ = `SECULAR_TP_PROMINENCE_DEG` / `ZLK_TP_PROMINENCE_DEG` (5°),
$f_r$ = `SECULAR_TP_RIPPLE_FACTOR` / `ZLK_TP_RIPPLE_FACTOR` (4) and
$r = Q_{0.95}\big(|\sigma_f - \operatorname{LP}_{P_{\text{floor}}}\sigma|\big)$ on the edge-trimmed window: what $\sigma_f$
still holds faster than the effective shortest libration $P_{\text{floor}} = \max(P_{\min},\ f_{\min}P_c)$, the same
floor as for episodes ($r = 0$ when $P_{\text{floor}} \le P_c$). For ZLK with $P_c = 2\times937$ yr this is 3748 yr,
not 2 kyr. The filter acts on the unwrapped angle with its trend removed and put back, so it keeps the number of
turns and cannot turn a circulation into a libration.

## angle_status

| condition | angle_status |
|---|---|
| an accepted episode (≥ `LIBRATION_MIN_TURNING_POINTS` = 3 turning points, i.e. one full cycle; a stationary or modulated centre; a period not below the shortest libration; no centre revolution inside it) covering the record (both ends within `LIBRATION_COVERAGE_PERIODS` periods) | `libration` (`stationary`, `modulated`) |
| accepted episodes, none covering | `episodic` |
| no episode; the centre's largest monotone pass ≥ 1 turn | `circulation` |
| no turning point; the filtered angle turns ≥ 1 time | `circulation` |
| no turning point, less than a turn | `insufficient_time` (`arc` or `stall`) |
| runs rejected for a moving centre cover ≥ `LIBRATION_DRIFT_MIN_WINDOW_FRACTION` (0.5) of the window | `drift` |
| otherwise (fewer turning points than one cycle) | `insufficient_time` (`few_cycles`) |

**The scale of the centre.** Without a libration at $P_c$, the geometry is measured again on the angle filtered at
$W_c$ = `LIBRATION_CENTRE_CYCLES` × $P_{\text{cyc}}$ (3 cycles), with the shortest libration $\max(P_{\min}, 2W_c)$ and
the centre-turn prominence. This scale replaces the fine one when it shows a libration (or libration episodes the
fine scale lacks): a slow secular libration often carries faster waves, and its centre is what librates
(`sec_scale` = `centre`).

## Status

| angle_status | status / reason |
|---|---|
| `libration` | `resonant` / `dynamic_libration` |
| `episodic` | `transient` / `dynamic_episode` |
| `circulation` | `non_resonant` / `circulation` |
| `drift`, `insufficient_time` | `uncertain` / `baseline_needed` (action `extend`) |

A secular angle has no further physics in the status: its $e$/$i$ response (`check_secular_e_response`,
`SECULAR_RESPONSE_R2_MIN`) is a column only.

**ZLK: the free-omega gate.** `omega_free_gate.evaluate_gate` ([Free elements](free-elements.md)) gives an outcome
only; `decide_status` turns it into the status. The gate judges every $\omega$ that stays within a turn (`libration`,
`episodic`, `drift`, `insufficient_time`):

| angle_status | gate outcome | status / reason / origin |
|---|---|---|
| `libration` | confirmed | `resonant` / `dynamic_libration` (check `check_zlk_free_omega` pass) |
| `libration` | kinematic (the free $\omega$ winds) | `non_resonant` / `kinematic_libration` / `kinematic`, detail `free_omega_winds` |
| `libration` | no verdict (short window, phase undefined, a residual at a planetary rate, no free elements) | `uncertain` / `gate_unavailable` / `unknown` |
| `episodic`, `drift`, `insufficient_time` | kinematic with $\rho_{\max} < 1$ (the free circle never encloses the origin) | `non_resonant` / `kinematic_libration` |
| `episodic`, `drift`, `insufficient_time` | anything else | unchanged (check not measured) |

`kinematic_libration` (a measurement: the libration is kinematic) and `gate_unavailable` (no measurement) are
different statuses and must never be merged in a statistic. The $e$–$i$ coherence (`check_zlk_ei`, `confirmed_ei`,
[Coherence](coherence.md)) is a column only.

## Confidence

| margin | what | full when |
|---|---|---|
| N (S) | turning points of the decisive episode | above the minimum (thin at it, with the cap `three_tp_episode`, ≤ high) |
| S (S) | its centre | every cycle's band overlaps, with enough independent cycles; not measured with one; not applicable to a modulated centre |
| F (S) | the filter | the libration period ≥ `FILTER_MIN_LIBRATION_CUTOFFS` × $P_c$; else not measured |
| P (D) | a circulation | its largest centre pass ≥ 1 turn at every smoothing (×0.5, ×1, ×2); else thin |
| G (D) | ZLK: the gate | confirmed with its cycle count clear of the minimum by `CONFIDENCE_THIN_BAND`; a kinematic outcome is full |

The level follows the shared rule ([reference §14](libration-analysis.md#confidence)); a visible exit or
entry caps it at medium. For ZLK the physics margin is G; a secular angle has none.

## Output

The summary row carries the analysis' columns (`flt_*`, `geo_*`, `sec_*`, `rsp_*`, `zlk_gate*`, `ep_*`); the runs of
turning points go to episodes.csv; the figure is the one-page `combined` figure (`SecularPlotter`, [Plots](plots.md)).

!!! note "Legacy integer codes"
    `status_code` keeps the old integers for comparison with old runs: kinematic libration −2, uncertain
    `insufficient_time` −4, uncertain drift −6, other uncertain (including `gate_unavailable`) −9. Old files are read
    back through `verdict_from_status_code`. −2 and −9 are never merged.
