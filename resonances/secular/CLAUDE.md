# secular (and Lidov-Kozai) — science invariants

Covers `resonances/secular/`, `resonances/lidov_kozai/`, `resonance/omega_free_gate.py`, `resonance/cross_spectrum.py`,
`resonance/coherence_analysis.py` and the decision in `resonance/analysis/secular/` and `analysis/zlk/`. User docs:
`docs/secular.md`, `docs/physics/secular-zlk.md` (the decision), `docs/free-elements.md`, `docs/coherence.md`. Reasons: `dev/physics-notes.md`.
One home per topic in the docs, the others link: why the angle is built so (`secular.md`), the steps, angle_status,
status and the gate's outcome → status (`physics/secular-zlk.md`), the gate's outcomes (`free-elements.md` "The gate"),
Lidov's integrals (`lidov-kozai.md`), −2 vs −9 (`physics/status-tree.md`), the baseline rule (`physics/confidence.md`).

## Modules

- `secular/proper_angle.py`: the classified secular angle: the body's ϖ and Ω against the planets' mode longitudes
  g_j·t + phase_j (`PLANETARY_FREQUENCIES`, `PLANETARY_PHASES`) or the measured ones, reduced by default — forced terms
  at the fundamentals not in the angle subtracted (`reduced_vectors`). Also `free_rates`/`free_divisor` (diagnostic
  only).
- `secular/planet_modes.py`: the planets' mode longitudes measured in the integration
  (`secular_planet_phases='measured'`, needs `save_planets`): phase of mode j in planet j's own vector, least squares
  on all modes in sliding 5 Myr windows. The linear law drifts by tens of degrees over 20 Myr (inner modes, and s6).
- `secular/free_elements.py`: forced/free split by least squares at the planetary fundamentals: cluster the candidates
  at 2/T, fit one representative each, mask each free vector near its own origin, measure the three rates. Pure signal
  processing, no Body or status.
- `secular/combinations.py`: d'Alembert combinations near a body.
- `resonance/omega_free_gate.py`: the free-omega gate (`evaluate_gate`): an outcome only (confirmed, kinematic, no
  verdict, disabled); `analysis.secular.decide.decide_status` turns it into the ZLK status.
- `resonance/cross_spectrum.py`: Welch coherence + cross-phase, line extraction, phase criterion, e_forced/e_free (in
  the ω plane, not ϖ). Pure math, no Body/Simulation.
- `resonance/coherence_analysis.py`: per-resonance-type pair presets, `analyse_resonance`, the strongest-line columns
  (`zlk_coherent`, `zlk_check`), CSV row builders. Changes no status.
- `resonance/analysis/zlk/diagnostics.py`: columns of a ZLK row that change no status: the e-i coupling at the decisive
  period (`zlk_ei_*`, `confirmed_ei`, `check_zlk_ei`), the attribution of a kinematic libration (`zlk_forcing_*`), and
  the centre-scale guard (`forced_beat_centre`). `resonance/analysis/secular/slow.py`: the slow line of the centre
  (`zlk_centre_*`, `sec_centre_*`). `data/invariable_plane.py`: the exact rotation (shared with the MMR analysis).

## Invariants (the root CLAUDE.md states them as rules; the reasons are here)

- **No osculating planet elements in secular angles** (the bug fixed 2026-10-12: every s_j-term status before it was
  wrong). From the ecliptic no planet's node circulates (the 1.58° invariable-plane tilt dominates) and only Mercury's,
  Jupiter's and Saturn's perihelia follow their mode. `secular_angle_mode='proper'` (default) is what the classifier
  reads; the osculating series is kept for reference only.
- **Inner-planet frequencies and phases** in `const.py` are measured on the package's own Newtonian model
  (`experiments/2026-10-planetary-frequencies`), not taken from the literature: the model has no GR, so g1, g2, s1, s2
  differ from La90 by 0.1–0.3″/yr.
- **The cluster representative** (`PRIORITY_E`, `PRIORITY_I`) is a modelling assumption, not a free convention: a
  different member changes the verdict on 3752 and 591986. What must stay true, and is tested: no choice ever turns a
  non-librator into a libration.
- **Free elements do not apply to co-orbitals of a planet whose modes are not in the fitted basis** (Mercury to Mars:
  g1-g4, s1-s4 are not candidates): their forced term is the planet's own vector; a confident-looking straight line
  there is not a measurement. Guard (`omega_free_gate.coorbital_planet`, author's variant (a), 2026-10-28):
  |<a>/a_P - 1| < `ZLK_COORBITAL_HILL_FACTOR` (m_P/3)^(1/3) and the planet's modes not fitted -> outcome unavailable,
  detail `coorbital_unfitted_forcing` (uncertain/gate_unavailable for a libration; an episodic row keeps its status:
  2020 PP1 stays transient). Never not_applicable (that would keep the geometric status and make 591986 resonant).
  Jupiter Trojans are covered: their forced vector is Jupiter's, g5+g6 (591986: fitted g5 1.02 x Jupiter's own term).
- **The ZLK status** comes from the gate's outcome in `decide_status`: confirmed → resonant; kinematic →
  non_resonant/kinematic_libration (−2); no verdict → uncertain/gate_unavailable (−9). A transient, drift or short
  record changes only on a kinematic outcome with rho_max < 1: max over the record of rollmed(|z_free|, W) /
  rollmed(|z_forced|, W), W = max(beat window, `ZLK_RHO_WINDOW_CYCLES` P_cyc) (`rho_local_max`; 555527: 0.66 at
  100 kyr, 0.63 at 500 kyr). The former median ratio (one e_free over the smallest forced radius) is the column
  `zlk_gate_rho_global` and `free_rho_max`, descriptive only. Gate **disabled** keeps
  the geometric status with `check_zlk_free_omega` not_measured — not the same as unavailable. −2 and −9 are never
  merged in a statistic.
- **e–i coupling is a column** (`zlk_ei_coupling`; `confirmed_ei` = coupling `zlk`, kept for compatibility), measured
  for every ZLK row at its decisive period (P_cyc; a circulating ω: pi/|omega_dot|), i from the invariable plane (tilt
  for i > 90°, an orbit flip is not measured): `zlk` = antiphase at the constant-sqrt(1-e^2)cos i amplitude,
  `forced_beat` = in phase at the constant-sqrt(1-e^2)(1-cos i) amplitude (the momentum conjugate to Omega when the
  forcing acts on varpi only). Coupling is not libration: 15527 circulates and shows `zlk` at its 2.05-kyr harmonic;
  591986's 6-kyr cycle is `forced_beat`. The Welch line (`zlk_coherent`, `zlk_check`) stays: its band T/(5·n_seg)
  sits below real ZLK periods, and it is skipped for non_resonant rows; the period-specific coupling is not.
- **The gate**: signed rates with tolerance 1.296e6/T ″/yr (one turn over the record); a confined residual is
  tested against the g and s modes (and zero for e when the constant term was clustered away), a winding one also
  against every g_j − s_k (s5 = 0 included); no amplitude floor (the prefit
  estimates e_free, not e_forced); calibrated at 500 kyr (`T ≥ max(500 kyr, 10·P_ZLK)`), where longer records answer
  "does it stay" rather than "is it ZLK".
- **Secular cutoff** includes the angle's longest forced wave (4305 and 8089 hid real librations behind a fixed
  window). `SECULAR_LIBRATION_PERIOD_MIN_YR` (200 kyr) and the ZLK floor (2 kyr) are conventions.
- **The action of an uncertain verdict** (`analysis/secular/extend.py`) changes no status: extend with a next check
  (N by cause, `base_extend_cause`) or review where no length helps (undefined_phase, unavailable, unusable_series,
  a planetary residual beyond `ZLK_EXTEND_MAX_YR`). The gate's `min_distance` (d_min) feeds it; keep the extension
  local when changing the gate.
- **Status statistics use the 21 literature secular resonances**, not broad order-≤4 scans (their hits were
  near-commensurabilities with g1/g2/s1/s2, wrong without GR). The finder itself scans the 75 `SECULAR_FORMULAS` by
  divisor; an explicit formula is classified as given.
- Lidov's names: c1 = (1 − e²) cos² i; the code's `c` is Lidov's c2 (renamed per Lidov). The columns c1, c2, h_quad
  are osculating, at the epoch, in the invariable plane (`evaluate_invariable`, exact rotation; 1373's c2 is +0.0014
  from the ecliptic, −0.0011 from the invariable plane).
- **Two scales of ω** (591986): the fast cycle (6 kyr, the free-forced beat 2pi/|b_varpi − g5|) decides; the centre's
  slow wave (54 kyr) is the g6−g5 beat of the forced vector, ±arcsin(|c6|/|c5|) = 22.8°, measured as a column
  (`zlk_centre_*`, origin `forced_beat:g6-g5`, reconstruction r 0.99, gain 1.00). A centre-scale libration at a forced
  beat while rho_max < 1 never becomes the decisive period (`sec_centre_rejected`; a clean synthetic otherwise decided
  at 54 kyr), only if the beat is measured (two nonzero per-mode amplitudes; old files: no beats) and its forced phase
  wave explains the centre in shape and size (`slow.forced_beat_model`: r and `ZLK_CENTRE_MODEL_AMP_FACTOR`; r alone
  is blind to size), and only where the gate acts (enabled, a split, no unfitted co-orbital): a disabled gate keeps the
  geometric scale (Codex W25 review, findings 1 and 5). Attribution (`zlk_forcing_*`) needs the per-mode amplitudes,
  persisted in omega-free.csv (`amp_*`); old files: none. Its strict pin bounds the free radius by its maximum over the
  record, never the median (finding 2). The e-i coupling is `none` only on finite numbers; an undersampled or
  unfilterable band is not_measured (finding 4).

## Integration notes

- SABA cannot follow Earth-crossers (3752, 10563 end at −99); use ias15. 162474 needs SABA dt = 0.5.
- About 0.7 GB per body in memory at Nout 100k for 10 Myr; arrays of unplotted rows are dropped after the verdict.
- Formula strings went through `detect_resonance_type`, which once read 9 catalogue formulas as MMRs (fixed; an
  MMR key starting with `2S…` once went to the secular parser). Keep a test when touching that detection; passing
  `SecularResonance` objects avoids it altogether.
- 3040 ZLK over 500 kyr (SABA dt = 1) depends on the start epoch within weeks: pin the date in tests.

## Open

591986: free-element g 220″/yr vs 6.65″/yr in the proper catalogue (the tadpole sidebands are not modelled); its node
librates against g5 only over ≥ 3 Myr (Ω − ϖ5 within 52°–129° over 4 Myr): a node-lock flag waits for the author's
decision; 88899 two-island episodes (tier D); secular bounded angles take a dynamic origin directly. See `dev/status.md`.
