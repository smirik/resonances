# secular (and Lidov-Kozai) — science invariants

Covers `resonances/secular/`, `resonances/lidov_kozai/`, `resonance/omega_free_gate.py`, `resonance/cross_spectrum.py`,
`resonance/coherence_analysis.py` and the decision in `resonance/analysis/secular/` and `analysis/zlk/`. User docs:
`docs/secular.md`, `docs/classify.md`, `docs/free-elements.md`, `docs/coherence.md`. Reasons: `dev/physics-notes.md`.

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
- `resonance/coherence_analysis.py`: per-resonance-type pair presets, `analyse_resonance`, the `confirmed_ei` flag,
  CSV row builders. Changes no status.

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
- **Free elements do not apply to co-orbitals**: their forced term is the perturber's own vector, not a planetary
  fundamental; a confident-looking straight line there is not a measurement. `analyse_zlk`/`evaluate_gate` have no
  co-orbital guard yet (open, `dev/status.md`).
- **The ZLK status** comes from the gate's outcome in `decide_status`: confirmed → resonant; kinematic →
  non_resonant/kinematic_libration (−2); no verdict → uncertain/gate_unavailable (−9). A transient, drift or short
  record changes only on a kinematic outcome (with the free circle never around the origin). Gate **disabled** keeps
  the geometric status with `check_zlk_free_omega` not_measured — not the same as unavailable. −2 and −9 are never
  merged in a statistic.
- **e–i coherence is a flag** (`confirmed_ei`): its validity band T/(5·n_seg) (25 kyr at 500 kyr) sits below real ZLK
  periods (27–58 kyr); e–i exchange also happens with circulating ω. `ZLK_MIN_ANTICORRELATION` = 0.5 is provisional.
- **The gate**: signed rates with tolerance 1.296e6/T ″/yr (one turn over the record); a confined residual is
  tested against the g and s modes (and zero for e when the constant term was clustered away), a winding one also
  against every g_j − s_k (s5 = 0 included); no amplitude floor (the prefit
  estimates e_free, not e_forced); calibrated at 500 kyr (`T ≥ max(500 kyr, 10·P_ZLK)`), where longer records answer
  "does it stay" rather than "is it ZLK".
- **Secular cutoff** includes the angle's longest forced wave (4305 and 8089 hid real librations behind a fixed
  window). `SECULAR_LIBRATION_PERIOD_MIN_YR` (200 kyr) and the ZLK floor (2 kyr) are conventions.
- **Status statistics use the 21 literature secular resonances**, not broad order-≤4 scans (their hits were
  near-commensurabilities with g1/g2/s1/s2, wrong without GR). The finder itself scans the 75 `SECULAR_FORMULAS` by
  divisor; an explicit formula is classified as given.
- Lidov's names: c1 = (1 − e²) cos² i; the code's `c` is Lidov's c2 (renamed per Lidov).

## Integration notes

- SABA cannot follow Earth-crossers (3752, 10563 end at −99); use ias15. 162474 needs SABA dt = 0.5.
- About 0.7 GB per body in memory at Nout 100k for 10 Myr; arrays of unplotted rows are dropped after the verdict.
- Formula strings went through `detect_resonance_type`, which once read 9 catalogue formulas as MMRs (fixed; an
  MMR key starting with `2S…` once went to the secular parser). Keep a test when touching that detection; passing
  `SecularResonance` objects avoids it altogether.
- 3040 ZLK over 500 kyr (SABA dt = 1) depends on the start epoch within weeks: pin the date in tests.

## Open

591986: free-element g 220″/yr vs 6.65″/yr in the proper catalogue; 88899 two-island episodes (tier D); secular
bounded angles take a dynamic origin directly. See `dev/status.md`.
