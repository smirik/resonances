# Checked and correct

Facts verified against independent sources, mostly in the package review of 2026-10-13 and the MPC/clones work of
2026-10-26/27. Re-check one only if the code it describes changed.

## Epochs, frames, units

- AstDyS `allnum.cat`: ecliptic J2000, heliocentric Keplerian elements, all rows at MJD 61200 = 2026-06-09, angles in
  radians.
- Horizons: `CENTER=@0`, ecliptic J2000; astroquery elements and rebound vectors are both TDB (a bare calendar date is
  read as TDB).
- The asteroid is added with `primary=Sun` and its orbit read heliocentric.
- 463 against JPL end to end: 233 km at t0, 873 km at +50 yr (ias15); Jupiter barycentre to 1e-8 km. Hektor through
  Horizons: 2e-15 AU. AstDyS against Horizons at the catalogue epoch: 2.1e-6 AU.
- Masses: GM from DE440 (Sun, Jupiter and Saturn systems, EMB = 399 + 301, Uranus, Neptune, Pluto + Charon). With the
  Moon: Earth + Moon = EMB to 2e-16 AU; the Moon is added last, planet indices kept; ias15 only.
- yr2pi = 365.25689833 d (Gaussian year); k against IAU 9e-12. The simulation's GM_sun differs from k² by −5e-12, so
  elements given with μ = k² (MPC, AstDyS, Horizons) convert consistently.
- `times/(2π)` is applied exactly once everywhere (classifier, periodograms, planet modes, free elements, proper
  angle, coherence, plotters, CSV); `restore` multiplies back.
- SABA(10,6,4) at dt = 1 matches ias15 for Earth, Jupiter and a main-belt orbit over 2 kyr; safe_mode 0 and 1 agree.
- MPCORB: μ = k² holds on all 1.58 M rows within rounding; epochs TT; standard epoch equal to AstDyS's. MPCORB vs
  AstDyS (objects 1–32, same epoch): |Δa| ≤ 1.1e-7 AU. 463 from both catalogues over 2 kyr: Δa ≤ 5.8e-7 AU, the
  4J-2S-1 angle within 0.012°. Eros, 719 and 887 propagated by the package reproduce MPC's own propagation
  (Δa ≤ 5e-8 AU, Δλ ≤ 1.3e-5°).
- Covariances: the AstDyS/NEODyS OEF `COV` has λ in degrees; SBDB's covariance is at the solution epoch with its own
  nominal; MPC eigenvalues are square-root eigenvalues.

## MMR

- d'Alembert, sign and gcd checks, angle formulas, planet indexing (with and without the Moon).
- Two-body resonant axes match Standish within ≤ 0.0012 AU (Neptune 0.016); three-body axes match Nesvorný &
  Morbidelli 1998, Table 1, within 1.6e-3 AU (4J-2S-1: 2.3985 vs 2.3977). `PLANETS_AXIS` = Standish 3000 BC–3000 AD.
- 153 Hilda and 1911 Schubart librate in 3J-2.

## Secular

- g5…g8, s6…s8 match Knežević & Milani / AstDyS. `PLANETARY_PHASES` against Laskar 1990 (Tables II–III): 0.5–2°.
- Invariable plane = Souami & Souchay 2012 (1.5787°, 107.5822°); the model's own plane 1.5788°/107.575°. Linear
  subtraction against rotation: ≤ 0.1° at i = 10°.
- All `SECULAR_FORMULAS` satisfy d'Alembert; `combinations.near_resonant` matches a full enumeration.
- `free_elements` recovers e_free = 0.1000, b_ϖ = 40.000, b_Ω = −45.000 on synthetics at 0.5/1/10 Myr;
  `planet_modes` recovers phases to 0.05–2°.

## ZLK

- c1 = (1 − e²) cos² i; the code's `c` (Lidov's c2) with the < 0 criterion matches RK4 integration of the quadrupole
  equations. Vinogradova 2024 confirms the shifted ZLK centres (Hildas 0/180°, Trojans ±60°).

## Signal processing

- filtfilt gain: the digital law G = 1/(1 + (tan(π dt/P)/tan(π dt/P_c))^(2N)), zero phase
  (`filtering.butterworth_gain`); 1/(1 + (P_c/P)^(2N)) is its dt → 0 limit. No aliasing at Hilda/Griqua (identical spectra at a 2-yr step).
- Cross-spectrum phase sign: σ–a target +90° (Hilda: 271.7 yr at +89.9°). γ²_crit per bin holds for non-overlapping
  segments only. e–i antiphase follows from the Hamiltonian; the |di/de| formula is correct.
- The circular mean is correct across 0/2π; the FFT autocorrelation equals `np.correlate`.

## I/O and batches

- summary.csv status equals `body.statuses`; `restore` with `save='all'` reproduces statuses to 1.7e-13.
- `_append_csv` locks: 8 processes × 40 writes without loss or duplicates; since W24 the holder removes `<table>.lock`
  (re-checked under a 4-process stress test that breaks a naive unlink).
- Styles are applied only through `rc_context`; no figure leaks over 40 iterations; FAIR coordinates and time reversal
  in the diagnostics are correct.
