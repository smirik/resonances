<!-- Generated from docs/matrix.md by docs/tools/build_skill.py (make skill). Edit the docs, not this file. -->

# Resonance matrices

A resonance matrix is a catalogue of mean-motion resonances with their nominal resonant semi-major axes $a_{\rm res}$.
It is used to find the resonances near a body before integrating it.

- **Two-body** (`TwoBody`): Kepler's third law at the commensurability $k_P n_P + k n = 0$,
  $a_{\rm res} = a_P\,(-k/k_P)^{2/3}$, with $a_P$ the planet's semi-major axis.
- **Three-body** (`ThreeBody`): one perturbative correction of the commensurability
  $k_1 n_1 + k_2 n_2 + k\,n + p\,\dot\varpi = 0$ (the planets' own precession neglected). First the Keplerian
  $n_K = -(k_1 n_1 + k_2 n_2)/k$ and $a_0 = (K/n_K)^{2/3}$; then the body's perihelion rate $g_0$ at $a_0$, and
  $n = -(k_1 n_1 + k_2 n_2 + p\,g_0)/k$, $a_{\rm res} = (K/n)^{2/3}$. Here $n_j = K a_j^{-3/2}$ (rad/day,
  $K = 0.0172020989484$, $a$ in au) and, summed over all planets with $\mu_j = m_j/M_\odot$ and $D = 365.2422$ days,
  $g_0 = \frac{3\pi}{2D}\sum_j \mu_j F_j$, $F_j = a_0^{3/2}/a_j^3$ for $a_0 \le a_j$ and
  $F_j = a_j^2/[a_0^{7/2}(1+\mu_j)^{3/2}]$ for $a_0 > a_j$ (the leading terms $\tfrac34 n\mu_j (a/a_j)^3$ of an outer planet and $\tfrac34 n\mu_j (a_j/a)^2$ of an inner one).
  $\dot\varpi$ is not iterated to the corrected axis.

These are nominal values from fixed planetary axes. The classifier does not use them: it measures the exact
resonance $a_*$ of each angle from the integration ([The resonant pendulum](physics/pendulum.md)).

## Initialisation

When the app needs to know the value of the resonant semi-major axis for the first time, it calculates it and stores it in the file (the config items `MATRIX_3BODY_FILE`, `MATRIX_2BODY_FILE` and `MATRIX_2BODY_RETRO_FILE`). The files have the following structure:

1. The number of an item
1. The short notation of resonance (i.e. `4J-2S-1` or `1J-1`)
1. The planet(s), one per column (i.e. `Jupiter` and `Saturn`)
1. Two or three integers depending on the type of a MMR (p, q and k for a retrograde one)
1. The order of a resonance
1. The value of the resonant semi-major axis

The first line of the file is a stamp with the class and the catalog version (e.g. `# TwoBodyMatrix v2`). If the app generates the file, it will use it next time; a file with another stamp, written by an older version of the package, is rebuilt automatically. If you change the `MATRIX_*` limits, delete the output file to recalculate the matrix.

Next to the file the app writes its build record, `<file>.meta.json` (e.g. `cache/mmr-2body.csv.meta.json`): the `MATRIX_*` limits, the planets, the catalog version and the package version. When a cached catalog is read, the record is compared with the current settings. Other limits or planets give a warning in the log naming the file and each differing setting (cached and current); a file without a record (written by an older version) gives one warning that its limits are unknown. The file is still read as it is, nothing is rebuilt or deleted: the paths are relative to the working directory, so a run started elsewhere can silently use a catalog built with other limits, and the warning is how you see it. To rebuild, delete the file (or call `TwoBodyMatrix.load(reload=True)`), or point `MATRIX_*_FILE` to another file. The retrograde catalog holds its limits in the stamp and is rebuilt when they change.

### Prograde resonances

`TwoBody` and `ThreeBody` are resonances of prograde orbits. A two-body resonance `m_p P + m` (angle `m_p λ_p + m λ + p ϖ`) needs `n = −m_p/m · n_p > 0`, so the body's coefficient `m` must be negative: `3J-2` is a resonance, `3J+2` is not. `TwoBody.calculate_resonant_axis` raises for `m ≥ 0`, and `TwoBodyMatrix` does not list such combinations. Their angle circulates at about `6 n_J` for `3J+2`, far above the Nyquist rate of any reasonable output step, so its series would be pure aliasing. `ThreeBody` likewise raises when the mean motion comes out negative.

### Retrograde resonances

A body with $i > 90°$ can be in a p/−q resonance with a planet (Morais & Namouni 2013a, b; Li, Huang & Gong 2019,
A&A 630, A60): $q n = p n'$, at the same axis as the prograde p/q, $a = (q/p)^{2/3} a'$. With
$\varpi^* = \omega - \Omega$ and $\lambda^* = M + \omega - \Omega$ the arguments are

$$
\varphi = q\lambda^* - p\lambda' - (p + q - 2k)\,\varpi^* + 2k\,\Omega = qM + (2k - p)\,\omega + p\,\Omega - p\lambda',
\qquad 0 \le 2k \le p + q .
$$

Each k is a separate angle; the order is p + q, the amplitude $\propto \cos^{2k}(i/2)\,e^{p+q-2k}$. Near the ecliptic k = 0 dominates, at polar orbits higher k do (2011 KT19 in 7/−9 with Neptune: k = 6).

Notation: `<p><planet>-<q>r<k>`, planet first, k required. `1J-3r0` is 2000 DG8 (1/−3 with Jupiter), `1J-1r1` is 2015 BZ509 (Ka'epaoka'awela), `7N-9r6` is 2011 KT19. `to_s()` and `to_short()` give the same string; it is the key in summary.csv and the data files.

```python
import resonances

mmr = resonances.create_mmr('1J-3r0')  # or RetrogradeTwoBody(p=1, q=3, k=0, planet='Jupiter')
# 2_000 years here; 10_000 and more for a verdict
sim = resonances.check('2000 DG8', '1J-3r0', integrator='ias15', integration_years=2_000, Nout=2_001)
sim.run()
```

- `RetrogradeTwoBodyMatrix` lists every coprime p, q ≤ `MATRIX_2BODY_RETRO_COEF_MAX` (15) with p + q ≤ `MATRIX_2BODY_RETRO_ORDER_MAX` (30) and every k: 143 resonances per planet as in Li et al., 1198 angles. The cache stamp holds both limits, so changing either rebuilds the cached catalog.
- `find` gives a body with `i > 90°` only retrograde candidates, within `sigma_retro = 0.5` au (Li et al.), and any other body only prograde ones. In `find_mmrs`, a single `sigma` overrides `sigma_retro` too, and `retrograde=True` with `mmr_type='two-body'` or `'three-body'` is a `ValueError`. `check` accepts any combination; a resonance of the other direction is only warned about, since orbits can flip. A run whose inclination crosses 90° is warned about too.
- The text of Li et al. writes the condition as `pn − qn' ≈ 0`; their eqs. (1), (2) and the objects give `qn − pn' ≈ 0`.
- The angle is built from the body's Ω, ω and M only. For `i > 90°` REBOUND returns `l = −λ*` and `pomega = −ϖ*`; an angle from those runs off on a pure Kepler orbit. The `longitude` (REBOUND's `l`) and `varpi` (Ω + ω) columns of the data files are therefore in different conventions for such orbits, and the FAIR plot is not drawn for retrograde resonances.
- The libration periods of retrograde Centaurs are 0.7–3 kyr (2000 DG8: 0.71 kyr, 2015 BZ509: 0.70 kyr). The classifier chooses the filter cutoff of each angle itself, from its synodic periods and the shortest libration it allows ([From the integration to the angle](physics/signal.md)), so these librations are kept; what they need is an output step of 1–2 yr (the [dry-run](dry-run.md) checks it). `oscillations_cutoff` changes only the body-level filter of the `evolution` figure and the periodograms: set it to `0.005` (200 yr) to see these librations there too. Slow TNO librations (2011 KT19: a period of 41 kyr) need a long run.

## Usage

There are three classes, `ThreeBodyMatrix`, `TwoBodyMatrix` and `RetrogradeTwoBodyMatrix`, responsible for building the resonant matrices. You might use any of them to find the possible resonances close to the given value of the semi-major axis.

The following code will print the list of three-body MMRs close to the value `2.39` AU of the semi-major axis.

```python
from resonances.mmr.three_body_matrix import ThreeBodyMatrix  # noqa: E402

mmrs = ThreeBodyMatrix.find_resonances(2.39, planets=['Jupiter', 'Saturn', 'Uranus'])
for mmr in mmrs:
    print(mmr.to_short())
```

Note that there is an optional parameter `planets` representing the array of the planets involved in the MMR. The app will try all possible configurations. In this particular case, there are three combinations: Jupiter and Saturn, Jupiter and Uranus, and Saturn and Uranus.

If you omit this parameter, the app will use all possible combinations of all planets in the Solar System.

The same workflow is for two-body MMRs. The following code will find all two-body MMRs with Jupiter near `5.2` AU (Trojans, actually):

```python
from resonances.mmr.two_body_matrix import TwoBodyMatrix  # noqa: E402

mmrs = TwoBodyMatrix.find_resonances(5.2, planets=['Jupiter'])
for mmr in mmrs:
    print(mmr.to_short())
```

The method `find_resonances` has one more optional parameter `sigma` (default `0.1` au): the largest distance between the given semi-major axis and the resonant one.

The following code will find all two-body MMRs with Jupiter between `4.2` and `6.2` AU:

```python
mmrs = TwoBodyMatrix.find_resonances(5.2, planets=['Jupiter'], sigma=1.0)
for mmr in mmrs:
    print(mmr.to_short())
```

`resonances.find` and `find_mmrs` have their own tolerances: `sigma2 = 0.1` au for two-body, `sigma3 = 0.02` au for three-body and `sigma_retro = 0.5` au for retrograde resonances. A wide tolerance only adds candidates; the integration decides.

### Candidates from a catalogue

The bodies that might be in a given resonance, from their semi-major axis alone (an integration decides), come from
`search_by_axis` of the `astdys` package (AstDyS) or of `resonances.mpcorb` (Minor Planet Center, see
[Configuration](config.md#minor-planet-center-sourcempc)); both take `sigma` (default 0.1 au).

The following code will print the numbers of asteroids that might be in the three-body resonance `4J-2S-1`:

```python
from resonances import mpcorb  # noqa: E402

df = mpcorb.search_by_axis(resonances.create_mmr('4J-2S-1').resonant_axis)  # or astdys.search_by_axis (AstDyS)
print(df['number'].tolist())
```

Note that the method `search_by_axis` returns [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), not simply a list or a dict. The DataFrame consists of the number of an asteroid and its orbital elements. You can use `df.head()` to see its structure.
