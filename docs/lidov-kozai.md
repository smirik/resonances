# Lidov-Kozai resonances

In the classical prograde quadrupole problem the Lidov-Kozai (ZLK) resonance is a libration of the argument of
perihelion $\omega$ about 90° or 270°, with $e$ and $i$ varying in antiphase at constant $\sqrt{1-e^2}\cos i$.
Inside a mean-motion resonance the centre can shift, so the classifier tests that $\omega$ and its free counterpart
are confined ([free-omega gate](free-elements.md)), not that the centre is at 90° or 270°; the e–i phase is a flag
([Coherence](coherence.md)). The package checks it as a resonance with the angle
$\omega$: `resonances.LidovKozaiResonance()`, key `LK`. A run may combine it with MMRs and secular resonances.

## Lidov's integrals

In the quadrupole approximation of the doubly averaged, circular restricted three-body problem (an inner body, a
distant perturber), $a$ is constant and the motion has two more integrals (Lidov 1962, eqs. 58–60; Kozai 1962,
eqs. 26–28):

$$
c_1 = (1 - e^2)\cos^2 i, \qquad c_2 = e^2\left(\tfrac{2}{5} - \sin^2 i \,\sin^2\omega\right).
$$

Here $i$ is the inclination to the perturber's orbital plane and $\omega$ is measured from the node on that plane;
$c_1$ is then the squared normalised component of the angular momentum normal to the perturber's orbit. In that limit
$\omega$ librates about 90° or 270° when $c_1 < 3/5$ and $c_2 < 0$, and circulates otherwise; for an initially
circular orbit this needs $\arccos\sqrt{3/5} \approx 39.2° < i < 140.8°$ (not a classifier threshold).
`LidovKozaiParameters` rotates nothing: it uses the angles it is given. The ecliptic osculating elements below are
an approximate Solar-system diagnostic. The quadrupole energy,
without its constant factor, is

$$
h_\text{quad} = (2 + 3e^2)(3\cos^2 i - 1) + 15 e^2 \sin^2 i \cos 2\omega = 6c_1 - 2 + 30 c_2 .
$$

```python
--8<-- "lidov_kozai.py:integrals"
```

```text
1373 Cincinnati  c1 = 0.545  c2 = +0.0015  h_quad = 1.311
3040 Kozai       c1 = 0.452  c2 = -0.0026  h_quad = 0.635
15527            c1 = 0.848  c2 = +0.0093  h_quad = 3.367
```

These numbers are **diagnostics, not the verdict**. The criterion is exact only for $a/a_\text{pert} \to 0$; a
main-belt asteroid has $a/a_J \approx 0.5$ and several perturbers, and 1373 Cincinnati librates although $c_2 > 0$.
The sign of $h_\text{quad}$ is not a criterion either: $e = 0.1$, $i = 70°$, $\omega = 0$ gives
$h_\text{quad} < 0$ while $\omega$ circulates ($c_2 > 0$). The osculating elements are used here, at one instant.

## Checking a body

```python
--8<-- "lidov_kozai.py:run"
```

The cycles of $\omega$ last tens of thousands of years, so a verdict needs a long record: 500 000 years with a saved
point every 10 years is a good default. `SABA(10,6,4)` with `dt = 0.5` (29 days) is safe for perihelion distances
$q \gtrsim 1.3$ au; closer to the planets use `ias15` ([dry-run](dry-run.md)). The short run above gives
`uncertain` / `insufficient_time`; over 500 000 years 3040 Kozai is `resonant` with `certain` confidence
([Gallery](physics/gallery.md#zlk-resonant)).

The angle is a series like any other:

```python
--8<-- "lidov_kozai.py:angle"
```

## How the status is decided

The angle $\omega$ goes through the same geometry as every other angle (turning points, cycles, episodes;
[What the angle does](physics/geometry.md)). A libration of the *osculating* $\omega$ is not enough: the forced
eccentricity, driven by the planets, can hold the osculating perihelion while the body's own (free) perihelion
circulates. The **free-ω gate** splits $e$ and $i$ into forced and free parts and asks whether the free
$\omega$ is confined ([Free elements](free-elements.md), [Secular and Lidov-Kozai](physics/secular-zlk.md)).

On a libration of $\omega$:

| the gate's outcome | status | reason |
|---|---|---|
| confirmed: the free $\omega$ is confined | `resonant` | `dynamic_libration` |
| kinematic: the free $\omega$ winds | `non_resonant` | `kinematic_libration` (origin `kinematic`) |
| no outcome (too short a window, an undefined phase, a residual at a planetary rate, no free elements) | `uncertain` | `gate_unavailable` |

On episodes, a drift or too few cycles only a kinematic outcome changes the status, and only when the free
eccentricity vector never circles the origin; the status then becomes `non_resonant` / `kinematic_libration`.

A kinematic libration is a measurement; `gate_unavailable` is the absence of one. Never count them together.
[Gallery](physics/gallery.md#zlk-kinematic) shows the Jupiter Trojan 591986, whose osculating $\omega$ librates
while its free $\omega$ winds.

The e–i cross-spectrum ([Cross-spectral coherence](coherence.md)) adds the flag `confirmed_ei` when $e$ and $i$
oscillate in antiphase at the libration period; it does not change the status.
