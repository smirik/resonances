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
`LidovKozaiParameters.evaluate` rotates nothing: it uses the angles it is given. The package reads them from the
**invariable plane** (`evaluate_invariable`: the angular momentum and the perihelion rotated exactly, no angle
subtracted), the plane of the planets' total angular momentum, as an approximate Solar-system stand-in for the
perturber's plane. The summary columns `c1`, `c2`, `h_quad` are these: osculating, at the epoch, invariable plane. From
the ecliptic $c_1$ moves by up to 0.02 and the sign of $c_2$ can flip (1373 Cincinnati: $+0.0014$ ecliptic, $-0.0011$
invariable). The quadrupole energy,
without its constant factor, is

$$
h_\text{quad} = (2 + 3e^2)(3\cos^2 i - 1) + 15 e^2 \sin^2 i \cos 2\omega = 6c_1 - 2 + 30 c_2 .
$$

```python
--8<-- "lidov_kozai.py:integrals"
```

```text
1373 Cincinnati  c1 = 0.521  c2 = -0.0011  h_quad = 1.091
3040 Kozai       c1 = 0.474  c2 = -0.0021  h_quad = 0.778
15527            c1 = 0.855  c2 = +0.0092  h_quad = 3.404
```

These numbers are **diagnostics, not the verdict**. The criterion is exact only for $a/a_\text{pert} \to 0$; a
main-belt asteroid has $a/a_J \approx 0.5$ and several perturbers, and a value this close to zero (1373's
$c_2 = -0.0011$; $+0.0014$ from the ecliptic) says nothing on its own. The sign of $h_\text{quad}$ is not a criterion
either: $e = 0.1$, $i = 70°$, $\omega = 0$ gives $h_\text{quad} < 0$ while $\omega$ circulates ($c_2 > 0$). The
osculating elements are used here, at one instant.

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

On a librating $\omega$ the gate's outcome sets the status: confirmed keeps `resonant`, kinematic gives
`non_resonant` / `kinematic_libration`, no verdict gives `uncertain` / `gate_unavailable`; the full rule, also for
episodes and short records: [Secular and Lidov-Kozai](physics/secular-zlk.md#zlk-free-omega).
[Gallery](physics/gallery.md#zlk-kinematic) shows the Jupiter Trojan 591986, whose osculating $\omega$ librates
while its free $\omega$ winds.

Four measurements sit next to the status and change none of it ([Free elements](free-elements.md#beyond-the-gate)):

- **e–i coupling** at the row's decisive period (`zlk_ei_coupling`, every row whatever its status): `zlk` when $e$
  and the inclination to the invariable plane trade in antiphase at the amplitude of constant $\sqrt{1-e^2}\cos i$,
  `forced_beat` when they move in phase at the amplitude of constant $\sqrt{1-e^2}(1-\cos i)$ (a forced
  eccentricity beating against the free one), `none`, or `not_measured`. Coupling is not libration: 15527 circulates
  and still trades $e$ for $i$ at its 2.05-kyr $\cos 2\omega$ harmonic. `confirmed_ei` (and `check_zlk_ei`) is
  `zlk_ei_coupling == zlk`.
- **Attribution** of a kinematic libration (`zlk_forcing_*`): the forced mode that holds $\varpi$ (591986: g5 with the
  modes merged into it), when the free circle stays off the origin, the predicted beat is the libration period and
  the node follows the mode (an episodic row demoted inside its episodes: scope `episodes`, the node free to turn);
  otherwise "forcing not attributed". The figure states it as the cause and draws panel (a′).
- **Node lock** (`zlk_node_lock`, a flag): the node's rate one line with a forced mode's and $\Omega - \varpi_j$
  librating through at least one full cycle (591986 over 4 Myr: g5; over 500 kyr not resolved).
- **The slow line of the centre** (`zlk_centre_*`): its period and half-amplitude, and its origin (591986: the g6−g5
  beat of the forced vector, 54 kyr, ±22°).
- **Co-orbitals** of Mercury to Mars have no gate verdict (`zlk_gate_detail` = `coorbital_unfitted_forcing`): their
  forced term is the planet's own vector, which the split does not fit; Jupiter's Trojans are covered.
