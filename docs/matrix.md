# Resonance Matrices

The Resonance Matrices allows us to find the preliminary (theoretical) value of a resonant semi-major axis for the resonance. While for a two-body mean motion resonance the calculation follows obviously from the Third Kepler Law, it is not so straightforward for three-body mean motion resonances.

This app calculates the value of the resonant semi-major axis for three-body resonance based on the paper: _Smirnov, E. A., Dovgalev, I. S. & Popova, E. A. Asteroids in three-body mean motion resonances with planets. Icarus (2017) doi:10.1016/j.icarus.2017.09.032._ The algorithms are in the classes `TwoBody` and `ThreeBody`.

## Initialisation

When the app needs to know the value of the resonant semi-major axis for the first time, it calculates it and stores it in the file (the config items `matrix.3body.file` and `matrix.2body.file`). The files have the following structure:

1. The number of an item
1. The short notation of resonance (i.e. `4J-2S-1` or `1J-1`)
1. The planet(s), one per column (i.e. `Jupiter` and `Saturn`)
1. Two or three integers depending on the type of a MMR
1. The order of a resonance
1. The value of the resonant semi-major axis

If the app generates the file, it will use it next time. If you need to recalculate the matrix, simply delete the output file.

## Usage

There are two classes, `ThreeBodyMatrix` and `TwoBodyMatrix`, responsible for building the resonant matrices. You might use any of them to find the possible resonances close to the given value of the semi-major axis.

The following code will print the list of three-body MMRs close to the value `2.39` AU of the semi-major axis.

```python
from resonances.matrix.three_body_matrix import ThreeBodyMatrix

mmrs = ThreeBodyMatrix.find_resonances(2.39, planets=['Jupiter', 'Saturn', 'Uranus'])
for mmr in mmrs:
    print(mmr.to_short())
```

Note that there is an optional parameter `planets` representing the array of the planets involved in the MMR. The app will try all possible configurations. In this particular case, there are three combinations: Jupiter and Saturn, Jupiter and Uranus, and Saturn and Uranus.

If you omit this parameter, the app will use all possible combinations of all planets in the Solar System.

The same workflow is for two-body MMRs. The following code will find all two-body MMRs with Jupiter near `5.2` AU (Trojans, actually):

```python
from resonances.matrix.two_body_matrix import TwoBodyMatrix

mmrs = TwoBodyMatrix.find_resonances(5.2, planets=['Jupiter'])
for mmr in mmrs:
    print(mmr.to_short())
```

The method `find_resonance` has one more optional parameter `sigma` representing the maximum distance between the given value of the semi-major axis and the value of a resonant semi-major axis of a possible MMR.

The following code will find all two-body MMRs with Jupiter between `4.2` and `6.2` AU:

```python
from resonances.matrix.two_body_matrix import TwoBodyMatrix

mmrs = TwoBodyMatrix.find_resonances(5.2, planets=['Jupiter'], sigma=1.0)
for mmr in mmrs:
    print(mmr.to_short())
```

While sometimes it is useful to make `sigma` greater, the default value is `0.02`. For three-body MMRs, it is recommended to use `0.1` or `0.2` for the asteroids with high eccentricity and `0.05` otherwise.

### Mix with AstDyS

You might want to find all possible resonant asteroids for a given resonance. While you have to integrate them to confirm their status, you can get the possible candidates based only on the value of the semi-major axis. To perform this, there is a method `search_by_axis` in `astdys` class.

The following code will print the numbers of asteroids that might be in the three-body resonance `4J-2S-1`:

```python
import resonances

from resonances.data.astdys import astdys
df = astdys.search_by_axis(resonances.create_mmr('4J-2S-1').resonant_axis)
print(df['num'].tolist())
```

Note that the method `search_by_axis` returns [Pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), not simply a list or a dict. The DataFrame consists of the number of an asteroid and its orbital elements. You can use `df.head()` to see its structure.

You can easily combine this method with the Simulation to verify are these candidates in the resonance or not. An example of such implementation is described in [Console commands](console.md).
