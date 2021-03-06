# Console commands

While `resonances` are designed to be used as a package, which means that you import them into your project and manipulate the data in the way you want, there are some console commands that are made mainly as examples of what one can do with this package.

## Quick identification

If you want to check if the given asteroid is in the given resonance, you can run:

```bash
poetry run identify-quick 463 4J-2S-1
```

The first argument specifies the number of the asteroid, the second one — the corresponding resonance in a short form.

As a result, the app will integrate this asteroid and identify whether or not it is in the resonance. The results will be placed in `cache` folder: the data file, the figure, and the quick summary in `summary.csv`.

## Identify several asteroids

You might want to check several asteroids whether or not they are resonant. Moreover, for each asteroid, you might want to specify several resonances. It is possible through the command:

```bash
poetry run identify-asteroids --config=docs/examples/asteroids.json
```

You have to specify the input data in `json` format. There is a self-explanatory example in `docs/examples/asteroids.json`. Note that you may specify many possible resonances for many asteroids.

## Resonance shape

There is another example available in `docs/examples/simulation-shape.json`. It shows how one might run a simulation that includes many test particles.

```bash
poetry run simulation-shape --config=docs/examples/simulation-shape.json
```

This command creates a set of test particles specified in `simulation-shape.json`, integrate them for `100,000` years, and identify whether or not they are in the resonance.

The parameter `variations` in `json` file specifies the variations of initial data that are used to create test particles. In the example, there are three groups of variations:

1. Variate `a` between `2.389` and `2.401` (approx.) and use linear distribution with 2 points (yes, it is very small, but this is only for test purposes) and `e` between `0.0` and `0.3` with 2 points.
2. Variate `a` between `2.389` and `2.401` (approx.) and use linear distribution with 3 points and `e` between `0.3` and `0.6` with 3 points.
3. Variate `a` between `2.389` and `2.401` (approx.) and use linear distribution with 2 points and `e` between `0.6` and `0.9` with 2 points.

In total, the app will create `2*2+3*3+2*2=17` test particles. Of course, you may change these values to the appropriate (i.e. `100` points per interval).

The results of the integration will be saved in `cache/simulation-shape` directory. The main files are `summary.csv` and `ae-plane.csv`. Note that the flag `save.only.undetermined` is set to `true`. Therefore, the data and the figures will be only for the bodies whose statuses are uncertain.

## Study an asteroid

If you want to know the list of the MMRs that a given asteroid might be in, you can use the console command `identify-resonances`:

```bash
poetry run identify-resonances 588
```

This command will find all possible two-body and three-body resonances for the asteroid with the number `588`. You may specify any asteroid number that is in the AstDyS catalogue.

The command has an optional parameter `planets`, which might be involved in the MMR. For example, the following command will analyse only the resonances with Mars, Jupiter, and Saturn (and all their possible unique combinations):

```bash
poetry run ir 588 --planets="Mars, Jupiter, Saturn"
```

By default, the results are saved in `cache/identifier` directory.

## Find asteroids in the resonance

You might want to run the task, which is reversal to the previous one, --- to find asteroids in the given resonance:

```bash
poetry run asteroids-in-resonance 4J-2S-1
```

This task will automatically identify all possible resonant asteroids (the MMR: 4J-2S-1), integrate their orbits for 100,000 years, and run the identification procedure.

However, it might be the case that the number of possible resonant asteroids is large. Thus, the execution time can also be very large. To limit the number of possible resonant asteroids, there is an option `--iterations`, which limits the number of iterations.

Each iteration analyses `100` asteroids. If `iterations` is equal to `3`, then `300` asteroids only will be checked. So, this option allows limiting the results by first `100*N` objects.

```bash
poetry run asteroids-in-resonance 4J-2S-1 --iterations=3
```
