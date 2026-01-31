"""Command-line interface for resonances package."""

import click
import resonances


@click.group()
def cli():
    """Resonances CLI - Asteroid resonance analysis toolkit."""
    pass


@cli.command()
@click.argument('resonance_str')
@click.option('--threshold', type=float, default=0.3, help="Threshold for candidate finding.")
@click.option('--threshold_negative', type=float, default=None, help="Threshold < 0 for candidate finding.")
@click.option('--threshold_positive', type=float, default=None, help="Threshold > 0 for candidate finding.")
@click.option('--numbered', is_flag=True, default=False, help="Return only numbered asteroids.")
@click.option('--debug', is_flag=False, default=False, help="Print debug information.")
def secular_candidates(resonance_str, threshold, threshold_negative, threshold_positive, numbered, debug):
    """Find secular resonance candidates for a given resonance.

    Examples:

        resonances secular-candidates z1 --threshold=0.3 --debug

        resonances secular-candidates g-g6 --threshold_negative=-6 --threshold_positive=0.3 --numbered
    """
    click.echo(f"Running secular candidates for resonance: {resonance_str}")
    resonance = resonances.create_resonance(resonance_str)
    print(f"Working with {resonance.to_s()}") if debug else None
    if threshold_negative is not None or threshold_positive is not None:
        finder = resonances.SecularResonanceFinder(
            threshold_negative=threshold_negative,
            threshold_positive=threshold_positive,
        )
    else:
        finder = resonances.SecularResonanceFinder(threshold=threshold)
    candidates = finder.find_candidates(resonance)
    print(f"Found {len(candidates)}") if debug else None

    if numbered:
        candidates = candidates[candidates['name'].apply(lambda x: str(x).isdigit())]
    print(f"Numbered candidates: {len(candidates)}") if debug else None
    print(candidates["name"].tolist())


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def find(ctx):
    """Find and analyze resonances for given asteroids.

    All parameters passed dynamically. Required: --asteroids

    List format: Use comma-separated values (no shell escaping needed)
        --asteroids=463,490
        --planets=Jupiter,Saturn
        --type=mmr,secular

    Examples:

        # Single asteroid
        resonances find --asteroids=463

        # Multiple asteroids (comma-separated)
        resonances find --asteroids=463,490,588

        # With common parameters
        resonances find --asteroids=463,490 --planets=Jupiter,Saturn --integration-years=50000

        # Full example with save/plot
        resonances find --asteroids=463 --planets=Mars,Jupiter,Saturn --integration-years=100000 --type=mmr,secular --save=all --plot=resonant

        # With advanced parameters
        resonances find --asteroids=463 --dt=0.01 --integrator=SABA --save-path=./results --tmax=1000000
    """
    kwargs = _parse_extra_args(ctx.args)

    if 'asteroids' not in kwargs:
        click.echo("Error: --asteroids is required", err=True)
        ctx.exit(1)

    click.echo(f"Running with parameters: {kwargs}")

    sim = resonances.find(**kwargs)
    sim.run(progress=True)

    click.echo(f"✓ Complete! Results in: {sim.config.save_path}")


@cli.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def check(ctx):
    """Check specific resonances for given asteroids.

    All parameters passed dynamically. Required: --asteroids, --resonances

    List format: Use comma-separated values (no shell escaping needed)
        --asteroids=463,490
        --resonances=4J-2S-1,3J-1S-1
        --planets=Jupiter,Saturn

    Examples:

        # Single asteroid, single resonance
        resonances check --asteroids=463 --resonances=4J-2S-1

        # Multiple asteroids and resonances (comma-separated)
        resonances check --asteroids=463,490 --resonances=4J-2S-1,3J-1S-1

        # With integration time
        resonances check --asteroids=463 --resonances=4J-2S-1 --integration-years=50000

        # Full example
        resonances check --asteroids=463,490 --resonances=4J-2S-1,3J-1S-1 --planets=Jupiter,Saturn,Uranus --save=all --plot=all

        # With advanced parameters
        resonances check --asteroids=463 --resonances=4J-2S-1 --dt=0.01 --integrator=SABA
    """
    kwargs = _parse_extra_args(ctx.args)

    if 'asteroids' not in kwargs:
        click.echo("Error: --asteroids is required", err=True)
        ctx.exit(1)

    if 'resonances' not in kwargs:
        click.echo("Error: --resonances is required", err=True)
        ctx.exit(1)

    click.echo(f"Running with parameters: {kwargs}")

    sim = resonances.check(**kwargs)
    sim.run(progress=True)

    click.echo(f"✓ Complete! Results in: {sim.config.save_path}")


# flake8: noqa: C901
def _parse_param(value):
    """Parse CLI parameter - handles comma-separated lists, single values, JSON.

    Supports:
        - Comma-separated: "463,490" -> [463, 490]
        - Comma-separated strings: "Jupiter,Saturn" -> ["Jupiter", "Saturn"]
        - Single values: "463" -> 463
        - JSON (if quoted): "[463,490]" -> [463, 490]
        - Booleans: "true", "false", "yes", "no"
    """
    if not isinstance(value, str):
        return value

    # Try comma-separated list
    if ',' in value:
        parts = [v.strip() for v in value.split(',')]
        result = []
        for part in parts:
            # Try int
            try:
                result.append(int(part))
                continue
            except ValueError:
                pass
            # Try float
            try:
                result.append(float(part))
                continue
            except ValueError:
                pass
            # Keep as string
            result.append(part)
        return result

    # Try parsing as int
    try:
        return int(value)
    except ValueError:
        pass

    # Try parsing as float
    try:
        return float(value)
    except ValueError:
        pass

    # Try parsing as boolean
    if value.lower() in ('true', 'yes', '1'):
        return True
    if value.lower() in ('false', 'no', '0'):
        return False

    # Return as string
    return value


def _parse_extra_args(args):
    """Parse extra CLI arguments into a dictionary.

    Handles formats like:
        --key=value
        --key value
        --long-key=value (converts to long_key)
    """
    kwargs = {}
    i = 0
    while i < len(args):
        arg = args[i]

        if arg.startswith('--'):
            # Remove -- prefix
            arg = arg[2:]

            # Check if it's --key=value format
            if '=' in arg:
                key, value = arg.split('=', 1)
                key = key.replace('-', '_')  # Convert kebab-case to snake_case
                kwargs[key] = _parse_param(value)
                i += 1
            # Check if next arg is the value (--key value format)
            elif i + 1 < len(args) and not args[i + 1].startswith('--'):
                key = arg.replace('-', '_')
                value = args[i + 1]
                kwargs[key] = _parse_param(value)
                i += 2
            else:
                # Flag without value, treat as True
                key = arg.replace('-', '_')
                kwargs[key] = True
                i += 1
        else:
            i += 1

    return kwargs


if __name__ == '__main__':
    cli()
