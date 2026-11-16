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

        # With 100 clones from the orbit's covariance (docs/clones.md; also --clones-source, --clones-seed)
        resonances check --asteroids=463 --resonances=4J-2S-1 --source=astdys --clones=100
    """
    kwargs = _parse_extra_args(ctx.args)

    if 'asteroids' not in kwargs:
        click.echo("Error: --asteroids is required", err=True)
        ctx.exit(1)

    if 'resonances' not in kwargs:
        click.echo("Error: --resonances is required", err=True)
        ctx.exit(1)

    click.echo(f"Running with parameters: {kwargs}")

    kwargs['resonance'] = kwargs.pop('resonances')  # finder.check names it `resonance` (one or a list)
    sim = resonances.check(**kwargs)
    sim.run(progress=True)  # Simulation.run starts with the static dry-run (warnings, OUTPUT_STEP=auto)

    click.echo(f"✓ Complete! Results in: {sim.config.save_path}")


@cli.command(name='mpc-update')
@click.option('--catalog', default=None, help="MPC catalogue: mpcorb, nea, neam00, distant, unusual or a URL (default MPC_CATALOG).")
@click.option('--force', is_flag=True, default=False, help="Download even when the server reports no change.")
def mpc_update(catalog, force):
    """Download the MPC catalogue again (source=mpc) and show its snapshot: the only way it is refreshed.

    Examples:

        resonances mpc-update

        resonances mpc-update --catalog=nea --force
    """
    from resonances import mpcorb
    from resonances.simulation.mpc_source import mpc_settings

    snapshot = mpcorb.update_catalog(force=force, settings=mpc_settings(catalog))
    click.echo(
        f"MPC catalogue {snapshot['catalog']}: {snapshot['rows']} rows, standard epoch MJD {snapshot['standard_epoch_mjd']} TT "
        f"({mpcorb.datetime_from_mjd(snapshot['standard_epoch_mjd']):%Y-%m-%d}), {snapshot['other_epoch_rows']} rows at other epochs"
    )
    click.echo(f"Last-Modified {snapshot.get('last_modified')}, sha256 {snapshot['sha256']}, from {snapshot['url']}")


@cli.group()
def skill():
    """The agent skill shipped with the package (how to set up runs and read their results)."""


@skill.command(name='install')
@click.option('--user', is_flag=True, default=False, help="Install into ~/.claude/skills (every project of this user).")
@click.option('--path', 'project', type=click.Path(file_okay=False), default=None, help="The project folder (default: the current one).")
@click.option('--force', is_flag=True, default=False, help="Replace an installed copy (e.g. after upgrading the package).")
def skill_install(user, project, force):
    """Copy the skill into <project>/.claude/skills/resonances (or ~/.claude/skills/resonances with --user).

    Examples:

        resonances skill install

        resonances skill install --user --force
    """
    from resonances.skills import SkillExistsError, install_skill

    if user and project:
        raise click.UsageError("--user and --path exclude each other")
    try:
        target = install_skill(project=project, user=user, force=force)
    except SkillExistsError as error:
        click.echo(f"The skill is already installed in {error}. Pass --force to replace it with this version.", err=True)
        raise SystemExit(1) from error
    n_files = sum(1 for path in target.rglob('*') if path.is_file())
    click.echo(f"Installed the resonances skill (package {resonances.__version__}, {n_files} files) into {target}")
    click.echo("Agents that read .claude/skills (Claude Code and others) load it in their next session.")


def parse_range(text):
    """'0:1' -> (0.0, 1.0), ':0.3' -> (None, 0.3), '2.5:' -> (2.5, None); None for no option."""
    if text is None:
        return None
    low, sep, high = str(text).partition(':')
    if not sep:
        raise click.BadParameter(f"a range is min:max (either may be empty), got {text!r}")
    try:
        return tuple(float(part) if part.strip() else None for part in (low, high))
    except ValueError as error:
        raise click.BadParameter(f"a range is min:max of numbers, got {text!r}") from error


# mpc-search's range options -> mpcorb.query's arguments ('--Q' needs its own name: click lower-cases option names)
MPC_SEARCH_RANGES = {
    'a': 'a', 'e': 'e', 'i': 'i', 'q': 'q', 'big_q': 'Q', 'h': 'H', 'node': 'node', 'peri': 'peri', 'm': 'M',
    'n_obs': 'n_obs', 'n_opp': 'n_opp', 'arc_years': 'arc_years', 'u': 'U',
}  # fmt: skip
MPC_SEARCH_FLAGS = ('neo', 'km_neo', 'pha', 'one_opp_earlier', 'critical_list')
MPC_SEARCH_SHOWN = ['number', 'designation', 'name', 'a', 'e', 'inc_deg', 'q', 'Q', 'H', 'U', 'n_opp', 'arc_years', 'orbit_type', 'flags']


def _range_option(*decls, text):
    return click.option(*decls, default=None, metavar='MIN:MAX', help=text)


@cli.command(name='mpc-search')
@_range_option('--a', text="Semi-major axis, AU (e.g. 0:1).")
@_range_option('--e', text="Eccentricity (e.g. :0.3).")
@_range_option('--i', text="Inclination, deg.")
@_range_option('--q', text="Perihelion distance a(1-e), AU.")
@_range_option('--Q', 'big_q', text="Aphelion distance a(1+e), AU.")
@_range_option('--H', 'h', text="Absolute magnitude H.")
@_range_option('--node', text="Longitude of the node, deg (350:10 wraps through 0).")
@_range_option('--peri', text="Argument of perihelion, deg (may wrap).")
@_range_option('--M', 'm', text="Mean anomaly, deg (may wrap).")
@_range_option('--n-obs', text="Number of observations.")
@_range_option('--n-opp', text="Number of oppositions.")
@_range_option('--arc-years', text="Observed arc, years (one-opposition arcs: days / 365.25).")
@_range_option('--U', 'u', text="Uncertainty parameter U, 0..9 (rows with a letter or none fail).")
@click.option('--epoch', type=click.Choice(['any', 'standard']), default='any', help="Rows at any epoch or only the standard one.")
@click.option('--numbered/--provisional', default=None, help="Only numbered objects, or only provisional designations.")
@click.option(
    '--orbit-type', default=None, help="MPC orbit types, comma-separated: atira, aten, apollo, amor, q_lt_1665, hungaria, hilda, ..."
)
@click.option('--neo/--not-neo', default=None, help="MPC's NEO flag set / clear.")
@click.option('--km-neo/--not-km-neo', default=None, help="MPC's 1-km NEO flag.")
@click.option('--pha/--not-pha', default=None, help="MPC's PHA flag.")
@click.option('--one-opp-earlier/--not-one-opp-earlier', default=None, help="1-opposition object seen at an earlier opposition.")
@click.option('--critical-list/--not-critical-list', default=None, help="Critical-list numbered object.")
@click.option('--catalog', default=None, help="MPC catalogue: mpcorb, nea, ... or a URL (default MPC_CATALOG).")
@click.option('--limit', type=int, default=20, show_default=True, help="Rows printed (the CSV gets every row).")
@click.option('--output', type=click.Path(dir_okay=False), default=None, help="Write every matching row to this CSV.")
@click.option('--ids', is_flag=True, default=False, help="Print the identifiers, comma-separated (for find --asteroids=...).")
def mpc_search(catalog, limit, output, ids, epoch, numbered, orbit_type, **criteria):
    """Select objects of the MPC catalogue by their elements and MPC's flags (resonances.mpcorb.query).

    Ranges are MIN:MAX, inclusive, either end may be empty.

    Examples:

        resonances mpc-search --a 0:1 --e :0.3 --limit 20

        resonances mpc-search --a 2.38:2.42 --numbered --U 0:2 --output lola-zone.csv

        resonances mpc-search --orbit-type=apollo --pha --H :18 --ids
    """
    from resonances import mpcorb
    from resonances.simulation.mpc_source import mpc_settings

    kwargs = {MPC_SEARCH_RANGES[key]: parse_range(criteria.pop(key)) for key in MPC_SEARCH_RANGES}
    kwargs.update(criteria)  # the flags
    types = [name.strip() for name in orbit_type.split(',') if name.strip()] if orbit_type else None
    try:
        found = mpcorb.query(epoch=epoch, numbered=numbered, orbit_type=types, settings=mpc_settings(catalog), **kwargs)
    except (TypeError, ValueError) as error:
        raise click.UsageError(str(error)) from error
    if output:
        found.to_csv(output, index=False)
        click.echo(f"{len(found)} rows written to {output}")
    if ids:
        click.echo(','.join(str(ident) for ident in mpcorb.identifiers(found)))
        return
    click.echo(f"{len(found)} rows match")
    if len(found) and limit > 0:
        click.echo(found[MPC_SEARCH_SHOWN].head(limit).to_string(index=False, float_format=lambda x: f"{x:.6g}"))


# Options of `dry-run` itself; every other argument goes to find (or check, with --resonances) as in those commands.
DRY_RUN_OPTIONS = ('pilot', 'years', 'step', 'fair', 'out')


@cli.command(name='dry-run', context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
def dry_run(ctx):
    """Check a run before integrating it (docs/dry-run.md): alias of the angles at the output step, the 'auto' step,
    the integrator and its step, the cost. Takes the arguments of find (or of check when --resonances is given) and:

        --pilot           a short fine integration per body (a few seconds each) instead of the elements only
        --years N         the span of the run (= --integration-years)
        --step auto|YR    the output step of the run (= --output-step)
        --fair            the step must resolve the FAIR plot (DRYRUN_FAIR=required)
        --out DIR         write dryrun_rows.csv, dryrun_bodies.csv and dryrun_run.json there

    Examples:

        resonances dry-run --asteroids=463 --years=100000 --step=auto

        resonances dry-run --asteroids=1865 --resonances=10E-11+0+1 --years=20000 --integrator=ias15 --Nout=4000 --pilot
    """
    kwargs = _parse_extra_args(ctx.args)
    if 'asteroids' not in kwargs:
        click.echo("Error: --asteroids is required", err=True)
        ctx.exit(1)
    options = {key: kwargs.pop(key) for key in DRY_RUN_OPTIONS if key in kwargs}
    if 'years' in options:
        kwargs['integration_years'] = options['years']
    if 'step' in options:
        kwargs['output_step'] = options['step']
    if options.get('fair'):
        kwargs['dryrun_fair'] = 'required'
    kwargs.setdefault('save', None)
    kwargs.setdefault('plot', None)
    if 'resonances' in kwargs:
        kwargs['resonance'] = kwargs.pop('resonances')
        sim = resonances.check(**kwargs)
    else:
        sim = resonances.find(**kwargs)
    report = sim.dry_run(level='pilot' if options.get('pilot') else 'static')
    click.echo(report.summary(echo=False))
    if options.get('out'):
        files = report.to_csv(str(options['out']))
        click.echo(f"Written: {', '.join(files.values())}")


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
