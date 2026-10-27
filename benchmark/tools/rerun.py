"""Re-integrate one benchmark body in another regime (step, span) and write the result as a variant example.

    uv run python benchmark/tools/rerun.py --name "1999 UQ" --source nea \
        --resonances "7E+2M-9+0+0+0,8M-9J-4+0+0+5" --span 20000 --step 1 --variant step1yr \
        --regime B-nea --like mmr3__1999_UQ__7E+2M-9+0+0+0

Uses integrate.run_task, so the example layout, meta.json and provenance are those of the batches. The source
example (`--like`) gives the rtype and the old labels; labels.csv is not touched (a rerun is not a new label).
Every rerun is agreed with the author first (CLAUDE.md); the list lives in RERUNS.md. A body rerun before keeps its
done marker; the new one is done/R__<body>__<variant>.json. The run's run.json keeps the bodies already in it: the
rerun's body and elements source are added to its lists (bench_io.merge_run_record).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import integrate as ig  # noqa: E402  (sets the BLAS thread variables before numpy)
import bench_io as io  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--source', required=True, choices=['nea', 'astdys', 'coorbital', 'clone', 'series'])
    ap.add_argument('--resonances', required=True, help='comma list')
    ap.add_argument('--span', type=float, required=True, help='years')
    ap.add_argument('--step', type=float, required=True, help='output step, years')
    ap.add_argument('--variant', default=None, help='example id suffix; none for a resonance the body has no example of yet')
    ap.add_argument('--regime', required=True, help='integrate.regime_settings name: A, B-nea, B-pair, C, D-saba, D-ias15')
    ap.add_argument('--like', required=True, help='source example id (rtype, labels)')
    ap.add_argument('--batch', default='R')
    ap.add_argument('--rtype', default=None, help='override the rtype of --like (e.g. a two-body owner of a three-body angle)')
    args = ap.parse_args()

    src = json.load(open(io.EXAMPLES / args.like / 'meta.json'))
    if args.source == 'series':  # osculating elements of the --like series at t = 0, its epoch
        row = io.load_example(args.like, with_planets=False).series.iloc[0]
        got = (
            {k: float(row[k]) for k in ('a', 'e', 'inc', 'Omega', 'omega', 'M')},
            float(src['regime']['epoch_jd']),
            f'first row of {args.like}',
        )
    else:
        got = getattr(ig.Elements(), args.source)(args.name)
    if got is None:
        raise SystemExit(f'{args.name}: not found in source {args.source}')
    elements, epoch, el_source = got
    settings = ig.regime_settings(args.regime)
    settings.update(span_yr=args.span, step_yr=args.step)
    spec = ig.BodySpec(
        key=args.name,
        batch=args.batch,
        regime=args.regime,
        elements=elements,
        epoch_jd=epoch,
        elements_source=el_source,
        **settings,
        note=f'rerun of {args.like}: {args.resonances}, step {args.step} yr, span {args.span} yr',
    )
    spec.run_id = spec.run_key()
    for res in args.resonances.split(','):
        rtype = args.rtype or src['rtype']
        eid = io.make_example_id(rtype, args.name, res, args.variant)
        if (io.EXAMPLES / eid).exists():
            raise SystemExit(f'{eid} exists')
        spec.examples.append(
            ig.ExampleSpec(
                example_id=eid,
                candidate_id=src.get('candidate_id'),
                resonance=res,
                rtype=rtype,
                needs_integration='rerun',
                labels=dict(src.get('labels', {}), rerun_of=args.like),
            )
        )
    runs_dir = ig.RUNS_ROOT / args.batch
    opts = {'runs_dir': str(runs_dir), 'data_root': None, 'dry_run': False, 'years_override': None}
    ig.prefetch_solar_systems([spec])
    ig.write_run_records([spec], runs_dir, opts, merge=True)
    task = ig.make_tasks([spec], 1)[0]
    # run_task names its done marker after the body; a body rerun before keeps its marker, the new one gets the
    # variant (verify reads every marker of done/).
    marker = runs_dir / 'done' / ig.done_name(args.batch, args.name)
    previous = marker.read_bytes() if marker.exists() else None
    res = ig.run_task(task, opts)
    if previous is not None:
        suffix = args.variant or io.safe_token(args.resonances)
        marker.rename(marker.with_name(ig.done_name(args.batch, f'{args.name}__{suffix}')))
        marker.write_bytes(previous)
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
