"""Build the author's labelling page: review_page/index.html + review_page/data.js.

The page works from file:// (no server, no network): data.js sets `window.BENCH`, images are referenced as
../data/examples/<id>/combined.png. The author's labels live in the browser's localStorage and leave it only
through "Экспорт CSV" (author_labels_<date>.csv), which tools/import_author_labels.py merges into labels.csv.

Read at build time (anything missing is reported and skipped):
  labels.csv, candidates.csv, data/examples/<id>/meta.json
  review_claude/wave*/all.json                     blind verdicts
  review_claude/wave*/adjudication-*.json,
  review_claude/adjudication/batch-*.json          adjudications; they supersede the blind verdict
  tmp/status-design-2026-10-13/prototype/author_labels.py
  experiments/nea-mmrs/research/2026-09-classifier/visual_cases.tsv
                                                   the atlas: the author's comments on 118 NEA PNGs and
                                                   Claude's reading of them, matched by name + resonance

Usage: uv run python benchmark/tools/build_review_page.py
"""

from __future__ import annotations

import ast
import csv
import datetime
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as bio  # noqa: E402

BENCH, REPO, EXAMPLES = bio.BENCH, bio.REPO, bio.EXAMPLES
OUT = BENCH / 'review_page'
REVIEW = BENCH / 'review_claude'
ATLAS_LABELS = REPO / 'tmp' / 'status-design-2026-10-13' / 'prototype' / 'author_labels.py'
ATLAS_TSV = REPO / 'experiments' / 'nea-mmrs' / 'research' / '2026-09-classifier' / 'visual_cases.tsv'

RTYPES = ['mmr2', 'mmr3', 'coorbital', 'secular', 'zlk']
STATUSES = bio.STATUSES
# Old bucket (labels.csv) -> statuses of the new model it is compatible with (`bench_io.BUCKET_STATUSES`)
BUCKET_TO_STATUS = {bucket: set(statuses) for bucket, statuses in bio.BUCKET_STATUSES.items()}
ATLAS_CLASSES = {
    'LIB': 'резонанс / либрация на всём прогоне',
    'TRN': 'транзиент (захват, эпизод)',
    'SLOW': 'медленная циркуляция / дрейф',
    'NON': 'скорее нерезонансный',
    'UNC': 'неясно / мало базы / «возможно»',
}

missing: list[str] = []


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        missing.append(f'{path.relative_to(REPO)}: нет файла')
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def items_of(data) -> list[dict]:
    """A review file is a list, or a dict holding the list under 'items'/'verdicts'."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ('items', 'verdicts', 'adjudications'):
            if isinstance(data.get(key), list):
                return data[key]
        for value in data.values():  # any other list of records
            if isinstance(value, list) and value and isinstance(value[0], dict) and 'example_id' in value[0]:
                return value
    return []


def load_json_items(paths: list[Path]) -> list[dict]:
    out = []
    for p in paths:
        try:
            out.extend(items_of(json.loads(p.read_text())))
        except Exception as exc:  # a half-written batch must not stop the build
            missing.append(f'{p.relative_to(REPO)}: не читается ({exc})')
    return out


def load_verdicts() -> dict[str, dict]:
    verdicts = {}
    waves = sorted(REVIEW.glob('wave*/all.json'))
    if not waves:
        missing.append('review_claude/wave*/all.json: нет слепых вердиктов')
    for p in waves:
        wave = p.parent.name
        for item in load_json_items([p]):
            if item.get('example_id'):
                verdicts[item['example_id']] = dict(item, wave=wave)
    return verdicts


def load_adjudications() -> dict[str, dict]:
    # Older location first, so the newer review_claude/adjudication/ wins on a repeated id.
    old = sorted(REVIEW.glob('wave*/adjudication-*.json'))
    new = sorted((REVIEW / 'adjudication').glob('batch-*.json'))
    if not new:
        missing.append('review_claude/adjudication/batch-*.json: ещё нет (взяты только разборы из wave*/adjudication-*.json)')
    adj = {}
    for p in old + new:
        for item in load_json_items([p]):
            if item.get('example_id'):
                adj[item['example_id']] = dict(item, source=str(p.relative_to(BENCH)))
    return adj


def load_disputed() -> set[str]:
    ids = set()
    for p in REVIEW.glob('*disputed*.txt'):
        ids.update(line.strip() for line in p.read_text().splitlines() if line.strip())
    return ids


def load_atlas() -> dict[str, dict]:
    """{'<name>-<resonance>': {cls, cls_text, sure, observation, group}}; parsed, never executed."""
    if not ATLAS_LABELS.exists():
        missing.append(f'{ATLAS_LABELS.relative_to(REPO)}: нет файла (атлас не показан)')
        return {}
    labels = {}
    for node in ast.parse(ATLAS_LABELS.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'LABELS' for t in node.targets):
            labels = ast.literal_eval(node.value)
    observations = {}
    if ATLAS_TSV.exists():
        with open(ATLAS_TSV, newline='') as f:
            for row in csv.DictReader(f, delimiter='\t'):
                observations[row['stem']] = row
    else:
        missing.append(f'{ATLAS_TSV.relative_to(REPO)}: нет файла (текст комментариев атласа не показан)')
    out = {}
    for stem, (cls, sure) in labels.items():
        row = observations.get(stem, {})
        out[stem] = {
            'cls': cls,
            'cls_text': ATLAS_CLASSES.get(cls, cls),
            'sure': bool(sure),
            'observation': row.get('user_observation', ''),
            'group': row.get('group', ''),
        }
    return out


def fmt_years(y) -> str:
    if y is None:
        return '?'
    y = abs(float(y))
    for scale, unit in ((1e6, 'Myr'), (1e3, 'kyr')):
        if y >= scale * 0.999:
            return f'{y / scale:.3g} {unit}'
    return f'{y:.3g} yr'


def clean(value):
    if value is None:
        return ''
    if isinstance(value, list):
        return ', '.join(str(v) for v in value)
    return str(value).strip()


def build_example(eid, label, cand, meta, verdict, adj, atlas, disputed):
    regime = meta.get('regime') or {}
    old = meta.get('labels') or {}
    parts = eid.split('__')
    name, resonance = str(meta.get('name', parts[1] if len(parts) > 1 else eid)), str(meta.get('resonance', ''))
    claude_status = (adj or {}).get('recommended_status') or (verdict or {}).get('status') or ''
    claude_angle = (adj or {}).get('recommended_angle_status') or (verdict or {}).get('angle_status') or ''
    claude_conf = (adj or {}).get('confidence') or (verdict or {}).get('confidence') or ''
    compatible = BUCKET_TO_STATUS.get(label.get('proposed_bucket', ''), set())
    blind_status = (verdict or {}).get('status')
    discrepancy = bool(
        verdict and (eid in disputed or (compatible and (blind_status not in compatible or claude_status not in compatible)))
    )
    needs_discussion = bool(adj and adj.get('needs_discussion'))
    conflict = bool(adj and adj.get('conflict_with_author'))
    tier = 0 if needs_discussion else 1 if conflict else 2
    notes = clean(meta.get('notes'))
    cand_notes = clean(cand.get('notes'))
    atlas_hit = atlas.get(f'{name}-{resonance}')
    return {
        'id': eid,
        'cand': label.get('candidate_id', ''),
        'name': name,
        'res': resonance,
        'rtype': label.get('rtype') or meta.get('rtype', ''),
        'variant': parts[3] if len(parts) > 3 else '',
        'regime': {
            'integrator': clean(regime.get('integrator')),
            'dt': regime.get('dt'),
            'span': fmt_years(regime.get('span_yr')),
            'step': fmt_years(regime.get('step_yr')),
            'direction': clean(regime.get('direction')),
            'epoch': clean(regime.get('epoch_date'))[:10],
            'n_rows': regime.get('n_rows'),
            'moon': bool(regime.get('moon')),
            'phases': clean(regime.get('secular_phases')),
            'decimation': (regime.get('decimation') or {}).get('factor'),
        },
        'proposed': label.get('proposed_bucket', ''),
        'pconf': label.get('confidence', ''),
        'old': {
            'author_verdict': clean(old.get('author_verdict')) or clean(cand.get('author_verdict')),
            'old_statuses': clean(old.get('old_statuses')) or clean(cand.get('old_statuses')),
            'source_status': clean(old.get('source_status')),
            'reliability': clean(old.get('reliability')) or clean(cand.get('best_reliability')),
            'sources': clean(old.get('sources')) or clean(cand.get('sources')),
            'tags': clean(old.get('tags')) or clean(cand.get('tags')).replace('|', ', '),
            'notes': notes,
            'cand_notes': cand_notes if cand_notes and cand_notes not in notes else '',
        },
        'claude': verdict,
        'adj': adj,
        'eff': {'status': claude_status, 'angle': claude_angle, 'conf': claude_conf},
        'atlas': atlas_hit,
        'discrepancy': discrepancy,
        'needs_discussion': needs_discussion,
        'conflict': conflict,
        'tier': tier,
        # What import_author_labels.py already wrote into labels.csv: seeds a browser with no local labels.
        'prior': {
            k: label.get(f'author_{k}', '')
            for k in ('bucket', 'confidence', 'angle_status', 'comment', 'discuss', 'agreed_with_claude', 'labelled_at')
        },
    }


def natural_key(text: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', text)]


def main() -> None:
    labels = read_csv(BENCH / 'labels.csv')
    candidates = {r['id']: r for r in read_csv(BENCH / 'candidates.csv')}
    verdicts = load_verdicts()
    adjudications = load_adjudications()
    disputed = load_disputed()
    atlas = load_atlas()

    examples = []
    no_meta, no_png = [], []
    for label in labels:
        eid = label['example_id']
        meta_path = EXAMPLES / eid / 'meta.json'
        if not meta_path.exists():
            no_meta.append(eid)
            meta = {}
        else:
            meta = json.loads(meta_path.read_text())
        if not (EXAMPLES / eid / 'combined.png').exists():
            no_png.append(eid)
        ex = build_example(
            eid,
            label,
            candidates.get(label.get('candidate_id', ''), {}),
            meta,
            verdicts.get(eid),
            adjudications.get(eid),
            atlas,
            disputed,
        )
        ex['has_png'] = eid not in no_png
        examples.append(ex)

    order = {r: i for i, r in enumerate(RTYPES)}
    examples.sort(key=lambda e: (e['tier'], order.get(e['rtype'], 99), natural_key(e['id'])))

    ids = {e['id'] for e in examples}
    orphans = sorted((set(verdicts) | set(adjudications)) - ids)
    no_verdict = [e['id'] for e in examples if not e['claude']]
    if no_meta:
        missing.append(f'meta.json нет у {len(no_meta)} примеров: {", ".join(no_meta[:5])}{" …" if len(no_meta) > 5 else ""}')
    if no_png:
        missing.append(f'combined.png нет у {len(no_png)} примеров: {", ".join(no_png[:5])}{" …" if len(no_png) > 5 else ""}')
    if no_verdict:
        by_type = {}
        for eid in no_verdict:
            rt = next(e['rtype'] for e in examples if e['id'] == eid)
            by_type[rt] = by_type.get(rt, 0) + 1
        missing.append(
            f'нет слепого вердикта Claude у {len(no_verdict)} примеров (' + ', '.join(f'{k}: {v}' for k, v in sorted(by_type.items())) + ')'
        )
    if orphans:
        missing.append(f'вердикты/разборы Claude для id, которых нет в labels.csv: {len(orphans)} ({", ".join(orphans[:5])})')

    stats = {
        'examples': len(examples),
        'verdicts': sum(1 for e in examples if e['claude']),
        'adjudications': sum(1 for e in examples if e['adj']),
        'needs_discussion': sum(e['needs_discussion'] for e in examples),
        'conflicts': sum(e['conflict'] for e in examples),
        'discrepancies': sum(e['discrepancy'] for e in examples),
        'atlas': sum(1 for e in examples if e['atlas']),
    }
    payload = {
        'built': datetime.datetime.now().isoformat(timespec='seconds'),
        'bench_dir': str(BENCH.relative_to(BENCH.parent)),
        # the page opens from file://, so figures are addressed relative to review_page/
        'examples_url': Path(os.path.relpath(EXAMPLES, OUT)).as_posix(),
        'rtypes': RTYPES,
        'statuses': STATUSES,
        'buckets': sorted({e['proposed'] for e in examples if e['proposed']}),
        'stats': stats,
        'missing': missing,
        'examples': examples,
    }
    OUT.mkdir(exist_ok=True)
    js = json.dumps(payload, ensure_ascii=False, separators=(',', ':'))
    # '</' would end an inline <script>; data.js is external, but keep it safe anyway.
    (OUT / 'data.js').write_text('window.BENCH = ' + js.replace('</', '<\\/') + ';\n')
    (OUT / 'index.html').write_text(PAGE)
    print(f'{OUT / "index.html"}: {len(examples)} карточек')
    for k, v in stats.items():
        print(f'  {k}: {v}')
    for m in missing:
        print(f'  нет данных: {m}')


PAGE = r'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка бенчмарка</title>
<style>
:root {
  --bg: #f6f5f1; --panel: #ffffff; --ink: #1d1d1b; --muted: #6b6a64; --line: #dedcd4;
  --accent: #2f5d8a; --accent-ink: #ffffff; --soft: #eef2f6;
  --warn: #9a5b00; --warn-bg: #fff3df; --bad: #a33a2b; --bad-bg: #fbe9e5; --ok: #2d6a3e; --ok-bg: #e5f2e8;
  --hl-bg: #fff8d6; --hl-line: #e3c84e; --focus: #2f5d8a;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #17181a; --panel: #212326; --ink: #e8e6e1; --muted: #9a988f; --line: #393b3f;
    --accent: #7fb0e0; --accent-ink: #10161c; --soft: #262c33;
    --warn: #e7b25a; --warn-bg: #3a2e18; --bad: #f08c7a; --bad-bg: #3d221d; --ok: #8fd19e; --ok-bg: #1f3325;
    --hl-bg: #33301c; --hl-line: #8f7d2b; --focus: #7fb0e0;
    color-scheme: dark;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; }
code, .mono { font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12.5px; }
button, select, input, textarea { font: inherit; color: inherit; }
button { cursor: pointer; }
:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }

.topbar { position: sticky; top: 0; z-index: 20; background: var(--panel); border-bottom: 1px solid var(--line); padding: 6px 16px; box-shadow: 0 1px 6px rgba(0,0,0,.06); }
.row { display: flex; flex-wrap: wrap; gap: 6px 14px; align-items: center; }
.row + .row { margin-top: 4px; }
.topbar h1 { font-size: 15px; margin: 0 8px 0 0; }
.topbar label { display: inline-flex; gap: 5px; align-items: center; white-space: nowrap; }
.topbar select, .topbar input[type=search] { background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 2px 6px; max-width: 170px; }
.topbar input[type=search] { width: 170px; }
.counters { margin-left: auto; display: flex; gap: 12px; white-space: nowrap; }
.counters b { font-variant-numeric: tabular-nums; }
.btn { border: 1px solid var(--line); background: var(--bg); border-radius: 6px; padding: 3px 10px; }
.btn:hover { border-color: var(--accent); }
.btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.hint { color: var(--muted); font-size: 12.5px; }
.savepath { color: var(--warn); }

main { max-width: 1560px; margin: 0 auto; padding: 12px 16px 80px; }
.notice { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 13px; }
.notice summary { cursor: pointer; }
.notice.bad { background: var(--bad-bg); border-color: var(--bad); }
.section-h { margin: 22px 0 8px; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); border-bottom: 1px solid var(--line); padding-bottom: 4px; }

.card { background: var(--panel); border: 1px solid var(--line); border-left: 4px solid var(--line); border-radius: 8px; margin: 0 0 14px; scroll-margin-top: calc(var(--topbar, 120px) + 8px); }
.card.current { box-shadow: 0 0 0 2px var(--focus); }
.card.done { border-left-color: var(--ok); }
.card.discuss { border-left-color: var(--warn); }
.card-h { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; padding: 8px 12px; border-bottom: 1px solid var(--line); }
.card-h .name { font-size: 17px; font-weight: 650; }
.card-h .res { font-size: 14px; }
.card-h .regime { color: var(--muted); font-size: 12.5px; }
.card-h .idx { margin-left: auto; color: var(--muted); font-size: 12px; }
.badge { display: inline-block; border-radius: 10px; padding: 0 8px; font-size: 12px; border: 1px solid var(--line); background: var(--bg); white-space: nowrap; }
.badge.rt { background: var(--soft); }
.badge.warn { color: var(--warn); background: var(--warn-bg); border-color: transparent; }
.badge.bad { color: var(--bad); background: var(--bad-bg); border-color: transparent; }
.badge.ok { color: var(--ok); background: var(--ok-bg); border-color: transparent; }

.card-b { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(330px, 1fr); }
@media (max-width: 1100px) { .card-b { grid-template-columns: 1fr; } }
.fig { padding: 8px; border-right: 1px solid var(--line); min-width: 0; }
@media (max-width: 1100px) { .fig { border-right: 0; border-bottom: 1px solid var(--line); } }
.fig img { display: block; width: 100%; height: auto; aspect-ratio: 1760 / 1091; object-fit: contain; background: #fff; border-radius: 4px; cursor: zoom-in; }
.fig .nopng { padding: 40px; text-align: center; color: var(--muted); }
.side { padding: 10px 12px; min-width: 0; display: flex; flex-direction: column; gap: 10px; }
.blk h3 { margin: 0 0 4px; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); font-weight: 600; }
.kv { display: grid; grid-template-columns: max-content 1fr; gap: 1px 10px; font-size: 13px; }
.kv dt { color: var(--muted); }
.kv dd { margin: 0; overflow-wrap: anywhere; }
.st { font-weight: 650; }
.argument { background: var(--hl-bg); border-left: 3px solid var(--hl-line); padding: 6px 9px; border-radius: 4px; font-size: 13.5px; }
.problem { background: var(--bad-bg); color: var(--bad); padding: 4px 8px; border-radius: 4px; font-size: 13px; }
.prev { background: var(--soft); padding: 6px 9px; border-radius: 4px; font-size: 13.5px; }
details.more > summary { cursor: pointer; color: var(--muted); font-size: 12.5px; }
details.more[open] > summary { margin-bottom: 4px; }

.ctl { border-bottom: 1px solid var(--line); padding-bottom: 10px; display: flex; flex-direction: column; gap: 7px; }
.btns { display: flex; flex-wrap: wrap; gap: 4px; }
.sbtn { border: 1px solid var(--line); background: var(--bg); border-radius: 6px; padding: 3px 8px; font-size: 13px; }
.sbtn kbd { font: 11px ui-monospace, Menlo, monospace; color: var(--muted); margin-right: 3px; }
.sbtn.on { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.sbtn.on kbd { color: inherit; opacity: .75; }
.sbtn:disabled { opacity: .4; cursor: default; }
.sbtn.agree { border-style: dashed; }
.sbtn.agree.on { background: var(--ok); border-color: var(--ok); color: var(--accent-ink); }
.ctl select { background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 3px 6px; }
.ctl textarea { width: 100%; min-height: 54px; resize: vertical; background: var(--bg); border: 1px solid var(--line); border-radius: 6px; padding: 5px 7px; }
.ctl .stamp { color: var(--muted); font-size: 12px; }

#lightbox { position: fixed; inset: 0; z-index: 50; background: rgba(0,0,0,.88); display: none; overflow: auto; cursor: zoom-out; }
#lightbox.open { display: block; }
#lightbox img { display: block; margin: 0 auto; max-width: none; background: #fff; }
#lightbox .cap { position: sticky; top: 0; color: #eee; background: rgba(0,0,0,.6); padding: 6px 12px; font-size: 13px; }
#toast { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); background: var(--ink); color: var(--bg); padding: 7px 14px; border-radius: 8px; font-size: 13px; opacity: 0; transition: opacity .2s; pointer-events: none; z-index: 60; max-width: calc(100vw - 32px); }
#toast.show { opacity: .95; }
.empty { text-align: center; color: var(--muted); padding: 40px; }
</style>
</head>
<body>
<div class="topbar" id="topbar">
  <div class="row">
    <h1>Разметка бенчмарка</h1>
    <label>тип <select id="f-rtype"></select></label>
    <label>статус Claude <select id="f-claude"></select></label>
    <label>предложенная корзина <select id="f-bucket"></select></label>
    <input type="search" id="f-search" placeholder="поиск: имя или резонанс" aria-label="поиск по имени">
    <div class="counters" aria-live="polite">
      <span>всего <b id="c-total">0</b></span>
      <span>размечено <b id="c-done">0</b></span>
      <span>осталось <b id="c-left">0</b></span>
      <span class="hint">показано <b id="c-shown">0</b></span>
    </div>
  </div>
  <div class="row">
    <label><input type="checkbox" id="t-disc"> только расхождения</label>
    <label><input type="checkbox" id="t-need"> только требующие обсуждения</label>
    <label><input type="checkbox" id="t-conf"> только конфликт с вашим вердиктом</label>
    <label><input type="checkbox" id="t-todo"> неразмеченные</label>
    <span style="margin-left:auto; display:flex; gap:6px; align-items:center; flex-wrap:wrap">
      <button class="btn primary" id="b-export">Экспорт CSV</button>
      <button class="btn" id="b-import">Импорт CSV</button>
      <input type="file" id="f-import" accept=".csv,text/csv" hidden>
    </span>
  </div>
  <div class="row hint">
    <span class="savepath">Разметка хранится только в этом браузере. Экспортированный файл сохраните в <code id="benchdir"></code> — оттуда его прочитает Claude.</span>
    <span id="lastexport"></span>
    <span>j/k — карточки, 1–6 — статус, a — согласен с Claude, c — комментарий, d — обсудить, Esc — выйти из поля</span>
  </div>
</div>
<main>
  <div id="notices"></div>
  <div id="list"></div>
</main>
<div id="lightbox" role="dialog" aria-label="рисунок в полном размере"><div class="cap" id="lb-cap"></div><img id="lb-img" alt=""></div>
<div id="toast" role="status"></div>
<script src="data.js"></script>
<script>
(function () {
'use strict';
const D = window.BENCH;
if (!D) { document.getElementById('list').textContent = 'data.js не загрузился: пересоберите страницу (tools/build_review_page.py).'; return; }

const KEY = 'benchv2.authorLabels.v1', PREFS = 'benchv2.reviewPrefs.v1', EXPORTED = 'benchv2.lastExport.v1';
const STATUSES = D.statuses;
const CONFS = ['certain', 'high', 'medium', 'low'];
const ANGLES = ['libration', 'episodic', 'near_separatrix', 'circulation', 'drift', 'insufficient_time', 'undefined'];
const NO_CONF = new Set(['uncertain', 'invalid']);  // DECISIONS.md: confidence is empty for these
const TIER = ['Требуют обсуждения', 'Конфликт с вашим вердиктом', 'Остальные'];
const RU = {
  resonant: 'резонанс', transient: 'транзиент', near_separatrix: 'у сепаратрисы', non_resonant: 'не резонанс',
  uncertain: 'неопределённо', invalid: 'невозможно классифицировать'
};

// ------------------------------------------------------------ storage
let storageOK = true;
function load(key, fallback) {
  try { const v = localStorage.getItem(key); return v ? JSON.parse(v) : fallback; }
  catch (e) { storageOK = false; return fallback; }
}
function save(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); return true; }
  catch (e) { storageOK = false; return false; }
}
const labels = load(KEY, {});
// A browser without local labels starts from what import_author_labels.py already put into labels.csv.
for (const ex of D.examples) {
  const p = ex.prior || {};
  if (labels[ex.id] || !(p.bucket || p.comment || p.discuss)) continue;
  labels[ex.id] = {
    status: (p.bucket || '').replace(/-/g, '_'), confidence: p.confidence || '', angle_status: p.angle_status || '',
    comment: p.comment || '', discuss: p.discuss === '1', agreed: p.agreed_with_claude === '1', ts: p.labelled_at || ''
  };
}
function persist() { if (!save(KEY, labels)) toast('localStorage недоступен: разметка не сохранится, экспортируйте CSV!'); }
function isDone(id) { const l = labels[id]; return !!(l && l.status); }

// ------------------------------------------------------------ helpers
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== '') el.append(kid.nodeType ? kid : String(kid));
  return el;
}
function kv(pairs) {
  const dl = h('dl', {class: 'kv'});
  for (const [k, v] of pairs) if (v !== '' && v !== null && v !== undefined) dl.append(h('dt', {}, k), h('dd', {}, v));
  return dl;
}
let toastTimer;
function toast(msg) {
  const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 3500);
}
const norm = s => String(s || '').toLowerCase().replace(/[_\s]+/g, ' ').trim();

// ------------------------------------------------------------ cards
const cards = new Map();   // id -> {el, ex, refresh}
function statusBadge(s) { return s ? h('span', {class: 'st', title: RU[s] || ''}, s) : '—'; }

function claudeBlock(ex) {
  const v = ex.claude, a = ex.adj;
  const blk = h('div', {class: 'blk'});
  if (!v && !a) { blk.append(h('h3', {}, 'Claude'), h('div', {class: 'hint'}, 'Claude ещё не смотрел этот пример.')); return blk; }
  const blind = v ? [
    ['статус', statusBadge(v.status)],
    ['угол', [v.angle_status, v.angle_subtype].filter(Boolean).join(' / ')],
    ['уверенность', v.confidence], ['причина', v.reason || ''], ['эпизоды', v.episodes || ''],
    ['проверки', v.checks || ''], ['комментарий', v.comment || '']
  ] : [];
  if (a) {
    blk.append(h('h3', {}, 'Рекомендация Claude (после разбора)'));
    blk.append(kv([['статус', statusBadge(a.recommended_status)], ['угол', a.recommended_angle_status], ['уверенность', a.confidence]]));
    const flags = h('div', {class: 'btns', style: 'margin:4px 0'});
    if (a.needs_discussion) flags.append(h('span', {class: 'badge warn'}, 'Claude просит обсудить'));
    if (a.conflict_with_author) flags.append(h('span', {class: 'badge bad'}, 'расходится с вашим вердиктом'));
    if (flags.childNodes.length) blk.append(flags);
    if (a.argument_ru) blk.append(h('div', {class: 'argument'}, a.argument_ru));
    blk.append(kv([['причина', a.reason || ''], ['ваш вердикт (как его прочёл Claude)', a.author_said || ''],
      ['откуда старая метка', a.old_label_source || ''], ['к правилу', a.rule_issue || '']]));
    if (v) {
      const d = h('details', {class: 'more'}, h('summary', {}, `слепой вердикт (${v.wave || ''}): ${v.status}, ${v.confidence}`), kv(blind));
      if (v.plot_problem) d.append(h('div', {class: 'problem'}, 'рисунок: ' + v.plot_problem));
      blk.append(d);
    }
  } else {
    blk.append(h('h3', {}, `Claude, слепой вердикт (${v.wave || ''})`), kv(blind));
    if (v.plot_problem) blk.append(h('div', {class: 'problem'}, 'проблема рисунка: ' + v.plot_problem));
  }
  return blk;
}

function oldBlock(ex) {
  const o = ex.old, blk = h('div', {class: 'blk'}, h('h3', {}, 'Прежние метки'));
  blk.append(o.author_verdict ? h('div', {class: 'prev'}, h('b', {}, 'Ваш прежний вердикт: '), o.author_verdict)
                              : h('div', {class: 'hint'}, 'Вашего прежнего вердикта нет.'));
  if (ex.atlas) {
    const at = ex.atlas;
    blk.append(h('div', {class: 'prev', style: 'margin-top:6px'},
      h('b', {}, 'Ваш комментарий к атласу (прочтение Claude): '),
      at.observation ? h('i', {}, '«' + at.observation + '»') : '', ' → ',
      h('b', {}, at.cls), ' (' + at.cls_text + (at.sure ? ', категорично' : ', с оговоркой') + ')',
      h('div', {class: 'hint'}, `старая папка: ${at.group || '?'}; атлас NEA (сент. 2026), режим мог быть другим — только подсказка`)));
  }
  const r = ex.regime;
  blk.append(h('details', {class: 'more'}, h('summary', {}, 'остальное: старые статусы, источники, режим'),
    kv([
      ['предложено', `${ex.proposed || '—'} / ${ex.pconf || '—'}`], ['старые статусы', o.old_statuses],
      ['статус прогона', o.source_status], ['надёжность', o.reliability], ['источники', o.sources], ['теги', o.tags],
      ['заметки', o.notes], ['заметки кандидата', o.cand_notes],
      ['кандидат', ex.cand], ['id', h('code', {}, ex.id)],
      ['режим', `${r.integrator}, dt=${r.dt}, ${r.direction}, эпоха ${r.epoch}, ${r.n_rows} строк${r.moon ? ', Луна' : ''}${r.phases ? ', фазы ' + r.phases : ''}${r.decimation ? ', прорежено ×' + r.decimation : ''}`]
    ])));
  return blk;
}

function controls(ex, card) {
  const id = ex.id, eff = ex.eff;
  const get = () => labels[id] || (labels[id] = {status: '', confidence: '', angle_status: '', comment: '', discuss: false, agreed: false, ts: ''});
  const touch = () => { get().ts = new Date().toISOString(); persist(); refresh(); updateCounters(); };
  const sBtns = STATUSES.map((s, i) => h('button', {class: 'sbtn', title: RU[s], onclick: () => setStatus(s)}, h('kbd', {}, i + 1), s));
  const agree = h('button', {class: 'sbtn agree', disabled: !eff.status, title: 'взять статус, уверенность и угол Claude',
    onclick: () => agreeClaude()}, h('kbd', {}, 'a'), 'согласен с Claude' + (eff.status ? ` (${eff.status})` : ''));
  const cBtns = CONFS.map(c => h('button', {class: 'sbtn', onclick: () => { const l = get(); l.confidence = l.confidence === c ? '' : c; recheckAgree(); touch(); }}, c));
  const angle = h('select', {'aria-label': 'angle_status', onchange: e => { get().angle_status = e.target.value; recheckAgree(); touch(); }},
    h('option', {value: ''}, '— не указан —'), ANGLES.map(a => h('option', {value: a}, a)));
  const comment = h('textarea', {placeholder: 'комментарий (c)', rows: 2});
  let t;
  comment.addEventListener('input', () => { clearTimeout(t); t = setTimeout(() => { get().comment = comment.value; touch(); }, 300); });
  comment.addEventListener('blur', () => { if (get().comment !== comment.value) { get().comment = comment.value; touch(); } });
  const disc = h('input', {type: 'checkbox', onchange: e => { get().discuss = e.target.checked; touch(); }});
  const stamp = h('span', {class: 'stamp'});

  function recheckAgree() {
    const l = get();
    if (l.agreed && (l.status !== eff.status || (l.angle_status && l.angle_status !== eff.angle) ||
        (l.confidence && l.confidence !== eff.conf))) l.agreed = false;
  }
  function setStatus(s) {
    const l = get(); l.status = l.status === s ? '' : s;
    if (NO_CONF.has(l.status)) l.confidence = '';
    recheckAgree(); touch();
  }
  function agreeClaude() {
    if (!eff.status) return;
    const l = get();
    l.status = eff.status; l.confidence = NO_CONF.has(eff.status) ? '' : (CONFS.includes(eff.conf) ? eff.conf : '');
    l.angle_status = ANGLES.includes(eff.angle) ? eff.angle : ''; l.agreed = true; touch();
  }
  function refresh() {
    const l = labels[id] || {};
    sBtns.forEach((b, i) => b.classList.toggle('on', l.status === STATUSES[i]));
    agree.classList.toggle('on', !!l.agreed);
    cBtns.forEach((b, i) => { b.classList.toggle('on', l.confidence === CONFS[i]); b.disabled = NO_CONF.has(l.status); });
    angle.value = l.angle_status || '';
    if (document.activeElement !== comment) comment.value = l.comment || '';
    disc.checked = !!l.discuss;
    stamp.textContent = l.ts ? 'изменено ' + l.ts.replace('T', ' ').slice(0, 16) : '';
    card.classList.toggle('done', !!l.status);
    card.classList.toggle('discuss', !!l.discuss);
  }
  const el = h('div', {class: 'ctl'},
    h('div', {class: 'blk'}, h('h3', {}, 'Ваша разметка')),
    h('div', {class: 'btns'}, sBtns), h('div', {class: 'btns'}, agree),
    h('div', {class: 'btns'}, h('span', {class: 'hint', style: 'align-self:center'}, 'уверенность:'), cBtns),
    h('label', {class: 'hint'}, 'angle_status: ', angle),
    comment,
    h('div', {class: 'row', style: 'justify-content:space-between'}, h('label', {}, disc, ' обсудить'), stamp));
  return {el, refresh, setStatus, agreeClaude, toggleDiscuss: () => { get().discuss = !get().discuss; touch(); }, comment};
}

function buildCard(ex) {
  const r = ex.regime;
  const card = h('article', {class: 'card', id: 'card-' + ex.id, 'data-id': ex.id});
  const badges = [h('span', {class: 'badge rt'}, ex.rtype)];
  if (ex.variant) badges.push(h('span', {class: 'badge'}, ex.variant));
  if (ex.proposed) badges.push(h('span', {class: 'badge', title: 'предложенная корзина / уверенность'}, `${ex.proposed} · ${ex.pconf || '—'}`));
  if (ex.needs_discussion) badges.push(h('span', {class: 'badge warn'}, 'обсудить'));
  if (ex.conflict) badges.push(h('span', {class: 'badge bad'}, 'конфликт'));
  if (ex.discrepancy) badges.push(h('span', {class: 'badge warn', title: 'Claude расходится с предложенной корзиной'}, 'расхождение'));
  const idx = h('span', {class: 'idx'});
  card.append(h('header', {class: 'card-h'},
    h('span', {class: 'name'}, ex.name), h('span', {class: 'res mono'}, ex.res), badges,
    h('span', {class: 'regime'}, `${r.span} · ${r.integrator} · шаг ${r.step}`), idx));
  const src = (D.examples_url || '../data/examples') + '/' + encodeURIComponent(ex.id) + '/combined.png';
  const fig = h('div', {class: 'fig'}, ex.has_png
    ? h('img', {src, loading: 'lazy', decoding: 'async', alt: `${ex.name} ${ex.res}`, onclick: () => openLightbox(src, ex)})
    : h('div', {class: 'nopng'}, 'combined.png нет'));
  const ctl = controls(ex, card);
  card.append(h('div', {class: 'card-b'}, fig, h('div', {class: 'side'}, ctl.el, claudeBlock(ex), oldBlock(ex))));
  card.addEventListener('focusin', () => setCurrent(visible.indexOf(ex.id), false));
  card.addEventListener('mousedown', () => setCurrent(visible.indexOf(ex.id), false));
  ctl.refresh();
  cards.set(ex.id, {el: card, ex, ctl, idx});
}

// ------------------------------------------------------------ filters
const prefs = Object.assign({rtype: '', claude: '', bucket: '', q: '', disc: false, need: false, conf: false, todo: false}, load(PREFS, {}));
const $ = id => document.getElementById(id);
function fillSelect(sel, options, value) {
  sel.append(h('option', {value: ''}, 'все'));
  for (const [v, t] of options) sel.append(h('option', {value: v}, t));
  sel.value = value;
}
function count(pred) { return D.examples.filter(pred).length; }
fillSelect($('f-rtype'), D.rtypes.map(r => [r, `${r} (${count(e => e.rtype === r)})`]), prefs.rtype);
fillSelect($('f-claude'), STATUSES.map(s => [s, `${s} (${count(e => e.eff.status === s)})`]).concat([['__none', `нет вердикта (${count(e => !e.eff.status)})`]]), prefs.claude);
fillSelect($('f-bucket'), D.buckets.map(b => [b, `${b} (${count(e => e.proposed === b)})`]).concat([['__none', `без корзины (${count(e => !e.proposed)})`]]), prefs.bucket);
$('f-search').value = prefs.q;
for (const k of ['disc', 'need', 'conf', 'todo']) $('t-' + k).checked = !!prefs[k];

function matches(ex) {
  if (prefs.rtype && ex.rtype !== prefs.rtype) return false;
  if (prefs.claude && (prefs.claude === '__none' ? ex.eff.status : ex.eff.status !== prefs.claude)) return false;
  if (prefs.bucket && (prefs.bucket === '__none' ? ex.proposed : ex.proposed !== prefs.bucket)) return false;
  if (prefs.disc && !ex.discrepancy) return false;
  if (prefs.need && !ex.needs_discussion) return false;
  if (prefs.conf && !ex.conflict) return false;
  if (prefs.todo && isDone(ex.id)) return false;
  if (prefs.q) {
    const q = norm(prefs.q);
    if (!norm(ex.name).includes(q) && !norm(ex.res).includes(q) && !norm(ex.id).includes(q)) return false;
  }
  return true;
}

let visible = [], current = -1;
function applyFilters() {
  const list = $('list'); list.replaceChildren();
  visible = [];
  let lastHead = null;
  for (const ex of D.examples) {
    if (!matches(ex)) continue;
    const head = TIER[ex.tier] + ' · ' + ex.rtype;
    if (head !== lastHead) { list.append(h('h2', {class: 'section-h'}, head)); lastHead = head; }
    visible.push(ex.id);
    list.append(cards.get(ex.id).el);
  }
  visible.forEach((id, i) => { cards.get(id).idx.textContent = `${i + 1} / ${visible.length}`; });
  if (!visible.length) list.append(h('div', {class: 'empty'}, 'Под фильтр ничего не попало.'));
  current = -1;
  updateCounters();
}
function onFilter() {
  prefs.rtype = $('f-rtype').value; prefs.claude = $('f-claude').value; prefs.bucket = $('f-bucket').value;
  prefs.q = $('f-search').value;
  for (const k of ['disc', 'need', 'conf', 'todo']) prefs[k] = $('t-' + k).checked;
  save(PREFS, prefs); applyFilters(); window.scrollTo(0, 0);
}
['f-rtype', 'f-claude', 'f-bucket', 't-disc', 't-need', 't-conf', 't-todo'].forEach(id => $(id).addEventListener('change', onFilter));
let st; $('f-search').addEventListener('input', () => { clearTimeout(st); st = setTimeout(onFilter, 200); });

function updateCounters() {
  const total = D.examples.length, done = D.examples.filter(e => isDone(e.id)).length;
  $('c-total').textContent = total; $('c-done').textContent = done; $('c-left').textContent = total - done;
  $('c-shown').textContent = visible.length;
}

// ------------------------------------------------------------ navigation and keys
function setCurrent(i, scroll = true) {
  if (i < 0 || i >= visible.length) return;
  if (current >= 0 && visible[current]) cards.get(visible[current]).el.classList.remove('current');
  current = i;
  const c = cards.get(visible[i]);
  c.el.classList.add('current');
  if (scroll) c.el.scrollIntoView({block: 'start', behavior: 'auto'});
}
function inView(el) {
  const r = el.getBoundingClientRect(), top = $('topbar').getBoundingClientRect().bottom;
  return r.bottom > top + 40 && r.top < window.innerHeight - 40;
}
// Keep the current card while it is on screen; otherwise take the first card in view. True if it was kept.
function ensureCurrent() {
  if (current >= 0 && inView(cards.get(visible[current]).el)) return true;
  const i = visible.findIndex(id => inView(cards.get(id).el));
  if (i >= 0) setCurrent(i, false);
  return false;
}
document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const lb = $('lightbox');
  if (lb.classList.contains('open')) { if (e.key === 'Escape') closeLightbox(); return; }
  const tag = (e.target.tagName || '').toLowerCase();
  if (tag === 'textarea' || tag === 'input' || tag === 'select') { if (e.key === 'Escape') e.target.blur(); return; }
  const code = e.code;   // physical keys, so the Russian layout works too
  if (code === 'KeyJ' || code === 'KeyK') {
    e.preventDefault();
    // after scrolling away, j/k first pick the card in view; then they move
    if (!ensureCurrent()) { if (current >= 0) setCurrent(current); return; }
    setCurrent(code === 'KeyJ' ? Math.min(current + 1, visible.length - 1) : Math.max(current - 1, 0));
    return;
  }
  ensureCurrent();
  if (current < 0) return;
  const c = cards.get(visible[current]);
  const m = /^Digit([1-6])$/.exec(code) || /^Numpad([1-6])$/.exec(code);
  if (m) { e.preventDefault(); c.ctl.setStatus(STATUSES[+m[1] - 1]); return; }
  if (code === 'KeyA') { e.preventDefault(); c.ctl.agreeClaude(); return; }
  if (code === 'KeyC') { e.preventDefault(); c.ctl.comment.focus(); return; }
  if (code === 'KeyD') { e.preventDefault(); c.ctl.toggleDiscuss(); return; }
});

// ------------------------------------------------------------ lightbox
function openLightbox(src, ex) {
  $('lb-img').src = src; $('lb-cap').textContent = `${ex.name}  ${ex.res}  —  щелчок или Esc закрывает`;
  $('lightbox').classList.add('open'); $('lightbox').scrollTop = 0;
}
function closeLightbox() { $('lightbox').classList.remove('open'); $('lb-img').removeAttribute('src'); }
$('lightbox').addEventListener('click', closeLightbox);

// ------------------------------------------------------------ CSV export / import
const COLS = ['example_id', 'status', 'confidence', 'angle_status', 'comment', 'discuss', 'agreed_with_claude', 'timestamp'];
function csvCell(v) { v = String(v ?? ''); return /[",\n\r]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function exportCSV() {
  const rows = [COLS.join(',')];
  const ids = new Set(D.examples.map(e => e.id));
  for (const id of Object.keys(labels).sort()) {
    const l = labels[id];
    if (!ids.has(id) || !(l.status || l.comment || l.discuss)) continue;
    rows.push([id, l.status, l.confidence, l.angle_status, l.comment, l.discuss ? 1 : 0, l.agreed ? 1 : 0, l.ts].map(csvCell).join(','));
  }
  const date = new Date().toISOString().slice(0, 10);
  const name = `author_labels_${date}.csv`;
  const blob = new Blob([rows.join('\n') + '\n'], {type: 'text/csv;charset=utf-8'});
  const a = h('a', {href: URL.createObjectURL(blob), download: name});
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  save(EXPORTED, {at: new Date().toISOString(), n: rows.length - 1});
  showLastExport();
  toast(`${name}: ${rows.length - 1} строк. Положите файл в ${D.bench_dir}/`);
}
function parseCSV(text) {
  const out = []; let row = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"') { if (text[i + 1] === '"') { cell += '"'; i++; } else q = false; }
      else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === ',') { row.push(cell); cell = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); cell = ''; if (row.some(x => x !== '')) out.push(row); row = [];
    } else cell += ch;
  }
  row.push(cell); if (row.some(x => x !== '')) out.push(row);
  return out;
}
function importCSV(file) {
  const reader = new FileReader();
  reader.onload = () => {
    const rows = parseCSV(String(reader.result).replace(/^﻿/, ''));
    const head = rows.shift() || [];
    const col = Object.fromEntries(head.map((n, i) => [n.trim(), i]));
    if (!('example_id' in col)) { toast('Нет колонки example_id: это не экспорт этой страницы.'); return; }
    const ids = new Set(D.examples.map(e => e.id));
    let n = 0, skipped = 0, older = 0;
    for (const r of rows) {
      const g = k => (k in col ? (r[col[k]] || '').trim() : '');
      const id = g('example_id');
      if (!ids.has(id)) { skipped++; continue; }
      const ts = g('timestamp'), cur = labels[id];
      if (cur && cur.ts && ts && cur.ts > ts) { older++; continue; }   // keep the newer local label
      labels[id] = {
        status: STATUSES.includes(g('status')) ? g('status') : '', confidence: CONFS.includes(g('confidence')) ? g('confidence') : '',
        angle_status: ANGLES.includes(g('angle_status')) ? g('angle_status') : '', comment: 'comment' in col ? (r[col.comment] || '') : '',
        discuss: g('discuss') === '1' || g('discuss').toLowerCase() === 'true',
        agreed: g('agreed_with_claude') === '1' || g('agreed_with_claude').toLowerCase() === 'true', ts
      };
      n++;
    }
    persist();
    for (const c of cards.values()) c.ctl.refresh();
    updateCounters();
    toast(`Импортировано ${n}` + (older ? `, ${older} пропущено (локально новее)` : '') + (skipped ? `, ${skipped} неизвестных id` : ''));
  };
  reader.readAsText(file);
}
$('b-export').addEventListener('click', exportCSV);
$('b-import').addEventListener('click', () => $('f-import').click());
$('f-import').addEventListener('change', e => { if (e.target.files[0]) importCSV(e.target.files[0]); e.target.value = ''; });
function showLastExport() {
  const x = load(EXPORTED, null);
  $('lastexport').textContent = x ? `последний экспорт: ${x.at.replace('T', ' ').slice(0, 16)} (${x.n} строк)` : 'экспорта ещё не было';
}
$('benchdir').textContent = D.bench_dir + '/';
window.addEventListener('beforeunload', () => { const a = document.activeElement; if (a && a.tagName === 'TEXTAREA') a.blur(); });

// ------------------------------------------------------------ start
const s = D.stats, notices = $('notices');
const info = h('details', {class: 'notice'},
  h('summary', {}, `Собрано ${D.built.replace('T', ' ')}: ${s.examples} примеров, вердиктов Claude ${s.verdicts}, разборов ${s.adjudications} (обсудить ${s.needs_discussion}, конфликтов ${s.conflicts}), расхождений с корзиной ${s.discrepancies}, совпадений с атласом ${s.atlas}.`),
  D.missing.length ? h('ul', {}, D.missing.map(m => h('li', {}, m))) : h('div', {}, 'Все источники на месте.'),
  h('div', {class: 'hint'}, 'Порядок: сначала требующие обсуждения, затем конфликты с вашим вердиктом, затем остальные; внутри — по типу. «Расхождение» — слепой вердикт или рекомендация Claude не совпадает с предложенной корзиной (или пример в списке спорных). «Согласен с Claude» берёт статус, уверенность и угол рекомендации (после разбора, если он есть).'));
notices.append(info);
try { localStorage.setItem('__probe', '1'); localStorage.removeItem('__probe'); } catch (e) { storageOK = false; }
if (!storageOK) notices.append(h('div', {class: 'notice bad'}, 'localStorage недоступен: разметка не сохранится между открытиями. Экспортируйте CSV почаще.'));
function topbarHeight() {
  document.documentElement.style.setProperty('--topbar', $('topbar').getBoundingClientRect().height + 'px');
}
topbarHeight();
window.addEventListener('resize', topbarHeight);
D.examples.forEach(buildCard);
applyFilters();
showLastExport();
})();
</script>
</body>
</html>
'''


if __name__ == '__main__':
    main()
