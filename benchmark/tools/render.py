"""Draw one combined review figure per benchmark example, from the saved files only.

    uv run python benchmark/tools/render.py                 # everything missing, then the tree
    uv run python benchmark/tools/render.py --sample 3      # 3 examples per rtype
    uv run python benchmark/tools/render.py --only ID,ID --force
    uv run python benchmark/tools/render.py --tree-only     # rebuild data/review only
    uv run python benchmark/tools/render.py --v2 --only ID  # the v2 combined figure (combined_v2.png)

Nothing is integrated: `bench_io.load_example` + `to_body` rebuild the body from series.csv.gz and
meta.json, and the package's own plotters draw it.

    mmr2, mmr3, coorbital  the v2 verdict figure (combined_v2.png, `render_v2`: the analysis of
                           reclassify.py). The old MMR review page (MMRPlotter.plot_combined) is removed
                           from the package; without --v2 the MMR examples go through `render_v2` too.
    secular                SecularPlotter.plot_combined() on angle_reduced (the classified angle as the
                           current package builds it; raw angle in grey).
    zlk                    composed here: omega(t) raw + filtered, e(t) and i(t) (anti-correlation),
                           (e cos omega, e sin omega) by time, free omega against osculating (forced/free
                           split, `Body.build_free_elements`), and the e-i coherence (Welch, package
                           presets). No status, no gate verdict.

Every figure carries a header: name, resonance, rtype, regime (span, integrator, step, cutoff). Old
labels are off by default (they bias the re-review); `--show-labels` adds them in the top-right corner.
Outputs per example: combined.png and render.json (settings, cutoff and why, FAIR or why not, time).
Then data/review/<rtype>/<bucket>/<confidence>/ is rebuilt from labels.csv with an index.html per folder.
"""

from __future__ import annotations

import os

# One BLAS thread per worker: the workers are the parallelism (see the ACF/OpenBLAS stall of Sept 2026).
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_var, '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import argparse  # noqa: E402
import datetime  # noqa: E402
import html  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_io as bio  # noqa: E402

RENDER_VERSION = 3
PNG = 'combined.png'
SIDECAR = 'render.json'
HEADER_IN = 0.62  # inches of header band added above every figure
LOG = bio.BENCH / 'render.log'


# ------------------------------------------------------------------ regime


NEA_Q = bio.NEA_Q  # au: perihelion below this = NEA (as integrate.py)
DAYS_PER_TIME_UNIT = 365.256898 / (2 * np.pi)  # REBOUND time unit (G=1, au, Msun) in days: dt is in these units


def choose_cutoff(meta: Dict[str, Any], series: Optional[pd.DataFrame] = None) -> Tuple[float, str]:
    """Low-pass cutoff (1/yr) for the figure, and why.

    The package default 0.0005/yr (2 kyr) suits main-belt three-body angles on 100 kyr, but it erases the
    200-1000 yr librations of co-orbitals, NEAs and two-body main-belt MMRs. The cutoff only shapes the
    black (filtered) curves, the portrait and the cycles; the raw angle is always drawn in grey.
    The regime is read from the orbit (perihelion at the epoch, median period of the saved series), not from the
    output step: batch A writes main-belt bodies at 1 yr, batch B NEAs at 0.2-0.5 yr. The cutoff never
    exceeds half the orbital frequency, so a TNO's orbital-period terms are filtered out too.
    """
    rtype, reg = meta['rtype'], meta['regime']
    step, span = float(reg['step_yr']), float(reg['span_yr'])
    nyquist = 0.5 / step
    q = period = None
    if series is not None and {'a', 'e'} <= set(series.columns):
        a = series['a'].to_numpy(dtype=float)
        i0 = int(np.argmin(np.abs(series['t_yr'].to_numpy(dtype=float)))) if 't_yr' in series.columns else 0
        q = float(a[i0] * (1 - float(series['e'].iloc[i0])))  # at the epoch: the NEA definition, as integrate.py
        period = float(np.nanmedian(np.abs(a))) ** 1.5
    if rtype == 'coorbital':
        cutoff, why = 0.02, 'co-orbital: tadpole/horseshoe/QS periods of 100s yr; 50 yr cutoff'
    elif rtype in ('mmr2', 'mmr3') and q is not None and q < NEA_Q:
        cutoff, why = 0.01, f'NEA MMR (q0 = {q:.2f} au): librations of 100s yr; 100 yr cutoff'
    elif rtype == 'mmr2':
        cutoff, why = 0.005, 'two-body MMR (q >= 1.3 au): librations of 300-3000 yr; 200 yr cutoff'
    elif rtype == 'mmr3':
        cutoff, why = 0.002, 'three-body MMR (q >= 1.3 au): librations of kyr; 500 yr cutoff'
    else:
        source = (meta.get('classifier_config') or {}).get('oscillations_cutoff') or 0.0005
        cutoff, why = float(source), f'{rtype}: source run cutoff (slow angles); smoothing of the secular figure is its own'
    if period and rtype in ('mmr2', 'mmr3', 'coorbital') and cutoff > 0.5 / period:
        cutoff, why = 0.5 / period, why + f'; capped at half the orbital frequency (P = {period:.3g} yr)'
    if cutoff > 0.4 * nyquist:
        cutoff, why = 0.4 * nyquist, why + f'; capped at 0.4 Nyquist ({0.4 * nyquist:.3g}/yr)'
    if span * cutoff < 5:
        why += f'; NOTE baseline holds only {span * cutoff:.1f} cutoff periods'
    return float(cutoff), why


def fmt_years(value: float) -> str:
    value = float(f'{abs(float(value)):.4g}')
    if value >= 1e6:
        return f'{value / 1e6:.4g} Myr'
    if value >= 1e3:
        return f'{value / 1e3:.4g} kyr'
    if value >= 1:
        return f'{value:.4g} yr'
    return f'{value * 365.25:.3g} d'


def regime_line(meta: Dict[str, Any], cutoff: Optional[float], extra: str = '') -> str:
    reg = meta['regime']
    integrator = str(reg.get('integrator') or '?')
    # dt is in REBOUND units (yr/2pi = 58.13 d), not days; for ias15 it is only the first step
    if reg.get('dt') not in (None, '') and 'ias15' not in integrator.lower():
        dt = reg['dt']
        integrator += f' dt={dt:g} ({dt * DAYS_PER_TIME_UNIT:.3g} d)' if isinstance(dt, (int, float)) else f' dt={dt}'
    direction = reg.get('direction') or ''
    parts = [
        f'span {fmt_years(reg["span_yr"])}{" " + direction if direction else ""}',
        integrator,
        f'step {fmt_years(reg["step_yr"])}',
        f'{reg.get("n_rows", "?")} rows',
    ]
    if cutoff:
        parts.append(f'cutoff {cutoff:g}/yr ({fmt_years(1 / cutoff)})')
    dec = reg.get('decimation')
    if isinstance(dec, dict) and dec.get('factor', 1) not in (None, 1):
        parts.append(f'thinned x{dec["factor"]}')
    if reg.get('epoch_date'):
        parts.append(f'epoch {str(reg["epoch_date"])[:10]}')
    parts.append(f'run {meta.get("provenance", {}).get("run_id", "?")}')
    if extra:
        parts.extend(p.strip() for p in extra.split('|'))
    return '  |  '.join(parts)


def wrap_parts(text: str, max_chars: int) -> List[str]:
    """Break a '  |  '-joined line into lines of at most `max_chars`, only at the separators."""
    lines: List[str] = []
    for part in text.split('  |  '):
        if lines and len(lines[-1]) + 5 + len(part) <= max_chars:
            lines[-1] += '  |  ' + part
        else:
            lines.append(part)
    return lines


def label_text(meta: Dict[str, Any], width: int = 95) -> str:
    """Old labels, shortened to at most four lines (only with --show-labels)."""
    lab = meta.get('labels') or {}

    def short(text, n):
        text = ' '.join(str(text or '').split())
        return text if len(text) <= n else text[: n - 1] + '…'

    lines = [
        f'proposed: {lab.get("proposed_bucket") or "?"} ({lab.get("confidence") or "?"})   source status: {lab.get("source_status")}',
        'old: ' + short(lab.get('old_statuses'), width),
    ]
    if lab.get('author_verdict'):
        lines.append('author: ' + short(lab.get('author_verdict'), width))
    if lab.get('tags'):
        tags = lab['tags'] if isinstance(lab['tags'], str) else ', '.join(lab['tags'])
        lines.append('tags: ' + short(tags, width))
    return '\n'.join(lines)


# ------------------------------------------------------------------ header


def add_header(fig, meta: Dict[str, Any], cutoff: Optional[float], show_labels: bool, extra: str = '', band: float = HEADER_IN):
    """Grow the figure by `band` inches at the top and write the header there; panels keep their size."""
    width, height = fig.get_size_inches()
    # 9.5 pt sans-serif is about 0.52 em per character; the labels box takes the right 40 % when shown
    room = width * (0.57 if show_labels else 0.97)
    regime = wrap_parts(regime_line(meta, cutoff, extra), int(room * 72 / (9.5 * 0.52)))
    band += 0.17 * (len(regime) - 1)
    new_height = height + band
    scale = height / new_height
    for ax in fig.axes:
        x0, y0, w, h = ax.get_position().bounds
        ax.set_position([x0, y0 * scale, w, h * scale])
    if getattr(fig, '_suptitle', None) is not None:
        fig._suptitle.remove()
        fig._suptitle = None
    for text in list(fig.texts):  # anything placed in figure fractions moves down with the panels
        x, y = text.get_position()
        text.set_position((x, y * scale))
    fig.set_size_inches(width, new_height, forward=True)

    top = 1.0 - 0.08 / new_height
    title = f'{meta["name"]}   {meta["resonance"]}   [{meta["rtype"]}]'
    fig.text(0.012, top, title, ha='left', va='top', fontsize=15, fontweight='bold')
    fig.text(0.012, top - 0.32 / new_height, '\n'.join(regime), ha='left', va='top', fontsize=9.5, color='#333333', linespacing=1.3)
    fig.text(0.988, top, meta['example_id'], ha='right', va='top', fontsize=8, color='#777777', family='monospace')
    if show_labels:
        fig.text(
            0.988,
            top - 0.24 / new_height,
            label_text(meta),
            ha='right',
            va='top',
            multialignment='left',
            fontsize=7,
            color='#5a3a00',
            family='monospace',
            bbox={'facecolor': '#fff8e6', 'edgecolor': '#d9b25a', 'lw': 0.6, 'pad': 2.5, 'alpha': 0.92},
            zorder=20,
        )


# ------------------------------------------------------------------ figures


def render_secular(ex, cutoff: float):
    from resonances.plotting.secular_plots import SecularPlotter

    body, resonance, sim = bio.to_body(ex, secular_angle='reduced')
    plotter = SecularPlotter.from_body(body, resonance, sim, options={'style': 'screen'}).plot_combined()
    angles = ex.meta.get('angles') or {}
    phases = angles.get('phases') or ex.meta['regime'].get('secular_phases') or '?'
    extra = f'black: reduced angle, {phases} mode phases, smoothed over {fmt_years(plotter.d.window)}'
    worst = ((angles.get('check') or {}).get('reduced_vs_src_angle') or {}).get('max_deg')
    if worst is not None and worst > 1:
        extra += f'  |  source run angle (older definition) differs by up to {worst:.0f}°'
    info = {'angle': 'angle_reduced', 'phases': phases, 'smoothing_yr': float(plotter.d.window), 'reduced_vs_src_max_deg': worst}
    return plotter.figure, info, plotter, extra


def _filtered(values: np.ndarray, config) -> np.ndarray:
    from resonances.resonance.filtering import filter_angle

    return filter_angle(np.asarray(values, dtype=float), config)


def render_zlk(ex, cutoff: float):
    """Composed Lidov-Kozai figure; see the module docstring."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    from resonances.plotting.base import time_unit
    from resonances.plotting.style import CENTRE, INK, RAW, STYLES, panel_letter, time_cmap, time_colorbar, time_label
    from resonances.resonance.coherence_analysis import analyse_resonance

    body, resonance, sim = bio.to_body(ex, oscillations_cutoff=cutoff)
    cfg = sim.config
    style = STYLES['screen']
    t_yr = ex.t_yr
    order = np.argsort(t_yr)
    divisor, unit = time_unit(abs(t_yr[-1] - t_yr[0]))
    t = t_yr / divisor
    info: Dict[str, Any] = {}

    W = 16.0
    L, R, B, G = 1.0, 0.25, 0.65, 0.95  # left, right, bottom margins; gap between columns
    stack_h, stack_gap, row_gap = 1.55, 0.08, 1.0
    lower_h = 3.6
    H = B + lower_h + row_gap + 3 * stack_h + 2 * stack_gap + 0.35
    right_w = 3 * stack_h + 2 * stack_gap  # square portrait as tall as the stack
    series_w = W - L - R - G - right_w - 0.95  # room for the portrait's colour bar

    with mpl.rc_context(style.rc()):
        fig = plt.figure(figsize=(W, H), dpi=style.dpi)

        def axes(x, y, w, h):
            return fig.add_axes((x / W, y / H, w / W, h / H))

        y_top = B + lower_h + row_gap
        ax_w = axes(L, y_top + 2 * (stack_h + stack_gap), series_w, stack_h)
        ax_e = axes(L, y_top + (stack_h + stack_gap), series_w, stack_h)
        ax_i = axes(L, y_top, series_w, stack_h)
        for ax in (ax_e, ax_i):
            ax.sharex(ax_w)

        # (a) omega: raw unwrapped grey, filtered black, wrapped on a twin as faint dots
        omega_raw = np.degrees(body.angles_unwrapped[resonance.to_s()])
        omega_f = np.degrees(body.angles_filtered_unwrapped[resonance.to_s()])
        ax_w.plot(t, omega_raw, color=RAW, lw=style.raw_line)
        ax_w.plot(t, omega_f, color=INK, lw=style.line)
        ax_w.set_ylabel(r'$\omega$ unwrapped (deg)')
        if np.ptp(omega_f) < 400:
            for level in (90, 270, -90, 450):
                if np.nanmin(omega_raw) - 20 < level < np.nanmax(omega_raw) + 20:
                    ax_w.axhline(level, color=CENTRE, lw=0.6, ls=':')

        # (b, c) e and i: raw grey, filtered black — ZLK exchanges them in antiphase
        inc_deg = np.degrees(body.inc)
        for ax, raw, label in ((ax_e, body.ecc, '$e$'), (ax_i, inc_deg, '$i$ (deg)')):
            ax.plot(t, raw, color=RAW, lw=style.raw_line)
            ax.plot(t, _filtered(raw, cfg), color=INK, lw=style.line)
            ax.set_ylabel(label)
        r = float(np.corrcoef(body.ecc, body.inc)[0, 1]) if np.std(body.ecc) > 0 and np.std(body.inc) > 0 else float('nan')
        info['r_e_i'] = r
        ax_e.text(
            0.995,
            0.95,
            f'corr(e, i) = {r:+.2f}',
            transform=ax_e.transAxes,
            ha='right',
            va='top',
            fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.8, 'pad': 1},
        )
        for ax in (ax_w, ax_e):
            ax.tick_params(labelbottom=False)
        ax_i.set_xlabel(time_label(unit))
        ax_w.set_xlim(t.min(), t.max())
        for ax in (ax_w, ax_e, ax_i):
            ax.grid(alpha=0.15, lw=0.5)
            ax.yaxis.set_label_coords(-0.055 * 11 / series_w, 0.5)
        for ax, letter in zip((ax_w, ax_e, ax_i), 'abc'):
            panel_letter(ax, letter, inside=True)

        # (d) eccentricity vector against omega, coloured by time
        x_port = L + series_w + G
        ax_p = axes(x_port, y_top, right_w, right_w)
        norm = Normalize(t.min(), t.max())
        k, h = body.ecc * np.cos(body.omega), body.ecc * np.sin(body.omega)
        ax_p.scatter(k[order], h[order], c=t[order], cmap=time_cmap(), norm=norm, s=style.marker, linewidths=0, rasterized=True)
        lim = 1.05 * float(np.nanmax(body.ecc)) or 1.0
        ax_p.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')
        ax_p.axhline(0, color=RAW, lw=0.5, ls='--')
        ax_p.axvline(0, color=RAW, lw=0.5, ls='--')
        ax_p.set_xlabel(r'$e\cos\omega$')
        ax_p.set_ylabel(r'$e\sin\omega$')
        ax_p.grid(alpha=0.15, lw=0.5)
        panel_letter(ax_p, 'd')
        time_colorbar(fig, norm, time_label(unit), cax=axes(x_port + right_w + 0.12, y_top, 0.15, right_w))

        # (e) free omega against osculating | (f) e-i plane | (g) e-i coherence
        coh_w = 3.4
        lower_w = W - L - R - 2 * G - lower_h - 0.95 - coh_w
        ax_f = axes(L, B, lower_w, lower_h)
        _draw_free_omega(ax_f, body, cfg, t_yr[order], divisor, info)
        ax_f.set_xlabel(time_label(unit))
        ax_f.grid(alpha=0.15, lw=0.5)
        panel_letter(ax_f, 'e')

        # (f) e against i coloured by time: the Kozai exchange is a curve with negative slope
        x_ei = L + lower_w + G
        ax_ei = axes(x_ei, B, lower_h, lower_h)
        ax_ei.plot(inc_deg[order], body.ecc[order], color=RAW, lw=0.4, zorder=1)
        ax_ei.scatter(
            inc_deg[order],
            body.ecc[order],
            c=t[order],
            cmap=time_cmap(),
            norm=norm,
            s=style.marker,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        ax_ei.set_xlabel('$i$ (deg)')
        ax_ei.set_ylabel('$e$')
        ax_ei.grid(alpha=0.15, lw=0.5)
        panel_letter(ax_ei, 'f')

        # (g) e-i coherence (decisive Welch level of the package's lidov_kozai preset)
        x_coh = W - R - coh_w
        ax_c = axes(x_coh, B, coh_w, lower_h)
        _draw_coherence(ax_c, analyse_resonance(body, resonance, cfg).get('e-i'), info)
        panel_letter(ax_c, 'g')
    return fig, info


def _draw_free_omega(ax_f, body, cfg, t_sorted: np.ndarray, divisor: float, info: Dict[str, Any]):
    """Free omega (forced terms at the planetary fundamentals removed) against the osculating one."""
    from resonances.plotting.style import RAW

    elements = None
    try:
        sampling = cfg.free_elements_sampling_years or 500.0
        elements = body.build_free_elements(sampling_years=sampling, mask_quantile=cfg.free_gate_mask_quantile)
    except Exception as exc:  # the panel then says why
        info['free_error'] = repr(exc)
    if elements is not None and len(elements.omega_times) > 1:
        grid = elements.omega_times
        tt = grid / divisor
        free = np.degrees(elements.omega_free)
        osc_omega = np.degrees(np.unwrap(np.interp(grid, t_sorted, np.unwrap(body.omega[np.argsort(body.times)]))))
        ax_f.plot(tt, osc_omega, color=RAW, lw=3.0, solid_capstyle='round', label=r'osculating $\omega$')
        ax_f.plot(tt, free, color='#1f6fb4', lw=1.1, label=r'free $\omega$ (forced terms removed)')
        span = max(np.ptp(free), np.ptp(osc_omega))
        if span > 1080:
            ax_f.set_ylabel(r'$\omega$ (deg, unwrapped; > 3 turns)')
        else:
            ax_f.set_ylabel(r'$\omega$ (deg, unwrapped)')
        ax_f.legend(frameon=False, fontsize=9, loc='best')
        ax_f.set_xlim(t_sorted[0] / divisor, t_sorted[-1] / divisor)
        info['free'] = {
            'sampling_yr': sampling,
            'mask_e': float(elements.mask_fraction_e),
            'mask_i': float(elements.mask_fraction_i),
            'basis_e': list(elements.basis_e),
            'basis_i': list(elements.basis_i),
        }
    else:
        ax_f.text(
            0.5,
            0.5,
            'free elements unavailable\n' + info.get('free_error', ''),
            transform=ax_f.transAxes,
            ha='center',
            va='center',
            color='#555555',
        )
        ax_f.set_yticks([])


def _draw_coherence(ax, analysis, info: Dict[str, Any]):
    from resonances.plotting.style import CENTRE, INK, RAW

    level = getattr(analysis, 'decisive', None)
    if level is None or not level.valid or level.frequency is None:
        reasons = [lv.reason for lv in (analysis.levels if analysis is not None else []) if lv.reason]
        valid_hi = [lv.period_max for lv in (analysis.levels if analysis is not None else []) if np.isfinite(lv.period_max)]
        if any('empty validity band' in r for r in reasons) and valid_hi:
            short = f'search band empty:\nP_min of the LK preset\nabove T/(5 n_seg) <= {fmt_years(max(valid_hi))}'
        else:
            short = '\n'.join(sorted(set(reasons)))[:120] or 'no pair analysed'
        ax.text(
            0.5,
            0.5,
            'e-i coherence\nunavailable\n\n' + short,
            transform=ax.transAxes,
            ha='center',
            va='center',
            color='#555555',
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        info['coherence'] = {'available': False, 'reason': '; '.join(sorted(set(reasons)))}
        return
    band = level.in_band & (level.frequency > 0)
    period = 1.0 / level.frequency[band]
    divisor, unit = (1e3, 'kyr') if np.nanmax(period) >= 1e4 else (1.0, 'yr')
    ax.plot(period / divisor, level.coherence[band], color=INK, lw=1.0, marker='o', ms=2.5)
    ax.axhline(level.gamma2_crit, color=RAW, ls='--', lw=0.9, label=rf'threshold ($n_{{seg}}={level.n_seg}$)')
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(f'Period ({unit})')
    ax.set_ylabel(r'$\gamma^2_{e,i}$')
    ax.grid(alpha=0.15, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    entry: Dict[str, Any] = {'available': True, 'n_seg': level.n_seg, 'band_yr': [level.period_min, level.period_max]}
    top = level.top_line
    if top is not None:
        ax.axvline(top.period / divisor, color=CENTRE, lw=0.9)
        title = f'e-i coherence: top line {fmt_years(top.period)}\nphase {top.phase_deg:+.0f}° (antiphase ±180°)'
        if np.isfinite(top.r_band):
            title += f', band r {top.r_band:+.2f}'
        entry.update(period=top.period, phase_deg=top.phase_deg, gamma2=top.gamma2, r_band=top.r_band)
    else:
        title = 'e-i coherence: no coherent line'
    ax.set_title(title, fontsize=9, loc='right')
    info['coherence'] = entry


# ------------------------------------------------------------------ one example


def _sidecar(example_id: str) -> Path:
    return bio.example_dir(example_id) / SIDECAR


def up_to_date(example_id: str, show_labels: bool) -> bool:
    png, side = bio.example_dir(example_id) / PNG, _sidecar(example_id)
    if not png.exists() or not side.exists():
        return False
    try:
        info = json.loads(side.read_text())
    except Exception:
        return False
    return info.get('show_labels') == show_labels and info.get('render_version') == RENDER_VERSION


def render_one(args) -> Dict[str, Any]:
    """Render one example; never raises (errors come back in the result)."""
    example_id, show_labels, force = args
    if not force and up_to_date(example_id, show_labels):
        return {'example_id': example_id, 'status': 'skipped'}
    import matplotlib.pyplot as plt

    logging.getLogger('resonances').setLevel(logging.ERROR)
    start = time.perf_counter()
    fig = None
    try:
        ex = bio.load_example(example_id)
        meta = ex.meta
        cutoff, why = choose_cutoff(meta, ex.series)
        extra, shown_cutoff = '', cutoff
        if meta['rtype'] == 'secular':
            fig, info, _, extra = render_secular(ex, cutoff)
            shown_cutoff, why = None, 'secular: the figure smooths on its own window (centre check), the cutoff is unused'
            band = 0.4
        elif meta['rtype'] == 'zlk':
            fig, info = render_zlk(ex, cutoff)
            if not str(meta['regime'].get('elements_source') or ''):
                extra = 'elements source unrecorded'
            band = HEADER_IN
        else:
            raise ValueError(f'unknown rtype {meta["rtype"]}')
        add_header(fig, meta, shown_cutoff, show_labels, extra, band=band)
        out = ex.path / PNG
        bio.save_figure(fig, out, 110)
        seconds = time.perf_counter() - start
        bio.write_json(
            _sidecar(example_id),
            {
                'render_version': RENDER_VERSION,
                'script': 'benchmark/tools/render.py',
                'package_commit': bio.package_commit(),
                'created': datetime.datetime.now().isoformat(timespec='seconds'),
                'show_labels': show_labels,
                'cutoff_per_yr': cutoff,
                'cutoff_reason': why,
                'seconds': round(seconds, 2),
                **info,
            },
        )
        return {'example_id': example_id, 'status': 'ok', 'seconds': seconds, 'rtype': meta['rtype']}
    except Exception as exc:
        return {'example_id': example_id, 'status': 'error', 'error': f'{exc!r}', 'trace': traceback.format_exc()}
    finally:
        if fig is not None:
            plt.close(fig)
        plt.close('all')


# ------------------------------------------------------------------ labels and review tree


def ensure_labels() -> None:
    """Create labels.csv from the examples and candidates.csv (proposed_bucket, confidence) if it is absent."""
    if bio.LABELS.exists():
        return
    candidates = pd.read_csv(bio.BENCH / 'candidates.csv', dtype=str, keep_default_na=False).set_index('id')
    rows = []
    for eid in bio.list_examples():
        meta = bio.read_meta(eid)
        cid = meta.get('candidate_id', '')
        cand = candidates.loc[cid] if cid in candidates.index else None
        rows.append(
            {
                'example_id': eid,
                'candidate_id': cid,
                'rtype': meta['rtype'],
                'proposed_bucket': cand['proposed_bucket'] if cand is not None else (meta.get('labels') or {}).get('proposed_bucket', ''),
                'confidence': cand['confidence'] if cand is not None else (meta.get('labels') or {}).get('confidence', ''),
            }
        )
    bio.upsert_labels(rows)


INDEX_CSS = (
    'body{font:14px -apple-system,Helvetica,Arial,sans-serif;margin:16px;background:#fafafa;color:#222}'
    'a{color:#1f5fa8;text-decoration:none}.grid{display:flex;flex-wrap:wrap;gap:12px}'
    '.card{background:#fff;border:1px solid #ddd;padding:6px;width:480px}'
    '.card img{width:480px;height:auto;display:block}.id{font:12px monospace;margin-top:4px;word-break:break-all}'
    'table{border-collapse:collapse}td,th{padding:3px 10px;border-bottom:1px solid #e3e3e3;text-align:left}'
)


def _page(title: str, body: str) -> str:
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{html.escape(title)}</title><style>{INDEX_CSS}</style></head><body>{body}</body></html>\n'
    )


def write_indexes() -> int:
    """index.html in every review leaf folder (thumbnails linking to the full PNG) and one at review/."""
    if not bio.REVIEW.exists():
        return 0
    leaves = sorted({p.parent for p in bio.REVIEW.rglob('*.png')})
    rows = []
    for folder in leaves:
        pngs = sorted(folder.glob('*.png'))
        rel = folder.relative_to(bio.REVIEW)
        cards = ''.join(
            f'<div class="card"><a href="{html.escape(p.name)}"><img loading="lazy" src="{html.escape(p.name)}"></a>'
            f'<div class="id">{html.escape(p.stem)}</div></div>'
            for p in pngs
        )
        depth = '../' * len(rel.parts)
        body = f'<p><a href="{depth}index.html">&larr; all folders</a></p><h2>{html.escape(str(rel))} ({len(pngs)})</h2><div class="grid">{cards}</div>'
        (folder / 'index.html').write_text(_page(str(rel), body))
        rows.append((rel, len(pngs)))
    table = ''.join(
        f'<tr><td><a href="{html.escape(str(rel))}/index.html">{html.escape(str(rel))}</a></td><td>{n}</td></tr>' for rel, n in rows
    )
    total = sum(n for _, n in rows)
    body = f'<h2>Benchmark v2 review ({total} figures)</h2><p>rtype / bucket / confidence</p><table><tr><th>folder</th><th>n</th></tr>{table}</table>'
    (bio.REVIEW / 'index.html').write_text(_page('Benchmark v2 review', body))
    return len(rows)


def rebuild_tree() -> Tuple[int, int]:
    ensure_labels()
    links = bio.build_review_tree(PNG)
    return links, write_indexes()


# ------------------------------------------------------------------ main


def select(args) -> List[str]:
    ids = bio.list_examples()
    if args.only:
        wanted = [s.strip() for s in args.only.split(',') if s.strip()]
        missing = [w for w in wanted if w not in ids]
        if missing:
            raise SystemExit(f'unknown example ids: {missing}')
        ids = wanted
    if args.rtype:
        allowed = set(args.rtype.split(','))
        ids = [i for i in ids if i.split('__', 1)[0] in allowed]
    if args.sample:
        rng = np.random.default_rng(args.seed)
        by_type: Dict[str, List[str]] = {}
        for i in ids:
            by_type.setdefault(i.split('__', 1)[0], []).append(i)
        ids = sorted(x for group in by_type.values() for x in rng.choice(group, size=min(args.sample, len(group)), replace=False))
    return ids


def render_v2(ids: List[str], workers: int) -> int:
    """combined_v2.png of every selected MMR example (the analysis of reclassify.py, its CSVs not written)."""
    import reclassify

    ids = [i for i in ids if i.split('__', 1)[0] in reclassify.MMR_RTYPES]
    print(f'{len(ids)} MMR examples, {workers} workers (v2 figures)')
    jobs = [(i, True) for i in ids]
    errors = 0
    with mp.get_context('spawn').Pool(max(1, workers), maxtasksperchild=20) as pool:
        for n, res in enumerate(pool.imap_unordered(reclassify._job, jobs), 1):
            ok = res['status'] == 'ok'
            errors += not ok
            line = f"{res['seconds']:.1f}s" if ok else res['error']
            print(f'[{n}/{len(jobs)}] {res["status"]:5s} {res["example_id"]}  {line}', flush=True)
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--only', help='comma-separated example ids')
    ap.add_argument('--rtype', help='comma-separated rtypes (mmr2,mmr3,coorbital,secular,zlk)')
    ap.add_argument('--sample', type=int, help='random N examples per rtype')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--force', action='store_true', help='re-render existing figures')
    ap.add_argument('--show-labels', action='store_true', help='add the OLD labels box (off by default: it biases the review)')
    ap.add_argument('--workers', type=int, default=max(1, min(12, (os.cpu_count() or 2) - 2)))
    ap.add_argument('--no-tree', action='store_true', help='do not rebuild data/review')
    ap.add_argument('--tree-only', action='store_true', help='only rebuild data/review and its index pages')
    ap.add_argument('--v2', action='store_true', help='the v2 combined figure (combined_v2.png, MMR examples only; no CSV)')
    args = ap.parse_args()

    if args.v2:
        return render_v2(select(args), args.workers)

    if args.tree_only:
        links, folders = rebuild_tree()
        print(f'review tree: {links} links in {folders} folders -> {bio.REVIEW}')
        return 0

    ids = select(args)
    mmr_ids = [i for i in ids if i.split('__', 1)[0] in ('mmr2', 'mmr3', 'coorbital')]
    if mmr_ids:  # the old MMR review page is gone from the package: the v2 figure instead
        render_v2(mmr_ids, args.workers)
        ids = [i for i in ids if i not in set(mmr_ids)]
    jobs = [(i, args.show_labels, args.force) for i in ids]
    print(f'{len(jobs)} examples, {args.workers} workers')
    start = time.perf_counter()
    results = []
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers, maxtasksperchild=25) as pool, open(LOG, 'a') as log:
        log.write(f'# {datetime.datetime.now().isoformat(timespec="seconds")} {" ".join(sys.argv[1:])}\n')
        for n, res in enumerate(pool.imap_unordered(render_one, jobs), 1):
            results.append(res)
            line = f'[{n}/{len(jobs)}] {res["status"]:7s} {res["example_id"]}'
            if res['status'] == 'ok':
                line += f'  {res["seconds"]:.1f}s'
            elif res['status'] == 'error':
                line += f'  {res["error"]}'
                log.write(res['trace'])
            print(line, flush=True)
            log.write(line + '\n')
    wall = time.perf_counter() - start

    ok = [r for r in results if r['status'] == 'ok']
    errors = [r for r in results if r['status'] == 'error']
    print(f'\nrendered {len(ok)}, skipped {sum(r["status"] == "skipped" for r in results)}, errors {len(errors)}; wall {wall:.0f}s')
    if ok:
        frame = pd.DataFrame(ok)
        print(frame.groupby('rtype')['seconds'].describe()[['count', 'mean', 'min', 'max']].round(1).to_string())
    for r in errors:
        print('ERROR', r['example_id'], r['error'])
    if not args.no_tree:
        links, folders = rebuild_tree()
        print(f'review tree: {links} links in {folders} folders -> {bio.REVIEW}')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
