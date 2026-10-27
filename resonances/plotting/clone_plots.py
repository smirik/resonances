"""The figure of an ensemble of clones in one resonance (docs/clones.md): three panels.

(a) the angle of every clone against time, coloured by its status, the nominal in black;
(b) the status of every clone against its initial semi-major axis, with the resonance's a* and, when the clones' accepted
    MMR episodes measured it, a* +- the libration half-width: where the resonance's border cuts the cloud;
(c) the share of each status with its Wilson 95 % interval; the nominal's status marked.
"""

from typing import Optional

import numpy as np

from .base import BasePlotter, time_unit
from .style import STATUS_COLOURS

STATUS_ORDER = ('resonant', 'transient', 'near_separatrix', 'non_resonant', 'uncertain', 'invalid')
NO_VERDICT = '#bbbbbb'


class EnsemblePlotter(BasePlotter):
    """The ensemble figure; `plot(...)` draws it and returns self (save/show/close from BasePlotter)."""

    def plot(
        self,
        parent: str,
        key: str,
        members: list,
        row: dict,
        nominal_series=None,
        nominal_a0: Optional[float] = None,
        nominal_status: Optional[str] = None,
        a_star: Optional[float] = None,
        half_width: Optional[float] = None,
    ):
        """`members`: one dict per clone (status or None, a0 [AU], series = (years, degrees) or None); `row`: the
        clones.csv row (shares and their Wilson bounds)."""
        import matplotlib.pyplot as plt

        self.close()
        figure, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw={'width_ratios': [2.2, 1.4, 1.2]})
        self._figure = figure
        self._angles(axes[0], members, nominal_series)
        self._status_vs_axis(axes[1], members, nominal_a0, nominal_status, a_star, half_width)
        self._shares(axes[2], row, nominal_status)
        n_valid = row.get('n_classified', 0)
        source = row.get('source', '')
        figure.suptitle(
            f"{parent}: {key} — {row.get('n_clones', len(members))} clones ({n_valid} classified, source {source}, "
            f"epoch MJD {row.get('epoch_mjd', float('nan')):.3f}); modal {row.get('modal_status', '')} "
            f"{100 * row.get('modal_share', float('nan')):.0f} %, nominal {nominal_status or '-'}",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        return self

    @staticmethod
    def _colour(status) -> str:
        return STATUS_COLOURS.get(str(status), NO_VERDICT) if status else NO_VERDICT

    def _angles(self, ax, members, nominal_series):
        spans = [np.nanmax(np.abs(m['series'][0])) for m in members if m.get('series') and len(m['series'][0])]
        if nominal_series is not None and len(nominal_series[0]):
            spans.append(np.nanmax(np.abs(nominal_series[0])))
        divisor, unit = time_unit(max(spans) if spans else 0.0)
        drawn = set()
        drawable = sum(1 for member in members if member.get('series'))
        alpha = float(np.clip(3.0 / np.sqrt(max(drawable, 1)), 0.08, 0.6))  # a cloud of 1000 stays readable
        for member in members:
            if not member.get('series'):
                continue
            t, angle = (np.asarray(x, dtype=float) for x in member['series'])
            status = member['status']
            label = status if status not in drawn else None
            drawn.add(status)
            ax.plot(t / divisor, angle, '.', color=self._colour(status), ms=2.0, alpha=alpha, mew=0, label=label)
        if nominal_series is not None:
            t, angle = nominal_series
            ax.plot(np.asarray(t) / divisor, angle, '.', color='black', ms=0.8, mew=0, label='nominal')
        ax.set_xlabel(f'time [{unit}]')
        ax.set_ylabel('angle [deg]')
        ax.set_ylim(0, 360)
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.set_title('(a) angle of every clone, coloured by its status')
        if drawn or nominal_series is not None:
            ax.legend(loc='upper right', fontsize=8, markerscale=12)

    def _status_vs_axis(self, ax, members, nominal_a0, nominal_status, a_star, half_width):
        from matplotlib.ticker import MaxNLocator

        position = {status: i for i, status in enumerate(STATUS_ORDER)}
        values = np.array([m['a0'] for m in members], dtype=float)
        finite = values[np.isfinite(values)]
        # a around a reference value, so a cloud of 1e-9 AU and a* 1e-4 AU away both read
        reference = nominal_a0 if nominal_a0 is not None else (float(np.median(finite)) if len(finite) else 0.0)
        rng = np.random.default_rng(0)  # vertical jitter only, to see overlapping clones
        for member in members:
            status = member['status']
            if status not in position or not np.isfinite(member['a0']):
                continue
            ax.plot(
                member['a0'] - reference, position[status] + rng.uniform(-0.15, 0.15), 'o', ms=3.5, color=self._colour(status), alpha=0.7
            )
        if nominal_a0 is not None and nominal_status in position:
            ax.plot(0.0, position[nominal_status], '*', ms=13, color='black', label='nominal')
        if a_star is not None and np.isfinite(a_star):
            ax.axvline(a_star - reference, color='#4d4d4d', ls=':', lw=1.2, label='a* (exact commensurability)')
            if half_width is not None and np.isfinite(half_width):
                for edge in (a_star - half_width, a_star + half_width):
                    ax.axvline(edge - reference, color='#2c7bb6', ls='--', lw=1.0)
                ax.plot([], [], color='#2c7bb6', ls='--', label='a* +- half-width (clones\' episodes)')
        ax.set_yticks(range(len(STATUS_ORDER)))
        ax.set_yticklabels(STATUS_ORDER)
        ax.set_ylim(-0.6, len(STATUS_ORDER) - 0.4)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3), useOffset=False)
        ax.set_xlabel(f'initial a - {reference:.8f} [AU] (osculating, at the run date)')
        ax.set_title('(b) status against initial a')
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='best', fontsize=8)

    def _shares(self, ax, row, nominal_status):
        shares = np.array([row.get(f'f_{s}', np.nan) for s in STATUS_ORDER], dtype=float)
        low = np.array([row.get(f'f_{s}_lo', np.nan) for s in STATUS_ORDER], dtype=float)
        high = np.array([row.get(f'f_{s}_hi', np.nan) for s in STATUS_ORDER], dtype=float)
        y = np.arange(len(STATUS_ORDER))
        values = np.nan_to_num(shares)
        ax.barh(y, values, color=[self._colour(s) for s in STATUS_ORDER], alpha=0.85)
        # clipped at 0: at 0/N or N/N a bound equals the share up to round-off
        errors = np.clip(np.vstack([np.nan_to_num(values - low), np.nan_to_num(high - values)]), 0.0, None)
        ax.errorbar(values, y, xerr=errors, fmt='none', ecolor='black', capsize=3, lw=1)
        for yi, value in zip(y, values):
            ax.text(min(value, 1.0) + 0.02, yi, f'{100 * value:.0f} %', va='center', fontsize=8)
        if nominal_status in STATUS_ORDER:
            ax.plot(-0.04, STATUS_ORDER.index(nominal_status), '*', ms=12, color='black', clip_on=False)
        ax.set_yticks(y)
        ax.set_yticklabels(STATUS_ORDER)
        ax.set_xlim(0, 1.15)
        ax.set_ylim(-0.6, len(STATUS_ORDER) - 0.4)
        ax.set_xlabel('share of clones (bars: Wilson 95 %)')
        ax.set_title('(c) status shares (* nominal)')
