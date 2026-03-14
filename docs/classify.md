# Classification

The classification module determines whether a resonant angle time series represents libration (resonance), transient resonance, circulation, or a near-separatrix regime.

## Architecture

The classifier has three layers:

```
classify_resonance(body, resonance, config)       # Body → arrays → delegates
  └── classify_from_data(times, σ_w, σ_u, params) # arrays → metrics → delegates
       └── classify_from_metrics(metrics, counts, segments, params)  # pure decision logic
```

- **`classify_resonance`** — top layer, extracts arrays from a `Body` object, checks for unphysical orbits, then delegates downward. Use this during a simulation run.
- **`classify_from_data`** — middle layer, accepts raw NumPy arrays (times in years, wrapped and unwrapped angles). Computes metrics and segments, then delegates. Use this when you have angle arrays but no `Body`.
- **`classify_from_metrics`** — bottom layer, pure decision logic on pre-computed `SegmentMetrics`, `SegmentCounts`, and per-window segment metrics. Use this for reclassification from saved CSV data.

## ClassifyParams

All thresholds are collected in a single `ClassifyParams` dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_steps` | `[0.1, 0.2, 0.3]` | Window length fractions for segment analysis |
| `window_step_percentage` | `0.05` | Sliding step as fraction of total length |
| `rev_libration` | `1.0` | Max revolutions for global libration branch |
| `tto_pure_libration` | `0.5` | Max TTO for pure libration |
| `tto_partial_libration` | `2.5` | Max TTO for partial libration |
| `tto_non_resonant` | `6.0` | TTO above which → non-resonant |
| `tto_transient_global` | `3.0` | Max global TTO for transient |
| `tto_near_separatrix` | `6.0` | Max global TTO for near-separatrix |
| `good_seg_max_rev` | `1.0` | Max rev for "good" segment |
| `good_seg_max_tto` | `0.5` | Max TTO for "good" segment |
| `reasonable_seg_max_rev_1` | `2.0` | Max rev (condition 1) for "reasonable" |
| `reasonable_seg_max_tto_1` | `1.0` | Max TTO (condition 1) for "reasonable" |
| `reasonable_seg_max_rev_2` | `1.0` | Max rev (condition 2) for "reasonable" |
| `reasonable_seg_max_tto_2` | `1.5` | Max TTO (condition 2) for "reasonable" |

Defaults can be overridden via `.env`, `SimulationConfig` kwargs, or by passing a `ClassifyParams` directly.

## Decision tree

1. **Unphysical orbit** → `CHAOTIC`
2. **Global libration** (`rev_true <= rev_libration`):
   - TTO < `tto_pure_libration` → `LIBRATION` (pure, high confidence)
   - TTO < `tto_partial_libration` → `LIBRATION` (partial, medium confidence)
   - Otherwise → `PROBABLY_SLOW_CIRCULATION` (low confidence)
3. **Strong circulation** (`TTO > tto_non_resonant`) → `NON_RESONANT`
4. **Segment analysis** (good + reasonable segments across all windows):
   - Good segments + low global TTO → `TRANSIENT`
   - Good segments + high global TTO → `NEAR_SEPARATRIX`
   - No good but reasonable segments + moderate TTO → `PROBABLY_NEAR_SEPARATRIX`
   - Nothing matched → `NON_RESONANT` (remaining)

## Reclassification from CSV

```python
from resonances.resonance.classify import (
    classify_from_metrics,
    ClassifyParams,
    SegmentMetrics,
    SegmentCounts,
)

# Load pre-computed metrics from summary.csv
metrics = SegmentMetrics(revolutions_true=0.3, trend_to_oscillation=0.1, ...)
counts = SegmentCounts(...)
segments = {}  # or reconstructed from segments.csv

# Reclassify with stricter thresholds
params = ClassifyParams(tto_pure_libration=0.3)
result = classify_from_metrics(metrics, counts, segments, params)
print(result["result"].status)
```

To reclassify from raw arrays:

```python
from resonances.resonance.classify import classify_from_data, ClassifyParams

result = classify_from_data(times_yrs, sigma_wrapped, sigma_unwrapped,
                            params=ClassifyParams(tto_non_resonant=5.0))
```
