"""Classification benchmark: integrate ~270 curated body-resonance pairs
(100 kyr, SABA) and score the classifier against the adjudicated ground truth.

Run:
    make test-benchmark                 # metrics + human-readable report
    BENCHMARK_REPORT=1 make test-benchmark KEEP=1
        # additionally saves ALL data CSVs and plots (evolution + phase
        # portraits, grouped into folders by the PREDICTED status) and writes
        # review.csv (all rows, disagreements first) + mismatches.csv into the
        # run folder under cache/tests/. KEEP=1 prevents the cleanup fixture
        # from deleting the artifacts.

Ground truth: ground_truth.csv, built by
experiments/classify/build_ground_truth.py (legacy dataset + 70 independently
re-verified cases + the author's plot-review adjudication of 2026-07-04).

How to add a new object:
1. Append a row to ground_truth.csv:
   name,resonance,category,primary,acceptable,assert_group,source,comment
   - `primary` — the expected status; `acceptable` — semicolon-separated
     statuses counted as correct for the relaxed metric (list 2-3 statuses
     for genuinely ambiguous cases, e.g. "1;-1;-3").
   - `assert_group`: strict2 / strict1 (hard recall families), soft
     (scored by `acceptable`), exclude (report only).
2. For a body that is not in the AstDyS catalog (clones, custom initial
   conditions), add its elements to clones.json under the same name.

Statuses 2/-2 and 1/-1 are scored as one family each: periodograms are built
over the full series, so the period-match split carries no extra information
for transients.

Metrics reported: family recalls, strict accuracy (predicted == primary),
relaxed accuracy (predicted in acceptable), review load (share of statuses an
astronomer must eyeball), and the full confusion matrix. The asserts are
regression gates calibrated to the current classifier; the targets are
recall = 1.0 for both strict families.
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import pytest

import resonances

BENCHMARK_DIR = Path(__file__).parent

# Regression gates (calibrated 2026-07-04 on the adjudicated ground truth;
# targets are higher: 1.0 / 1.0 for the recall families)
MIN_RECALL_FAMILY2 = 0.93
MIN_RECALL_FAMILY1 = 0.85
MIN_RELAXED_ACCURACY = 0.80
MAX_REVIEW_LOAD = 0.65

REVIEW_STATUSES = {2, 1, -1, -2}
REPORT_MODE = os.environ.get("BENCHMARK_REPORT", "") not in ("", "0")


def _load_truth() -> pd.DataFrame:
    truth = pd.read_csv(BENCHMARK_DIR / "ground_truth.csv")
    truth["name"] = truth["name"].astype(str)
    return truth


def _build_simulation(truth: pd.DataFrame) -> resonances.Simulation:
    clones = json.load(open(BENCHMARK_DIR / "clones.json"))

    kwargs = dict(
        name="test_benchmark",
        integrator="SABA(10,6,4)",
        integration_years=100_000,
        source="astdys",
    )
    if REPORT_MODE:
        # Batched workers save the data and render the plots in parallel;
        # predictions are read back from the accumulated summary.csv.
        kwargs.update(
            save="all",
            plot="all",
            plots=["evolution", "phase_portrait"],
            plot_subfolder_strategy="status",
            batch_size=50,
            batch_threshold=50,
        )
    else:
        # No I/O; unbatched so that statuses stay on the parent's bodies.
        kwargs.update(save="none", plot="none", batch_threshold=100_000)

    sim = resonances.Simulation(**kwargs)

    resonances_per_body = defaultdict(list)
    for _, row in truth.iterrows():
        resonances_per_body[row["name"]].append(row["resonance"])
    for name, res_list in resonances_per_body.items():
        if name in clones:
            sim.add_body(clones[name]["initial_data"], res_list, name=name)
        else:
            sim.add_body(name, res_list, name=name)
    return sim


def _collect_predictions(sim) -> dict:
    """(name, resonance) -> (status, subtype)."""
    if REPORT_MODE:
        summary = pd.read_csv(f"{sim.config.save_path}/summary.csv", dtype=str)
        # Batch workers append concurrently; drop repeated header rows
        summary = summary[summary["status"] != "status"]
        return {(str(r["name"]), r["resonance"]): (int(r["status"]), str(r.get("subtype", ""))) for _, r in summary.iterrows()}
    predictions = {}
    for body in sim.bodies:
        for resonance in body.resonances():
            result = body.librations[resonance.to_s()]
            predictions[(body.name, resonance.to_s())] = (int(result.status), result.subtype)
    return predictions


def _score(truth: pd.DataFrame, predictions: dict) -> pd.DataFrame:
    rows = []
    for _, row in truth.iterrows():
        key = (row["name"], row["resonance"])
        if key not in predictions:
            continue
        pred, subtype = predictions[key]
        acceptable = {int(s) for s in str(row["acceptable"]).split(";")}
        primary = None if str(row["primary"]) in ("", "nan") else int(float(row["primary"]))
        rows.append(
            {
                "name": key[0],
                "resonance": key[1],
                "group": row["assert_group"],
                "primary": primary,
                "acceptable": row["acceptable"],
                "predicted": pred,
                "subtype": subtype,
                "strict_ok": primary is not None and pred == primary,
                "relaxed_ok": pred in acceptable,
                "comment": row["comment"],
            }
        )
    return pd.DataFrame(rows)


def _metrics(df: pd.DataFrame) -> dict:
    strict2 = df[df.group == "strict2"]
    strict1 = df[df.group == "strict1"]
    labeled = df[df.primary.notna()]
    return {
        "recall_family2": strict2.predicted.isin([2, -2]).mean() if len(strict2) else 1.0,
        "recall_family1": strict1.predicted.isin([1, -1]).mean() if len(strict1) else 1.0,
        "strict_accuracy": labeled.strict_ok.mean() if len(labeled) else 1.0,
        "relaxed_accuracy": labeled.relaxed_ok.mean() if len(labeled) else 1.0,
        "review_load": df.predicted.isin(list(REVIEW_STATUSES)).mean(),
    }


def _print_report(df: pd.DataFrame, metrics: dict) -> None:
    lines = [
        "",
        "================= classification benchmark =================",
        f"rows scored: {len(df)}",
        f"recall family {{2,-2}}  (target 1.00, gate {MIN_RECALL_FAMILY2:.2f}): {metrics['recall_family2']:.3f}",
        f"recall family {{1,-1}}  (target 1.00, gate {MIN_RECALL_FAMILY1:.2f}): {metrics['recall_family1']:.3f}",
        f"strict accuracy  (pred == primary)      : {metrics['strict_accuracy']:.3f}",
        f"relaxed accuracy (pred in acceptable, gate {MIN_RELAXED_ACCURACY:.2f}): {metrics['relaxed_accuracy']:.3f}",
        f"review load      (2/1/-1/-2, gate {MAX_REVIEW_LOAD:.2f})   : {metrics['review_load']:.1%}",
        "",
        "confusion matrix (true primary -> predicted):",
    ]
    conf = defaultdict(Counter)
    for _, r in df.iterrows():
        if pd.notna(r.primary):
            conf[int(r.primary)][int(r.predicted)] += 1
    statuses = sorted(set(conf) | {p for c in conf.values() for p in c}, reverse=True)
    header = "  true\\pred " + " ".join(f"{s:>5}" for s in statuses)
    lines.append(header)
    for s in sorted(conf, reverse=True):
        lines.append(f"  {s:>9} " + " ".join(f"{conf[s].get(p, 0):>5}" for p in statuses))

    mismatches = df[~df.relaxed_ok].sort_values(["group", "name"])
    lines.append("")
    lines.append(f"disagreements (predicted not in acceptable): {len(mismatches)}")
    if len(mismatches):
        lines.append(f"  {'object':>10} {'resonance':>18} {'group':>8} {'expected':>8} {'acceptable':>14} {'got':>5}  subtype")
        for _, r in mismatches.iterrows():
            expected = "" if pd.isna(r.primary) else str(int(r.primary))
            lines.append(f"  {r['name']:>10} {r.resonance:>18} {r.group:>8} {expected:>8} {r.acceptable:>14} {r.predicted:>5}  {r.subtype}")

    report = "\n".join(lines)
    print(report)
    resonances.logger.info(report)


def _save_artifacts(sim, df: pd.DataFrame) -> None:
    out = sim.config.save_path
    review = df.sort_values(["relaxed_ok", "group", "name"])
    review.to_csv(f"{out}/review.csv", index=False)
    review[~review.relaxed_ok].to_csv(f"{out}/mismatches.csv", index=False)
    print(f"\nartifacts: {out}/review.csv, {out}/mismatches.csv")
    print(f"plots (by predicted status): {sim.config.plot_path}/<status folder>/")


@pytest.mark.benchmark
def test_classification_ground_truth():
    truth = _load_truth()
    sim = _build_simulation(truth)
    sim.create_solar_system()
    sim.run(progress=True)

    df = _score(truth, _collect_predictions(sim))
    assert len(df) > 200, "benchmark lost most of its rows: check body setup"

    metrics = _metrics(df)
    _print_report(df, metrics)
    if REPORT_MODE:
        _save_artifacts(sim, df)

    assert metrics["recall_family2"] >= MIN_RECALL_FAMILY2
    assert metrics["recall_family1"] >= MIN_RECALL_FAMILY1
    assert metrics["relaxed_accuracy"] >= MIN_RELAXED_ACCURACY
    assert metrics["review_load"] <= MAX_REVIEW_LOAD
