"""Benchmark for the classification algorithm."""

import shutil
from datetime import datetime
import pandas as pd
from pathlib import Path

from resonances.resonance.classify import classify_angle


STATUS_MAP = {
    "libration": 2,
    "transient": 1,
    "non-resonant": 0,
}

STATUS_NAMES = {0: "non-resonant", 1: "transient", 2: "libration"}


def run_benchmark(directory: str) -> dict:
    """Run classification benchmark on a directory with labeled examples.

    The directory should contain subfolders: libration, transient, non-resonant.
    Each subfolder contains CSV files with 'times' and 'angle' columns.
    """
    base_path = Path(directory)
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "failed_by_status": {0: 0, 1: 0, 2: 0},
        "missed_resonant": 0,
        "failures": [],
        "successes": [],
    }

    for folder_name, expected_status in STATUS_MAP.items():
        folder_path = base_path / folder_name
        if not folder_path.exists():
            continue

        csv_files = list(folder_path.rglob("*.csv"))
        for csv_file in csv_files:
            results["total"] += 1
            filepath = str(csv_file.relative_to(base_path))

            try:
                df = pd.read_csv(csv_file)
                if "times" not in df.columns or "angle" not in df.columns:
                    results["failed"] += 1
                    results["failures"].append(
                        {
                            "file": filepath,
                            "expected": expected_status,
                            "actual": None,
                            "error": "missing columns",
                            "diag": None,
                        }
                    )
                    continue

                times = df["times"].values
                angles = df["angle"].values

                classification = classify_angle(times, angles)
                actual_status = classification["status"]

                diag = {
                    "r_sq": classification["r_squared"],
                    "lib_frac": classification["libration_fraction"],
                    "unif": classification["angle_uniformity"],
                    "is_circ": classification["is_circulation"],
                    "cycles": classification["total_drift_cycles"],
                    "total_lib_time": classification.get("total_libration_time_frac", 0),
                    "libration_periods": classification.get("libration_periods", []),
                }

                entry = {
                    "file": filepath,
                    "expected": expected_status,
                    "actual": actual_status,
                    "error": None,
                    "diag": diag,
                }

                if actual_status == expected_status:
                    results["passed"] += 1
                    results["successes"].append(entry)
                else:
                    results["failed"] += 1
                    results["failed_by_status"][actual_status] += 1
                    if expected_status in (1, 2) and actual_status == 0:
                        results["missed_resonant"] += 1
                    results["failures"].append(entry)

            except Exception as e:
                results["failed"] += 1
                results["failures"].append(
                    {
                        "file": filepath,
                        "expected": expected_status,
                        "actual": None,
                        "error": str(e),
                        "diag": None,
                    }
                )

    return results


def _format_diag(diag: dict) -> str:
    """Format diagnostics for display."""
    if diag is None:
        return ""
    return (
        f"r_sq={diag['r_sq']:.3f}, lib_frac={diag['lib_frac']:.3f}, "
        f"unif={diag['unif']:.3f}, cycles={diag['cycles']:.1f}, is_circ={diag['is_circ']}"
    )


def _format_libration_periods(diag: dict) -> str:
    """Format libration periods for display."""
    if diag is None:
        return ""

    total_lib_time = diag.get("total_lib_time", 0)
    periods = diag.get("libration_periods", [])

    lines = [f"    Total libration time: {total_lib_time*100:.1f}%"]

    if periods:
        lines.append(f"    Libration periods ({len(periods)}):")
        for i, p in enumerate(periods):
            lines.append(f"      {i+1}. {p['start_frac']*100:.1f}%-{p['end_frac']*100:.1f}% " f"(duration: {p['duration_frac']*100:.1f}%)")
    else:
        lines.append("    No libration periods detected")

    return "\n".join(lines)


def print_benchmark_results(results: dict, verbose: int = 0):
    """Print benchmark results.

    Args:
        results: Benchmark results from run_benchmark()
        verbose: 0=summary only, 1=show failed diagnostics, 2=show all diagnostics
    """
    print(f"\n{'='*60}")
    print("CLASSIFICATION BENCHMARK RESULTS")
    print(f"{'='*60}\n")

    print(f"Total files:     {results['total']}")
    print(f"Passed:          {results['passed']}")
    print(f"Failed:          {results['failed']}")

    if results["total"] > 0:
        accuracy = results["passed"] / results["total"] * 100
        print(f"Accuracy:        {accuracy:.1f}%")

    print(f"\nFailed by predicted status:")
    print(f"  Status 0 (non-resonant): {results['failed_by_status'][0]}")
    print(f"  Status 1 (transient):    {results['failed_by_status'][1]}")
    print(f"  Status 2 (libration):    {results['failed_by_status'][2]}")

    print(f"\nMissed resonant (expected 1/2, got 0): {results['missed_resonant']}")

    if results["failures"]:
        print(f"\n{'='*60}")
        print("FAILED FILES:")
        print(f"{'='*60}")
        for entry in results["failures"]:
            if entry["error"]:
                print(f"  {entry['file']}: ERROR - {entry['error']}")
            else:
                exp_name = STATUS_NAMES[entry["expected"]]
                act_name = STATUS_NAMES[entry["actual"]]
                print(f"  {entry['file']}: expected={entry['expected']} ({exp_name}), got={entry['actual']} ({act_name})")
                if verbose >= 1 and entry["diag"]:
                    print(f"    {_format_diag(entry['diag'])}")
                    print(_format_libration_periods(entry['diag']))

    if verbose >= 2 and results["successes"]:
        print(f"\n{'='*60}")
        print("PASSED FILES:")
        print(f"{'='*60}")
        for entry in results["successes"]:
            status_name = STATUS_NAMES[entry["actual"]]
            print(f"  {entry['file']}: status={entry['actual']} ({status_name})")
            if entry["diag"]:
                print(f"    {_format_diag(entry['diag'])}")


def save_failed_files(results: dict, directory: str):
    """Save misclassified files to a separate folder organized by expected/actual status.

    Creates a timestamped folder with subfolders like exp_0_got_1, exp_1_got_2, etc.
    Only creates subfolders that contain at least one file.
    """
    if not results["failures"]:
        print("\nNo failures to save.")
        return

    base_path = Path(directory)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = base_path / f"misclassified-{timestamp}"

    copied_count = 0
    for entry in results["failures"]:
        if entry["error"] or entry["actual"] is None:
            continue

        expected = entry["expected"]
        actual = entry["actual"]
        folder_name = f"exp_{expected}_got_{actual}"
        dest_folder = output_dir / folder_name

        src_csv = base_path / entry["file"]
        if src_csv.exists():
            dest_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_csv, dest_folder / src_csv.name)
            copied_count += 1

            src_png = src_csv.with_suffix(".png")
            if src_png.exists():
                shutil.copy2(src_png, dest_folder / src_png.name)

    print(f"\nSaved {copied_count} misclassified files to: {output_dir}")
