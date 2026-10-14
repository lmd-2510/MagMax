"""Export the completed MagMax result to CSV/JSON without loading a model."""
import argparse
import csv
import json
import math
from pathlib import Path


def summarize(run_dir):
    result = json.loads((run_dir / "result.json").read_text())
    expected = {"method": "MagMax", "dataset": "CIFAR100", "model": "ViT-B-16",
                "n_splits": 20, "seed": 5, "epochs_per_task": 10,
                "merge_coefficient": 0.5,
                "evaluation": "final full-dataset test, full-class head, val_preprocess"}
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Unexpected {key}: {result.get(key)!r}; expected {value!r}")
    if len(result["task_checkpoints"]) != 20:
        raise ValueError("Expected 20 task checkpoint records.")
    accuracy = float(result["top1"])
    percent = float(result["top1_percent"])
    if (not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or
            not math.isclose(percent, 100 * accuracy, abs_tol=1e-8)):
        raise ValueError("Invalid or inconsistent top-1 accuracy.")
    exit_path = run_dir / "exit_code.txt"
    if exit_path.exists() and exit_path.read_text().strip() != "0":
        raise ValueError("The latest full pipeline has not completed successfully.")
    row = {
        "method": "MagMax", "dataset": "CIFAR100", "model": "ViT-B-16",
        "tasks": 20, "epochs_per_task": 10, "seed": 5, "training_runs": 1,
        "merge_coefficient": 0.5, "evaluation": "full CIFAR100 test / 100-class head",
        "test_examples": 10000, "top1_percent": percent,
    }
    report = {
        "summary": row,
        "note": "One training seed; no standard deviation across seeds. "
                "This full-test result uses a different evaluation protocol from the target-subset results in the Tunable MagMax experiment.",
        "original_result": result,
        "run_metadata": {},
    }
    for name in ("source_commit.txt", "gpu.txt", "training_command.txt", "runtime.txt",
                 "total_seconds.txt", "stage.txt", "exit_code.txt"):
        path = run_dir / name
        if path.is_file():
            report["run_metadata"][name] = path.read_text().strip()
    # Preserve the commit recorded during training/evaluation; summary generation
    # may run after pulling a newer commit and must not replace that provenance.
    (run_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"MagMax | CIFAR100/20 | ViT-B-16 | seed5 | full test top-1: {percent:.2f}%")
    print(f"[SAVED] {run_dir / 'summary.csv'}")
    print(f"[SAVED] {run_dir / 'summary.json'}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summarize(args.run_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"[STOP] Cannot summarize MagMax: {exc}") from exc
