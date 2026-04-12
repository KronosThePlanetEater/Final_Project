import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from path_layout import OUTPUT_ROOT, ensure_standard_directories, resolve_existing_path, resolve_output_path
from progress_utils import emit_progress


NUMERIC_COLUMNS = [
    "pure_fps",
    "mean_motion",
    "mean_mask_coverage",
    "failure_count",
    "empty_mask_count",
    "audio_runtime_seconds",
    "mean_iou",
    "si_sdr",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_float(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_run_rows(experiment_root: str) -> List[Dict[str, object]]:
    root = resolve_existing_path(experiment_root, extra_search_dirs=[OUTPUT_ROOT / "experiments", OUTPUT_ROOT]).resolve()
    rows = []
    for tracking_summary_path in root.rglob("metrics/tracking_summary.json"):
        if "_tracking_cache" in tracking_summary_path.parts:
            continue
        run_dir = tracking_summary_path.parent.parent
        eval_summary_path = run_dir / "eval" / "evaluation_summary.json"
        with open(tracking_summary_path, "r", encoding="utf-8") as handle:
            tracking = json.load(handle)
        evaluation = {}
        if eval_summary_path.exists():
            with open(eval_summary_path, "r", encoding="utf-8") as handle:
                evaluation = json.load(handle)

        row = {
            "run_dir": str(run_dir),
            "clip_id": tracking.get("clip_id"),
            "model_size": evaluation.get("model_size"),
            "model_id": evaluation.get("model_id"),
            "prompt_mode": evaluation.get("prompt_mode"),
            "backend": evaluation.get("backend"),
            "predict_spans": evaluation.get("predict_spans"),
            "reranking_candidates": evaluation.get("reranking_candidates"),
            "audio_runtime_seconds": evaluation.get("audio_runtime_seconds"),
            "pure_fps": tracking.get("pure_fps"),
            "mean_motion": tracking.get("mean_motion"),
            "mean_mask_coverage": tracking.get("mean_mask_coverage"),
            "failure_count": tracking.get("failure_count"),
            "empty_mask_count": tracking.get("empty_mask_count"),
            "mean_iou": evaluation.get("mean_iou"),
            "si_sdr": evaluation.get("si_sdr"),
            "si_sdr_evaluated": evaluation.get("si_sdr_evaluated"),
            "si_sdr_skip_reason": evaluation.get("si_sdr_skip_reason"),
        }
        rows.append(row)
    return rows


def write_aggregate_csv(rows: List[Dict[str, object]], csv_path: str) -> None:
    ensure_dir(Path(csv_path).parent)
    fieldnames = [
        "run_dir",
        "clip_id",
        "model_size",
        "model_id",
        "prompt_mode",
        "backend",
        "predict_spans",
        "reranking_candidates",
        "audio_runtime_seconds",
        "pure_fps",
        "mean_motion",
        "mean_mask_coverage",
        "failure_count",
        "empty_mask_count",
        "mean_iou",
        "si_sdr",
        "si_sdr_evaluated",
        "si_sdr_skip_reason",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scatter_plot(rows: List[Dict[str, object]], x_key: str, y_key: str, output_path: str) -> None:
    points = [
        (safe_float(row.get(x_key)), safe_float(row.get(y_key)))
        for row in rows
        if safe_float(row.get(x_key)) is not None and safe_float(row.get(y_key)) is not None
    ]
    if not points:
        return
    xs, ys = zip(*points)
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, alpha=0.8)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def correlation_outputs(rows: List[Dict[str, object]], output_dir: str) -> None:
    numeric_rows = []
    for row in rows:
        values = [safe_float(row.get(column)) for column in NUMERIC_COLUMNS]
        if any(value is not None for value in values):
            numeric_rows.append(values)
    if not numeric_rows:
        return

    matrix = np.array(
        [[np.nan if value is None else value for value in row] for row in numeric_rows],
        dtype=float,
    )
    corr = np.full((len(NUMERIC_COLUMNS), len(NUMERIC_COLUMNS)), np.nan)
    for i in range(len(NUMERIC_COLUMNS)):
        for j in range(len(NUMERIC_COLUMNS)):
            valid = ~np.isnan(matrix[:, i]) & ~np.isnan(matrix[:, j])
            if np.count_nonzero(valid) >= 2:
                corr[i, j] = np.corrcoef(matrix[valid, i], matrix[valid, j])[0, 1]

    corr_csv = Path(output_dir) / "correlation_matrix.csv"
    with open(corr_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + NUMERIC_COLUMNS)
        for metric, row in zip(NUMERIC_COLUMNS, corr):
            writer.writerow([metric] + list(row))

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.xticks(range(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS, rotation=45, ha="right")
    plt.yticks(range(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "correlation_matrix.png", dpi=150)
    plt.close()


def model_comparison_plot(rows: List[Dict[str, object]], metric_key: str, output_path: str) -> None:
    grouped = {}
    for row in rows:
        model = row.get("model_id") or row.get("model_size")
        value = safe_float(row.get(metric_key))
        if model is None or value is None:
            continue
        grouped.setdefault(model, []).append(value)
    if not grouped:
        return

    models = sorted(grouped)
    means = [float(np.mean(grouped[model])) for model in models]
    plt.figure(figsize=(7, 5))
    plt.bar(models, means)
    plt.ylabel(metric_key)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def analyze_results(experiment_root: str, output_dir: str | None, progress_callback=None) -> Dict[str, str]:
    ensure_standard_directories()
    emit_progress(progress_callback, stage="analytics", stage_progress=0.0, status="running", message="Loading run summaries for analytics.")
    rows = find_run_rows(experiment_root)
    out_dir = resolve_output_path(output_dir, "analysis").resolve()
    ensure_dir(out_dir)
    aggregate_csv = out_dir / "aggregate_results.csv"
    write_aggregate_csv(rows, str(aggregate_csv))
    emit_progress(progress_callback, stage="analytics", stage_progress=0.25, status="running", message="Building summary plots.", stats={"run_count": len(rows)})

    scatter_plot(rows, "mean_motion", "mean_iou", str(out_dir / "motion_vs_iou.png"))
    scatter_plot(rows, "mean_motion", "si_sdr", str(out_dir / "motion_vs_si_sdr.png"))
    scatter_plot(rows, "mean_mask_coverage", "pure_fps", str(out_dir / "coverage_vs_fps.png"))
    model_comparison_plot(rows, "mean_iou", str(out_dir / "model_mean_iou.png"))
    model_comparison_plot(rows, "si_sdr", str(out_dir / "model_si_sdr.png"))
    correlation_outputs(rows, str(out_dir))
    emit_progress(progress_callback, stage="analytics", stage_progress=0.85, status="running", message="Writing analytics summary.", stats={"run_count": len(rows)})

    summary = {
        "aggregate_csv": str(aggregate_csv),
        "run_count": len(rows),
        "analysis_dir": str(out_dir),
    }
    with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_path"] = str(out_dir / "analysis_summary.json")
    emit_progress(progress_callback, stage="analytics", stage_progress=1.0, status="completed", message="Analytics ready.", outputs={"analysis_dir": str(out_dir), "summary_path": summary["summary_path"]}, stats={"run_count": len(rows)})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate experiment outputs and generate plots.")
    parser.add_argument("experiment_root", nargs="?", default="experiments", help="Root directory containing experiment run folders. Defaults to output/experiments.")
    parser.add_argument("--output-dir", default=None, help="Directory for aggregated analysis outputs. Defaults to output/analysis.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    result = analyze_results(args.experiment_root, args.output_dir)
    print(json.dumps(result, indent=2))
