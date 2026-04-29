import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

AGGREGATE_FIELDNAMES = [
    "run_dir",
    "run_id",
    "job_id",
    "display_label",
    "clip_id",
    "motion_level",
    "tracker_variant_key",
    "tracker_variant_label",
    "audio_model_size",
    "model_size",
    "model_id",
    "prompt_mode",
    "backend",
    "predict_spans",
    "reranking_candidates",
    "requested_audio_precision",
    "effective_audio_precision",
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
    "tracking_summary_path",
    "audio_metadata_path",
    "evaluation_summary_path",
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


def load_json_if_exists(path_value: Optional[str]) -> Dict[str, object]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_job_id_from_run_dir(run_dir: Path) -> Optional[str]:
    parts = list(run_dir.parts)
    if "ui_runs" in parts:
        index = parts.index("ui_runs")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def fallback_clip_id(tracking: Dict[str, object], run_dir: Path) -> Optional[str]:
    clip_id = tracking.get("clip_id")
    if isinstance(clip_id, str) and clip_id and clip_id != "tracked":
        return clip_id
    video_path = tracking.get("video_path")
    if isinstance(video_path, str) and video_path:
        return Path(video_path).stem
    return run_dir.name


def derive_row(
    run_dir: Path,
    tracking_summary_path: str,
    audio_metadata_path: Optional[str] = None,
    evaluation_summary_path: Optional[str] = None,
    metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    tracking = load_json_if_exists(tracking_summary_path)
    if not tracking:
        raise FileNotFoundError(f"Missing tracking summary: {tracking_summary_path}")
    audio = load_json_if_exists(audio_metadata_path)
    evaluation = load_json_if_exists(evaluation_summary_path)
    metadata = metadata or {}

    clip_id = (
        evaluation.get("clip_id")
        or audio.get("clip_id")
        or metadata.get("clip_id")
        or fallback_clip_id(tracking, run_dir)
    )

    run_id = metadata.get("run_id") or run_dir.name
    tracker_variant_key = metadata.get("tracker_variant_key") or str(run_id).split("__")[0]
    tracker_variant_label = metadata.get("tracker_variant_label") or tracker_variant_key
    audio_model_size = (
        metadata.get("audio_model_size")
        or audio.get("model_size")
        or evaluation.get("model_size")
        or (str(run_id).split("__", 1)[1] if "__" in str(run_id) else None)
    )
    display_label = metadata.get("display_label") or f"{tracker_variant_label} | {audio_model_size or 'audio'}"

    return {
        "run_dir": str(run_dir.resolve()),
        "run_id": run_id,
        "job_id": metadata.get("job_id") or infer_job_id_from_run_dir(run_dir),
        "display_label": display_label,
        "clip_id": clip_id,
        "motion_level": metadata.get("motion_level"),
        "tracker_variant_key": tracker_variant_key,
        "tracker_variant_label": tracker_variant_label,
        "audio_model_size": audio_model_size,
        "model_size": evaluation.get("model_size") or audio.get("model_size"),
        "model_id": evaluation.get("model_id") or audio.get("model_id"),
        "prompt_mode": evaluation.get("prompt_mode") or audio.get("prompt_mode"),
        "backend": evaluation.get("backend") or audio.get("backend"),
        "predict_spans": evaluation.get("predict_spans") if evaluation else audio.get("predict_spans"),
        "reranking_candidates": evaluation.get("reranking_candidates") if evaluation else audio.get("reranking_candidates"),
        "requested_audio_precision": audio.get("requested_audio_precision"),
        "effective_audio_precision": audio.get("effective_audio_precision"),
        "audio_runtime_seconds": evaluation.get("audio_runtime_seconds") if evaluation else audio.get("runtime_seconds"),
        "pure_fps": tracking.get("pure_fps"),
        "mean_motion": tracking.get("mean_motion"),
        "mean_mask_coverage": tracking.get("mean_mask_coverage"),
        "failure_count": tracking.get("failure_count"),
        "empty_mask_count": tracking.get("empty_mask_count"),
        "mean_iou": evaluation.get("mean_iou"),
        "si_sdr": evaluation.get("si_sdr"),
        "si_sdr_evaluated": evaluation.get("si_sdr_evaluated"),
        "si_sdr_skip_reason": evaluation.get("si_sdr_skip_reason"),
        "tracking_summary_path": str(Path(tracking_summary_path).resolve()),
        "audio_metadata_path": str(Path(audio_metadata_path).resolve()) if audio_metadata_path else None,
        "evaluation_summary_path": str(Path(evaluation_summary_path).resolve()) if evaluation_summary_path else None,
    }


def collect_run_rows_from_manifest(run_manifest_path: str) -> Tuple[List[Dict[str, object]], int, int]:
    path = resolve_existing_path(run_manifest_path, extra_search_dirs=[OUTPUT_ROOT / "analysis_sets", OUTPUT_ROOT]).resolve()
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    entries = payload.get("runs") if isinstance(payload, dict) else payload
    entries = entries or []

    rows: List[Dict[str, object]] = []
    skipped_count = 0
    for entry in entries:
        tracking_summary_path = entry.get("tracking_summary_path")
        if not tracking_summary_path or not Path(tracking_summary_path).exists():
            skipped_count += 1
            continue
        try:
            run_dir = Path(entry.get("run_dir") or Path(tracking_summary_path).parent.parent)
            rows.append(
                derive_row(
                    run_dir=run_dir,
                    tracking_summary_path=tracking_summary_path,
                    audio_metadata_path=entry.get("audio_metadata_path"),
                    evaluation_summary_path=entry.get("evaluation_summary_path"),
                    metadata=entry,
                )
            )
        except Exception:
            skipped_count += 1
    return rows, len(rows), skipped_count


def collect_run_rows_from_root(experiment_root: str) -> Tuple[List[Dict[str, object]], int, int]:
    root = resolve_existing_path(experiment_root, extra_search_dirs=[OUTPUT_ROOT / "experiments", OUTPUT_ROOT]).resolve()
    rows: List[Dict[str, object]] = []
    skipped_count = 0
    seen_run_dirs = set()
    for tracking_summary_path in root.rglob("metrics/tracking_summary.json"):
        if "tracking_cache" in tracking_summary_path.parts or "analysis_sets" in tracking_summary_path.parts:
            continue
        run_dir = tracking_summary_path.parent.parent
        run_dir_key = str(run_dir.resolve())
        if run_dir_key in seen_run_dirs:
            continue
        seen_run_dirs.add(run_dir_key)
        eval_summary_path = run_dir / "eval" / "evaluation_summary.json"
        audio_metadata_path = run_dir / "audio" / "audio_run_metadata.json"
        try:
            rows.append(
                derive_row(
                    run_dir=run_dir,
                    tracking_summary_path=str(tracking_summary_path),
                    audio_metadata_path=str(audio_metadata_path) if audio_metadata_path.exists() else None,
                    evaluation_summary_path=str(eval_summary_path) if eval_summary_path.exists() else None,
                    metadata=None,
                )
            )
        except Exception:
            skipped_count += 1
    return rows, len(rows), skipped_count


def find_run_rows(experiment_root: str) -> List[Dict[str, object]]:
    rows, _, _ = collect_run_rows_from_root(experiment_root)
    return rows


def write_aggregate_csv(rows: List[Dict[str, object]], csv_path: str) -> None:
    ensure_dir(Path(csv_path).parent)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_placeholder_plot(output_path: str, title: str, message: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.axis("off")
    plt.title(title)
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def scatter_plot(rows: List[Dict[str, object]], x_key: str, y_key: str, output_path: str) -> bool:
    points = [
        (safe_float(row.get(x_key)), safe_float(row.get(y_key)))
        for row in rows
        if safe_float(row.get(x_key)) is not None and safe_float(row.get(y_key)) is not None
    ]
    if len(points) < 2:
        write_placeholder_plot(output_path, f"{x_key} vs {y_key}", "Insufficient data to plot this comparison.")
        return False
    xs, ys = zip(*points)
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys, alpha=0.8)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def correlation_outputs(rows: List[Dict[str, object]], output_dir: str) -> bool:
    numeric_rows = []
    for row in rows:
        values = [safe_float(row.get(column)) for column in NUMERIC_COLUMNS]
        if any(value is not None for value in values):
            numeric_rows.append(values)

    corr_csv = Path(output_dir) / "correlation_matrix.csv"
    corr_png = Path(output_dir) / "correlation_matrix.png"
    corr = np.full((len(NUMERIC_COLUMNS), len(NUMERIC_COLUMNS)), np.nan)

    if numeric_rows:
        matrix = np.array(
            [[np.nan if value is None else value for value in row] for row in numeric_rows],
            dtype=float,
        )
        for i in range(len(NUMERIC_COLUMNS)):
            for j in range(len(NUMERIC_COLUMNS)):
                valid = ~np.isnan(matrix[:, i]) & ~np.isnan(matrix[:, j])
                if np.count_nonzero(valid) >= 2:
                    values_i = matrix[valid, i]
                    values_j = matrix[valid, j]
                    if np.allclose(values_i, values_i[0]) or np.allclose(values_j, values_j[0]):
                        continue
                    corr[i, j] = np.corrcoef(values_i, values_j)[0, 1]

    with open(corr_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric"] + NUMERIC_COLUMNS)
        for metric, row in zip(NUMERIC_COLUMNS, corr):
            writer.writerow([metric] + list(row))

    if np.any(~np.isnan(corr)):
        plt.figure(figsize=(8, 6))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS, rotation=45, ha="right")
        plt.yticks(range(len(NUMERIC_COLUMNS)), NUMERIC_COLUMNS)
        plt.tight_layout()
        plt.savefig(corr_png, dpi=150)
        plt.close()
        return True

    write_placeholder_plot(str(corr_png), "Correlation Matrix", "Insufficient data to compute correlations. At least two valid runs are needed.")
    return False


def model_comparison_plot(rows: List[Dict[str, object]], metric_key: str, output_path: str) -> bool:
    grouped = {}
    for row in rows:
        model = row.get("model_id") or row.get("audio_model_size") or row.get("model_size")
        value = safe_float(row.get(metric_key))
        if model is None or value is None:
            continue
        grouped.setdefault(model, []).append(value)
    if not grouped:
        write_placeholder_plot(output_path, metric_key, "No valid runs contained this metric.")
        return False

    models = sorted(grouped)
    means = [float(np.mean(grouped[model])) for model in models]
    plt.figure(figsize=(7, 5))
    plt.bar(models, means)
    plt.ylabel(metric_key)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def categorical_mean_plot(rows: List[Dict[str, object]], category_key: str, metric_key: str, output_path: str) -> bool:
    grouped = {}
    for row in rows:
        category = row.get(category_key)
        value = safe_float(row.get(metric_key))
        if category in (None, "") or value is None:
            continue
        grouped.setdefault(str(category), []).append(value)
    if not grouped:
        write_placeholder_plot(output_path, f"{category_key} vs {metric_key}", "No valid runs contained this metric.")
        return False

    categories = sorted(grouped)
    means = [float(np.mean(grouped[category])) for category in categories]
    plt.figure(figsize=(8, 5))
    plt.bar(categories, means)
    plt.xlabel(category_key)
    plt.ylabel(metric_key)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def analyze_results(
    experiment_root: Optional[str],
    output_dir: str | None,
    progress_callback=None,
    run_manifest_path: Optional[str] = None,
) -> Dict[str, object]:
    ensure_standard_directories()
    emit_progress(progress_callback, stage="analytics", stage_progress=0.0, status="running", message="Loading run summaries for analytics.")

    if run_manifest_path:
        rows, eligible_count, skipped_count = collect_run_rows_from_manifest(run_manifest_path)
        source_mode = "explicit_manifest"
    else:
        root_value = experiment_root or "experiments"
        rows, eligible_count, skipped_count = collect_run_rows_from_root(root_value)
        source_mode = "root_scan"

    out_dir = resolve_output_path(output_dir, "analysis").resolve()
    ensure_dir(out_dir)
    aggregate_csv = out_dir / "aggregate_results.csv"
    write_aggregate_csv(rows, str(aggregate_csv))
    emit_progress(progress_callback, stage="analytics", stage_progress=0.25, status="running", message="Building summary plots.", stats={"run_count": len(rows), "eligible_run_count": eligible_count, "skipped_run_count": skipped_count})

    scatter_plot(rows, "mean_motion", "mean_iou", str(out_dir / "motion_vs_iou.png"))
    scatter_plot(rows, "mean_motion", "si_sdr", str(out_dir / "motion_vs_si_sdr.png"))
    scatter_plot(rows, "mean_iou", "si_sdr", str(out_dir / "iou_vs_si_sdr.png"))
    scatter_plot(rows, "mean_mask_coverage", "si_sdr", str(out_dir / "coverage_vs_si_sdr.png"))
    scatter_plot(rows, "mean_mask_coverage", "pure_fps", str(out_dir / "coverage_vs_fps.png"))
    categorical_mean_plot(rows, "motion_level", "mean_iou", str(out_dir / "motion_level_mean_iou.png"))
    categorical_mean_plot(rows, "motion_level", "si_sdr", str(out_dir / "motion_level_si_sdr.png"))
    categorical_mean_plot(rows, "tracker_variant_label", "mean_iou", str(out_dir / "tracker_mean_iou.png"))
    categorical_mean_plot(rows, "tracker_variant_label", "si_sdr", str(out_dir / "tracker_si_sdr.png"))
    categorical_mean_plot(rows, "audio_model_size", "si_sdr", str(out_dir / "audio_model_si_sdr.png"))
    model_comparison_plot(rows, "mean_iou", str(out_dir / "model_mean_iou.png"))
    model_comparison_plot(rows, "si_sdr", str(out_dir / "model_si_sdr.png"))
    correlation_outputs(rows, str(out_dir))
    emit_progress(progress_callback, stage="analytics", stage_progress=0.85, status="running", message="Writing analytics summary.", stats={"run_count": len(rows), "eligible_run_count": eligible_count, "skipped_run_count": skipped_count})

    summary = {
        "aggregate_csv": str(aggregate_csv),
        "run_count": int(eligible_count + skipped_count),
        "eligible_run_count": int(eligible_count),
        "skipped_run_count": int(skipped_count),
        "analysis_dir": str(out_dir),
        "source_mode": source_mode,
        "run_manifest_path": str(resolve_existing_path(run_manifest_path).resolve()) if run_manifest_path else None,
    }
    with open(out_dir / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_path"] = str(out_dir / "analysis_summary.json")
    emit_progress(progress_callback, stage="analytics", stage_progress=1.0, status="completed", message="Analytics ready.", outputs={"analysis_dir": str(out_dir), "summary_path": summary["summary_path"]}, stats={"run_count": summary["run_count"], "eligible_run_count": eligible_count, "skipped_run_count": skipped_count})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate experiment outputs and generate plots.")
    parser.add_argument("experiment_root", nargs="?", default="experiments", help="Root directory containing experiment run folders. Defaults to output/experiments.")
    parser.add_argument("--output-dir", default=None, help="Directory for aggregated analysis outputs. Defaults to output/analysis.")
    parser.add_argument("--run-manifest", default=None, help="Optional explicit run manifest JSON to analyze instead of scanning an experiment root.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    result = analyze_results(
        experiment_root=None if args.run_manifest else args.experiment_root,
        output_dir=args.output_dir,
        run_manifest_path=args.run_manifest,
    )
    print(json.dumps(result, indent=2))
