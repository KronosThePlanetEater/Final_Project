import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.io import wavfile

from path_layout import INPUT_ROOT, OUTPUT_ROOT, PLACEHOLDER_ROOT, ensure_standard_directories, resolve_existing_path, resolve_output_path
from progress_utils import emit_progress


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_mask_data(mask_path: str) -> np.ndarray:
    path = Path(mask_path)
    if path.is_dir():
        pngs = sorted(path.glob("*.png"))
        if not pngs:
            raise ValueError(f"No PNG masks found in {path}")
        masks = [(cv2.imread(str(png), cv2.IMREAD_GRAYSCALE) > 0) for png in pngs]
        return np.stack(masks, axis=0).astype(bool)
    if path.suffix == ".npy":
        return np.load(path).astype(bool)
    if path.suffix == ".npz":
        return np.load(path)["masks"].astype(bool)
    raise ValueError("Mask input must be a directory of PNGs, .npy, or .npz.")


def compute_frame_ious(pred_masks: np.ndarray, gt_masks: np.ndarray) -> List[Dict[str, float]]:
    frame_count = min(pred_masks.shape[0], gt_masks.shape[0])
    rows = []
    for frame_idx in range(frame_count):
        pred = pred_masks[frame_idx].astype(bool)
        gt = gt_masks[frame_idx].astype(bool)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = 1.0 if union == 0 else float(intersection / union)
        rows.append({"frame_idx": frame_idx, "iou": iou})
    return rows


def write_iou_csv(rows: List[Dict[str, float]], csv_path: str) -> None:
    ensure_dir(Path(csv_path).parent)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["frame_idx", "iou"])
        writer.writeheader()
        writer.writerows(rows)


def resample_audio(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float64)
    duration = len(audio) / float(src_rate)
    src_positions = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    dst_length = int(round(duration * dst_rate))
    dst_positions = np.linspace(0.0, duration, num=dst_length, endpoint=False)
    return np.interp(dst_positions, src_positions, audio).astype(np.float64)


def load_audio(audio_path: str) -> Tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(audio_path)
    audio = audio.astype(np.float64)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return sample_rate, audio


def compute_si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    estimate = estimate.astype(np.float64)
    min_len = min(len(reference), len(estimate))
    reference = reference[:min_len]
    estimate = estimate[:min_len]

    if np.allclose(reference, 0) or np.allclose(estimate, 0):
        return float("-inf")

    scale = np.dot(estimate, reference) / np.dot(reference, reference)
    projection = scale * reference
    noise = estimate - projection
    if np.allclose(noise, 0):
        return float("inf")
    ratio = np.sum(projection ** 2) / np.sum(noise ** 2)
    return float(10.0 * np.log10(ratio))


def evaluate_run(
    output_dir: Optional[str],
    clip_id: str,
    model_size: Optional[str] = None,
    model_id: Optional[str] = None,
    predicted_mask_path: Optional[str] = None,
    ground_truth_mask_path: Optional[str] = None,
    estimated_audio_path: Optional[str] = None,
    reference_audio_path: Optional[str] = None,
    audio_metadata_path: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, object]:
    ensure_standard_directories()
    emit_progress(progress_callback, stage="evaluation", stage_progress=0.0, status="running", clip_id=clip_id, message="Starting evaluation.")

    resolved_predicted_mask_path = (
        str(resolve_existing_path(predicted_mask_path, extra_search_dirs=[OUTPUT_ROOT])) if predicted_mask_path else None
    )
    resolved_ground_truth_mask_path = (
        str(resolve_existing_path(ground_truth_mask_path, extra_search_dirs=[INPUT_ROOT, OUTPUT_ROOT]))
        if ground_truth_mask_path
        else None
    )
    resolved_estimated_audio_path = (
        str(resolve_existing_path(estimated_audio_path, extra_search_dirs=[OUTPUT_ROOT])) if estimated_audio_path else None
    )
    resolved_reference_audio_path = (
        str(resolve_existing_path(reference_audio_path, extra_search_dirs=[INPUT_ROOT, OUTPUT_ROOT]))
        if reference_audio_path
        else None
    )
    resolved_audio_metadata_path = (
        str(resolve_existing_path(audio_metadata_path, extra_search_dirs=[OUTPUT_ROOT])) if audio_metadata_path else None
    )

    placeholder_backend = False
    if resolved_audio_metadata_path:
        with open(resolved_audio_metadata_path, "r", encoding="utf-8") as handle:
            preview_audio_metadata = json.load(handle)
        placeholder_backend = preview_audio_metadata.get("backend") == "placeholder_passthrough"
    default_eval_relative = f"placeholder_test/eval/{clip_id}" if placeholder_backend else f"eval/{clip_id}"
    eval_dir = resolve_output_path(output_dir, default_eval_relative).resolve()
    ensure_dir(eval_dir)
    iou_csv_path = eval_dir / "frame_iou.csv"
    summary_path = eval_dir / "evaluation_summary.json"

    summary = {
        "clip_id": clip_id,
        "model_size": model_size,
        "model_id": model_id,
        "predicted_mask_path": resolved_predicted_mask_path,
        "ground_truth_mask_path": resolved_ground_truth_mask_path,
        "estimated_audio_path": resolved_estimated_audio_path,
        "reference_audio_path": resolved_reference_audio_path,
        "audio_metadata_path": resolved_audio_metadata_path,
        "mean_iou": None,
        "si_sdr": None,
        "si_sdr_evaluated": False,
        "si_sdr_skip_reason": None,
    }

    audio_metadata = None
    if resolved_audio_metadata_path:
        with open(resolved_audio_metadata_path, "r", encoding="utf-8") as handle:
            audio_metadata = json.load(handle)
        summary["prompt_mode"] = audio_metadata.get("prompt_mode")
        summary["backend"] = audio_metadata.get("backend")
        summary["predict_spans"] = audio_metadata.get("predict_spans")
        summary["reranking_candidates"] = audio_metadata.get("reranking_candidates")
        summary["audio_runtime_seconds"] = audio_metadata.get("runtime_seconds")
    else:
        summary["prompt_mode"] = None
        summary["backend"] = None
        summary["predict_spans"] = None
        summary["reranking_candidates"] = None
        summary["audio_runtime_seconds"] = None

    if resolved_predicted_mask_path and resolved_ground_truth_mask_path:
        emit_progress(progress_callback, stage="evaluation", stage_progress=0.25, status="running", clip_id=clip_id, message="Computing frame IoU.")
        pred_masks = load_mask_data(resolved_predicted_mask_path)
        gt_masks = load_mask_data(resolved_ground_truth_mask_path)
        iou_rows = compute_frame_ious(pred_masks, gt_masks)
        write_iou_csv(iou_rows, str(iou_csv_path))
        summary["iou_csv_path"] = str(iou_csv_path)
        summary["mean_iou"] = float(np.mean([row["iou"] for row in iou_rows])) if iou_rows else None
    else:
        summary["iou_csv_path"] = None

    if audio_metadata and audio_metadata.get("backend") == "placeholder_passthrough":
        summary["si_sdr_skip_reason"] = "placeholder_audio_is_not_real_separation"
    elif resolved_estimated_audio_path and resolved_reference_audio_path:
        emit_progress(progress_callback, stage="evaluation", stage_progress=0.7, status="running", clip_id=clip_id, message="Computing SI-SDR.")
        est_rate, estimate = load_audio(resolved_estimated_audio_path)
        ref_rate, reference = load_audio(resolved_reference_audio_path)
        if ref_rate != est_rate:
            reference = resample_audio(reference, ref_rate, est_rate)
            ref_rate = est_rate
        summary["si_sdr"] = compute_si_sdr(reference, estimate)
        summary["si_sdr_evaluated"] = True
    elif resolved_estimated_audio_path and not resolved_reference_audio_path:
        summary["si_sdr_skip_reason"] = "missing_reference_audio"
    elif resolved_reference_audio_path and not resolved_estimated_audio_path:
        summary["si_sdr_skip_reason"] = "missing_estimated_audio"
    else:
        summary["si_sdr_skip_reason"] = "missing_audio_inputs"

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_path"] = str(summary_path)
    emit_progress(progress_callback, stage="evaluation", stage_progress=1.0, status="completed", clip_id=clip_id, message="Evaluation complete.", outputs={"summary_path": str(summary_path), "iou_csv_path": summary.get("iou_csv_path")}, stats={"mean_iou": summary.get("mean_iou"), "si_sdr": summary.get("si_sdr"), "si_sdr_evaluated": summary.get("si_sdr_evaluated")})
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate mask IoU and audio SI-SDR for experiment runs.")
    parser.add_argument("--output-dir", default=None, help="Directory to store evaluation artifacts. Defaults to output/eval/<clip_id>.")
    parser.add_argument("--clip-id", required=True, help="Clip identifier.")
    parser.add_argument("--model-size", default=None, help="Optional model size label.")
    parser.add_argument("--model-id", default=None, help="Optional explicit model id.")
    parser.add_argument("--predicted-mask-path", default=None, help="Predicted masks (.npy/.npz or PNG dir).")
    parser.add_argument("--ground-truth-mask-path", default=None, help="Ground-truth masks (.npy/.npz or PNG dir).")
    parser.add_argument("--estimated-audio-path", default=None, help="Estimated target audio WAV.")
    parser.add_argument("--reference-audio-path", default=None, help="Clean reference audio WAV.")
    parser.add_argument("--audio-metadata-path", default=None, help="Audio run metadata JSON from audio_pipeline.py.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    result = evaluate_run(
        output_dir=args.output_dir,
        clip_id=args.clip_id,
        model_size=args.model_size,
        model_id=args.model_id,
        predicted_mask_path=args.predicted_mask_path,
        ground_truth_mask_path=args.ground_truth_mask_path,
        estimated_audio_path=args.estimated_audio_path,
        reference_audio_path=args.reference_audio_path,
        audio_metadata_path=args.audio_metadata_path,
    )
    print(json.dumps(result, indent=2))
