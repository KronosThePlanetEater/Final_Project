from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from scipy.io import wavfile

from audio_pipeline import (
    create_audio_aligned_segment,
    mux_audio_with_video,
    probe_video,
    run_audio_pipeline,
)
from evaluation import evaluate_run
from tracker import (
    PreviewHandler,
    get_box_from_mask,
    save_mask_outputs,
    selection_from_dict,
    selection_to_dict,
    track_object,
    transcode_video_for_browser,
    write_metrics_csv,
    write_summary_json,
)


BOOL_FIELDS = {"is_keyframe", "flow_valid", "mask_empty", "tracking_failed", "object_missing"}
INT_FIELDS = {"frame_idx", "mask_area_px", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_width", "bbox_height"}
FLOAT_FIELDS = {"timestamp_sec", "motion_magnitude", "mask_coverage_ratio", "inference_time_ms"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def validate_segment_settings(
    segment_processing: bool,
    segment_length_seconds: float,
    segment_overlap_seconds: float,
) -> None:
    if not segment_processing:
        return
    if segment_length_seconds <= 0:
        raise ValueError("segment_length_seconds must be greater than 0.")
    if segment_overlap_seconds < 0:
        raise ValueError("segment_overlap_seconds must be >= 0.")
    if segment_overlap_seconds >= segment_length_seconds:
        raise ValueError("segment_overlap_seconds must be smaller than segment_length_seconds.")


def plan_segment_windows(
    total_video_frames: int,
    fps: float,
    start_time_seconds: float,
    max_frames: Optional[int],
    segment_length_seconds: float,
    segment_overlap_seconds: float,
) -> Dict[str, Any]:
    if fps <= 0:
        raise RuntimeError("Prepared clip FPS must be positive for segmented processing.")
    start_frame = max(0, int(float(start_time_seconds or 0.0) * fps))
    if start_frame >= total_video_frames:
        raise RuntimeError(
            f"Start time {start_time_seconds}s maps to frame {start_frame}, beyond the prepared clip length of {total_video_frames} frames."
        )

    remaining_frames = total_video_frames - start_frame
    selected_total_frames = remaining_frames if max_frames in (None, "") else min(remaining_frames, int(max_frames))
    if selected_total_frames <= 0:
        raise RuntimeError("Segmented processing resolved to zero frames. Increase max_frames or choose an earlier start time.")

    segment_length_frames = max(1, int(round(segment_length_seconds * fps)))
    segment_overlap_frames = max(0, int(round(segment_overlap_seconds * fps)))
    if segment_overlap_frames >= segment_length_frames:
        raise ValueError("Segment overlap must be smaller than segment length after frame conversion.")

    windows: List[Dict[str, Any]] = []
    relative_start = 0
    while relative_start < selected_total_frames:
        relative_end = min(selected_total_frames, relative_start + segment_length_frames)
        absolute_start = start_frame + relative_start
        absolute_end = start_frame + relative_end
        overlap_from_previous = 0
        if windows:
            overlap_from_previous = max(0, windows[-1]["end_frame"] - absolute_start)
        window = {
            "segment_index": len(windows),
            "segment_id": f"segment_{len(windows):03d}",
            "relative_start_frame": relative_start,
            "relative_end_frame": relative_end,
            "start_frame": absolute_start,
            "end_frame": absolute_end,
            "frame_count": absolute_end - absolute_start,
            "start_time_seconds": absolute_start / fps,
            "end_time_seconds": absolute_end / fps,
            "overlap_from_previous": overlap_from_previous,
        }
        windows.append(window)
        if relative_end >= selected_total_frames:
            break
        next_relative_start = relative_end - segment_overlap_frames
        if next_relative_start <= relative_start:
            next_relative_start = relative_start + 1
        relative_start = next_relative_start

    return {
        "fps": float(fps),
        "selected_start_frame": start_frame,
        "selected_total_frames": selected_total_frames,
        "selected_start_time_seconds": start_frame / fps,
        "selected_end_frame": start_frame + selected_total_frames,
        "segment_length_frames": segment_length_frames,
        "segment_overlap_frames": segment_overlap_frames,
        "segments": windows,
    }


def _coerce_metric_value(field: str, value: str) -> Any:
    if field in BOOL_FIELDS:
        return str(value).strip().lower() in {"1", "true", "yes"}
    if field in INT_FIELDS:
        return int(float(value)) if value not in (None, "") else 0
    if field in FLOAT_FIELDS:
        return float(value) if value not in (None, "") else 0.0
    return value


def _load_segment_metrics_rows(metrics_csv_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(metrics_csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _coerce_metric_value(key, value) for key, value in row.items()})
    return rows


def _load_mask_bundle(mask_stack_path: str) -> Dict[str, Any]:
    payload = np.load(mask_stack_path, allow_pickle=True)
    masks = payload["masks"].astype(bool)
    frame_names_raw = payload["frame_names"] if "frame_names" in payload else None
    if frame_names_raw is None:
        frame_names = [f"{idx:05d}.jpg" for idx in range(masks.shape[0])]
    else:
        frame_names = [str(name) for name in frame_names_raw.tolist()]
    frame_sources = [int(Path(name).stem) for name in frame_names]
    return {
        "masks": masks,
        "frame_names": frame_names,
        "frame_sources": frame_sources,
    }


def derive_segment_initial_selection(
    previous_segment: Dict[str, Any],
    next_window: Dict[str, Any],
    padding: int = 10,
):
    bundle = _load_mask_bundle(previous_segment["tracking_summary"]["mask_stack_path"])
    masks = bundle["masks"]
    frame_sources = bundle["frame_sources"]
    frame_lookup = {frame_source: idx for idx, frame_source in enumerate(frame_sources)}

    overlap_start = int(next_window["start_frame"])
    overlap_end = min(
        int(next_window["start_frame"] + next_window["overlap_from_previous"]),
        int(previous_segment["window"]["end_frame"]),
    )

    candidate_frames: List[int] = [overlap_start]
    for frame_source in range(overlap_start, overlap_end):
        if frame_source != overlap_start:
            candidate_frames.append(frame_source)

    for frame_source in candidate_frames:
        idx = frame_lookup.get(frame_source)
        if idx is None:
            continue
        mask = masks[idx]
        if not mask.any():
            continue
        bbox = get_box_from_mask(mask, padding=padding)
        if bbox is not None:
            return selection_from_dict({"mode": "bbox", "bbox": bbox.astype(float).tolist()})

    for idx in range(len(masks) - 1, -1, -1):
        mask = masks[idx]
        if not mask.any():
            continue
        bbox = get_box_from_mask(mask, padding=padding)
        if bbox is not None:
            return selection_from_dict({"mode": "bbox", "bbox": bbox.astype(float).tolist()})

    raise RuntimeError(
        f"Could not derive a carry-over bounding box for {next_window['segment_id']}. No valid non-empty overlap mask was found."
    )


def _map_stage_event(segment_index: int, segment_count: int, event: Dict[str, Any]) -> Dict[str, Any]:
    mapped = dict(event or {})
    local_progress = float(mapped.get("stage_progress", 0.0) or 0.0)
    mapped["stage_progress"] = min(1.0, max(0.0, (segment_index + local_progress) / max(segment_count, 1)))
    if mapped.get("message"):
        mapped["message"] = f"Segment {segment_index + 1}/{segment_count}: {mapped['message']}"
    stats = dict(mapped.get("stats") or {})
    stats.setdefault("segment_index", segment_index + 1)
    stats.setdefault("segment_count", segment_count)
    mapped["stats"] = stats
    if mapped.get("status") == "completed" and segment_index < segment_count - 1:
        mapped["status"] = "running"
    return mapped


def build_segment_stage_callback(base_callback, segment_index: int, segment_count: int):
    if base_callback is None:
        return None

    def callback(event: Dict[str, Any]) -> None:
        base_callback(_map_stage_event(segment_index, segment_count, event))

    return callback


def _write_masked_preview_video(
    source_video_path: str,
    output_video_path: str,
    mask_stack: np.ndarray,
    background_color: Optional[tuple[int, int, int]],
    ffmpeg_bin: str,
) -> str:
    source_meta = probe_video(source_video_path)
    cap = cv2.VideoCapture(source_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source video for stitched tracking preview: {source_video_path}")

    output_path = Path(output_video_path)
    ensure_dir(output_path.parent)
    raw_path = output_path.with_name(f"{output_path.stem}.raw.mp4")
    writer = cv2.VideoWriter(
        str(raw_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(source_meta["fps"] or 30.0),
        (int(source_meta["width"]), int(source_meta["height"])),
    )

    written = 0
    try:
        for frame_index in range(mask_stack.shape[0]):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = PreviewHandler.apply_mask_visuals(frame, mask_stack[frame_index], background_color)
            writer.write(frame)
            written += 1
    finally:
        writer.release()
        cap.release()

    if written != mask_stack.shape[0]:
        raise RuntimeError(
            f"Stitched tracking preview wrote {written} frames, but the final mask stack has {mask_stack.shape[0]} frames."
        )

    transcode_video_for_browser(str(raw_path), str(output_path), ffmpeg_bin=ffmpeg_bin)
    try:
        raw_path.unlink(missing_ok=True)
    except Exception:
        pass
    return str(output_path.resolve())


def _to_float32_wave(audio: np.ndarray) -> np.ndarray:
    if audio.dtype.kind in {"i", "u"}:
        max_value = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        return audio.astype(np.float32) / float(max_value)
    return audio.astype(np.float32)


def _load_wave(path: str) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return int(sample_rate), _to_float32_wave(audio)


def _stitch_audio_waveforms(audio_paths: List[str], overlap_samples: List[int]) -> tuple[int, np.ndarray]:
    if not audio_paths:
        raise RuntimeError("No audio paths were provided for stitching.")

    sample_rate, stitched = _load_wave(audio_paths[0])
    for boundary_index, audio_path in enumerate(audio_paths[1:]):
        next_sample_rate, next_audio = _load_wave(audio_path)
        if next_sample_rate != sample_rate:
            raise RuntimeError("All segment audio outputs must use the same sample rate before stitching.")

        overlap = int(overlap_samples[boundary_index]) if boundary_index < len(overlap_samples) else 0
        overlap = max(0, min(overlap, stitched.shape[0], next_audio.shape[0]))
        if overlap > 0:
            if overlap == 1:
                fade_out = np.array([0.5], dtype=np.float32)
                fade_in = np.array([0.5], dtype=np.float32)
            else:
                fade_out = np.linspace(1.0, 0.0, overlap, endpoint=True, dtype=np.float32)
                fade_in = np.linspace(0.0, 1.0, overlap, endpoint=True, dtype=np.float32)
            blended = stitched[-overlap:] * fade_out + next_audio[:overlap] * fade_in
            stitched = np.concatenate([stitched[:-overlap], blended, next_audio[overlap:]], axis=0)
        else:
            stitched = np.concatenate([stitched, next_audio], axis=0)
    return sample_rate, stitched


def _persist_waveform(path: Path, sample_rate: int, audio: np.ndarray) -> str:
    ensure_dir(path.parent)
    clipped = np.clip(audio.astype(np.float32), -1.0, 1.0)
    wavfile.write(str(path), sample_rate, (clipped * 32767.0).astype(np.int16))
    return str(path.resolve())


def _stitch_tracking_outputs(
    run_dir: Path,
    clip_id: str,
    selected_span_video_path: str,
    selection_mode: str,
    initial_selection,
    tracker_variant: Dict[str, Any],
    segment_length_seconds: float,
    segment_overlap_seconds: float,
    segment_results: List[Dict[str, Any]],
    save_mask_pngs: bool,
    ffmpeg_bin: str,
) -> Dict[str, Any]:
    if not segment_results:
        raise RuntimeError("No segment tracking results were produced.")

    final_masks: List[np.ndarray] = []
    final_frame_names: List[str] = []
    final_metrics_rows: List[Dict[str, Any]] = []
    total_sam_frames = 0
    total_flow_frames = 0
    total_pure_time = 0.0

    for segment_result in segment_results:
        tracking_summary = segment_result["tracking_summary"]
        window = segment_result["window"]
        keep_from = 0 if window["segment_index"] == 0 else min(window["overlap_from_previous"], int(window["frame_count"]))

        mask_bundle = _load_mask_bundle(tracking_summary["mask_stack_path"])
        masks = mask_bundle["masks"]
        frame_names = mask_bundle["frame_names"]
        if keep_from > masks.shape[0]:
            raise RuntimeError(
                f"Overlap for {window['segment_id']} exceeds the available segment masks ({keep_from} > {masks.shape[0]})."
            )
        final_masks.extend(masks[keep_from:])
        final_frame_names.extend(frame_names[keep_from:])

        rows = _load_segment_metrics_rows(tracking_summary["metrics_csv_path"])
        for row in rows[keep_from:]:
            row["frame_idx"] = len(final_metrics_rows)
            final_metrics_rows.append(row)

        total_sam_frames += int(tracking_summary.get("sam_frames", 0) or 0)
        total_flow_frames += int(tracking_summary.get("flow_frames", 0) or 0)
        total_pure_time += float(tracking_summary.get("pure_time", 0.0) or 0.0)

    mask_array = np.stack(final_masks, axis=0).astype(bool)
    selected_meta = probe_video(selected_span_video_path)
    expected_frames = int(selected_meta.get("frame_count") or 0)
    if expected_frames and mask_array.shape[0] != expected_frames:
        raise RuntimeError(
            f"Final stitched mask count ({mask_array.shape[0]}) does not match the selected span video frame count ({expected_frames})."
        )

    masks_dir = run_dir / "masks"
    metrics_dir = run_dir / "metrics"
    video_dir = run_dir / "video"
    ensure_dir(masks_dir)
    ensure_dir(metrics_dir)
    ensure_dir(video_dir)

    mask_stack_path = masks_dir / "masks.npz"
    mask_png_dir = masks_dir / "png"
    metrics_csv_path = metrics_dir / "frame_metrics.csv"
    summary_json_path = metrics_dir / "tracking_summary.json"
    output_video_path = video_dir / "tracked.mp4"

    save_mask_outputs(
        list(mask_array),
        final_frame_names,
        str(mask_stack_path),
        save_mask_pngs=save_mask_pngs,
        mask_png_dir=str(mask_png_dir),
    )
    write_metrics_csv(final_metrics_rows, str(metrics_csv_path))
    _write_masked_preview_video(
        source_video_path=selected_span_video_path,
        output_video_path=str(output_video_path),
        mask_stack=mask_array,
        background_color=None,
        ffmpeg_bin=ffmpeg_bin,
    )

    empty_mask_count = sum(1 for row in final_metrics_rows if row.get("mask_empty"))
    failure_count = sum(1 for row in final_metrics_rows if row.get("tracking_failed"))
    object_missing_count = sum(1 for row in final_metrics_rows if row.get("object_missing"))
    mean_motion = float(np.mean([row["motion_magnitude"] for row in final_metrics_rows])) if final_metrics_rows else 0.0
    mean_mask_coverage = float(np.mean([row["mask_coverage_ratio"] for row in final_metrics_rows])) if final_metrics_rows else 0.0
    pure_fps = (mask_array.shape[0] / total_pure_time) if total_pure_time > 0 else 0.0

    summary = {
        "clip_id": clip_id,
        "video_path": str(Path(selected_span_video_path).resolve()),
        "output_video_path": str(output_video_path.resolve()),
        "frames_dir": str((run_dir / "segments").resolve()),
        "artifacts_dir": str(run_dir.resolve()),
        "metrics_csv_path": str(metrics_csv_path.resolve()),
        "summary_json_path": str(summary_json_path.resolve()),
        "mask_stack_path": str(mask_stack_path.resolve()),
        "mask_png_dir": str(mask_png_dir.resolve()) if save_mask_pngs else None,
        "fps": float(selected_meta.get("fps") or 0.0),
        "width": int(selected_meta.get("width") or 0),
        "height": int(selected_meta.get("height") or 0),
        "total_frames": int(mask_array.shape[0]),
        "sam_frames": total_sam_frames,
        "flow_frames": total_flow_frames,
        "pure_time": total_pure_time,
        "pure_fps": pure_fps,
        "empty_mask_count": empty_mask_count,
        "failure_count": failure_count,
        "object_missing_count": object_missing_count,
        "mean_motion": mean_motion,
        "mean_mask_coverage": mean_mask_coverage,
        "selection_mode": selection_mode,
        "initial_selection": selection_to_dict(selection_mode, initial_selection),
        "sam_interval": int(tracker_variant.get("sam_interval", 1) or 1),
        "dynamic_interval": list(tracker_variant.get("dynamic_interval")) if tracker_variant.get("dynamic_interval") else None,
        "start_time_seconds": float(segment_results[0]["window"]["start_time_seconds"]),
        "background_color_bgr": None,
        "segmented": True,
        "segment_count": len(segment_results),
        "segment_length_seconds": float(segment_length_seconds),
        "segment_overlap_seconds": float(segment_overlap_seconds),
        "segments": [segment_result["manifest_entry"] for segment_result in segment_results],
    }
    write_summary_json(summary, str(summary_json_path))
    return summary


def _stitch_audio_outputs(
    run_dir: Path,
    clip_id: str,
    selected_span_video_path: str,
    final_mask_path: str,
    segment_length_seconds: float,
    segment_overlap_seconds: float,
    segment_results: List[Dict[str, Any]],
    ffmpeg_bin: str,
) -> Dict[str, Any]:
    if not segment_results:
        raise RuntimeError("No segment audio results were produced.")

    audio_dir = run_dir / "audio"
    ensure_dir(audio_dir)

    target_audio_paths = [segment_result["audio_summary"]["target_audio_path"] for segment_result in segment_results]
    residual_audio_paths = [segment_result["audio_summary"].get("residual_audio_path") for segment_result in segment_results]
    overlap_samples: List[int] = []
    reference_sample_rate = int(segment_results[0]["audio_summary"].get("sample_rate") or 16000)
    fps = float(segment_results[0]["tracking_summary"].get("fps") or 0.0)
    for segment_result in segment_results[1:]:
        overlap_frames = int(segment_result["window"].get("overlap_from_previous", 0) or 0)
        overlap_samples.append(int(round((overlap_frames / fps) * reference_sample_rate)) if fps > 0 else 0)

    sample_rate, stitched_target = _stitch_audio_waveforms(target_audio_paths, overlap_samples)
    target_audio_path = _persist_waveform(audio_dir / "target.wav", sample_rate, stitched_target)

    residual_audio_path: Optional[str] = None
    if all(path for path in residual_audio_paths):
        residual_sample_rate, stitched_residual = _stitch_audio_waveforms([str(path) for path in residual_audio_paths], overlap_samples)
        if residual_sample_rate != sample_rate:
            raise RuntimeError("Residual audio sample rate did not match target audio sample rate during stitching.")
        residual_audio_path = _persist_waveform(audio_dir / "residual.wav", residual_sample_rate, stitched_residual)

    target_video_path = str((audio_dir / "target_video.mp4").resolve())
    residual_video_path = str((audio_dir / "residual_video.mp4").resolve()) if residual_audio_path else None
    mux_audio_with_video(selected_span_video_path, target_audio_path, target_video_path, ffmpeg_bin=ffmpeg_bin)
    if residual_audio_path:
        mux_audio_with_video(selected_span_video_path, residual_audio_path, residual_video_path, ffmpeg_bin=ffmpeg_bin)

    first_audio_summary = segment_results[0]["audio_summary"]
    selected_meta = probe_video(selected_span_video_path)
    metadata_path = audio_dir / "audio_run_metadata.json"
    payload = {
        "clip_id": clip_id,
        "backend": first_audio_summary.get("backend"),
        "prompt_mode": first_audio_summary.get("prompt_mode"),
        "model_size": first_audio_summary.get("model_size"),
        "model_id": first_audio_summary.get("model_id"),
        "device": first_audio_summary.get("device"),
        "requested_audio_precision": first_audio_summary.get("requested_audio_precision"),
        "effective_audio_precision": first_audio_summary.get("effective_audio_precision"),
        "predict_spans": bool(first_audio_summary.get("predict_spans", False)),
        "reranking_candidates": int(first_audio_summary.get("reranking_candidates", 1) or 1),
        "video_path": str(Path(selected_span_video_path).resolve()),
        "mask_path": str(Path(final_mask_path).resolve()),
        "packaged_mask_path": None,
        "target_audio_path": target_audio_path,
        "residual_audio_path": residual_audio_path,
        "target_video_path": target_video_path,
        "residual_video_path": residual_video_path,
        "mux_video_outputs": True,
        "runtime_seconds": float(sum(segment_result["audio_summary"].get("runtime_seconds", 0.0) or 0.0 for segment_result in segment_results)),
        "frame_count": int(selected_meta.get("frame_count") or 0),
        "fps": float(selected_meta.get("fps") or 0.0),
        "width": int(selected_meta.get("width") or 0),
        "height": int(selected_meta.get("height") or 0),
        "frame_span": [
            int(segment_results[0]["window"]["start_frame"]),
            int(segment_results[-1]["window"]["end_frame"] - 1),
        ],
        "alignment_adjustments": [],
        "visual_mask_semantics": first_audio_summary.get("visual_mask_semantics", "tracker_target_true_inverted_for_sam_audio"),
        "sample_rate": sample_rate,
        "segmented": True,
        "segment_count": len(segment_results),
        "segment_length_seconds": float(segment_length_seconds),
        "segment_overlap_seconds": float(segment_overlap_seconds),
        "segments": [segment_result["manifest_entry"] for segment_result in segment_results],
    }
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["metadata_path"] = str(metadata_path.resolve())
    return payload


def run_segmented_pipeline(
    prepared_video_path: str,
    canonical_video_path: str,
    run_dir: str,
    clip_id: str,
    checkpoint_path: str,
    config_path: str,
    selection_mode: str,
    initial_selection,
    max_frames: Optional[int],
    start_time: float,
    scale: float,
    tracker_variant: Dict[str, Any],
    save_mask_pngs: bool,
    audio_model_size: str,
    audio_model_id: Optional[str] = None,
    audio_device: Optional[str] = None,
    audio_precision: str = "auto",
    predict_spans: bool = False,
    reranking_candidates: int = 1,
    allow_placeholder_audio: bool = False,
    release_memory_after_run: bool = True,
    reference_audio_path: Optional[str] = None,
    ground_truth_mask_path: Optional[str] = None,
    ffmpeg_bin: str = "ffmpeg",
    segment_length_seconds: float = 15.0,
    segment_overlap_seconds: float = 2.0,
    tracking_progress_callback=None,
    audio_progress_callback=None,
    evaluation_progress_callback=None,
    cleanup_callback=None,
) -> Dict[str, Any]:
    validate_segment_settings(True, float(segment_length_seconds), float(segment_overlap_seconds))

    run_root = Path(run_dir)
    ensure_dir(run_root)
    ensure_dir(run_root / "segments")

    prepared_meta = probe_video(prepared_video_path)
    plan = plan_segment_windows(
        total_video_frames=int(prepared_meta.get("frame_count") or 0),
        fps=float(prepared_meta.get("fps") or 0.0),
        start_time_seconds=float(start_time or 0.0),
        max_frames=max_frames,
        segment_length_seconds=float(segment_length_seconds),
        segment_overlap_seconds=float(segment_overlap_seconds),
    )

    selected_span_video_path = create_audio_aligned_segment(
        source_video_path=canonical_video_path,
        output_video_path=str(run_root / "selected_span" / "input_segment.mp4"),
        start_time_seconds=float(plan["selected_start_time_seconds"]),
        frame_count=int(plan["selected_total_frames"]),
        fps=float(plan["fps"]),
        ffmpeg_bin=ffmpeg_bin,
    )

    segment_results: List[Dict[str, Any]] = []
    current_selection = initial_selection
    segment_windows = plan["segments"]

    for segment_index, window in enumerate(segment_windows):
        if segment_index > 0:
            current_selection = derive_segment_initial_selection(segment_results[-1], window)
            current_selection_mode = "bbox"
        else:
            current_selection_mode = selection_mode

        segment_dir = run_root / "segments" / window["segment_id"]
        tracking_dir = segment_dir / "tracking"
        frames_dir = tracking_dir / "frames"
        ensure_dir(segment_dir)

        tracking_summary = track_object(
            video_path=prepared_video_path,
            output_path=str(tracking_dir / "video" / "tracked.mp4"),
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            frames_dir=str(frames_dir),
            extract_frames=True,
            selection_mode=current_selection_mode,
            max_frames=int(window["frame_count"]),
            start_time=float(window["start_time_seconds"]),
            scale=scale,
            initial_selection=current_selection,
            skip_mask_confirmation=True,
            sam_interval=int(tracker_variant.get("sam_interval", 1) or 1),
            dynamic_interval=tuple(tracker_variant["dynamic_interval"]) if tracker_variant.get("dynamic_interval") else None,
            background_color=None,
            show_preview=False,
            artifacts_dir=str(tracking_dir),
            save_mask_pngs=save_mask_pngs,
            ffmpeg_bin=ffmpeg_bin,
            progress_callback=build_segment_stage_callback(tracking_progress_callback, segment_index, len(segment_windows)),
        )

        audio_input_path = create_audio_aligned_segment(
            source_video_path=canonical_video_path,
            output_video_path=str(segment_dir / "audio_input" / "input_segment.mp4"),
            start_time_seconds=float(window["start_time_seconds"]),
            frame_count=int(window["frame_count"]),
            fps=float(plan["fps"]),
            ffmpeg_bin=ffmpeg_bin,
        )
        audio_summary = run_audio_pipeline(
            video_path=audio_input_path,
            mask_path=tracking_summary["mask_stack_path"],
            output_dir=str(segment_dir / "audio"),
            clip_id=f"{clip_id}_{window['segment_id']}",
            model_size=audio_model_size,
            model_id=audio_model_id,
            prompt_mode="visual",
            device=audio_device,
            audio_precision=audio_precision,
            predict_spans=predict_spans,
            reranking_candidates=reranking_candidates,
            allow_placeholder=allow_placeholder_audio,
            mux_video_outputs=True,
            ffmpeg_bin=ffmpeg_bin,
            progress_callback=build_segment_stage_callback(audio_progress_callback, segment_index, len(segment_windows)),
        )

        segment_manifest_entry = {
            "segment_id": window["segment_id"],
            "segment_index": int(segment_index),
            "start_frame": int(window["start_frame"]),
            "end_frame": int(window["end_frame"]),
            "frame_count": int(window["frame_count"]),
            "start_time_seconds": float(window["start_time_seconds"]),
            "end_time_seconds": float(window["end_time_seconds"]),
            "overlap_from_previous": int(window["overlap_from_previous"]),
            "selection_mode": current_selection_mode,
            "tracking_summary_path": tracking_summary["summary_json_path"],
            "audio_metadata_path": audio_summary["metadata_path"],
        }
        segment_results.append(
            {
                "window": window,
                "tracking_summary": tracking_summary,
                "audio_summary": audio_summary,
                "manifest_entry": segment_manifest_entry,
            }
        )

        if cleanup_callback is not None and release_memory_after_run:
            cleanup_callback()

    effective_initial_selection = initial_selection
    if effective_initial_selection is None:
        effective_initial_selection = selection_from_dict(segment_results[0]["tracking_summary"].get("initial_selection"))

    final_tracking_summary = _stitch_tracking_outputs(
        run_dir=run_root,
        clip_id=clip_id,
        selected_span_video_path=selected_span_video_path,
        selection_mode=selection_mode,
        initial_selection=effective_initial_selection,
        tracker_variant=tracker_variant,
        segment_length_seconds=float(segment_length_seconds),
        segment_overlap_seconds=float(segment_overlap_seconds),
        segment_results=segment_results,
        save_mask_pngs=save_mask_pngs,
        ffmpeg_bin=ffmpeg_bin,
    )

    final_audio_summary = _stitch_audio_outputs(
        run_dir=run_root,
        clip_id=clip_id,
        selected_span_video_path=selected_span_video_path,
        final_mask_path=final_tracking_summary["mask_stack_path"],
        segment_length_seconds=float(segment_length_seconds),
        segment_overlap_seconds=float(segment_overlap_seconds),
        segment_results=segment_results,
        ffmpeg_bin=ffmpeg_bin,
    )

    final_evaluation_summary = evaluate_run(
        output_dir=str(run_root / "eval"),
        clip_id=clip_id,
        model_size=audio_model_size,
        model_id=final_audio_summary.get("model_id"),
        predicted_mask_path=final_tracking_summary["mask_stack_path"],
        ground_truth_mask_path=ground_truth_mask_path,
        estimated_audio_path=final_audio_summary["target_audio_path"],
        reference_audio_path=reference_audio_path,
        audio_metadata_path=final_audio_summary["metadata_path"],
        progress_callback=evaluation_progress_callback,
    )

    if cleanup_callback is not None and release_memory_after_run:
        cleanup_callback()

    return {
        "tracking_summary": final_tracking_summary,
        "audio_summary": final_audio_summary,
        "evaluation_summary": final_evaluation_summary,
        "segment_manifest": [segment_result["manifest_entry"] for segment_result in segment_results],
        "selected_span_video_path": str(Path(selected_span_video_path).resolve()),
    }
