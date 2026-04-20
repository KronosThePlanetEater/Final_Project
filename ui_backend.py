from __future__ import annotations

import gc
import io
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from analyze_results import analyze_results
from audio_pipeline import create_audio_aligned_segment, run_audio_pipeline
from dataset_prep import prepare_clip
from evaluation import evaluate_run
from path_layout import INPUT_ROOT, OUTPUT_ROOT, ensure_standard_directories, resolve_existing_path
from posthoc_analysis import create_posthoc_analysis_set, discover_mergeable_runs, get_recent_analysis_set_ids, group_runs_by_clip, load_analysis_set
from progress_utils import iso_utc_now
from run_experiments import materialize_tracking_outputs
from tracker import PreviewHandler, SAM2ImagePredictor, build_sam2, get_box_from_mask, mask_to_bool, normalize_sam2_config_path, selection_from_dict, setup_device, track_object

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a"}
MASK_SUFFIXES = {".npy", ".npz"}
JOB_ROOT = OUTPUT_ROOT / "ui_runs"
VALID_AUDIO_MODEL_SIZES = ["small-tv", "small", "base-tv", "base", "large-tv", "large"]
TRACKER_PRESETS = {
    "dynamic_optical_flow": "Dynamic SAM + Optical Flow",
    "full_sam": "Full SAM (every frame)",
    "static_interval_5": "Static SAM + Optical Flow (every 5 frames)",
    "static_interval_10": "Static SAM + Optical Flow (every 10 frames)",
}
TRACKER_JOB_TAGS = {
    "dynamic_optical_flow": "dynam",
    "full_sam": "fullS",
    "static_interval_5": "stat5",
    "static_interval_10": "stat10",
}
STAGE_WEIGHTS = {
    "dataset_prep": 0.1,
    "tracking": 0.55,
    "audio": 0.2,
    "evaluation": 0.1,
    "analytics": 0.05,
}
ACTIVE_THREADS: Dict[str, threading.Thread] = {}
STATE_LOCK = threading.Lock()


def cleanup_runtime_memory() -> Dict[str, Any]:
    stats: Dict[str, Any] = {"cleanup_mode": "aggressive", "gc_collected": 0, "cuda_available": False, "cuda_cache_cleared": False}
    stats["gc_collected"] = int(gc.collect())
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        stats["cuda_available"] = cuda_available
        if cuda_available:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            stats["cuda_cache_cleared"] = True
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception as exc:
        stats["cleanup_error"] = str(exc)
    return stats


def ensure_ui_dirs() -> None:
    ensure_standard_directories()
    JOB_ROOT.mkdir(parents=True, exist_ok=True)


def list_input_files(suffixes: set[str]) -> List[str]:
    ensure_ui_dirs()
    return sorted(path.name for path in INPUT_ROOT.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def list_input_videos() -> List[str]:
    return list_input_files(VIDEO_SUFFIXES)


def list_reference_audio_files() -> List[str]:
    return list_input_files(AUDIO_SUFFIXES)


def list_ground_truth_mask_files() -> List[str]:
    files = list_input_files(MASK_SUFFIXES)
    files.extend(sorted(path.name for path in INPUT_ROOT.iterdir() if path.is_dir()))
    return sorted(set(files))


def job_dir(job_id: str) -> Path:
    return JOB_ROOT / job_id


def job_state_path(job_id: str) -> Path:
    return job_dir(job_id) / "job_state.json"


def job_output_root(job_id: str, allow_placeholder: bool = False) -> Path:
    if allow_placeholder:
        return OUTPUT_ROOT / "placeholder_test" / "ui_runs" / job_id
    return job_dir(job_id)


def slugify_job_token(value: str) -> str:
    slug_chars: List[str] = []
    previous_was_dash = False
    for char in str(value).lower():
        if char.isalnum():
            slug_chars.append(char)
            previous_was_dash = False
        elif not previous_was_dash:
            slug_chars.append("-")
            previous_was_dash = True
    return "".join(slug_chars).strip("-") or "tracker"


def build_job_tracker_tag(config: Dict[str, Any]) -> str:
    if config.get("run_mode") == "comparison":
        selected = config.get("tracker_variants") or ["dynamic_optical_flow", "full_sam"]
    else:
        selected = [config.get("single_tracker_preset", "dynamic_optical_flow")]

    tags: List[str] = []
    for key in selected:
        tag = TRACKER_JOB_TAGS.get(str(key), slugify_job_token(str(key)))
        if tag not in tags:
            tags.append(tag)

    if not tags:
        return "tracker"
    if len(tags) == 1:
        return tags[0]
    return "-vs-".join(tags)


def generate_job_id(config: Dict[str, Any], now: Optional[datetime] = None) -> str:
    local_now = now.astimezone() if now is not None else datetime.now().astimezone()
    date_prefix = local_now.strftime("%m-%d-%y")
    tracker_tag = build_job_tracker_tag(config)
    next_index = 1
    prefix = f"{date_prefix}-{tracker_tag}-"
    if JOB_ROOT.exists():
        for path in JOB_ROOT.iterdir():
            if not path.is_dir() or not path.name.startswith(prefix):
                continue
            suffix = path.name[len(prefix):]
            if suffix.isdigit():
                next_index = max(next_index, int(suffix) + 1)
    return f"{prefix}{next_index}"


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Optional[Exception] = None
    for attempt in range(8):
        temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            os.replace(temp_path, path)
            return
        except PermissionError as exc:
            last_error = exc
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
        except Exception:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            raise
    if last_error is not None:
        raise last_error


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def deep_update(target: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def compute_overall_progress(state: Dict[str, Any]) -> float:
    weighted = 0.0
    total_weight = sum(STAGE_WEIGHTS.values()) or 1.0
    for stage_name, weight in STAGE_WEIGHTS.items():
        stage_state = state.get("stages", {}).get(stage_name, {})
        progress = float(stage_state.get("progress", 0.0) or 0.0)
        if stage_state.get("status") == "completed":
            progress = 1.0
        weighted += weight * progress
    return min(1.0, max(0.0, weighted / total_weight))


def build_tracker_variants(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    dynamic_interval = [int(config.get("dynamic_min", 2)), int(config.get("dynamic_max", 20))]
    if config.get("run_mode") != "comparison":
        selected = config.get("single_tracker_preset", "dynamic_optical_flow")
        if selected == "dynamic_optical_flow":
            return [{
                "key": f"dynamic_{dynamic_interval[0]}_{dynamic_interval[1]}",
                "preset_key": selected,
                "label": f"Dynamic SAM + Optical Flow ({dynamic_interval[0]}-{dynamic_interval[1]})",
                "sam_interval": 1,
                "dynamic_interval": dynamic_interval,
            }]
        if selected == "full_sam":
            return [{
                "key": "full_sam",
                "preset_key": selected,
                "label": TRACKER_PRESETS[selected],
                "sam_interval": 1,
                "dynamic_interval": None,
            }]
        if selected == "static_interval_10":
            return [{
                "key": "static_10",
                "preset_key": selected,
                "label": TRACKER_PRESETS[selected],
                "sam_interval": 10,
                "dynamic_interval": None,
            }]
        return [{
            "key": "static_5",
            "preset_key": "static_interval_5",
            "label": TRACKER_PRESETS["static_interval_5"],
            "sam_interval": 5,
            "dynamic_interval": None,
        }]

    selected = config.get("tracker_variants") or ["dynamic_optical_flow", "full_sam"]
    variants: List[Dict[str, Any]] = []
    for key in selected:
        if key == "dynamic_optical_flow":
            variants.append({
                "key": f"dynamic_{dynamic_interval[0]}_{dynamic_interval[1]}",
                "preset_key": key,
                "label": f"Dynamic SAM + Optical Flow ({dynamic_interval[0]}-{dynamic_interval[1]})",
                "sam_interval": 1,
                "dynamic_interval": dynamic_interval,
            })
        elif key == "full_sam":
            variants.append({
                "key": "full_sam",
                "preset_key": key,
                "label": TRACKER_PRESETS[key],
                "sam_interval": 1,
                "dynamic_interval": None,
            })
        elif key == "static_interval_5":
            variants.append({
                "key": "static_5",
                "preset_key": key,
                "label": TRACKER_PRESETS[key],
                "sam_interval": 5,
                "dynamic_interval": None,
            })
        elif key == "static_interval_10":
            variants.append({
                "key": "static_10",
                "preset_key": key,
                "label": TRACKER_PRESETS[key],
                "sam_interval": 10,
                "dynamic_interval": None,
            })
    return variants


def build_audio_model_list(config: Dict[str, Any]) -> List[str]:
    if config.get("run_mode") == "comparison":
        models = config.get("audio_model_sizes") or [config.get("audio_model_size", "small-tv")]
        return [model for model in models if model in VALID_AUDIO_MODEL_SIZES]
    model = config.get("audio_model_size", "small-tv")
    return [model]


def build_initial_job_state(job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    clip_id = Path(config["video_name"]).stem
    output_root = job_output_root(job_id, bool(config.get("allow_placeholder_audio", False)))
    tracker_variants = build_tracker_variants(config)
    audio_models = build_audio_model_list(config)
    planned_runs = max(1, len(tracker_variants) * len(audio_models))
    return {
        "job_id": job_id,
        "clip_id": clip_id,
        "status": "queued",
        "current_stage": None,
        "stage_progress": 0.0,
        "overall_progress": 0.0,
        "started_at": iso_utc_now(),
        "finished_at": None,
        "updated_at": iso_utc_now(),
        "error_message": None,
        "config": deepcopy(config),
        "job_dir": str(job_dir(job_id).resolve()),
        "run_dir": str((output_root / "runs").resolve()),
        "analysis_dir": str((output_root / "analysis").resolve()),
        "planned_runs": planned_runs,
        "completed_runs": 0,
        "current_run": None,
        "stages": {
            stage_name: {"status": "pending", "progress": 0.0, "message": None, "stats": {}, "outputs": {}}
            for stage_name in STAGE_WEIGHTS
        },
        "results": {"run_manifest": []},
    }


def load_job_state(job_id: str) -> Optional[Dict[str, Any]]:
    path = job_state_path(job_id)
    if not path.exists():
        return None
    return read_json(path)


def save_job_state(job_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    state["updated_at"] = iso_utc_now()
    state["overall_progress"] = compute_overall_progress(state)
    write_json_atomic(job_state_path(job_id), state)
    return state


def patch_job_state(job_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
    with STATE_LOCK:
        state = load_job_state(job_id)
        if state is None:
            raise FileNotFoundError(f"Job state not found for {job_id}")
        deep_update(state, updates)
        return save_job_state(job_id, state)


def record_stage_event(job_id: str, stage_name: str, event: Dict[str, Any]) -> Dict[str, Any]:
    with STATE_LOCK:
        state = load_job_state(job_id)
        if state is None:
            raise FileNotFoundError(f"Job state not found for {job_id}")

        stage_state = state.setdefault("stages", {}).setdefault(
            stage_name,
            {"status": "pending", "progress": 0.0, "message": None, "stats": {}, "outputs": {}},
        )
        now = event.get("timestamp", iso_utc_now())
        status = event.get("status") or stage_state.get("status") or "running"
        progress = event.get("stage_progress")
        if progress is not None:
            stage_state["progress"] = float(progress)
        if event.get("message") is not None:
            current_run = state.get("current_run") or {}
            prefix = current_run.get("label")
            stage_state["message"] = f"{prefix}: {event['message']}" if prefix else event["message"]
        if event.get("stats"):
            stage_state["stats"] = event["stats"]
        if event.get("outputs"):
            stage_state["outputs"] = event["outputs"]
        stage_state["status"] = status
        stage_state.setdefault("started_at", now)
        if status in {"completed", "failed"}:
            stage_state["finished_at"] = now

        state["status"] = "failed" if status == "failed" else "running"
        state["current_stage"] = stage_name
        state["stage_progress"] = float(stage_state.get("progress", 0.0) or 0.0)
        state["updated_at"] = now
        if event.get("clip_id"):
            state["clip_id"] = event["clip_id"]
        if status == "failed" and event.get("message"):
            state["error_message"] = event["message"]
        return save_job_state(job_id, state)


def stage_callback(job_id: str, stage_name: str):
    def callback(event: Dict[str, Any]) -> None:
        record_stage_event(job_id, stage_name, event)
    return callback


def mark_job_completed(job_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
    with STATE_LOCK:
        state = load_job_state(job_id)
        if state is None:
            raise FileNotFoundError(f"Job state not found for {job_id}")
        state["status"] = "completed"
        state["finished_at"] = iso_utc_now()
        state["current_stage"] = "analytics"
        state["stage_progress"] = 1.0
        state["results"] = results
        for stage_name in STAGE_WEIGHTS:
            stage_state = state.setdefault("stages", {}).setdefault(stage_name, {})
            if stage_state.get("status") != "completed":
                stage_state["status"] = "completed"
                stage_state["progress"] = 1.0
        return save_job_state(job_id, state)


def mark_job_failed(job_id: str, stage_name: str, exc: Exception) -> Dict[str, Any]:
    error_message = f"{type(exc).__name__}: {exc}"
    traceback_text = traceback.format_exc()
    with STATE_LOCK:
        state = load_job_state(job_id)
        if state is None:
            raise FileNotFoundError(f"Job state not found for {job_id}")
        stage_state = state.setdefault("stages", {}).setdefault(stage_name, {})
        stage_state["status"] = "failed"
        stage_state["message"] = error_message
        stage_state["traceback"] = traceback_text
        stage_state["finished_at"] = iso_utc_now()
        state["status"] = "failed"
        state["current_stage"] = stage_name
        state["error_message"] = error_message
        state["traceback"] = traceback_text
        state["finished_at"] = iso_utc_now()
        return save_job_state(job_id, state)


def get_recent_job_ids(limit: int = 10) -> List[str]:
    ensure_ui_dirs()
    jobs = []
    for path in JOB_ROOT.glob("*/job_state.json"):
        try:
            state = read_json(path)
        except Exception:
            continue
        jobs.append((state.get("started_at") or "", path.parent.name))
    jobs.sort(reverse=True)
    return [job_id for _, job_id in jobs[:limit]]


def get_latest_job_state() -> Optional[Dict[str, Any]]:
    for job_id in get_recent_job_ids(limit=1):
        return load_job_state(job_id)
    return None


def get_active_job_state() -> Optional[Dict[str, Any]]:
    for job_id in get_recent_job_ids(limit=25):
        state = load_job_state(job_id)
        if state and state.get("status") in {"queued", "running"}:
            return state
    return None


def list_posthoc_runs_by_clip() -> Dict[str, List[Dict[str, Any]]]:
    return group_runs_by_clip(discover_mergeable_runs())


def draw_selection_overlay(frame_rgb: np.ndarray, selection: Dict[str, Any]) -> np.ndarray:
    canvas = frame_rgb.copy()
    if selection.get("mode") == "bbox":
        bbox = selection.get("bbox") or []
        if len(bbox) == 4:
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 165, 0), 2)
    else:
        for point, label in zip(selection.get("points", []), selection.get("labels", []), strict=False):
            x, y = [int(round(value)) for value in point]
            color = (0, 255, 0) if int(label) == 1 else (255, 0, 0)
            cv2.circle(canvas, (x, y), 7, color, -1)
            cv2.circle(canvas, (x, y), 12, color, 2)
    return canvas


def preview_initial_mask(
    video_name: str,
    selection: Dict[str, Any],
    checkpoint_path: str,
    config_path: str,
    start_time: float = 0.0,
    scale: float = 1.0,
) -> Dict[str, Any]:
    frame_info = load_selection_frame(video_name, start_time=start_time, scale=scale)
    frame_rgb = np.asarray(frame_info["image"], dtype=np.uint8)
    resolved_checkpoint = resolve_existing_path(checkpoint_path)
    resolved_config = normalize_sam2_config_path(config_path)
    (points, labels), bbox = selection_from_dict(selection)

    model = None
    predictor = None
    try:
        device = setup_device()
        model = build_sam2(str(resolved_config), str(resolved_checkpoint), device)
        predictor = SAM2ImagePredictor(model)
        predictor.set_image(frame_rgb)
        if bbox is None:
            masks, scores, _ = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=False,
            )
        else:
            masks, scores, _ = predictor.predict(box=bbox, multimask_output=False)

        mask = mask_to_bool(masks[0], frame_rgb.shape[:2])
        overlay_rgb = PreviewHandler.apply_mask_visuals(frame_rgb, mask, None)
        overlay_rgb = draw_selection_overlay(overlay_rgb, selection)
        bbox_from_mask = get_box_from_mask(mask, padding=0)
        buffer = io.BytesIO()
        Image.fromarray(overlay_rgb).save(buffer, format="PNG")
        mask_area = int(mask.sum())
        mask_coverage = float(mask_area / mask.size) if mask.size else 0.0
        return {
            "image_bytes": buffer.getvalue(),
            "mask_area_px": mask_area,
            "mask_coverage_ratio": round(mask_coverage, 8),
            "mask_empty": bool(mask_area == 0),
            "score": float(scores[0]) if scores is not None and len(scores) else None,
            "mask_bbox": bbox_from_mask.astype(int).tolist() if bbox_from_mask is not None else None,
            "warning": "Mask covers almost the entire frame. Recheck your selection." if mask_coverage >= 0.9 else None,
            "frame_width": int(frame_rgb.shape[1]),
            "frame_height": int(frame_rgb.shape[0]),
        }
    finally:
        del predictor
        del model
        cleanup_runtime_memory()


def load_selection_frame(video_name: str, start_time: float = 0.0, scale: float = 1.0) -> Dict[str, Any]:
    video_path = resolve_existing_path(video_name, extra_search_dirs=[INPUT_ROOT])
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if start_time > 0 and fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read a frame from {video_path}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if scale != 1.0:
        frame_rgb = cv2.resize(frame_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    image = Image.fromarray(frame_rgb)
    return {
        "image": image,
        "width": image.width,
        "height": image.height,
        "video_path": str(video_path.resolve()),
    }


def parse_selection_objects(objects: List[Dict[str, Any]], selection_mode: str, scale_x: float, scale_y: float) -> Optional[Dict[str, Any]]:
    if selection_mode == "point":
        points: List[List[float]] = []
        labels: List[int] = []
        for obj in objects:
            obj_type = obj.get("type")
            if obj_type not in {"circle", "point"}:
                continue
            radius = float(obj.get("radius", 3.0))
            cx = float(obj.get("left", 0.0)) + radius
            cy = float(obj.get("top", 0.0)) + radius
            points.append([round(cx * scale_x, 2), round(cy * scale_y, 2)])
            labels.append(1)
        if not points:
            return None
        return {"mode": "point", "points": points, "labels": labels}

    rects = [obj for obj in objects if obj.get("type") == "rect"]
    if not rects:
        return None
    rect = rects[-1]
    width = float(rect.get("width", 0.0)) * float(rect.get("scaleX", 1.0))
    height = float(rect.get("height", 0.0)) * float(rect.get("scaleY", 1.0))
    x1 = float(rect.get("left", 0.0)) * scale_x
    y1 = float(rect.get("top", 0.0)) * scale_y
    x2 = (float(rect.get("left", 0.0)) + width) * scale_x
    y2 = (float(rect.get("top", 0.0)) + height) * scale_y
    return {"mode": "bbox", "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]}


def list_analysis_images(analysis_dir: str) -> List[str]:
    path = Path(analysis_dir)
    if not path.exists():
        return []
    return sorted(str(image.resolve()) for image in path.glob("*.png"))


def load_metrics_preview(csv_path: str, max_rows: int = 200) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).head(max_rows)


def load_json_if_exists(path_value: Optional[str]) -> Dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    return read_json(path)


def format_run_label(entry: Dict[str, Any]) -> str:
    return f"{entry.get('tracker_variant_label', entry.get('tracker_variant_key', 'tracker'))} | {entry.get('audio_model_size', 'audio')}"


def summarize_job_outputs(state: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
    results = state.get("results", {})
    manifest = results.get("run_manifest") or []
    if manifest:
        selected = None
        if run_id:
            for entry in manifest:
                if entry.get("run_id") == run_id:
                    selected = entry
                    break
        if selected is None:
            selected = manifest[0]
        tracking_summary = load_json_if_exists(selected.get("tracking_summary_path"))
        audio_summary = load_json_if_exists(selected.get("audio_metadata_path"))
        evaluation_summary = load_json_if_exists(selected.get("evaluation_summary_path"))
        analysis_summary = results.get("analysis_summary") or load_json_if_exists((Path(state.get("analysis_dir", "")) / "analysis_summary.json") if state.get("analysis_dir") else None)
        return {
            "run_manifest": manifest,
            "selected_run": selected,
            "selected_run_id": selected.get("run_id"),
            "tracked_video": tracking_summary.get("output_video_path"),
            "target_video": audio_summary.get("target_video_path"),
            "residual_video": audio_summary.get("residual_video_path"),
            "target_audio": audio_summary.get("target_audio_path"),
            "residual_audio": audio_summary.get("residual_audio_path"),
            "metrics_csv": tracking_summary.get("metrics_csv_path"),
            "tracking_summary": tracking_summary,
            "audio_summary": audio_summary,
            "evaluation_summary": evaluation_summary,
            "analysis_summary": analysis_summary,
            "analysis_images": list_analysis_images(analysis_summary.get("analysis_dir", "")) if analysis_summary else [],
        }

    tracking_summary = results.get("tracking_summary", {}) or {}
    audio_summary = results.get("audio_summary", {}) or {}
    evaluation_summary = results.get("evaluation_summary", {}) or {}
    analysis_summary = results.get("analysis_summary", {}) or {}
    return {
        "run_manifest": [],
        "selected_run": None,
        "selected_run_id": None,
        "tracked_video": tracking_summary.get("output_video_path"),
        "target_video": audio_summary.get("target_video_path"),
        "residual_video": audio_summary.get("residual_video_path"),
        "target_audio": audio_summary.get("target_audio_path"),
        "residual_audio": audio_summary.get("residual_audio_path"),
        "metrics_csv": tracking_summary.get("metrics_csv_path"),
        "tracking_summary": tracking_summary,
        "audio_summary": audio_summary,
        "evaluation_summary": evaluation_summary,
        "analysis_summary": analysis_summary,
        "analysis_images": list_analysis_images(analysis_summary.get("analysis_dir", "")) if analysis_summary else [],
    }


def build_run_manifest_entry(
    run_id: str,
    run_dir: Path,
    tracker_variant: Dict[str, Any],
    audio_model_size: str,
    tracking_summary: Dict[str, Any],
    audio_summary: Dict[str, Any],
    evaluation_summary: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "display_label": f"{tracker_variant['label']} | {audio_model_size}",
        "tracker_variant_key": tracker_variant["key"],
        "tracker_variant_label": tracker_variant["label"],
        "audio_model_size": audio_model_size,
        "run_dir": str(run_dir.resolve()),
        "tracking_summary_path": tracking_summary.get("summary_json_path"),
        "audio_metadata_path": audio_summary.get("metadata_path"),
        "evaluation_summary_path": evaluation_summary.get("summary_path"),
    }


def _run_job(job_id: str, config: Dict[str, Any]) -> None:
    clip_id = Path(config["video_name"]).stem
    output_root = job_output_root(job_id, bool(config.get("allow_placeholder_audio", False)))
    tracking_root = output_root / "tracking_cache"
    runs_root = output_root / "runs"
    prepared_root = output_root / "prepared"
    analysis_root = output_root / "analysis"
    selection = config["selection"]
    tracker_variants = build_tracker_variants(config)
    audio_models = build_audio_model_list(config)
    manifest: List[Dict[str, Any]] = []
    completed_runs = 0
    cleanup_done = False

    try:
        patch_job_state(job_id, {"status": "running", "clip_id": clip_id})
        prepared = prepare_clip(
            video_path=config["video_name"],
            output_root=str(prepared_root),
            clip_id=clip_id,
            target_fps=config.get("target_fps"),
            target_width=config.get("target_width"),
            target_height=config.get("target_height"),
            reference_audio=config.get("reference_audio_path"),
            ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"),
            progress_callback=stage_callback(job_id, "dataset_prep"),
        )
        patch_job_state(job_id, {"results": {"prepared_clip": prepared}})

        for tracker_variant in tracker_variants:
            tracking_cache_dir = tracking_root / tracker_variant["key"]
            patch_job_state(job_id, {
                "current_run": {
                    "index": completed_runs + 1,
                    "total": len(tracker_variants) * len(audio_models),
                    "label": tracker_variant["label"],
                    "tracker_variant": tracker_variant["label"],
                    "audio_model": None,
                }
            })
            tracking_summary = track_object(
                video_path=prepared["prepared_video_path"],
                output_path=str(tracking_cache_dir / "video" / "tracked.mp4"),
                checkpoint_path=config["checkpoint_path"],
                config_path=config["config_path"],
                frames_dir=str(tracking_cache_dir / "frames"),
                extract_frames=True,
                selection_mode=config["selection_mode"],
                max_frames=config.get("max_frames"),
                start_time=config.get("start_time", 0.0),
                scale=config.get("scale", 1.0),
                initial_selection=selection_from_dict(selection),
                skip_mask_confirmation=True,
                sam_interval=tracker_variant["sam_interval"],
                dynamic_interval=tuple(tracker_variant["dynamic_interval"]) if tracker_variant.get("dynamic_interval") else None,
                background_color=None,
                show_preview=False,
                artifacts_dir=str(tracking_cache_dir),
                save_mask_pngs=config.get("save_mask_pngs", False),
                ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"),
                progress_callback=stage_callback(job_id, "tracking"),
            )

            for audio_model_size in audio_models:
                run_id = f"{tracker_variant['key']}__{audio_model_size}"
                run_dir = runs_root / run_id
                patch_job_state(job_id, {
                    "current_run": {
                        "index": completed_runs + 1,
                        "total": len(tracker_variants) * len(audio_models),
                        "label": f"{tracker_variant['label']} | {audio_model_size}",
                        "tracker_variant": tracker_variant["label"],
                        "audio_model": audio_model_size,
                    }
                })
                copied_tracking_summary = materialize_tracking_outputs(tracking_summary, run_dir)
                audio_input_video_path = create_audio_aligned_segment(
                    source_video_path=prepared["canonical_video_path"],
                    output_video_path=str(run_dir / "audio_input" / "input_segment.mp4"),
                    start_time_seconds=float(copied_tracking_summary.get("start_time_seconds", 0.0) or 0.0),
                    frame_count=int(copied_tracking_summary.get("total_frames") or 0),
                    fps=float(copied_tracking_summary.get("fps") or prepared.get("fps") or 0.0),
                    ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"),
                )
                audio_summary = run_audio_pipeline(
                    video_path=audio_input_video_path,
                    mask_path=copied_tracking_summary["mask_stack_path"],
                    output_dir=str(run_dir / "audio"),
                    clip_id=clip_id,
                    model_size=audio_model_size,
                    model_id=config.get("audio_model_id"),
                    prompt_mode="visual",
                    device=config.get("audio_device"),
                    audio_precision=config.get("audio_precision", "auto"),
                    predict_spans=config.get("predict_spans", False),
                    reranking_candidates=config.get("reranking_candidates", 1),
                    allow_placeholder=config.get("allow_placeholder_audio", False),
                    mux_video_outputs=True,
                    ffmpeg_bin=config.get("ffmpeg_bin", "ffmpeg"),
                    progress_callback=stage_callback(job_id, "audio"),
                )
                evaluation_summary = evaluate_run(
                    output_dir=str(run_dir / "eval"),
                    clip_id=clip_id,
                    model_size=audio_model_size,
                    model_id=audio_summary.get("model_id"),
                    predicted_mask_path=copied_tracking_summary["mask_stack_path"],
                    ground_truth_mask_path=config.get("ground_truth_mask_path"),
                    estimated_audio_path=audio_summary.get("target_audio_path"),
                    reference_audio_path=config.get("reference_audio_path"),
                    audio_metadata_path=audio_summary.get("metadata_path"),
                    progress_callback=stage_callback(job_id, "evaluation"),
                )
                manifest.append(
                    build_run_manifest_entry(
                        run_id=run_id,
                        run_dir=run_dir,
                        tracker_variant=tracker_variant,
                        audio_model_size=audio_model_size,
                        tracking_summary=copied_tracking_summary,
                        audio_summary=audio_summary,
                        evaluation_summary=evaluation_summary,
                    )
                )
                completed_runs += 1
                patch_job_state(job_id, {"completed_runs": completed_runs, "results": {"run_manifest": manifest}})
                if config.get("release_memory_after_run", True):
                    cleanup_runtime_memory()
                del copied_tracking_summary
                del audio_summary
                del evaluation_summary

            del tracking_summary
            if config.get("release_memory_after_run", True):
                cleanup_runtime_memory()

        analysis_summary = analyze_results(
            experiment_root=str(runs_root),
            output_dir=str(analysis_root),
            progress_callback=stage_callback(job_id, "analytics"),
        )
        final_results: Dict[str, Any] = {
            "prepared_clip": prepared,
            "run_manifest": manifest,
            "analysis_summary": analysis_summary,
        }
        if manifest:
            first_outputs = summarize_job_outputs({"results": {"run_manifest": manifest, "analysis_summary": analysis_summary}})
            final_results["tracking_summary"] = first_outputs.get("tracking_summary", {})
            final_results["audio_summary"] = first_outputs.get("audio_summary", {})
            final_results["evaluation_summary"] = first_outputs.get("evaluation_summary", {})
            del first_outputs
        if config.get("release_memory_after_run", True):
            cleanup_stats = cleanup_runtime_memory()
            cleanup_done = True
            final_results["cleanup"] = cleanup_stats
            patch_job_state(job_id, {
                "results": {"cleanup": cleanup_stats},
                "stages": {
                    "analytics": {
                        "stats": cleanup_stats,
                        "message": "Run finished and memory cleanup completed.",
                    }
                },
            })
        mark_job_completed(job_id, final_results)
    except Exception as exc:
        active_state = load_job_state(job_id)
        active_stage = active_state.get("current_stage") if active_state else "tracking"
        mark_job_failed(job_id, active_stage or "tracking", exc)
    finally:
        for name in ["prepared", "manifest", "analysis_summary", "final_results"]:
            if name in locals():
                try:
                    del locals()[name]
                except Exception:
                    pass
        if config.get("release_memory_after_run", True) and not cleanup_done:
            cleanup_stats = cleanup_runtime_memory()
            try:
                patch_job_state(job_id, {
                    "results": {"cleanup": cleanup_stats},
                    "stages": {
                        "analytics": {
                            "stats": cleanup_stats,
                            "message": "Memory cleanup completed after the run stopped.",
                        }
                    },
                })
            except Exception:
                pass
        ACTIVE_THREADS.pop(job_id, None)


def run_ui_job(job_id: str) -> None:
    state = load_job_state(job_id)
    if state is None:
        raise FileNotFoundError(f"Job state not found for {job_id}")
    config = deepcopy(state.get("config") or {})
    _run_job(job_id, config)


def start_ui_job(config: Dict[str, Any]) -> str:
    ensure_ui_dirs()
    active = get_active_job_state()
    if active is not None:
        raise RuntimeError(f"Job {active['job_id']} is already running. Wait for it to finish before starting another.")

    job_id = generate_job_id(config)
    initial_state = build_initial_job_state(job_id, config)
    save_job_state(job_id, initial_state)

    worker_script = Path(__file__).resolve().parent / "ui_worker.py"
    popen_kwargs = {
        "cwd": str(Path(__file__).resolve().parent),
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [sys.executable, str(worker_script), "--job-id", job_id],
        **popen_kwargs,
    )
    patch_job_state(job_id, {"worker_pid": int(process.pid)})
    return job_id
