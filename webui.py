from __future__ import annotations

import json
import time

from runtime_console import apply_runtime_console_config

apply_runtime_console_config()

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from path_layout import INPUT_ROOT, PROJECT_ROOT
from ui_backend import (
    TRACKER_PRESETS,
    VALID_AUDIO_MODEL_SIZES,
    create_posthoc_analysis_set,
    enqueue_ui_job,
    enqueue_ui_jobs,
    format_run_label,
    get_active_job_state,
    get_latest_job_state,
    get_queue_state,
    get_recent_analysis_set_ids,
    get_recent_job_ids,
    list_ground_truth_mask_files,
    list_input_videos,
    list_reference_audio_files,
    list_posthoc_runs_by_clip,
    load_analysis_set,
    load_job_state,
    load_metrics_preview,
    load_selection_frame,
    preview_initial_mask,
    parse_selection_objects,
    remove_queued_job,
    start_ui_job,
    start_queue_worker_if_needed,
    summarize_job_outputs,
)

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:  # pragma: no cover - optional runtime dependency
    st_canvas = None


st.set_page_config(page_title="SAM Tracker + Audio UI", layout="wide")
try:
    st.set_option("global.dataFrameSerialization", "legacy")
except Exception:
    pass


def rerun_app() -> None:
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")
    rerun()


def draw_divider() -> None:
    if hasattr(st, "divider"):
        st.divider()
    else:
        st.markdown("---")


def show_progress(value: float, text: Optional[str] = None):
    normalized = max(0.0, min(1.0, float(value or 0.0)))
    try:
        return st.progress(normalized, text=text)
    except TypeError:
        progress_bar = st.progress(normalized)
        if text:
            st.caption(text)
        return progress_bar


def compat_button(*args, **kwargs):
    try:
        return st.button(*args, **kwargs)
    except TypeError:
        kwargs.pop("use_container_width", None)
        return st.button(*args, **kwargs)


def compat_column_button(column, *args, **kwargs):
    try:
        return column.button(*args, **kwargs)
    except TypeError:
        kwargs.pop("use_container_width", None)
        return column.button(*args, **kwargs)


def compat_download_button(column, *args, **kwargs):
    try:
        return column.download_button(*args, **kwargs)
    except TypeError:
        kwargs.pop("use_container_width", None)
        return column.download_button(*args, **kwargs)


def render_metric(label: str, value: Any) -> None:
    display_value = "-" if value in (None, "", []) else value
    st.metric(label, display_value)


def render_compact_kv(rows: list[tuple[str, Any]]) -> None:
    if not rows:
        return
    items = []
    for label, value in rows:
        display_value = "-" if value in (None, "", []) else value
        items.append(
            "<div class='compact-kv-item'>"
            f"<div class='compact-kv-label'>{label}</div>"
            f"<div class='compact-kv-value'>{display_value}</div>"
            "</div>"
        )
    st.markdown(
        """
        <style>
        .compact-kv-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.45rem 0.65rem;
            margin: 0.25rem 0 0.75rem 0;
        }
        .compact-kv-item {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 0.45rem;
            padding: 0.45rem 0.55rem;
            background: rgba(250, 250, 250, 0.72);
        }
        .compact-kv-label {
            color: rgba(49, 51, 63, 0.62);
            font-size: 0.68rem;
            font-weight: 600;
            line-height: 1.1;
            text-transform: uppercase;
            letter-spacing: 0.025em;
            margin-bottom: 0.15rem;
        }
        .compact-kv-value {
            color: rgb(49, 51, 63);
            font-size: 0.82rem;
            font-weight: 500;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        </style>
        """
        + f"<div class='compact-kv-grid'>{''.join(items)}</div>",
        unsafe_allow_html=True,
    )


def format_seconds(value: Any) -> str:
    if value in (None, "", []):
        return "-"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric:.2f} s"


STAGE_LABELS = {
    "dataset_prep": "Dataset Prep",
    "tracking": "Tracking",
    "audio": "SAM-Audio",
    "evaluation": "Evaluation",
    "analytics": "Analyze Results",
}


def format_stage_status(stage_state: Dict[str, Any]) -> str:
    status = (stage_state.get("status") or "pending").lower()
    if status == "completed":
        return "Completed"
    if status == "running":
        return "Running"
    if status == "failed":
        return "Failed"
    return "Pending"


def render_stage_overview(job_state: Dict[str, Any]) -> None:
    st.markdown("**Pipeline Stages**")
    stages = job_state.get("stages", {}) or {}
    for stage_key in ["dataset_prep", "tracking", "audio", "evaluation", "analytics"]:
        stage_state = stages.get(stage_key, {}) or {}
        cols = st.columns([1.15, 0.7, 0.45])
        cols[0].markdown(f"**{STAGE_LABELS.get(stage_key, stage_key)}**")
        cols[1].caption(format_stage_status(stage_state))
        cols[2].caption(f"{int(round(float(stage_state.get('progress', 0.0) or 0.0) * 100))}%")
        message = stage_state.get("message") or ""
        if message:
            st.caption(message)


def render_html_table(rows, columns: Optional[list[str]] = None, max_rows: Optional[int] = None) -> None:
    if rows is None:
        return
    if hasattr(rows, "to_dict"):
        data = rows.to_dict(orient="records")
        columns = columns or list(rows.columns)
    else:
        data = list(rows)
        if not data:
            st.info("No rows to display.")
            return
        columns = columns or list(data[0].keys())
    if max_rows is not None:
        data = data[:max_rows]
    if not data:
        st.info("No rows to display.")
        return

    header_html = "".join(f"<th style='text-align:left;padding:6px;border-bottom:1px solid #ddd;'>{column}</th>" for column in columns)
    body_rows = []
    for row in data:
        cells = []
        for column in columns:
            value = row.get(column, "") if isinstance(row, dict) else ""
            cells.append(f"<td style='padding:6px;border-bottom:1px solid #eee;'>{value}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    html = (
        "<div style='overflow-x:auto;'>"
        "<table style='border-collapse:collapse;width:100%;font-size:0.9rem;'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_metrics_plot(df, chart_cols: list[str]) -> None:
    if df is None or df.empty or not chart_cols:
        return
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for column in chart_cols:
        ax.plot(df["frame_idx"], df[column], label=column)
    ax.set_xlabel("frame_idx")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def resolve_media_path(path_value: str) -> Optional[str]:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    return str(path.resolve())


def render_video_file(path_value: Optional[str], label: Optional[str] = None, container=None) -> None:
    if not path_value:
        return
    resolved_path = resolve_media_path(path_value)
    if resolved_path is None:
        return
    target = container or st
    if label:
        target.markdown(label)
    target.video(resolved_path)


def render_audio_file(path_value: Optional[str], container=None) -> None:
    if not path_value:
        return
    resolved_path = resolve_media_path(path_value)
    if resolved_path is None:
        return
    target = container or st
    target.audio(resolved_path)


def render_image_file(path_value: Optional[str], caption: Optional[str] = None, container=None) -> None:
    if not path_value:
        return
    resolved_path = resolve_media_path(path_value)
    if resolved_path is None:
        return
    target = container or st
    target.image(resolved_path, caption=caption)


def clear_preview_state() -> None:
    st.session_state["mask_preview"] = None
    st.session_state["mask_preview_signature"] = None

def options_or_none(label: str, options: list[str], key: str, help_text: Optional[str] = None) -> Optional[str]:
    values = ["(None)"] + options
    choice = st.selectbox(label, values, key=key, help=help_text)
    return None if choice == "(None)" else choice


def list_local_audio_model_ids() -> list[str]:
    model_root = PROJECT_ROOT / "sam_audio_models"
    if not model_root.exists():
        return []
    return sorted(str(Path("sam_audio_models") / child.name) for child in model_root.iterdir() if child.is_dir())


def ensure_session_defaults() -> None:
    st.session_state.setdefault("ui_selection", None)
    st.session_state.setdefault("selection_config", None)
    st.session_state.setdefault("selection_config_error", None)
    st.session_state.setdefault("queue_manifest", None)
    st.session_state.setdefault("queue_manifest_error", None)
    st.session_state.setdefault("auto_start_queue_worker", False)
    st.session_state.setdefault("last_job_id", None)
    st.session_state.setdefault("selection_mode", "point")
    st.session_state.setdefault("scale", 1.0)
    st.session_state.setdefault("start_time", 0.0)
    st.session_state.setdefault("run_mode", "single")
    st.session_state.setdefault("single_tracker_preset", "dynamic_optical_flow")
    st.session_state.setdefault("comparison_tracker_variants", ["dynamic_optical_flow", "full_sam"])
    st.session_state.setdefault("comparison_audio_model_sizes", ["small-tv", "base-tv"])
    st.session_state.setdefault("audio_precision", "auto")
    st.session_state.setdefault("mask_preview", None)
    st.session_state.setdefault("mask_preview_signature", None)
    st.session_state.setdefault("observed_job_id", None)
    st.session_state.setdefault("observed_job_status", None)
    st.session_state.setdefault("selected_posthoc_set_id", None)
    st.session_state.setdefault("segment_processing", False)
    st.session_state.setdefault("segment_length_seconds", 15.0)
    st.session_state.setdefault("segment_overlap_seconds", 2.0)


def parse_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def selection_summary(selection: Optional[Dict[str, Any]]) -> str:
    if not selection:
        return "No selection captured yet."
    if selection.get("mode") == "bbox":
        bbox = selection.get("bbox", [])
        return f"Bounding box: {bbox}"
    return f"Points: {selection.get('points', [])}"


def normalize_selection_payload(selection: Any, label: str = "selection") -> Dict[str, Any]:
    if not isinstance(selection, dict):
        raise ValueError(f"{label} must be an object.")
    mode = selection.get("mode")
    if mode == "bbox":
        bbox = selection.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{label}.bbox must contain exactly four numbers.")
        try:
            normalized_bbox = [float(value) for value in bbox]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label}.bbox must contain only numeric values.") from exc
        return {"mode": "bbox", "bbox": normalized_bbox}

    if mode == "point":
        points = selection.get("points")
        labels = selection.get("labels")
        if not isinstance(points, list) or not isinstance(labels, list) or len(points) != len(labels):
            raise ValueError(f"{label}.points and {label}.labels must be equal-length lists.")
        normalized_points = []
        normalized_labels = []
        for index, point in enumerate(points):
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"{label}.points[{index}] must contain exactly two numbers.")
            try:
                normalized_points.append([float(point[0]), float(point[1])])
                normalized_labels.append(int(labels[index]))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label}.points and {label}.labels must contain numeric values.") from exc
        if not normalized_points:
            raise ValueError(f"{label}.points must contain at least one point.")
        return {"mode": "point", "points": normalized_points, "labels": normalized_labels}

    raise ValueError(f"{label}.mode must be either 'bbox' or 'point'.")


def parse_selection_config_bytes(raw_bytes: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
    except Exception as exc:
        raise ValueError("Selection config must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Selection config root must be a JSON object.")
    selections = payload.get("selections")
    if not isinstance(selections, dict) or not selections:
        raise ValueError("Selection config must contain a non-empty 'selections' object.")
    normalized = {
        str(key): normalize_selection_payload(value, label=f"selections.{key}")
        for key, value in selections.items()
    }
    return {"version": int(payload.get("version", 1) or 1), "selections": normalized}


def find_config_selection(selection_config: Optional[Dict[str, Any]], video_name: Optional[str]) -> Optional[Dict[str, Any]]:
    if not selection_config or not video_name:
        return None
    selections = selection_config.get("selections") or {}
    if video_name in selections:
        return selections[video_name]
    stem = Path(video_name).stem
    return selections.get(stem)


def build_selection_config_download(video_name: Optional[str], selection: Optional[Dict[str, Any]]) -> str:
    key = video_name or "video_name_or_clip_id"
    payload = {
        "version": 1,
        "selections": {
            key: selection or {"mode": "bbox", "bbox": [100, 120, 420, 560]},
        },
    }
    return json.dumps(payload, indent=2)


DEFAULT_QUEUE_JOB_CONFIG: Dict[str, Any] = {
    "selection_mode": "point",
    "checkpoint_path": "checkpoints/sam2.1_hiera_tiny.pt",
    "config_path": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "start_time": 0.0,
    "scale": 1.0,
    "max_frames": None,
    "run_mode": "single",
    "single_tracker_preset": "dynamic_optical_flow",
    "dynamic_min": 2,
    "dynamic_max": 20,
    "tracker_variants": [],
    "save_mask_pngs": True,
    "audio_model_size": "small-tv",
    "audio_model_sizes": [],
    "audio_model_id": None,
    "audio_device": None,
    "audio_precision": "auto",
    "predict_spans": False,
    "reranking_candidates": 1,
    "allow_placeholder_audio": False,
    "release_memory_after_run": True,
    "reference_audio_path": None,
    "ground_truth_mask_path": None,
    "target_fps": None,
    "target_width": None,
    "target_height": None,
    "ffmpeg_bin": "ffmpeg",
    "segment_processing": False,
    "segment_length_seconds": 15.0,
    "segment_overlap_seconds": 2.0,
}


def normalize_queue_job_config(job: Any, defaults: Optional[Dict[str, Any]] = None, index: int = 0) -> Dict[str, Any]:
    if not isinstance(job, dict):
        raise ValueError(f"jobs[{index}] must be an object.")
    merged = dict(DEFAULT_QUEUE_JOB_CONFIG)
    if defaults:
        merged.update(defaults)
    merged.update(job)
    if not merged.get("video_name"):
        raise ValueError(f"jobs[{index}].video_name is required.")
    merged["selection"] = normalize_selection_payload(merged.get("selection"), label=f"jobs[{index}].selection")
    merged["selection_mode"] = merged.get("selection_mode") or merged["selection"].get("mode", "point")
    merged["start_time"] = float(merged.get("start_time") or 0.0)
    merged["scale"] = float(merged.get("scale") or 1.0)
    merged["max_frames"] = parse_optional_int(merged.get("max_frames"))
    merged["dynamic_min"] = int(merged.get("dynamic_min") or 2)
    merged["dynamic_max"] = int(merged.get("dynamic_max") or 20)
    merged["reranking_candidates"] = int(merged.get("reranking_candidates") or 1)
    merged["segment_length_seconds"] = float(merged.get("segment_length_seconds") or 15.0)
    merged["segment_overlap_seconds"] = float(merged.get("segment_overlap_seconds") or 2.0)
    for key in ["predict_spans", "allow_placeholder_audio", "release_memory_after_run", "save_mask_pngs", "segment_processing"]:
        merged[key] = bool(merged.get(key))
    return merged


def parse_queue_manifest_bytes(raw_bytes: bytes) -> Dict[str, Any]:
    try:
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
    except Exception as exc:
        raise ValueError("Queue manifest must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Queue manifest root must be a JSON object.")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Queue manifest must contain a non-empty 'jobs' list.")
    defaults = payload.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise ValueError("Queue manifest 'defaults' must be an object when provided.")
    normalized_jobs = [
        normalize_queue_job_config(job, defaults=defaults, index=index)
        for index, job in enumerate(jobs)
    ]
    return {
        "version": int(payload.get("version", 1) or 1),
        "motion_type": payload.get("motion_type"),
        "jobs": normalized_jobs,
    }


def build_queue_manifest_template(video_name: Optional[str], selection: Optional[Dict[str, Any]]) -> str:
    job = dict(DEFAULT_QUEUE_JOB_CONFIG)
    job.update({
        "video_name": video_name or "video_name.mp4",
        "selection": selection or {"mode": "bbox", "bbox": [100, 120, 420, 560]},
        "selection_mode": (selection or {}).get("mode", "bbox"),
        "audio_precision": "fp16",
        "reranking_candidates": 1,
    })
    payload = {
        "version": 1,
        "motion_type": "low",
        "defaults": {
            "checkpoint_path": "checkpoints/sam2.1_hiera_tiny.pt",
            "config_path": "configs/sam2.1/sam2.1_hiera_t.yaml",
            "segment_processing": False,
            "segment_length_seconds": 15.0,
            "segment_overlap_seconds": 2.0,
        },
        "jobs": [job],
    }
    return json.dumps(payload, indent=2)


def build_preview_signature(video_name: Optional[str], selection: Optional[Dict[str, Any]]) -> Optional[str]:
    if not video_name or not selection:
        return None
    payload = {
        "video_name": video_name,
        "selection": selection,
        "selection_mode": st.session_state.get("selection_mode", "point"),
        "checkpoint_path": st.session_state.get("checkpoint_path", "checkpoints/sam2.1_hiera_tiny.pt"),
        "config_path": st.session_state.get("config_path", "configs/sam2.1/sam2.1_hiera_t.yaml"),
        "start_time": float(st.session_state.get("start_time", 0.0) or 0.0),
        "scale": float(st.session_state.get("scale", 1.0) or 1.0),
    }
    return json.dumps(payload, sort_keys=True)


def prepare_canvas_image(image: Image.Image, max_width: int = 640) -> tuple[Image.Image, float, float]:
    if image.width <= max_width:
        return image, 1.0, 1.0
    ratio = max_width / image.width
    resized = image.resize((int(image.width * ratio), int(image.height * ratio)))
    scale_x = image.width / resized.width
    scale_y = image.height / resized.height
    return resized, scale_x, scale_y


def build_job_config(video_name: str) -> Dict[str, Any]:
    run_mode = st.session_state.get("run_mode", "single")
    return {
        "video_name": video_name,
        "selection": st.session_state.get("ui_selection"),
        "selection_mode": st.session_state.get("selection_mode", "point"),
        "checkpoint_path": st.session_state.get("checkpoint_path", "checkpoints/sam2.1_hiera_tiny.pt"),
        "config_path": st.session_state.get("config_path", "configs/sam2.1/sam2.1_hiera_t.yaml"),
        "start_time": float(st.session_state.get("start_time", 0.0) or 0.0),
        "scale": float(st.session_state.get("scale", 1.0) or 1.0),
        "max_frames": parse_optional_int(st.session_state.get("max_frames")),
        "run_mode": run_mode,
        "single_tracker_preset": st.session_state.get("single_tracker_preset", "dynamic_optical_flow"),
        "dynamic_min": int(st.session_state.get("dynamic_min", 2) or 2),
        "dynamic_max": int(st.session_state.get("dynamic_max", 20) or 20),
        "tracker_variants": st.session_state.get("comparison_tracker_variants", []),
        "save_mask_pngs": bool(st.session_state.get("save_mask_pngs", True)),
        "audio_model_size": st.session_state.get("audio_model_size", "small-tv"),
        "audio_model_sizes": st.session_state.get("comparison_audio_model_sizes", []),
        "audio_model_id": st.session_state.get("audio_model_id_override") or None,
        "audio_device": st.session_state.get("audio_device") or None,
        "audio_precision": st.session_state.get("audio_precision", "auto"),
        "predict_spans": bool(st.session_state.get("predict_spans", False)),
        "reranking_candidates": int(st.session_state.get("reranking_candidates", 1) or 1),
        "allow_placeholder_audio": bool(st.session_state.get("allow_placeholder_audio", False)),
        "release_memory_after_run": bool(st.session_state.get("release_memory_after_run", True)),
        "reference_audio_path": st.session_state.get("reference_audio_path") or None,
        "ground_truth_mask_path": st.session_state.get("ground_truth_mask_path") or None,
        "target_fps": parse_optional_float(st.session_state.get("target_fps")),
        "target_width": parse_optional_int(st.session_state.get("target_width")),
        "target_height": parse_optional_int(st.session_state.get("target_height")),
        "ffmpeg_bin": st.session_state.get("ffmpeg_bin", "ffmpeg") or "ffmpeg",
        "segment_processing": bool(st.session_state.get("segment_processing", False)),
        "segment_length_seconds": float(st.session_state.get("segment_length_seconds", 15.0) or 15.0),
        "segment_overlap_seconds": float(st.session_state.get("segment_overlap_seconds", 2.0) or 2.0),
    }


def render_selection_config_controls(video_name: Optional[str]) -> None:
    st.markdown("**Selection Config**")
    uploaded = st.file_uploader(
        "Upload selection config",
        type=["json"],
        key="selection_config_upload",
        help="Load saved point or bbox selections keyed by input filename or clip id.",
    )
    if uploaded is not None:
        try:
            st.session_state["selection_config"] = parse_selection_config_bytes(uploaded.getvalue())
            st.session_state["selection_config_error"] = None
        except Exception as exc:
            st.session_state["selection_config"] = None
            st.session_state["selection_config_error"] = str(exc)

    if st.session_state.get("selection_config_error"):
        st.error(st.session_state["selection_config_error"])

    selection_config = st.session_state.get("selection_config")
    config_selection = find_config_selection(selection_config, video_name)
    if selection_config:
        count = len(selection_config.get("selections") or {})
        st.caption(f"Loaded {count} saved selection{'s' if count != 1 else ''}.")
        if config_selection:
            st.info(selection_summary(config_selection))
        else:
            st.warning("No matching selection found for this video.")

    cols = st.columns(2)
    if compat_column_button(cols[0], "Apply config selection", use_container_width=True, disabled=config_selection is None):
        st.session_state["ui_selection"] = config_selection
        st.session_state["selection_mode"] = config_selection.get("mode", st.session_state.get("selection_mode", "point"))
        clear_preview_state()
        rerun_app()

    export_payload = build_selection_config_download(video_name, st.session_state.get("ui_selection"))
    compat_download_button(
        cols[1],
        "Export current selection",
        data=export_payload,
        file_name=f"{Path(video_name or 'selection').stem}_selection_config.json",
        mime="application/json",
        use_container_width=True,
        disabled=st.session_state.get("ui_selection") is None,
    )


def render_tracker_controls() -> None:
    st.subheader("Tracker")
    st.text_input("SAM checkpoint", value="checkpoints/sam2.1_hiera_tiny.pt", key="checkpoint_path", help="Path to the SAM2 checkpoint used for visual tracking. Changing models can affect accuracy, speed, and VRAM usage.")
    st.text_input("SAM config", value="configs/sam2.1/sam2.1_hiera_t.yaml", key="config_path", help="Hydra config for the SAM2 checkpoint. It must match the checkpoint family, or tracking quality and loading can fail.")
    st.radio("Selection mode", ["point", "bbox"], key="selection_mode", horizontal=True, help="Point mode uses clicked positive prompts. Bounding box mode gives SAM a tighter region and can reduce accidental full-frame masks.")
    st.radio(
        "Run mode",
        [("Single run", "single"), ("Comparison experiment", "comparison")],
        key="run_mode_radio",
        format_func=lambda item: item[0],
        index=0 if st.session_state.get("run_mode", "single") == "single" else 1,
        help="Single run tests one tracker/audio configuration. Comparison experiment runs multiple tracker and/or audio variants on the same clip for side-by-side analysis.",
    )
    st.session_state["run_mode"] = st.session_state["run_mode_radio"][1]
    st.slider("Scale", min_value=0.25, max_value=1.0, value=1.0, step=0.05, key="scale", help="Downscale the video before tracking. Lower values run faster and use less memory, but can reduce mask detail and audio-prompt precision.")
    st.number_input("Start time (seconds)", min_value=0.0, step=0.5, key="start_time", help="Skip the beginning of the clip. Useful for testing a later segment without processing the whole video.")
    st.text_input("Max frames", value="", key="max_frames", help="Leave blank to process the full clip. Lower values make test runs much faster but reduce total tracking/audio coverage.")

    if st.session_state.get("run_mode") == "comparison":
        st.multiselect(
            "Tracker presets to compare",
            list(TRACKER_PRESETS.keys()),
            default=st.session_state.get("comparison_tracker_variants", ["dynamic_optical_flow", "full_sam"]),
            format_func=lambda key: TRACKER_PRESETS[key],
            key="comparison_tracker_variants",
            help="Run the same clip repeatedly with different tracker settings, including optical-flow interpolation and full SAM baselines.",
        )
    else:
        st.selectbox(
            "Tracker preset",
            list(TRACKER_PRESETS.keys()),
            index=list(TRACKER_PRESETS.keys()).index(st.session_state.get("single_tracker_preset", "dynamic_optical_flow")),
            format_func=lambda key: TRACKER_PRESETS[key],
            key="single_tracker_preset",
            help="Choose the tracking strategy. Full SAM is strongest but slowest. Dynamic SAM + optical flow is faster and usually a better efficiency baseline. Static sparse SAM trades refresh rate for speed.",
        )

    if (st.session_state.get("run_mode") == "comparison") or (st.session_state.get("single_tracker_preset") == "dynamic_optical_flow"):
        st.caption("Dynamic SAM + Optical Flow uses the min/max interval values below.")
        cols = st.columns(2)
        cols[0].number_input("Dynamic min interval", min_value=1, step=1, value=2, key="dynamic_min", help="Shortest allowed gap between full SAM refreshes. Lower values improve robustness during motion but cost more runtime.")
        cols[1].number_input("Dynamic max interval", min_value=1, step=1, value=20, key="dynamic_max", help="Longest allowed gap between full SAM refreshes. Higher values reduce compute but can drift more if optical flow is wrong.")

    st.checkbox("Save mask PNGs", value=True, key="save_mask_pngs", help="Stores per-frame mask PNGs for debugging. Useful for inspection, but it creates more files and uses more disk space.")


def render_audio_controls() -> None:
    st.subheader("Audio")
    if st.session_state.get("run_mode") == "comparison":
        st.multiselect(
            "Audio models to compare",
            VALID_AUDIO_MODEL_SIZES,
            default=st.session_state.get("comparison_audio_model_sizes", ["small-tv", "base-tv"]),
            key="comparison_audio_model_sizes",
            help="Runs every selected tracker preset against every selected audio model. Great for comparing quality versus runtime across configurations.",
        )
    else:
        st.selectbox("Audio model size", VALID_AUDIO_MODEL_SIZES, key="audio_model_size", help="Chooses the SAM-Audio model family. Larger models may separate better but usually take longer and use more memory.")
    local_model_options = list_local_audio_model_ids()
    selected_model_override = options_or_none(
        "Audio model id override",
        local_model_options,
        key="audio_model_id_select",
        help_text="Choose a local SAM-Audio model folder like sam_audio_models/small-tv to avoid typing it each run. Leave as (None) to use the default remote model id for the chosen model size.",
    )
    st.session_state["audio_model_id_override"] = selected_model_override
    st.text_input("Audio device override", value="", key="audio_device", help="Optional device override such as cuda or cpu. GPU is usually much faster, while CPU is slower but can be safer for memory-constrained systems.")
    st.selectbox("Audio precision", ["auto", "fp32", "bf16", "fp16"], key="audio_precision", help="Controls SAM-Audio inference precision. Auto uses bf16 on newer CUDA GPUs, fp16 on older CUDA GPUs, and fp32 elsewhere. Lower precision can reduce VRAM use and sometimes improve speed, but may affect stability or output quality.")
    st.checkbox("Predict spans", value=False, key="predict_spans", help="Lets SAM-Audio predict when the target sound is active in time. This can help intermittent sounds, but for continuous speech it may not always improve results.")
    st.number_input("Reranking candidates", min_value=1, max_value=8, value=1, step=1, key="reranking_candidates", help="How many candidate separations SAM-Audio considers before picking one. Higher values may improve quality but increase runtime and memory use.")
    st.checkbox("Allow placeholder audio", value=False, key="allow_placeholder_audio", help="Smoke-test mode. This does not perform real audio separation, so it should not be used for final evaluation metrics.")
    st.checkbox("Release RAM/VRAM after run", value=True, key="release_memory_after_run", help="Free PyTorch caches and run garbage collection after the job finishes instead of keeping models warm in memory.")


def render_left_panel(video_name: Optional[str], active_job: Optional[Dict[str, Any]]) -> None:
    st.subheader("Run Configuration")
    if not video_name:
        st.info("Add a video file to the input folder to start.")
        return

    render_tracker_controls()
    draw_divider()
    render_audio_controls()
    draw_divider()

    st.subheader("Optional Evaluation Inputs")
    ref_audio = options_or_none("Reference audio", list_reference_audio_files(), key="reference_audio_select", help_text="Optional clean reference audio for SI-SDR evaluation. Without it, audio-quality evaluation is skipped.")
    gt_mask = options_or_none("Ground-truth masks", list_ground_truth_mask_files(), key="ground_truth_mask_select", help_text="Optional ground-truth masks for IoU evaluation. Without them, visual-tracking evaluation is skipped.")
    st.session_state["reference_audio_path"] = str((INPUT_ROOT / ref_audio).resolve()) if ref_audio else None
    st.session_state["ground_truth_mask_path"] = str((INPUT_ROOT / gt_mask).resolve()) if gt_mask else None

    with st.expander("Advanced Prep"):
        st.text_input("Target FPS", value="", key="target_fps", help="Leave blank to keep the original FPS.")
        st.text_input("Target width", value="", key="target_width", help="Leave blank to keep the original width.")
        st.text_input("Target height", value="", key="target_height", help="Leave blank to keep the original height.")
        st.text_input("ffmpeg binary", value="ffmpeg", key="ffmpeg_bin", help="Command name or full path for ffmpeg. Leave as ffmpeg for the normal default, or point to a custom binary. Repo-local tools/ffmpeg binaries are auto-detected when available.")
        st.checkbox(
            "Segment processing for long clips",
            value=False,
            key="segment_processing",
            help="Split the selected clip span into overlapping subclips, run tracking and SAM-Audio per segment, then stitch the final outputs back together. This can help longer runs fit on smaller GPUs.",
        )
        segment_cols = st.columns(2)
        segment_cols[0].number_input(
            "Segment length (seconds)",
            min_value=1.0,
            step=1.0,
            value=float(st.session_state.get("segment_length_seconds", 15.0) or 15.0),
            key="segment_length_seconds",
            help="Target duration for each processing chunk before stitching. Shorter segments lower memory pressure but increase boundary management.",
        )
        segment_cols[1].number_input(
            "Segment overlap (seconds)",
            min_value=0.0,
            step=0.5,
            value=float(st.session_state.get("segment_overlap_seconds", 2.0) or 2.0),
            key="segment_overlap_seconds",
            help="Shared context between neighboring segments. Overlap helps prompt continuity and smoother stitched audio, but it must stay smaller than the segment length.",
        )

    st.info(selection_summary(st.session_state.get("ui_selection")))
    queue_state = get_queue_state()
    queue_busy = bool(active_job) or bool(queue_state.get("worker_running"))
    missing_selection = st.session_state.get("ui_selection") is None
    button_label = "Start comparison now" if st.session_state.get("run_mode") == "comparison" else "Start now"
    start_col, queue_col = st.columns(2)
    if compat_column_button(start_col, button_label, type="primary", disabled=queue_busy or missing_selection, use_container_width=True):
        config = build_job_config(video_name)
        try:
            job_id = start_ui_job(config)
            clear_preview_state()
            st.session_state["last_job_id"] = job_id
            st.success(f"Started job {job_id}")
            rerun_app()
        except Exception as exc:
            st.error(str(exc))
    if compat_column_button(queue_col, "Add to queue", disabled=missing_selection, use_container_width=True):
        config = build_job_config(video_name)
        try:
            job_id = enqueue_ui_job(config)
            clear_preview_state()
            st.session_state["last_job_id"] = job_id
            st.success(f"Queued job {job_id}")
            rerun_app()
        except Exception as exc:
            st.error(str(exc))
    if queue_busy:
        st.caption("A job or queue worker is active. Wait for it to finish before starting another direct run.")
    elif queue_state.get("queued"):
        st.caption("Queued jobs are waiting, but idle queued jobs will not block `Start now`.")
    if active_job is not None:
        st.warning(f"Job {active_job['job_id']} is currently running.")


def render_center_panel(video_name: Optional[str]) -> None:
    st.subheader("Selection and Progress")
    if not video_name:
        st.info("No video selected.")
        return

    try:
        frame_info = load_selection_frame(
            video_name,
            start_time=float(st.session_state.get("start_time", 0.0) or 0.0),
            scale=float(st.session_state.get("scale", 1.0) or 1.0),
        )
    except Exception as exc:
        st.error(str(exc))
        return

    image = frame_info["image"]
    render_image, scale_x, scale_y = prepare_canvas_image(image, max_width=640)
    st.caption(f"Selection preview: {render_image.width}x{render_image.height}px")

    if st_canvas is None:
        st.image(render_image, caption="Install streamlit-drawable-canvas to enable click/box selection.", use_column_width=False)
        st.warning("Missing dependency: streamlit-drawable-canvas")
        return

    drawing_mode = "point" if st.session_state.get("selection_mode", "point") == "point" else "rect"
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",
        stroke_width=3,
        stroke_color="#f97316",
        background_image=render_image,
        update_streamlit=True,
        height=render_image.height,
        width=render_image.width,
        drawing_mode=drawing_mode,
        key=f"selection-canvas-{video_name}-{drawing_mode}-{st.session_state.get('scale')}-{st.session_state.get('start_time')}-{render_image.width}",
    )

    selection = None
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        selection = parse_selection_objects(
            canvas_result.json_data.get("objects", []),
            st.session_state.get("selection_mode", "point"),
            scale_x,
            scale_y,
        )

    selection_for_preview = selection or st.session_state.get("ui_selection")
    current_preview_signature = build_preview_signature(video_name, selection_for_preview)

    render_selection_config_controls(video_name)

    cols = st.columns(3)
    if compat_column_button(cols[0], "Use current selection", use_container_width=True, disabled=selection is None):
        st.session_state["ui_selection"] = selection
        rerun_app()
    if compat_column_button(cols[1], "Preview initial mask", use_container_width=True, disabled=selection_for_preview is None):
        try:
            preview = preview_initial_mask(
                video_name=video_name,
                selection=selection_for_preview,
                checkpoint_path=st.session_state.get("checkpoint_path", "checkpoints/sam2.1_hiera_tiny.pt"),
                config_path=st.session_state.get("config_path", "configs/sam2.1/sam2.1_hiera_t.yaml"),
                start_time=float(st.session_state.get("start_time", 0.0) or 0.0),
                scale=float(st.session_state.get("scale", 1.0) or 1.0),
            )
            st.session_state["mask_preview"] = preview
            st.session_state["mask_preview_signature"] = current_preview_signature
        except Exception as exc:
            st.error(str(exc))
    if compat_column_button(cols[2], "Clear selection", use_container_width=True):
        st.session_state["ui_selection"] = None
        st.session_state["mask_preview"] = None
        st.session_state["mask_preview_signature"] = None
        rerun_app()

    if selection:
        st.code(str(selection), language="python")

    preview_state = st.session_state.get("mask_preview")
    if preview_state and st.session_state.get("mask_preview_signature") == current_preview_signature:
        st.markdown("**Initial Mask Preview**")
        st.image(preview_state.get("image_bytes"), caption="First-frame mask preview")
        preview_cols = st.columns(4)
        preview_cols[0].metric("Mask area", preview_state.get("mask_area_px", "-"))
        preview_cols[1].metric("Coverage", preview_state.get("mask_coverage_ratio", "-"))
        preview_cols[2].metric("Empty mask", preview_state.get("mask_empty", "-"))
        preview_cols[3].metric("Score", preview_state.get("score", "-"))
        if preview_state.get("mask_bbox"):
            st.caption(f"Predicted mask bbox: {preview_state['mask_bbox']}")
        if preview_state.get("warning"):
            st.warning(preview_state["warning"])
    elif st.session_state.get("mask_preview") and current_preview_signature is not None:
        st.info("Selection changed since the last preview. Click `Preview initial mask` again to refresh it.")


def render_right_panel(job_state: Optional[Dict[str, Any]]) -> None:
    st.subheader("Current Run")
    if not job_state:
        st.info("No runs yet.")
        return

    show_progress(float(job_state.get("overall_progress", 0.0) or 0.0), text=f"Overall progress: {job_state.get('status', 'idle')}")
    st.caption(f"Job ID: {job_state['job_id']}")
    render_compact_kv([
        ("Status", job_state.get("status")),
        ("Stage", job_state.get("current_stage")),
        ("Clip", job_state.get("clip_id")),
        ("Completed runs", f"{job_state.get('completed_runs', 0)}/{job_state.get('planned_runs', 0)}"),
    ])

    current_run = job_state.get("current_run") or {}
    if current_run:
        st.caption("Current variant")
        render_compact_kv([
            ("Variant", current_run.get("label")),
            ("Tracker", current_run.get("tracker_variant")),
            ("Audio model", current_run.get("audio_model")),
        ])

    current_stage = job_state.get("current_stage")
    if current_stage:
        stage_state = job_state.get("stages", {}).get(current_stage, {})
        show_progress(float(stage_state.get("progress", 0.0) or 0.0), text=stage_state.get("message") or STAGE_LABELS.get(current_stage, current_stage))
        stats = stage_state.get("stats", {}) or {}
        for key in ["sam_frames", "flow_frames", "failure_count", "empty_mask_count", "mean_motion", "runtime_seconds", "mean_iou", "si_sdr", "gc_collected", "cuda_cache_cleared"]:
            if key in stats:
                render_metric(key.replace("_", " ").title(), stats[key])

    render_stage_overview(job_state)

    if job_state.get("error_message"):
        st.error(job_state["error_message"])


def render_queue_panel(queue_state: Dict[str, Any]) -> None:
    draw_divider()
    st.subheader("Queue")
    running = queue_state.get("running")
    queued = queue_state.get("queued") or []
    history = queue_state.get("history") or []
    worker_running = bool(queue_state.get("worker_running"))

    control_cols = st.columns([1.0, 1.0])
    control_cols[0].checkbox(
        "Auto-start queue worker",
        key="auto_start_queue_worker",
        help="Leave off to keep queued jobs idle until you explicitly start the queue. This saves RAM/VRAM while using Start now.",
    )
    if compat_column_button(
        control_cols[1],
        "Start queue worker",
        disabled=not queued or bool(running) or worker_running,
        use_container_width=True,
    ):
        try:
            pid = start_queue_worker_if_needed()
            if pid:
                st.success(f"Started queue worker PID {pid}.")
            else:
                st.info("Queue worker was not started because no queued job is ready or another job is active.")
            rerun_app()
        except Exception as exc:
            st.error(str(exc))
    st.caption("RAM-safe default: queued jobs wait here until you click `Start queue worker` or enable auto-start.")

    uploaded = st.file_uploader(
        "Import queue manifest",
        type=["json"],
        key="queue_manifest_upload",
        help="Load a full list of queued job configs. Use one file per motion type if that matches your experiment setup.",
    )
    if uploaded is not None:
        try:
            st.session_state["queue_manifest"] = parse_queue_manifest_bytes(uploaded.getvalue())
            st.session_state["queue_manifest_error"] = None
        except Exception as exc:
            st.session_state["queue_manifest"] = None
            st.session_state["queue_manifest_error"] = str(exc)

    if st.session_state.get("queue_manifest_error"):
        st.error(st.session_state["queue_manifest_error"])

    manifest = st.session_state.get("queue_manifest")
    import_cols = st.columns(2)
    if manifest:
        motion_label = f" for {manifest.get('motion_type')}" if manifest.get("motion_type") else ""
        import_cols[0].caption(f"Loaded {len(manifest.get('jobs', []))} queued job configs{motion_label}.")
    if compat_column_button(import_cols[0], "Add manifest to queue", disabled=not manifest, use_container_width=True):
        try:
            job_ids = enqueue_ui_jobs(manifest["jobs"], start_worker=bool(st.session_state.get("auto_start_queue_worker", False)))
            st.session_state["last_job_id"] = job_ids[0] if job_ids else st.session_state.get("last_job_id")
            st.success(f"Queued {len(job_ids)} jobs from manifest.")
            rerun_app()
        except Exception as exc:
            st.error(str(exc))
    compat_download_button(
        import_cols[1],
        "Download queue template",
        data=build_queue_manifest_template(None, None),
        file_name="motion_queue_manifest_template.json",
        mime="application/json",
        use_container_width=True,
    )

    if running:
        st.caption(f"Running: {running.get('job_id')} ({running.get('clip_id')})")
    elif worker_running:
        st.caption(f"Queue worker active: PID {queue_state.get('worker_pid')}")
    elif queued and not st.session_state.get("auto_start_queue_worker", False):
        st.info("Queued jobs are waiting. Click `Start queue worker` to run them.")

    if queued:
        st.markdown("**Waiting**")
        for item in queued:
            cols = st.columns([0.25, 1.0, 0.55])
            cols[0].caption(f"#{item.get('position')}")
            cols[1].caption(f"{item.get('job_id')} | {item.get('clip_id')}")
            if compat_column_button(cols[2], "Remove", key=f"remove-queued-{item.get('job_id')}", use_container_width=True):
                try:
                    remove_queued_job(item["job_id"])
                    rerun_app()
                except Exception as exc:
                    st.error(str(exc))
    elif not running and not worker_running:
        st.info("Queue is empty.")

    if history:
        st.markdown("**Recent queue results**")
        rows = [
            {
                "Job": item.get("job_id"),
                "Clip": item.get("clip_id"),
                "Status": item.get("status"),
                "Done": f"{item.get('completed_runs', 0)}/{item.get('planned_runs', 0)}",
            }
            for item in history[:5]
        ]
        render_html_table(rows, columns=["Job", "Clip", "Status", "Done"])


def render_results_panel(job_state: Optional[Dict[str, Any]]) -> None:
    st.subheader("Outputs and Analytics")
    if not job_state:
        st.info("Run something to populate outputs.")
        return

    manifest = (job_state.get("results", {}) or {}).get("run_manifest", [])
    selected_run_id = None
    if manifest:
        selected_label = st.selectbox(
            "Run outputs to inspect",
            [entry["display_label"] for entry in manifest],
            index=0,
            key=f"run-output-select-{job_state['job_id']}",
            help="Choose which run variant to inspect when a comparison experiment produced multiple outputs.",
        )
        selected_entry = next(entry for entry in manifest if entry["display_label"] == selected_label)
        selected_run_id = selected_entry["run_id"]

    outputs = summarize_job_outputs(job_state, run_id=selected_run_id)
    tracking_summary = outputs.get("tracking_summary", {}) or {}
    audio_summary = outputs.get("audio_summary", {}) or {}
    evaluation_summary = outputs.get("evaluation_summary", {}) or {}
    analysis_summary = outputs.get("analysis_summary", {}) or {}
    original_video_path = None
    config = job_state.get("config", {}) if job_state else {}
    video_name = config.get("video_name")
    if video_name:
        original_candidate = INPUT_ROOT / video_name
        if original_candidate.exists():
            original_video_path = str(original_candidate.resolve())

    tabs = st.tabs(["Outputs", "Stats", "Analytics"])

    with tabs[0]:
        comparison_cols = st.columns(2)
        if original_video_path:
            render_video_file(original_video_path, label="**Original Input Video**", container=comparison_cols[0])
        if outputs.get("tracked_video") and Path(outputs["tracked_video"]).exists():
            render_video_file(outputs.get("tracked_video"), label="**Tracked Video**", container=comparison_cols[1])

        processed_video_cols = st.columns(2)
        if outputs.get("target_video") and Path(outputs["target_video"]).exists():
            render_video_file(outputs.get("target_video"), label="**Target Video**", container=processed_video_cols[0])
        if outputs.get("residual_video") and Path(outputs["residual_video"]).exists():
            render_video_file(outputs.get("residual_video"), label="**Residual Video**", container=processed_video_cols[1])

        audio_cols = st.columns(2)
        if outputs.get("target_audio") and Path(outputs["target_audio"]).exists():
            audio_cols[0].markdown("**Target Audio**")
            render_audio_file(outputs.get("target_audio"), container=audio_cols[0])
        if outputs.get("residual_audio") and Path(outputs["residual_audio"]).exists():
            audio_cols[1].markdown("**Residual Audio**")
            render_audio_file(outputs.get("residual_audio"), container=audio_cols[1])

    with tabs[1]:
        st.caption("Tracking FPS reflects visual throughput. Tracking time and SAM-Audio time help you compare runtime cost across runs.")
        metrics_cols = st.columns(6)
        metrics_cols[0].metric("Tracking FPS", tracking_summary.get("pure_fps", "-"))
        metrics_cols[1].metric("Tracking time", format_seconds(tracking_summary.get("pure_time")))
        metrics_cols[2].metric("SAM-Audio time", format_seconds(audio_summary.get("runtime_seconds", evaluation_summary.get("audio_runtime_seconds"))))
        metrics_cols[3].metric("Failure count", tracking_summary.get("failure_count", "-"))
        metrics_cols[4].metric("Mean IoU", evaluation_summary.get("mean_iou", "-"))
        metrics_cols[5].metric("SI-SDR", evaluation_summary.get("si_sdr", "-"))
        st.caption(f"Audio precision: requested {audio_summary.get('requested_audio_precision', '-')}, effective {audio_summary.get('effective_audio_precision', '-')}")
        if manifest:
            st.markdown("**Run Manifest**")
            manifest_rows = []
            for entry in manifest:
                entry_tracking = {}
                entry_audio = {}
                tracking_path = entry.get("tracking_summary_path")
                audio_path = entry.get("audio_metadata_path")
                if tracking_path and Path(tracking_path).exists():
                    try:
                        entry_tracking = json.loads(Path(tracking_path).read_text(encoding="utf-8"))
                    except Exception:
                        entry_tracking = {}
                if audio_path and Path(audio_path).exists():
                    try:
                        entry_audio = json.loads(Path(audio_path).read_text(encoding="utf-8"))
                    except Exception:
                        entry_audio = {}
                manifest_rows.append({
                    "Run": entry["display_label"],
                    "Tracker": entry["tracker_variant_label"],
                    "Audio": entry["audio_model_size"],
                    "Precision": entry_audio.get("effective_audio_precision", entry_audio.get("requested_audio_precision", "-")),
                    "Tracking time": format_seconds(entry_tracking.get("pure_time")),
                    "SAM-Audio time": format_seconds(entry_audio.get("runtime_seconds")),
                })
            render_html_table(manifest_rows, columns=["Run", "Tracker", "Audio", "Precision", "Tracking time", "SAM-Audio time"])
        st.markdown("**Tracking Summary**")
        st.json(tracking_summary)
        st.markdown("**Evaluation Summary**")
        st.json(evaluation_summary)

    with tabs[2]:
        metrics_csv = outputs.get("metrics_csv")
        if metrics_csv and Path(metrics_csv).exists():
            df = load_metrics_preview(metrics_csv)
            if not df.empty:
                render_html_table(df, max_rows=200)
                chart_cols = [column for column in ["motion_magnitude", "mask_coverage_ratio", "inference_time_ms"] if column in df.columns]
                if chart_cols:
                    render_metrics_plot(df, chart_cols)
        if analysis_summary:
            st.json(analysis_summary)
        image_cols = st.columns(2)
        for idx, image_path in enumerate(outputs.get("analysis_images", [])):
            render_image_file(image_path, caption=Path(image_path).name, container=image_cols[idx % 2])


def render_posthoc_analysis_results(bundle: Optional[Dict[str, Any]]) -> None:
    st.subheader("Post-Hoc Analysis")
    if not bundle:
        st.info("Create or select an analysis set to inspect merged results.")
        return

    metadata = bundle.get("metadata", {}) or {}
    manifest = bundle.get("run_manifest", []) or []
    analysis_summary = bundle.get("analysis_summary", {}) or {}
    st.caption(
        f"Set {bundle.get('set_id', '-')} | Clip {metadata.get('clip_id', '-')} | "
        f"Selected runs {metadata.get('selected_run_count', len(manifest))}"
    )

    tabs = st.tabs(["Summary", "Manifest", "Analytics"])
    with tabs[0]:
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Clip", metadata.get("clip_id", "-"))
        metrics_cols[1].metric("Selected runs", metadata.get("selected_run_count", len(manifest)))
        metrics_cols[2].metric("Eligible runs", analysis_summary.get("eligible_run_count", "-"))
        metrics_cols[3].metric("Skipped runs", analysis_summary.get("skipped_run_count", "-"))
        st.markdown("**Set Metadata**")
        st.json(metadata)
        st.markdown("**Analysis Summary**")
        st.json(analysis_summary)

    with tabs[1]:
        if manifest:
            rows = []
            for entry in manifest:
                rows.append({
                    "Run": entry.get("display_label") or entry.get("run_id"),
                    "Tracker": entry.get("tracker_variant_label") or entry.get("tracker_variant_key"),
                    "Audio": entry.get("audio_model_size") or "-",
                    "Precision": entry.get("effective_audio_precision") or entry.get("requested_audio_precision") or "-",
                    "Job": entry.get("job_id") or "-",
                    "Run ID": entry.get("run_id") or "-",
                })
            render_html_table(rows, columns=["Run", "Tracker", "Audio", "Precision", "Job", "Run ID"])
        else:
            st.info("No run manifest entries were found for this analysis set.")

    with tabs[2]:
        aggregate_csv = bundle.get("aggregate_csv")
        if aggregate_csv and Path(aggregate_csv).exists():
            df = load_metrics_preview(aggregate_csv, max_rows=500)
            if not df.empty:
                render_html_table(df, max_rows=500)
        image_cols = st.columns(2)
        for idx, image_path in enumerate(bundle.get("analysis_images", [])):
            render_image_file(image_path, caption=Path(image_path).name, container=image_cols[idx % 2])


def render_posthoc_analysis_panel() -> None:
    grouped_runs = list_posthoc_runs_by_clip()
    recent_set_ids = get_recent_analysis_set_ids(limit=20)

    if grouped_runs:
        clip_id = st.selectbox(
            "Clip for merged analysis",
            list(grouped_runs.keys()),
            key="posthoc_clip_selector",
            help="Only runs from the same clip can be merged into one post-hoc analysis set.",
        )
        clip_runs = grouped_runs.get(clip_id, [])
        option_map = {
            f"{entry.get('display_label') or entry.get('run_id')} [{entry.get('job_id') or 'manual'} / {entry.get('run_id')}]": entry
            for entry in clip_runs
        }
        selected_labels = st.multiselect(
            "Completed runs to merge",
            list(option_map.keys()),
            key="posthoc_run_selector",
            help="Select previously completed runs for the same clip, then build one merged analysis set without rerunning tracking or audio.",
        )
        if compat_button("Create post-hoc analysis set", disabled=len(selected_labels) == 0, use_container_width=True):
            try:
                result = create_posthoc_analysis_set([option_map[label] for label in selected_labels])
                st.session_state["selected_posthoc_set_id"] = result["set_id"]
                st.success(f"Created analysis set {result['set_id']}")
                rerun_app()
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("No eligible completed runs were found for post-hoc merging yet.")

    selected_set_id = st.session_state.get("selected_posthoc_set_id")
    available_set_ids = recent_set_ids
    if available_set_ids:
        default_index = 0
        if selected_set_id in available_set_ids:
            default_index = available_set_ids.index(selected_set_id)
        selected_set_id = st.selectbox(
            "Recent analysis sets",
            available_set_ids,
            index=default_index,
            key="posthoc_set_selector",
            help="Inspect a previously created merged analysis set without rerunning any model stages.",
        )
        st.session_state["selected_posthoc_set_id"] = selected_set_id
        render_posthoc_analysis_results(load_analysis_set(selected_set_id))
    else:
        render_posthoc_analysis_results(None)


def main() -> None:
    ensure_session_defaults()
    previous_job_status = st.session_state.get("observed_job_status")
    previous_job_id = st.session_state.get("observed_job_id")
    st.title("SAM Tracking + SAM-Audio Web UI")
    st.caption("Single-clip local UI for configuring, running, and reviewing the pipeline.")

    videos = list_input_videos()
    active_job = get_active_job_state()
    queue_state = get_queue_state()
    if active_job is None and queue_state.get("queued") and st.session_state.get("auto_start_queue_worker", False):
        start_queue_worker_if_needed()
        queue_state = get_queue_state()
        active_job = get_active_job_state()
    recent_job_ids = get_recent_job_ids(limit=20)

    header_cols = st.columns([1.5, 1.0])
    if videos:
        video_name = header_cols[0].selectbox("Input video", videos, index=0)
    else:
        video_name = None
        header_cols[0].info("No videos found in the input folder.")

    if recent_job_ids:
        selected_job_id = header_cols[1].selectbox(
            "Recent jobs",
            recent_job_ids,
            index=0,
            key="job_selector",
            help="Inspect a previous run without starting a new one. Helpful for comparing outputs and summaries after experiments finish.",
        )
    else:
        selected_job_id = None
        header_cols[1].info("No prior jobs yet.")

    job_state = None
    if active_job is not None:
        job_state = active_job
        st.session_state["last_job_id"] = active_job["job_id"]
    elif selected_job_id:
        job_state = load_job_state(selected_job_id)
    elif st.session_state.get("last_job_id"):
        job_state = load_job_state(st.session_state["last_job_id"])
    else:
        job_state = get_latest_job_state()

    current_job_id = job_state.get("job_id") if job_state else None
    current_job_status = job_state.get("status") if job_state else None
    if previous_job_status == "running" and current_job_status in {"completed", "failed"}:
        clear_preview_state()
    st.session_state["observed_job_id"] = current_job_id
    st.session_state["observed_job_status"] = current_job_status

    main_tab, outputs_tab, posthoc_tab = st.tabs(
        ["Main", "Outputs and Analytics", "Post-Hoc Analysis"]
    )

    with main_tab:
        left, center, right = st.columns([1.15, 1.35, 1.0])
        with left:
            render_left_panel(video_name, active_job)
        with center:
            render_center_panel(video_name)
        with right:
            render_right_panel(job_state)
        render_queue_panel(queue_state)

    with outputs_tab:
        render_results_panel(job_state)

    with posthoc_tab:
        render_posthoc_analysis_panel()

    if active_job is not None or queue_state.get("queued") or queue_state.get("worker_running"):
        time.sleep(2)
        rerun_app()


if __name__ == "__main__":
    main()
