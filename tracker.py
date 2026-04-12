import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from path_layout import ensure_standard_directories, resolve_existing_path, resolve_ffmpeg_binary, resolve_input_path, resolve_output_path
from progress_utils import emit_progress


SelectionTuple = Tuple[Tuple[Optional[np.ndarray], Optional[np.ndarray]], Optional[np.ndarray]]


def normalize_sam2_config_path(config_path: str) -> str:
    config_str = str(config_path).replace('\\', '/')
    candidate = Path(config_str)
    parts = [part for part in candidate.parts if part not in {'/', '.'}]

    if not candidate.is_absolute() and len(parts) >= 2 and parts[0] == 'configs':
        return '/'.join(parts)

    if 'configs' in parts:
        config_index = parts.index('configs')
        trailing = parts[config_index:]
        if len(trailing) >= 2:
            return '/'.join(trailing)

    name = candidate.name
    parent_name = candidate.parent.name
    if parent_name.startswith('sam2') and name.endswith('.yaml'):
        return f'configs/{parent_name}/{name}'
    if name.startswith('sam2.1_') and name.endswith('.yaml'):
        return f'configs/sam2.1/{name}'
    if name.startswith('sam2_') and name.endswith('.yaml'):
        return f'configs/sam2/{name}'
    return config_str


class ObjectSelector:
    """Interactive object selector using OpenCV GUI."""

    def __init__(self, frame, points=None, labels=None, bbox=None):
        self.frame = frame.copy()
        self.original_frame = frame.copy()
        self.points = points or []
        self.labels = labels or []
        self.bbox = bbox or []
        self.drawing = False
        self.mode = None

    def mouse_callback(self, event, x, y, flags, param):
        if self.mode == "point":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append([x, y])
                self.labels.append(1)
                cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("Select Object", self.frame)
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.points.append([x, y])
                self.labels.append(0)
                cv2.circle(self.frame, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Object", self.frame)
        elif self.mode == "bbox":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.bbox = [x, y, x, y]
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.frame = self.original_frame.copy()
                self.bbox[2] = x
                self.bbox[3] = y
                cv2.rectangle(
                    self.frame,
                    (self.bbox[0], self.bbox[1]),
                    (self.bbox[2], self.bbox[3]),
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Select Object", self.frame)
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.bbox[2] = x
                self.bbox[3] = y

    def select(self, mode="point"):
        self.mode = mode
        window_name = "Select Object"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        if mode == "point":
            instruction = "Left click on an object to track. Right click to exclude. Press ENTER when done,"
            instruction_2 = "R to reset, ESC to cancel."
        else:
            instruction = "Draw a bounding box around the object. Press ENTER when done, R to reset, ESC to cancel."
            instruction_2 = ""

        print(f"\n{instruction} {instruction_2}".strip())

        frame_with_text = self.frame.copy()
        cv2.putText(
            frame_with_text,
            instruction,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
        )
        if mode == "point":
            cv2.putText(
                frame_with_text,
                instruction_2,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )
        cv2.imshow(window_name, frame_with_text)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                break
            if key == ord("r"):
                self.frame = self.original_frame.copy()
                self.points = []
                self.labels = []
                self.bbox = []
                cv2.imshow(window_name, self.frame)
            elif key == 27:
                cv2.destroyAllWindows()
                return (None, None), None

        cv2.destroyAllWindows()

        if mode == "point":
            return (np.array(self.points, dtype=np.float32), np.array(self.labels, np.int32)), None

        if len(self.bbox) == 4:
            x1, y1, x2, y2 = self.bbox
            bbox = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            return (None, None), np.array(bbox)
        return (None, None), None


class PreviewHandler:
    """Handles all mask visualization and OpenCV window management."""

    def __init__(self, window_name="Tracking Preview"):
        self.window_name = window_name
        self.window_created = False

    @staticmethod
    def apply_mask_visuals(frame, mask, background_color=None):
        display_frame = frame.copy()
        if mask is None:
            mask = np.zeros(display_frame.shape[:2], dtype=bool)
        if len(mask.shape) == 3:
            mask = mask[0]
        if mask.shape != display_frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (display_frame.shape[1], display_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        if background_color is not None:
            output = np.full_like(display_frame, background_color, dtype=np.uint8)
            output[mask] = display_frame[mask]
            return output

        display_frame[mask] = display_frame[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
        return display_frame

    def _ensure_window(self, frame):
        if not self.window_created:
            cv2.namedWindow(self.window_name)
            h, w = frame.shape[:2]
            if w > 1280 or h > 720:
                cv2.resizeWindow(self.window_name, min(w, 1280), min(h, 720))
            self.window_created = True

    def show_interactive_mask(self, frame, mask):
        display_frame = self.apply_mask_visuals(frame, mask, None)
        instruction = "Press ENTER to confirm, ESC to reject."
        (text_w, text_h), _ = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(display_frame, (10, 10), (20 + text_w, 20 + text_h), (0, 0, 0), -1)
        cv2.putText(
            display_frame,
            instruction,
            (15, 15 + text_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        self._ensure_window(display_frame)
        cv2.imshow(self.window_name, display_frame)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 13:
                self.close()
                return True
            if key == 27:
                return False

    def update_live_preview(self, frame, mask, background_color=None, stats=None):
        display_frame = self.apply_mask_visuals(frame, mask, background_color)

        if stats:
            overlay = display_frame.copy()
            box_h = 20 + (len(stats) * 25)
            cv2.rectangle(overlay, (10, 10), (420, 10 + box_h), (0, 0, 0), -1)
            display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
            y = 35
            for key, (val, color) in stats.items():
                text = f"{key}: {val}"
                cv2.putText(
                    display_frame,
                    text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )
                y += 25

        self._ensure_window(display_frame)
        cv2.imshow(self.window_name, display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()
            return True
        return False

    def close(self):
        if self.window_created:
            cv2.destroyWindow(self.window_name)
            self.window_created = False


class VideoHandler:
    """Reads metadata, extracts frames, and writes preview video."""

    def __init__(self, video_path, frames_dir, output_path):
        self.video_path = video_path
        self.frames_dir = frames_dir
        self.output_path = output_path
        self.load_metadata()
        ensure_dir(self.frames_dir)

    def load_metadata(self):
        cap = cv2.VideoCapture(self.video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    def extract_frames(self, max_frames=None, start_time=0.0, scale=1.0):
        cap = cv2.VideoCapture(self.video_path)
        print(f"Loaded video from {self.video_path}")
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"{self.width}x{self.height} @ {self.fps:.3f} fps, {total_frames} total frames")

        start_frame = max(0, int(start_time * self.fps))
        if start_frame >= total_frames:
            print(
                f"Start time {start_time}s maps to frame {start_frame}, beyond video length ({total_frames} frames)."
            )
            cap.release()
            return

        if max_frames is None:
            end_frame = total_frames
            print(f"Extracting frames from {start_frame} to {end_frame - 1} (to end of video)")
        else:
            end_frame = min(total_frames, start_frame + int(max_frames))
            print(f"Extracting frames {start_frame}..{end_frame - 1} ({end_frame - start_frame} frames)")

        if scale != 1.0:
            self.width = int(self.width * scale)
            self.height = int(self.height * scale)
            print(f"Resizing frames to {self.width}x{self.height} (scale {scale}x)")

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for source_frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            if scale != 1.0:
                frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
            frame_filename = os.path.join(self.frames_dir, f"{source_frame_idx:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
        cap.release()

    def clean_frames(self):
        for file_name in os.listdir(self.frames_dir):
            file_path = os.path.join(self.frames_dir, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def write_video(self, frame_names, mask_stack, background_color):
        ensure_dir(Path(self.output_path).parent)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))

        for frame_idx, frame_name in tqdm(
            enumerate(frame_names), total=len(frame_names), desc="Writing frames"
        ):
            frame = cv2.imread(os.path.join(self.frames_dir, frame_name))
            mask = mask_stack[frame_idx] if frame_idx < len(mask_stack) else None
            if mask is not None:
                frame = PreviewHandler.apply_mask_visuals(frame, mask, background_color)
            elif background_color is not None:
                frame = np.full_like(frame, background_color, dtype=np.uint8)
            out.write(frame)
        out.release()


def ensure_dir(path: Any) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def run_ffmpeg(command: List[str]) -> None:
    resolved_command = [resolve_ffmpeg_binary(command[0]), *command[1:]] if command else command
    try:
        subprocess.run(resolved_command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for browser-compatible video export but was not found on PATH or in tools/ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed during tracking video export: {stderr.strip()}") from exc


def transcode_video_for_browser(input_path: str, output_path: str, ffmpeg_bin: str = "ffmpeg") -> str:
    temp_output = str(Path(output_path).with_suffix(".browser.mp4"))
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        temp_output,
    ]
    run_ffmpeg(command)
    os.replace(temp_output, output_path)
    return output_path


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("Enabled TF32 mode for matmul and cudnn")
    return device


def initialize_video_predictor(config_path, checkpoint_path, device, frames_dir):
    predictor = build_sam2_video_predictor(normalize_sam2_config_path(config_path), checkpoint_path, device)
    inference_state = predictor.init_state(video_path=frames_dir)
    return predictor, inference_state


def list_frame_names(frames_dir: str) -> List[str]:
    frame_names = [name for name in os.listdir(frames_dir) if name.lower().endswith(".jpg")]
    frame_names.sort(key=lambda name: int(Path(name).stem))
    return frame_names


def mask_to_bool(mask: Any, frame_shape: Tuple[int, int]) -> np.ndarray:
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    is_float_mask = np.issubdtype(mask.dtype, np.floating)
    resize_source = mask.astype(np.float32) if is_float_mask else mask.astype(np.uint8)
    if resize_source.shape != frame_shape:
        resize_source = cv2.resize(
            resize_source,
            (frame_shape[1], frame_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    if is_float_mask:
        return resize_source > 0.0
    return resize_source.astype(bool)


def get_box_from_mask(mask, padding=10):
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return None

    h, w = mask.shape[:2]
    x_min, x_max = x_indices.min(), x_indices.max()
    y_min, y_max = y_indices.min(), y_indices.max()
    return np.array(
        [
            max(0, x_min - padding),
            max(0, y_min - padding),
            min(w - 1, x_max + padding),
            min(h - 1, y_max + padding),
        ]
    )


def bbox_fields_from_mask(mask: np.ndarray) -> Dict[str, Any]:
    bbox = get_box_from_mask(mask, padding=0)
    if bbox is None:
        return {
            "bbox_x1": "",
            "bbox_y1": "",
            "bbox_x2": "",
            "bbox_y2": "",
            "bbox_width": "",
            "bbox_height": "",
        }
    x1, y1, x2, y2 = [int(value) for value in bbox]
    return {
        "bbox_x1": x1,
        "bbox_y1": y1,
        "bbox_x2": x2,
        "bbox_y2": y2,
        "bbox_width": max(0, x2 - x1 + 1),
        "bbox_height": max(0, y2 - y1 + 1),
    }


def selection_from_dict(selection_data: Optional[Dict[str, Any]]) -> Optional[SelectionTuple]:
    if selection_data is None:
        return None

    bbox = selection_data.get("bbox")
    if bbox is not None:
        return (None, None), np.array(bbox, dtype=np.float32)

    points = selection_data.get("points")
    labels = selection_data.get("labels")
    if points is None or labels is None:
        raise ValueError("Selection JSON must contain either bbox or points+labels.")

    return (
        np.array(points, dtype=np.float32),
        np.array(labels, dtype=np.int32),
    ), None


def load_selection_from_file(selection_json_path: str) -> SelectionTuple:
    with open(selection_json_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return selection_from_dict(payload)


def selection_to_dict(selection_mode: str, initial_selection: SelectionTuple) -> Dict[str, Any]:
    (points, labels), bbox = initial_selection
    payload = {"mode": selection_mode}
    if bbox is not None:
        payload["bbox"] = np.asarray(bbox).astype(float).tolist()
    else:
        payload["points"] = np.asarray(points).astype(float).tolist() if points is not None else []
        payload["labels"] = np.asarray(labels).astype(int).tolist() if labels is not None else []
    return payload


def default_artifacts_dir(output_path: str) -> str:
    output = Path(output_path)
    return str(output.parent / f"{output.stem}_artifacts")


def build_artifact_paths(artifacts_dir: str) -> Dict[str, str]:
    artifacts = Path(artifacts_dir)
    masks_dir = artifacts / "masks"
    metrics_dir = artifacts / "metrics"
    ensure_dir(masks_dir)
    ensure_dir(metrics_dir)
    return {
        "artifacts_dir": str(artifacts),
        "mask_stack_path": str(masks_dir / "masks.npz"),
        "mask_png_dir": str(masks_dir / "png"),
        "metrics_csv_path": str(metrics_dir / "frame_metrics.csv"),
        "summary_json_path": str(metrics_dir / "tracking_summary.json"),
    }


METRIC_FIELDNAMES = [
    "frame_idx",
    "timestamp_sec",
    "frame_source",
    "is_keyframe",
    "motion_magnitude",
    "mask_area_px",
    "mask_coverage_ratio",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "bbox_width",
    "bbox_height",
    "inference_time_ms",
    "flow_valid",
    "mask_empty",
    "tracking_failed",
    "object_missing",
]


def append_metrics_row(
    rows: List[Dict[str, Any]],
    frame_idx: int,
    fps: float,
    start_time_seconds: float,
    frame_source: str,
    is_keyframe: bool,
    motion_magnitude: float,
    mask: np.ndarray,
    inference_time_ms: float,
    flow_valid: bool,
    tracking_failed: bool,
    consecutive_empty: int,
) -> Tuple[Dict[str, Any], int]:
    mask_area = int(mask.sum())
    mask_empty = mask_area == 0
    consecutive_empty = consecutive_empty + 1 if mask_empty else 0
    object_missing = consecutive_empty >= 2
    row = {
        "frame_idx": frame_idx,
        "timestamp_sec": round(start_time_seconds + (frame_idx / fps if fps else 0.0), 6),
        "frame_source": frame_source,
        "is_keyframe": bool(is_keyframe),
        "motion_magnitude": round(float(motion_magnitude), 6),
        "mask_area_px": mask_area,
        "mask_coverage_ratio": round(mask_area / mask.size, 8) if mask.size else 0.0,
        "inference_time_ms": round(float(inference_time_ms), 6),
        "flow_valid": bool(flow_valid),
        "mask_empty": bool(mask_empty),
        "tracking_failed": bool(tracking_failed),
        "object_missing": bool(object_missing),
    }
    row.update(bbox_fields_from_mask(mask))
    rows.append(row)
    return row, consecutive_empty


def write_metrics_csv(metrics_rows: List[Dict[str, Any]], metrics_csv_path: str) -> None:
    ensure_dir(Path(metrics_csv_path).parent)
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)


def save_mask_outputs(
    mask_stack: List[np.ndarray],
    frame_names: List[str],
    mask_stack_path: str,
    save_mask_pngs: bool = False,
    mask_png_dir: Optional[str] = None,
) -> None:
    if not mask_stack:
        raise ValueError("No masks were generated; cannot save mask outputs.")

    mask_array = np.stack(mask_stack, axis=0).astype(bool)
    ensure_dir(Path(mask_stack_path).parent)
    np.savez_compressed(
        mask_stack_path,
        masks=mask_array,
        frame_names=np.array(frame_names),
        frame_indices=np.arange(mask_array.shape[0], dtype=np.int32),
    )

    if save_mask_pngs:
        if mask_png_dir is None:
            raise ValueError("mask_png_dir must be provided when save_mask_pngs=True.")
        ensure_dir(mask_png_dir)
        for frame_idx, mask in enumerate(mask_stack):
            png_path = Path(mask_png_dir) / f"{frame_idx:05d}.png"
            cv2.imwrite(str(png_path), (mask.astype(np.uint8) * 255))


def build_summary(
    video_path: str,
    output_path: str,
    frame_names: List[str],
    metrics_rows: List[Dict[str, Any]],
    video_handler: VideoHandler,
    pure_duration: float,
    sam_frames: int,
    flow_frames: int,
    artifacts: Dict[str, str],
    selection_mode: str,
    initial_selection: SelectionTuple,
    dynamic_interval: Optional[Tuple[int, int]],
    sam_interval: int,
    start_time_seconds: float,
    background_color: Optional[Tuple[int, int, int]],
) -> Dict[str, Any]:
    empty_mask_count = sum(1 for row in metrics_rows if row["mask_empty"])
    failure_count = sum(1 for row in metrics_rows if row["tracking_failed"])
    object_missing_count = sum(1 for row in metrics_rows if row["object_missing"])
    mean_motion = float(np.mean([row["motion_magnitude"] for row in metrics_rows])) if metrics_rows else 0.0
    mean_mask_coverage = (
        float(np.mean([row["mask_coverage_ratio"] for row in metrics_rows])) if metrics_rows else 0.0
    )
    pure_fps = (len(frame_names) / pure_duration) if pure_duration > 0 else 0.0

    return {
        "clip_id": Path(output_path).stem,
        "video_path": os.path.abspath(video_path),
        "output_video_path": os.path.abspath(output_path),
        "frames_dir": os.path.abspath(video_handler.frames_dir),
        "artifacts_dir": os.path.abspath(artifacts["artifacts_dir"]),
        "metrics_csv_path": os.path.abspath(artifacts["metrics_csv_path"]),
        "summary_json_path": os.path.abspath(artifacts["summary_json_path"]),
        "mask_stack_path": os.path.abspath(artifacts["mask_stack_path"]),
        "mask_png_dir": os.path.abspath(artifacts["mask_png_dir"]) if os.path.isdir(artifacts["mask_png_dir"]) else None,
        "fps": float(video_handler.fps),
        "width": int(video_handler.width),
        "height": int(video_handler.height),
        "total_frames": len(frame_names),
        "sam_frames": sam_frames,
        "flow_frames": flow_frames,
        "pure_time": pure_duration,
        "pure_fps": pure_fps,
        "empty_mask_count": empty_mask_count,
        "failure_count": failure_count,
        "object_missing_count": object_missing_count,
        "mean_motion": mean_motion,
        "mean_mask_coverage": mean_mask_coverage,
        "selection_mode": selection_mode,
        "initial_selection": selection_to_dict(selection_mode, initial_selection),
        "sam_interval": sam_interval,
        "dynamic_interval": list(dynamic_interval) if dynamic_interval else None,
        "start_time_seconds": start_time_seconds,
        "background_color_bgr": list(background_color) if background_color is not None else None,
    }


def write_summary_json(summary: Dict[str, Any], summary_json_path: str) -> None:
    ensure_dir(Path(summary_json_path).parent)
    with open(summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def emit_tracking_progress(
    progress_callback,
    frame_idx: int,
    total_frames: int,
    metrics_rows: List[Dict[str, Any]],
    sam_frames: int,
    flow_frames: int,
    frame_source: str,
    message: Optional[str] = None,
) -> None:
    if total_frames <= 0:
        stage_progress = 0.0
    else:
        stage_progress = min(1.0, max(0.0, (frame_idx + 1) / total_frames))

    failure_count = sum(1 for row in metrics_rows if row["tracking_failed"])
    empty_mask_count = sum(1 for row in metrics_rows if row["mask_empty"])
    mean_motion = float(np.mean([row["motion_magnitude"] for row in metrics_rows])) if metrics_rows else 0.0
    last_row = metrics_rows[-1] if metrics_rows else {}

    emit_progress(
        progress_callback,
        stage="tracking",
        stage_progress=stage_progress,
        status="running" if stage_progress < 1.0 else "completed",
        current_frame=frame_idx + 1,
        total_frames=total_frames,
        message=message or f"Processed frame {frame_idx + 1} of {total_frames}.",
        stats={
            "sam_frames": sam_frames,
            "flow_frames": flow_frames,
            "failure_count": failure_count,
            "empty_mask_count": empty_mask_count,
            "mean_motion": mean_motion,
            "frame_source": frame_source,
            "tracking_fps": last_row.get("inference_time_ms"),
        },
    )


lk_params = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
feature_params = dict(maxCorners=200, qualityLevel=0.01, minDistance=5, blockSize=7)


def calculate_optical_flow(prev_gray, curr_gray, prev_mask):
    if np.sum(prev_mask) == 0:
        return 0.0, None, False

    detection_mask = cv2.dilate(prev_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=detection_mask, **feature_params)
    if p0 is None or len(p0) == 0:
        return 0.0, None, False

    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, **lk_params)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    if len(good_new) == 0:
        return 0.0, None, False

    dists = np.linalg.norm(good_new - good_old, axis=1)
    motion = np.mean(dists)
    matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new)
    return motion, matrix, (matrix is not None)


def request_initial_selection(
    frame_path: str,
    selection_mode: str,
    initial_selection: Optional[SelectionTuple],
) -> SelectionTuple:
    if initial_selection is not None:
        print("Using predefined initial selection.")
        return initial_selection

    selector = ObjectSelector(cv2.imread(frame_path))
    (points, labels), bbox = selector.select(mode=selection_mode)
    if points is None and bbox is None:
        raise RuntimeError("Object selection cancelled.")
    return (points, labels), bbox


def track_object(
    video_path,
    output_path,
    checkpoint_path,
    config_path,
    frames_dir,
    extract_frames=False,
    selection_mode="point",
    max_frames=None,
    start_time=0.0,
    scale=1.0,
    initial_selection=None,
    skip_mask_confirmation=False,
    sam_interval=1,
    dynamic_interval=None,
    background_color=None,
    show_preview=False,
    artifacts_dir=None,
    save_mask_pngs=False,
    ffmpeg_bin="ffmpeg",
    progress_callback=None,
):
    """
    Track an object in a video using SAM and optical flow.

    Returns a summary dictionary with output paths and aggregate statistics.
    """

    emit_progress(progress_callback, stage="tracking", stage_progress=0.0, status="running", message="Initializing tracker.")
    device = setup_device()
    if sam_interval and not dynamic_interval:
        dynamic_interval = (sam_interval, sam_interval)

    video_handler = VideoHandler(video_path, frames_dir, output_path)
    if extract_frames:
        video_handler.clean_frames()
        video_handler.extract_frames(max_frames=max_frames, start_time=start_time, scale=scale)

    emit_progress(progress_callback, stage="tracking", stage_progress=0.05, status="running", message="Loading extracted frames.")
    frame_names = list_frame_names(frames_dir)
    if not frame_names:
        raise RuntimeError(f"No extracted frames found in {frames_dir}.")

    first_frame_path = os.path.join(frames_dir, frame_names[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame at {first_frame_path}.")
    frame_shape = first_frame.shape[:2]

    preview_handler = PreviewHandler()
    artifacts_dir = artifacts_dir or default_artifacts_dir(output_path)
    artifacts = build_artifact_paths(artifacts_dir)

    emit_progress(progress_callback, stage="tracking", stage_progress=0.08, status="running", message="Preparing initial object selection.")
    initial_selection = request_initial_selection(first_frame_path, selection_mode, initial_selection)
    (points, labels), bbox = initial_selection

    metrics_rows: List[Dict[str, Any]] = []
    mask_stack: List[Optional[np.ndarray]] = [None] * len(frame_names)
    consecutive_empty = 0
    previous_valid_mask: Optional[np.ndarray] = None
    sam_frames = 0
    flow_frames = 0
    initial_processing_seconds = 0.0

    if dynamic_interval == (1, 1):
        predictor, inference_state = initialize_video_predictor(config_path, checkpoint_path, device, frames_dir)
        preview_accepted = False
        accepted_mask = None
        accepted_inference_ms = 0.0

        while not preview_accepted:
            inference_start = time.time()
            if bbox is None:
                _, _, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    points=points,
                    labels=labels,
                )
            else:
                _, _, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    box=bbox,
                )
            accepted_inference_ms = (time.time() - inference_start) * 1000.0
            mask = mask_to_bool(out_mask_logits[0], frame_shape)

            if skip_mask_confirmation:
                preview_accepted = True
                print("Skipping mask confirmation.")
            else:
                preview_accepted = preview_handler.show_interactive_mask(first_frame, mask)
                if not preview_accepted:
                    (points, labels), bbox = request_initial_selection(
                        first_frame_path, selection_mode, initial_selection=None
                    )
            accepted_mask = mask

        initial_processing_seconds = accepted_inference_ms / 1000.0
        final_initial_mask = accepted_mask.copy()
        initial_failed = False
        if not final_initial_mask.any():
            initial_failed = True
        else:
            previous_valid_mask = final_initial_mask.copy()

        mask_stack[0] = final_initial_mask
        sam_frames += 1
        _, consecutive_empty = append_metrics_row(
            metrics_rows,
            frame_idx=0,
            fps=video_handler.fps,
            start_time_seconds=start_time,
            frame_source="sam",
            is_keyframe=True,
            motion_magnitude=0.0,
            mask=final_initial_mask,
            inference_time_ms=accepted_inference_ms,
            flow_valid=False,
            tracking_failed=initial_failed,
            consecutive_empty=consecutive_empty,
        )
        emit_tracking_progress(progress_callback, 0, len(frame_names), metrics_rows, sam_frames, flow_frames, "sam", "Tracker initialized with first frame.")

        print("Starting SAM processing loop...")
        processing_start_time = time.time()
        propagation = predictor.propagate_in_video(inference_state)

        while True:
            try:
                frame_start = time.time()
                out_frame_idx, out_obj_ids, out_mask_logits = next(propagation)
                inference_ms = (time.time() - frame_start) * 1000.0
            except StopIteration:
                break

            if out_frame_idx == 0:
                continue

            if out_mask_logits is None or len(out_mask_logits) == 0:
                predicted_mask = np.zeros(frame_shape, dtype=bool)
            else:
                predicted_mask = mask_to_bool(out_mask_logits[0], frame_shape)

            tracking_failed = False
            final_mask = predicted_mask
            if not predicted_mask.any():
                tracking_failed = True
                if previous_valid_mask is not None:
                    final_mask = previous_valid_mask.copy()
            else:
                previous_valid_mask = predicted_mask.copy()

            mask_stack[out_frame_idx] = final_mask
            sam_frames += 1
            _, consecutive_empty = append_metrics_row(
                metrics_rows,
                frame_idx=out_frame_idx,
                fps=video_handler.fps,
                start_time_seconds=start_time,
                frame_source="sam",
                is_keyframe=True,
                motion_magnitude=0.0,
                mask=final_mask,
                inference_time_ms=inference_ms,
                flow_valid=False,
                tracking_failed=tracking_failed,
                consecutive_empty=consecutive_empty,
            )
            emit_tracking_progress(progress_callback, out_frame_idx, len(frame_names), metrics_rows, sam_frames, flow_frames, "sam")

            if show_preview:
                current_frame = cv2.imread(os.path.join(frames_dir, frame_names[out_frame_idx]))
                stats = {
                    "Frame": (f"{out_frame_idx}/{len(frame_names) - 1}", (255, 255, 255)),
                    "Type": ("SAM", (0, 255, 0)),
                    "Failed": (str(tracking_failed), (0, 165, 255) if tracking_failed else (255, 255, 255)),
                }
                should_quit = preview_handler.update_live_preview(
                    current_frame,
                    final_mask,
                    background_color,
                    stats,
                )
                if should_quit:
                    show_preview = False

        processing_end_time = time.time()
    else:
        sam2_model = build_sam2(normalize_sam2_config_path(config_path), checkpoint_path, device)
        predictor = SAM2ImagePredictor(sam2_model)
        min_interval, max_interval = dynamic_interval
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        predictor.set_image(frame_rgb)
        preview_accepted = False
        accepted_mask = None
        accepted_inference_ms = 0.0

        while not preview_accepted:
            inference_start = time.time()
            if bbox is None:
                masks, scores, logits = predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    multimask_output=False,
                )
            else:
                masks, scores, logits = predictor.predict(box=bbox, multimask_output=False)
            accepted_inference_ms = (time.time() - inference_start) * 1000.0
            mask = mask_to_bool(masks[0], frame_shape)
            if skip_mask_confirmation:
                preview_accepted = True
                print("Skipping mask confirmation.")
            else:
                preview_accepted = preview_handler.show_interactive_mask(first_frame, mask)
                if not preview_accepted:
                    (points, labels), bbox = request_initial_selection(
                        first_frame_path, selection_mode, initial_selection=None
                    )
                    predictor.set_image(frame_rgb)
            accepted_mask = mask

        initial_processing_seconds = accepted_inference_ms / 1000.0
        initial_failed = False
        final_initial_mask = accepted_mask.copy()
        if not final_initial_mask.any():
            initial_failed = True
        else:
            previous_valid_mask = final_initial_mask.copy()

        mask_stack[0] = final_initial_mask
        sam_frames += 1
        _, consecutive_empty = append_metrics_row(
            metrics_rows,
            frame_idx=0,
            fps=video_handler.fps,
            start_time_seconds=start_time,
            frame_source="sam",
            is_keyframe=True,
            motion_magnitude=0.0,
            mask=final_initial_mask,
            inference_time_ms=accepted_inference_ms,
            flow_valid=False,
            tracking_failed=initial_failed,
            consecutive_empty=consecutive_empty,
        )
        emit_tracking_progress(progress_callback, 0, len(frame_names), metrics_rows, sam_frames, flow_frames, "sam", "Tracker initialized with first frame.")

        prev_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        frames_since_keyframe = 0
        is_fixed_interval = min_interval == max_interval

        print("Starting processing loop with adaptive SAM/flow scheduling...")
        processing_start_time = time.time()

        for frame_idx in tqdm(
            range(1, len(frame_names)),
            total=len(frame_names) - 1,
            desc="Processing frames",
        ):
            frame_start = time.time()
            current_frame = cv2.imread(os.path.join(frames_dir, frame_names[frame_idx]))
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            previous_frame_mask = (
                mask_stack[frame_idx - 1].copy()
                if mask_stack[frame_idx - 1] is not None
                else np.zeros(frame_shape, dtype=bool)
            )

            frames_since_keyframe += 1
            forced_keyframe = frames_since_keyframe >= max_interval
            needs_motion_check = (not is_fixed_interval) and (not forced_keyframe)

            mean_motion = 0.0
            affine_matrix = None
            flow_valid = False
            flow_calculated = False
            adaptive_interval = min_interval

            if needs_motion_check:
                mean_motion, affine_matrix, flow_valid = calculate_optical_flow(
                    prev_frame_gray,
                    current_gray,
                    previous_frame_mask,
                )
                flow_calculated = True
                adaptive_interval = int(
                    np.clip(
                        max_interval - (mean_motion / 5.0) * (max_interval - min_interval),
                        min_interval,
                        max_interval,
                    )
                )
            elif is_fixed_interval:
                adaptive_interval = min_interval
            else:
                adaptive_interval = max_interval

            run_sam = (frames_since_keyframe >= adaptive_interval) or forced_keyframe
            tracking_failed = False

            if run_sam:
                frame_source = "sam"
                is_keyframe = True
                prompt_box = get_box_from_mask(previous_frame_mask, padding=10)
                if prompt_box is not None:
                    frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    predictor.set_image(frame_rgb)
                    masks, scores, logits = predictor.predict(box=prompt_box, multimask_output=False)
                    predicted_mask = mask_to_bool(masks[0], frame_shape)
                else:
                    predicted_mask = np.zeros(frame_shape, dtype=bool)

                if predicted_mask.any():
                    final_mask = predicted_mask
                    previous_valid_mask = predicted_mask.copy()
                    frames_since_keyframe = 0
                else:
                    tracking_failed = True
                    if previous_valid_mask is not None:
                        final_mask = previous_valid_mask.copy()
                    else:
                        final_mask = predicted_mask
                sam_frames += 1
            else:
                frame_source = "flow"
                is_keyframe = False
                if not flow_calculated:
                    mean_motion, affine_matrix, flow_valid = calculate_optical_flow(
                        prev_frame_gray,
                        current_gray,
                        previous_frame_mask,
                    )

                if flow_valid and previous_frame_mask.any():
                    warped_mask = cv2.warpAffine(
                        previous_frame_mask.astype(np.uint8),
                        affine_matrix,
                        (previous_frame_mask.shape[1], previous_frame_mask.shape[0]),
                        flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0,
                    )
                    final_mask = binary_dilation(warped_mask.astype(bool), iterations=1)
                else:
                    final_mask = previous_frame_mask.copy()

                if final_mask.any():
                    previous_valid_mask = final_mask.copy()
                else:
                    tracking_failed = True
                flow_frames += 1

            inference_time_ms = (time.time() - frame_start) * 1000.0
            mask_stack[frame_idx] = final_mask
            _, consecutive_empty = append_metrics_row(
                metrics_rows,
                frame_idx=frame_idx,
                fps=video_handler.fps,
                start_time_seconds=start_time,
                frame_source=frame_source,
                is_keyframe=is_keyframe,
                motion_magnitude=mean_motion,
                mask=final_mask,
                inference_time_ms=inference_time_ms,
                flow_valid=flow_valid,
                tracking_failed=tracking_failed,
                consecutive_empty=consecutive_empty,
            )
            emit_tracking_progress(progress_callback, frame_idx, len(frame_names), metrics_rows, sam_frames, flow_frames, frame_source)
            prev_frame_gray = current_gray.copy()

            if show_preview:
                status_text = "KEYFRAME (SAM)" if is_keyframe else "INTERPOLATED"
                status_color = (0, 255, 0) if is_keyframe else (0, 165, 255)
                stats = {
                    "Frame": (f"{frame_idx}/{len(frame_names) - 1}", (255, 255, 255)),
                    "Motion": (f"{mean_motion:.2f}", (255, 255, 255)),
                    "Interval": (f"{adaptive_interval}", (255, 255, 255)),
                    "Type": (status_text, status_color),
                    "Failed": (str(tracking_failed), (0, 165, 255) if tracking_failed else (255, 255, 255)),
                }
                should_quit = preview_handler.update_live_preview(
                    current_frame,
                    final_mask,
                    background_color,
                    stats,
                )
                if should_quit:
                    show_preview = False

        processing_end_time = time.time()
        print(f"Total SAM frames: {sam_frames}, Total Optical Flow frames: {flow_frames}")

    preview_handler.close()

    finalized_mask_stack = []
    fallback_mask = np.zeros(frame_shape, dtype=bool)
    for mask in mask_stack:
        if mask is None:
            if previous_valid_mask is not None:
                mask = previous_valid_mask.copy()
            else:
                mask = fallback_mask.copy()
        finalized_mask_stack.append(mask.astype(bool))

    emit_progress(progress_callback, stage="tracking", stage_progress=0.97, status="running", message="Writing tracked preview video and artifacts.")
    print(f"Saving video to {output_path}")
    video_handler.write_video(frame_names, finalized_mask_stack, background_color)
    transcode_video_for_browser(output_path, output_path, ffmpeg_bin=ffmpeg_bin)
    print("Video saved.")

    save_mask_outputs(
        finalized_mask_stack,
        frame_names,
        artifacts["mask_stack_path"],
        save_mask_pngs=save_mask_pngs,
        mask_png_dir=artifacts["mask_png_dir"],
    )
    write_metrics_csv(metrics_rows, artifacts["metrics_csv_path"])

    pure_duration = max(0.0, (processing_end_time - processing_start_time) + initial_processing_seconds)
    summary = build_summary(
        video_path=video_path,
        output_path=output_path,
        frame_names=frame_names,
        metrics_rows=metrics_rows,
        video_handler=video_handler,
        pure_duration=pure_duration,
        sam_frames=sam_frames,
        flow_frames=flow_frames,
        artifacts=artifacts,
        selection_mode=selection_mode,
        initial_selection=((points, labels), bbox),
        dynamic_interval=dynamic_interval,
        sam_interval=sam_interval,
        start_time_seconds=start_time,
        background_color=background_color,
    )
    write_summary_json(summary, artifacts["summary_json_path"])
    emit_progress(progress_callback, stage="tracking", stage_progress=1.0, status="completed", message="Tracking complete.", outputs={"output_video_path": summary["output_video_path"], "metrics_csv_path": summary["metrics_csv_path"], "summary_json_path": summary["summary_json_path"], "mask_stack_path": summary["mask_stack_path"], "mask_png_dir": summary.get("mask_png_dir")}, stats={"total_frames": summary["total_frames"], "sam_frames": summary["sam_frames"], "flow_frames": summary["flow_frames"], "pure_fps": summary["pure_fps"], "failure_count": summary["failure_count"], "empty_mask_count": summary["empty_mask_count"], "mean_motion": summary["mean_motion"]})
    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Track objects in video using SAM and optical flow")
    parser.add_argument("video_path", type=str, help="Path to input video file. Relative paths are resolved from the input folder.")
    parser.add_argument(
        "output_path",
        nargs="?",
        default=None,
        type=str,
        help="Path to save output video. Defaults to output/tracker/<clip_id>/tracked.mp4.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/sam2.1_hiera_tiny.pt",
        type=str,
        help="Path to SAM checkpoint (.pt)",
    )
    parser.add_argument(
        "--config",
        default="configs/sam2.1/sam2.1_hiera_t.yaml",
        type=str,
        help="Path to SAM config (.yaml)",
    )
    parser.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Directory to store extracted frames. Defaults to output/tracker/<clip_id>/frames.",
    )
    parser.add_argument("--no-extract", action="store_true", help="Skip frame extraction and reuse existing frames")
    parser.add_argument("--mode", choices=["point", "bbox"], default="point", help="Selection mode")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start processing from timestamp (seconds)")
    parser.add_argument("--scale", type=float, default=1.0, help="Resize frames (e.g. 0.5 for half size)")
    parser.add_argument("--sam-interval", type=int, default=1, help="Run SAM every N frames (static mode)")
    parser.add_argument(
        "--dynamic-interval",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Use motion-adaptive interval (e.g. --dynamic-interval 2 30). Overrides sam-interval.",
    )
    parser.add_argument(
        "--bg-color",
        type=int,
        nargs=3,
        metavar=("B", "G", "R"),
        help="Replace background with solid color (omit to keep original video).",
    )
    parser.add_argument("--preview", action="store_true", help="Show live preview window during processing")
    parser.add_argument(
        "--selection-json",
        type=str,
        help="Path to a JSON file containing either bbox or points+labels for non-interactive selection.",
    )
    parser.add_argument(
        "--skip-mask-confirmation",
        action="store_true",
        help="Skip the interactive mask confirmation window.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Directory where masks, metrics, and summary JSON will be stored.",
    )
    parser.add_argument(
        "--save-mask-pngs",
        action="store_true",
        help="Also save per-frame binary mask PNGs for debugging.",
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_standard_directories()

    resolved_video_path = resolve_input_path(args.video_path)
    resolved_output_path = resolve_output_path(args.output_path, f"tracker/{resolved_video_path.stem}/tracked.mp4")
    resolved_frames_dir = resolve_output_path(args.frames_dir, f"tracker/{resolved_video_path.stem}/frames")
    resolved_checkpoint_path = resolve_existing_path(args.checkpoint)
    resolved_config_path = resolve_existing_path(args.config)
    resolved_selection_json = resolve_existing_path(args.selection_json) if args.selection_json else None
    resolved_artifacts_dir = resolve_output_path(args.artifacts_dir, f"tracker/{resolved_video_path.stem}/artifacts")

    do_extract = not args.no_extract
    bg_color = tuple(args.bg_color) if args.bg_color else None
    dyn_interval = tuple(args.dynamic_interval) if args.dynamic_interval else None
    selection = load_selection_from_file(str(resolved_selection_json)) if resolved_selection_json else None

    try:
        result = track_object(
            video_path=str(resolved_video_path),
            output_path=str(resolved_output_path),
            checkpoint_path=str(resolved_checkpoint_path),
            config_path=str(resolved_config_path),
            frames_dir=str(resolved_frames_dir),
            extract_frames=do_extract,
            selection_mode=args.mode,
            max_frames=args.max_frames,
            start_time=args.start_time,
            scale=args.scale,
            initial_selection=selection,
            skip_mask_confirmation=args.skip_mask_confirmation,
            sam_interval=args.sam_interval,
            dynamic_interval=dyn_interval,
            background_color=bg_color,
            show_preview=args.preview,
            artifacts_dir=str(resolved_artifacts_dir),
            save_mask_pngs=args.save_mask_pngs,
            ffmpeg_bin=args.ffmpeg_bin,
        )
        print(json.dumps(result, indent=2))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\nError: {exc}")
        sys.exit(1)
