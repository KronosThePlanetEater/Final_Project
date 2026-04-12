import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from path_layout import ensure_standard_directories, resolve_existing_path, resolve_ffmpeg_binary, resolve_input_path, resolve_output_path
from progress_utils import emit_progress


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def probe_video(video_path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    metadata = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }
    cap.release()
    return metadata


def run_ffmpeg(command: List[str]) -> None:
    resolved_command = [resolve_ffmpeg_binary(command[0]), *command[1:]] if command else command
    try:
        subprocess.run(resolved_command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for dataset preparation but was not found on PATH or in tools/ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed: {stderr.strip()}") from exc


def normalize_video(
    source_video: str,
    normalized_video: str,
    target_fps: Optional[float],
    target_width: Optional[int],
    target_height: Optional[int],
    ffmpeg_bin: str,
) -> None:
    vf_parts = []
    if target_width is not None and target_height is not None:
        vf_parts.append(f"scale={target_width}:{target_height}")
    elif target_width is not None or target_height is not None:
        raise ValueError("target_width and target_height must be provided together.")

    command = [ffmpeg_bin, "-y", "-i", source_video]
    if target_fps is not None:
        command.extend(["-r", str(target_fps)])
    if vf_parts:
        command.extend(["-vf", ",".join(vf_parts)])
    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", "-b:a", "192k", normalized_video])
    run_ffmpeg(command)


def extract_audio(source_video: str, audio_output: str, ffmpeg_bin: str) -> None:
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        source_video,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_output,
    ]
    run_ffmpeg(command)


def prepare_clip(
    video_path: str,
    output_root: str,
    clip_id: Optional[str] = None,
    target_fps: Optional[float] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    reference_audio: Optional[str] = None,
    ffmpeg_bin: str = "ffmpeg",
    progress_callback=None,
) -> Dict[str, str]:
    source = resolve_input_path(video_path).resolve()
    emit_progress(progress_callback, stage="dataset_prep", stage_progress=0.0, status="running", clip_id=clip_id or source.stem, message="Preparing clip inputs.")
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    clip_id = clip_id or source.stem
    clip_dir = resolve_output_path(output_root, "prepared").resolve() / clip_id
    video_dir = clip_dir / "video"
    audio_dir = clip_dir / "audio"
    metadata_dir = clip_dir / "metadata"
    ensure_dir(video_dir)
    ensure_dir(audio_dir)
    ensure_dir(metadata_dir)

    normalized_video_path = video_dir / "input_normalized.mp4"
    source_audio_path = audio_dir / "source.wav"
    metadata_path = metadata_dir / "clip_metadata.json"

    emit_progress(progress_callback, stage="dataset_prep", stage_progress=0.2, status="running", clip_id=clip_id, message="Normalizing video.")
    normalize_video(
        str(source),
        str(normalized_video_path),
        target_fps=target_fps,
        target_width=target_width,
        target_height=target_height,
        ffmpeg_bin=ffmpeg_bin,
    )
    emit_progress(progress_callback, stage="dataset_prep", stage_progress=0.7, status="running", clip_id=clip_id, message="Extracting source audio.")
    extract_audio(str(normalized_video_path), str(source_audio_path), ffmpeg_bin=ffmpeg_bin)

    metadata = probe_video(str(normalized_video_path))
    payload = {
        "clip_id": clip_id,
        "source_video_path": str(source),
        "prepared_video_path": str(normalized_video_path),
        "canonical_video_path": str(normalized_video_path),
        "audio_path": str(source_audio_path),
        "canonical_audio_path": str(source_audio_path),
        "reference_audio_path": str(resolve_existing_path(reference_audio)) if reference_audio else None,
        "fps": metadata["fps"],
        "width": metadata["width"],
        "height": metadata["height"],
        "frame_count": metadata["frame_count"],
        "target_fps": target_fps,
        "target_width": target_width,
        "target_height": target_height,
    }

    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    payload["metadata_path"] = str(metadata_path)
    payload["clip_dir"] = str(clip_dir)
    emit_progress(progress_callback, stage="dataset_prep", stage_progress=1.0, status="completed", clip_id=clip_id, message="Prepared clip assets.", outputs={"clip_dir": str(clip_dir), "metadata_path": str(metadata_path), "video_path": str(normalized_video_path), "audio_path": str(source_audio_path)})
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare videos and audio for tracking/audio experiments.")
    parser.add_argument("video_paths", nargs="+", help="One or more source videos to prepare.")
    parser.add_argument("--output-dir", default=None, help="Root directory for prepared clips. Defaults to output/prepared.")
    parser.add_argument("--target-fps", type=float, default=None, help="Normalize prepared video to this FPS.")
    parser.add_argument("--target-width", type=int, default=None, help="Normalize prepared video width.")
    parser.add_argument("--target-height", type=int, default=None, help="Normalize prepared video height.")
    parser.add_argument(
        "--reference-audio-map",
        type=str,
        default=None,
        help="Path to a JSON object mapping clip ids to clean reference audio paths.",
    )
    parser.add_argument("--ffmpeg-bin", type=str, default="ffmpeg", help="ffmpeg executable name or path.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    ensure_standard_directories()

    reference_audio_map = {}
    if args.reference_audio_map:
        reference_audio_map_path = resolve_existing_path(args.reference_audio_map)
        with open(reference_audio_map_path, "r", encoding="utf-8") as handle:
            reference_audio_map = json.load(handle)

    prepared_output_dir = resolve_output_path(args.output_dir, "prepared")

    prepared = []
    for video_path in args.video_paths:
        clip_id = Path(video_path).stem
        result = prepare_clip(
            video_path=video_path,
            output_root=str(prepared_output_dir),
            clip_id=clip_id,
            target_fps=args.target_fps,
            target_width=args.target_width,
            target_height=args.target_height,
            reference_audio=reference_audio_map.get(clip_id),
            ffmpeg_bin=args.ffmpeg_bin,
        )
        prepared.append(result)

    print(json.dumps(prepared, indent=2))
