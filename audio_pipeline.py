import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from path_layout import INPUT_ROOT, OUTPUT_ROOT, PLACEHOLDER_ROOT, PROJECT_ROOT, ensure_standard_directories, resolve_existing_path, resolve_ffmpeg_binary, resolve_output_path
from progress_utils import emit_progress


VALID_MODEL_SIZES = {"small", "base", "large", "small-tv", "base-tv", "large-tv"}
MODEL_SIZE_TO_ID = {
    "small": "facebook/sam-audio-small",
    "base": "facebook/sam-audio-base",
    "large": "facebook/sam-audio-large",
    "small-tv": "facebook/sam-audio-small-tv",
    "base-tv": "facebook/sam-audio-base-tv",
    "large-tv": "facebook/sam-audio-large-tv",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cleanup_inference_memory() -> Dict[str, Any]:
    stats: Dict[str, Any] = {"gc_collected": int(gc.collect()), "cuda_cache_cleared": False}
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            stats["cuda_cache_cleared"] = True
    except Exception as exc:
        stats["cleanup_error"] = str(exc)
    return stats


def run_ffmpeg(command: List[str]) -> None:
    resolved_command = [resolve_ffmpeg_binary(command[0]), *command[1:]] if command else command
    try:
        subprocess.run(resolved_command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for SAM-Audio video prompting but was not found on PATH or in tools/ffmpeg.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg failed: {stderr.strip()}") from exc


def extract_audio_from_video(video_path: str, audio_output: str, ffmpeg_bin: str = "ffmpeg") -> str:
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
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
    return audio_output


def mux_audio_with_video(video_path: str, audio_path: str, output_video_path: str, ffmpeg_bin: str = "ffmpeg") -> str:
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_video_path,
    ]
    run_ffmpeg(command)
    return output_video_path


def load_mask_bundle(mask_path: str) -> Dict[str, Any]:
    mask_file = Path(mask_path)
    if mask_file.suffix == ".npy":
        masks = np.load(mask_file)
        frame_names = None
        frame_indices = None
    elif mask_file.suffix == ".npz":
        bundle = np.load(mask_file, allow_pickle=True)
        masks = bundle["masks"]
        frame_names = bundle["frame_names"].tolist() if "frame_names" in bundle.files else None
        frame_indices = bundle["frame_indices"].tolist() if "frame_indices" in bundle.files else None
    else:
        raise ValueError("Mask stack must be .npy or .npz.")

    if masks.ndim != 3:
        raise ValueError(f"Expected masks shaped (T, H, W); got {masks.shape}.")

    decoded_frame_names = None
    if frame_names is not None:
        decoded_frame_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in frame_names]

    return {
        "masks": masks.astype(bool),
        "frame_names": decoded_frame_names,
        "frame_indices": frame_indices,
    }


def package_masks(
    mask_bundle: Dict[str, Any],
    packaged_path: str,
) -> str:
    packaged = Path(packaged_path)
    ensure_dir(packaged.parent)
    payload = {
        "masks": mask_bundle["masks"].astype(bool),
        "frame_count": int(mask_bundle["masks"].shape[0]),
        "height": int(mask_bundle["masks"].shape[1]),
        "width": int(mask_bundle["masks"].shape[2]),
    }
    if mask_bundle.get("frame_names") is not None:
        payload["frame_names"] = np.array(mask_bundle["frame_names"])
    if mask_bundle.get("frame_indices") is not None:
        payload["frame_indices"] = np.array(mask_bundle["frame_indices"])
    np.savez_compressed(packaged, **payload)
    return str(packaged)


def probe_video(video_path: str) -> Dict[str, Any]:
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


def resolve_model_id(model_size: Optional[str], model_id: Optional[str]) -> Tuple[str, Optional[str]]:
    if model_id:
        local_candidate = resolve_existing_path(model_id, extra_search_dirs=[PROJECT_ROOT / "sam_audio_models"])
        if local_candidate.exists():
            return str(local_candidate), model_size
        return model_id, model_size
    if not model_size:
        model_size = "small-tv"
    if model_size not in VALID_MODEL_SIZES:
        raise ValueError(f"model_size must be one of {sorted(VALID_MODEL_SIZES)}")
    return MODEL_SIZE_TO_ID[model_size], model_size


def choose_device(device: Optional[str]) -> str:
    if device:
        return device
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to run SAM-Audio.") from exc

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_video_frames(video_path: str, expected_count: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for frame loading: {video_path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if len(frames) != expected_count:
        raise RuntimeError(
            f"Video frame count ({len(frames)}) does not match mask frame count ({expected_count}). "
            "Use a prepared video clip that exactly matches the tracker segment."
        )
    return frames


def infer_frame_span(frame_names: Optional[List[str]]) -> Optional[Tuple[int, int]]:
    if not frame_names:
        return None
    try:
        indices = [int(Path(name).stem) for name in frame_names]
    except ValueError:
        return None
    return min(indices), max(indices)


def validate_alignment(mask_bundle: Dict[str, Any], video_metadata: Dict[str, Any]) -> Dict[str, Any]:
    masks = mask_bundle["masks"]
    frame_count, height, width = masks.shape

    if height != video_metadata["height"] or width != video_metadata["width"]:
        raise RuntimeError(
            f"Mask resolution {width}x{height} does not match video resolution "
            f"{video_metadata['width']}x{video_metadata['height']}."
        )
    if frame_count != video_metadata["frame_count"]:
        raise RuntimeError(
            f"Mask frame count ({frame_count}) does not match video frame count ({video_metadata['frame_count']}). "
            "Tracker and audio clip segments must match exactly."
        )

    frame_span = infer_frame_span(mask_bundle.get("frame_names"))
    if frame_span is not None:
        start_idx, end_idx = frame_span
        expected_last = frame_count - 1
        if start_idx != 0 or end_idx != expected_last:
            raise RuntimeError(
                "Tracker mask frame names indicate a non-zero-based or partial segment "
                f"({start_idx}..{end_idx}) while the supplied video has {frame_count} frames. "
                "Use the exact prepared video segment that was tracked."
            )

    return {
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "fps": video_metadata["fps"],
        "frame_span": frame_span,
        "alignment_adjustments": [],
    }


def lazy_import_sam_audio():
    try:
        import torch
        import torchaudio
        from sam_audio import SAMAudio, SAMAudioProcessor
    except ImportError as exc:
        raise RuntimeError(
            "The official sam_audio package is not installed. Install the facebookresearch/sam-audio repo "
            "and authenticate with Hugging Face before running real audio inference."
        ) from exc
    return torch, torchaudio, SAMAudio, SAMAudioProcessor


def normalize_audio_tensor(audio_tensor):
    if isinstance(audio_tensor, list):
        if not audio_tensor:
            raise RuntimeError("SAM-Audio returned an empty audio list.")
        audio_tensor = audio_tensor[0]
    tensor = audio_tensor.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor[0]
    return tensor.to(dtype=tensor.dtype).float()


def video_frames_to_tensor(torch_module, frames: List[np.ndarray]):
    if not frames:
        raise RuntimeError("No video frames were loaded for SAM-Audio visual prompting.")
    stacked = np.stack(frames, axis=0)
    return torch_module.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()


def mask_stack_to_video_tensor(torch_module, masks: np.ndarray):
    if masks.ndim != 3:
        raise ValueError(f"Expected masks with shape (T, H, W); got {masks.shape}.")
    # Tracker masks use 1=True for the target object, but SAM-Audio's processor
    # zeros pixels where the mask equals 1. Invert here so the target stays visible
    # and the background is suppressed in the visual prompt.
    inverted_masks = (~masks.astype(bool)).astype(np.uint8)
    stacked = torch_module.from_numpy(inverted_masks).unsqueeze(1)
    return stacked.expand(-1, 3, -1, -1).contiguous()


def run_sam_audio_visual_prompt(
    video_path: str,
    mask_bundle: Dict[str, Any],
    model_id: str,
    device: str,
    predict_spans: bool,
    reranking_candidates: int,
    output_root: Path,
) -> Tuple[str, str, int]:
    torch, torchaudio, SAMAudio, SAMAudioProcessor = lazy_import_sam_audio()

    model = None
    processor = None
    frames = None
    video_tensor = None
    mask_tensor = None
    masked_videos = None
    batch = None
    result = None
    try:
        model = SAMAudio.from_pretrained(model_id)
        processor = SAMAudioProcessor.from_pretrained(model_id)
        model = model.eval().to(device)

        frames = load_video_frames(video_path, expected_count=mask_bundle["masks"].shape[0])
        video_tensor = video_frames_to_tensor(torch, frames)
        mask_tensor = mask_stack_to_video_tensor(torch, mask_bundle["masks"])
        masked_videos = processor.mask_videos([video_tensor], [mask_tensor])
        batch = processor(
            audios=[video_path],
            descriptions=[""],
            masked_videos=masked_videos,
        ).to(device)

        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
            )

        sample_rate = int(processor.audio_sampling_rate)
        target_audio_path = output_root / "target.wav"
        residual_audio_path = output_root / "residual.wav"
        target_tensor = normalize_audio_tensor(result.target)
        residual_tensor = normalize_audio_tensor(result.residual)
        torchaudio.save(str(target_audio_path), target_tensor, sample_rate)
        torchaudio.save(str(residual_audio_path), residual_tensor, sample_rate)
        del target_tensor
        del residual_tensor
        return str(target_audio_path), str(residual_audio_path), sample_rate
    finally:
        result = None
        batch = None
        masked_videos = None
        mask_tensor = None
        video_tensor = None
        frames = None
        processor = None
        model = None
        cleanup_inference_memory()


def run_audio_pipeline(
    video_path: str,
    mask_path: str,
    output_dir: Optional[str],
    clip_id: Optional[str],
    model_size: Optional[str] = "small-tv",
    model_id: Optional[str] = None,
    prompt_mode: str = "visual",
    device: Optional[str] = None,
    predict_spans: bool = False,
    reranking_candidates: int = 1,
    allow_placeholder: bool = False,
    mux_video_outputs: bool = True,
    ffmpeg_bin: str = "ffmpeg",
    progress_callback=None,
) -> Dict[str, Any]:
    if prompt_mode != "visual":
        raise ValueError("This pipeline currently supports only prompt_mode='visual'.")
    if reranking_candidates < 1:
        raise ValueError("reranking_candidates must be >= 1.")

    ensure_standard_directories()
    emit_progress(progress_callback, stage="audio", stage_progress=0.0, status="running", clip_id=clip_id, message="Preparing SAM-Audio inputs.")

    mask_bundle = None
    video_metadata = None
    alignment_info = None
    packaged_input = None

    try:
        resolved_video_path = resolve_existing_path(
            video_path,
            extra_search_dirs=[INPUT_ROOT, OUTPUT_ROOT / "prepared", OUTPUT_ROOT / "tracker", OUTPUT_ROOT],
        )
        resolved_mask_path = resolve_existing_path(mask_path, extra_search_dirs=[OUTPUT_ROOT / "tracker", OUTPUT_ROOT])
        clip_id = clip_id or resolved_video_path.stem
        default_output_relative = f"placeholder_test/audio/{clip_id}" if allow_placeholder else f"audio/{clip_id}"
        output_root = resolve_output_path(output_dir, default_output_relative).resolve()
        ensure_dir(output_root)
        packaged_mask_path = output_root / "sam_audio_input.npz"
        metadata_path = output_root / "audio_run_metadata.json"

        resolved_model_id, normalized_model_size = resolve_model_id(model_size, model_id)
        selected_device = choose_device(device) if not allow_placeholder else (device or "cpu")

        mask_bundle = load_mask_bundle(str(resolved_mask_path))
        video_metadata = probe_video(str(resolved_video_path))
        alignment_info = validate_alignment(mask_bundle, video_metadata)
        packaged_input = package_masks(mask_bundle, str(packaged_mask_path))
        emit_progress(progress_callback, stage="audio", stage_progress=0.2, status="running", clip_id=clip_id, message="Inputs aligned for audio separation.", stats={"frame_count": int(alignment_info["frame_count"]), "fps": float(alignment_info["fps"]), "width": int(alignment_info["width"]), "height": int(alignment_info["height"])})

        started = time.time()
        backend = "sam_audio_visual"
        target_audio_path: Optional[str] = None
        residual_audio_path: Optional[str] = None
        target_video_path: Optional[str] = None
        residual_video_path: Optional[str] = None
        sample_rate: Optional[int] = None

        if allow_placeholder:
            backend = "placeholder_passthrough"
            emit_progress(progress_callback, stage="audio", stage_progress=0.45, status="running", clip_id=clip_id, message="Generating placeholder audio outputs.")
            target_audio_path = str(output_root / "target.wav")
            residual_audio_path = str(output_root / "residual.wav")
            extract_audio_from_video(str(resolved_video_path), target_audio_path, ffmpeg_bin=ffmpeg_bin)
            extract_audio_from_video(str(resolved_video_path), residual_audio_path, ffmpeg_bin=ffmpeg_bin)
            sample_rate = 16000
        else:
            emit_progress(progress_callback, stage="audio", stage_progress=0.45, status="running", clip_id=clip_id, message="Running SAM-Audio separation.", stats={"model_id": resolved_model_id, "device": selected_device})
            target_audio_path, residual_audio_path, sample_rate = run_sam_audio_visual_prompt(
                video_path=str(resolved_video_path),
                mask_bundle=mask_bundle,
                model_id=resolved_model_id,
                device=selected_device,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
                output_root=output_root,
            )

        emit_progress(progress_callback, stage="audio", stage_progress=0.8, status="running", clip_id=clip_id, message="Audio outputs ready; muxing video previews." if mux_video_outputs else "Audio outputs ready.")

        if mux_video_outputs:
            target_video_path = str(output_root / "target_video.mp4")
            residual_video_path = str(output_root / "residual_video.mp4")
            mux_audio_with_video(str(resolved_video_path), target_audio_path, target_video_path, ffmpeg_bin=ffmpeg_bin)
            mux_audio_with_video(str(resolved_video_path), residual_audio_path, residual_video_path, ffmpeg_bin=ffmpeg_bin)

        runtime_seconds = time.time() - started
        payload = {
            "clip_id": clip_id,
            "backend": backend,
            "prompt_mode": prompt_mode,
            "model_size": normalized_model_size,
            "model_id": resolved_model_id,
            "device": selected_device,
            "predict_spans": bool(predict_spans),
            "reranking_candidates": int(reranking_candidates),
            "video_path": str(resolved_video_path),
            "mask_path": str(resolved_mask_path),
            "packaged_mask_path": packaged_input,
            "target_audio_path": str(Path(target_audio_path).resolve()),
            "residual_audio_path": str(Path(residual_audio_path).resolve()) if residual_audio_path else None,
            "target_video_path": str(Path(target_video_path).resolve()) if target_video_path else None,
            "residual_video_path": str(Path(residual_video_path).resolve()) if residual_video_path else None,
            "mux_video_outputs": bool(mux_video_outputs),
            "runtime_seconds": runtime_seconds,
            "frame_count": int(alignment_info["frame_count"]),
            "fps": float(alignment_info["fps"]),
            "width": int(alignment_info["width"]),
            "height": int(alignment_info["height"]),
            "frame_span": alignment_info["frame_span"],
            "alignment_adjustments": alignment_info["alignment_adjustments"],
            "visual_mask_semantics": "tracker_target_true_inverted_for_sam_audio",
            "sample_rate": sample_rate,
        }
        with open(metadata_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        payload["metadata_path"] = str(metadata_path)
        cleanup_stats = cleanup_inference_memory()
        payload["cleanup_stats"] = cleanup_stats
        emit_progress(progress_callback, stage="audio", stage_progress=1.0, status="completed", clip_id=clip_id, message="Audio stage complete.", outputs={"target_audio_path": payload["target_audio_path"], "residual_audio_path": payload.get("residual_audio_path"), "target_video_path": payload.get("target_video_path"), "residual_video_path": payload.get("residual_video_path"), "metadata_path": payload["metadata_path"]}, stats={"runtime_seconds": runtime_seconds, "sample_rate": sample_rate, "backend": backend})
        return payload
    finally:
        mask_bundle = None
        video_metadata = None
        alignment_info = None
        packaged_input = None
        cleanup_inference_memory()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAM-Audio with visual prompting from tracker-exported masks.")
    parser.add_argument("--video-path", required=True, help="Prepared video path aligned to the tracker mask stack.")
    parser.add_argument("--mask-path", required=True, help="Path to tracker masks (.npy or .npz).")
    parser.add_argument("--output-dir", default=None, help="Directory for SAM-Audio outputs. Defaults to output/audio/<clip_id>.")
    parser.add_argument("--clip-id", default=None, help="Clip identifier for metadata. Defaults to the video filename stem.")
    parser.add_argument(
        "--model-size",
        default="small-tv",
        choices=sorted(VALID_MODEL_SIZES),
        help="Model size alias. Defaults to the visual-prompting-friendly small-tv variant.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Optional explicit Hugging Face model id or local from_pretrained directory. Overrides --model-size.",
    )
    parser.add_argument(
        "--prompt-mode",
        default="visual",
        choices=["visual"],
        help="Prompt mode for SAM-Audio. Only visual prompting is currently supported here.",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda, cpu, or mps.")
    parser.add_argument(
        "--predict-spans",
        action="store_true",
        help="Enable SAM-Audio span prediction during separation.",
    )
    parser.add_argument(
        "--reranking-candidates",
        type=int,
        default=1,
        help="Number of SAM-Audio reranking candidates to generate.",
    )
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="Smoke-test mode only: extract passthrough audio instead of running real SAM-Audio inference.",
    )
    parser.add_argument(
        "--no-mux-video",
        action="store_true",
        help="Skip creating target/residual MP4 files with the separated audio muxed back onto the input video.",
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name or path.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    result = run_audio_pipeline(
        video_path=args.video_path,
        mask_path=args.mask_path,
        output_dir=args.output_dir,
        clip_id=args.clip_id,
        model_size=args.model_size,
        model_id=args.model_id,
        prompt_mode=args.prompt_mode,
        device=args.device,
        predict_spans=args.predict_spans,
        reranking_candidates=args.reranking_candidates,
        allow_placeholder=args.allow_placeholder,
        mux_video_outputs=not args.no_mux_video,
        ffmpeg_bin=args.ffmpeg_bin,
    )
    print(json.dumps(result, indent=2))
