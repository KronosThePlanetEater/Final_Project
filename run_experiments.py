import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

from audio_pipeline import create_audio_aligned_segment, run_audio_pipeline
from dataset_prep import prepare_clip
from evaluation import evaluate_run
from tracker import selection_from_dict, track_object
from path_layout import PLACEHOLDER_ROOT, ensure_standard_directories, resolve_existing_path, resolve_input_path, resolve_output_path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_optional_json_map(json_path: Optional[str]) -> Dict[str, str]:
    if not json_path:
        return {}
    with open(resolve_existing_path(json_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def materialize_tracking_outputs(tracking_summary: Dict[str, object], run_dir: Path) -> Dict[str, object]:
    video_dir = run_dir / "video"
    masks_dir = run_dir / "masks"
    metrics_dir = run_dir / "metrics"
    ensure_dir(video_dir)
    ensure_dir(masks_dir)
    ensure_dir(metrics_dir)

    output_video_path = Path(tracking_summary["output_video_path"])
    metrics_csv_path = Path(tracking_summary["metrics_csv_path"])
    summary_json_path = Path(tracking_summary["summary_json_path"])
    mask_stack_path = Path(tracking_summary["mask_stack_path"])
    mask_png_dir = Path(tracking_summary["mask_png_dir"]) if tracking_summary.get("mask_png_dir") else None

    copied_video_path = video_dir / output_video_path.name
    copied_metrics_csv_path = metrics_dir / metrics_csv_path.name
    copied_summary_json_path = metrics_dir / summary_json_path.name
    copied_mask_stack_path = masks_dir / mask_stack_path.name
    copied_mask_png_dir = masks_dir / "png"

    copy_path(output_video_path, copied_video_path)
    copy_path(metrics_csv_path, copied_metrics_csv_path)
    copy_path(mask_stack_path, copied_mask_stack_path)
    if mask_png_dir and mask_png_dir.exists():
        copy_path(mask_png_dir, copied_mask_png_dir)

    with open(summary_json_path, "r", encoding="utf-8") as handle:
        copied_summary = json.load(handle)
    copied_summary["output_video_path"] = str(copied_video_path.resolve())
    copied_summary["artifacts_dir"] = str(run_dir.resolve())
    copied_summary["metrics_csv_path"] = str(copied_metrics_csv_path.resolve())
    copied_summary["summary_json_path"] = str(copied_summary_json_path.resolve())
    copied_summary["mask_stack_path"] = str(copied_mask_stack_path.resolve())
    copied_summary["mask_png_dir"] = str(copied_mask_png_dir.resolve()) if copied_mask_png_dir.exists() else None

    with open(copied_summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(copied_summary, handle, indent=2)
    return copied_summary


def run_batch_experiments(
    video_paths,
    output_dir,
    checkpoint_path,
    config_path,
    selection_map=None,
    reference_audio_map=None,
    ground_truth_mask_map=None,
    audio_model_sizes=None,
    allow_placeholder_audio=False,
    audio_model_id=None,
    audio_device=None,
    predict_spans=False,
    reranking_candidates=1,
    target_fps=None,
    target_width=None,
    target_height=None,
    ffmpeg_bin="ffmpeg",
    selection_mode="point",
    max_frames=None,
    start_time=0.0,
    scale=1.0,
    sam_interval=1,
    dynamic_interval=None,
    skip_mask_confirmation=False,
    save_mask_pngs=False,
    show_preview=False,
):
    ensure_standard_directories()

    default_output_relative = "placeholder_test/experiments" if allow_placeholder_audio else "experiments"
    output_root = resolve_output_path(output_dir, default_output_relative).resolve()
    ensure_dir(output_root)
    checkpoint_path = str(resolve_existing_path(checkpoint_path))
    config_path = str(resolve_existing_path(config_path))
    selection_map = selection_map or {}
    reference_audio_map = reference_audio_map or {}
    ground_truth_mask_map = ground_truth_mask_map or {}
    audio_model_sizes = audio_model_sizes or ["small-tv"]

    prepared_root = output_root / "prepared"
    tracking_cache_root = output_root / "_tracking_cache"
    ensure_dir(prepared_root)
    ensure_dir(tracking_cache_root)

    manifest = []
    for video_path in video_paths:
        resolved_video_path = resolve_input_path(video_path)
        clip_id = resolved_video_path.stem
        prepared = prepare_clip(
            video_path=str(resolved_video_path),
            output_root=str(prepared_root),
            clip_id=clip_id,
            target_fps=target_fps,
            target_width=target_width,
            target_height=target_height,
            reference_audio=reference_audio_map.get(clip_id),
            ffmpeg_bin=ffmpeg_bin,
        )

        tracking_cache_dir = tracking_cache_root / clip_id
        tracking_video_dir = tracking_cache_dir / "video"
        tracking_frames_dir = tracking_video_dir / "frames"
        ensure_dir(tracking_video_dir)
        selection = selection_from_dict(selection_map.get(clip_id))

        tracking_summary = track_object(
            video_path=prepared["prepared_video_path"],
            output_path=str(tracking_video_dir / "tracked.mp4"),
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            frames_dir=str(tracking_frames_dir),
            extract_frames=True,
            selection_mode=selection_mode,
            max_frames=max_frames,
            start_time=start_time,
            scale=scale,
            initial_selection=selection,
            skip_mask_confirmation=skip_mask_confirmation,
            sam_interval=sam_interval,
            dynamic_interval=dynamic_interval,
            background_color=None,
            show_preview=show_preview,
            artifacts_dir=str(tracking_cache_dir),
            save_mask_pngs=save_mask_pngs,
        )

        for model_size in audio_model_sizes:
            run_dir = output_root / f"{clip_id}_{model_size}"
            ensure_dir(run_dir)
            copied_tracking_summary = materialize_tracking_outputs(tracking_summary, run_dir)
            audio_input_video_path = create_audio_aligned_segment(
                source_video_path=prepared["canonical_video_path"],
                output_video_path=str(run_dir / "audio_input" / "input_segment.mp4"),
                start_time_seconds=float(copied_tracking_summary.get("start_time_seconds", 0.0) or 0.0),
                frame_count=int(copied_tracking_summary.get("total_frames") or 0),
                fps=float(copied_tracking_summary.get("fps") or prepared.get("fps") or 0.0),
                ffmpeg_bin=ffmpeg_bin,
            )

            audio_result = run_audio_pipeline(
                video_path=audio_input_video_path,
                mask_path=copied_tracking_summary["mask_stack_path"],
                output_dir=str(run_dir / "audio"),
                clip_id=clip_id,
                model_size=model_size,
                model_id=audio_model_id,
                prompt_mode="visual",
                device=audio_device,
                predict_spans=predict_spans,
                reranking_candidates=reranking_candidates,
                allow_placeholder=allow_placeholder_audio,
                ffmpeg_bin=ffmpeg_bin,
            )

            evaluation_result = evaluate_run(
                output_dir=str(run_dir / "eval"),
                clip_id=clip_id,
                model_size=model_size,
                model_id=audio_result["model_id"],
                predicted_mask_path=copied_tracking_summary["mask_stack_path"],
                ground_truth_mask_path=ground_truth_mask_map.get(clip_id),
                estimated_audio_path=audio_result["target_audio_path"],
                reference_audio_path=reference_audio_map.get(clip_id),
                audio_metadata_path=audio_result["metadata_path"],
            )

            manifest.append(
                {
                    "clip_id": clip_id,
                    "model_size": model_size,
                    "run_dir": str(run_dir),
                    "prepared_clip": prepared,
                    "tracking_summary_path": copied_tracking_summary["summary_json_path"],
                    "audio_metadata_path": audio_result["metadata_path"],
                    "evaluation_summary_path": evaluation_result["summary_path"],
                }
            )

    manifest_path = output_root / "experiment_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return {"manifest_path": str(manifest_path), "run_count": len(manifest)}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the full prep -> tracking -> audio -> evaluation pipeline.")
    parser.add_argument("video_paths", nargs="+", help="Input videos to process.")
    parser.add_argument("--output-dir", default=None, help="Root output directory for experiments. Defaults to output/experiments.")
    parser.add_argument("--checkpoint", default="checkpoints/sam2.1_hiera_tiny.pt", help="SAM checkpoint path.")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_t.yaml", help="SAM config path.")
    parser.add_argument("--mode", choices=["point", "bbox"], default="point", help="Tracker selection mode.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for tracking.")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--scale", type=float, default=1.0, help="Tracker frame scale.")
    parser.add_argument("--sam-interval", type=int, default=1, help="Static SAM interval.")
    parser.add_argument(
        "--dynamic-interval",
        nargs=2,
        type=int,
        default=None,
        metavar=("MIN", "MAX"),
        help="Adaptive tracker interval range.",
    )
    parser.add_argument(
        "--selection-map",
        default=None,
        help="JSON mapping clip ids to tracker selection objects (bbox or points+labels).",
    )
    parser.add_argument(
        "--reference-audio-map",
        default=None,
        help="JSON mapping clip ids to clean reference audio WAVs.",
    )
    parser.add_argument(
        "--ground-truth-mask-map",
        default=None,
        help="JSON mapping clip ids to ground-truth masks (.npy/.npz or PNG dir).",
    )
    parser.add_argument(
        "--audio-model-sizes",
        nargs="+",
        default=["small-tv"],
        help="Audio model sizes to evaluate.",
    )
    parser.add_argument(
        "--audio-model-id",
        default=None,
        help="Optional explicit Hugging Face model id or local from_pretrained directory.",
    )
    parser.add_argument(
        "--allow-placeholder-audio",
        action="store_true",
        help="Allow passthrough placeholder audio if no SAM-audio backend is configured.",
    )
    parser.add_argument("--audio-device", default=None, help="Torch device override for SAM-Audio.")
    parser.add_argument("--predict-spans", action="store_true", help="Enable SAM-Audio span prediction.")
    parser.add_argument(
        "--reranking-candidates",
        type=int,
        default=1,
        help="Number of SAM-Audio reranking candidates.",
    )
    parser.add_argument("--target-fps", type=float, default=None, help="Prepared video FPS.")
    parser.add_argument("--target-width", type=int, default=None, help="Prepared video width.")
    parser.add_argument("--target-height", type=int, default=None, help="Prepared video height.")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg", help="ffmpeg executable name or path.")
    parser.add_argument(
        "--skip-mask-confirmation",
        action="store_true",
        help="Skip tracker mask confirmation dialogs.",
    )
    parser.add_argument("--save-mask-pngs", action="store_true", help="Export per-frame mask PNGs.")
    parser.add_argument("--preview", action="store_true", help="Show tracker preview during tracking.")
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    result = run_batch_experiments(
        video_paths=args.video_paths,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        selection_map=load_optional_json_map(args.selection_map),
        reference_audio_map=load_optional_json_map(args.reference_audio_map),
        ground_truth_mask_map=load_optional_json_map(args.ground_truth_mask_map),
        audio_model_sizes=args.audio_model_sizes,
        allow_placeholder_audio=args.allow_placeholder_audio,
        audio_model_id=args.audio_model_id,
        audio_device=args.audio_device,
        predict_spans=args.predict_spans,
        reranking_candidates=args.reranking_candidates,
        target_fps=args.target_fps,
        target_width=args.target_width,
        target_height=args.target_height,
        ffmpeg_bin=args.ffmpeg_bin,
        selection_mode=args.mode,
        max_frames=args.max_frames,
        start_time=args.start_time,
        scale=args.scale,
        sam_interval=args.sam_interval,
        dynamic_interval=tuple(args.dynamic_interval) if args.dynamic_interval else None,
        skip_mask_confirmation=args.skip_mask_confirmation,
        save_mask_pngs=args.save_mask_pngs,
        show_preview=args.preview,
    )
    print(json.dumps(result, indent=2))
