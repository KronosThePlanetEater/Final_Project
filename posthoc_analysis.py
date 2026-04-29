from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from analyze_results import analyze_results
from path_layout import OUTPUT_ROOT, PLACEHOLDER_ROOT, ensure_standard_directories, resolve_existing_path, resolve_output_path
from progress_utils import iso_utc_now


ANALYSIS_SET_ROOT = OUTPUT_ROOT / "analysis_sets"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json_if_exists(path_value: Optional[str]) -> Dict[str, Any]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def slugify(value: str) -> str:
    chars: List[str] = []
    previous_dash = False
    for char in str(value).lower():
        if char.isalnum():
            chars.append(char)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-") or "analysis"


def build_analysis_set_id(clip_id: str) -> str:
    stamp = datetime.now().strftime("%m-%d-%y-%H%M%S")
    return f"{stamp}-{slugify(clip_id)}"


def infer_job_id_from_run_dir(run_dir: Path) -> Optional[str]:
    parts = list(run_dir.parts)
    if "ui_runs" in parts:
        index = parts.index("ui_runs")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def resolve_clip_id(tracking: Dict[str, Any], audio: Dict[str, Any], evaluation: Dict[str, Any], run_dir: Path) -> str:
    clip_id = evaluation.get("clip_id") or audio.get("clip_id")
    if clip_id:
        return str(clip_id)
    tracking_clip = tracking.get("clip_id")
    if tracking_clip and tracking_clip != "tracked":
        return str(tracking_clip)
    video_path = tracking.get("video_path")
    if isinstance(video_path, str) and video_path:
        return Path(video_path).stem
    return run_dir.name


def build_candidate_entry(
    run_dir: Path,
    tracking_summary_path: Path,
    audio_metadata_path: Optional[Path] = None,
    evaluation_summary_path: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if not tracking_summary_path.exists():
        return None
    tracking = load_json_if_exists(str(tracking_summary_path))
    if not tracking:
        return None
    audio = load_json_if_exists(str(audio_metadata_path)) if audio_metadata_path else {}
    evaluation = load_json_if_exists(str(evaluation_summary_path)) if evaluation_summary_path else {}
    metadata = metadata or {}

    clip_id = resolve_clip_id(tracking, audio, evaluation, run_dir)
    run_id = metadata.get("run_id") or run_dir.name
    tracker_variant_key = metadata.get("tracker_variant_key") or str(run_id).split("__")[0]
    tracker_variant_label = metadata.get("tracker_variant_label") or tracker_variant_key
    audio_model_size = metadata.get("audio_model_size") or audio.get("model_size") or evaluation.get("model_size")
    display_label = metadata.get("display_label") or f"{tracker_variant_label} | {audio_model_size or 'audio'}"

    return {
        "run_id": run_id,
        "job_id": metadata.get("job_id") or infer_job_id_from_run_dir(run_dir),
        "clip_id": clip_id,
        "motion_level": metadata.get("motion_level"),
        "display_label": display_label,
        "tracker_variant_key": tracker_variant_key,
        "tracker_variant_label": tracker_variant_label,
        "audio_model_size": audio_model_size,
        "model_id": evaluation.get("model_id") or audio.get("model_id"),
        "prompt_mode": evaluation.get("prompt_mode") or audio.get("prompt_mode"),
        "backend": evaluation.get("backend") or audio.get("backend"),
        "predict_spans": evaluation.get("predict_spans") if evaluation else audio.get("predict_spans"),
        "reranking_candidates": evaluation.get("reranking_candidates") if evaluation else audio.get("reranking_candidates"),
        "requested_audio_precision": audio.get("requested_audio_precision"),
        "effective_audio_precision": audio.get("effective_audio_precision"),
        "run_dir": str(run_dir.resolve()),
        "tracking_summary_path": str(tracking_summary_path.resolve()),
        "audio_metadata_path": str(audio_metadata_path.resolve()) if audio_metadata_path and audio_metadata_path.exists() else None,
        "evaluation_summary_path": str(evaluation_summary_path.resolve()) if evaluation_summary_path and evaluation_summary_path.exists() else None,
    }


def discover_runs_from_job_states(root_dir: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if not root_dir.exists():
        return candidates

    for state_path in root_dir.glob("*/job_state.json"):
        try:
            with open(state_path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except Exception:
            continue
        if state.get("status") != "completed":
            continue
        manifest = (state.get("results") or {}).get("run_manifest") or []
        for entry in manifest:
            tracking_summary_path = entry.get("tracking_summary_path")
            if not tracking_summary_path:
                continue
            candidate = build_candidate_entry(
                run_dir=Path(entry.get("run_dir") or Path(tracking_summary_path).parent.parent),
                tracking_summary_path=Path(tracking_summary_path),
                audio_metadata_path=Path(entry["audio_metadata_path"]) if entry.get("audio_metadata_path") else None,
                evaluation_summary_path=Path(entry["evaluation_summary_path"]) if entry.get("evaluation_summary_path") else None,
                metadata={**entry, "job_id": state.get("job_id")},
            )
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def discover_runs_from_generic_root(root_dir: Path) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    if not root_dir.exists():
        return candidates

    seen = set()
    for tracking_summary_path in root_dir.rglob("metrics/tracking_summary.json"):
        if "tracking_cache" in tracking_summary_path.parts or "analysis_sets" in tracking_summary_path.parts or "ui_runs" in tracking_summary_path.parts:
            continue
        run_dir = tracking_summary_path.parent.parent
        run_dir_key = str(run_dir.resolve())
        if run_dir_key in seen:
            continue
        seen.add(run_dir_key)
        audio_metadata_path = run_dir / "audio" / "audio_run_metadata.json"
        evaluation_summary_path = run_dir / "eval" / "evaluation_summary.json"
        candidate = build_candidate_entry(
            run_dir=run_dir,
            tracking_summary_path=tracking_summary_path,
            audio_metadata_path=audio_metadata_path if audio_metadata_path.exists() else None,
            evaluation_summary_path=evaluation_summary_path if evaluation_summary_path.exists() else None,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def discover_mergeable_runs() -> List[Dict[str, Any]]:
    ensure_standard_directories()
    ensure_dir(ANALYSIS_SET_ROOT)
    candidates: List[Dict[str, Any]] = []
    candidates.extend(discover_runs_from_job_states(OUTPUT_ROOT / "ui_runs"))
    candidates.extend(discover_runs_from_job_states(PLACEHOLDER_ROOT / "ui_runs"))
    candidates.extend(discover_runs_from_generic_root(OUTPUT_ROOT / "experiments"))
    candidates.extend(discover_runs_from_generic_root(OUTPUT_ROOT / "terminal_runs"))

    deduped: Dict[str, Dict[str, Any]] = {}
    for entry in candidates:
        deduped[entry["run_dir"]] = entry
    return sorted(
        deduped.values(),
        key=lambda entry: (str(entry.get("clip_id") or ""), str(entry.get("job_id") or ""), str(entry.get("display_label") or "")),
    )


def group_runs_by_clip(entries: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        clip_id = str(entry.get("clip_id") or "unknown")
        grouped.setdefault(clip_id, []).append(entry)
    for clip_id in grouped:
        grouped[clip_id] = sorted(grouped[clip_id], key=lambda entry: str(entry.get("display_label") or entry.get("run_id") or ""))
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def resolve_selected_runs(selected_runs: List[Any]) -> List[Dict[str, Any]]:
    resolved: List[Dict[str, Any]] = []
    for item in selected_runs:
        if isinstance(item, dict):
            resolved.append(item)
            continue
        run_dir = resolve_existing_path(str(item), extra_search_dirs=[OUTPUT_ROOT]).resolve()
        tracking_summary_path = run_dir / "metrics" / "tracking_summary.json"
        audio_metadata_path = run_dir / "audio" / "audio_run_metadata.json"
        evaluation_summary_path = run_dir / "eval" / "evaluation_summary.json"
        candidate = build_candidate_entry(
            run_dir=run_dir,
            tracking_summary_path=tracking_summary_path,
            audio_metadata_path=audio_metadata_path if audio_metadata_path.exists() else None,
            evaluation_summary_path=evaluation_summary_path if evaluation_summary_path.exists() else None,
        )
        if candidate is None:
            raise FileNotFoundError(f"Could not load run summaries from {run_dir}")
        resolved.append(candidate)
    return resolved


def create_posthoc_analysis_set(
    selected_runs: List[Any],
    output_dir: Optional[str] = None,
    set_id: Optional[str] = None,
    progress_callback=None,
) -> Dict[str, Any]:
    ensure_standard_directories()
    ensure_dir(ANALYSIS_SET_ROOT)
    resolved_runs = resolve_selected_runs(selected_runs)
    if not resolved_runs:
        raise ValueError("At least one completed run must be selected.")

    clip_ids = {str(entry.get("clip_id") or "") for entry in resolved_runs}
    if len(clip_ids) != 1:
        raise ValueError("Post-hoc analysis sets may only merge runs from the same clip.")
    clip_id = next(iter(clip_ids))

    chosen_set_id = set_id or build_analysis_set_id(clip_id)
    set_root = resolve_output_path(output_dir, f"analysis_sets/{chosen_set_id}").resolve()
    ensure_dir(set_root)
    analysis_dir = set_root / "analysis"
    ensure_dir(analysis_dir)

    manifest_payload = {"runs": resolved_runs}
    run_manifest_path = set_root / "run_manifest.json"
    with open(run_manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    analysis_summary = analyze_results(
        experiment_root=None,
        output_dir=str(analysis_dir),
        progress_callback=progress_callback,
        run_manifest_path=str(run_manifest_path),
    )

    metadata = {
        "set_id": chosen_set_id,
        "clip_id": clip_id,
        "created_at": iso_utc_now(),
        "analysis_source_mode": "posthoc_merge",
        "selected_run_count": len(resolved_runs),
        "selected_run_ids": [entry.get("run_id") for entry in resolved_runs],
        "selected_job_ids": sorted({entry.get("job_id") for entry in resolved_runs if entry.get("job_id")}),
        "set_dir": str(set_root),
        "run_manifest_path": str(run_manifest_path),
        "analysis_dir": str(analysis_dir),
        "analysis_summary_path": analysis_summary.get("summary_path"),
    }
    metadata_path = set_root / "set_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "set_id": chosen_set_id,
        "clip_id": clip_id,
        "set_dir": str(set_root),
        "metadata_path": str(metadata_path),
        "run_manifest_path": str(run_manifest_path),
        "analysis_summary": analysis_summary,
    }


def get_recent_analysis_set_ids(limit: int = 20) -> List[str]:
    ensure_standard_directories()
    ensure_dir(ANALYSIS_SET_ROOT)
    sets = []
    for metadata_path in ANALYSIS_SET_ROOT.glob("*/set_metadata.json"):
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            continue
        sets.append((metadata.get("created_at") or "", metadata_path.parent.name))
    sets.sort(reverse=True)
    return [set_id for _, set_id in sets[:limit]]


def load_analysis_set(set_id: str) -> Dict[str, Any]:
    set_root = resolve_existing_path(set_id, extra_search_dirs=[ANALYSIS_SET_ROOT]).resolve()
    if set_root.is_file():
        set_root = set_root.parent
    metadata = load_json_if_exists(str(set_root / "set_metadata.json"))
    manifest_payload = load_json_if_exists(str(set_root / "run_manifest.json"))
    analysis_summary = load_json_if_exists(str(set_root / "analysis" / "analysis_summary.json"))
    aggregate_csv = analysis_summary.get("aggregate_csv") if analysis_summary else None
    images = sorted(str(path.resolve()) for path in (set_root / "analysis").glob("*.png")) if (set_root / "analysis").exists() else []
    return {
        "set_id": metadata.get("set_id") or set_root.name,
        "metadata": metadata,
        "run_manifest": manifest_payload.get("runs", []) if isinstance(manifest_payload, dict) else manifest_payload,
        "analysis_summary": analysis_summary,
        "aggregate_csv": aggregate_csv,
        "analysis_images": images,
        "analysis_dir": str((set_root / "analysis").resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a post-hoc merged analysis set from completed runs.")
    parser.add_argument("--run-dir", action="append", default=[], help="Run directory to include. Provide multiple times to merge multiple runs.")
    parser.add_argument("--clip-id", default=None, help="Optional clip id to include all discovered completed runs for that clip when --run-dir is omitted.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for the analysis set. Defaults to output/analysis_sets/<set_id>.")
    parser.add_argument("--set-id", default=None, help="Optional explicit analysis set id.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.run_dir:
        selected = args.run_dir
    elif args.clip_id:
        selected = [entry for entry in discover_mergeable_runs() if entry.get("clip_id") == args.clip_id]
    else:
        parser.error("Provide at least one --run-dir or a --clip-id.")

    result = create_posthoc_analysis_set(selected, output_dir=args.output_dir, set_id=args.set_id)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
