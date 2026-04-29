from __future__ import annotations

import argparse
import traceback

from runtime_console import apply_runtime_console_config
from ui_backend import enqueue_ui_jobs, load_job_state, parse_queue_manifest_path, run_queue_until_empty


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run queued UI jobs, optionally loading a queue manifest first.")
    parser.add_argument(
        "manifest",
        nargs="?",
        help="Optional queue manifest JSON, such as input/medium_movement_main_8_run_queue.json.",
    )
    return parser


def raise_if_manifest_jobs_failed(job_ids: list[str]) -> None:
    failed = []
    for job_id in job_ids:
        state = load_job_state(job_id)
        if state and state.get("status") == "failed":
            failed.append(f"{job_id}: {state.get('error_message') or 'failed'}")
    if failed:
        raise RuntimeError("One or more manifest jobs failed:\n" + "\n".join(failed))


def main() -> None:
    apply_runtime_console_config()
    args = build_arg_parser().parse_args()
    job_ids: list[str] = []
    try:
        if args.manifest:
            manifest = parse_queue_manifest_path(args.manifest)
            job_ids = enqueue_ui_jobs(manifest["jobs"], start_worker=False)
            print(f"Queued {len(job_ids)} jobs from {args.manifest}.")
        run_queue_until_empty()
        if job_ids:
            raise_if_manifest_jobs_failed(job_ids)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
