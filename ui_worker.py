from __future__ import annotations

import argparse
import traceback

from runtime_console import apply_runtime_console_config
from ui_backend import mark_job_failed, run_ui_job


def main() -> None:
    apply_runtime_console_config()
    parser = argparse.ArgumentParser(description="Run one UI pipeline job in a separate worker process.")
    parser.add_argument("--job-id", required=True, help="Job id to execute.")
    args = parser.parse_args()

    try:
        run_ui_job(args.job_id)
    except Exception as exc:
        try:
            mark_job_failed(args.job_id, "tracking", exc)
        except Exception:
            traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
