from __future__ import annotations

import traceback

from runtime_console import apply_runtime_console_config
from ui_backend import run_queue_until_empty


def main() -> None:
    apply_runtime_console_config()
    try:
        run_queue_until_empty()
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
