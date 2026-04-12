from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

ProgressCallback = Optional[Callable[[Dict[str, Any]], None]]


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit_progress(callback: ProgressCallback, **payload: Any) -> None:
    if callback is None:
        return
    event = dict(payload)
    event.setdefault("timestamp", iso_utc_now())
    callback(event)
