from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "runtime_console_config.json"
NOISY_PATTERNS = (
    "Accessing `__path__` from ",
    "Examining the path of torch.classes raised",
    "The pynvml package is deprecated",
)


class _NoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        return not any(pattern in message for pattern in NOISY_PATTERNS)


class _FilteringStream:
    def __init__(self, wrapped, patterns):
        self._wrapped = wrapped
        self._patterns = tuple(patterns)
        self._buffer = ""

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        self._buffer += data
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not any(pattern in line for pattern in self._patterns):
                self._wrapped.write(line + "\n")
        return len(data)

    def flush(self):
        if self._buffer and not any(pattern in self._buffer for pattern in self._patterns):
            self._wrapped.write(self._buffer)
        self._buffer = ""
        self._wrapped.flush()

    def isatty(self):
        return getattr(self._wrapped, "isatty", lambda: False)()

    def fileno(self):
        return self._wrapped.fileno()

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def load_runtime_console_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"suppress_console_noise": False}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {"suppress_console_noise": False}
    if not isinstance(payload, dict):
        return {"suppress_console_noise": False}
    return payload


def apply_runtime_console_config() -> Dict[str, Any]:
    config = load_runtime_console_config()
    if not bool(config.get("suppress_console_noise", False)):
        return config

    warnings.filterwarnings("ignore", message=r"Accessing `__path__` from .*", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*", category=FutureWarning)

    noise_filter = _NoiseFilter()
    logging.getLogger().addFilter(noise_filter)
    for logger_name in [
        "streamlit",
        "streamlit.runtime.scriptrunner.script_runner",
        "transformers",
        "transformers.utils.import_utils",
        "torch",
    ]:
        logging.getLogger(logger_name).addFilter(noise_filter)

    if not isinstance(sys.stdout, _FilteringStream):
        sys.stdout = _FilteringStream(sys.stdout, NOISY_PATTERNS)
    if not isinstance(sys.stderr, _FilteringStream):
        sys.stderr = _FilteringStream(sys.stderr, NOISY_PATTERNS)

    return config
