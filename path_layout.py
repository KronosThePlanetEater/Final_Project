from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = PROJECT_ROOT / "input"
OUTPUT_ROOT = PROJECT_ROOT / "output"
PLACEHOLDER_ROOT = OUTPUT_ROOT / "placeholder_test"
TOOLS_ROOT = PROJECT_ROOT / "tools"
FFMPEG_ROOT = TOOLS_ROOT / "ffmpeg"
FFMPEG_WINDOWS = FFMPEG_ROOT / "windows"
FFMPEG_LINUX = FFMPEG_ROOT / "linux"


def ensure_standard_directories() -> None:
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    PLACEHOLDER_ROOT.mkdir(parents=True, exist_ok=True)
    FFMPEG_WINDOWS.mkdir(parents=True, exist_ok=True)
    FFMPEG_LINUX.mkdir(parents=True, exist_ok=True)


def resolve_existing_path(path_value: str, extra_search_dirs: Optional[Iterable[Path]] = None) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [PROJECT_ROOT]
    if extra_search_dirs:
        search_roots.extend(Path(path) for path in extra_search_dirs)
    search_roots.extend([INPUT_ROOT, OUTPUT_ROOT])

    for root_dir in search_roots:
        resolved = (root_dir / candidate).resolve()
        if resolved.exists():
            return resolved
    return (PROJECT_ROOT / candidate).resolve()


def resolve_input_path(path_value: str, extra_search_dirs: Optional[Iterable[Path]] = None) -> Path:
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    search_roots = [INPUT_ROOT]
    if extra_search_dirs:
        search_roots.extend(Path(path) for path in extra_search_dirs)
    search_roots.append(PROJECT_ROOT)

    for root_dir in search_roots:
        resolved = (root_dir / candidate).resolve()
        if resolved.exists():
            return resolved
    return (INPUT_ROOT / candidate).resolve()


def resolve_output_path(path_value: Optional[str], default_relative: str) -> Path:
    candidate = Path(path_value) if path_value else Path(default_relative)
    if candidate.is_absolute():
        return candidate.resolve()
    return (OUTPUT_ROOT / candidate).resolve()


def resolve_ffmpeg_binary(ffmpeg_bin: Optional[str] = None) -> str:
    ensure_standard_directories()
    candidate = (ffmpeg_bin or "ffmpeg").strip() or "ffmpeg"

    explicit = Path(candidate).expanduser()
    if candidate != "ffmpeg":
        if explicit.is_absolute():
            return str(explicit.resolve())
        for root_dir in (PROJECT_ROOT, TOOLS_ROOT, FFMPEG_ROOT, FFMPEG_WINDOWS, FFMPEG_LINUX):
            resolved = (root_dir / explicit).resolve()
            if resolved.exists():
                return str(resolved)
        return candidate

    windows_local = FFMPEG_WINDOWS / "ffmpeg.exe"
    linux_local = FFMPEG_LINUX / "ffmpeg"
    if os.name == "nt":
        if windows_local.exists():
            return str(windows_local.resolve())
        if linux_local.exists():
            return str(linux_local.resolve())
    else:
        if linux_local.exists():
            return str(linux_local.resolve())
        if windows_local.exists():
            return str(windows_local.resolve())
    return "ffmpeg"
