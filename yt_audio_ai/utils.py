from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_info_json(audio_file: Path) -> Dict[str, Any]:
    info_path = audio_file.with_suffix(".info.json")
    data: Dict[str, Any] = {}
    if info_path.exists():
        data = json.loads(info_path.read_text(encoding="utf-8"))
    return data


def get_video_id_from_audio(audio_file: Path) -> str:
    # audio files are saved as {id}.mp3
    return audio_file.stem


def build_youtube_link(video_id: str, start_seconds: int | float | None = None) -> str:
    base = f"https://youtu.be/{video_id}"
    if start_seconds is None:
        return base
    try:
        t = int(start_seconds)
    except Exception:
        t = 0
    return f"{base}?t={t}"
