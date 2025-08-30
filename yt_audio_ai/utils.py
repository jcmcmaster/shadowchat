from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Any


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists, creating it and any parent directories if needed.
    
    Args:
        path: The directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def read_info_json(audio_file: Path) -> Dict[str, Any]:
    """Read the associated .info.json file for an audio file.
    
    Args:
        audio_file: Path to the audio file (.mp3). The corresponding .info.json 
                   file is expected to be in the same directory.
    
    Returns:
        Dictionary containing the JSON data from the info file, or empty dict if not found.
    """
    info_path = audio_file.with_suffix(".info.json")
    data: Dict[str, Any] = {}
    if info_path.exists():
        data = json.loads(info_path.read_text(encoding="utf-8"))
    return data


def get_video_id_from_audio(audio_file: Path) -> str:
    """Extract the video ID from an audio file path.
    
    Audio files are saved as {id}.mp3, so this returns the filename without extension.
    
    Args:
        audio_file: Path to the audio file.
        
    Returns:
        The video ID (filename without extension).
    """
    # audio files are saved as {id}.mp3
    return audio_file.stem


def build_youtube_link(video_id: str, start_seconds: int | float | None = None) -> str:
    """Build a YouTube URL for a video, optionally with a timestamp.
    
    Args:
        video_id: The YouTube video ID.
        start_seconds: Optional timestamp to start playback at (in seconds).
        
    Returns:
        A YouTube URL, with timestamp parameter if start_seconds is provided.
    """
    base = f"https://youtu.be/{video_id}"
    if start_seconds is None:
        return base
    try:
        t = int(start_seconds)
    except Exception:
        t = 0
    return f"{base}?t={t}"
