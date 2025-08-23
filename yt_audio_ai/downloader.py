from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any
from rich import print

import yt_dlp

from .utils import ensure_dir


class YouTubeAudioDownloader:
    def __init__(self, audio_dir: Path) -> None:
        self.audio_dir = audio_dir
        ensure_dir(self.audio_dir)

    def _already_downloaded(self, video_id: str) -> bool:
        return (self.audio_dir / f"{video_id}.mp3").exists()

    def download_many(self, urls: Iterable[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        urls = [u.strip() for u in urls if u and u.strip()]
        if not urls:
            print("[yellow]No URLs provided[/yellow]")
            return results
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(self.audio_dir / "%(id)s.%(ext)s"),
            "writethumbnail": False,
            "writeinfojson": True,
            "ignoreerrors": True,
            "extract_flat": False,  # let yt-dlp expand playlists itself
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "quiet": True,
            "noprogress": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for url in urls:
                try:
                    # This handles both single videos and playlists
                    info = ydl.extract_info(url, download=False)
                    entries = info.get("entries") if isinstance(info, dict) else None
                    if entries:
                        iterable = entries
                    else:
                        iterable = [info]

                    for item in iterable:
                        if not item:
                            continue
                        vid = item.get("id")
                        if not vid:
                            continue
                        if self._already_downloaded(vid):
                            print(f"[cyan]Skip[/cyan] already have {vid}")
                            continue
                        # download the concrete video URL
                        ydl.download([item.get("webpage_url") or item.get("url") or url])
                        audio_path = self.audio_dir / f"{vid}.mp3"
                        info_path = self.audio_dir / f"{vid}.info.json"
                        results.append(
                            {
                                "video_id": vid,
                                "audio_path": str(audio_path),
                                "info_path": str(info_path),
                                "original_url": url,
                            }
                        )
                        print(f"[green]Downloaded[/green] {vid} -> {audio_path}")
                except Exception as e:
                    print(f"[red]Failed[/red] {url}: {e}")
        return results
