from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Dict, Any, List, Optional
import json
from rich import print
from tqdm import tqdm
from datetime import datetime
import os
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Only for type hints; actual import happens after DLL paths are set
    from faster_whisper import WhisperModel  # noqa: F401

from .utils import ensure_dir, get_video_id_from_audio


class Transcriber:
    def __init__(
        self,
        transcripts_dir: Path,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: Optional[str] = None,
        beam_size: int = 1,
        overwrite: bool = False,
        archive_existing: bool = False,
        archive_subdir: str = "old",
        enable_diarization: bool = False,
        diarization_token: Optional[str] = None,
    ) -> None:
        self.transcripts_dir = transcripts_dir
        ensure_dir(self.transcripts_dir)
        self.model_size = model_size
        self.device = device
        self.beam_size = beam_size
        self.overwrite = overwrite
        self.archive_existing = archive_existing
        self.archive_subdir = archive_subdir
        self.enable_diarization = enable_diarization
        self.diarization_token = diarization_token
        # On Windows + CUDA, ensure CUDA/cuDNN DLLs are discoverable
        if os.name == "nt" and self.device == "cuda":
            def _add_dir(path: Path) -> None:
                try:
                    real = path.resolve()
                    if real.exists():
                        os.add_dll_directory(str(real))  # type: ignore[attr-defined]
                        os.environ["PATH"] = f"{real};{os.environ.get('PATH','')}"
                except Exception:
                    pass

            def _add_from_namespace(pkg_name: str) -> None:
                try:
                    spec = importlib.util.find_spec(pkg_name)
                except Exception:
                    spec = None
                if not spec or not getattr(spec, "submodule_search_locations", None):
                    return
                for loc in spec.submodule_search_locations or []:  # type: ignore[attr-defined]
                    base = Path(loc)
                    for cand in [base, base / "bin", base / "lib", base / "lib" / "x64"]:
                        _add_dir(cand)
                    # Fall back to searching for specific DLLs within the package tree
                    try:
                        dll_parent_dirs = set()
                        patterns = [
                            "cudnn_ops64_9.dll",
                            "cudnn_cnn64_9.dll",
                            "cudnn64_9.dll",
                            "cublas64*.dll",
                            "cudart64_*.dll",
                        ]
                        for pat in patterns:
                            for p in base.rglob(pat):
                                dll_parent_dirs.add(p.parent)
                        for d in dll_parent_dirs:
                            _add_dir(d)
                    except Exception:
                        pass

            for ns in ("nvidia.cudnn", "nvidia.cublas", "nvidia.cuda_runtime"):
                _add_from_namespace(ns)
        if compute_type is None:
            self.compute_type = "int8" if device == "cpu" else "float16"
        else:
            self.compute_type = compute_type

        # Import after DLL directories are configured on Windows
        from faster_whisper import WhisperModel  # type: ignore
        self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        
        # Initialize speaker diarization if enabled
        self.diarization_pipeline = None
        if self.enable_diarization:
            try:
                from pyannote.audio import Pipeline  # type: ignore
                
                # Prefer environment variable, fallback to parameter for backward compatibility
                token = self.diarization_token or os.getenv('HUGGINGFACE_TOKEN')
                
                if token:
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=token
                    )
                else:
                    # Try without token (may work for public models)
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1"
                    )
                print(f"[green]Loaded[/green] speaker diarization pipeline")
            except Exception as e:
                print(f"[yellow]Warning[/yellow] Failed to load diarization pipeline: {e}")
                print(f"[yellow]Tip[/yellow] You need to set HUGGINGFACE_TOKEN in your .env file")
                print(f"[yellow]Tip[/yellow] Get your token from https://huggingface.co/settings/tokens")
                self.enable_diarization = False

    def transcribe_audio_file(self, audio_file: Path) -> Path:
        video_id = get_video_id_from_audio(audio_file)
        out_path = self.transcripts_dir / f"{video_id}.json"
        if out_path.exists():
            if self.overwrite:
                if self.archive_existing:
                    archive_dir = self.transcripts_dir / self.archive_subdir
                    ensure_dir(archive_dir)
                    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
                    archived_path = archive_dir / f"{video_id}-{ts}.json"
                    try:
                        out_path.replace(archived_path)
                        print(f"[cyan]Archived[/cyan] {out_path} -> {archived_path}")
                    except Exception as e:
                        print(f"[yellow]Warning[/yellow] failed to archive {out_path}: {e}")
                else:
                    print(f"[yellow]Overwriting[/yellow] existing transcript: {out_path}")
            else:
                print(f"[cyan]Skip[/cyan] transcript exists: {out_path}")
                return out_path
        print(
            f"Transcribing {audio_file} with model={self.model_size} device={self.device} compute_type={self.compute_type} beam_size={self.beam_size} ..."
        )
        segments, info = self.model.transcribe(
            str(audio_file),
            beam_size=self.beam_size,
            vad_filter=True,
            word_timestamps=False,
        )
        
        # Perform speaker diarization if enabled
        speaker_mapping = {}
        if self.enable_diarization and self.diarization_pipeline:
            try:
                print(f"[cyan]Running[/cyan] speaker diarization...")
                diarization = self.diarization_pipeline(str(audio_file))
                
                # Build speaker mapping for each time segment
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    start_time = turn.start
                    end_time = turn.end
                    speaker_mapping[(start_time, end_time)] = speaker
                
                print(f"[green]Found[/green] {len(set(speaker_mapping.values()))} speakers")
            except Exception as e:
                print(f"[yellow]Warning[/yellow] Diarization failed: {e}")
        
        data: Dict[str, Any] = {"video_id": video_id, "segments": []}
        for s in segments:
            segment_data = {
                "text": s.text.strip(),
                "start": float(s.start),
                "end": float(s.end),
            }
            
            # Assign speaker if diarization was performed
            if speaker_mapping:
                segment_start = float(s.start)
                segment_end = float(s.end)
                
                # Find the speaker for this segment by checking overlap with diarization
                assigned_speaker = None
                max_overlap = 0
                
                for (dia_start, dia_end), speaker in speaker_mapping.items():
                    # Calculate overlap between segment and diarization turn
                    overlap_start = max(segment_start, dia_start)
                    overlap_end = min(segment_end, dia_end)
                    overlap = max(0, overlap_end - overlap_start)
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        assigned_speaker = speaker
                
                if assigned_speaker:
                    segment_data["speaker"] = assigned_speaker
            
            data["segments"].append(segment_data)
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[green]Wrote[/green] {out_path}")
        return out_path

    def transcribe_many(self, audio_dir: Path) -> List[Path]:
        audio_files = sorted(audio_dir.glob("*.mp3"))
        if not audio_files:
            print("[yellow]No audio files found[/yellow]")
            return []
        results: List[Path] = []
        for f in tqdm(audio_files, desc="Transcribing"):
            try:
                results.append(self.transcribe_audio_file(f))
            except Exception as e:
                print(f"[red]Failed[/red] {f}: {e}")
        return results
