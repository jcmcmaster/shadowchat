from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from yt_audio_ai.downloader import YouTubeAudioDownloader
from yt_audio_ai.transcriber import Transcriber
from yt_audio_ai.indexer import Indexer
from yt_audio_ai.qa import QAClient
from yt_audio_ai.utils import read_info_json
from yt_audio_ai.utils import ensure_dir


def get_transcripts_dir() -> Path:
    """Get the transcripts directory path, potentially versioned based on configuration.
    
    Returns:
        Path to transcripts directory, which may include a version subdirectory
        if TRANSCRIPT_VERSION environment variable is set.
    """
    data_dir = Path("data")
    transcript_version = os.getenv("TRANSCRIPT_VERSION")
    if transcript_version:
        return data_dir / "transcripts" / transcript_version
    return data_dir / "transcripts"


DATA_DIR = Path("data")
AUDIO_DIR = DATA_DIR / "audio"
INDEX_DIR = DATA_DIR / "index"

# TRANSCRIPTS_DIR will be set in main() after loading environment variables
TRANSCRIPTS_DIR: Path


def cmd_download(args: argparse.Namespace) -> None:
    """Download audio from YouTube URLs and playlists.
    
    Args:
        args: Parsed command line arguments containing URL sources and options.
    """
    urls: List[str] = []
    # individual URLs
    if args.url:
        urls.extend(args.url)
    # file with URLs (one per line)
    if args.urls_file:
        urls.extend([l.strip() for l in Path(args.urls_file).read_text(encoding="utf-8").splitlines()])
    # playlist IDs -> convert to full playlist URLs
    if args.playlist_id:
        urls.extend([f"https://www.youtube.com/playlist?list={pid.strip()}" for pid in args.playlist_id])
    # playlist URLs
    if args.playlist_url:
        urls.extend(args.playlist_url)

    dl = YouTubeAudioDownloader(AUDIO_DIR)
    dl.download_many(urls)


def cmd_transcribe(args: argparse.Namespace) -> None:
    """Transcribe downloaded audio files to text using Whisper.
    
    Args:
        args: Parsed command line arguments containing transcription options.
    """
    tr = Transcriber(
        TRANSCRIPTS_DIR,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        overwrite=args.overwrite,
        archive_existing=args.archive,
        archive_subdir=args.archive_subdir,
        enable_diarization=args.enable_diarization,
        diarization_token=args.diarization_token,
    )
    tr.transcribe_many(AUDIO_DIR)


def cmd_index(args: argparse.Namespace) -> None:
    """Index transcript chunks into a vector database for semantic search.
    
    Args:
        args: Parsed command line arguments containing indexing options.
    """
    indexer = Indexer(
        INDEX_DIR,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    indexer.index_many(TRANSCRIPTS_DIR, AUDIO_DIR)


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question and display relevant transcript excerpts.
    
    Args:
        args: Parsed command line arguments containing the question and search options.
    """
    qa = QAClient(
        INDEX_DIR,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    qa.ask(args.question, top_k=args.top_k)


def cmd_chat(args: argparse.Namespace) -> None:
    """Start an interactive chat session for asking questions about transcripts.
    
    Args:
        args: Parsed command line arguments containing chat options and LLM settings.
    """
    qa = QAClient(
        INDEX_DIR,
        collection_name=args.collection,
        embedding_model=args.embedding_model,
    )
    print("Type your question and press Enter. Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if args.analysis:
            qa.analyze(q, top_k=args.top_k, provider=args.llm_provider, model=args.llm_model)
        else:
            qa.ask(q, top_k=args.top_k)


def cmd_stats(args: argparse.Namespace) -> None:
    """Display statistics about available transcripts and sessions.
    
    Args:
        args: Parsed command line arguments (unused for this command).
    """
    # Enumerate sessions from transcripts/audio info
    entries = []
    for p in TRANSCRIPTS_DIR.glob("*.json"):
        video_id = p.stem
        info = read_info_json(AUDIO_DIR / f"{video_id}.mp3")
        title = info.get("title") or video_id
        ts = info.get("timestamp")
        upload_date = info.get("upload_date")
        if not isinstance(ts, int) or ts <= 0:
            # fallback parse YYYYMMDD
            try:
                if isinstance(upload_date, str) and len(upload_date) == 8:
                    from datetime import datetime, timezone
                    dt = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
                    ts = int(dt.timestamp())
                else:
                    ts = 0
            except Exception:
                ts = 0
        entries.append((ts, title, video_id))
    entries.sort(key=lambda x: x[0])
    print(f"Total sessions: {len(entries)}")
    for i, (_, title, vid) in enumerate(entries, start=1):
        print(f"{i:02d}. {title} ({vid})")


def build_parser() -> argparse.ArgumentParser:
    """Build and configure the command line argument parser.
    
    Returns:
        Configured ArgumentParser with all subcommands and options.
    """
    p = argparse.ArgumentParser(
        description="YouTube audio QA (local). Set TRANSCRIPT_VERSION env var to organize transcripts in versioned subdirectories."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("download", help="Download audio for URLs or playlists")
    sp.add_argument("--url", action="append", help="YouTube video URL (repeatable)")
    sp.add_argument("--urls-file", help="Path to text file with one URL per line (videos or playlists)")
    sp.add_argument("--playlist-id", action="append", help="YouTube playlist ID (repeatable)")
    sp.add_argument("--playlist-url", action="append", help="Full YouTube playlist URL (repeatable)")
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser("transcribe", help="Transcribe downloaded audio")
    sp.add_argument("--model-size", default="small.en", help="Whisper model size (e.g., small.en, small, medium, large-v3)")
    sp.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to run transcription")
    sp.add_argument(
        "--compute-type",
        default=None,
        help="Override compute type (e.g., int8, int8_float16, float16, float32); defaults based on device",
    )
    sp.add_argument("--beam-size", type=int, default=1, help="Decoding beam size (1 is fastest)")
    sp.add_argument("--overwrite", action="store_true", help="Re-generate transcripts even if they exist")
    sp.add_argument("--archive", action="store_true", help="When overwriting, move old transcript into archive subdir")
    sp.add_argument("--archive-subdir", default="old", help="Subdirectory under transcripts to place archived files")
    sp.add_argument("--enable-diarization", action="store_true", help="Enable speaker diarization using pyannote")
    sp.add_argument("--diarization-token", default=None, help="Hugging Face token for pyannote models (optional, prefers HUGGINGFACE_TOKEN env var)")
    sp.set_defaults(func=cmd_transcribe)

    sp = sub.add_parser("index", help="Chunk, embed, and index transcripts")
    sp.add_argument("--collection", default="youtube_audio")
    sp.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    sp.set_defaults(func=cmd_index)

    sp = sub.add_parser("ask", help="Query the index")
    sp.add_argument("--collection", default="youtube_audio")
    sp.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    sp.add_argument("--question", required=True)
    sp.add_argument("--top-k", type=int, default=5)
    sp.set_defaults(func=cmd_ask)

    sp = sub.add_parser("chat", help="Interactive chat against the index")
    sp.add_argument("--collection", default="youtube_audio")
    sp.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name",
    )
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--analysis", action="store_true", help="Use LLM analysis mode (aggregated reasoning)")
    sp.add_argument("--llm-provider", default="auto", choices=["auto", "openai"], help="LLM provider")
    sp.add_argument("--llm-model", default=None, help="LLM model name (provider-specific)")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("stats", help="Show session count and titles from transcripts")
    sp.set_defaults(func=cmd_stats)

    return p


def main() -> None:
    """Main entry point for the YouTube audio QA application.
    
    Loads environment variables, ensures required directories exist,
    parses command line arguments, and executes the appropriate command.
    """
    global TRANSCRIPTS_DIR
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Set TRANSCRIPTS_DIR after loading environment variables
    TRANSCRIPTS_DIR = get_transcripts_dir()
    
    ensure_dir(DATA_DIR)
    ensure_dir(AUDIO_DIR)
    ensure_dir(TRANSCRIPTS_DIR)
    ensure_dir(INDEX_DIR)

    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
