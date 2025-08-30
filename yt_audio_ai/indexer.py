from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timezone
import json
from rich import print

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from sentence_transformers import SentenceTransformer

from .utils import ensure_dir, read_info_json


def _default_chunker(segments: List[Dict[str, Any]], max_chars: int = 1500) -> List[List[Dict[str, Any]]]:
    """Split transcript segments into chunks based on character count.
    
    Args:
        segments: List of transcript segments, each with 'text', 'start', and 'end' keys.
        max_chars: Maximum number of characters per chunk.
        
    Returns:
        List of chunks, where each chunk is a list of segments.
    """
    chunks: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    total_len = 0
    for seg in segments:
        text = seg.get("text", "")
        if not text:
            continue
        if total_len + len(text) > max_chars and current:
            chunks.append(current)
            current = []
            total_len = 0
        current.append(seg)
        total_len += len(text) + 1
    if current:
        chunks.append(current)
    return chunks


def _chunk_to_text(chunk: List[Dict[str, Any]]) -> str:
    """Convert a chunk of segments to a single text string.
    
    Args:
        chunk: List of transcript segments with 'text' keys.
        
    Returns:
        Concatenated text from all segments in the chunk.
    """
    return " ".join(seg.get("text", "").strip() for seg in chunk).strip()


def _chunk_bounds(chunk: List[Dict[str, Any]]) -> Tuple[float, float]:
    """Get the start and end timestamps for a chunk of segments.
    
    Args:
        chunk: List of transcript segments with 'start' and 'end' keys.
        
    Returns:
        Tuple of (start_time, end_time) for the chunk.
    """
    return float(chunk[0]["start"]), float(chunk[-1]["end"])


class SentenceTransformerEmbedding(EmbeddingFunction):
    """ChromaDB embedding function using SentenceTransformers."""
    
    def __init__(self, model_name: str) -> None:
        """Initialize with a specific SentenceTransformer model.
        
        Args:
            model_name: Name of the SentenceTransformer model to use.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            List of embedding vectors as lists of floats.
        """
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]




class Indexer:
    """Indexes transcript chunks into a ChromaDB vector database for semantic search."""
    
    def __init__(
        self,
        index_dir: Path,
        collection_name: str = "youtube_audio",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the indexer with a ChromaDB collection.
        
        Args:
            index_dir: Directory for the persistent ChromaDB storage.
            collection_name: Name of the ChromaDB collection to use.
            embedding_model: SentenceTransformer model name for embeddings.
        """
        ensure_dir(index_dir)
        self.client = chromadb.PersistentClient(path=str(index_dir))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=SentenceTransformerEmbedding(embedding_model),
            metadata={"embedding_model": embedding_model},
        )

    def index_transcript(self, transcript_path: Path, audio_dir: Path) -> int:
        """Index a single transcript file into the vector database.
        
        Args:
            transcript_path: Path to the transcript JSON file.
            audio_dir: Directory containing the corresponding audio files and metadata.
            
        Returns:
            Number of chunks indexed from this transcript.
        """
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        video_id = data.get("video_id")
        segments = data.get("segments", [])
        if not segments:
            print(f"[yellow]No segments in {transcript_path}")
            return 0
        chunks = _default_chunker(segments)
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        audio_file = audio_dir / f"{video_id}.mp3"
        info = read_info_json(audio_file)
        title = info.get("title") or video_id
        original_url = info.get("webpage_url") or info.get("original_url")
        # Prefer exact epoch seconds if present; fall back to parsing YYYYMMDD
        upload_epoch: int = 0
        try:
            ts = info.get("timestamp")
            if isinstance(ts, (int, float)) and ts > 0:
                upload_epoch = int(ts)
            else:
                upload_date = info.get("upload_date")
                if isinstance(upload_date, str) and len(upload_date) == 8:
                    dt = datetime.strptime(upload_date, "%Y%m%d").replace(tzinfo=timezone.utc)
                    upload_epoch = int(dt.timestamp())
        except Exception:
            upload_epoch = 0

        for i, chunk in enumerate(chunks):
            text = _chunk_to_text(chunk)
            start, end = _chunk_bounds(chunk)
            documents.append(text)
            meta: Dict[str, Any] = {
                "video_id": video_id,
                "title": title,
                "start": start,
                "end": end,
                "url": original_url,
                "chunk_index": i,
            }
            # Persist upload date metadata when available
            if isinstance(info.get("upload_date"), str):
                meta["upload_date"] = info["upload_date"]
            if upload_epoch:
                meta["upload_timestamp"] = upload_epoch
                # Global timeline across videos using upload time as base
                try:
                    meta["global_start"] = int(upload_epoch + int(start))
                    meta["global_end"] = int(upload_epoch + int(end))
                except Exception:
                    pass
            metadatas.append(meta)
            ids.append(f"{video_id}:{i}")

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[green]Indexed[/green] {video_id} chunks={len(ids)}")
        return len(ids)

    def index_many(self, transcripts_dir: Path, audio_dir: Path) -> int:
        """Index all transcript files in a directory.
        
        Args:
            transcripts_dir: Directory containing transcript JSON files.
            audio_dir: Directory containing corresponding audio files and metadata.
            
        Returns:
            Total number of chunks indexed across all transcripts.
        """
        total = 0
        for p in sorted(transcripts_dir.glob("*.json")):
            try:
                total += self.index_transcript(p, audio_dir)
            except Exception as e:
                print(f"[red]Failed[/red] index {p}: {e}")
        return total
