from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple
from rich import print

import chromadb
from sentence_transformers import SentenceTransformer

from .utils import build_youtube_link
from .llm import make_llm, ChatMessage


class QAClient:
    """Client for querying and analyzing indexed transcript data."""
    
    def __init__(
        self,
        index_dir: Path,
        collection_name: str = "youtube_audio",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        """Initialize the QA client with access to an existing ChromaDB collection.
        
        Args:
            index_dir: Directory containing the ChromaDB persistent storage.
            collection_name: Name of the ChromaDB collection to query.
            embedding_model: SentenceTransformer model name used for query embeddings.
        """
        self.client = chromadb.PersistentClient(path=str(index_dir))
        self.collection = self.client.get_collection(collection_name)
        self.encoder = SentenceTransformer(embedding_model)

    def ask(self, question: str, top_k: int = 5, story_mode: bool = True) -> None:
        """Ask a question and display relevant transcript excerpts.
        
        Args:
            question: The question to ask.
            top_k: Number of most relevant chunks to retrieve.
            story_mode: If True, sort results chronologically instead of by relevance.
        """
        qv = self.encoder.encode([question], normalize_embeddings=True)[0].tolist()
        res = self.collection.query(query_embeddings=[qv], n_results=top_k)
        docs: List[str] = res.get("documents", [[]])[0]
        metas: List[Dict[str, Any]] = res.get("metadatas", [[]])[0]
        if story_mode:
            # Sort results by global timeline when available
            def sort_key(m: Dict[str, Any]) -> int:
                gs = m.get("global_start")
                if isinstance(gs, int) and gs > 0:
                    return gs
                return int(m.get("upload_timestamp", 0)) + int(m.get("start", 0))

            order = sorted(range(len(metas)), key=lambda i: sort_key(metas[i]))
            docs = [docs[i] for i in order]
            metas = [metas[i] for i in order]
        print(f"\n[bold]Q:[/bold] {question}\n")
        for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
            video_id = meta.get("video_id")
            start = int(meta.get("start", 0))
            end = int(meta.get("end", 0))
            url = meta.get("url") or build_youtube_link(video_id, start)
            title = meta.get("title") or video_id
            gs = meta.get("global_start")
            timeline_suffix = f" @t={gs}" if isinstance(gs, int) and gs > 0 else ""
            print(f"[bold]{i}.[/bold] {title} [{start}-{end}]{timeline_suffix}\n{doc[:400]}…\n{url if url else build_youtube_link(video_id, start)}\n")

    def analyze(self, question: str, top_k: int = 12, provider: str = "auto", model: str | None = None) -> None:
        """Analyze a question using LLM reasoning over relevant transcript excerpts.
        
        Args:
            question: The question to analyze.
            top_k: Number of most relevant chunks to use as context.
            provider: LLM provider to use for analysis ('auto' or 'openai').
            model: Specific model name to use for analysis.
        """
        qv = self.encoder.encode([question], normalize_embeddings=True)[0].tolist()
        res = self.collection.query(query_embeddings=[qv], n_results=top_k)
        docs: List[str] = res.get("documents", [[]])[0]
        metas: List[Dict[str, Any]] = res.get("metadatas", [[]])[0]
        # Order by global timeline
        def sort_key(m: Dict[str, Any]) -> int:
            gs = m.get("global_start")
            if isinstance(gs, int) and gs > 0:
                return gs
            return int(m.get("upload_timestamp", 0)) + int(m.get("start", 0))
        order = sorted(range(len(metas)), key=lambda i: sort_key(metas[i]))
        docs = [docs[i] for i in order]
        metas = [metas[i] for i in order]

        # Build context with citations
        context_lines: List[str] = []
        for doc, meta in zip(docs, metas):
            video_id = meta.get("video_id")
            start = int(meta.get("start", 0))
            title = meta.get("title") or video_id
            url = meta.get("url") or build_youtube_link(video_id, start)
            context_lines.append(f"[{title} @ {start}s] {doc}")
            context_lines.append(f"Source: {url}")
        context = "\n\n".join(context_lines)

        system = ChatMessage(
            role="system",
            content=(
                "You are a meticulous analyst of long-form RPG session transcripts. "
                "Synthesize and infer across all provided excerpts. "
                "Cite specific sessions/timecodes inline when making claims (use the provided Source URLs). "
                "If uncertain, say so. Keep the answer structured and concise."
            ),
        )
        user = ChatMessage(
            role="user",
            content=(
                f"Question:\n{question}\n\n"
                f"Context (chronological excerpts with sources):\n{context}"
            ),
        )
        llm = make_llm(provider, model)
        answer = llm.chat([system, user], max_tokens=800)
        print(f"\n[bold]Q:[/bold] {question}\n\n[bold]Analysis:[/bold]\n{answer}\n")
