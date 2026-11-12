"""Chunking and ChromaDB vector store for the Dice & Dough knowledge base."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from embeddings import embed_query, embed_texts

CHROMA_DIR = os.getenv("CHROMA_DIR", str(Path(__file__).parent / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dicedough_knowledge")
KNOWLEDGE_DIR = Path(os.getenv("KNOWLEDGE_DIR", str(Path(__file__).parent / "knowledge")))

MAX_CHARS = 400
OVERLAP = 60

_lock = threading.Lock()
_client: chromadb.api.ClientAPI | None = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def split_text(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP) -> list[str]:
    """Split Thai text safely without relying on whitespace.

    Paragraphs/sections are merged up to ``max_chars``. A very long paragraph
    is split into overlapping character windows (~400 chars, ~60 overlap).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            chunks.append(buffer.strip())
        buffer = ""

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            if len(candidate) <= max_chars:
                buffer = candidate
            else:
                flush()
                buffer = paragraph
            continue

        # Long paragraph: flush what we have, then slide a character window.
        flush()
        step = max_chars - overlap
        start = 0
        while start < len(paragraph):
            piece = paragraph[start : start + max_chars].strip()
            if piece:
                chunks.append(piece)
            if start + max_chars >= len(paragraph):
                break
            start += step

    flush()

    # Merge tiny title-only chunks into the next section so headings carry
    # searchable content instead of consuming a retrieval slot on their own.
    merged: list[str] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        if (
            len(current) < 80
            and index + 1 < len(chunks)
            and len(current) + 2 + len(chunks[index + 1]) <= int(max_chars * 1.2)
        ):
            merged.append(f"{current}\n\n{chunks[index + 1]}")
            index += 2
        else:
            merged.append(current)
            index += 1
    return [c for c in merged if c]


def index_text(text: str, source: str) -> int:
    """Replace all chunks for ``source`` with freshly embedded chunks."""
    chunks = split_text(text)
    if not chunks:
        raise ValueError("Document has no indexable text")

    collection = get_collection()
    with _lock:
        collection.delete(where={"source": source})
        vectors = embed_texts(chunks)
        collection.add(
            ids=[f"{source}:{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=vectors,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
    return len(chunks)


def index_file(path: Path, source: str | None = None) -> int:
    source = source or path.name
    return index_text(path.read_text(encoding="utf-8-sig"), source)


def index_directory(directory: Path | str = KNOWLEDGE_DIR) -> dict[str, int]:
    directory = Path(directory)
    results: dict[str, int] = {}
    for path in sorted(directory.glob("*.txt")):
        results[path.name] = index_file(path)
    return results


def search(question: str, top_k: int = 4) -> list[dict[str, Any]]:
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return []

    vector = embed_query(question)
    result = collection.query(
        query_embeddings=[vector],
        n_results=min(max(top_k, 1), total),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[dict[str, Any]] = []
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for document, metadata, distance in zip(documents, metadatas, distances):
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        hits.append(
            {
                "source": (metadata or {}).get("source", "unknown"),
                "chunk": int((metadata or {}).get("chunk", 0)),
                "text": document,
                "similarity": round(similarity, 4),
            }
        )
    return hits


def search_multi(question: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve for the full question plus each "และ"-joined clause.

    Multi-part Thai questions often retrieve well for each clause but not for
    the concatenation. Merging candidates keeps the best chunks for every part.
    """
    queries = [question]
    parts = [p.strip() for p in re.split(r"\s*และ\s*", question) if len(p.strip()) >= 5]
    if len(parts) > 1:
        queries.extend(parts)

    collected: dict[tuple[str, int], dict[str, Any]] = {}
    for query in queries:
        for hit in search(query, top_k=top_k):
            key = (hit["source"], hit["chunk"])
            previous = collected.get(key)
            if previous is None or hit["similarity"] > previous["similarity"]:
                collected[key] = hit
    return sorted(collected.values(), key=lambda item: item["similarity"], reverse=True)[:top_k]


def stats() -> dict[str, Any]:
    collection = get_collection()
    sources = collection.get(include=["metadatas"]).get("metadatas") or []
    unique = {m.get("source") for m in sources if m}
    return {"documents": len(unique), "chunks": collection.count()}


def reset() -> None:
    global _client, _collection
    with _lock:
        if _collection is not None:
            _client.delete_collection(COLLECTION_NAME)
            _collection = None
