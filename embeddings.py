"""Embedding helpers for the Dice & Dough RAG service.

Uses an Ollama embedding model (default: embeddinggemma, multilingual).
"""

from __future__ import annotations

import os
import re
from typing import Iterable

import ollama

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

# EmbeddingGemma's retrieval instruction for queries. Documents are embedded
# as-is; this asymmetric setup gave the best Thai in-scope/out-of-scope margin
# in our calibration (in-scope min 0.505 vs out-of-scope max 0.477).
QUERY_PREFIX = os.getenv("EMBEDDING_QUERY_PREFIX", "task: search result | query: ")

_COLLOQUIAL_REPLACEMENTS = [
    ("เข้าเล่น", "ค่าเข้าร้าน"),
    ("กี่บาท", "ราคาเท่าไหร่"),
    ("เท่าไร", "เท่าไหร่"),
    ("ยังไง", "อย่างไร"),
    ("ทำไง", "ทำอย่างไร"),
    ("มั้ย", "ไหม"),
    ("รึเปล่า", "หรือไม่"),
]
_TRAILING_PARTICLES = re.compile(r"(อ่ะ|อะ|จ้ะ|จ้า|น้า|เนอะ|ค่ะ|ครับ)\s*$")


def normalize_query(text: str) -> str:
    """Light Thai normalization so colloquial questions retrieve better."""
    normalized = text.strip()
    for source, target in _COLLOQUIAL_REPLACEMENTS:
        normalized = normalized.replace(source, target)
    normalized = _TRAILING_PARTICLES.sub("", normalized)
    return normalized.strip() or text.strip()

_client: ollama.Client | None = None


def get_client() -> ollama.Client:
    global _client
    if _client is None:
        _client = ollama.Client(host=os.getenv("OLLAMA_HOST") or None)
    return _client


def embed_texts(texts: Iterable[str]) -> list[list[float]]:
    """Embed a list of texts, batching requests to Ollama."""
    items = [t for t in texts]
    if not items:
        return []

    vectors: list[list[float]] = []
    client = get_client()
    for start in range(0, len(items), BATCH_SIZE):
        batch = items[start : start + BATCH_SIZE]
        response = client.embed(model=EMBEDDING_MODEL, input=batch)
        vectors.extend(response["embeddings"])
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([QUERY_PREFIX + normalize_query(text)])[0]
