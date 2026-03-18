"""
ChromaDB vector store with sentence-transformers embeddings.

Uses:
  - chromadb.PersistentClient for local persistence
  - SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2') for embeddings
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from ingestion import TextChunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "construction_market_docs"
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

# ---------------------------------------------------------------------------
# Singleton helpers (cache within process)
# ---------------------------------------------------------------------------

_embed_model: SentenceTransformer | None = None
_chroma_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _chroma_client


def _get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = _get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts and return dense vectors."""
    model = _get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.tolist()


def add_chunks(chunks: list[TextChunk]) -> int:
    """
    Embed and store chunks in ChromaDB.
    Returns the number of chunks actually added (skips duplicates by ID).
    """
    if not chunks:
        return 0

    collection = _get_collection()

    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)

    ids: list[str] = []
    metadatas: list[dict] = []
    documents: list[str] = []

    for chunk, emb in zip(chunks, embeddings):
        chunk_id = (
            f"{chunk.source_filename}__chunk_{chunk.chunk_index}"
        )
        ids.append(chunk_id)
        metadatas.append(chunk.to_metadata())
        documents.append(chunk.text)

    # Upsert to handle re-ingestion of same file gracefully
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(chunks)


def query_chunks(
    query: str,
    n_results: int = 10,
    country_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-n most relevant chunks for a query string.
    Optionally filter by country metadata.
    Returns list of dicts with keys: text, score, metadata.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    query_embedding = embed_texts([query])[0]

    where: dict | None = None
    if country_filter:
        where = {"country": {"$eq": country_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    output: list[dict[str, Any]] = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for doc, meta, dist in zip(docs, metas, distances):
        output.append(
            {
                "text": doc,
                "score": 1 - dist,  # cosine similarity
                "metadata": meta,
            }
        )

    return output


def get_document_stats() -> dict[str, Any]:
    """Return summary statistics about stored documents."""
    collection = _get_collection()
    total = collection.count()

    if total == 0:
        return {"total_chunks": 0, "documents": {}}

    # Retrieve all metadata to compute per-document stats
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas", [])

    doc_stats: dict[str, dict] = {}
    for meta in metadatas:
        fname = meta.get("source_filename", "unknown")
        if fname not in doc_stats:
            doc_stats[fname] = {
                "chunk_count": 0,
                "country": meta.get("country", ""),
                "upload_date": meta.get("upload_date", ""),
            }
        doc_stats[fname]["chunk_count"] += 1

    return {"total_chunks": total, "documents": doc_stats}


def delete_document(filename: str) -> int:
    """Delete all chunks associated with a specific file. Returns deleted count."""
    collection = _get_collection()
    result = collection.get(where={"source_filename": {"$eq": filename}})
    ids = result.get("ids", [])
    if ids:
        collection.delete(ids=ids)
    return len(ids)
