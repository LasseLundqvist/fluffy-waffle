"""
Document ingestion pipeline:
  1. Parse PDFs with PyMuPDF (fitz)
  2. Chunk text with a recursive character splitter
  3. Return chunks with metadata for storage in ChromaDB
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TextChunk:
    text: str
    chunk_index: int
    source_filename: str
    country: str
    upload_date: str
    page_numbers: list[int] = field(default_factory=list)

    def to_metadata(self) -> dict:
        return {
            "source_filename": self.source_filename,
            "country": self.country,
            "upload_date": self.upload_date,
            "chunk_index": self.chunk_index,
            "page_numbers": ",".join(str(p) for p in self.page_numbers),
        }


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def parse_pdf_bytes(pdf_bytes: bytes, filename: str) -> list[tuple[int, str]]:
    """
    Parse a PDF from raw bytes.
    Returns a list of (page_number, page_text) tuples.
    Uses PyMuPDF (fitz).
    """
    import fitz  # PyMuPDF

    pages: list[tuple[int, str]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")  # plain text extraction
            # Also attempt to extract tables as plain text
            try:
                tabs = page.find_tables()
                for tab in tabs.tables:
                    rows = tab.extract()
                    for row in rows:
                        cleaned = "\t".join(
                            str(cell) if cell is not None else "" for cell in row
                        )
                        text += "\n" + cleaned
            except Exception:
                pass  # table extraction is best-effort
            if text.strip():
                pages.append((page_num, text))
    return pages


# ---------------------------------------------------------------------------
# Text chunking (recursive character splitter, no external deps)
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Recursive character-based splitter similar to LangChain's
    RecursiveCharacterTextSplitter, but pure Python.
    chunk_size and chunk_overlap are measured in whitespace-tokenised words
    (a good proxy for tokens without requiring tiktoken at this stage).
    """
    if separators is None:
        separators = _SEPARATORS

    def word_len(s: str) -> int:
        return len(s.split())

    def _split(text: str, seps: list[str]) -> list[str]:
        if not seps:
            return [text]
        sep = seps[0]
        splits = text.split(sep) if sep else list(text)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for fragment in splits:
            flen = word_len(fragment)
            if current_len + flen + (1 if current else 0) > chunk_size and current:
                chunks.append(sep.join(current))
                # Overlap: keep tail of current in next chunk
                overlap_words: list[str] = []
                overlap_len = 0
                for part in reversed(current):
                    plen = word_len(part)
                    if overlap_len + plen <= chunk_overlap:
                        overlap_words.insert(0, part)
                        overlap_len += plen
                    else:
                        break
                current = overlap_words
                current_len = overlap_len

            current.append(fragment)
            current_len += flen

        if current:
            chunks.append(sep.join(current))

        # Recursively split chunks that are still too large
        result: list[str] = []
        for chunk in chunks:
            if word_len(chunk) > chunk_size and len(seps) > 1:
                result.extend(_split(chunk, seps[1:]))
            else:
                result.append(chunk)
        return result

    raw_chunks = _split(text, separators)
    return [c.strip() for c in raw_chunks if c.strip()]


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_bytes: bytes,
    filename: str,
    country: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> list[TextChunk]:
    """
    Full pipeline: parse PDF → chunk → return TextChunk list.
    """
    upload_date = datetime.now().isoformat()
    pages = parse_pdf_bytes(pdf_bytes, filename)

    # Combine all pages into one string while tracking page boundaries
    full_text_parts: list[tuple[int, str]] = pages  # (page_num, text)

    # Build chunks with page tracking
    chunks: list[TextChunk] = []
    chunk_index = 0

    for page_num, page_text in full_text_parts:
        page_chunks = _split_text(page_text, chunk_size, chunk_overlap)
        for chunk_text in page_chunks:
            if len(chunk_text.split()) < 10:
                continue  # skip near-empty chunks
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    source_filename=filename,
                    country=country,
                    upload_date=upload_date,
                    page_numbers=[page_num],
                )
            )
            chunk_index += 1

    return chunks
