"""Chunking: split each page into overlapping pieces small enough to embed.

Why per page? So every chunk knows its page number, which we need for citations.
Why overlap? A sentence cut at a chunk border still appears whole in the next chunk.
"""

from dataclasses import dataclass

from app.services.pdf import Page


@dataclass(frozen=True)
class Chunk:
    index: int  # position in the whole document (0, 1, 2, ...)
    page: int
    text: str


def chunk_pages(pages: list[Page], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks: list[Chunk] = []
    for page in pages:
        start = 0
        text = page.text
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                # Prefer to cut at the end of a sentence or word, not mid-word.
                cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
                if cut > start + size // 2:
                    end = cut + 1
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(index=len(chunks), page=page.number, text=piece))
            if end >= len(text):
                break
            # Step back by `overlap`, then forward to the next word so we never start mid-word.
            next_start = end - overlap
            space = text.find(" ", next_start, end)
            start = space + 1 if space != -1 else next_start
    return chunks
