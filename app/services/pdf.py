"""Text extraction: PDF file -> list of (page number, text)."""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as a human would cite it
    text: str


def _clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)  # join words hyphenated across lines: "atten-\ntion"
    text = re.sub(r"[ \t]+", " ", text)  # collapse runs of spaces
    return text.strip()


def extract_pages(path: Path) -> list[Page]:
    reader = PdfReader(path)
    pages = [
        Page(number=i + 1, text=_clean(p.extract_text() or "")) for i, p in enumerate(reader.pages)
    ]
    return [p for p in pages if p.text]  # skip empty pages (e.g. full-page figures)


def extract_title(path: Path, fallback: str) -> str:
    """Best-effort title: PDF metadata first, else the first line of page 1."""
    reader = PdfReader(path)
    meta_title = reader.metadata.title if reader.metadata else None
    if meta_title and len(meta_title.strip()) > 5:
        return meta_title.strip()[:500]
    first_page = reader.pages[0].extract_text() if reader.pages else ""
    first_line = next((ln.strip() for ln in (first_page or "").splitlines() if ln.strip()), "")
    return (first_line or fallback)[:500]
