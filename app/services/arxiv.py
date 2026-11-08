"""arXiv: download a paper's PDF and read its metadata (the real title)."""

import re
import xml.etree.ElementTree as ET

import httpx

ATOM = "{http://www.w3.org/2005/Atom}"


def base_id(arxiv_id: str) -> str:
    """'1706.03762v7' -> '1706.03762'. The same paper has several versions."""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def download_arxiv_pdf(arxiv_id: str, timeout: float = 60.0) -> bytes:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = httpx.get(url, follow_redirects=True, timeout=timeout)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"arXiv did not return a PDF for {arxiv_id}")
    return response.content


def fetch_arxiv_title(arxiv_id: str, timeout: float = 30.0) -> str | None:
    """Title from arXiv's public API (Atom XML). None if unavailable: the caller falls back."""
    try:
        response = httpx.get(
            "https://export.arxiv.org/api/query",
            params={"id_list": base_id(arxiv_id)},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
        title = ET.fromstring(response.text).findtext(f"{ATOM}entry/{ATOM}title")
    except (httpx.HTTPError, ET.ParseError):
        return None
    return " ".join(title.split()) if title else None  # titles contain line breaks
