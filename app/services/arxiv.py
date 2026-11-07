"""Download a paper's PDF from arXiv by its id."""

import httpx


def download_arxiv_pdf(arxiv_id: str, timeout: float = 60.0) -> bytes:
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    response = httpx.get(url, follow_redirects=True, timeout=timeout)
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        raise ValueError(f"arXiv did not return a PDF for {arxiv_id}")
    return response.content
