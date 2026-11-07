import pytest

from app.services.chunking import chunk_pages
from app.services.pdf import Page


def test_short_page_is_one_chunk() -> None:
    chunks = chunk_pages([Page(1, "Attention is all you need.")], size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].page == 1


def test_long_page_is_split_with_overlap_and_keeps_page_numbers() -> None:
    words = " ".join(f"word{i}" for i in range(400))  # ~3000 chars
    chunks = chunk_pages([Page(1, "short page"), Page(2, words)], size=500, overlap=100)

    assert chunks[0].page == 1
    assert all(c.page == 2 for c in chunks[1:])
    assert all(len(c.text) <= 500 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))  # global, in order
    # overlap: the end of one chunk appears at the start of the next
    assert chunks[1].text[-30:].split()[-1] in chunks[2].text
    # no chunk starts or ends mid-word
    assert all(c.text.split()[0].startswith("word") for c in chunks[1:])
    assert all(c.text.split()[-1].removeprefix("word").isdigit() for c in chunks[1:])


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "x")], size=100, overlap=100)
