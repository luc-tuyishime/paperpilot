from collections.abc import Callable

from fastapi.testclient import TestClient

from app.services.vector_store import VectorStore

PAGE_1 = "The Transformer relies entirely on attention mechanisms. " * 10
PAGE_2 = "Generative adversarial networks train a generator against a discriminator. " * 10


def upload(client: TestClient, data: bytes, name: str = "paper.pdf") -> dict:  # type: ignore[type-arg]
    response = client.post("/documents", files={"file": (name, data, "application/pdf")})
    return {"status_code": response.status_code, **response.json()}


def test_upload_ingests_in_background(
    client: TestClient, store: VectorStore, make_pdf: Callable[..., bytes]
) -> None:
    created = upload(client, make_pdf([PAGE_1, PAGE_2]))
    assert created["status_code"] == 201
    assert created["status"] == "pending"  # the response returns before ingestion runs

    # TestClient runs background tasks right after the response, so it's done now.
    doc = client.get(f"/documents/{created['id']}").json()
    assert doc["status"] == "ready", doc["error"]
    assert doc["title"] == "A Test Paper"
    assert doc["num_pages"] == 2
    assert doc["num_chunks"] == store.count() > 2


def test_same_pdf_twice_is_idempotent(
    client: TestClient, store: VectorStore, make_pdf: Callable[..., bytes]
) -> None:
    pdf = make_pdf([PAGE_1])
    first = upload(client, pdf, "a.pdf")
    second = upload(client, pdf, "copy-of-a.pdf")

    assert second["status_code"] == 200  # existing document returned, not created
    assert second["id"] == first["id"]
    assert len(client.get("/documents").json()) == 1
    chunks_before = store.count()

    client.post(f"/documents/{first['id']}/reingest")  # run the whole pipeline again
    assert store.count() == chunks_before  # no duplicate vectors


def test_rejects_non_pdf(client: TestClient) -> None:
    response = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 422


def test_pdf_without_text_is_marked_failed(
    client: TestClient, make_pdf: Callable[..., bytes]
) -> None:
    created = upload(client, make_pdf([""]))
    doc = client.get(f"/documents/{created['id']}").json()
    assert doc["status"] == "failed"
    assert "no extractable text" in doc["error"]


def test_search_finds_the_right_page(client: TestClient, make_pdf: Callable[..., bytes]) -> None:
    upload(client, make_pdf([PAGE_1, PAGE_2]))

    hits = client.get("/search", params={"q": "generator and discriminator"}).json()

    assert hits[0]["page"] == 2
    assert "discriminator" in hits[0]["text"]


def test_invalid_arxiv_id_is_rejected(client: TestClient) -> None:
    response = client.post("/documents/arxiv", json={"arxiv_id": "not-an-id"})
    assert response.status_code == 422
