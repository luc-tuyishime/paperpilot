# Day 1 — Ingestion: PDF → chunks → embeddings → Qdrant

By tonight you can upload your 5 papers, watch them become `ready`, and search them by meaning.

```
POST /documents (PDF)                       ┌──────────── background task ────────────┐
   │ 1. check it's a PDF, hash the bytes    │ 4. extract text per page (pypdf)         │
   │ 2. same hash exists? → return it (200) │ 5. chunk each page (1000 chars, 150 over)│
   │ 3. save file + row status=PENDING ─────┼▶6. embed chunks in batches (Ollama)      │
   ▼                                        │ 7. delete old vectors, upsert new (Qdrant)│
 201 response, immediately                  │ 8. status=READY (or FAILED + error)      │
                                            └──────────────────────────────────────────┘
GET /documents/{id}   → watch status: pending → processing → ready
GET /search?q=...     → embed the question, nearest chunks from Qdrant, with page numbers
```

---

## Part A — PostgreSQL and the documents table (~2 h)

**Run it**

```bash
cp .env.example .env
uv sync
docker compose up -d postgres qdrant     # start the two databases only
uv run alembic upgrade head              # create the documents table
uv run uvicorn app.main:app --reload
```

Check the table exists: `docker compose exec postgres psql -U paperpilot -c "\d documents"`

**Read, in this order**

1. `app/db/session.py`: engine (connection pool) → session factory → `get_db` (one session per request, always closed).
2. `app/db/models.py`: the `Document` table. Notice `content_hash` is **UNIQUE**, and notice the `status` column.
3. `migrations/versions/0001_create_documents_table.py`: the SQL Alembic generated from the model. Like TypeORM migrations: the schema changes are versioned in git.
4. `app/repositories/documents.py`: the only file that queries documents.
5. `app/schemas/document.py`: API shapes. Kept separate from the ORM model on purpose.

**Understand**: why have both `models.py` *and* `schemas.py`? The database can change (a new internal column) without changing what clients see.

## Part B — The ingestion pipeline (~4 h)

**Run it**

```bash
ollama pull nomic-embed-text     # once, about 270 MB
ollama serve                     # if it isn't already running in the menu bar
```

Upload a paper:

```bash
curl -F "file=@research-paper/1706.03762.pdf" localhost:8000/documents
```

You get `201` with `"status": "pending"`. Copy the `id`, then poll:

```bash
curl localhost:8000/documents/<id>
```

It should become `"ready"` with `num_pages` and `num_chunks` filled in. Now upload all five:

```bash
for f in research-paper/*.pdf; do curl -s -F "file=@$f" localhost:8000/documents; echo; done
curl -s localhost:8000/documents | python3 -m json.tool
```

Open the Qdrant dashboard at http://localhost:6333/dashboard → collection `chunks` → look at a point's **payload** (document_id, title, page, text).

**Read, in this order**

1. `app/services/pdf.py`: text per page, plus some light cleaning (re-joining hyphenated words).
2. `app/services/chunking.py`: why per page (citations need page numbers) and why overlap (a sentence cut at the border survives in the next chunk).
3. `app/services/embeddings.py`: the `Embedder` **Protocol** is an interface. Note the `search_document:` / `search_query:` prefixes that nomic-embed-text expects.
4. `app/services/vector_store.py`: `point_id()` uses `uuid5(document_id, chunk_index)`, so the same chunk always gets the same id.
5. `app/services/ingestion.py`: the pipeline. It opens **its own** DB session, and every failure becomes `FAILED` + `error`.
6. `app/api/routes/documents.py`: `BackgroundTasks`, and status 201 (new) vs 200 (already exists).

**Test idempotency yourself**

```bash
curl -F "file=@research-paper/1706.03762.pdf" localhost:8000/documents   # → 200, same id
curl -X POST localhost:8000/documents/<id>/reingest                       # pipeline runs again
```

In the dashboard, the number of points in `chunks` doesn't change. Two layers protect against duplicates:
- **Same file twice**: the SHA-256 hash is UNIQUE in PostgreSQL, so the existing document is returned.
- **Same document re-processed**: its old vectors are deleted, and point ids are deterministic.

## Part C — Search and tests (~2 h)

```bash
curl -s "localhost:8000/search?q=what+BLEU+score+did+the+Transformer+get" | python3 -m json.tool
curl -s "localhost:8000/search?q=how+is+the+generator+trained" | python3 -m json.tool
```

Do the top hits come from the right paper and page? Check them against the 15 questions you wrote.

Also try the arXiv import: `curl -X POST localhost:8000/documents/arxiv -H "Content-Type: application/json" -d '{"arxiv_id":"1409.0473"}'`. If you already uploaded that PDF, you get back the existing document.

**Tests**: `uv run pytest -v` gives 11 passing tests. Read `tests/conftest.py`: every external system is replaced with a fake using `dependency_overrides` (SQLite, in-memory Qdrant, a word-count embedder). That's lesson 1's `Depends` paying off.

Then commit and push:

```bash
git add . && git commit -m "Day 1: ingestion pipeline (Postgres, Qdrant, Ollama embeddings)" && git push
```

---

## Interview talking points (say these out loud once)

1. **Why a background task?** Embedding a 15-page paper takes seconds to minutes. The API answers right away and the client polls the status. *At scale:* move to a real queue (Redis + arq/Celery, or Pub/Sub on GCP), so a crash doesn't lose jobs and workers scale separately from the API. FastAPI's `BackgroundTasks` runs in the same process: if the server restarts mid-ingestion, the job is lost. That's acceptable for a demo, and a known limit.
2. **Why PostgreSQL *and* Qdrant?** PostgreSQL is the source of truth: documents, status, users later. Qdrant is a search index that can be rebuilt from PostgreSQL plus the files. You could use pgvector to have one fewer system; Qdrant gives faster approximate search (HNSW), payload filtering and horizontal scaling.
3. **Idempotency**: content hash plus deterministic point ids. Retries are safe, which matters as soon as a queue retries failed jobs.
4. **Chunking choices**: 1000 characters (about 250 tokens) with 150 overlap, per page for citations. *Trade-off:* small chunks are precise but lose context; large ones keep context but dilute the embedding. Day 2's evaluation set is how you'd *measure* the best size instead of guessing.
5. **Failure handling**: a scanned PDF has no text, so the document becomes `FAILED` with a readable error. Nothing crashes silently. *Next:* OCR for scanned PDFs, a retry limit, a dead-letter list.

## If something breaks

| Symptom | Fix |
|---|---|
| `connection refused` on 5432 or 6333 | `docker compose up -d postgres qdrant`, check with `docker compose ps` |
| `relation "documents" does not exist` | `uv run alembic upgrade head` |
| status `failed`, error mentions `11434` | Ollama isn't running: open the Ollama app or run `ollama serve` |
| error `model "nomic-embed-text" not found` | `ollama pull nomic-embed-text` |
| everything is slow | Close Chrome tabs. 8 GB is tight with Docker running; give Docker at most 3 GB in Docker Desktop → Settings → Resources |
