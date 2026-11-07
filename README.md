# PaperPilot

An AI assistant that answers questions about research papers, with the sources cited (paper, page).
A personal learning project covering FastAPI, RAG, vector databases, evaluation, agents and observability.

## Run it

Needs Docker Desktop, [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com) (`ollama pull nomic-embed-text`).

```bash
cp .env.example .env
uv sync                                  # install dependencies into .venv
docker compose up -d postgres qdrant     # databases
uv run alembic upgrade head              # create tables
uv run uvicorn app.main:app --reload     # API on http://localhost:8000
uv run pytest -v                         # tests (no Docker or Ollama needed)
```

```bash
curl -F "file=@paper.pdf" localhost:8000/documents          # upload → ingested in background
curl localhost:8000/documents                               # status of each paper
curl "localhost:8000/search?q=what+is+multi-head+attention" # semantic search with page numbers
```

Open http://localhost:8000/docs for the interactive API docs (generated from the code).

## Stack (100% free)

| Need | Tool |
|---|---|
| API | FastAPI |
| LLM + embeddings | Ollama (Llama 3.1 8B, nomic-embed-text), or the Gemini free tier |
| Vector database | Qdrant (Docker) |
| Database / memory | PostgreSQL (Docker) |
| Agent | LangGraph |
| Evaluation | Ragas, run in GitHub Actions |
| Observability | Prometheus + Grafana (Docker) |
| Front end | Next.js |
| Deployment | Docker Compose, local Kubernetes (kind) |

## Roadmap

- [x] **Lesson 1.** Project skeleton, `/health`, tests, CI, Docker
- [x] **Day 1. Ingestion:** PostgreSQL + Alembic, PDF/arXiv upload, chunking, Ollama embeddings, Qdrant, background tasks, `/search`
- [ ] **Day 2. RAG with citations + evaluation in CI** (answers with the Gemini free tier)
- [ ] **Day 3. LangGraph agent with tools and memory** + architecture docs
- [ ] Later: Next.js chat, Grafana dashboard, Kubernetes manifests

## Layout

```
app/
  main.py              # builds the FastAPI app and plugs in the routers
  core/config.py       # settings from environment variables
  api/deps.py          # shared dependencies (DB, embedder, vector store), overridable in tests
  api/routes/          # HTTP layer: documents, search, health
  services/            # pdf, chunking, embeddings, vector_store, ingestion, arxiv
  repositories/        # database queries
  db/                  # engine, session, ORM models
  schemas/             # API request/response models
migrations/            # Alembic database migrations
tests/                 # pytest, with fakes for Postgres, Qdrant and Ollama
docs/                  # day-by-day guides, later ADRs
```
