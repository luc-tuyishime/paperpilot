# PaperPilot

An AI assistant that answers questions about research papers, with the sources cited (paper, page).
A personal learning project covering FastAPI, RAG, vector databases, evaluation, agents and observability.

## Run it

```bash
uv sync                                  # install dependencies into .venv
uv run uvicorn app.main:app --reload     # start the API on http://localhost:8000
uv run pytest -v                         # run the tests
docker compose up --build                # or run it in Docker
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

- [ ] **Step 1. FastAPI + ingestion + Qdrant**
  - [x] Lesson 1: project skeleton, `/health`, tests, CI, Docker
  - [ ] Lesson 2: PostgreSQL + `/documents` endpoints (SQLAlchemy, Alembic)
  - [ ] Lesson 3: PDF/arXiv ingestion → chunks → embeddings → Qdrant (background tasks)
- [ ] **Step 2. RAG with citations + evaluation in CI**
- [ ] **Step 3. LangGraph agent with tools and memory**
- [ ] **Step 4. Next.js chat + Grafana dashboard + Kubernetes manifests**

## Layout

```
app/
  main.py              # builds the FastAPI app and plugs in the routers
  core/config.py       # settings from environment variables
  api/routes/          # one file per group of endpoints
tests/                 # pytest
docs/                  # lesson notes and, later, architecture decision records (ADRs)
```
