# Lesson 1 — The skeleton

Goal: a FastAPI app that runs, is tested, is checked by CI and runs in Docker.
No AI yet. Every later lesson plugs into this structure, so it's worth understanding it well.

## 1. How a request flows

```
GET /health
   → uvicorn (the web server)
   → app.main:app (the FastAPI app)
   → router from app/api/routes/health.py
   → FastAPI calls get_settings() because of Depends(...)
   → health() returns a HealthResponse
   → FastAPI validates it and sends JSON
```

## 2. The files, in reading order

**`pyproject.toml`** — the project's identity card: dependencies, plus config for ruff, mypy and pytest.
It's the Python equivalent of `package.json`. `uv sync` reads it and creates `.venv`.
`uv.lock` pins exact versions (like `package-lock.json`); commit it.

**`app/core/config.py`** — settings with pydantic-settings. `Settings()` reads
`PAPERPILOT_ENVIRONMENT` from the environment or `.env`, validates types, and falls back to defaults.
Same idea as `@nestjs/config`. `@lru_cache` makes it a singleton.

**`app/api/routes/health.py`** — an `APIRouter` (like a NestJS controller).
Two FastAPI ideas to remember:
- `response_model=HealthResponse`: the Pydantic model is the contract. It validates the output and
  appears in `/docs` automatically (that's the "generated OpenAPI" from the plan).
- `Depends(get_settings)`: dependency injection. Later we'll inject a database session and a Qdrant
  client the same way, and tests will swap them for fakes.

**`app/main.py`** — `create_app()` is an *app factory*. Each test calls it to get a clean app,
so tests don't leak state into each other.

**`tests/test_health.py`** — `TestClient` sends requests to the app in memory (no server needed).
The second test uses `app.dependency_overrides` — this is *why* we use `Depends`.

**`Dockerfile`** — copies dependency files before code so Docker caches the install layer.
**`docker-compose.yml`** — today just the API; lessons 2 and 3 add `postgres` and `qdrant` here.
**`.github/workflows/ci.yml`** — on every push: lint (ruff), format check, types (mypy strict), tests.

## 3. Do it yourself

1. Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. `uv sync`, then `uv run uvicorn app.main:app --reload`
3. Open http://localhost:8000/docs and try `/health` there.
4. Run `uv run pytest -v`, `uv run ruff check .`, `uv run mypy app`.
5. `docker compose up --build`, then `curl localhost:8000/health` (`environment` should say `docker`).
6. Create the public GitHub repo `paperpilot`, push, and watch the CI tab turn green.

## 4. Exercises (to check you understood)

1. Add a `PAPERPILOT_VERSION=0.2.0` line to `.env`. Restart. What does `/health` show? Why?
2. Change `status: str` in `HealthResponse` to `status: int`. Run the tests. Which tool catches it first, mypy or pytest?
3. Add a `GET /health/time` endpoint that returns the server time, with its own response model and a test.

## Next: lesson 2

PostgreSQL in Docker Compose, a `documents` table with SQLAlchemy, migrations with Alembic,
and `POST /documents` + `GET /documents` — the records that ingestion (lesson 3) will fill in.
