"""Entry point. `uvicorn app.main:app` imports this file and serves `app`."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agent.runtime import close_pools
from app.api.routes import agent, ask, documents, health, search
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield  # the app runs here
    close_pools()  # on shutdown: close the agent's database connections


def create_app() -> FastAPI:
    """App factory: builds a fresh FastAPI app. Tests call it to get a clean instance."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    app.include_router(agent.router)
    return app


app = create_app()
