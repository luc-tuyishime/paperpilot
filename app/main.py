"""Entry point. `uvicorn app.main:app` imports this file and serves `app`."""

import logging

from fastapi import FastAPI

from app.api.routes import ask, documents, health, search
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    """App factory: builds a fresh FastAPI app. Tests call it to get a clean instance."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.version)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    app.include_router(ask.router)
    return app


app = create_app()
