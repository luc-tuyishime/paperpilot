"""Health endpoint: tells Docker, Kubernetes or a load balancer "the API process is alive"."""

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.config import Settings, get_settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """The response schema. FastAPI uses it to validate output and to document /docs."""

    status: str
    app: str
    version: str
    environment: str


@router.get("/health", response_model=HealthResponse)
def health(settings: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
    # Depends(get_settings) = dependency injection: FastAPI calls get_settings()
    # and passes the result in. Tests can swap it for fake settings.
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.version,
        environment=settings.environment,
    )
