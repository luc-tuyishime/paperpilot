from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app


def test_health_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_uses_injected_settings() -> None:
    """Shows why we used Depends(): a test can replace the settings without touching .env."""
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: Settings(environment="ci")
    client = TestClient(app)

    response = client.get("/health")

    assert response.json()["environment"] == "ci"
