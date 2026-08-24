import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.api.main import app
from app.config import reset_settings_cache


@pytest.fixture
def client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    database_url = (
        os.getenv("TEST_DATABASE_URL") or f"sqlite+pysqlite:///{tmp_path / 'test.sqlite3'}"
    )
    monkeypatch.setenv("DATABASE_URL", database_url)
    reset_settings_cache()
    with TestClient(app) as test_client:
        yield test_client
    reset_settings_cache()


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-API-Key": "dev-triage-api-key"}
