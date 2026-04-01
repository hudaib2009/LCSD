import os
from base64 import b64encode

import pytest
from httpx import ASGITransport, AsyncClient

from backend.app.main import create_app


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_health_endpoint_returns_success() -> None:
    os.environ["CSD_SKIP_MODEL_LOAD"] = "1"
    os.environ.pop("CSD_BASIC_AUTH_USERNAME", None)
    os.environ.pop("CSD_BASIC_AUTH_PASSWORD", None)
    os.environ["CSD_ALLOW_UNAUTHENTICATED"] = "0"
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json()["ok"] is True


@pytest.mark.anyio
async def test_protected_endpoint_requires_auth_when_configured() -> None:
    os.environ["CSD_SKIP_MODEL_LOAD"] = "1"
    os.environ["CSD_ALLOW_UNAUTHENTICATED"] = "0"
    os.environ["CSD_BASIC_AUTH_USERNAME"] = "clinician"
    os.environ["CSD_BASIC_AUTH_PASSWORD"] = "s3cret"
    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/infer", json={})

    assert response.status_code == 401


@pytest.mark.anyio
async def test_protected_endpoint_accepts_valid_basic_auth() -> None:
    os.environ["CSD_SKIP_MODEL_LOAD"] = "1"
    os.environ["CSD_ALLOW_UNAUTHENTICATED"] = "0"
    os.environ["CSD_BASIC_AUTH_USERNAME"] = "clinician"
    os.environ["CSD_BASIC_AUTH_PASSWORD"] = "s3cret"
    app = create_app()
    transport = ASGITransport(app=app)
    auth_header = b64encode(b"clinician:s3cret").decode("ascii")

    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            "/infer",
            json={},
            headers={"Authorization": f"Basic {auth_header}"},
        )

    assert response.status_code != 401
