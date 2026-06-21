"""Smoke-тесты REST-сервиса на лёгкой baseline-модели."""
import os

os.environ["MODEL"] = "baseline"

from fastapi.testclient import TestClient  # noqa: E402

from src.mapping import GROUPS, RESTRICTIONS  # noqa: E402
from src.serve import app  # noqa: E402


def test_ping():
    with TestClient(app) as client:
        assert client.get("/ping").json()["status"] == "ok"


def test_groups_cover_all():
    with TestClient(app) as client:
        body = client.get("/groups").json()
    assert len(body) == len(GROUPS)
    assert {row["group"] for row in body} == set(GROUPS)


def test_predict_shape():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"text": "изжога и тяжесть в животе после еды"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["group"] in GROUPS
    assert body["restriction"] == RESTRICTIONS[body["group"]]
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_empty():
    with TestClient(app) as client:
        assert client.post("/predict", json={"text": ""}).status_code == 422
