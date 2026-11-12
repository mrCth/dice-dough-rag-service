import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from main import app  # noqa: E402
from rag import REFUSAL  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["documents"] == 8
    assert data["chunks"] > 0
    assert data["chat_model"]
    assert data["embedding_model"]
    assert data["warmup_ok"] is True


def test_models(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert data["chat_model"] == "qwen3.5:4b"
    assert data["embedding_model"] == "embeddinggemma"
    assert data["refusal_message"] == REFUSAL


def test_search_returns_expected_thai_document(client):
    response = client.post(
        "/search",
        json={"question": "ค่าเข้าร้านวันธรรมดาเท่าไหร่", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["results"], "search should return at least one chunk"
    assert data["results"][0]["source"] == "pricing.txt"
    assert data["results"][0]["similarity"] > 0


def test_ask_returns_answer_and_sources(client):
    response = client.post(
        "/ask",
        json={"question": "ค่าเข้าร้านวันธรรมดาเท่าไหร่", "top_k": 5},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["refused"] is False
    assert "60" in data["answer"]
    assert data["sources"], "in-scope answer should include sources"
    assert any(source["source"] == "pricing.txt" for source in data["sources"])
    assert data["model"] == "qwen3.5:4b"
    assert data["latency_ms"] >= 0


def test_out_of_scope_question_is_refused(client):
    response = client.post("/ask", json={"question": "ราคาหุ้นวันนี้เท่าไหร่"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == REFUSAL
    assert data["refused"] is True
    assert data["sources"] == []


def test_empty_question_rejected(client):
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_invalid_top_k_rejected(client):
    response = client.post("/search", json={"question": "ค่าเข้าเท่าไหร่", "top_k": 99})
    assert response.status_code == 422


def test_document_path_traversal_rejected(client):
    response = client.post(
        "/documents",
        json={"name": "../x.txt", "text": "เนื้อหาทดสอบที่ยาวเพียงพอสำหรับการทดสอบระบบ"},
    )
    assert response.status_code == 422


def test_web_client_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ไดซ์แอนด์โดว์" in response.text
