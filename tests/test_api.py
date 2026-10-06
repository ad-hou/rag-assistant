from fastapi.testclient import TestClient

from api import main

client = TestClient(main.app)
FAKE = {"answer": "Utilisez `None` [1].", "refused": False, "retrieval_s": 0.4,
        "generation_s": 3.0,
        "sources": [{"score": 0.86, "source": "tutorial/query-params.md",
                     "section": "Query > Optional", "text": "Set default to None."}]}


def test_ask_returns_numbered_sources(monkeypatch):
    seen = {}

    def fake(question, **kw):
        seen.update(kw, question=question)
        return FAKE

    monkeypatch.setattr(main, "answer", fake)
    r = client.post("/ask", json={"question": "Comment faire ?", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"][0]["n"] == 1 and body["refused"] is False
    assert seen["k"] == 3 and seen["collection"] == "fastapi_code" and seen["prompt"] == "strict"


def test_validation_errors():
    assert client.post("/ask", json={"question": "a"}).status_code == 422
    assert client.post("/ask", json={"question": "Une vraie question", "k": 50}).status_code == 422
    assert client.post("/ask", json={"question": "Une vraie question",
                                     "prompt": "inconnu"}).status_code == 422


def test_llm_down_gives_503(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("Ollama ne répond pas")

    monkeypatch.setattr(main, "answer", boom)
    assert client.post("/ask", json={"question": "Une vraie question"}).status_code == 503
