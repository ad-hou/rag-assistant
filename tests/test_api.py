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


def test_upload_and_delete_documents(monkeypatch):
    monkeypatch.setattr(main, "ingest_files", lambda files: {
        "collection": "user_ab12cd34", "documents": [n for n, _ in files], "passages": 5})
    r = client.post("/documents", files=[("files", ("a.txt", b"contenu", "text/plain"))])
    assert r.status_code == 200 and r.json()["collection"] == "user_ab12cd34"
    assert r.json()["documents"] == ["a.txt"]
    monkeypatch.setattr(main, "delete_collection", lambda name: None)
    assert client.delete("/documents/user_ab12cd34").status_code == 204


def test_cannot_delete_the_demo_collection():
    assert client.delete("/documents/fastapi_code").status_code == 422


def test_upload_rejected_gives_422(monkeypatch):
    def refuse(files):
        raise main.DocumentError("format non pris en charge")

    monkeypatch.setattr(main, "ingest_files", refuse)
    r = client.post("/documents", files=[("files", ("a.png", b"x", "image/png"))])
    assert r.status_code == 422 and "format" in r.json()["detail"]


def test_user_collection_uses_generic_prompt(monkeypatch):
    seen = {}
    monkeypatch.setattr(main, "collection_exists", lambda name: True)
    monkeypatch.setattr(main, "answer", lambda q, **kw: seen.update(kw) or FAKE)
    r = client.post("/ask", json={"question": "Une vraie question", "collection": "user_ab12cd34"})
    assert r.status_code == 200 and seen["prompt"] == "generic"


def test_missing_user_collection_gives_404(monkeypatch):
    monkeypatch.setattr(main, "collection_exists", lambda name: False)
    r = client.post("/ask", json={"question": "Une vraie question", "collection": "user_zzzzzzzz"})
    assert r.status_code == 404
