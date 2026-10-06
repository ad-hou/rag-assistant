from src import generate

HITS = [
    {"score": 0.86, "source": "tutorial/query-params.md", "section": "Query > Optional",
     "text": "Set the default to None."},
    {"score": 0.80, "source": "tutorial/body.md", "section": "Body", "text": "Use a model."},
]


def test_build_messages_numbers_passages_and_asks_citations():
    msgs = generate.build_messages("Ma question ?", HITS)
    assert msgs[0]["role"] == "system" and "[1]" in msgs[0]["content"]
    assert generate.REFUSAL in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "[1] (tutorial/query-params.md > Query > Optional)" in user
    assert "[2] (tutorial/body.md > Body)" in user
    assert user.rstrip().endswith("Question : Ma question ?")


def test_below_min_score_refuses_without_calling_llm(monkeypatch):
    monkeypatch.setattr(generate, "retrieve", lambda *a, **k: HITS)

    def boom(*a, **k):
        raise AssertionError("le LLM ne doit pas etre appele")

    monkeypatch.setattr(generate, "ask_llm", boom)
    r = generate.answer("hors sujet", min_score=0.9)
    assert r["refused"] and r["answer"] == generate.REFUSAL


def test_answer_uses_llm_when_score_ok(monkeypatch):
    monkeypatch.setattr(generate, "retrieve", lambda *a, **k: HITS)
    monkeypatch.setattr(generate, "ask_llm", lambda m, model=None: "Réponse [1].")
    r = generate.answer("q", min_score=0.5)
    assert not r["refused"] and r["answer"] == "Réponse [1]." and len(r["sources"]) == 2
