from src.evaluate import (best_threshold, first_rank, load_questions,
                          retrieval_metrics, valid_citation)


def H(*sources):
    return [{"source": s, "score": 0.8} for s in sources]


def test_first_rank():
    assert first_rank(H("a", "b", "c"), {"c"}) == 3
    assert first_rank(H("a", "b"), {"z"}) == 0


def test_retrieval_metrics():
    qs = [{"id": "1", "type": "in_corpus", "gold": ["a"]},
          {"id": "2", "type": "in_corpus", "gold": ["b"]},
          {"id": "3", "type": "off_topic", "gold": []}]
    hits = {"1": H("a", "x"), "2": H("x", "y", "z", "w", "b"), "3": H("x")}
    m = retrieval_metrics(qs, hits)
    assert m["n"] == 2
    assert m["recall@1"] == 0.5 and m["recall@3"] == 0.5 and m["recall@5"] == 1.0
    assert abs(m["mrr"] - (1 + 1 / 5) / 2) < 1e-9


def test_best_threshold_separates_when_possible():
    t = best_threshold([0.82, 0.85, 0.9], [0.70, 0.75])
    assert 0.75 < t <= 0.82


def test_valid_citation():
    assert valid_citation("Voir [1] et [3].", 4)
    assert not valid_citation("Voir [5].", 4)
    assert not valid_citation("Aucune citation.", 4)


def test_question_file_is_consistent():
    qs = load_questions()
    ids = [q["id"] for q in qs]
    assert len(ids) == len(set(ids))
    assert sum(q["type"] == "in_corpus" for q in qs) >= 30
    assert all(q["gold"] for q in qs if q["type"] == "in_corpus")
    assert all(not q["gold"] for q in qs if q["type"] != "in_corpus")
