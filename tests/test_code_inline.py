from src.ingest import chunk_document, clean_markdown
from src.evaluate import unsupported_code

MD = "# Titre\n\nTexte d'introduction assez long pour etre garde.\n\n{* ../../docs_src/demo/ex1.py hl[2] *}\n\nSuite du texte explicatif.\n"


def test_include_replaced_by_code_when_available(tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "ex1.py").write_text("x = 1\nprint(x)\n", encoding="utf-8")
    out = clean_markdown(MD, tmp_path)
    assert "```python\nx = 1\nprint(x)\n```" in out and "docs_src" not in out


def test_include_dropped_without_code_dir_or_file(tmp_path):
    assert "docs_src" not in clean_markdown(MD)
    assert "docs_src" not in clean_markdown(MD, tmp_path)


def test_code_ends_up_in_chunks(tmp_path):
    (tmp_path / "demo").mkdir()
    (tmp_path / "demo" / "ex1.py").write_text("value = compute_total(items)\n", encoding="utf-8")
    _, items = chunk_document(MD, code_dir=tmp_path)
    assert any("value = compute_total(items)" in t for _, t in items)


def test_unsupported_code_detection():
    hits = [{"text": "```python\nx = 1\nprint(x)\n```"}]
    assert not unsupported_code("Voici :\n```python\nx = 1\n```", hits)
    assert unsupported_code("Voici :\n```python\ny = secret_call()\n```", hits)
    assert not unsupported_code("Pas de code ici.", hits)
