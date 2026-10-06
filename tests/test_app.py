from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"
RESULT = {"answer": "Utilisez `None` comme valeur par défaut [1].", "refused": False,
          "retrieval_s": 0.4, "generation_s": 3.0,
          "sources": [{"n": 1, "source": "tutorial/query-params.md", "section": "Query > Optional",
                       "score": 0.865, "text": "Set the default to None."},
                      {"n": 2, "source": "tutorial/body.md", "section": "Body",
                       "score": 0.80, "text": "Declare the body."}]}


def fake_get(*a, **k):
    m = MagicMock()
    m.json.return_value = {"ollama": True}
    return m


def fake_post(result):
    def post(*a, **k):
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = result
        return m
    return post


def test_question_shows_answer_and_numbered_sources():
    with patch("requests.get", fake_get), patch("requests.post", fake_post(RESULT)):
        at = AppTest.from_file(str(APP), default_timeout=20).run()
        assert not at.exception
        at.chat_input[0].set_value("Comment faire ?").run()
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        assert "Comment faire ?" in html
        assert '<span class="cite">1</span>' in html          # citation surlignée
        assert 'class="note"' in html and 'class="note off"' in html  # cité / non cité
        assert "query-params.md" in html


def test_refusal_is_shown_as_such():
    refused = {**RESULT, "answer": "Je ne trouve pas la réponse dans la documentation fournie.",
               "refused": True}
    with patch("requests.get", fake_get), patch("requests.post", fake_post(refused)):
        at = AppTest.from_file(str(APP), default_timeout=20).run()
        at.chat_input[0].set_value("Recette de tarte ?").run()
        html = " ".join(m.value for m in at.markdown)
        assert "Pas dans les documents" in html and "non retenus" in html


def test_user_documents_tab_asks_for_files_first():
    with patch("requests.get", fake_get):
        at = AppTest.from_file(str(APP), default_timeout=20).run()
        at.segmented_control[0].set_value("Mes documents").run()
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        assert "Déposez vos documents" in html
        assert at.chat_input[0].disabled
