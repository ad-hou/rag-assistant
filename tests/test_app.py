from pathlib import Path
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"
RESULT = {"answer": "Utilisez `None` [1].", "refused": False, "retrieval_s": 0.4,
          "generation_s": 3.0,
          "sources": [{"n": 1, "source": "tutorial/query-params.md", "section": "Query > Optional",
                       "score": 0.865, "text": "Set the default to None."}]}


def fake_get(*a, **k):
    m = MagicMock()
    m.json.return_value = {"ollama": True}
    return m


def fake_post(*a, **k):
    m = MagicMock()
    m.json.return_value = RESULT
    m.raise_for_status.return_value = None
    return m


def test_question_shows_answer_and_sources():
    with patch("requests.get", fake_get), patch("requests.post", fake_post):
        at = AppTest.from_file(str(APP), default_timeout=20).run()
        assert not at.exception
        at.chat_input[0].set_value("Comment faire ?").run()
        assert not at.exception
        assert any("Utilisez" in m.value for m in at.markdown)
        assert any("tutorial/query-params.md" in m.value for m in at.markdown)
