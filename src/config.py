"""Reglages partages par l'ingestion et la recherche."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"
CHROMA_DIR = ROOT / "chroma"

EMBED_MODEL = "intfloat/multilingual-e5-small"
COLLECTION = "fastapi_docs"
MAX_CHARS = 1000
OVERLAP = 150


def passage_prefix(model: str) -> str:
    """Les modeles E5 exigent un prefixe different pour passages et requetes."""
    return "passage: " if "e5" in model.lower() else ""


def query_prefix(model: str) -> str:
    return "query: " if "e5" in model.lower() else ""
