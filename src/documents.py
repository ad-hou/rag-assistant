"""Documents fournis par l'utilisateur : extraction (PDF, Markdown, texte), decoupage,
index temporaire (une collection Chroma `user_xxxxxxxx` par envoi)."""
import io
import re
import uuid
from pathlib import Path

from src.config import CHROMA_DIR, EMBED_MODEL, MAX_CHARS, OVERLAP
from src.ingest import MIN_CHARS, chunk_blocks, chunk_document, index_chunks

PREFIX = "user_"
ALLOWED = (".pdf", ".md", ".markdown", ".txt")
MAX_FILES = 10
MAX_BYTES = 15 * 1024 * 1024
MAX_PASSAGES = 3000


class DocumentError(ValueError):
    """Fichier refuse ou illisible (message affichable a l'utilisateur)."""


def _pack_sentences(text: str, max_chars: int):
    out, cur = [], ""
    for p in re.split(r"(?<=[.!?])\s+", text):
        while len(p) > max_chars:  # phrase geante : coupe dure
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:max_chars])
            p = p[max_chars:]
        if cur and len(cur) + 1 + len(p) > max_chars:
            out.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur:
        out.append(cur)
    return out


def paragraphs(text: str, max_chars: int = MAX_CHARS):
    """Paragraphes separes par une ligne vide ; les retours a la ligne internes (PDF)
    sont recolles, puis les paragraphes trop longs sont coupes entre deux phrases."""
    blocks = []
    for raw in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        flat = " ".join(raw.split())
        if flat:
            blocks += _pack_sentences(flat, max_chars)
    return blocks


def _pdf_pages(name: str, data: bytes):
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise DocumentError(f"{name} : PDF protégé par mot de passe")
        return [(f"p. {i}", p.extract_text() or "") for i, p in enumerate(reader.pages, 1)]
    except DocumentError:
        raise
    except Exception as e:
        raise DocumentError(f"{name} : PDF illisible") from e


def _decode(name: str, data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def build_chunks(files, max_chars: int = MAX_CHARS, overlap: int = OVERLAP):
    """files : liste de (nom, octets) -> passages {id, source, title, section, text}."""
    chunks = []
    for fi, (name, data) in enumerate(files):
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED:
            raise DocumentError(f"{name} : format non pris en charge (PDF, Markdown, texte)")
        if len(data) > MAX_BYTES:
            raise DocumentError(f"{name} : fichier trop volumineux (maximum 15 Mo)")
        if ext in (".md", ".markdown"):
            title, items = chunk_document(_decode(name, data), max_chars, overlap)
            for i, (section, text) in enumerate(items):
                chunks.append({"id": f"{fi}:{name}::{i}", "source": name,
                               "title": title or name, "section": section, "text": text})
            continue
        if ext == ".pdf":
            pages = _pdf_pages(name, data)
            if sum(len(t.strip()) for _, t in pages) < 50:
                raise DocumentError(f"{name} : aucun texte extractible "
                                    "(PDF scanné ? l'OCR n'est pas géré)")
        else:
            pages = [("", _decode(name, data))]
        i = 0
        for label, text in pages:
            for c in chunk_blocks(paragraphs(text, max_chars), max_chars, overlap):
                if len(c) >= MIN_CHARS:
                    chunks.append({"id": f"{fi}:{name}::{i}", "source": name, "title": name,
                                   "section": label, "text": c})
                    i += 1
    return chunks


def ingest_files(files, model: str = EMBED_MODEL) -> dict:
    if not files:
        raise DocumentError("aucun fichier")
    if len(files) > MAX_FILES:
        raise DocumentError(f"trop de fichiers (maximum {MAX_FILES})")
    chunks = build_chunks(files)
    if not chunks:
        raise DocumentError("aucun texte exploitable dans ces fichiers")
    if len(chunks) > MAX_PASSAGES:
        raise DocumentError(f"documents trop longs ({len(chunks)} passages, "
                            f"maximum {MAX_PASSAGES})")
    collection = f"{PREFIX}{uuid.uuid4().hex[:8]}"
    n = index_chunks(chunks, model, collection)
    names = []
    for name, _ in files:
        if name not in names:
            names.append(name)
    return {"collection": collection, "documents": names, "passages": n}


def _client():
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def collection_exists(name: str) -> bool:
    try:
        _client().get_collection(name)
        return True
    except Exception:
        return False


def delete_collection(name: str) -> None:
    if not name.startswith(PREFIX):
        raise DocumentError("collection non supprimable")
    try:
        _client().delete_collection(name)
    except Exception:
        pass
    from src.retrieve import _collection
    _collection.cache_clear()


def purge_user_collections() -> int:
    """Supprime les collections `user_*` d'une session precedente."""
    client, n = _client(), 0
    for col in client.list_collections():
        name = col if isinstance(col, str) else col.name
        if name.startswith(PREFIX):
            client.delete_collection(name)
            n += 1
    return n
