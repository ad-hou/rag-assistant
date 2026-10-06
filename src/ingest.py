"""Ingestion : lecture des .md, nettoyage, decoupage, embeddings, index Chroma.

Usage : python -m src.ingest [--max-chars 1000] [--overlap 150] [--model ...] [--collection ...]
"""
import argparse
import re
import statistics
import time
from pathlib import Path

from src.config import (CHROMA_DIR, COLLECTION, DOCS_DIR, EMBED_MODEL,
                        MAX_CHARS, OVERLAP, passage_prefix)

FENCE = re.compile(r"^\s*```")
INCLUDE = re.compile(r"^\s*\{\*.*\*\}\s*$")      # {* ../../docs_src/x.py *} : code non present
ADMONITION = re.compile(r"^\s*///.*$")           # /// note | /// tip | ///
IMAGE = re.compile(r"^\s*!\[.*?\]\(.*?\)\s*(\{.*\})?\s*$")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}\s*$")     # titre { #ancre }
FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.S)
MIN_CHARS = 40


def clean_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = FRONT_MATTER.sub("", text, count=1)
    out, in_fence = [], False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and (INCLUDE.match(line) or ADMONITION.match(line)
                               or IMAGE.match(line)):
            continue
        out.append(line)
    return "\n".join(out)


def parse_sections(text: str):
    """Texte nettoye -> liste de (chemin_de_titres, [blocs]).
    Un bloc = un paragraphe ou un bloc de code complet (jamais coupe)."""
    sections, stack, blocks, para, fence = [], [], [], [], None
    path = ""

    def end_para():
        if para:
            blocks.append("\n".join(para).strip())
            para.clear()

    def end_section():
        end_para()
        if blocks:
            sections.append((path, list(blocks)))
            blocks.clear()

    for line in text.split("\n"):
        if fence is not None:
            fence.append(line)
            if FENCE.match(line):
                blocks.append("\n".join(fence))
                fence = None
            continue
        if FENCE.match(line):
            end_para()
            fence = [line]
            continue
        m = HEADING.match(line)
        if m:
            end_section()
            level, title = len(m.group(1)), ANCHOR.sub("", m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            path = " > ".join(t for _, t in stack)
            continue
        if not line.strip():
            end_para()
            continue
        para.append(line)
    if fence:
        blocks.append("\n".join(fence))
    end_section()
    return sections


def _split_oversized(text: str, max_chars: int):
    if len(text) <= max_chars:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        while len(line) > max_chars:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:max_chars])
            line = line[max_chars:]
        if cur and len(cur) + 1 + len(line) > max_chars:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts


def chunk_blocks(blocks, max_chars: int = MAX_CHARS, overlap: int = OVERLAP):
    """Regroupe des blocs en chunks <= max_chars, avec chevauchement de blocs entiers."""
    pieces = [p for b in blocks for p in _split_oversized(b, max_chars)]
    chunks, cur = [], []
    for p in pieces:
        size = sum(len(x) + 2 for x in cur)
        if cur and size + len(p) > max_chars:
            chunks.append("\n\n".join(cur))
            carry, total = [], 0
            for x in reversed(cur):
                if total + len(x) > overlap:
                    break
                carry.insert(0, x)
                total += len(x) + 2
            cur = carry
            if sum(len(x) + 2 for x in cur) + len(p) > max_chars:
                cur = []
        cur.append(p)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def chunk_document(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP):
    """-> (titre, [(section, texte_du_chunk)])"""
    sections = parse_sections(clean_markdown(text))
    title = sections[0][0].split(" > ")[0] if sections else ""
    out = []
    for path, blocks in sections:
        for c in chunk_blocks(blocks, max_chars, overlap):
            if len(c) >= MIN_CHARS:
                out.append((path or title, c))
    return title, out


def load_chunks(docs_dir: Path = DOCS_DIR, max_chars: int = MAX_CHARS,
                overlap: int = OVERLAP):
    chunks = []
    for f in sorted(Path(docs_dir).rglob("*.md")):
        source = f.relative_to(docs_dir).as_posix()
        title, items = chunk_document(f.read_text(encoding="utf-8"), max_chars, overlap)
        for i, (section, text) in enumerate(items):
            chunks.append({"id": f"{source}::{i}", "source": source,
                           "title": title or f.stem, "section": section, "text": text})
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--overlap", type=int, default=OVERLAP)
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--collection", default=COLLECTION)
    a = ap.parse_args()

    chunks = load_chunks(DOCS_DIR, a.max_chars, a.overlap)
    sizes = [len(c["text"]) for c in chunks]
    n_docs = len({c["source"] for c in chunks})
    print(f"{n_docs} documents -> {len(chunks)} passages "
          f"(taille moyenne {statistics.mean(sizes):.0f}, mediane {statistics.median(sizes):.0f}, "
          f"max {max(sizes)} caracteres)")

    from sentence_transformers import SentenceTransformer
    import chromadb

    model = SentenceTransformer(a.model)
    pre = passage_prefix(a.model)
    texts = [f"{pre}{c['section']}\n{c['text']}" for c in chunks]
    t0 = time.time()
    emb = model.encode(texts, batch_size=32, normalize_embeddings=True,
                       show_progress_bar=True)
    print(f"embeddings : {time.time() - t0:.1f} s, dimension {emb.shape[1]}")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(a.collection)
    except Exception:
        pass
    col = client.create_collection(a.collection,
                                   configuration={"hnsw": {"space": "cosine"}})
    for i in range(0, len(chunks), 256):
        part = chunks[i:i + 256]
        col.add(ids=[c["id"] for c in part],
                embeddings=emb[i:i + 256].tolist(),
                documents=[c["text"] for c in part],
                metadatas=[{"source": c["source"], "title": c["title"],
                            "section": c["section"]} for c in part])
    print(f"index '{a.collection}' : {col.count()} passages dans {CHROMA_DIR}")


if __name__ == "__main__":
    main()
