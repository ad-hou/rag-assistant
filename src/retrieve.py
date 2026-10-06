"""Recherche : question -> k passages les plus proches, avec score (similarite cosinus).

Usage : python -m src.retrieve "Comment declarer un parametre de requete optionnel ?" -k 5
"""
import argparse
import os
import sys
import time
from functools import lru_cache

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from src.config import CHROMA_DIR, COLLECTION, EMBED_MODEL, query_prefix


@lru_cache(maxsize=2)
def _model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


@lru_cache(maxsize=4)
def _collection(name: str):
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(name)


def retrieve(question: str, k: int = 5, model: str = EMBED_MODEL,
             collection: str = COLLECTION):
    """-> liste de dicts {score, source, section, text}, du plus proche au moins proche."""
    emb = _model(model).encode([query_prefix(model) + question],
                               normalize_embeddings=True)
    res = _collection(collection).query(query_embeddings=emb.tolist(), n_results=k)
    return [{"score": 1.0 - d, "source": m["source"], "section": m["section"], "text": t}
            for d, m, t in zip(res["distances"][0], res["metadatas"][0],
                               res["documents"][0])]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--model", default=EMBED_MODEL)
    ap.add_argument("--collection", default=COLLECTION)
    a = ap.parse_args()

    _model(a.model)  # charge le modele hors chronometre
    t0 = time.perf_counter()
    hits = retrieve(a.question, a.k, a.model, a.collection)
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n{a.question}  ({ms:.0f} ms)\n")
    for i, h in enumerate(hits, 1):
        snippet = " ".join(h["text"].split())[:200]
        print(f"{i}. [{h['score']:.3f}] {h['source']} > {h['section']}\n   {snippet}\n")


if __name__ == "__main__":
    main()
