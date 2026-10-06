"""Generation : question -> passages -> reponse en francais avec citations [n] (Ollama local).

Usage : python -m src.generate "Comment declarer un parametre de requete optionnel ?" [-k 4]
"""
import argparse
import sys
import time

import requests

from src.config import COLLECTION, EMBED_MODEL
from src.retrieve import retrieve

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen2.5:7b-instruct"
REFUSAL = "Je ne trouve pas la réponse dans la documentation fournie."

SYSTEM = (
    "Tu es un assistant qui répond aux questions sur la documentation FastAPI.\n"
    "Règles :\n"
    "- Réponds toujours en français, de façon concise.\n"
    "- Utilise uniquement les extraits fournis, jamais tes propres connaissances.\n"
    "- Cite les extraits utilisés avec leur numéro entre crochets, par exemple [1] ou [2].\n"
    f"- Si les extraits ne contiennent pas la réponse, réponds exactement : « {REFUSAL} »"
)


def build_messages(question: str, hits: list):
    ctx = "\n\n".join(f"[{i}] ({h['source']} > {h['section']})\n{h['text']}"
                      for i, h in enumerate(hits, 1))
    user = f"Extraits :\n{ctx}\n\nQuestion : {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def ask_llm(messages: list, model: str = LLM_MODEL) -> str:
    try:
        r = requests.post(OLLAMA_URL, timeout=300, json={
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096}})
        r.raise_for_status()
    except requests.ConnectionError as e:
        raise RuntimeError("Ollama ne répond pas sur localhost:11434 "
                           "(l'application Ollama est-elle lancée ?)") from e
    return r.json()["message"]["content"].strip()


def answer(question: str, k: int = 4, min_score: float | None = None,
           llm: str = LLM_MODEL, embed_model: str = EMBED_MODEL,
           collection: str = COLLECTION) -> dict:
    t0 = time.perf_counter()
    hits = retrieve(question, k, embed_model, collection)
    t1 = time.perf_counter()
    if min_score is not None and hits[0]["score"] < min_score:
        text = REFUSAL  # refus sans appeler le modele
    else:
        text = ask_llm(build_messages(question, hits), llm)
    t2 = time.perf_counter()
    return {"question": question, "answer": text, "refused": REFUSAL in text,
            "sources": hits, "retrieval_s": t1 - t0, "generation_s": t2 - t1,
            "total_s": t2 - t0}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--llm", default=LLM_MODEL)
    a = ap.parse_args()

    r = answer(a.question, a.k, a.min_score, a.llm)
    print(f"\n{r['answer']}\n\nSources :")
    for i, h in enumerate(r["sources"], 1):
        print(f"  [{i}] {h['source']} > {h['section']}  (score {h['score']:.3f})")
    print(f"\nrecherche {r['retrieval_s'] * 1000:.0f} ms, "
          f"génération {r['generation_s']:.1f} s")


if __name__ == "__main__":
    main()
