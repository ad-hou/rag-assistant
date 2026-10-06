"""Evaluation du RAG sur eval/questions.jsonl.

  python -m src.evaluate --retrieval                    # recherche seule (rapide, sans LLM)
  python -m src.evaluate --generation --tag base        # chaine complete (Ollama requis)
  python -m src.evaluate --faith eval/faithfulness_sample.csv   # score de fidelite note a la main

Reglages comparables : --k, --embed-model, --collection, --llm, --min-score, --tag.
"""
import argparse
import csv
import json
import random
import re
import statistics
import sys
from pathlib import Path

from src.config import COLLECTION, EMBED_MODEL, ROOT
from src.retrieve import retrieve

QUESTIONS = ROOT / "eval" / "questions.jsonl"
OUT = ROOT / "eval" / "results"
KS = (1, 3, 5)
DEEP = 10  # profondeur de recherche pour le MRR


def load_questions(path=QUESTIONS):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def first_rank(hits, gold):
    """Rang (1 = premier) du premier passage dont la page est dans gold ; 0 si absent."""
    for i, h in enumerate(hits, 1):
        if h["source"] in gold:
            return i
    return 0


def retrieval_metrics(questions, hits_by_id):
    ranks = [first_rank(hits_by_id[q["id"]], set(q["gold"]))
             for q in questions if q["type"] == "in_corpus"]
    n = len(ranks)
    out = {f"recall@{k}": sum(1 for r in ranks if 0 < r <= k) / n for k in KS}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    out["n"] = n
    return out


def best_threshold(in_scores, off_scores):
    """Seuil t (refus si score < t) maximisant la moyenne [questions valides acceptees,
    questions hors corpus refusees]."""
    best = (-1.0, None)
    for t in sorted(set(in_scores) | set(off_scores)):
        acc = sum(s >= t for s in in_scores) / len(in_scores)
        ref = sum(s < t for s in off_scores) / len(off_scores)
        if (acc + ref) / 2 > best[0]:
            best = ((acc + ref) / 2, t)
    return best[1]


def threshold_report(questions, hits_by_id):
    """Seuil choisi sur une moitie des questions, evalue sur l'autre (pas de triche)."""
    def top1(qs):
        return [hits_by_id[q["id"]][0]["score"] for q in qs]
    inc = [q for q in questions if q["type"] == "in_corpus"]
    off = [q for q in questions if q["type"] != "in_corpus"]
    t = best_threshold(top1(inc[0::2]), top1(off[0::2]))
    in_t, off_t = top1(inc[1::2]), top1(off[1::2])
    return {"threshold": t,
            "valid_accepted": sum(s >= t for s in in_t) / len(in_t),
            "offcorpus_refused": sum(s < t for s in off_t) / len(off_t),
            "min_top1_in_corpus": min(top1(inc)), "max_top1_off": max(top1(off))}


def run_retrieval(args, questions):
    hits = {q["id"]: retrieve(q["question"], DEEP, args.embed_model, args.collection)
            for q in questions}
    m = retrieval_metrics(questions, hits)
    t = threshold_report(questions, hits)
    print(f"\nRecherche ({m['n']} questions du corpus, modele {args.embed_model}, "
          f"collection {args.collection})")
    for k in KS:
        print(f"  Recall@{k} : {m[f'recall@{k}']:.1%}")
    print(f"  MRR (profondeur {DEEP}) : {m['mrr']:.3f}")
    print("\nSeuil de refus sur le score du 1er passage (choisi sur une moitie, mesure sur l'autre)")
    print(f"  seuil {t['threshold']:.3f} : questions valides acceptees {t['valid_accepted']:.1%}, "
          f"hors corpus refusees {t['offcorpus_refused']:.1%}")
    print(f"  plus bas score valide {t['min_top1_in_corpus']:.3f} / "
          f"plus haut score hors corpus {t['max_top1_off']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"retrieval_{args.tag}.json").write_text(
        json.dumps({"metrics": m, "threshold": t}, indent=2), encoding="utf-8")


def valid_citation(answer, k):
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    return bool(nums) and all(1 <= n <= k for n in nums)


def pct(values, p):
    s = sorted(values)
    return s[int(p * (len(s) - 1))]


def run_generation(args, questions):
    from src.generate import REFUSAL, answer, ask_llm
    retrieve("echauffement", 1, args.embed_model, args.collection)
    ask_llm([{"role": "user", "content": "ok"}], args.llm)  # charge le modele en VRAM
    rows = []
    for i, q in enumerate(questions, 1):
        r = answer(q["question"], args.k, args.min_score, args.llm,
                   args.embed_model, args.collection)
        r.update(id=q["id"], type=q["type"], gold=q["gold"])
        rows.append(r)
        print(f"  {i}/{len(questions)} {q['id']} {r['total_s']:.1f} s "
              f"{'REFUS' if r['refused'] else 'reponse'}", flush=True)

    inc = [r for r in rows if r["type"] == "in_corpus"]
    off = [r for r in rows if r["type"] != "in_corpus"]
    answered = [r for r in inc if not r["refused"]]
    lat = [r["total_s"] for r in rows]
    print(f"\nChaine complete ({len(inc)} valides, {len(off)} hors corpus), "
          f"LLM {args.llm}, k={args.k}, seuil {args.min_score}")
    print(f"  Refus correct (hors corpus) : {sum(r['refused'] for r in off) / len(off):.1%}")
    print(f"  Faux refus (questions valides) : {1 - len(answered) / len(inc):.1%}")
    print(f"  Reponses avec citation valide : "
          f"{sum(valid_citation(r['answer'], args.k) for r in answered) / max(len(answered), 1):.1%}"
          f" ({len(answered)} reponses)")
    print(f"  Recall@{args.k} (page attendue dans les passages fournis) : "
          f"{sum(first_rank(r['sources'], set(r['gold'])) > 0 for r in inc) / len(inc):.1%}")
    print(f"  Latence de bout en bout : mediane {statistics.median(lat):.1f} s, "
          f"p95 {pct(lat, 0.95):.1f} s")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"generation_{args.tag}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    pool = [r for r in answered if r["answer"].strip() != REFUSAL]
    sample = random.Random(0).sample(pool, min(15, len(pool)))
    with open(ROOT / "eval" / f"faithfulness_{args.tag}.csv", "w", encoding="utf-8-sig",
              newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "question", "reponse", "passages", "fidele (1/0)"])
        for r in sample:
            ctx = "\n---\n".join(f"[{i}] {h['text']}" for i, h in enumerate(r["sources"], 1))
            w.writerow([r["id"], r["question"], r["answer"], ctx, ""])
    print(f"\nA noter a la main : eval/faithfulness_{args.tag}.csv "
          f"(colonne 'fidele (1/0)'), puis --faith")


def run_faith(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        vals = [row["fidele (1/0)"].strip() for row in csv.DictReader(f, delimiter=";")]
    vals = [int(v) for v in vals if v in ("0", "1")]
    print(f"Fidelite : {sum(vals)}/{len(vals)} = {sum(vals) / len(vals):.1%}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--generation", action="store_true")
    ap.add_argument("--faith", metavar="CSV")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--llm", default="qwen2.5:7b-instruct")
    ap.add_argument("--limit", type=int, default=None, help="n premieres questions (test rapide)")
    a = ap.parse_args()

    if a.faith:
        return run_faith(a.faith)
    qs = load_questions()
    if a.limit:
        qs = qs[:a.limit]
    if a.retrieval:
        run_retrieval(a, qs)
    if a.generation:
        run_generation(a, qs)


if __name__ == "__main__":
    main()
