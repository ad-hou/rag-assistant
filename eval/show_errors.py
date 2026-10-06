"""Diagnostic des erreurs d'une evaluation : python eval/show_errors.py base"""
import json
import re
import sys
from pathlib import Path

tag = sys.argv[1] if len(sys.argv) > 1 else "base"
rows = json.loads((Path(__file__).parent / "results" / f"generation_{tag}.json")
                  .read_text(encoding="utf-8"))
sys.stdout.reconfigure(encoding="utf-8")
inc = [r for r in rows if r["type"] == "in_corpus"]


def rank(r):
    for i, h in enumerate(r["sources"], 1):
        if h["source"] in r["gold"]:
            return i
    return 0


print("FAUX REFUS")
for r in inc:
    if r["refused"]:
        print(f"  {r['id']} {r['question']}\n    page attendue dans les passages : "
              f"{'rang ' + str(rank(r)) if rank(r) else 'NON'}"
              f"\n    1er passage : {r['sources'][0]['source']}")

print("\nREPONSES SANS CITATION VALIDE")
k = len(inc[0]["sources"])
for r in inc:
    if r["refused"]:
        continue
    nums = [int(x) for x in re.findall(r"\[(\d+)\]", r["answer"])]
    if not nums or any(n < 1 or n > k for n in nums):
        print(f"  {r['id']} {r['question']}\n    {' '.join(r['answer'].split())[:160]}")

print("\nPAGE ATTENDUE ABSENTE DES PASSAGES (reponse donnee quand meme)")
for r in inc:
    if not r["refused"] and not rank(r):
        print(f"  {r['id']} {r['question']} -> 1er passage : {r['sources'][0]['source']}")
