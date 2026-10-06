"""Copie dans data/docs_src les fichiers de code references par les pages du corpus.

Prerequis (une fois) :
  git -C data\raw\fastapi sparse-checkout set docs/en/docs docs_src
Usage : python -m src.fetch_code
"""
import re
import shutil

from src.config import CODE_DIR, DOCS_DIR, ROOT

SRC = ROOT / "data" / "raw" / "fastapi" / "docs_src"
REF = re.compile(r"\{\*\s*\S*docs_src/(\S+\.py)")


def referenced_files(docs_dir=DOCS_DIR):
    return sorted({m for f in docs_dir.rglob("*.md")
                   for m in REF.findall(f.read_text(encoding="utf-8"))})


def main():
    needed = referenced_files()
    missing = []
    for rel in needed:
        src = SRC / rel
        if not src.exists():
            missing.append(rel)
            continue
        dst = CODE_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print(f"{len(needed)} fichiers de code references, "
          f"{len(needed) - len(missing)} copies dans {CODE_DIR}, {len(missing)} manquants")
    for m in missing[:10]:
        print("  manquant :", m)


if __name__ == "__main__":
    main()
