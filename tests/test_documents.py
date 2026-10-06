import pytest

from src import documents
from src.documents import DocumentError, build_chunks, paragraphs


def make_pdf(pages):
    """PDF minimal (une ligne de texte par page), sans dependance."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = ["<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>",
            "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, t in enumerate(pages):
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Contents {5 + 2 * i} 0 R /Resources << /Font << /F1 3 0 R >> >> >>")
        stream = f"BT /F1 12 Tf 72 720 Td ({t}) Tj ET"
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF").encode()
    return out


def test_pdf_pages_become_cited_passages():
    pdf = make_pdf(["Le contrat prend effet le premier janvier et dure douze mois.",
                    "La resiliation exige un preavis de trois mois par ecrit."])
    chunks = build_chunks([("contrat.pdf", pdf)])
    assert [c["section"] for c in chunks] == ["p. 1", "p. 2"]
    assert all(c["source"] == "contrat.pdf" for c in chunks)
    assert "preavis de trois mois" in chunks[1]["text"]


def test_txt_and_markdown():
    txt = ("Premier paragraphe assez long pour etre conserve dans l'index.\n\n"
           "Second paragraphe, lui aussi assez long pour etre conserve ici.").encode()
    md = b"# Titre\n\nUn texte d'introduction suffisamment long pour etre garde.\n"
    c1 = build_chunks([("notes.txt", txt)])
    c2 = build_chunks([("doc.md", md)])
    assert c1 and c1[0]["source"] == "notes.txt"
    assert c2[0]["section"] == "Titre" and c2[0]["title"] == "Titre"


def test_unique_ids_for_same_filename():
    t = b"Un paragraphe suffisamment long pour etre indexe correctement."
    ids = [c["id"] for c in build_chunks([("a.txt", t), ("a.txt", t)])]
    assert len(ids) == len(set(ids)) == 2


def test_refusals():
    with pytest.raises(DocumentError, match="format"):
        build_chunks([("image.png", b"x")])
    with pytest.raises(DocumentError, match="illisible"):
        build_chunks([("faux.pdf", b"pas un pdf")])
    with pytest.raises(DocumentError, match="aucun texte"):
        build_chunks([("vide.pdf", make_pdf([""]))])
    with pytest.raises(DocumentError, match="volumineux"):
        build_chunks([("gros.txt", b"a" * (documents.MAX_BYTES + 1))])


def test_long_paragraph_is_cut_between_sentences():
    text = " ".join(f"Phrase numero {i} qui parle d'un sujet." for i in range(80))
    blocks = paragraphs(text, 300)
    assert len(blocks) > 5 and all(len(b) <= 300 for b in blocks)
    assert all(b.endswith(".") for b in blocks)


def test_ingest_validates_before_indexing(monkeypatch):
    monkeypatch.setattr(documents, "index_chunks", lambda *a, **k: 7)
    r = documents.ingest_files([("a.txt", b"Un paragraphe suffisamment long pour etre indexe.")])
    assert r["collection"].startswith("user_") and r["passages"] == 7
    assert r["documents"] == ["a.txt"]
    with pytest.raises(DocumentError):
        documents.ingest_files([])
    with pytest.raises(DocumentError, match="trop de fichiers"):
        documents.ingest_files([("a.txt", b"x")] * 11)


def test_only_user_collections_can_be_deleted():
    with pytest.raises(DocumentError):
        documents.delete_collection("fastapi_code")
