"""Interface Streamlit : questions sur un corpus de demonstration ou sur vos documents.

Lancer (2 terminaux) :
  python -m uvicorn api.main:app --port 8000
  python -m streamlit run app.py
"""
import html
import os
import re
from pathlib import PurePosixPath

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
FASTAPI_CORPUS, USER_CORPUS = "Tutoriel FastAPI", "Mes documents"
EXAMPLES = ["Comment déclarer un paramètre de requête optionnel ?",
            "Comment protéger une route avec un token JWT ?",
            "Comment téléverser un fichier ?"]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&family=Source+Serif+4:ital,wght@0,400;1,400&display=swap');
:root { --paper:#F3F5F7; --surface:#FFFFFF; --ink:#16212C; --slate:#5B6B7B;
        --rule:#D9E0E7; --mark:#FFE066; }
html, body, [data-testid="stApp"], button, input, textarea {
  font-family:'Schibsted Grotesk', system-ui, sans-serif !important; color:var(--ink); }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
footer, #MainMenu { display:none !important; }
.block-container { max-width:1080px; padding:1.6rem 1.5rem 7rem; }
[data-testid="stMarkdownContainer"] p { font-size:1.02rem; line-height:1.62; }

.brand { font-weight:700; font-size:1.05rem; letter-spacing:-.01em; padding-top:.35rem; }
.hero h1 { font-size:clamp(2.3rem,5.2vw,3.7rem); line-height:1.02; letter-spacing:-.04em;
           font-weight:700; margin:3.2rem 0 .9rem; max-width:13ch; }
.hero p { color:var(--slate); font-size:1.12rem; line-height:1.5; max-width:44ch; margin-bottom:1.6rem; }

.q { font-size:1.55rem; line-height:1.2; letter-spacing:-.022em; font-weight:600;
     margin:2.6rem 0 1.1rem; max-width:34ch; }
.cite { background:var(--mark); font-weight:700; font-size:.72em; padding:.06em .4em;
        border-radius:3px; margin:0 .12em; vertical-align:.1em; }
.refusal { border:1px dashed var(--slate); border-radius:4px; padding:1rem 1.1rem;
           color:var(--slate); line-height:1.5; }
.refusal b { color:var(--ink); }
.metrics { color:var(--slate); font-size:.8rem; margin-top:.7rem; }

.note { display:flex; gap:.75rem; padding:.85rem 1rem; background:var(--surface);
        border-left:4px solid var(--mark); border-radius:0 4px 4px 0; margin-bottom:.65rem; }
.note.off { border-left-color:var(--rule); opacity:.72; }
.note .n { flex:none; font-weight:700; font-size:.78rem; background:var(--mark);
           height:1.45rem; min-width:1.45rem; display:flex; align-items:center;
           justify-content:center; border-radius:3px; }
.note.off .n { background:var(--rule); }
.note .doc { font-weight:600; font-size:.9rem; word-break:break-word; }
.note .dir { color:var(--slate); font-weight:400; }
.note .sec { color:var(--slate); font-size:.78rem; margin:.1rem 0 .5rem; }
.note blockquote { margin:0; font-family:'Source Serif 4', Georgia, serif; font-style:italic;
                   font-size:.9rem; line-height:1.5; color:#2A3744; }
.meter { height:3px; background:var(--rule); border-radius:2px; margin:.6rem 0 .15rem; }
.meter i { display:block; height:3px; background:var(--ink); border-radius:2px; }
.score { color:var(--slate); font-size:.72rem; }
.legend { color:var(--slate); font-size:.78rem; margin:.2rem 0 .6rem; }

.dossier { display:flex; flex-wrap:wrap; gap:.5rem; margin:.4rem 0 .3rem; }
.dossier span { background:var(--surface); border-bottom:3px solid var(--mark);
                padding:.3rem .65rem; font-size:.88rem; font-weight:500; border-radius:3px 3px 0 0; }
.fine { color:var(--slate); font-size:.82rem; max-width:60ch; line-height:1.45; }

.stButton > button { background:transparent; border:0; border-bottom:1px solid var(--rule);
  border-radius:0; justify-content:flex-start; text-align:left; font-weight:500;
  font-size:1.02rem; padding:.95rem .1rem; color:var(--ink); }
.stButton > button { justify-content:flex-start !important; }
.stButton > button > div, .stButton > button > div > span {
  width:100%; display:flex; justify-content:flex-start !important; }
.stButton > button p { text-align:left !important; margin:0; }
.stButton > button div[data-testid="stMarkdownContainer"] { width:100%; text-align:left !important; }
.stButton > button [data-testid="stMarkdownContainer"] p { text-align:left; font-size:1.02rem; }
.stButton > button[kind="primary"] { justify-content:center !important; }
.stButton > button[kind="primary"] div[data-testid="stMarkdownContainer"] { text-align:center !important; }
[data-testid="stBottomBlockContainer"] { max-width:1080px; padding-left:1.5rem; padding-right:1.5rem; }
.stButton > button:hover { border-bottom-color:var(--ink); background:transparent; color:var(--ink); }
.stButton > button:focus-visible { outline:3px solid var(--mark); outline-offset:2px; }
.stButton > button[kind="primary"] { background:var(--ink); color:#fff; border:0; border-radius:4px;
  justify-content:center; padding:.6rem 1.1rem; font-weight:600; }
.stButton > button[kind="primary"]:hover { background:#24364A; color:#fff; }
[data-testid="stFileUploaderDropzone"] { background:var(--surface); border:1.5px dashed var(--slate);
  border-radius:4px; }
</style>
"""


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def style_citations(text: str) -> str:
    """Remplace [n] par un surlignage, hors blocs de code ; neutralise le HTML du modele."""
    parts = re.split(r"(```.*?```)", text, flags=re.S)
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(part)
        else:
            part = part.replace("<", "&lt;")
            out.append(re.sub(r"\[(\d+)\]", r'<span class="cite">\1</span>', part))
    return "".join(out)


def cited_numbers(text: str) -> set:
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}


def note_html(s: dict, used: bool) -> str:
    path = PurePosixPath(s["source"])
    parent = f'<span class="dir">{esc(str(path.parent))}/</span>' if str(path.parent) != "." else ""
    section = f'<div class="sec">{esc(s["section"])}</div>' if s["section"] else ""
    excerpt = " ".join(s["text"].split())
    excerpt = excerpt[:170] + ("…" if len(excerpt) > 170 else "")
    level = max(0.0, min(1.0, (s["score"] - 0.70) / 0.20))
    return (f'<div class="note{"" if used else " off"}"><div class="n">{s["n"]}</div><div>'
            f'<div class="doc">{parent}{esc(path.name)}</div>{section}'
            f'<blockquote>{esc(excerpt)}</blockquote>'
            f'<div class="meter" title="similarité cosinus"><i style="width:{level:.0%}"></i></div>'
            f'<div class="score">score {s["score"]:.3f}</div></div></div>')


def render_turn(h: dict):
    st.markdown(f'<div class="q">{esc(h["question"])}</div>', unsafe_allow_html=True)
    left, right = st.columns([0.57, 0.43], gap="large")
    used = cited_numbers(h["answer"])
    with left:
        if h["refused"]:
            st.markdown('<div class="refusal"><b>Pas dans les documents.</b><br>'
                        "Aucun passage ne répond à cette question. Reformulez, ou vérifiez "
                        "que le document concerné est indexé.</div>", unsafe_allow_html=True)
        else:
            st.markdown(style_citations(h["answer"]), unsafe_allow_html=True)
        st.markdown(f'<div class="metrics">recherche {h["retrieval_s"] * 1000:.0f} ms · '
                    f'génération {h["generation_s"]:.1f} s</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="legend">'
                    + ("Passages les plus proches, non retenus" if h["refused"] else
                       "Surlignés : cités dans la réponse. Estompés : fournis au modèle, non cités.")
                    + "</div>", unsafe_allow_html=True)
        st.markdown("".join(note_html(s, s["n"] in used and not h["refused"])
                            for s in h["sources"]), unsafe_allow_html=True)
        with st.expander("Texte intégral des passages"):
            for s in h["sources"]:
                st.markdown(f"**[{s['n']}]** {esc(s['source'])} {esc(s['section'])}")
                st.text(s["text"])


def api_status():
    try:
        return requests.get(f"{API_URL}/health", timeout=3).json().get("ollama")
    except requests.RequestException:
        return None


st.set_page_config(page_title="Assistant RAG", layout="wide", initial_sidebar_state="collapsed")
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("history", {FASTAPI_CORPUS: [], USER_CORPUS: []})
st.session_state.setdefault("user_docs", None)

top = st.columns([3, 4, 1.4], vertical_alignment="center")
top[0].markdown('<div class="brand">Assistant RAG</div>', unsafe_allow_html=True)
corpus = top[1].segmented_control("Corpus", [FASTAPI_CORPUS, USER_CORPUS],
                                  default=FASTAPI_CORPUS, label_visibility="collapsed",
                                  key="corpus") or FASTAPI_CORPUS
with top[2].popover("Réglages", width="stretch"):
    k = st.slider("Passages fournis au modèle", 1, 8, 4)
    prompt = st.radio("Prompt", ["strict", "base"], horizontal=True,
                      help="strict : pas de code absent des extraits, citations obligatoires")
    status = api_status()
    st.caption({True: "API et Ollama joignables.", False: "API joignable, Ollama arrêté.",
                None: f"API injoignable ({API_URL})."}[status])

history = st.session_state.history[corpus]
docs = st.session_state.user_docs
ready = corpus == FASTAPI_CORPUS or docs is not None

if corpus == USER_CORPUS:
    if docs is None:
        st.markdown('<div class="hero"><h1>Déposez vos documents.</h1>'
                    "<p>PDF, Markdown ou texte. L'assistant les lit, puis répond en citant "
                    "le document et la page.</p></div>", unsafe_allow_html=True)
        files = st.file_uploader("Fichiers", type=["pdf", "md", "txt"],
                                 accept_multiple_files=True, label_visibility="collapsed")
        if files and st.button(f"Indexer {len(files)} fichier{'s' if len(files) > 1 else ''}",
                               type="primary"):
            try:
                with st.spinner("Lecture et indexation…"):
                    r = requests.post(f"{API_URL}/documents", timeout=600, files=[
                        ("files", (f.name, f.getvalue(), "application/octet-stream"))
                        for f in files])
                if r.status_code == 422:
                    st.error(f"Fichier refusé : {r.json()['detail']}")
                else:
                    r.raise_for_status()
                    st.session_state.user_docs = r.json()
                    st.session_state.history[USER_CORPUS] = []
                    st.rerun()
            except requests.RequestException as e:
                st.error(f"Indexation impossible ({API_URL}) : {e}")
        st.markdown('<p class="fine">Maximum 10 fichiers de 15 Mo. Les PDF scannés (images) ne '
                    "sont pas lisibles : l'OCR n'est pas géré. Les documents restent sur votre "
                    "machine et sont supprimés au redémarrage de l'API.</p>", unsafe_allow_html=True)
    else:
        chips = "".join(f"<span>{esc(n)}</span>" for n in docs["documents"])
        st.markdown(f'<div class="dossier">{chips}</div>'
                    f'<p class="fine">{docs["passages"]} passages indexés. La qualité n\'est pas '
                    "mesurée sur vos documents : les chiffres du README concernent le corpus "
                    "FastAPI.</p>", unsafe_allow_html=True)
        if st.button("Changer de documents"):
            try:
                requests.delete(f"{API_URL}/documents/{docs['collection']}", timeout=10)
            except requests.RequestException:
                pass
            st.session_state.user_docs = None
            st.rerun()
elif not history:
    st.markdown('<div class="hero"><h1>Interrogez la documentation FastAPI.</h1>'
                "<p>51 pages du tutoriel officiel. Chaque réponse cite ses passages sources, "
                "et l'assistant refuse quand la réponse n'y est pas.</p></div>",
                unsafe_allow_html=True)
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.pending = ex
            st.rerun()

for h in history:
    render_turn(h)

question = st.chat_input("Posez une question précise" if ready
                         else "Indexez d'abord vos documents", disabled=not ready)
question = question or st.session_state.pop("pending", None)
if question and ready:
    payload = {"question": question, "k": k, "prompt": prompt}
    if corpus == USER_CORPUS:
        payload["collection"] = docs["collection"]
    try:
        with st.spinner("Recherche et génération…"):
            r = requests.post(f"{API_URL}/ask", json=payload, timeout=300)
        if r.status_code == 404:
            st.session_state.user_docs = None
            st.error("Les documents ne sont plus indexés (l'API a redémarré). Déposez-les à nouveau.")
        elif r.status_code == 503:
            st.error("Ollama ne répond pas. Lancez l'application Ollama, puis reposez la question.")
        else:
            r.raise_for_status()
            history.append({"question": question, **r.json()})
            st.rerun()
    except requests.RequestException as e:
        st.error(f"L'API ne répond pas ({API_URL}) : {e}")
