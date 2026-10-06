"""Interface Streamlit : pose une question a l'API et affiche reponse + sources.

Lancer (2 terminaux) :
  python -m uvicorn api.main:app --port 8000
  python -m streamlit run app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
EXAMPLES = ["Comment déclarer un paramètre de requête optionnel ?",
            "Comment protéger une route avec un token JWT ?",
            "Comment téléverser un fichier ?"]

st.set_page_config(page_title="Assistant RAG · FastAPI", layout="centered")
st.title("Assistant documentation FastAPI")
st.caption("Répond uniquement à partir du tutoriel officiel FastAPI (51 pages), "
           "cite ses sources et refuse quand la réponse n'y est pas.")

with st.sidebar:
    st.header("Réglages")
    k = st.slider("Passages fournis au modèle (k)", 1, 8, 4)
    prompt = st.radio("Prompt", ["strict", "base"],
                      help="strict : interdit le code absent des extraits, citations obligatoires")
    try:
        ok = requests.get(f"{API_URL}/health", timeout=3).json().get("ollama")
    except requests.RequestException:
        ok = None
    st.write({True: "API et Ollama : OK", False: "API OK, Ollama injoignable",
              None: "API injoignable"}[ok])


def ask(question: str) -> dict:
    r = requests.post(f"{API_URL}/ask", timeout=300,
                      json={"question": question, "k": k, "prompt": prompt})
    r.raise_for_status()
    return r.json()


def show(msg: dict):
    if msg["refused"]:
        st.warning(msg["answer"])
    else:
        st.markdown(msg["answer"])
    with st.expander(f"Sources ({len(msg['sources'])})"):
        for s in msg["sources"]:
            st.markdown(f"**[{s['n']}]** `{s['source']}` › {s['section']} — score {s['score']:.3f}")
            st.text(s["text"][:700])
    st.caption(f"recherche {msg['retrieval_s'] * 1000:.0f} ms · "
               f"génération {msg['generation_s']:.1f} s")


if "history" not in st.session_state:
    st.session_state.history = []

for h in st.session_state.history:
    with st.chat_message("user"):
        st.write(h["question"])
    with st.chat_message("assistant"):
        show(h)

if not st.session_state.history:
    st.write("Essayez :")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.pending = ex
            st.rerun()

question = st.chat_input("Posez une question sur FastAPI") or st.session_state.pop("pending", None)
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Recherche et génération…"):
                result = ask(question)
        except requests.RequestException as e:
            st.error(f"Erreur d'appel à l'API ({API_URL}) : {e}")
        else:
            show(result)
            st.session_state.history.append({"question": question, **result})
