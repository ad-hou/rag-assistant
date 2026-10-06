"""API : POST /ask -> reponse citee + passages ; GET /health.

Lancer : python -m uvicorn api.main:app --port 8000
"""
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.documents import (PREFIX, DocumentError, collection_exists, delete_collection,
                           ingest_files, purge_user_collections)
from src.generate import OLLAMA_URL, PROMPTS, answer
from src.retrieve import retrieve

# Reglage retenu apres comparaison (voir README) : code des exemples + prompt strict
DEFAULT_COLLECTION = "fastapi_code"
DEFAULT_PROMPT = "strict"



@asynccontextmanager
async def lifespan(_app):
    try:
        purge_user_collections()  # les documents d'une session precedente ne sont pas conserves
    except Exception as e:
        print(f"Nettoyage impossible : {e}")
    try:  # charge le modele d'embeddings et l'index au demarrage (sinon ~12 s a la 1re question)
        retrieve("echauffement", 1, collection=DEFAULT_COLLECTION)
    except Exception as e:  # index absent : l'API demarre quand meme
        print(f"Echauffement impossible : {e}")
    yield


app = FastAPI(title="Assistant RAG - documentation FastAPI", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(4, ge=1, le=8)
    prompt: str = DEFAULT_PROMPT
    collection: str = DEFAULT_COLLECTION


class Source(BaseModel):
    n: int
    source: str
    section: str
    score: float
    text: str


class DocumentsResponse(BaseModel):
    collection: str
    documents: list[str]
    passages: int


class AskResponse(BaseModel):
    answer: str
    refused: bool
    sources: list[Source]
    retrieval_s: float
    generation_s: float


@app.get("/health")
def health():
    try:
        requests.get(OLLAMA_URL.replace("/api/chat", "/api/tags"), timeout=3).raise_for_status()
        return {"status": "ok", "ollama": True}
    except requests.RequestException:
        return {"status": "degraded", "ollama": False}


@app.post("/documents", response_model=DocumentsResponse)
def upload_documents(files: list[UploadFile]):
    """Indexe des PDF / Markdown / texte dans une collection temporaire `user_xxxxxxxx`."""
    try:
        return ingest_files([(f.filename or "document", f.file.read()) for f in files])
    except DocumentError as e:
        raise HTTPException(422, str(e)) from e


@app.delete("/documents/{collection}", status_code=204)
def remove_documents(collection: str):
    try:
        delete_collection(collection)
    except DocumentError as e:
        raise HTTPException(422, str(e)) from e


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.prompt not in PROMPTS:
        raise HTTPException(422, f"prompt inconnu : {req.prompt}")
    prompt = req.prompt
    if req.collection.startswith(PREFIX):
        if not collection_exists(req.collection):
            raise HTTPException(404, "documents introuvables : réindexez-les")
        if prompt == "strict":
            prompt = "generic"  # prompt sans mention de FastAPI pour les documents utilisateur
    try:
        r = answer(req.question, k=req.k, collection=req.collection, prompt=prompt)
    except RuntimeError as e:  # Ollama injoignable
        raise HTTPException(503, str(e)) from e
    return AskResponse(
        answer=r["answer"], refused=r["refused"],
        sources=[Source(n=i, **h) for i, h in enumerate(r["sources"], 1)],
        retrieval_s=r["retrieval_s"], generation_s=r["generation_s"])
