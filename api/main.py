"""API : POST /ask -> reponse citee + passages ; GET /health.

Lancer : python -m uvicorn api.main:app --port 8000
"""
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generate import OLLAMA_URL, PROMPTS, answer
from src.retrieve import retrieve

# Reglage retenu apres comparaison (voir README) : code des exemples + prompt strict
DEFAULT_COLLECTION = "fastapi_code"
DEFAULT_PROMPT = "strict"



@asynccontextmanager
async def lifespan(_app):
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


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if req.prompt not in PROMPTS:
        raise HTTPException(422, f"prompt inconnu : {req.prompt}")
    try:
        r = answer(req.question, k=req.k, collection=req.collection, prompt=req.prompt)
    except RuntimeError as e:  # Ollama injoignable
        raise HTTPException(503, str(e)) from e
    return AskResponse(
        answer=r["answer"], refused=r["refused"],
        sources=[Source(n=i, **h) for i, h in enumerate(r["sources"], 1)],
        retrieval_s=r["retrieval_s"], generation_s=r["generation_s"])
