from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from pydantic import BaseModel
import re

import chatbot as rag

app = FastAPI(title="YouTube RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_video_id(url_or_id: str) -> str:
    """Pulls the 11-character video ID out of a full YouTube URL, or returns it as-is if already an ID."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'youtu\.be\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    if len(url_or_id) == 11:
        return url_or_id
    raise ValueError("Could not extract a valid video ID from the input")


class IngestRequest(BaseModel):
    video_url: str


class AskRequest(BaseModel):
    video_id: str
    question: str


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        video_id = extract_video_id(req.video_url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    result = rag.ingest_video(video_id)

    if result["status"] == "error":
        raise HTTPException(status_code=422, detail=result["message"])

    return result


@app.post("/ask")
async def ask_question(req: AskRequest):
    if req.video_id not in rag.retrievers_by_video:
        raise HTTPException(status_code=400, detail="Video not ingested yet — call /ingest first")

    answer = await run_in_threadpool(rag.ask, req.video_id, req.question)
    return {"answer": answer}