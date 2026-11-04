"""FastAPI backend: JSON endpoint + Server-Sent Events stream of the agent's steps.

    uvicorn server.api:app --reload        # then open http://127.0.0.1:8000/
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from creatorguide.config import load_settings
from creatorguide.schema import ValidationError
from creatorguide.service import Service, build_service

log = logging.getLogger("creatorguide.api")
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="CreatorGuide Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@lru_cache(maxsize=1)
def service() -> Service:
    return build_service(load_settings())


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=600)
    mode: Literal["agent", "rag"] = "agent"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    s = service()
    return {"status": "ok", "model": s.settings.llm_model, "retrieval": s.retrieval_used,
            "documents": len(s.corpus.docs), "chunks": len(s.retriever.chunks)}


@app.post("/api/ask")
def ask(body: AskBody) -> dict:
    try:
        return service().ask(body.question, body.mode)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:  # noqa: BLE001 - details go to the log, not to the client
        log.exception("ask failed")
        raise HTTPException(status_code=502, detail="The model backend failed. Check the server log.") from None


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


def _events(question: str, mode: str) -> Iterator[str]:
    try:
        for event in service().stream(question, mode):
            yield _sse(event)
    except ValidationError as exc:
        yield _sse({"type": "error", "message": str(exc)})
    except Exception:  # noqa: BLE001
        log.exception("stream failed")
        yield _sse({"type": "error", "message": "The model backend failed. Check the server log."})
    yield _sse({"type": "done"})


@app.get("/api/ask/stream")
def ask_stream(q: str = Query(min_length=1, max_length=600), mode: Literal["agent", "rag"] = "agent") -> StreamingResponse:
    return StreamingResponse(_events(q, mode), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
