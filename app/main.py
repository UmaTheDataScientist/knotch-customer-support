from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import conversations
from app.models.schemas import ErrorResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Knotch Customer Support Agent",
    description="Plan -> Act -> Observe -> Verify agent with a Compliance Agent guardrail.",
    version="0.1.0",
)

# Permissive CORS for local development only (e.g. the devtools/ chat UI,
# which is a personal debugging aid and not part of the graded submission).
# In a real deployment this would be scoped to a specific origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(conversations.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content=ErrorResponse(error="bad_request", detail=str(exc)).model_dump())


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logging.exception("Unhandled exception")
    return JSONResponse(
        status_code=500, content=ErrorResponse(error="internal_error", detail=str(exc)).model_dump()
    )
