"""Rutas FastAPI del estado de vision."""
from __future__ import annotations

from fastapi import FastAPI

from .store import vision_store

app = FastAPI(title="Identificación biométrica facial API", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/v1/vision-status")
def get_vision_status() -> dict:
    return vision_store.get_snapshot().to_api_dict()
