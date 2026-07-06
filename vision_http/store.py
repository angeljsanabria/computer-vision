"""Store thread-safe del ultimo VisionSnapshot (productor: pipeline, consumidor: HTTP)."""
from __future__ import annotations

import threading

from .types import VisionSnapshot


class VisionStatusStore:
    """Memoria compartida protegida por Lock entre el bucle CV y el hilo API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = VisionSnapshot.no_deteccion_face()

    def publish(self, snapshot: VisionSnapshot) -> None:
        with self._lock:
            self._snapshot = snapshot

    def get_snapshot(self) -> VisionSnapshot:
        with self._lock:
            return self._snapshot


vision_store = VisionStatusStore()
