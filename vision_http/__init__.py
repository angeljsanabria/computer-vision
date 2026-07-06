"""API HTTP de estado de vision (FastAPI en hilo separado del pipeline)."""
from .derive import derive_vision_status
from .store import VisionStatusStore, vision_store
from .types import VisionPublicStatus, VisionSnapshot, now_iso


def start_api_thread():
    """Import lazy: requiere fastapi/uvicorn solo si se arranca la API."""
    from .server import start_api_thread as _start_api_thread

    return _start_api_thread()


def stop_api_thread(timeout_s: float = 5.0):
    """Import lazy: cierra uvicorn y libera el puerto."""
    from .server import stop_api_thread as _stop_api_thread

    return _stop_api_thread(timeout_s=timeout_s)


__all__ = [
    "VisionPublicStatus",
    "VisionSnapshot",
    "VisionStatusStore",
    "derive_vision_status",
    "now_iso",
    "start_api_thread",
    "stop_api_thread",
    "vision_store",
]
