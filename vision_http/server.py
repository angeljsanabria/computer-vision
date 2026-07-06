"""Arranque y cierre del servidor HTTP en hilo (mismo proceso que main_mov)."""
from __future__ import annotations

import logging
import threading

import uvicorn

from configs import settings as s

from .routes import app

_server: uvicorn.Server | None = None
_thread: threading.Thread | None = None


def _run_server() -> None:
    """Target del hilo: uvicorn con log de fallos (p. ej. puerto ocupado)."""
    try:
        if _server is not None:
            _server.run()
    except OSError as exc:
        logging.error(
            "API vision: no se pudo bind %s:%d — %s",
            s.HTTP_API_HOST,
            s.HTTP_API_PORT,
            exc,
        )
    except Exception as exc:
        logging.error("API vision: fallo uvicorn: %s", exc, exc_info=True)
    finally:
        logging.debug("API vision: hilo uvicorn finalizado.")


def start_api_thread() -> threading.Thread | None:
    """Inicia uvicorn en hilo si ENABLE_ENDPOINT=true."""
    global _server, _thread
    if not s.ENABLE_ENDPOINT:
        return None
    if _thread is not None and _thread.is_alive():
        return _thread

    config = uvicorn.Config(
        app,
        host=s.HTTP_API_HOST,
        port=s.HTTP_API_PORT,
        log_level=logging.WARNING,
        access_log=False,
        log_config=None,
    )
    _server = uvicorn.Server(config)
    _thread = threading.Thread(
        target=_run_server,
        name="H_VisionAPI",
        daemon=False,
    )
    _thread.start()
    logging.debug(
        "API vision: hilo iniciado en %s:%d",
        s.HTTP_API_HOST,
        s.HTTP_API_PORT,
    )
    return _thread


def stop_api_thread(timeout_s: float = 5.0) -> None:
    """Cierra uvicorn y libera el puerto (llamar desde finally de main_mov)."""
    global _server, _thread
    if _server is None or _thread is None:
        return

    logging.debug("API vision: solicitando cierre uvicorn...")
    _server.should_exit = True
    _thread.join(timeout=timeout_s)
    if _thread.is_alive():
        logging.warning(
            "API vision: hilo uvicorn no termino en %.1fs", timeout_s
        )

    _server = None
    _thread = None
    logging.debug("API vision: servidor cerrado.")

