"""Arranque y cierre del servidor HTTP en hilo (mismo proceso que main_mov)."""
from __future__ import annotations

import logging
import threading

import uvicorn

from .routes import app

_server: uvicorn.Server | None = None
_thread: threading.Thread | None = None
_api_host: str = "0.0.0.0"
_api_port: int = 8008


def _run_server() -> None:
    """Target del hilo: uvicorn con log de fallos (p. ej. puerto ocupado)."""
    try:
        if _server is not None:
            _server.run()
    except OSError as exc:
        logging.error(
            "API vision: no se pudo bind %s:%d — %s",
            _api_host,
            _api_port,
            exc,
        )
    except Exception as exc:
        logging.error("API vision: fallo uvicorn: %s", exc, exc_info=True)
    finally:
        logging.debug("API vision: hilo uvicorn finalizado.")


def start_api_thread(host: str, port: int) -> threading.Thread:
    """Inicia uvicorn en hilo dedicado."""
    global _server, _thread, _api_host, _api_port
    if _thread is not None and _thread.is_alive():
        return _thread

    _api_host = host
    _api_port = port

    config = uvicorn.Config(
        app,
        host=host,
        port=port,
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
    logging.debug("API vision: hilo iniciado en %s:%d", host, port)
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
