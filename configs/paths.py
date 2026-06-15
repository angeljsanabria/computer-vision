"""Rutas del repo (misma logica que scripts en export_models/)."""
from __future__ import annotations

from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """
    Raiz del repositorio: primer ancestro que contiene ``utils/aux_tools_retinaface.py``.

    Mismo criterio que ``_project_root_with_utils`` en export_models y WIP/main_mov.py.
    """
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for d in [cur, *cur.parents]:
        if (d / "utils" / "aux_tools_retinaface.py").is_file():
            return d
    raise RuntimeError(
        "No se encontro raiz del repo (utils/aux_tools_retinaface.py)."
    )


def resolve_repo_path(path: str | Path, *, start: Path | None = None) -> Path:
    """Ruta absoluta: absoluta tal cual; relativa respecto a ``project_root()``."""
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root(start) / p
