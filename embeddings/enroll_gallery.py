"""
Orquesta el enrolamiento de galeria en dos pasos:

  1. prepare_faces_refs.py       faces/ -> faces_upd/
  2. face_embeddings_npy_from_images_folder.py  faces_upd/ -> gallery.npy

Si el paso 1 falla, no se ejecuta el paso 2.

Ejemplo:
  python embeddings/enroll_gallery.py
"""
import logging
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PREPARE_SCRIPT = SCRIPT_DIR / "prepare_faces_refs.py"
EMBED_SCRIPT = SCRIPT_DIR / "face_embeddings_npy_from_images_folder.py"


def _run_step(label: str, cmd: list[str]) -> None:
    logging.info("=== %s ===", label)
    logging.info("Comando: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    python = sys.executable
    steps: list[tuple[str, list[str]]] = [
        ("Paso 1/2: prepare_faces_refs", [python, str(PREPARE_SCRIPT)]),
        (
            "Paso 2/2: face_embeddings_npy_from_images_folder",
            [python, str(EMBED_SCRIPT)],
        ),
    ]

    for label, cmd in steps:
        try:
            _run_step(label, cmd)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode) from exc

    logging.info(
        "Enrolamiento completo: %s y %s en %s",
        "gallery.npy",
        "gallery_meta.json",
        SCRIPT_DIR.resolve(),
    )


if __name__ == "__main__":
    main()
