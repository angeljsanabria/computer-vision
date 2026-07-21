"""
Extrae ONNX de referencia desde mesh-originals/032_FaceMesh.tar.gz.

No modifica models_onnx/; deja copias en facemesh-models/originals/ para
test, comparacion y documentacion del export RKNN.
"""
from __future__ import annotations

import hashlib
import io
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAR_PATH = ROOT.parent / "mesh-originals" / "032_FaceMesh.tar.gz"
OUT_DIR = ROOT / "originals"

# ONNX utiles del bundle PINTO (20_new_onnx_postprocess_N-batch)
_POST_BUNDLE = "032_FaceMesh/20_new_onnx_postprocess_N-batch/resources_post.tar.gz"
_FILES = (
    "face_mesh_192x192.onnx",
    "face_mesh_192x192_post.onnx",
    "post_process.onnx",
)

DEPLOYED_ONNX = ROOT.parent / "models_onnx" / "face_mesh_192x192.onnx"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not TAR_PATH.is_file():
        raise SystemExit(f"No existe archivo fuente: {TAR_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outer = tarfile.open(TAR_PATH)
    member = outer.extractfile(_POST_BUNDLE)
    if member is None:
        raise SystemExit(f"No se encontro {_POST_BUNDLE} en el tar")

    inner = tarfile.open(fileobj=io.BytesIO(member.read()))
    for name in _FILES:
        src = inner.extractfile(name)
        if src is None:
            print("WARN: falta", name)
            continue
        dst = OUT_DIR / name
        dst.write_bytes(src.read())
        print("OK", dst.relative_to(ROOT), f"md5={_md5(dst)[:8]}")

    raw = OUT_DIR / "face_mesh_192x192.onnx"
    if raw.is_file() and DEPLOYED_ONNX.is_file():
        same = _md5(raw) == _md5(DEPLOYED_ONNX)
        print(
            "Comparacion models_onnx/face_mesh_192x192.onnx:",
            "IDENTICO" if same else "DISTINTO (revisar cual desplegar)",
        )


if __name__ == "__main__":
    main()
