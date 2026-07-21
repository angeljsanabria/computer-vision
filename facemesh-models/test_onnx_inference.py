"""
Prueba ONNX FaceMesh 468 con el modulo inference/facemesh (sin webcam).

Usa RetinaFace del repo + estimate_from_det. Imagen por defecto: primera JPG
en mobilenet_modelos/calib/ (recorte de cara generico).

Ejemplo:
  cd facemesh-models
  python extract_from_mesh_originals.py
  python test_onnx_inference.py
  python test_onnx_inference.py --image ../mobilenet_modelos/calib/test.jpg --save out.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from configs import settings as s  # noqa: E402
from inference import build_face_detector, build_face_mesh  # noqa: E402
from inference.facemesh.from_detection import estimate_from_det  # noqa: E402
from inference.retinaface.select_best import mejores_caras  # noqa: E402
from ui.facemesh_overlay import draw_facemesh_landmarks  # noqa: E402

DEFAULT_ONNX = ROOT / "models_onnx" / "face_mesh_192x192.onnx"
DEFAULT_RETINA = ROOT / "models_onnx" / "RetinaFace_mobile320.onnx"
DEFAULT_IMAGE = None  # resuelto en main() via find_test_image o --image obligatorio


def _resolve_image(path: Path | None) -> Path:
    if path is not None and path.is_file():
        return path
    from find_test_image import _find_images  # noqa: WPS433

    found = _find_images()
    if found:
        print("Usando imagen:", found[0].relative_to(ROOT))
        return found[0]
    raise SystemExit(
        "Sin imagen de test. Opciones:\n"
        "  python find_test_image.py\n"
        "  python test_onnx_inference.py --image ruta/a/foto.jpg\n"
        "  Colocar JPG en mobilenet_modelos/calib/ y correr prepare_calib.py"
    )


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Test FaceMesh ONNX via inference/facemesh")
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--save", type=Path, default=None, help="Guardar overlay JPG")
    p.add_argument("--show", action="store_true", help="Mostrar ventana OpenCV")
    return p.parse_args()


def main() -> None:
    args = _parse()
    args.image = _resolve_image(args.image)
    if not args.onnx.is_file():
        raise SystemExit(f"No existe ONNX: {args.onnx}")

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"No se pudo leer: {args.image}")

    face = build_face_detector(
        "pc",
        str(DEFAULT_RETINA),
        s.RETINAFACE_SCORE_DETECCION,
        s.RETINAFACE_SCORE_PRE_NMS,
    )
    mesh = build_face_mesh("pc", str(args.onnx))
    if face is None or mesh is None:
        raise SystemExit("No se pudo crear RetinaFace o FaceMesh")

    try:
        dets = mejores_caras(face.detect(frame), top_n=1)
        landmarks = None
        if dets.has_faces:
            landmarks = estimate_from_det(
                frame,
                dets.dets[0],
                mesh,
                margin_frac=s.FACE_CROP_MARGIN_FRAC,
            )
        vis = draw_facemesh_landmarks(frame, landmarks, draw_crop_rect=True)
        n_pts = 0 if landmarks is None else landmarks.count
        print(f"landmarks: {n_pts} puntos en frame")

        if args.save:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save), vis)
            print("guardado:", args.save)

        if args.show:
            cv2.imshow("FaceMesh test", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    finally:
        mesh.release()


if __name__ == "__main__":
    main()
