"""
Imagen comun (sin transformar el archivo): OpenCV lee BGR tal cual.
RetinaFace ONNX hace letterbox 320x320, deteccion y NMS; luego se recorta la mejor cara
y MobileFaceNet obtiene el embedding (128).

Por defecto la entrada es test/micara1.jpg (relativo a la raiz del repo).

El embedding se guarda en .npy (float32 shape (128,)); MobileFaceNet: RGB [0,1] + Normalize ImageNet.

Por defecto se crea la carpeta embeddings/ en la raiz del repo y el archivo se llama como la imagen
(sin extension original), ej. test/micara1.jpg -> embeddings/micara1.npy.

Si la mejor cara tiene score RetinaFace por debajo de MIN_SCORE_MEJOR_CARA_EMBEDDING (constante
en el script, defecto 0.90), se aborta sin generar embedding.

Ejemplo:
  python export_models/face_embedding_from_image.py
  python export_models/face_embedding_from_image.py --img test/otra.jpg
  python export_models/face_embedding_from_image.py --embedding-dir /tmp/mis_emb
  python export_models/face_embedding_from_image.py --preprocess arcface_align --img test/angel1.jpg
  python export_models/face_embedding_from_image.py --preprocess roll_fix --embedding-dir embeddings_roll
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    raise SystemExit(
        "No se pudo importar onnxruntime.\n"
        f"  Python: {sys.executable}\n"
        "  pip install onnxruntime\n"
        f"  Detalle: {e!r}"
    ) from e


def _project_root_with_utils(start_dir: Path) -> Path:
    cur = start_dir.resolve()
    for d in [cur, *cur.parents]:
        u = d / "utils"
        if u.is_dir() and (u / "aux_tools_retinaface.py").is_file():
            return d
    raise RuntimeError(
        "No se encontro utils/aux_tools_retinaface.py desde " + str(start_dir)
    )


ROOT = _project_root_with_utils(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.aux_tools_retinaface import (  # noqa: E402
    RETINAFACE_INPUT_HEIGHT,
    RETINAFACE_INPUT_WIDTH,
    RETINAFACE_LETTERBOX_FILL,
    retinaface_dets_desde_rknn_outputs,
)
from utils.image_utils import letterbox_bgr  # noqa: E402
from inference.face_preprocess import prepare_face_patch  # noqa: E402

RETINAFACE_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)
RETINAFACE_SCORE_PRE_NMS = 0.02
RETINAFACE_SCORE_DETECCION = 0.2

# Score minimo de la mejor deteccion RetinaFace para generar embedding (ajustar si hace falta).
MIN_SCORE_MEJOR_CARA_EMBEDDING = 0.90

# MobileFaceNet (foamliu): torchvision val
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
_MOBILEFACENET_HW = (112, 112)


def _preprocess_retinaface(inp0_meta: ort.NodeArg, canvas_bgr: np.ndarray) -> np.ndarray:
    if canvas_bgr.dtype != np.uint8:
        canvas_bgr = canvas_bgr.astype(np.uint8)
    x32 = canvas_bgr.astype(np.float32)
    dims = inp0_meta.shape
    nchw = len(dims) == 4 and dims[1] == 3
    nhwc = len(dims) == 4 and dims[-1] == 3
    if len(dims) == 4 and dims[1] == RETINAFACE_INPUT_WIDTH:
        nhwc = True
    if not nchw and not nhwc:
        nchw = True
    if nhwc:
        x32 -= RETINAFACE_MEAN_BGR.reshape(1, 1, 3)
        return np.expand_dims(x32, axis=0)
    chw = np.transpose(x32, (2, 0, 1))
    feed = np.expand_dims(chw, axis=0)
    feed -= RETINAFACE_MEAN_BGR.reshape(1, 3, 1, 1)
    return feed


def _bbox_crop_with_margin(
    det: np.ndarray, img_w: int, img_h: int, margin: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    mx = bw * margin
    my = bh * margin
    nx1 = int(np.floor(x1 - mx))
    ny1 = int(np.floor(y1 - my))
    nx2 = int(np.ceil(x2 + mx))
    ny2 = int(np.ceil(y2 + my))
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w - 1, nx2)
    ny2 = min(img_h - 1, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return int(x1), int(y1), int(x2), int(y2)
    return nx1, ny1, nx2, ny2


def _crop_bgr_to_mobilefacenet_nchw(face_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 (H,W,3) -> float32 NCHW (1,3,112,112) foamliu val."""
    if face_bgr.size == 0:
        raise ValueError("recorte vacio")
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = cv2.resize(rgb, _MOBILEFACENET_HW, interpolation=cv2.INTER_LINEAR)
    norm = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(norm, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32), axis=0)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec.reshape(-1), ord=2))
    if n < 1e-12:
        return vec
    return (vec / n).astype(np.float32)


def main() -> None:
    default_retina = ROOT / "Retinaface-Models" / "RetinaFace_mobile320.onnx"
    default_mfn = ROOT / "mobilenet_modelos" / "MobileFaceNet.onnx"
    default_img = ROOT / "test" / "angel1.jpg"

    p = argparse.ArgumentParser(
        description="Embedding facial: RetinaFace ONNX + recorte + MobileFaceNet ONNX."
    )
    p.add_argument(
        "--img",
        type=str,
        default=str(default_img),
        help="Imagen entrada (foto comun, sin preproceso manual). Defecto: test/micara1.jpg bajo la raiz del repo.",
    )
    p.add_argument(
        "--retinaface-onnx",
        type=str,
        default=str(default_retina),
        help="ONNX RetinaFace",
    )
    p.add_argument(
        "--mobilefacenet-onnx",
        type=str,
        default=str(default_mfn),
        help="ONNX MobileFaceNet (requiere .onnx.data en la misma carpeta si aplica).",
    )
    p.add_argument(
        "--embedding-dir",
        type=str,
        default=str(ROOT / "embeddings"),
        help="Carpeta donde guardar el .npy si no se pasa --embedding-out (se crea si no existe).",
    )
    p.add_argument(
        "--embedding-out",
        type=str,
        default="",
        help="Ruta completa del .npy (si vacio: <embedding-dir>/<mismo nombre base que la imagen>.npy).",
    )
    p.add_argument(
        "--preprocess",
        type=str,
        choices=("crop", "roll_fix", "arcface_align"),
        default="crop",
        help=(
            "Parche 112x112 antes del embedder: crop (defecto), roll_fix hibrido "
            "o arcface_align (mismo criterio que FACE_ALIGNMENT_ENABLE en runtime)."
        ),
    )
    p.add_argument(
        "--roll-max-deg",
        type=float,
        default=10.0,
        help="Umbral |roll| para roll_fix (solo si --preprocess roll_fix).",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.15,
        help="Margen fraccion caja deteccion al recortar (0.15 = 15%% por lado).",
    )
    p.add_argument(
        "--flip-avg",
        action="store_true",
        help="Promedio embedding cara + espejo horizontal (como lfw_eval foamliu).",
    )
    p.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help="Providers ORT separados por coma.",
    )
    p.add_argument(
        "--meta-json",
        type=str,
        default="",
        help="Si se indica, guarda score y rutas en JSON auxiliar.",
    )
    args = p.parse_args()

    img_path = Path(args.img)
    if not img_path.is_file():
        raise SystemExit(f"No existe la imagen: {img_path}")

    rf_onnx = Path(args.retinaface_onnx)
    mfn_onnx = Path(args.mobilefacenet_onnx)
    for label, path in (("RetinaFace", rf_onnx), ("MobileFaceNet", mfn_onnx)):
        if not path.is_file():
            raise SystemExit(f"No existe ONNX {label}: {path}")

    providers = [x.strip() for x in args.providers.split(",") if x.strip()]
    sess_rf = ort.InferenceSession(str(rf_onnx), providers=providers)
    sess_mfn = ort.InferenceSession(str(mfn_onnx), providers=providers)

    rf_in = sess_rf.get_inputs()[0]
    rf_name = rf_in.name
    mfn_in = sess_mfn.get_inputs()[0]
    mfn_out = sess_mfn.get_outputs()[0].name
    mfn_in_name = mfn_in.name

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"No se pudo leer la imagen: {img_path}")
    h, w = img.shape[:2]

    letterbox_img, lb_meta = letterbox_bgr(
        img,
        (RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT),
        RETINAFACE_LETTERBOX_FILL,
    )
    tensor_rf = _preprocess_retinaface(rf_in, letterbox_img)
    out_rf = sess_rf.run(None, {rf_name: tensor_rf})
    dets = retinaface_dets_desde_rknn_outputs(
        list(out_rf),
        img_width=w,
        img_height=h,
        aspect_ratio=lb_meta.aspect_ratio,
        offset_x=lb_meta.offset_x,
        offset_y=lb_meta.offset_y,
        score_deteccion=RETINAFACE_SCORE_DETECCION,
        score_pre_nms=RETINAFACE_SCORE_PRE_NMS,
    )

    if dets.shape[0] == 0:
        raise SystemExit(
            "RetinaFace no detecto caras sobre el umbral. "
            "Prueba otra imagen o baja RETINAFACE_SCORE_DETECCION en el script."
        )

    best = dets[np.argmax(dets[:, 4])]
    score = float(best[4])
    if score < MIN_SCORE_MEJOR_CARA_EMBEDDING:
        raise SystemExit(
            "No se puede analizar: el score de la mejor cara ("
            f"{score:.4f}) no supera el minimo requerido "
            f"({MIN_SCORE_MEJOR_CARA_EMBEDDING}). "
            "Mejora iluminacion / encuadre o baja MIN_SCORE_MEJOR_CARA_EMBEDDING "
            "en export_models/face_embedding_from_image.py"
        )

    arcface = args.preprocess == "arcface_align"
    roll_fix = args.preprocess == "roll_fix"
    patch = prepare_face_patch(
        img,
        best,
        arcface_align_enable=arcface,
        rot_align_simple_enable=roll_fix,
        max_abs_roll_deg=args.roll_max_deg,
        crop_margin_frac=args.margin,
    )
    face112 = patch.bgr
    if args.save_crop:
        Path(args.save_crop).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.save_crop, face112)
        print("[save-crop]", args.save_crop)

    feed = _crop_bgr_to_mobilefacenet_nchw(face112)
    emb = sess_mfn.run([mfn_out], {mfn_in_name: feed})[0]

    if args.flip_avg:
        face_flip = cv2.flip(face112, 1)
        feed2 = _crop_bgr_to_mobilefacenet_nchw(face_flip)
        emb2 = sess_mfn.run([mfn_out], {mfn_in_name: feed2})[0]
        emb = emb + emb2

    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    emb = _l2_normalize(emb)

    if args.embedding_out:
        out_npy = Path(args.embedding_out)
    else:
        emb_dir = Path(args.embedding_dir)
        out_npy = emb_dir / f"{img_path.stem}.npy"
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_npy), emb)

    print("score_mejor_cara:", score)
    print("preprocess:", args.preprocess, "roll_deg:", round(patch.roll_deg, 2))
    print("embedding_shape:", emb.shape, "L2_norm:", float(np.linalg.norm(emb)))
    print("[guardado]", out_npy.resolve())

    if args.meta_json:
        meta = {
            "img": str(img_path.resolve()),
            "retinaface_onnx": str(rf_onnx.resolve()),
            "mobilefacenet_onnx": str(mfn_onnx.resolve()),
            "score": score,
            "preprocess": args.preprocess,
            "roll_deg": patch.roll_deg,
            "used_arcface_align": patch.used_arcface_align,
            "used_roll_fix": patch.used_roll_fix,
            "embedding_npy": str(out_npy.resolve()),
            "flip_avg": bool(args.flip_avg),
        }
        Path(args.meta_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.meta_json).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("[meta-json]", args.meta_json)


if __name__ == "__main__":
    main()
