"""
Experimento A/B: crop vs roll-fix vs align ArcFace.

Para cada imagen de entrada:
  1. RetinaFace ONNX detecta la mejor cara.
  2. Genera parches 112x112 con tres preprocess.
  3. MobileFaceNet ONNX obtiene embeddings.
  4. Compara similitud coseno vs galeria embeddings/*.npy (refs crop actuales).

Guarda parches en --out-dir y un reporte JSON.

Ejemplo:
  python export_models/ab_preprocess_compare.py --img test/angel1.jpg
  python export_models/ab_preprocess_compare.py --img test/otra.jpg --gallery-dir embeddings
  python export_models/ab_preprocess_compare.py --img test/angel1.jpg --out-dir test/ab_out
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
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

from inference.face_align import align_face_from_det_row  # noqa: E402
from inference.face_pose import eye_roll_deg_from_det_row  # noqa: E402
from inference.face_preprocess import crop_bbox_to_size  # noqa: E402
from inference.face_roll_fix import crop_roll_fix_to_size  # noqa: E402
from inference.mobilefacenet.norm import l2_normalize  # noqa: E402
from inference.mobilefacenet.preprocess import bgr112_to_onnx_nchw  # noqa: E402
from utils.aux_tools_retinaface import (  # noqa: E402
    RETINAFACE_INPUT_HEIGHT,
    RETINAFACE_INPUT_WIDTH,
    RETINAFACE_LETTERBOX_FILL,
    retinaface_dets_desde_rknn_outputs,
)
from utils.image_utils import letterbox_bgr  # noqa: E402

RETINAFACE_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)
RETINAFACE_SCORE_PRE_NMS = 0.02
RETINAFACE_SCORE_DETECCION = 0.2
MIN_SCORE_MEJOR_CARA = 0.5
DEFAULT_MARGIN = 0.15
MODES = ("crop", "roll_fix", "arcface_align")


@dataclass(frozen=True)
class ModeResult:
    mode: str
    roll_deg: float
    preprocess_ms: float
    embed_ms: float
    embedding: np.ndarray
    patch_bgr: np.ndarray
    gallery_sims: dict[str, float]
    best_label: str
    best_sim: float


def _preprocess_retinaface(inp0_meta: ort.NodeArg, canvas_bgr: np.ndarray) -> np.ndarray:
    if canvas_bgr.dtype != np.uint8:
        canvas_bgr = canvas_bgr.astype(np.uint8)
    x32 = canvas_bgr.astype(np.float32)
    if inp0_meta.shape[1] == 3:
        chw = np.transpose(x32, (2, 0, 1))
        feed = np.expand_dims(chw, axis=0)
        feed -= RETINAFACE_MEAN_BGR.reshape(1, 3, 1, 1)
        return feed
    return np.expand_dims(x32, axis=0)


def _detect_best_face(
    sess_rf: ort.InferenceSession,
    img_bgr: np.ndarray,
) -> tuple[np.ndarray, float]:
    h, w = img_bgr.shape[:2]
    rf_in = sess_rf.get_inputs()[0]
    rf_name = rf_in.name

    letterbox_img, lb_meta = letterbox_bgr(
        img_bgr,
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
        raise RuntimeError(
            "RetinaFace no detecto caras. Prueba otra imagen o baja el umbral."
        )
    best = dets[np.argmax(dets[:, 4])]
    score = float(best[4])
    if score < MIN_SCORE_MEJOR_CARA:
        raise RuntimeError(
            f"Score mejor cara {score:.4f} < minimo {MIN_SCORE_MEJOR_CARA}"
        )
    return best, score


def _build_patch(
    mode: str,
    img_bgr: np.ndarray,
    det_row: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, float]:
    roll_deg = eye_roll_deg_from_det_row(det_row)
    if mode == "crop":
        patch = crop_bbox_to_size(
            img_bgr,
            det_row,
            margin_frac=margin,
        )
        return patch, roll_deg
    if mode == "roll_fix":
        patch, roll_deg = crop_roll_fix_to_size(
            img_bgr,
            det_row,
            margin_frac=margin,
        )
        return patch, roll_deg
    if mode == "arcface_align":
        patch = align_face_from_det_row(img_bgr, det_row)
        return patch, roll_deg
    raise ValueError(f"mode invalido: {mode!r}")


def _load_gallery(gallery_dir: Path) -> dict[str, np.ndarray]:
    if not gallery_dir.is_dir():
        raise FileNotFoundError(f"No existe galeria: {gallery_dir}")
    refs: dict[str, np.ndarray] = {}
    for path in sorted(gallery_dir.glob("*.npy")):
        vec = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
        refs[path.stem] = l2_normalize(vec)
    if not refs:
        raise FileNotFoundError(f"Galeria vacia (sin .npy): {gallery_dir}")
    return refs


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a.reshape(-1), b.reshape(-1)))


def _run_mode(
    mode: str,
    img_bgr: np.ndarray,
    det_row: np.ndarray,
    margin: float,
    sess_mfn: ort.InferenceSession,
    gallery: dict[str, np.ndarray],
) -> ModeResult:
    mfn_in = sess_mfn.get_inputs()[0].name
    mfn_out = sess_mfn.get_outputs()[0].name

    t0 = time.perf_counter()
    patch, roll_deg = _build_patch(mode, img_bgr, det_row, margin)
    t_pre = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    feed = bgr112_to_onnx_nchw(patch)
    emb = l2_normalize(np.asarray(sess_mfn.run([mfn_out], {mfn_in: feed})[0], dtype=np.float32))
    t_emb = (time.perf_counter() - t1) * 1000.0

    sims = {label: _cosine_sim(emb, ref) for label, ref in gallery.items()}
    best_label = max(sims, key=sims.get)
    return ModeResult(
        mode=mode,
        roll_deg=roll_deg,
        preprocess_ms=t_pre,
        embed_ms=t_emb,
        embedding=emb,
        patch_bgr=patch,
        gallery_sims=sims,
        best_label=best_label,
        best_sim=sims[best_label],
    )


def _save_collage(results: list[ModeResult], out_path: Path) -> None:
    tiles = []
    for r in results:
        tile = r.patch_bgr.copy()
        label = (
            f"{r.mode} roll={r.preprocess_ms:.1f}ms sim={r.best_sim:.3f}"
        )
        cv2.putText(
            tile,
            r.mode,
            (4, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            f"sim={r.best_sim:.3f}",
            (4, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    collage = np.hstack(tiles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), collage)


def _print_table(results: list[ModeResult], gallery_labels: list[str]) -> None:
    header = f"{'mode':<14} {'roll':>6} {'pre_ms':>7} {'emb_ms':>7} {'best':>10} {'best_sim':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.mode:<14} {r.roll_deg:>6.1f} {r.preprocess_ms:>7.2f} "
            f"{r.embed_ms:>7.2f} {r.best_label:>10} {r.best_sim:>8.4f}"
        )
    print()
    for label in gallery_labels:
        row = f"{label:>10}"
        for r in results:
            row += f"  {r.gallery_sims[label]:>7.4f}"
        print(row)


def main() -> None:
    default_retina = ROOT / "Retinaface-Models" / "RetinaFace_mobile320.onnx"
    default_mfn = ROOT / "mobilenet_modelos" / "MobileFaceNet.onnx"
    default_img = ROOT / "test" / "angel1.jpg"

    p = argparse.ArgumentParser(
        description="A/B preprocess: crop vs roll_fix vs arcface_align."
    )
    p.add_argument("--img", type=str, default=str(default_img), help="Imagen de prueba")
    p.add_argument(
        "--gallery-dir",
        type=str,
        default=str(ROOT / "embeddings"),
        help="Carpeta con refs .npy (enroladas con crop)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "test" / "ab_preprocess_out"),
        help="Donde guardar parches y reporte JSON",
    )
    p.add_argument("--retinaface-onnx", type=str, default=str(default_retina))
    p.add_argument("--mobilefacenet-onnx", type=str, default=str(default_mfn))
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    p.add_argument(
        "--sim-min-match",
        type=float,
        default=0.45,
        help="Umbral MATCH (solo reporte; no afecta inferencia)",
    )
    p.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help="Providers ORT separados por coma",
    )
    args = p.parse_args()

    img_path = Path(args.img)
    gallery_dir = Path(args.gallery_dir)
    out_dir = Path(args.out_dir)
    rf_onnx = Path(args.retinaface_onnx)
    mfn_onnx = Path(args.mobilefacenet_onnx)

    for label, path in (
        ("imagen", img_path),
        ("RetinaFace", rf_onnx),
        ("MobileFaceNet", mfn_onnx),
    ):
        if not path.is_file():
            raise SystemExit(f"No existe {label}: {path}")

    gallery = _load_gallery(gallery_dir)
    gallery_labels = sorted(gallery.keys())

    img = cv2.imread(str(img_path))
    if img is None:
        raise SystemExit(f"No se pudo leer imagen: {img_path}")

    providers = [x.strip() for x in args.providers.split(",") if x.strip()]
    sess_rf = ort.InferenceSession(str(rf_onnx), providers=providers)
    sess_mfn = ort.InferenceSession(str(mfn_onnx), providers=providers)

    det_row, det_score = _detect_best_face(sess_rf, img)
    print(f"imagen: {img_path}")
    print(f"det_score: {det_score:.4f}  roll_ojos: {eye_roll_deg_from_det_row(det_row):.2f} deg")
    print(f"galeria: {gallery_dir} ({len(gallery)} refs)")
    print()

    results: list[ModeResult] = []
    for mode in MODES:
        r = _run_mode(mode, img, det_row, args.margin, sess_mfn, gallery)
        results.append(r)
        patch_path = out_dir / f"{img_path.stem}_{mode}.jpg"
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(patch_path), r.patch_bgr)
        print(f"[patch] {patch_path}")

    _save_collage(results, out_dir / f"{img_path.stem}_collage.jpg")
    _print_table(results, gallery_labels)

    report = {
        "img": str(img_path.resolve()),
        "det_score": det_score,
        "roll_deg": float(eye_roll_deg_from_det_row(det_row)),
        "gallery_dir": str(gallery_dir.resolve()),
        "sim_min_match": args.sim_min_match,
        "margin": args.margin,
        "modes": [],
    }
    for r in results:
        match = r.best_sim >= args.sim_min_match
        report["modes"].append(
            {
                "mode": r.mode,
                "roll_deg": r.roll_deg,
                "preprocess_ms": round(r.preprocess_ms, 3),
                "embed_ms": round(r.embed_ms, 3),
                "gallery_sims": {k: round(v, 6) for k, v in r.gallery_sims.items()},
                "best_label": r.best_label,
                "best_sim": round(r.best_sim, 6),
                "match": match,
            }
        )

    report_path = out_dir / f"{img_path.stem}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[collage] {out_dir / f'{img_path.stem}_collage.jpg'}")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
