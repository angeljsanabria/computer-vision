"""
RetinaFace con ONNX en PC (OnnxRuntime).

Copia adaptada desde RetinaFace_from_img.py: mismo letterbox y mismo postproceso
(aux_tools_retinaface.retinaface_dets_desde_rknn_outputs) para validar el pipeline
respecto al .rknn en placa.

Preproceso: letterbox BGR 320x320, conversion a tensor segun entrada del ONNX
(typico input0 float32 [1,3,H,W] BGR menos media Caffe RetinaFace 104,117,123).

Ejemplo:
  cd /mnt/c/code/computer-vision
  pip install onnxruntime  # si falta
  python export_models/RetinaFace_from_img_onnx.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _project_root_with_utils(start_dir: Path) -> Path:
    cur = start_dir.resolve()
    for d in [cur, *cur.parents]:
        u = d / "utils"
        if u.is_dir() and (u / "aux_tools_retinaface.py").is_file():
            return d
    raise RuntimeError(
        "No se encontro carpeta 'utils' con aux_tools_retinaface.py. "
        f"Origen de busqueda: {start_dir}"
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

try:
    import onnxruntime as ort
except Exception as e:
    raise SystemExit(
        "No se pudo importar onnxruntime.\n"
        f"  Python usado: {sys.executable}\n"
        "  Instala en ESE intérprete (con el venv activado): pip install onnxruntime\n"
        f"  Detalle: {e!r}"
    ) from e

RETINAFACE_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)

RETINAFACE_SCORE_PRE_NMS = 0.02
RETINAFACE_SCORE_DETECCION = 0.2


def _preprocess_for_onnx(inp0_meta: ort.NodeArg, canvas_bgr: np.ndarray) -> np.ndarray:
    """
    Canvas BGR uint8 (H,W,3) ya en 320x320 (letterbox).
    Produce tensor listo para session.run segun firma ONNX (NCHW o NHWC).
    """
    if canvas_bgr.dtype != np.uint8:
        canvas_bgr = canvas_bgr.astype(np.uint8)
    x32 = canvas_bgr.astype(np.float32)
    dims = inp0_meta.shape

    # Detectar orden NCHW vs NHWC usando dimensiones conocidas para 320 entrada
    nchw = False
    nhwc = False
    if len(dims) == 4 and dims[1] == 3:
        nchw = True
    elif len(dims) == 4 and dims[-1] == 3:
        nhwc = True
    elif len(dims) == 4 and dims[1] == RETINAFACE_INPUT_WIDTH:
        # A veces [1,H,W,C]
        nhwc = True
    else:
        # Por defecto: NCHW (export Rockchip habitual)
        nchw = True

    if nhwc:
        x32 -= RETINAFACE_MEAN_BGR.reshape(1, 1, 3)
        return np.expand_dims(x32, axis=0)

    if nchw:
        chw = np.transpose(x32, (2, 0, 1))
        feed = np.expand_dims(chw, axis=0)
        feed -= RETINAFACE_MEAN_BGR.reshape(1, 3, 1, 1)
        return feed

    raise RuntimeError("Forma de entrada ONNX no soportada: " + repr(dims))


if __name__ == "__main__":
    _repo = ROOT
    _default_onnx = _repo / "Retinaface-Models" / "RetinaFace_mobile320.onnx"
    _default_img = _repo / "Retinaface-Models" / "RK" / "model" / "test.jpg"
    if not _default_img.is_file():
        _default_img = _repo / "Retinaface-Models" / "test.jpg"
    _default_out = _repo / "export_models" / "result_retinaface_onnx.jpg"

    parser = argparse.ArgumentParser(
        description="RetinaFace ONNX en PC (validacion auxiliar antes de RKNN)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(_default_onnx),
        help="Ruta al .onnx (defecto: Retinaface-Models/RetinaFace_mobile320.onnx)",
    )
    parser.add_argument(
        "--img",
        type=str,
        default=str(_default_img),
        help="Imagen de entrada",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(_default_out),
        help="Imagen de salida",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help='OnnxRuntime providers separados por coma (ej: CPUExecutionProvider)',
    )
    args = parser.parse_args()

    if not Path(args.model_path).is_file():
        raise SystemExit(f"No existe el ONNX: {args.model_path}")

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    session = ort.InferenceSession(args.model_path, providers=providers)
    inp0 = session.get_inputs()[0]
    input_name = inp0.name

    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit("No se pudo leer la imagen: " + args.img)
    img_height, img_width = img.shape[:2]

    letterbox_img, lb_meta = letterbox_bgr(
        img,
        (RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT),
        RETINAFACE_LETTERBOX_FILL,
    )
    tensor = _preprocess_for_onnx(inp0, letterbox_img)
    ort_outputs = session.run(None, {input_name: tensor})

    dets = retinaface_dets_desde_rknn_outputs(
        list(ort_outputs),
        img_width=img_width,
        img_height=img_height,
        aspect_ratio=lb_meta.aspect_ratio,
        offset_x=lb_meta.offset_x,
        offset_y=lb_meta.offset_y,
        score_deteccion=RETINAFACE_SCORE_DETECCION,
        score_pre_nms=RETINAFACE_SCORE_PRE_NMS,
    )

    print(" onnx input:", inp0.name, "shape onnxmeta:", inp0.shape, "fed:", tensor.shape)

    for data in dets:
        print(
            "face @ (%d %d %d %d) %f"
            % (data[0], data[1], data[2], data[3], data[4])
        )
        text = "{:.4f}".format(data[4])
        data = list(map(int, data))
        cv2.rectangle(img, (data[0], data[1]), (data[2], data[3]), (0, 0, 255), 2)
        cx = data[0]
        cy = data[1] + 12
        cv2.putText(img, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        cv2.circle(img, (data[5], data[6]), 1, (0, 0, 255), 5)
        cv2.circle(img, (data[7], data[8]), 1, (0, 255, 255), 5)
        cv2.circle(img, (data[9], data[10]), 1, (255, 0, 255), 5)
        cv2.circle(img, (data[11], data[12]), 1, (0, 255, 0), 5)
        cv2.circle(img, (data[13], data[14]), 1, (255, 0, 0), 5)

    cv2.imwrite(args.out, img)
    print("save image in", args.out)
