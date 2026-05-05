"""
RetinaFace (.rknn) en la placa Rockchip: inferencia local con RKNN-Toolkit-Lite2.

Copia logica de RetinaFace.py del Zoo (letterbox, decode, NMS), pero usa rknnlite
(NPU local). Postproceso unificado en ``utils.aux_tools_retinaface``.

Requisito en la placa: pip3 install rknn_toolkit_lite2 (wheel acorde a Python).

Layout tipico (``utils`` a la misma altura que ``wip`` o ``scripts``):
  ./utils/
  ./wip/RetinaFace_lite.py   <- este script
  ./model/ ...

Tambien valido bajo ``cv/Retinaface-Models/RK/``: se sube por padres hasta
encontrar un directorio que contenga ``utils/aux_tools_retinaface.py``.

Ejemplo:
  cd /home/myir/Documents/git-ctk/anpr-core/wip
  python3 RetinaFace_lite.py

Otro .rknn (ruta absoluta o relativa a donde ejecutes):
  python3 RetinaFace_lite.py --model_path ../models/RetinaFace_mobile320.rknn
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def _project_root_with_utils(start_dir: Path) -> Path:
    """Sube desde start_dir hasta encontrar ./utils/ (RetinaFace + image_utils)."""
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

from utils.aux_tools_retinaface import (
    RETINAFACE_INPUT_HEIGHT,
    RETINAFACE_INPUT_WIDTH,
    RETINAFACE_LETTERBOX_FILL,
    retinaface_dets_desde_rknn_outputs,
)
from utils.image_utils import letterbox_bgr

try:
    from rknnlite.api import RKNNLite
except ImportError as e:
    raise SystemExit(
        "Instala RKNN-Toolkit-Lite2 en la placa (rknnlite). "
        "Ej.: pip3 install --user rknn_toolkit_lite2-....whl"
    ) from e

# Umbrales sobre conf[:, 1] (probabilidad de cara). Ajustar precision vs recall.
# Pre-NMS: descarta anclas muy debiles (valor del demo Zoo).
RETINAFACE_SCORE_PRE_NMS = 0.02
# Deteccion final: la utilidad solo devuelve filas con score >= este valor.
RETINAFACE_SCORE_DETECCION = 0.2


if __name__ == "__main__":
    # Script en .../wip/RetinaFace_lite.py -> una carpeta atras -> .../anpr-core/
    # Ahi estan model/ y test/ (no dentro de wip/).
    print("*+++++++++++++++")
    _wip = Path(__file__).resolve().parent
    _repo = _wip.parent
    print("*+++++++++++++++")
    print(_wip)
    print(_repo)
    _default_rknn = _repo / "model" / "RetinaFace_mobile320_2.rknn"
    #_default_img = _repo / "test" / "test.jpg"
    _default_img = _repo / "test" / "test3.jpg"
    _default_out = _repo / "test" / "result_lite.jpg"

    parser = argparse.ArgumentParser(
        description="RetinaFace RKNN en placa (RKNNLite, sin target remoto)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(_default_rknn),
        help="Ruta al .rknn en la placa",
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
    args = parser.parse_args()

    if not Path(args.model_path).is_file():
        raise SystemExit(f"No existe el modelo: {args.model_path}")

    rknn = RKNNLite()
    print("--> load_rknn", args.model_path)
    ret = rknn.load_rknn(args.model_path)
    if ret != 0:
        print('Load RKNN model "{}" failed!'.format(args.model_path))
        sys.exit(ret)
    print("done")

    print("--> init_runtime (NPU local, sin target PC)")
    ret = rknn.init_runtime()
    if ret != 0:
        print("init_runtime failed!")
        sys.exit(ret)
    print("done")

    img = cv2.imread(args.img)
    if img is None:
        raise SystemExit("No se pudo leer la imagen: " + args.img)
    img_height, img_width = img.shape[:2]

    letterbox_img, lb_meta = letterbox_bgr(
        img,
        (RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT),
        RETINAFACE_LETTERBOX_FILL,
    )
    infer_rgb = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
    input_tensor = np.expand_dims(infer_rgb, axis=0)

    print("--> Running model")
    outputs = rknn.inference(inputs=[input_tensor])
    if not outputs:
        rknn.release()
        raise SystemExit("inference sin salidas")

    dets = retinaface_dets_desde_rknn_outputs(
        outputs,
        img_width=img_width,
        img_height=img_height,
        aspect_ratio=lb_meta.aspect_ratio,
        offset_x=lb_meta.offset_x,
        offset_y=lb_meta.offset_y,
        score_deteccion=RETINAFACE_SCORE_DETECCION,
        score_pre_nms=RETINAFACE_SCORE_PRE_NMS,
    )

    for data in dets:
        print("face @ (%d %d %d %d) %f" % (data[0], data[1], data[2], data[3], data[4]))
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
    rknn.release()
