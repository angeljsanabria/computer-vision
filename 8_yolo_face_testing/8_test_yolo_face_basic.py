"""
Ejecutar con el entorno virtual .venvYoloFace (no con el venv de YOLO objeto).
  Activar: source .venvYoloFace/bin/activate
  Desde la raiz del proyecto: python 8_yolo_face_testing/8_test_yolo_face_basic.py

Script basico de prueba de deteccion de CARAS con YOLOv8-face (derronqi/yolov8-face).
Misma API que el ejemplo del repo: test_widerface.py (model.predict sin show).

Pesos: NO se descargan solos. Descargar desde derronqi/yolov8-face (enlaces
Google Drive en el README) y colocar en models/Yolo-Weights-face/ (en la raiz del proyecto).
  (yolov8-lite-t.pt, yolov8-lite-s.pt, yolov8n-face.pt)

Entorno: ver ENV_YOLOV8_FACE.md en esta carpeta.

Compatibilidad: el fork derronqi/yolov8-face con fuse=True deja dilation como (bool, bool)
en las Conv fusionadas; PyTorch exige tuple of ints. Este script aplica parches solo en
memoria (fuse=False + F.conv2d con tipos corregidos) para que la inferencia funcione sin
modificar el repo.

Parametros:
- M_LITE_T, M_LITE_S, M_NANO: activan cada variante (solo una en True).
- IMAGE: True = una imagen (images/lily2.jpg), False = video (videos/lily.mp4).
- SHOW: True = mostrar ventana (cv2.imshow); False = solo guardar en runs/detect/predict.
"""
import os
import sys
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils.torch_utils import select_device

# Parches de compatibilidad con derronqi/yolov8-face (fuse deja dilation como bool; Conv sin bias)
_conv2d_orig = F.conv2d

def _conv2d_compat(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    if bias is None:
        bias = torch.zeros(weight.shape[0], device=input.device, dtype=input.dtype)
    stride = (int(stride), int(stride)) if isinstance(stride, int) else (int(stride[0]), int(stride[1]))
    padding = padding if isinstance(padding, str) else (
        (int(padding), int(padding)) if isinstance(padding, int) else (int(padding[0]), int(padding[1]))
    )
    dilation = (int(dilation), int(dilation)) if isinstance(dilation, int) else (int(dilation[0]), int(dilation[1]))
    return _conv2d_orig(input, weight, bias, stride, padding, dilation, int(groups))

F.conv2d = _conv2d_compat

_orig_setup = BasePredictor.setup_model

def _setup_no_fuse(self, model, verbose=True):
    device = select_device(self.args.device, verbose=verbose)
    model = model or self.args.model
    self.args.half &= device.type != "cpu"
    self.model = AutoBackend(model, device=device, dnn=self.args.dnn, data=self.args.data,
                             fp16=self.args.half, fuse=False, verbose=verbose)
    self.device = device
    self.model.eval()

BasePredictor.setup_model = _setup_no_fuse

# Pesos en la raiz del proyecto: cv/models/Yolo-Weights-face
_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR_SCRIPT, ".."))
WEIGHTS_DIR = os.path.join(_ROOT, "models", "Yolo-Weights-face")

M_LITE_T = False
M_LITE_S = True
M_NANO = False

IMAGE = True
SHOW = True


def main():
    if M_LITE_T:
        ruta = os.path.join(WEIGHTS_DIR, "yolov8-lite-t.pt")
    elif M_LITE_S:
        ruta = os.path.join(WEIGHTS_DIR, "yolov8-lite-s.pt")
    else:
        ruta = os.path.join(WEIGHTS_DIR, "yolov8n-face.pt")

    if not os.path.isfile(ruta):
        print(f"Falta el archivo de pesos: {ruta}")
        print("Descargalos desde https://github.com/derronqi/yolov8-face (enlaces en el README)")
        sys.exit(1)

    model = YOLO(ruta)
    # Rutas relativas al directorio de trabajo (ejecutar desde raiz del proyecto)
    source = "images/lily2.jpg" if IMAGE else "videos/lily.mp4"

    results = model.predict(
        source=source,
        imgsz=640,
        conf=0.35,
        iou=0.5,
        save=True,
        verbose=True,
    )

    if SHOW and len(results) > 0:
        win = "YOLOv8-face"
        if len(results) == 1:
            r = results[0]
            if r.orig_img is not None:
                cv2.imshow(win, r.plot())
            cv2.waitKey(0)
        else:
            for r in results:
                if r.orig_img is not None:
                    cv2.imshow(win, r.plot())
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("Fin")


if __name__ == "__main__":
    main()
