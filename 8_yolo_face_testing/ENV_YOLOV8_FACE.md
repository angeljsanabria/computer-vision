# Entorno virtual para YOLOv8-face

Este venv es **solo para scripts que usan YOLOv8-face** (derronqi). No uses el venv principal de YOLO objeto.

**Python:** Usa **Python 3.10**.

## Instalacion

Usa las versiones fijas en `requirements-yolov8-face.txt` (en esta carpeta). El script incluye parches de compatibilidad necesarios con el fork derronqi:

```bash
cd /Users/angel-dev/PycharmProjects/cv

python3.10 -m venv .venvYoloFace
source .venvYoloFace/bin/activate

pip install --upgrade pip
pip install -r 8_yolo_face_testing/requirements-yolov8-face.txt
cd yolov8-face && pip install -e . --no-build-isolation && cd ..
```

## Ejecutar

Desde la **raiz del proyecto** (cv/), donde esta la carpeta `yolov8-face` y `8_yolo_face_testing`:

```bash
cd /Users/angel-dev/PycharmProjects/cv
source .venvYoloFace/bin/activate
python 8_yolo_face_testing/8_test_yolo_face_basic.py
```

Pesos: descargar de los enlaces del README de derronqi/yolov8-face y poner en `models/Yolo-Weights-face/` (en la raiz del proyecto).
