# Pruebas con RetinaFace (serengil)

Detector de caras basado en RetinaFace (TensorFlow). No mezclar con el venv de YOLOv8-face (PyTorch).

## Instalacion

```bash
cd /Users/angel-dev/PycharmProjects/cv
python3.10 -m venv .venvRetinaFace
source .venvRetinaFace/bin/activate
pip install --upgrade pip
pip install -r 9_retinaface_testing/requirements-retinaface.txt
pip list
pip freeze
```

## Ejecutar

Desde la raiz del proyecto:

```bash
source .venvRetinaFace/bin/activate
python 9_retinaface_testing/retinaface_test_basic.py
```

El script usa imagenes en `images/` o video en `videos/` (rutas relativas al directorio de trabajo). Pesos se descargan automaticamente la primera vez.
