# Export MobileFaceNet ONNX → RKNN (Python 3.11)

Guía para generar `MobileFaceNet.rknn` desde WSL/Linux en PC. El script es `exp_mobilefacenet_rknn.py`.

## Por qué este entorno (y no otro)

| Entorno | ¿Sirve para export? | Motivo |
|---------|---------------------|--------|
| Windows + `cvEnvOk` | No | No existe wheel de RKNN-Toolkit2 para Windows. |
| Placa RK3568 + `rknn_toolkit_lite2` (aarch64) | No | **Lite2** es solo para **inferencia** en la placa (`RKNNLite`). No incluye `load_onnx`, `build` ni cuantización INT8. |
| WSL/Linux + **RKNN-Toolkit2** (x86_64) | Sí | Es el toolkit de **conversión** ONNX → RKNN en PC. |

En la RK3568 usás Python **3.11.2** con:

`rknn_toolkit_lite2-2.3.2-cp311-...-aarch64.whl`

Para exportar en PC usás el **mismo Python 3.11** pero el paquete distinto:

`rknn_toolkit2-2.3.2-cp311-...-x86_64.whl`

Misma versión del SDK (2.3.2), mismo tag `cp311`, distinta arquitectura (`x86_64` vs `aarch64`) y distinto rol (export vs runtime).

## Requisitos previos

- WSL2 o Linux x86_64.
- Python **3.11** instalado en WSL (`python3.11 --version`).
- En esta carpeta: `MobileFaceNet.onnx`, `MobileFaceNet.onnx.data`.
- Para INT8: `dataset.txt` y las JPG listadas en `calib/`.

## 1. Instalar RKNN-Toolkit2 (Python 3.11)

Ejecutar en WSL:

```bash
cd /mnt/c/code/computer-vision
git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git

python3.11 -m venv ~/venv-rknn311
source ~/venv-rknn311/bin/activate
python -m pip install --upgrade pip

cd rknn-toolkit2/rknn-toolkit2/packages/x86_64
pip install -r requirements_cp311-2.3.2.txt
pip install rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# RKNN 2.3.2 no es compatible con ONNX >= 1.19 (falta onnx.mapping).
# Fijar version antes del export:
pip install onnx==1.18.0 onnxruntime==1.18.0

python -c "from rknn.api import RKNN; print('RKNN OK')"
```

Si el clone de `rknn-toolkit2` ya existe, omitir el `git clone`.

## 2. Export ONNX → RKNN

Activar el venv (si no está activo) y correr el script:

```bash
source ~/venv-rknn311/bin/activate
cd /mnt/c/code/computer-vision/mobilenet_modelos
python exp_mobilefacenet_rknn.py
```

Salida esperada: `MobileFaceNet.rknn` en esta carpeta.

## Flag `DO_QUANTIZATION`

En `exp_mobilefacenet_rknn.py` (constante al inicio del archivo):

| Valor | Efecto |
|-------|--------|
| `False` | RKNN flotante; no usa `dataset.txt`. Útil para verificar que el ONNX carga bien. |
| `True` | RKNN INT8; usa `dataset.txt` y las imágenes de `calib/` para calibración. |

Flujo sugerido: primero `DO_QUANTIZATION = False`, luego `True` con más caras en `calib/` y listadas en `dataset.txt`.

## Error `onnx has no attribute 'mapping'`

Si `load_onnx` falla con ese mensaje, el venv tiene ONNX demasiado nuevo (p. ej. 1.19+). El `requirements_cp311-2.3.2.txt` pide `onnx>=1.16.1` sin tope y pip instala la ultima.

Corregir en el venv activo:

```bash
source ~/venv-rknn311/bin/activate
pip install onnx==1.18.0 onnxruntime==1.18.0
python -c "import onnx; print(onnx.__version__)"
cd /mnt/c/code/computer-vision/mobilenet_modelos
python exp_mobilefacenet_rknn.py
```

## Después del export

Copiar `MobileFaceNet.rknn` a la placa. Allí la inferencia usa **RKNN-Toolkit-Lite2** (`rknnlite`), no este toolkit de PC.
