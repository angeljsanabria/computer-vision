# Nozzle YOLO — nueva version / otro dataset

## 1. Dataset Roboflow

Export YOLOv8 → carpeta en `yolo_train/`, ej. `nozzle_v3.yolov8/`  
(debe tener `train/valid/test` con `images/` + `labels/`)

## 2. Config (unico archivo a tocar)

`yolo_train/nozzle_config.py`:

```python
NOZZLE_VERSION = "v3"
DATASET_DIR = "nozzle_v3.yolov8"
CLASS_NAME = "nozzle"   # como en data.yaml del export
```

Opcional: actualizar `data_nozzle.yaml` (`path` + `names`) — el train regenera el resolved.

## 3. Entrenar → export → RKNN

```bash
python yolo_train/train_nozzle.py
python yolo_train/export_nozzle_onnx.py
python yolo_train/gen_nozzle_rknn_dataset.py
# WSL (rknn-toolkit2 2.3.2):
python yolo_train/exp_yolov8n_nozzle_rknn.py
# FP sin INT8 (solo comparar):
python yolo_train/exp_yolov8n_nozzle_rknn.py --no-quant
```

## 4. Salidas (version v3)

| Artefacto | Ruta |
|-----------|------|
| Pesos | `yolo_train/runs/detect/nozzle_v3/weights/best.pt` |
| ONNX | `Yolo-Weights/yolov8n_nozzle_v3.onnx` |
| RKNN | `Yolo-Weights/yolov8n_nozzle_v3.rknn` |
| Alias desplegado | `Yolo-Weights/yolov8n_nozzle.onnx` / `.rknn` (se sobrescriben al export) |

## 5. Backup version anterior

Antes de entrenar v3, copiar a mano si queres congelar v2:

- `nozzle.yolov8` → `nozzle_v2.yolov8` (ya hecho)
- `yolov8n_nozzle.onnx` → `yolov8n_nozzle_v2.onnx` (ya hecho)

## 6. Probar

```bash
python export_models/nozzle_yolo_v1.py --modo usb --display
# o ONNX versionado:
python export_models/nozzle_yolo_v1.py --modo usb --onnx Yolo-Weights/yolov8n_nozzle_v3.onnx
```
