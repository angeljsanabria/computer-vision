# Nozzle Bidon/Pico — nozzle_bidones_v7

## Objetivos

- Dataset Roboflow `picos-bidones-v7.yolov8` (2 clases).
- Entrada **416** (igual que v4).
- RKNN **hybrid INT8** (output0 FP16).
- Runtime: `inference/nozzle_bidon/`.

## Nombres de clase

| ID en labels | Roboflow | Producto (config) |
|--------------|----------|-------------------|
| 0 | Bidon | Bidon |
| 1 | nozzle | Pico |

## 1. Dataset

Export en:

`yolo_train/picos-bidones-v7.yolov8/`

Si las labels son poligono (seg), convertir a bbox:

```bash
python yolo_train/prepare_nozzle_detect_labels.py
```

Salida: `yolo_train/picos-bidones-v7.yolov8_detect/`

## 2. Config

Hub: `yolo_train/nozzle_config.py`

- `NOZZLE_VERSION = "nozzle_bidones_v7"`
- `DATASET_DIR = "picos-bidones-v7.yolov8_detect"`
- `CLASS_NAMES = ("Bidon", "Pico")`
- `IMGSZ = 416` / `RKNN_INPUT_SIZE = 416`

## 3. Entrenar → ONNX → calib → RKNN

```bash
python yolo_train/prepare_nozzle_detect_labels.py
python yolo_train/train_nozzle.py
python yolo_train/export_nozzle_onnx.py
python yolo_train/prepare_nozzle_calib_v7.py
# WSL rknn-toolkit2 2.3.2:
python yolo_train/exp_yolov8n_nozzle_rknn_v7.py
```

## 4. Artefactos

| Artefacto | Ruta |
|-----------|------|
| Pesos | `yolo_train/runs/detect/nozzle_bidones_v7/weights/best.pt` |
| ONNX | `Yolo-Weights/yolov8n_nozzle_bidones_v7.onnx` |
| RKNN | `Yolo-Weights/yolov8n_nozzle_bidones_v7.rknn` |
| Deploy placa | `models/yolov8n_nozzle_bidones_v7.rknn` |

## 5. Runtime placa

```bash
ENABLE_NOZZLE=true
NOZZLE_MODEL_RK3568=models/yolov8n_nozzle_bidones_v7.rknn
```

`main_track` usa solo `inference/nozzle_bidon/` + `build_nozzle_bidon_detector`.
