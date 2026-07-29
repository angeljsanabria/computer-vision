# Nozzle Bidon/Pico — nozzle_bidones_v4

## Objetivos

- Dataset Roboflow `picos_y_bidones` (2 clases).
- Entrada **416** (prueba vs v3@640).
- RKNN **hybrid INT8** (output0 FP16).
- Runtime: paquete nuevo `inference/nozzle_bidon/` (no muta `inference/nozzle`).

## Nombres de clase

| ID en labels | Roboflow | Producto |
|--------------|----------|----------|
| 0 | jerrycan | Bidon |
| 1 | nozzle | Pico |

Los `.txt` del export **no se reescriben** por rename; solo el mapa en config.

## 1. Dataset

Export Roboflow **YOLOv8** (no OBB) en:

`yolo_train/picos_y_bidones.v1i.yolov8/`

Si las labels son poligono (seg), convertir a bbox:

```bash
python yolo_train/prepare_nozzle_detect_labels.py
```

Salida: `yolo_train/picos_y_bidones.v1i.yolov8_detect/`

## 2. Config

Hub: `yolo_train/nozzle_config.py`

- `NOZZLE_VERSION = "nozzle_bidones_v4"`
- `DATASET_DIR = "picos_y_bidones.v1i.yolov8_detect"`
- `CLASS_NAMES = ("Bidon", "Pico")`
- `IMGSZ = 416` / `RKNN_INPUT_SIZE = 416`

## 3. Entrenar → ONNX → calib → RKNN

```bash
python yolo_train/prepare_nozzle_detect_labels.py
python yolo_train/train_nozzle.py
python yolo_train/export_nozzle_onnx.py
python yolo_train/prepare_nozzle_calib_v4.py
# WSL rknn-toolkit2 2.3.2:
python yolo_train/exp_yolov8n_nozzle_rknn_v4.py
```

## 4. Artefactos

| Artefacto | Ruta |
|-----------|------|
| Pesos | `yolo_train/runs/detect/nozzle_bidones_v4/weights/best.pt` |
| ONNX | `Yolo-Weights/yolov8n_nozzle_bidones_v4.onnx` |
| RKNN | `Yolo-Weights/yolov8n_nozzle_bidones_v4.rknn` |
| Deploy placa | `models/yolov8n_nozzle_bidones_v4.rknn` |

## 5. Runtime placa

```bash
ENABLE_NOZZLE=true
NOZZLE_MODEL_RK3568=models/yolov8n_nozzle_bidones_v4.rknn
```

`main_track` usa solo `inference/nozzle_bidon/` + `build_nozzle_bidon_detector`.  
El paquete `inference/nozzle/` (v3@640) permanece en el repo como modulo; no lo cablea este pipeline.
