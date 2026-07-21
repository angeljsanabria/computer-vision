# Origen del ONNX FaceMesh 468

El modelo desplegado en `models_onnx/face_mesh_192x192.onnx` proviene de **`mesh-originals/032_FaceMesh.tar.gz`** (export PINTO / MediaPipe Face Mesh, modelo 032).

## Que hay en `mesh-originals/`

| Carpeta / archivo | Landmarks | Input | Relacion con nuestro ONNX |
|-------------------|-----------|-------|---------------------------|
| **`032_FaceMesh.tar.gz`** | **468** | 192×192 RGB | **Fuente correcta** |
| `face-landmark-detection/` | 98 | 112×112 | PFLD distinto; no usar para FaceMesh 468 |
| `pfld_106_face_landmarks/` | 106 | 112×112 | PFLD distinto; no usar para FaceMesh 468 |

## Variantes ONNX dentro del tar (carpeta `20_new_onnx_postprocess_N-batch`)

| Archivo | Entradas | Salidas | Uso en este repo |
|---------|----------|---------|------------------|
| `face_mesh_192x192.onnx` | `input` [1,3,192,192] | `landmarks` [1,1,1,1404], `score` [1,1,1,1] | **Este exportamos a RKNN** (igual que `models_onnx/`) |
| `face_mesh_192x192_post.onnx` | `input` + crop_x1/y1/width/height | `final_landmarks` [1,468,3], `score` | Referencia dorada para validar `postprocess.py` |
| `post_process.onnx` | landmarks crudos + crop | `final_landmarks` | Solo postprocess MediaPipe |

## Preprocess (alineado con `inference/facemesh/preprocess.py`)

- ONNX PC: BGR → RGB float32 NCHW `/255` → [0,1]
- TFLite original (json en tar): NHWC `[1,192,192,3]` float32
- RKNN placa (planeado): RGB uint8 NHWC + `mean=0`, `std=255`

## Extraer artefactos al repo de trabajo

```bash
cd facemesh-models
python extract_from_mesh_originals.py
```

Copia los ONNX de referencia a `facemesh-models/originals/`. El script confirma MD5 **identico** con `models_onnx/face_mesh_192x192.onnx`.

## Validacion postprocess

`compare_postprocess.py` compara `landmarks_mesh_to_frame` (Python) vs `face_mesh_192x192_post.onnx` (PINTO). Error tipico ~0.4 px (sub-pixel); aceptable para overlay.

Inputs crop del ONNX `_post`: **int32** `[1,1]` para x1, y1, width, height.

## Imagenes de test / calibracion INT8

No hay JPG en `032_FaceMesh.tar.gz`. El script `prepare_calib.py` arma `calib/` desde:

| Fuente | Imagenes | Modo |
|--------|----------|------|
| `mobilenet_modelos/calib/` | 7 (cara2-7, test) | resize directo 192x192 |
| `embeddings/faces_upd/*_zero.jpg` | 3 | center crop + resize |
| `embeddings/faces/` | 4 enrolamiento | center crop + resize |
| `embeddings/faces_upd/*_{der,izq}.jpg` | 6 perfiles | center crop + resize |
| `camara_snap/latest_*retinaface*.jpg`, `latest_camara_snap.jpg` | 2 | center crop + resize |
| `export_models/result_retinaface_onnx.jpg` | 1 | center crop + resize |
| `mesh-originals/pfld_106_face_landmarks/1.png` | 1 | center crop + resize |

```bash
cd facemesh-models
py -3.10 prepare_calib.py              # 24 JPG -> calib/ + dataset.txt
py -3.10 prepare_calib.py --retinaface # recorte bbox si hay onnxruntime
```

**No usar** para calib: `get_dataset/`, `yolo_train/` (surtidor), COCO generico, duplicados RetinaFace `test.jpg`.
