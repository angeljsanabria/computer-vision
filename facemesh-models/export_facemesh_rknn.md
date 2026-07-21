# Export FaceMesh ONNX → RKNN (Python 3.11, WSL)

Guía rápida. **Paso a paso completo (entorno venv + toolkit + checklist):** `EXPORT_WSL_FACEMESH.md`.  
Plan: `PLAN_FACEMESH_RKNN.md`.

## Entorno

Igual que MobileFaceNet — **no Windows**, **no RKNN Lite en placa** para export.

```bash
source ~/venv-rknn311/bin/activate   # mismo venv que mobilenet_modelos
cd /mnt/c/code/computer-vision/facemesh-models
```

Si no tenés el venv: seguir `mobilenet_modelos/export_mobilefacenet_rknn.md` sección 1.

## Validacion PC (antes de WSL)

```bash
python extract_from_mesh_originals.py   # confirma MD5 vs models_onnx/
python compare_postprocess.py           # postprocess.py vs _post.onnx (sin OpenCV)
python find_test_image.py               # localizar JPG en el repo
python test_onnx_inference.py --save test_out.jpg   # requiere cv2 + RetinaFace ONNX
```

Origen ONNX: `mesh-originals/032_FaceMesh.tar.gz` — ver `MESH_ORIGINALS.md`.

## Archivos necesarios

| Archivo | Origen |
|---------|--------|
| `../models_onnx/face_mesh_192x192.onnx` | Ya en el repo (no copiar; el script usa ruta absoluta) |
| `calib/*.jpg` | `py -3.10 prepare_calib.py` (24 entradas desde el repo) |
| `dataset.txt` | Ya en esta carpeta |

## Pasos

```bash
# 1) Verificar ONNX
python inspect_onnx.py

# 2) Calibracion INT8 (opcional antes del primer export fp)
python prepare_calib.py

# 3) Export — primero sin INT8
#    Editar DO_QUANTIZATION = False en exp_facemesh_rknn.py
python exp_facemesh_rknn.py

# 4) Export INT8
#    DO_QUANTIZATION = True
python exp_facemesh_rknn.py
```

Salida: `facemesh-models/face_mesh_192x192.rknn`

## Despliegue

```bash
cp face_mesh_192x192.rknn ../models/
```

En placa: `INFERENCE_BACKEND=rk3568`, path `models/face_mesh_192x192.rknn` (ver `settings_track.py`).

## Preprocess RKNN

- Entrada: RGB uint8 192×192 (NHWC en runtime placa).
- Config: `mean=[0,0,0]`, `std=[255,255,255]` → mismo `/255` que ONNX en PC.
