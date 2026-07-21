# Plan: FaceMesh 468 ONNX → RKNN (RK3568)

Export de `face_mesh_192x192.onnx` a `face_mesh_192x192.rknn`, siguiendo el mismo flujo que ya usa el repo para **RetinaFace** y **MobileFaceNet**.

---

## 1. Referencia en el repo (como se hizo antes)

| Modelo | Carpeta / script | ONNX origen | RKNN salida | Preprocess en `rknn.config` |
|--------|------------------|-------------|-------------|-----------------------------|
| **RetinaFace** | `Retinaface-Models/` + `export_models/exp_retinaface_rknn.py` | `Retinaface-Models/RetinaFace_mobile320.onnx` | `Retinaface-Models/RetinaFace_mobile320.rknn` (copiado a `models/`) | BGR mean `[104,117,123]`, std `[1,1,1]` (Caffe) |
| **MobileFaceNet** | `mobilenet_modelos/exp_mobilefacenet_rknn.py` | `mobilenet_modelos/MobileFaceNet.onnx` (+ `.onnx.data`) | `mobilenet_modelos/MobileFaceNet.rknn` → `models/` | RGB ImageNet mean/std ×255 |
| **FaceMesh** (este plan) | `facemesh-models/exp_facemesh_rknn.py` | `models_onnx/face_mesh_192x192.onnx` | `facemesh-models/face_mesh_192x192.rknn` → `models/` | RGB `/255` → mean `[0,0,0]`, std `[255,255,255]` |

**RetinaFace** ya venía del zoo Rockchip; el `.rknn` en `models/` se generó con `exp_retinaface_rknn.py` (o el `convert.py` original en `Retinaface-Models/RK/`).

**MobileFaceNet** es el mejor template para FaceMesh: mismo entorno WSL, mismo venv RKNN 2.3.2 cp311, dataset INT8 en `calib/`.

---

## 2. Qué reutilizamos del proyecto

### Modelo ONNX (origen)

| Archivo | Ubicación | Notas |
|---------|-----------|--------|
| `face_mesh_192x192.onnx` | `models_onnx/` | ~2.4 MB, **un solo archivo** (sin `.onnx.data`) |
| Fuente PINTO / MediaPipe | `mesh-originals/032_FaceMesh.tar.gz` | Bundle **032 FaceMesh 468**; ver `MESH_ORIGINALS.md` |
| Referencia postprocess | `facemesh-models/originals/` | Tras `extract_from_mesh_originals.py` |
| Input | `input` | `[1, 3, 192, 192]` float32 NCHW |
| Output | `landmarks` | `[1, 1, 1, 1404]` = 468 × 3 (x, y, z) |
| Output extra | `score` | `[1, 1, 1, 1]` — no usado en `inference/facemesh/` (solo landmarks) |

**No confundir** con `mesh-originals/face-landmark-detection/` (PFLD 98, 112×112) ni `pfld_106_face_landmarks/` (106 pts): son modelos distintos.

El ONNX en `models_onnx/` coincide con el del tar PINTO. Runtime PC: `inference/facemesh/` (`estimator_pc.py`, `preprocess.py`, `postprocess.py`).

### Código de inferencia (destino post-export)

| Pieza | Archivo | Estado |
|-------|---------|--------|
| Preprocess RKNN | `inference/facemesh/preprocess.py` → `bgr192_to_rknn_nhwc` | Listo (RGB uint8 NHWC) |
| Estimator placa | `inference/facemesh/estimator_rk3568.py` | **Stub** — completar tras export |
| Settings | `configs/settings_track.py` → `FACEMESH_MODEL_RK3568=models/face_mesh_192x192.rknn` | Path ya definido |
| Postprocess | `inference/facemesh/postprocess.py` | Compartido PC/RKNN (remap mesh→frame) |

### Calibración INT8 (reutilizable)

Script `prepare_calib.py` recopila **24 parches 192×192** desde varias carpetas del repo (mobilenet calib, embeddings, camara_snap, etc.) y escribe `calib/` + `dataset.txt`. Ver tabla en `MESH_ORIGINALS.md`.

### Entorno RKNN (compartido)

Mismo venv que MobileFaceNet (`~/venv-rknn311`, WSL, `rknn_toolkit2` x86_64, `onnx==1.18.0`). Ver `mobilenet_modelos/export_mobilefacenet_rknn.md`.

---

## 3. Preprocess: alinear PC y RKNN

**PC (`bgr192_to_onnx_nchw`):**

```
BGR uint8 192×192 → RGB float32 → /255 → NCHW
```

**RKNN (propuesto en `exp_facemesh_rknn.py`):**

```
Entrada en placa: RGB uint8 NHWC (1, 192, 192, 3)  ← bgr192_to_rknn_nhwc
rknn.config mean_values=[[0,0,0]], std_values=[[255,255,255]]
→ equivalente a dividir por 255 en float
```

Debe coincidir con lo validado en PC antes de confiar en INT8.

---

## 4. Fases de trabajo

### Fase 0 — Preparar carpeta (hecho)

```
facemesh-models/
├── PLAN_FACEMESH_RKNN.md           ← este documento
├── MESH_ORIGINALS.md               ← fuente mesh-originals/032_FaceMesh
├── export_facemesh_rknn.md         ← guía rápida WSL (como mobilenet)
├── EXPORT_WSL_FACEMESH.md          ← paso a paso WSL (venv, toolkit, checklist)
├── exp_facemesh_rknn.py            ← script export
├── extract_from_mesh_originals.py  ← extrae ONNX de referencia del tar
├── test_onnx_inference.py          ← smoke test PC (RetinaFace + FaceMesh)
├── compare_postprocess.py          ← valida postprocess.py vs _post.onnx
├── inspect_onnx.py                 ← verifica I/O del ONNX
├── prepare_calib.py                ← calib 192×192 desde mobilenet_modelos/calib
├── dataset.txt                     ← listado para INT8
├── originals/                      ← ONNX extraidos (gitignore opcional)
└── calib/                          ← JPG 192×192 (generar con prepare_calib.py)
```

### Fase 0b — Validacion PC (antes de RKNN)

1. `python extract_from_mesh_originals.py` — confirma MD5 vs `models_onnx/`.
2. `python compare_postprocess.py` — error xy medio Python vs `face_mesh_192x192_post.onnx`.
3. `python test_onnx_inference.py --save test_out.jpg` — pipeline RetinaFace + `estimate_from_det`.

Imagenes de test: **no hay JPG en `mesh-originals/`**; usar `mobilenet_modelos/calib/*.jpg` o cualquier foto con `--image`.

### Fase 1 — Smoke test sin cuantizar

1. WSL + venv RKNN (ya usado para MobileFaceNet).
2. `python inspect_onnx.py` — confirmar shapes.
3. En `exp_facemesh_rknn.py`: `DO_QUANTIZATION = False`.
4. `python exp_facemesh_rknn.py` → `face_mesh_192x192.rknn`.

Si falla `load_onnx`, revisar versión ONNX (1.18.0) o ops no soportados.

### Fase 2 — INT8 con calibración

1. `python prepare_calib.py` — llena `calib/` con crops 192×192.
2. `DO_QUANTIZATION = True`.
3. Re-export → comparar landmarks vs ONNX en PC (`compare_pc_rknn.py` opcional, fase 3).

### Fase 3 — Validación numérica

1. Misma imagen de prueba: ONNX (PC) vs RKNN (simulador toolkit o placa).
2. Métrica: error medio en landmarks (468 puntos) en espacio mesh [0,192].
3. Umbral aceptable: definir tras primera corrida (p. ej. < 2 px en x/y).

### Fase 4 — Integración en placa

1. Copiar `face_mesh_192x192.rknn` → `models/`.
2. Completar `FaceMeshEstimatorRk3568.estimate()` (patrón `MobileFaceNetEmbedderRk3568`).
3. Probar `main3.py` / `main_track.py` con `INFERENCE_BACKEND=rk3568`, `ENABLE_FACEMESH=true`.

---

## 5. Riesgos conocidos

| Riesgo | Mitigación |
|--------|------------|
| Ops ONNX no soportados por RKNN | Fase 1 sin INT8; revisar log verbose de `build` |
| INT8 degrada landmarks | Más imágenes en `calib/`; probar `DO_QUANTIZATION=False` en prod si hace falta |
| Preprocess distinto PC/RK | Validar con mismo parche BGR 192×192 en compare |
| Modelo experimental | Si no se aprueba UX, se elimina módulo; export queda aislado en `facemesh-models/` |

---

## 6. Checklist final

- [ ] `face_mesh_192x192.rknn` generado en WSL
- [ ] Copiado a `models/face_mesh_192x192.rknn`
- [ ] `estimator_rk3568.py` implementado (no stub)
- [ ] Landmarks visuales OK en overlay (tracks desconocidos)
- [ ] Documentar versión RKNN toolkit usada en el export
