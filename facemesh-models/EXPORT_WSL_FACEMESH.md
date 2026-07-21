# Export FaceMesh 468 → RKNN en WSL (paso a paso)

Runbook para convertir `face_mesh_192x192.onnx` a `face_mesh_192x192.rknn` en **WSL2/Linux x86_64**, usando el **mismo entorno virtual** que MobileFaceNet.

Plan de fondo: `PLAN_FACEMESH_RKNN.md`. Origen del ONNX: `MESH_ORIGINALS.md`.

---

## 1. Entorno: que usar y que no

| Entorno | Sirve para export | Motivo |
|---------|-------------------|--------|
| Windows + Python nativo | **No** | No hay wheel de RKNN-Toolkit2 para Windows |
| Placa RK3568 + `rknn_toolkit_lite2` | **No** | Lite2 solo **inferencia** (`RKNNLite`); no tiene `load_onnx` ni `build` |
| WSL + **`~/venv-rknn311`** | **Si** | RKNN-Toolkit2 x86_64, Python 3.11, mismo flujo que MobileFaceNet |
| WSL + `~/venv-rknn310` | **No** (FaceMesh) | Python 3.10 + ONNX 1.16.1; quedo para YOLO (`NOTAS RK.txt`) |

### Entorno virtual en esta maquina (verificado)

```
Ruta:     ~/venv-rknn311   (ej. /root/venv-rknn311 en WSL)
Python:   3.11.x           (/usr/bin/python3.11)
Creado:   jun-2026         (misma sesion que MobileFaceNet.rknn)
```

Paquetes relevantes instalados:

| Paquete | Version | Rol |
|---------|---------|-----|
| `rknn-toolkit2` | **2.3.2** | Conversion ONNX → RKNN, cuantizacion INT8 |
| `onnx` | **1.18.0** | Parser ONNX (2.3.2 rompe con ONNX >= 1.19) |
| `onnxruntime` | **1.18.0** | Dependencia del toolkit |
| `numpy` | 1.26.4 | Arrays en calibracion |
| `opencv-python`, `torch`, etc. | (deps del requirements) | Soporte interno del toolkit |

Comprobar antes de exportar:

```bash
source ~/venv-rknn311/bin/activate
python --version
python -c "from rknn.api import RKNN; import onnx; print('RKNN OK, onnx', onnx.__version__)"
# Esperado: Python 3.11.x, RKNN OK, onnx 1.18.0
```

---

## 2. Rockchip toolkit: que hay en el repo

El repo incluye (o clona) **`rknn-toolkit2`** de airockchip. Estructura util:

```
computer-vision/
├── rknn-toolkit2/                          # clone del SDK (PC, x86_64)
│   └── rknn-toolkit2/
│       └── packages/
│           └── x86_64/
│               ├── requirements_cp311-2.3.2.txt
│               └── rknn_toolkit2-2.3.2-cp311-...-x86_64.whl   # wheel de export
├── rknn-toolkit2/rknn-toolkit-lite2/       # runtime en placa (aarch64) — NO usar para export
└── facemesh-models/
    ├── exp_facemesh_rknn.py                # script de conversion
    ├── dataset.txt                         # 24 rutas calib/ (INT8)
    └── calib/*.jpg                         # parches 192x192
```

**RKNN-Toolkit2** (PC): `from rknn.api import RKNN` — `config`, `load_onnx`, `build`, `export_rknn`.

**RKNN-Toolkit-Lite2** (placa): `from rknnlite.api import RKNNLite` — solo `load_rknn` + `inference`.

Misma version SDK (**2.3.2**), distinta arquitectura (`x86_64` export vs `aarch64` runtime).

Si falta el clone:

```bash
cd /mnt/c/code/computer-vision
git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git
```

---

## 3. Crear el venv (solo si no existe)

Si `~/venv-rknn311` ya existe y el smoke test de la seccion 1 pasa, **saltar esta seccion**.

```bash
cd /mnt/c/code/computer-vision

# Solo si no hay clone:
# git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git

python3.11 -m venv ~/venv-rknn311
source ~/venv-rknn311/bin/activate
python -m pip install --upgrade pip

cd rknn-toolkit2/rknn-toolkit2/packages/x86_64
pip install -r requirements_cp311-2.3.2.txt
pip install rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Fijar ONNX compatible con RKNN 2.3.2:
pip install onnx==1.18.0 onnxruntime==1.18.0

python -c "from rknn.api import RKNN; import onnx; print('RKNN OK', onnx.__version__)"
```

Guia equivalente para MobileFaceNet: `mobilenet_modelos/export_mobilefacenet_rknn.md`.

---

## 4. Archivos necesarios (estado actual)

| Archivo | Ruta | Estado |
|---------|------|--------|
| ONNX origen | `models_onnx/face_mesh_192x192.onnx` | En repo (~2.4 MB) |
| Script export | `facemesh-models/exp_facemesh_rknn.py` | Listo |
| Calibracion INT8 | `facemesh-models/calib/*.jpg` | **24 imagenes** 192x192 |
| Dataset INT8 | `facemesh-models/dataset.txt` | **24 entradas** |
| Salida esperada | `facemesh-models/face_mesh_192x192.rknn` | Se genera en WSL |
| Destino runtime | `models/face_mesh_192x192.rknn` | Copia manual post-export |

Regenerar calib desde Windows (opcional, si cambian fotos):

```powershell
cd C:\code\computer-vision\facemesh-models
py -3.10 prepare_calib.py
```

En WSL no hace falta OpenCV para el export; solo para regenerar calib.

---

## 5. Paso a paso en WSL

Abrir terminal WSL y ejecutar en orden.

### Paso 0 — Entrar al proyecto y activar venv

```bash
source ~/venv-rknn311/bin/activate
cd /mnt/c/code/computer-vision/facemesh-models
```

### Paso 1 — Verificar ONNX (I/O y tamano)

```bash
python inspect_onnx.py
```

Esperado:

- Input: `input` → `[1, 3, 192, 192]`
- Output: `landmarks` → 1404 (= 468 × 3), `score` auxiliar

### Paso 2 — Verificar calibracion INT8

```bash
wc -l dataset.txt
ls calib/*.jpg | wc -l
# Ambos deben coincidir (24)
head -3 dataset.txt
```

Si faltan JPG, volver a correr `prepare_calib.py` en Windows o copiar JPG a `calib/` y actualizar `dataset.txt`.

### Paso 3 — Export flotante (sin INT8)

Primera corrida: validar que el ONNX carga y compila.

1. Editar `exp_facemesh_rknn.py` linea ~32:

   ```python
   DO_QUANTIZATION = False
   ```

2. Ejecutar:

   ```bash
   python exp_facemesh_rknn.py
   ```

3. Esperado al final:

   ```
   OK -> .../facemesh-models/face_mesh_192x192.rknn
   Copiar a runtime: cp face_mesh_192x192.rknn ../models/
   ```

Si `load_onnx` o `build` fallan, ver seccion 7 (errores).

### Paso 4 — Export INT8 (produccion)

1. Editar `exp_facemesh_rknn.py`:

   ```python
   DO_QUANTIZATION = True
   ```

2. Ejecutar:

   ```bash
   python exp_facemesh_rknn.py
   ```

Usa `dataset.txt` + las 24 JPG de `calib/` para cuantizar.

### Paso 5 — Copiar a `models/`

```bash
cp face_mesh_192x192.rknn ../models/
ls -la ../models/face_mesh_192x192.rknn
```

### Paso 6 — (Opcional) Probar en placa

En RK3568, con `INFERENCE_BACKEND=rk3568` y `ENABLE_FACEMESH=true`:

- Path: `models/face_mesh_192x192.rknn` (`settings_track.py` → `FACEMESH_MODEL_RK3568`)
- Completar/implementar `inference/facemesh/estimator_rk3568.py` si sigue en stub
- Probar overlay en `main_track.py` o PoC `main3.py`

---

## 6. Config del export (referencia)

Definido en `exp_facemesh_rknn.py`:

| Parametro | Valor | Equivalente PC |
|-----------|-------|------------------|
| `TARGET_PLATFORM` | `rk3568` | Placa destino |
| `MEAN_VALUES` | `[0, 0, 0]` | — |
| `STD_VALUES` | `[255, 255, 255]` | `/255` en float (ONNX PC) |
| Entrada runtime placa | RGB uint8 192×192 NHWC | `bgr192_to_rknn_nhwc()` |

Preprocess alineado con `inference/facemesh/preprocess.py`.

---

## 7. Errores frecuentes

### `onnx has no attribute 'mapping'`

ONNX demasiado nuevo en el venv:

```bash
source ~/venv-rknn311/bin/activate
pip install onnx==1.18.0 onnxruntime==1.18.0
python -c "import onnx; print(onnx.__version__)"
```

### `No module named 'rknn'`

Venv incorrecto o no activado:

```bash
source ~/venv-rknn311/bin/activate
which python   # debe apuntar a ~/venv-rknn311/bin/python
```

### `Imagenes del dataset no encontradas`

Rutas en `dataset.txt` son relativas a `facemesh-models/`. Verificar:

```bash
cd /mnt/c/code/computer-vision/facemesh-models
test -f calib/mobilenet_calib_test.jpg && echo OK
```

### `load_onnx failed` / ops no soportados

Revisar log verbose de `build`. Probar primero `DO_QUANTIZATION = False`. Consultar `rknn-toolkit2/doc/RKNNToolKit2_OP_Support-2.3.2.md`.

### INT8 degrada landmarks

Agregar mas caras en `calib/` (regenerar con `prepare_calib.py`), re-export INT8, o usar FP (`DO_QUANTIZATION = False`) si la calidad no alcanza.

---

## 8. Checklist rapido

- [ ] WSL abierto, `source ~/venv-rknn311/bin/activate`
- [ ] `python -c "from rknn.api import RKNN"` → OK
- [ ] `onnx.__version__` → 1.18.0
- [ ] `models_onnx/face_mesh_192x192.onnx` existe
- [ ] `calib/` → 24 JPG, `dataset.txt` → 24 lineas
- [ ] `DO_QUANTIZATION = False` → export FP OK
- [ ] `DO_QUANTIZATION = True` → export INT8 OK
- [ ] `cp face_mesh_192x192.rknn ../models/`
- [ ] Probar en placa / completar `estimator_rk3568.py`

---

## 9. Orden de ejecucion resumido (copiar/pegar)

```bash
source ~/venv-rknn311/bin/activate
cd /mnt/c/code/computer-vision/facemesh-models

python inspect_onnx.py
wc -l dataset.txt && ls calib/*.jpg | wc -l

# FP (DO_QUANTIZATION = False en exp_facemesh_rknn.py)
python exp_facemesh_rknn.py

# INT8 (DO_QUANTIZATION = True)
python exp_facemesh_rknn.py

cp face_mesh_192x192.rknn ../models/
ls -la ../models/face_mesh_192x192.rknn
```
