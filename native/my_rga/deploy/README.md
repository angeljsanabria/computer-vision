# Wheels aarch64 — RKNN Lite + my_rga

Instalar en el **mismo Python** que ejecuta `main_track` (Conda `anpr` en produccion).

## Compatibilidad Python / arquitectura

Leer el nombre del archivo `.whl`, igual que RKNN Lite:

| Wheel | Para |
|-------|------|
| `...-cp312-cp312-linux_aarch64.whl` | **Python 3.12** en **aarch64** (RK3568) |
| `...-cp311-cp311-linux_aarch64.whl` | Python 3.11 aarch64 (otra placa / imagen) |

Verificar en dispositivo:

```bash
python --version
python -c "import platform; print(platform.machine())"
```

Si el Python no coincide: `pip` rechaza el wheel o falla `import my_rga` con
`Python version mismatch`. Usar el wheel con el tag `cp` correcto.

## Instalacion (bare metal)

```bash
/opt/conda/envs/anpr/bin/pip install --force-reinstall \
  /opt/anpr-core/rknn-toolkit-lite/rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

/opt/conda/envs/anpr/bin/pip install --force-reinstall \
  /opt/anpr-core/rknn-toolkit-lite/my_rga-0.1.0-cp312-cp312-linux_aarch64.whl
```

## Docker (ejemplo)

```dockerfile
RUN pip3 install --no-cache-dir \
    rknn-toolkit-lite/rknn_toolkit_lite2-2.3.2-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
    rknn-toolkit-lite/my_rga-0.1.0-cp312-cp312-linux_aarch64.whl
```

## Variables de entorno (pip no las setea)

```bash
export INFERENCE_BACKEND=rk3568
export USE_RGA=true
# Solo si import falla por librga:
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH
```

Requisito sistema: `librga.so` (tipico `/usr/lib/aarch64-linux-gnu/librga.so`).

## API `my_rga`

| Funcion | Entrada | Salida | Uso en pipeline |
|---------|---------|--------|-----------------|
| `resize_bgr(input, out_w, out_h)` | BGR uint8 (H,W,3) | `(ndarray, used_rga)` | MOG2, crops, nozzle |
| `letterbox_bgr(input, canvas_w, canvas_h, fill_value)` | BGR uint8 | `(canvas, scale, pad_x, pad_y, used_rga)` | RetinaFace 320 |
| `bgr_to_rgb(input)` | BGR uint8 | `(rgb, used_rga)` | Preprocess RKNN |

`used_rga=True` → driver RGA proceso; `False` → fallback CPU interno.

## Comprobar instalacion

```bash
python -c "import my_rga; help(my_rga)"
python -c "import my_rga, numpy as np; z=np.zeros((48,64,3),np.uint8); print(my_rga.resize_bgr(z,32,32))"
```

Monitor RGA (opcional): `sudo cat /sys/kernel/debug/rkrga/load`
