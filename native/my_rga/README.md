# my_rga — extension nativa RGA (RK3568)

Modulo Python `my_rga` con resize, letterbox y BGR->RGB via **librga** (IM2D API).
Basado en la logica de `rknn_model_zoo/utils/image_utils.c`.

Documentacion operativa (pip, cp312, env vars): `native/my_rga/deploy/README.md`.

## Build en WSL (cross aarch64)

```bash
chmod +x native/build_my_rga_wsl.sh
./native/build_my_rga_wsl.sh
```

Wheel de salida: `native/wheels/my_rga-0.1.0-cp312-cp312-linux_aarch64.whl`

## API

| Funcion | Retorno |
|---------|---------|
| `resize_bgr(input, out_w, out_h)` | `(ndarray, used_rga: bool)` |
| `letterbox_bgr(input, canvas_w, canvas_h, fill_value)` | `(canvas, scale, pad_x, pad_y, used_rga)` |
| `bgr_to_rgb(input)` | `(rgb, used_rga)` |

## Verificacion en placa

```bash
python native/my_rga_smoke.py
# o: python -c "import my_rga; help(my_rga)"
```

## Fallback

Si RGA no puede procesar, el codigo C hace fallback CPU. En PC (x86_64) no se instala;
el pipeline usa OpenCV via `utils/image_backend.py`.
