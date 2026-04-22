# Edge IoT MYD-LR3568-GK-B (RK3568)

Implementacion del plan: pipeline **ONNX / entrenamiento -> RKNN-Toolkit2 -> `.rknn` -> RKNN Lite** en la placa, con **manifiestos versionados** y entorno de export reproducible (Linux x86_64, recomendado via Docker).

## Contenido

| Ruta | Proposito |
|------|-----------|
| [models/](models/) | Esquema JSON y ejemplo de manifiesto por modelo |
| [docker/](docker/) | Imagen Ubuntu 22.04 para RKNN-Toolkit2 (export en PC) |
| [scripts/](scripts/) | Export demo a `.rknn` (host) e inferencia RKNN Lite (dispositivo) |
| `requirements-toolkit-host.txt` | Dependencias del **host de export** (Linux/Docker) |
| `requirements-device.txt` | Dependencias en la **placa** (aarch64) |

## Flujo rapido (E2E demo)

1. Completar versiones reales en [Docu/MYD-LR3568_BSP_versiones.md](../Docu/MYD-LR3568_BSP_versiones.md).
2. En **Mac**: clonar/abrir este repo; usar Docker Desktop para construir la imagen de `docker/` (ver [MAC_DESARROLLO.md](MAC_DESARROLLO.md)).
3. En el contenedor Linux: `./scripts/fetch_demo_onnx.sh` y `python3 scripts/export_demo_to_rknn.py --onnx models/demo_mobilenet_v2/mobilenetv2-7.onnx`.
4. En la placa: `sudo ./scripts/deploy_layout.sh demo_mobilenet_v2 0.1.0` y copiar `model.rknn` + `manifest.json` a `/opt/edge-models/...`.
5. En la **MYD-LR3568-GK-B**: `pip install -r requirements-device.txt` y `python3 scripts/infer_rknn_lite.py --rknn /opt/edge-models/demo_mobilenet_v2/0.1.0/model.rknn`.

## Nota sobre GPU en el Mac o PC

La GPU **no** es necesaria para exportar a `.rknn`; solo puede acelerar otras fases (p. ej. entrenamiento en PyTorch). El contenedor usa CPU.
