# Contenedor RKNN-Toolkit2 (export)

Imagen base **Ubuntu 22.04 amd64** con **Python 3.10** y dependencias para exportar modelos a `.rknn`.

## Construir

El `Dockerfile` espera como contexto el directorio `edge_iot_myd` (para copiar `requirements-toolkit-host.txt`):

```bash
cd edge_iot_myd
docker build --platform linux/amd64 -f docker/Dockerfile -t rknn-toolkit2-export .
```

En Mac **Intel**, puedes omitir `--platform linux/amd64`.

## Ejecutar (montando el repo)

```bash
cd /ruta/al/cv
docker run --rm -it \
  --platform linux/amd64 \
  -v "$(pwd):/workspace/cv" \
  -w /workspace/cv/edge_iot_myd \
  rknn-toolkit2-export \
  bash
```

Dentro del contenedor:

```bash
./scripts/fetch_demo_onnx.sh
python3 scripts/export_demo_to_rknn.py --onnx models/demo_mobilenet_v2/mobilenetv2-12.onnx
```

## Alineacion de versiones

Instala en la placa la misma **familia de versiones** de runtime RKNN que la usada por `rknn-toolkit2` en este contenedor. Anota todo en [Docu/MYD-LR3568_BSP_versiones.md](../../Docu/MYD-LR3568_BSP_versiones.md).

## Ajustar version de rknn-toolkit2

Edita [../requirements-toolkit-host.txt](../requirements-toolkit-host.txt) para fijar la version exacta (`rknn-toolkit2==X.Y.Z`) segun BSP MYIR / Rockchip, luego reconstruye la imagen.
