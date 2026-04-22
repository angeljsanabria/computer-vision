# Modelos versionados (`.rknn`)

## Convencion de rutas en la placa (SOM)

Recomendado:

```text
/opt/edge-models/<model_id>/<version>/
  model.rknn
  manifest.json
```

Opcional: enlace simbolico `/opt/edge-models/<model_id>/current` -> ultima version estable.

## Manifiesto

Cada artefacto lleva un `manifest.json` validado contra [manifest.schema.json](manifest.schema.json).

Campos principales:

- `model_id`, `version`, `sha256` del `.rknn`
- Forma de entrada y preprocesado (mean/std, RGB/BGR)
- `rknn_toolkit_version` usada en el export (debe alinearse con runtime en dispositivo)
- `target_platform`: p. ej. `rk3568`

## Entregables

Los binarios `.rknn` no suelen versionarse en Git; versiona el **manifiesto** y sube el binario a releases, artefacto CI o almacenamiento interno.
