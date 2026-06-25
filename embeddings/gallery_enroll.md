# Enrolamiento de galeria facial

Referencia detallada: [`README_enrolamiento_galeria.md`](README_enrolamiento_galeria.md).

Entrada recomendada: **`enroll_gallery.py`** (ejecuta los dos pasos en orden).

Scripts subyacentes en `embeddings/`:

1. **`prepare_faces_refs.py`** — prepara recortes en `faces_upd/` (roll, variantes ±7°, crop, flags `ENABLE_*`).
2. **`face_embeddings_npy_from_images_folder.py`** — genera `gallery.npy` + `gallery_meta.json` desde `faces_upd/`.

---

## Estructura de carpetas

```text
embeddings/
  enroll_gallery.py
  prepare_faces_refs.py
  face_embeddings_npy_from_images_folder.py
  faces/                    # entrada prepare (fotos crudas)
  faces_upd/                # salida prepare; entrada del enrolamiento
    or/                     # referencia original (no enrolar)
  gallery.npy
  gallery_meta.json
```

---

## Convencion de nombres (enrolamiento)

```text
{id}_{nombre-con-guiones}[_{zero|der|izq}].{ext}
```

| Sufijo | `rotacion` en JSON |
|--------|-------------------|
| *(ninguno)* o `_zero` | `"cero"` |
| `_der` | `"derecha"` |
| `_izq` | `"izquierda"` |

Referencia original: `faces_upd/or/{stem}.ext` (no va a `faces/` ni a la galeria).

Roll excedido (`|roll| > 25°` en prepare): prefijo `err_` + X roja en el crop; **no copiar** a `faces/` ni enrolar.

El campo **`rotacion`** va siempre en cada entrada del JSON.  
Varias fotos del mismo `id`/`nombre` (cero, der, izq) producen **varias filas** en la galeria.

---

## Flujo prepare → enrolar

```text
faces/  --prepare_faces_refs-->  faces_upd/  (+ or/ referencia)
                                      |
                    face_embeddings_npy_from_images_folder.py
                                      |
                              gallery.npy + gallery_meta.json

Orquestado: python embeddings/enroll_gallery.py
```

### Salida de `prepare_faces_refs` (por foto procesada)

| Archivo | Contenido | Enrolar |
|---------|-----------|---------|
| `or/{stem}.jpg` | Recorte de la imagen original (sin rotar) | No |
| `{stem}_zero.jpg` | Recorte a 0° (centrado) | Si |
| `{stem}_der.jpg` | Recorte rotado -7° desde 0° | Si |
| `{stem}_izq.jpg` | Recorte rotado +7° desde 0° | Si |

Si `|roll| > MAX_ABS_ROLL_DEG` (25°): warning, prefijo `err_` en todos los nombres anteriores y **X roja** en cada recorte (diagonal `(0,0)-(max_x,max_y)` y `(0,max_y)-(max_x,0)`).

Flags opcionales (default `True`): `ENABLE_PROCESS_ROLL_ZERO`, `ENABLE_PROCESS_ROLL_DER`, `ENABLE_PROCESS_ROLL_IZQ`, `ENABLE_SAVE_CROP_ORIGINAL`.

---

## Flujo de enrolamiento (por imagen)

```text
Imagen en faces_upd/
    -> RetinaFace
    -> score >= 0.90, |roll| <= 5°
    -> crop 112x112 (sin align)
    -> MobileFaceNet + L2 normalize
    -> fila en gallery.npy + entry en JSON (con rotacion)
```

---

## Ejemplo `entries[i]`

```json
{
  "id": "0000000001",
  "nombre": "Angel Sanabria",
  "img": "faces/1_Angel-Sanabria_der.jpg",
  "rotacion": "derecha",
  "score": 0.9912,
  "roll_deg": 0.8,
  "used_arcface_align": false,
  "used_roll_fix": false,
  "flip_avg": false
}
```

---

## Ejecucion

```bash
python embeddings/enroll_gallery.py
```

Paso a paso (equivalente):

```bash
python embeddings/prepare_faces_refs.py
python embeddings/face_embeddings_npy_from_images_folder.py
```

Requisito: `INFERENCE_BACKEND=pc` o `rk3568`.

---

## Resumen

Pon fotos crudas en `faces/`, ejecuta `enroll_gallery.py` (o los dos scripts en orden), y usa la galeria matriz en `main_mov` via el matcher. El enrolamiento lee `faces_upd/`; excluye automaticamente `err_*` y `or/`.
