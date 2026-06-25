# Enrolamiento de galeria facial

Scripts en `embeddings/`:

| Script | Rol |
|--------|-----|
| `enroll_gallery.py` | **Entrada unica:** ejecuta prepare y luego enrolamiento (en ese orden) |
| `prepare_faces_refs.py` | Prepara fotos: corrige roll, genera variantes y recortes → `faces_upd/` |
| `face_embeddings_npy_from_images_folder.py` | Enrola embeddings desde `faces_upd/` → `gallery.npy` + `gallery_meta.json` |

El reconocimiento en vivo lo hace `WIP/main_mov.py` via `inference/identity/matcher.py`.

---

## Flujo recomendado

```text
1. Fotos crudas       -> embeddings/faces/
2. enroll_gallery.py  -> prepare (faces/ -> faces_upd/) + enrolar (faces_upd/ -> gallery.npy)
```

Equivalente manual (mismo orden que `enroll_gallery.py`):

```text
1. Fotos crudas          -> embeddings/faces/
2. prepare_faces_refs    -> embeddings/faces_upd/  (3 recortes enrolables + or/ referencia)
3. face_embeddings_...   -> gallery.npy + gallery_meta.json  (lee faces_upd/ directamente)
```

Los scripts individuales sirven para depurar un solo paso. No hace falta copiar recortes a `faces/` si usas el flujo completo: el enrolamiento lee `faces_upd/`.

---

## Paso 1: `prepare_faces_refs.py`

Entrada fija: `embeddings/faces/`. Salida fija: `embeddings/faces_upd/`.  
Ambas carpetas deben existir antes de ejecutar.

Por cada imagen en `faces/`:

1. RetinaFace detecta la mejor cara y mide roll (linea entre ojos).
2. Si `|roll| > 25°` (`MAX_ABS_ROLL_DEG`): **warning**, se procesa igual; los recortes llevan prefijo `err_`, marca **X roja** (diagonales punta a punta) y **no deben enrolarse**.
3. Si `|roll| > 3°` (`MAX_TOLERANCE_ABS_ROLL_DEG`): rota el frame completo a ~0°.
4. Si `|roll| ≤ 3°`: usa el frame sin corregir.
5. Segun flags `ENABLE_*`, genera variantes rotadas; en cada una vuelve a detectar y **recorta bbox** (margen `FACE_CROP_MARGIN_FRAC`). Opcionalmente guarda crop de la imagen original sin rotar en `faces_upd/or/`.
6. Guarda en `faces_upd/`:

| Archivo salida | Contenido | Enrolar |
|----------------|-----------|---------|
| `or/{id}_{nombre}.jpg` | Crop de la imagen original (sin corregir roll) | No |
| `{id}_{nombre}_zero.jpg` | Crop del frame a 0° (centrado) | Si |
| `{id}_{nombre}_der.jpg` | Crop rotado -7° desde 0° | Si |
| `{id}_{nombre}_izq.jpg` | Crop rotado +7° desde 0° | Si |

Si `|roll| > 25°`, los nombres anteriores pasan a `err_{...}` (p. ej. `err_1_Angel-Sanabria_zero.jpg`, `or/err_1_Angel-Sanabria.jpg`).

```bash
python embeddings/prepare_faces_refs.py
```

Constantes (solo en el script):

| Constante | Valor | Efecto |
|-----------|-------|--------|
| `MAX_ABS_ROLL_DEG` | 25.0 | Warning + prefijo `err_` + X roja (no omite) |
| `MAX_TOLERANCE_ABS_ROLL_DEG` | 3.0 | Corregir roll del frame si se supera |
| `APPLY_ROT_ABS_ROLL_DEG` | 7.0 | Rotacion augment `_der` / `_izq` |
| `MIN_RETINAFACE_SCORE` | 0.90 | Score minimo RetinaFace |
| `ENABLE_PROCESS_ROLL_ZERO` | `True` | Guardar `{stem}_zero` |
| `ENABLE_PROCESS_ROLL_DER` | `True` | Guardar `{stem}_der` |
| `ENABLE_PROCESS_ROLL_IZQ` | `True` | Guardar `{stem}_izq` |
| `ENABLE_SAVE_CROP_ORIGINAL` | `True` | Guardar crop original en `or/` |

Al menos un `ENABLE_*` debe estar en `True`. La correccion de roll a 0° solo corre si alguna variante `_zero`/`_der`/`_izq` esta habilitada.

Usa `build_face_detector()` (`INFERENCE_BACKEND=pc` o `rk3568`).

---

## Paso 2: `face_embeddings_npy_from_images_folder.py`

### Que hace (resumen)

1. Lee imagenes validas en `embeddings/faces_upd/`.
2. Por cada foto: RetinaFace, valida score y roll.
3. Crop 112x112 (sin align ni roll-fix) → vector 128-D L2-normalizado.
4. Escribe `gallery.npy` y `gallery_meta.json`.

No modifica las fotos de entrada. Si `|roll| > 5°` al enrolar, omite la foto (warning).

---

## Entrada: carpetas `faces/` y `faces_upd/`

```text
embeddings/
  enroll_gallery.py
  prepare_faces_refs.py
  face_embeddings_npy_from_images_folder.py
  faces/                            # entrada de prepare (fotos crudas)
    1_Angel-Sanabria_zero.jpg         # rotacion: cero
    1_Angel-Sanabria_der.jpg          # rotacion: derecha
    1_Angel-Sanabria_izq.jpg          # rotacion: izquierda
  faces_upd/                          # salida prepare (enrolables en raiz)
    1_Angel-Sanabria_zero.jpg
    err_2_Juan-Perez_zero.jpg         # roll > 25 deg: no se enrola
    or/                               # referencia original (no se enrola)
  gallery.npy
  gallery_meta.json
```

### Nombre de archivo (obligatorio)

Patron:

```text
{id}_{nombre-con-guiones}[_{zero|der|izq}].{ext}
```

| Parte | Regla |
|-------|--------|
| `id` | Solo digitos, hasta el primer `_` |
| `nombre` | Texto entre `_` y el sufijo opcional; `-` → espacio en JSON |
| Sufijo | Opcional: `_zero`, `_der`, `_izq` |
| Prefijo | `err_*` en `faces_upd/` no se enrolan (roll excedido en prepare) |
| `ext` | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` |

El `id` en JSON: `str(numero).zfill(10)` (maximo 10 digitos en el archivo).

### Campo `rotacion` en JSON

Siempre presente en cada `entries[i]`:

| Sufijo en archivo | `rotacion` |
|-------------------|------------|
| *(ninguno)* | `"cero"` |
| `_zero` | `"cero"` |
| `_der` | `"derecha"` |
| `_izq` | `"izquierda"` |

Las referencias originales viven en `faces_upd/or/` (`{id}_{nombre}.jpg` o `err_{id}_{nombre}.jpg`) y **no se enrolan**.

Los archivos `err_*` (roll original > 25° en prepare) tampoco se enrolan: el script de enrolamiento los descarta por prefijo.

Una misma persona (`id` + `nombre`) puede tener **varias filas** en la galeria (p. ej. cero + der + izq).

Ejemplos:

| Archivo | `id` | `nombre` | `rotacion` |
|---------|------|----------|------------|
| `1_Angel-Sanabria.jpg` | `0000000001` | `Angel Sanabria` | `cero` |
| `1_Angel-Sanabria_der.jpg` | `0000000001` | `Angel Sanabria` | `derecha` |

Relacion con `prepare_faces_refs`: el enrolamiento lee la raiz de `faces_upd/` (solo `*_zero`, `*_der`, `*_izq` validos); ignora `or/` y archivos `err_*`.

---

## Filtros de calidad (enrolamiento)

| Constante | Valor | Efecto |
|-----------|-------|--------|
| `MIN_RETINAFACE_SCORE` | 0.90 | Score minimo de la mejor cara |
| `MAX_ABS_ROLL_DEG` | 5.0 | Si \|roll\| ojos > 5°, foto no procesable |

Roll: linea entre ojos (RetinaFace). Solo aviso y skip; no endereza en enrolamiento.

---

## Preprocess del embedding

Siempre **crop** bbox + margen (`prepare_face_patch`, align off).  
Inferencia: `build_face_detector()` + `build_embedder()` segun `INFERENCE_BACKEND`.

Opcional: `--flip-avg`.

### Normalizacion L2

Cada fila de `gallery.npy` se guarda L2-normalizada en enrolamiento. El matcher no re-normaliza (`gallery @ live`).

Cabecera JSON: `"data_normalizada": 1`. Si falta o ≠ 1, el matcher falla al arrancar.

---

## Salida: `gallery_meta.json`

```json
{
  "version": 1,
  "embed_dim": 128,
  "preprocess": "crop",
  "data_normalizada": 1,
  "generated_at": "2026-06-08T19:45:32+00:00",
  "entries": [
    {
      "id": "0000000001",
      "nombre": "Angel Sanabria",
      "img": "faces/1_Angel-Sanabria_izq.jpg",
      "rotacion": "izquierda",
      "score": 0.9965,
      "roll_deg": 1.2,
      "used_arcface_align": false,
      "used_roll_fix": false,
      "flip_avg": false
    }
  ],
  "detecciones": [
    {
      "count": 0,
      "last_seen": null,
      "autorizaciones_validas": 0,
      "autorizaciones_denegadas": 0
    }
  ]
}
```

- Sin campo `idx`: fila `i` del array = fila `i` de `gallery.npy`.
- `detecciones[i]`: placeholders; runtime los actualizara mas adelante.

---

## Uso

```bash
# Flujo completo (recomendado)
python embeddings/enroll_gallery.py
```

Paso a paso (equivalente al comando anterior):

```bash
# 1. Preparar recortes (faces/ -> faces_upd/)
python embeddings/prepare_faces_refs.py

# 2. Enrolar (faces_upd/ -> gallery.npy + JSON)
python embeddings/face_embeddings_npy_from_images_folder.py
```

---

## Matcher y `main_mov.py`

| Prioridad | Archivos | Comportamiento |
|-----------|----------|----------------|
| 1 | `gallery.npy` + `gallery_meta.json` | `data_normalizada: 1`, match `gallery @ live`, label = `entries[i].id` |
| 2 (legacy) | `.npy` sueltos | Solo si no hay par matriz/JSON |

Pendiente: UI con `nombre`, `rotacion`, `detecciones[i]`; borrar `.npy` legacy.

---

## Referencias

| Componente | Ruta |
|------------|------|
| Pipeline completo | `embeddings/enroll_gallery.py` |
| Preparar fotos | `embeddings/prepare_faces_refs.py` |
| Enrolamiento batch | `embeddings/face_embeddings_npy_from_images_folder.py` |
| Enrolamiento 1 foto (legacy) | `export_models/face_embedding_from_image.py` |
| Pipeline live | `WIP/main_mov.py` |
| Matcher | `inference/identity/matcher.py` |
| Preprocess crop | `inference/face_preprocess.py` |

---

## Estado de implementacion

| Item | Estado |
|------|--------|
| `enroll_gallery.py` | Hecho |
| `prepare_faces_refs.py` | Hecho |
| `faces/` + parsing id/nombre/rotacion | Hecho |
| Campo `rotacion` en JSON | Hecho |
| `gallery.npy` + `gallery_meta.json` | Hecho |
| `data_normalizada: 1` | Hecho |
| Matcher matriz + JSON | Hecho |
| UI / detecciones en runtime | Pendiente |
| Borrar `.npy` sueltos legacy | Pendiente |
