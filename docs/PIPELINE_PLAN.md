# Plan de trabajo — Pipeline biometria RK3568

Documento vivo del pipeline edge: captura, MOG2, FSM, RetinaFace, preprocess cara, MobileFaceNet e identidad.

Referencias internas: `mov_detect/mov_detect.md`, `Retinaface-Models/face_recognition_pipeline_rk3568.md`, `export_models/cara_detectada_pipeline.md`, `export_models/pipeline_v1.md`, `mobilenet_modelos/export_mobilefacenet_rknn.md`.

---

## Estado actual (hecho en PC)

| Componente | Ubicacion |
|------------|-----------|
| Captura USB/RTSP/SNAP | `utils/capture_cameras.py` |
| MOG2 + histéresis umbral | `mov_detect/sensor_mog2.py` |
| FSM MOV_* / FACE_* | `mov_detect/fsm.py` (`run_embedding` en `FACE_PROCESSED`) |
| RetinaFace PC/RK3568 | `inference/retinaface/` |
| Ranking top-N | `inference/retinaface/select_best.py` |
| Preprocess cara 112x112 | `inference/face_crop.py`, `face_preprocess.py`, `face_roll_fix.py`, `face_align.py` |
| MobileFaceNet PC/RK3568 | `inference/mobilefacenet/` + `build_embedder()` |
| Identidad 1:N coseno | `inference/identity/matcher.py` |
| Settings §6.1–6.4 | `configs/settings.py` |
| Orquestacion por ticks | `WIP/main_mov.py` (detect → embed → match → UI) |
| UI opcional (headless ok) | `ui/` (`FrameView.identity`, barra sim/MATCH) |
| Enrolamiento `.npy` | `export_models/face_embedding_from_image.py` (`--preprocess`) |
| Test A/B preprocess | `export_models/ab_preprocess_compare.py` |
| Export MobileFaceNet RKNN | `mobilenet_modelos/` |

---

## Arquitectura (runtime)

```text
Captura -> MOG2 -> FSM -> RetinaFace -> elegir cara (interocular)
    -> prepare_face_patch (112x112 BGR)
    -> MobileFaceNet -> vector L2
    -> FaceGalleryMatcher (coseno vs embeddings/*.npy)
    -> UI / logs [Embed] [ID]
```

### Modos de preprocess (`configs/settings.py` §6.1)

Prioridad si varios flags activos: **ArcFace > roll-fix simple > crop**.

| Modo | Setting | Enrolamiento | Uso |
|------|---------|--------------|-----|
| **Crop** (defecto prod) | ambos flags `false` | `--preprocess crop` | Galeria actual; sim genuina ~0.50–0.63 |
| **Roll-fix hibrido** | `FACE_ROT_ALIGNMENT_SIMPLE_ENABLE=true` | `--preprocess roll_fix` | Endereza roll en crop si `\|roll\| > FACE_ROLL_MAX_DEG`; mejora limitada en pruebas |
| **ArcFace siempre** | `FACE_ALIGNMENT_ENABLE=true` | `--preprocess arcface_align` | Cámara mala posicion; **refs y live deben coincidir** |

Regla critica: **mismo preprocess en enrolamiento y en live**. Mezclar crop vs align degrada el match.

Implementacion align v1: `cv2.estimateAffinePartial2D` + `warpAffine` en CPU (`inference/face_align.py`).

---

## Fase 1 — Preproceso basico

- [x] `inference/face_crop.py` — bbox + margen + resize
- [x] `inference/face_preprocess.py` — `prepare_face_patch`, `FacePatch`
- [x] Settings embed §6.1–6.2 + validacion en `validar_todo()`
- [x] `_tick_embed_if_needed` en `WIP/main_mov.py`

## Fase 2 — Alignment / normalizacion geometrica

- [x] Referencia upstream: `inference/reference/insightface_face_align.py`
- [x] Port: `inference/face_align.py` (OpenCV, sin skimage)
- [x] Roll-fix simple: `inference/face_roll_fix.py` + `FACE_ROT_ALIGNMENT_SIMPLE_ENABLE`
- [x] Integrado en preprocess / embed via `prepare_face_patch_from_settings`
- [x] Test A/B: `export_models/ab_preprocess_compare.py` (crop vs roll_fix vs arcface_align)
- [x] Enrolamiento alineado: `face_embedding_from_image.py --preprocess arcface_align`
- [ ] Mover `_elegir_fila_para_embed` de `main_mov.py` a `select_best.py`
- [ ] Validar pipeline ArcFace en produccion (galeria `embeddings_arcface/` + E2E live)

## Fase 3 — MobileFaceNet embedding

- [x] `inference/mobilefacenet/embedder_pc.py` (ONNX)
- [x] `inference/mobilefacenet/embedder_rk3568.py` (RKNNLite)
- [x] `build_embedder()` en `inference/__init__.py`
- [x] Rutas modelo en `settings.py` §6.3 + validacion
- [x] `_tick_embed_if_needed` completo (cooldown, `EMBED_MIN_SCORE`)

| | PC | RK3568 |
|---|-----|--------|
| Entrada | 112x112 RGB float, mean ImageNet en Python | 112x112 RGB uint8, mean en .rknn |

## Fase 4 — Identidad

- [x] `inference/identity/matcher.py` — coseno 1:N (`np.dot` sobre L2)
- [x] Galeria `.npy` + `EMBED_SIM_MIN_MATCH` + `EMBED_REF_GALLERY_DIR`
- [x] UI: `identity`, `similarity` en `FrameView` / `ui/overlay.py`
- [ ] Calibrar `EMBED_SIM_MIN_MATCH` en datos reales (crop: ~0.45–0.52 recomendado; default actual 0.60 estricto)
- [ ] Script sugerido: calibracion umbral por percentiles (genuinos vs impostores)

## Fase 5 — Calidad (post-MVP)

De `cara_detectada_pipeline.md` (M1–M13). Sin estado `FACE_STABLE`; reglas dentro de `FACE_PROCESSED`.

| Regla | Estado |
|-------|--------|
| M1 score minimo embed | [x] `EMBED_MIN_SCORE` |
| M10 cooldown embed | [x] `EMBED_COOLDOWN_S` |
| M2 tamano bbox minimo | [ ] |
| M7 nitidez / blur | [ ] |
| Gate roll (no embed si `\|roll\|` alto) | [ ] parcial: roll en log, sin rechazo |
| Multi-ref enrolamiento | [ ] manual (varios `.npy` por persona) |

## Fase 6 — Produccion RK3568

- [ ] `requirements-pc.txt` / `requirements-rk3568.txt` (`opencv-python` vs `opencv-python-headless`)
- [ ] SIGINT/SIGTERM en `main_mov.py` (cierre limpio con systemd)
- [ ] E2E headless en placa (`INFERENCE_BACKEND=rk3568`)
- [ ] Latencia por etapa (`eval_modelos/`)

## Fase 7 — RGA (optimizacion)

Solo tras medir en RK3568. Sustituir backends cuando `USE_RGA=true`:

| Operacion | Hoy | RGA |
|-----------|-----|-----|
| resize | MOG2, letterbox | `utils/image_utils.py` (TODO) |
| cvtColor BGR->RGB | RetinaFace RK3568, embed | preprocess |
| warpAffine | roll-fix / align | CPU v1; RGA no soporta afín arbitrario en RK3568 |

MOG2 sigue en CPU OpenCV.

## Fase 8 — Evolucion modelos (backlog)

SCRFD, AdaFace, liveness, FAISS — ver `export_models/pipeline_v1.md`.

---

## Hallazgos experimentales (2026-06)

| Camino | sim vs galeria crop | Notas |
|--------|---------------------|-------|
| crop live | ~0.50–0.63 | Produccion recomendada con refs crop |
| roll_fix live | ~0.45–0.50 (roll ~18°) | Poca mejora vs crop; flag off por defecto |
| arcface live vs refs crop | ~0.14 o negativo | Inutil sin re-enrolar |

---

## Orden de ejecucion (desde hoy)

```text
Calibrar EMBED_SIM_MIN_MATCH
  -> E2E RK3568 (Fase 6)
  -> Calidad M2/M7 (Fase 5)
  -> (opcional) ArcFace + embeddings_arcface si camara inclinada
  -> Medir latencia -> RGA (Fase 7)
  -> Refactor select_best (deuda menor)
```

---

## Settings clave (§6)

| Variable | Default | Rol |
|----------|---------|-----|
| `FACE_ALIGNMENT_ENABLE` | `false` | Align ArcFace siempre |
| `FACE_ROT_ALIGNMENT_SIMPLE_ENABLE` | `false` | Hibrido crop / roll-fix |
| `FACE_ROLL_MAX_DEG` | `10` | Umbral roll-fix |
| `EMBED_MIN_SCORE` | `RETINAFACE_SCORE_DETECCION` | Score min RetinaFace para embed |
| `EMBED_COOLDOWN_S` | `1.0` | Segundos entre embeds |
| `EMBED_SIM_MIN_MATCH` | `0.60` | Umbral coseno MATCH |
| `EMBED_REF_GALLERY_DIR` | `embeddings` | Galeria `.npy` |

Ultima actualizacion: 2026-06-08.
