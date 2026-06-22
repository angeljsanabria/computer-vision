# Plan de trabajo — Pipeline biometria RK3568

Documento vivo del pipeline edge: captura, MOG2, FSM, RetinaFace, alignment, MobileFaceNet e identidad.

Referencias internas: `mov_detect/mov_detect.md`, `Retinaface-Models/face_recognition_pipeline_rk3568.md`, `export_models/cara_detectada_pipeline.md`, `export_models/pipeline_v1.md`, `mobilenet_modelos/export_mobilefacenet_rknn.md`.

---

## Estado actual (hecho)

| Componente | Ubicacion |
|------------|-----------|
| Captura USB/RTSP/SNAP | `utils/capture_cameras.py` |
| MOG2 + histéresis umbral | `mov_detect/sensor_mog2.py` |
| FSM MOV_* / FACE_* | `mov_detect/fsm.py` (`run_embedding` en `FACE_PROCESSED`) |
| RetinaFace PC/RK3568 | `inference/retinaface/` |
| Ranking top-N | `inference/retinaface/select_best.py` |
| Orquestacion por ticks | `WIP/main_mov.py` |
| UI opcional (sin cv2 si headless) | `ui/` |
| Export MobileFaceNet RKNN | `mobilenet_modelos/` |

---

## Arquitectura objetivo

```text
Captura -> MOG2 -> FSM -> RetinaFace -> mejores_caras
    -> Face Alignment (112x112) -> MobileFaceNet -> coseno vs galeria
```

---

## Fase 1 — Preproceso basico (integracion, pendiente)

- [ ] `inference/face_preprocess.py` — crop bbox + margen (o delegar en align)
- [ ] Settings embed + `_tick_embed_if_needed` en main

## Fase 2 — Face Alignment

- [x] Referencia upstream: `inference/reference/insightface_face_align.py`
- [x] Port: `inference/face_align.py` (OpenCV, sin skimage)
- [ ] Integrar align en preprocess / embed (sin main hasta que se pida)
- [ ] Test A/B: crop vs align (similitud)

Implementacion v1: `cv2.estimateAffinePartial2D` + `warpAffine` en CPU (suficiente RK3568, 1 cara).
## Fase 3 — MobileFaceNet embedding

- [ ] `inference/mobilefacenet/embedder_pc.py` (ONNX)
- [ ] `inference/mobilefacenet/embedder_rk3568.py` (RKNNLite)
- [ ] `build_embedder()` en `inference/__init__.py`
- [ ] Rutas modelo en `settings.py` + validacion
- [ ] Completar `_tick_embed_if_needed`

| | PC | RK3568 |
|---|-----|--------|
| Entrada | 112x112 RGB float, mean ImageNet en Python | 112x112 RGB uint8, mean en .rknn |

## Fase 4 — Identidad

- [ ] `inference/identity/matcher.py` — coseno 1:1 / 1:N
- [ ] Galeria `.npy` + `EMBED_SIM_MIN_MATCH`
- [ ] UI: `identity`, `similarity` en `FrameView`

## Fase 5 — Calidad (post-MVP)

De `cara_detectada_pipeline.md` (M1–M13). Prioridad: M1 score embed, M10 cooldown, M2 tamano bbox, M7 nitidez.

Sin estado `FACE_STABLE`; reglas dentro de `FACE_PROCESSED`.

## Fase 6 — Produccion RK3568

- [ ] `requirements-pc.txt` / `requirements-rk3568.txt` (`opencv-python` vs `opencv-python-headless`)
- [ ] SIGINT/SIGTERM en `main_mov.py`
- [ ] E2E headless en placa
- [ ] Latencia por etapa (`eval_modelos/`)

## Fase 7 — RGA (optimizacion)

Solo tras medir. Sustituir backends cuando `USE_RGA=true`:

| Operacion | Hoy | RGA |
|-----------|-----|-----|
| resize | MOG2, letterbox | `utils/image_utils.py` |
| cvtColor BGR->RGB | RetinaFace RK3568, embed | preprocess |
| warpAffine | alignment | CPU v1; RGA opcional |

MOG2 sigue en CPU OpenCV.

## Fase 8 — Evolucion modelos (backlog)

SCRFD, AdaFace, liveness, FAISS — ver `export_models/pipeline_v1.md`.

---

## Orden de ejecucion

```text
Fase 1 (crop + stub) -> Fase 2 (align) -> Fase 3-4 (embed + match) -> Fase 6 (prod) -> Fase 7 (RGA)
```

Ultima actualizacion: 2026-06.
