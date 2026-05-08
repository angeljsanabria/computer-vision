# Pipeline cara detectada: diseno y opciones de evaluacion

Documento de trabajo para alinear el flujo de **deteccion de movimiento (MOG2)**, **RetinaFace**, **estabilidad**, **alineacion** y **embedding** en entornos tipo **Rockchip RK3568** (NPU + CPU). Sirve para comparar alternativas y elegir las mas aceptables antes de implementar.

---

## 1. Contexto

- **Entrada:** camara USB (o IP), frame rate acotado en codigo (p. ej. ~2 FPS en prototipo).
- **Deteccion ligera:** MOG2 (`cv2.createBackgroundSubtractorMOG2`) sobre frame reducido para disparar etapas pesadas solo cuando hay actividad.
- **Cara:** RetinaFace mobile 320 (ONNX en PC / RKNN en placa), salidas con caja + landmarks (5 puntos) tras postproceso en `utils/aux_tools_retinaface.py`.
- **Restriccion:** en placa, CPU y memoria limitadas; la NPU acelera modelos convertidos a **RKNN**; operaciones clasicas (warp, metricas) suelen ir en **CPU**.

---

## 2. Pipeline canonico (referencia industria / investigacion)

El orden que suelen seguir sistemas de reconocimiento facial en produccion y SDKs orientados a edge (p. ej. flujos tipo InsightFace / pipelines documentados para detect-align-recognize):

1. **Deteccion:** bounding box + landmarks.
2. **Filtrado de calidad / estabilidad** (opcional pero recomendado en edge): score, tamano, nitidez, pose, consistencia temporal.
3. **Alineacion 2D:** transformacion afine o similar a partir de landmarks → parche fijo (p. ej. **112×112** al estilo ArcFace).
4. **Embedding:** red tipo ArcFace / MobileFace / EdgeFace → vector (p. ej. 512-D), normalizado L2.
5. **Coincidencia:** similitud coseno (o distancia) frente a galeria / umbral.

En **RK3568**, lo habitual es: **deteccion (RKNN si esta el modelo)**, **alineacion y metricas en CPU**, **embedding RKNN si existe conversion**; si no, modelo de reconocimiento **pequeno en CPU/NEON**.

---

## 3. Maquina de estados actual (codigo `deteccion_movimiento_fsm.py`)

| Estado | Significado breve |
|--------|------------------|
| `IDLE` | Sin actividad MOG2 relevante (timeout o inicio). |
| `MOV_DETECTED` | MOG2 supera umbral en el frame. |
| `MOV_OUT` | MOG2 por debajo del umbral en el frame (no confundir con timeout de 10 s). |
| `FACE_PROCESSED` | Ultima inferencia RetinaFace encontro al menos una cara. |
| `FACE_OUT` | Frame sin cara (tras haber estado en fase cara); temporizador de “sin cara” desde ultima deteccion. |
| `FACE_PROCESSED_TIMEOUT` | Paso intermedio antes de `IDLE` por timeout de cara. |

**Timeouts:**

- **Sin movimiento MOG2** (`timeout_seg`): solo fuerza `IDLE` desde `MOV_DETECTED` / `MOV_OUT` (no desde `FACE_*`, para no expulsar por “fondo quieto” con persona quieta).
- **Sin cara** (`timeout_seg`) desde `t_ultima_cara` en fase `FACE_OUT`.

---

## 4. Propuesta: estado `FACE_STABLE` (o `EMBED_READY`)

**Problema que resuelve:** no lanzar **embedding** (ni alineacion costosa repetida) en cada frame con `FACE_PROCESSED`, sino cuando la escena y la cara cumplen criterios de “lista para identidad”.

**Definicion posible (borrador):** existe cara con score suficiente **y** se cumplen reglas de estabilidad/calidad (seccion 5) durante una ventana corta **o** un unico frame ganador.

**Alternativa sin estado explicito:** mismas reglas evaluadas en cada frame; si pasan → alinear + embed + marcar cooldown. Es mas compacto pero peor para logs y pruebas.

**Recomendacion de documentacion:** mantener un estado **`FACE_STABLE`** (o **`EMBED_READY`**) como **puerta logica** aunque internamente sea un subconjunto de condiciones sobre `FACE_PROCESSED`.

---

## 5. Ideas de mejora (lista para evaluar)

Cada fila es una **palanca** independente; se pueden combinar.

| ID | Idea | Objetivo | Coste aprox. en RK3568 |
|----|------|----------|-------------------------|
| M1 | Umbral minimo de **score** de cara (post-NMS) | Menos falsos antes de embed | Muy bajo |
| M2 | **Tamano minimo** de caja en pixeles | Ignorar caras lejanas | Muy bajo |
| M3 | **N frames consecutivos** con cara (o con score > S) | Estabilidad temporal | Bajo (contadores) |
| M4 | **IoU** entre cajas consecutivas > umbral | Caja estable (reduce jitter) | Bajo |
| M5 | Baja **varianza** de posiciones de landmarks entre frames | Cara “quieta” respecto al sensor | Medio (aritmetica) |
| M6 | **NOT_MOV** (MOG2) + cara presente | “Escena estabilizada” sin gastar embed en cambio de fondo | Ya disponible en logs MOG2 |
| M7 | **Nitidez** (p. ej. varianza Laplaciana en crop) | Descartar frames borrosos | Bajo/medio (crop + Laplaciano) |
| M8 | Filtro de **pose** aproximada (landmarks → yaw/pitch aprox.) | Solo frontales o rangos admisibles | Medio (geometria) |
| M9 | **Alineacion** a parche 112×112 (o 112×96) con matriz afine desde 5 puntos | Entrada estandar al modelo de embedding | Medio (warp OpenCV) |
| M10 | **Un embed cada T ms** o una vez por “sesion” estable | Limitar carga y evitar spam | Bajo |
| M11 | **Mejor frame** en ventana (p. ej. max nitidez o max score) | Mejor vector que el primero estable | Memoria corta de frames |
| M12 | Embedding en **RKNN** (modelo convertido + calibracion INT8) | Latencia y consumo | Esfuerzo toolchain + validacion |
| M13 | Galeria pequena + **cosine** en CPU | Identificacion 1:N ligera | Bajo para N pequeno |

---

## 6. Heuristica “MOG2 quieto + cara” (discusion)

- **Ventaja:** muy barata; encaja con “no embedar mientras el encuadre cambia mucho”.
- **Limite:** MOG2 quieto **no** implica cara nitida, ni caja estable, ni buena pose; es complementaria a M3–M9.

Uso recomendado: **M6 como condicion necesaria pero no suficiente**, junto con M1–M5 o M7–M9 segun presupuesto CPU.

---

## 7. Criterios para elegir “opciones mas aceptables”

Evaluar cada combinacion candidata con:

1. **Precision percibida:** menos embeddings basura y menos identidades equivocas.
2. **Latencia end-to-end** en RK3568 (ms por frame o por evento “embedding”).
3. **Complejidad de implementacion** (tiempo de desarrollo y riesgo de bugs).
4. **Mantenimiento:** parametros tunables (umbrales) sin recompilar.
5. **Alineacion con toolchain actual:** RetinaFace ya da landmarks → M9 es natural.

**Sugerencia de orden de pilotaje:**

1. M1 + M2 + M9 + M10 (pipeline minimo serio: score, tamano, alinear, rate-limit).
2. Anadir M3 o M4 si hay jitter de caja.
3. Anadir M6 si quieres acoplar a MOG2 sin mas CPU.
4. Anadir M7 si hay problemas de foco / movimiento suave.
5. M12 cuando el modelo de embedding y RKNN esten cerrados.

---

## 8. Relacion con `FACE_STABLE`

Transicion sugerida (conceptual):

```text
FACE_PROCESSED / FACE_OUT
    -> (cuando se cumplan {M1,M2,...} y opcionalmente M6)
    -> FACE_STABLE
    -> alineacion + embedding (+ cooldown)
    -> vuelta a FACE_PROCESSED o permanencia segun producto
```

Si **no** se introduce estado nuevo, equivalente: flag interno `embed_ready` o contador hasta disparo unico.

---

## 9. Proximos pasos de evaluacion (checklist)

- [ ] Fijar metricas: latencia objetivo, tasa de aceptacion de frames, FRR/FAR deseados (aunque sea informal en entorno real).
- [ ] Elegir modelo de **embedding** exportable a ONNX y reproducible en RKNN.
- [ ] Probar en PC: RetinaFace + warp + embedding ONNX (OnnxRuntime).
- [ ] Convertir embedding a RKNN, medir en RK3568.
- [ ] Tunear umbrales M1–M7 con dataset corto de videos de la camara real.

---

## 10. Referencias de apoyo (lectura)

- Pipelines tipo **detect → align → recognize** y modulos documentados en proyectos de reconocimiento facial open source (p. ej. documentacion de pipelines y reconocimiento en proyectos estilo Uniface / InsightFace ecosystem).
- SDKs que documentan despliegue en **NPU Rockchip** para pipeline completo (referencia orientativa: lineas de producto que citan RK3588/RKNN; extrapolar expectativas de latencia a RK3568 con margen).

---

*Ultima revision: documento vivo; actualizar cuando se cierren decisiones de implementacion.*
