# Paso a paso: Sistema de evaluacion de modelos (eval_modelos)

Guia secuencial para implementar y usar el sistema del [README.md](README.md). Cada paso termina con un **checkpoint** antes de pasar al siguiente.

---

## Que necesitas antes de empezar

- Proyecto con OpenCV y MediaPipe (ya tienes `7_ocv_scr_anonymize_v2py` y `utils/image_utils.py` con `bbox_relativo_a_absoluto`).
- Un **video de prueba** (p. ej. unos segundos con caras visibles) para generar el teacher y el student.
- Ejecutar los scripts desde la **raiz del proyecto** (donde esta la carpeta `utils/`).

---

## Paso 0: Definir la muestra (script de especificacion)

**Objetivo:** Decidir que frames se evaluaran segun la duracion o longitud del video, y guardar esa decision en un archivo. Teacher y student leen ese archivo y no hardcodean "cada N frames" ni duracion.

1. **Concepto:** Leer en el README la **seccion 3.2** (Especificacion de la muestra). La idea es tener una **unica fuente de verdad**: un JSON con la lista de indices de frame (y metadata del video) que teacher y student usan.
2. **Estrategias:** El script `gen_sampling_spec_v1.py` acepta **una** de estas opciones:
   - `--every_n_frames N`: un frame cada N (ej. 30 → en 30 FPS = 1 por segundo).
   - `--samples_per_second S`: objetivo de S muestras por segundo (repartidas en el tiempo).
   - `--interval_sec D`: un frame cada D segundos (ej. 1.0).
   - `--max_samples M`: como maximo M frames, repartidos uniformemente en el video (util si un video de 10 s y uno de 60 s deben dar cantidades manejables).
3. **Implementacion:** El script ya existe: `eval_modelos/gen_sampling_spec_v1.py`. Lee el video (duracion, FPS, total frames), aplica la estrategia y escribe un JSON con `video_path`, `duration_sec`, `fps`, `total_frames`, `strategy`, `strategy_params`, `num_samples`, `frames`, `timestamps`.
4. **Ejecucion (ejemplos):**
   ```bash
   # Un frame cada 30 (1/s en 30 FPS)
   python eval_modelos/gen_sampling_spec_v1.py --video ruta/video.mp4 --every_n_frames 30

   # Aprox 1 muestra por segundo
   python eval_modelos/gen_sampling_spec_v1.py --video ruta/video.mp4 --samples_per_second 1

   # Max 60 muestras (videos largos no explotan la cantidad)
   python eval_modelos/gen_sampling_spec_v1.py --video ruta/video.mp4 --max_samples 60
   ```
   Por defecto guarda en `eval_modelos/data/sampling_spec_<nombre_video>.json`.
5. **Verificacion:** Abrir el JSON generado y comprobar que `frames` tiene los indices esperados y que `num_samples` y la duracion tienen sentido.

**Checkpoint:** Tienes un archivo `sampling_spec_*.json` con la lista de frames a evaluar. Teacher y student usaran `--sampling_spec <este_archivo>` en lugar de `--every_n` o valores fijos.

---

## Paso 1: Entender el sistema (sin codigo)

**Objetivo:** Saber para que sirve cada pieza y en que orden se usan.

1. Leer en el README: **Objetivos del sistema** (seccion 1) y **Componentes principales** (seccion 2).
2. Entender el flujo:
   - **Teacher** (PC) recorre el video en ciertos frames y guarda detecciones en un JSON.
   - **Student** lee ese JSON, va a los mismos frames del mismo video y guarda sus detecciones en otro JSON.
   - **Metricas** comparan ambos JSON (matching por IoU) y calculan TP, FP, FN, Precision, Recall, F1.

**Checkpoint:** Puedes explicar con tus palabras: "Para que sirve el teacher" y "Por que teacher y student tienen que usar los mismos frames."

---

## Paso 2: Entender el formato JSON

**Objetivo:** Conocer el contrato de datos que comparten teacher y student.

1. Leer en el README la **seccion 3** (Formato de la matriz por frame).
2. Fijarte en la estructura: lista de objetos con `frame`, `timestamp`, `detecciones`; cada deteccion tiene `bbox` [x, y, w, h], `class`, `confidence`.
3. Entender por que bbox va en **pixeles absolutos**: para poder calcular IoU sin conversiones.

**Checkpoint:** Dado un frame 640x480, sabes que un `bbox: [100, 50, 80, 80]` significa x=100, y=50, ancho=80, alto=80 en pixeles.

---

## Paso 3: Implementar y ejecutar el script Teacher

**Objetivo:** Tener el primer archivo de pseudo-ground-truth.

1. **Concepto:** Repasar en el README la **seccion 4** (Script Teacher): rol, responsabilidades, ventajas.
2. **Implementacion:** Escribir (o revisar si ya existe) `eval_modelos/gen_teacher_gt_v1.py` que:
   - Acepte `--sampling_spec <ruta>.json` (archivo generado en Paso 0) y opcionalmente `--video` (override) y `--out_json`, `--model` (0 o 1).
   - Lea del JSON de sampling: `video_path` (o use `--video` si se pasa) y la lista `frames`.
   - Abra el video, para cada frame en `frames` haga seek, lea el frame, ejecute MediaPipe Face Detection, convierta cada bbox con `bbox_relativo_a_absoluto`, arme la lista con el formato del README (frame, timestamp, detecciones) y guarde JSON.
3. **Ejecucion:**  
   Desde la raiz del proyecto (primero generar el sampling spec en Paso 0):
   ```bash
   python eval_modelos/gen_teacher_gt_v1.py --sampling_spec eval_modelos/data/sampling_spec_video1.json
   ```
   Opcional: `--video otra_ruta.mp4` para sobreescribir la ruta del video; `--out_json` para elegir salida.
4. **Verificacion:** Abrir el JSON generado y comprobar que hay entradas por frame, con `frame`, `timestamp` y `detecciones` (lista de bbox, class, confidence).

**Checkpoint:** Tienes un archivo `teacher_faces_*.json` y puedes decir que frame X tiene N caras y cuales son (aproximadamente) las bbox.

---

## Paso 4: Implementar y ejecutar el script Student

**Objetivo:** Tener las detecciones del student en exactamente los mismos frames que el teacher.

1. **Concepto:** Leer en el README la **seccion 5** (Script Student): rol y responsabilidades.
2. **Implementacion:** Escribir `eval_modelos/run_student_v1.py` que:
   - Acepte `--sampling_spec <ruta>.json` (el mismo del Paso 0 / teacher) y opcionalmente `--video`, `--out_json`, `--model`.
   - Lea del JSON de sampling la lista `frames` y `video_path` (o use `--video` si se pasa).
   - Abra el video, para cada frame en `frames` haga seek, lea el frame, ejecute MediaPipe Face Detection, convierta bbox a [x,y,w,h], arme la misma estructura que el teacher y guarde JSON.
   - (Alternativa: puede seguir leyendo `--teacher_json` para extraer la lista de frames del propio output del teacher; si usas `--sampling_spec`, teacher y student comparten la misma fuente y no dependen del orden de ejecucion.)
3. **Ejecucion:**  
   Desde la raiz:
   ```bash
   python eval_modelos/run_student_v1.py --sampling_spec eval_modelos/data/sampling_spec_video1.json --out_json eval_modelos/data/student_mediapipe_model0_video1.json
   ```
   Opcional: `--video` para sobreescribir la ruta del video.
4. **Verificacion:** Abrir el JSON del student y comparar con el del teacher en 2–3 frames (mismo numero de entradas por frame, mismas keys).

**Checkpoint:** Tienes `student_*.json` y puedes decir: "En el frame X el teacher vio 2 caras y el student vio 1" (o lo que corresponda).

---

## Paso 5: Entender TP, FP, FN, IoU, Precision, Recall, F1

**Objetivo:** Saber que calcula el script de metricas antes de implementarlo.

1. **IoU:** Intersection over Union; umbral tipico 0.5 para "acierto" en deteccion.
2. **TP:** deteccion del student que hace match (IoU >= umbral) con una del teacher.
3. **FP:** deteccion del student sin match en el teacher.
4. **FN:** deteccion del teacher sin match en el student.
5. **Formulas:**  
   Precision = TP / (TP + FP), Recall = TP / (TP + FN), F1 = 2 * P * R / (P + R).

Leer la **seccion 6** del README (objetivo, entrada, proceso v1).

**Checkpoint:** Dado un ejemplo con 2 bbox teacher y 2 bbox student (con un match claro y un FP), puedes indicar TP/FP/FN y calcular Precision y Recall a mano.

---

## Paso 6: Implementar y ejecutar el script de metricas

**Objetivo:** Obtener numeros de calidad (P, R, F1, IoU medio) comparando teacher y student.

1. **Implementacion:** Escribir `eval_modelos/metrics_quality_v1.py` que:
   - Acepte `--teacher_json`, `--student_json`, `--iou_threshold` (default 0.5).
   - Cargue ambos JSON y los indexe por `frame`.
   - Por cada frame comun: matching por IoU (cada bbox student con el teacher de mayor IoU; si IoU >= umbral y el bbox teacher no usado → TP; student sin match → FP; teacher sin match → FN).
   - Acumule TP, FP, FN globales y calcule Precision, Recall, F1 e IoU medio de los TP.
   - Imprima los resultados y opcionalmente guarde un JSON de metricas.
2. **Ejecucion:**  
   ```bash
   python eval_modelos/metrics_quality_v1.py --teacher_json eval_modelos/data/teacher_faces_video1.json --student_json eval_modelos/data/student_mediapipe_model0_video1.json --iou_threshold 0.5
   ```
3. **Interpretacion:** Leer los numeros; probar otro `--iou_threshold` (ej. 0.3 y 0.6) y ver como cambian las metricas.

**Checkpoint:** Puedes explicar que significa "este student tiene Recall alto y Precision baja" (o el caso que te de) en terminos de que detecta de mas o de menos.

---

## Paso 7: Opcional – Script de performance (v2)

**Objetivo:** Medir FPS y latencia del pipeline, separado de la calidad.

1. Leer en el README la **seccion 7** (Script de Performance).
2. Implementar `eval_modelos/metrics_performance_v1.py` que, sobre el mismo video (o un segmento), ejecute el modelo N veces, mida tiempo por frame y calcule FPS y latencia (ms/frame). Opcional: uso de CPU/RAM si es relevante.
3. Ejecutarlo en PC; mas adelante en Raspberry Pi o UNIHIKER para comparar hardware.

**Checkpoint:** Tienes numeros de FPS/latencia para el modelo en un hardware y sabes que primero se valida calidad (pasos 1–6) y despues performance.

---

## Resumen del orden

| Paso | Contenido | Resultado |
|------|-----------|-----------|
| 0 | Definir la muestra (sampling spec) | `sampling_spec_*.json` – que frames evaluar, sin hardcodear |
| 1 | Entender el sistema | Claridad de roles y flujo |
| 2 | Formato JSON | Contrato de datos claro |
| 3 | Script Teacher | `teacher_faces_*.json` (usa `--sampling_spec`) |
| 4 | Script Student | `student_*.json` (usa `--sampling_spec`) |
| 5 | Conceptos TP/FP/FN, P, R, F1, IoU | Saber que calculan las metricas |
| 6 | Script Metricas | Numeros de calidad (P, R, F1, IoU medio) |
| 7 | (Opcional) Script Performance | FPS, latencia por hardware |

Seguir en orden; no pasar al siguiente paso hasta tener el checkpoint del anterior (o hasta que decidas avanzar). Para profundizar en mejoras (varios IoU, mAP, visualizacion, etc.) ver **seccion 9** del README; para herramientas profesionales, **seccion 10**.
