## Sistema de Evaluacion de Modelos y Pipelines

Este documento define **como vamos a evaluar modelos y pipelines de vision por computadora**
en el proyecto, de forma reproducible y profesional.

La idea es separar claramente:
- **Calidad del modelo/pipeline** (TP, FP, FN, Precision, Recall, F1, IoU, mAP)
- **Performance** (FPS, latencia, uso de CPU/RAM) en cada hardware

Primero construiremos una **version v1 a mano** para entender bien las metricas.  
Luego podremos migrar a herramientas usadas por profesionales (COCO, pycocotools, etc.).

---

## 1. Objetivos del Sistema de Evaluacion

1. **Comparar modelos** (teacher vs students) con la **misma entrada**.
2. **Comparar hardware** (PC, Raspberry Pi, UNIHIKER, ESP32-P4) con el mismo modelo.
3. Separar:
   - Calidad: que tan bien ve el modelo
   - Performance: que tan rapido y barato corre
4. Poder repetir las pruebas en el futuro con las mismas reglas.

---

## 2. Componentes Principales (v1)

En la version inicial trabajamos con **tres scripts principales**:

1. **Script Teacher** (`gen_teacher_gt_v1.py`)
   - Corre **solo en PC** (modelo pesado / mas preciso).
   - Recorre un video (o imagenes) y genera una **matriz por frame** con las detecciones del modelo teacher.
   - El resultado se usa como **pseudo-ground-truth** para comparar otros modelos.

2. **Script Student Runner** (`run_student_v1.py`)
   - Corre en PC, Raspberry Pi, UNIHIKER (y conceptualmente ESP32-P4).
   - Usa la misma fuente (video/imagenes) y **solo evalua los frames definidos por el teacher**.
   - Genera otra matriz por frame con las detecciones del student.

3. **Script de Metricas de Calidad** (`metrics_quality_v1.py`)
   - Corre en PC (analisis offline).
   - Lee la matriz del teacher y la del student.
   - Hace **matching por IoU** entre bounding boxes de teacher y student.
   - Calcula:
     - TP, FP, FN
     - Precision, Recall, F1
     - IoU medio
   - Opcional mas adelante: mAP.

En **v2** se agrega un cuarto componente separado:

4. **Script de Performance** (`metrics_performance_v1.py`)
   - Mide FPS, latencia, uso de CPU/RAM.
   - No mira calidad de deteccion, solo coste de ejecucion.
   - Se puede correr en cada hardware (PC, Raspberry, UNIHIKER, etc.).

---

## 3. Formato de la Matriz por Frame (Teacher y Student)

Tanto el teacher como el student producen archivos JSON con una **lista de frames evaluados**.

Ejemplo de estructura (simplificada):

```json
[
  {
    "frame": 120,
    "timestamp": 4.0,
    "detecciones": [
      {
        "bbox": [100, 50, 80, 80],
        "class": "face",
        "confidence": 0.97
      },
      {
        "bbox": [300, 100, 90, 90],
        "class": "face",
        "confidence": 0.88
      }
    ]
  },
  {
    "frame": 150,
    "timestamp": 5.0,
    "detecciones": []
  }
]
```

Donde:
- **frame**: numero de frame en el video original.
- **timestamp**: tiempo en segundos desde el inicio del video.
- **detecciones**: lista de objetos detectados en ese frame.
  - **bbox**: bounding box en coordenadas absolutas (x, y, w, h) en pixeles.
  - **class**: clase detectada (en nuestro caso, normalmente `"face"`).
  - **confidence**: confianza del modelo (0.0 a 1.0).

### 3.1. Notas importantes

- El **formato es el mismo para teacher y student**, lo que cambia es el origen de las detecciones.
- El script de metricas (`metrics_quality_v1.py`) asumira este formato para poder comparar.
- El uso de coordenadas **absolutas en pixeles** simplifica el calculo de IoU.

### 3.2. Especificacion de la muestra (sampling spec) – Paso 0

Para no hardcodear "cada cuantos frames" o "cuantos segundos" en teacher/student, un **script previo** genera un archivo JSON que define **que frames** se evaluaran. Teacher y student leen ese archivo y procesan solo esos frames.

**Estrategias posibles (se elige una):**

| Estrategia | Parametro | Ejemplo | Uso |
|------------|-----------|---------|-----|
| Cada N frames | `every_n_frames` | 30 | En video 30 FPS → 1 frame por segundo |
| Muestras por segundo | `samples_per_second` | 1 | Aprox 1 frame por segundo (independiente del FPS) |
| Intervalo en segundos | `interval_sec` | 1.0 | Un frame cada 1 s |
| Maximo de muestras | `max_samples` | 60 | Como mucho 60 frames, repartidos uniformemente en el video |

El script lee el video (duracion, FPS, total de frames), aplica la estrategia y escribe un JSON con la lista de indices de frame (y opcionalmente timestamps). Ese JSON es la **unica fuente de verdad** para "que frames evaluar"; teacher y student reciben `--sampling_spec <ruta>` y usan la lista de frames del archivo.

**Ejemplo de formato del archivo de sampling spec:**

```json
{
  "video_path": "ruta/al/video.mp4",
  "duration_sec": 10.5,
  "fps": 30,
  "total_frames": 315,
  "strategy": "every_n_frames",
  "strategy_params": { "every_n_frames": 30 },
  "num_samples": 11,
  "frames": [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300],
  "timestamps": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
}
```

- **Teacher** y **Student** reciben `--sampling_spec <archivo>.json` (y opcionalmente `--video` si se quiere sobreescribir la ruta del video). Leen `frames` y opcionalmente `video_path` para saber que video abrir y en que frames hacer seek.

---

## 4. Script Teacher (`gen_teacher_gt_v1.py`)

### 4.1. Rol del teacher

El teacher es un **modelo mas preciso y pesado**, ejecutado en PC, que usamos como referencia:
- Puede ser YOLO grande, MediaPipe con configuracion mas sensible, Detectron2, etc.
- No es "ground truth humano perfecto", pero nos sirve como **pseudo-ground-truth**.

### 4.2. Responsabilidades del script

1. Leer un video (o una carpeta de imagenes).
2. Definir una **estrategia de muestreo de frames**, por ejemplo:
   - Cada N frames (ej: 30 → 1 frame por segundo en video de 30 FPS).
   - O N frames aleatorios por video.
3. Para cada frame seleccionado:
   - Ejecutar el modelo teacher.
   - Convertir bounding boxes del modelo (por ejemplo, relativos) a **absolutos** usando funciones como `bbox_relativo_a_absoluto`.
   - Guardar la entrada en la matriz JSON con el formato definido.
4. Guardar el archivo, por ejemplo:
   - `eval_modelos/data/teacher_faces_video1.json`.

### 4.3. Ventajas de este enfoque

- El teacher se ejecuta **solo una vez por dataset**.
- No necesitamos re-ejecutarlo para cada student.
- Todos los students se comparan contra el **mismo conjunto de frames**.

---

## 5. Script Student Runner (`run_student_v1.py`)

### 5.1. Rol del student

El student es el modelo que queremos evaluar:
- Puede ser el mismo tipo de modelo con otra configuracion (por ej. MediaPipe `model=0` vs `model=1`).
- Puede ser un modelo distinto (YOLO tiny, etc.).
- Puede correr en otro hardware (Raspberry, UNIHIKER).

### 5.2. Responsabilidades del script

1. Leer el archivo JSON del teacher:
   - `teacher_faces_video1.json`.
   - Extraer la lista de `frame` o `timestamp` a evaluar.
2. Reproducir el mismo video (o imagenes):
   - Moverse a los frames indicados.
   - Ejecutar el student SOLO en esos frames.
3. Para cada frame evaluado:
   - Obtener las detecciones del student.
   - Convertir bounding boxes a formato **[x, y, w, h] en pixeles**.
   - Guardar la entrada en un JSON similar, por ejemplo:
     - `student_mediapipe_model0_video1.json`.

De esta forma, la matriz del student es **estructuralmente identica** a la del teacher.

---

## 6. Script de Metricas de Calidad (`metrics_quality_v1.py`)

### 6.1. Objetivo

Calcular que tan bien el **student** imita al **teacher** (o ground truth) en terminos de deteccion:
- TP (True Positives)
- FP (False Positives)
- FN (False Negatives)
- Precision, Recall, F1
- IoU medio

### 6.2. Entrada del script (v1)

- `--teacher_json`: ruta al JSON del teacher (ej: `teacher_faces_video1.json`).
- `--student_json`: ruta al JSON del student (ej: `student_mediapipe_model0_video1.json`).
- `--iou_threshold`: umbral de IoU para considerar una deteccion correcta (ej: 0.5).

### 6.3. Proceso (v1, a mano)

1. Leer ambos JSON y construir diccionarios por `frame`:
   - `teacher[frame] -> lista de bbox`.
   - `student[frame] -> lista de bbox`.
2. Para cada `frame` comun entre ambos:
   - Para cada bbox del student, buscar el bbox del teacher con **mayor IoU**.
   - Si `IoU >= iou_threshold` y ese bbox del teacher no fue usado aun → **TP**.
   - Si no hay match suficiente → **FP**.
   - Cualquier bbox del teacher sin match al final → **FN**.
3. Al final, acumular TP, FP, FN en todos los frames y calcular:
   - `Precision = TP / (TP + FP)` (si el denominador > 0).
   - `Recall = TP / (TP + FN)` (si el denominador > 0).
   - `F1 = 2 * (Precision * Recall) / (Precision + Recall)` (si el denominador > 0).
   - IoU medio: promedio de todos los IoU de los matches TP.
4. Mostrar los resultados y, opcionalmente, guardarlos en un JSON de metricas.

### 6.4. Futuro: uso de librerias profesionales

Una vez entendida la v1 a mano:
- Mapear nuestras matrices a **formato COCO**.
- Usar **pycocotools** para calcular mAP y otras metricas estandar.
- Integrar con herramientas como **FiftyOne** para visualizacion.

---

## 7. Script de Performance (`metrics_performance_v1.py`) – v2

Este script mide **performance**, separado de la calidad:

- FPS (frames por segundo) del pipeline completo.
- Latencia por frame (ms/frame).
- Uso de CPU y memoria (si es relevante).
- Comparacion entre hardware:
  - PC
  - Raspberry Pi Zero 2W
  - UNIHIKER M10
  - ESP32-P4 (conceptualmente, con modelos tiny).

Orden sugerido:
- Primero usar `metrics_quality_v1.py` para asegurarnos de que el modelo ve bien.
- Despues usar `metrics_performance_v1.py` para ver donde y que tan rapido podemos correrlo.

---

## 8. Resumen del nucleo (v1)

Este sistema de evaluacion busca:
- Ser **modular** (teacher, student, metricas, performance separados).
- Ser **reproducible** (mismas matrices, mismos frames para todos los modelos).
- Ser **educativo**: primero v1 a mano, luego migrar a librerias profesionales.

Proximo paso sugerido:
- Definir en codigo (paso a paso) el script `gen_teacher_gt_v1.py` usando
  el modelo de MediaPipe que ya tienes (`7_ocv_scr_anonymize_v2py`) y generar
  el primer `teacher_faces_*.json` de prueba.

---

## 9. Mejoras y extensiones (enfoque profesional / v2)

Las siguientes mejoras son las que un ingeniero en vision por computadora suele agregar sobre el nucleo v1. Se pueden implementar por fases.

### 9.1. Metricas de calidad

- **Varios umbrales de IoU (ej. 0.5 y 0.75)**  
  Reportar Precision, Recall y F1 para mas de un `iou_threshold` (p. ej. 0.5 y 0.75). Un solo umbral puede ocultar que el modelo localiza mal pero “pasa” en 0.5. Permite ver sensibilidad a la precision de la caja.

- **Umbral de confianza configurable en metricas**  
  En el script de metricas, considerar como deteccion solo si `confidence >= X` (parametro). Permite hacer curvas Precision–Recall o analizar el trade-off confianza vs calidad sin re-ejecutar teacher/student.

- **Metricas por clase**  
  Si hay varias clases (face, person, etc.): Precision, Recall y F1 **por clase**, y luego promedio (macro) o ponderado por frecuencia. Necesario cuando se comparan modelos multi-clase.

- **mAP (mean Average Precision)**  
  Metrica estandar en deteccion de objetos. Tras entender P/R/F1 a mano, anadir calculo de mAP (p. ej. con **pycocotools** o equivalente) para alinearse con benchmarks (COCO, etc.).

### 9.2. Analisis de errores

- **Desglose por frame**  
  No solo metricas globales: identificar frames con muchos FN, muchos FP o ambos (p. ej. tabla o CSV: frame_id, n_teacher, n_student, n_TP, n_FP, n_FN).

- **Desglose por tamano de objeto**  
  Clasificar detecciones por area (chico / mediano / grande) y reportar metricas por tamano. Los modelos suelen fallar mas en objetos pequenos.

- **Export para inspeccion**  
  Opcion de exportar una lista de “peores frames” (ej. mas FN o mas FP) para revisar a mano o con visualizacion.

### 9.3. Reproducibilidad

- **Archivo de configuracion (YAML o JSON)**  
  Concentrar en un solo archivo: rutas de video, JSON teacher/student, `every_n`, `iou_threshold`, `min_confidence`, nombre del experimento. Los scripts leen la config; asi se repite el mismo experimento sin depender de memoria o de flags en la linea de comandos.

- **Versionado**  
  Registrar version del codigo (commit o tag) y version/nombre del modelo teacher y student en el JSON de metricas o en un `run_metadata.json` (fecha, host, parametros usados).

- **Semilla fija**  
  Si hay aleatoriedad (p. ej. muestreo aleatorio de frames), fijar semilla (numpy, random) y documentarla en la config para reproducibilidad.

### 9.4. Visualizacion

- **Overlay teacher vs student**  
  Script o modo que, para uno o varios frames, dibuje en la imagen:
  - Bboxes del teacher (ej. color A).
  - Bboxes del student (ej. color B).
  - Opcional: valor de IoU en cada par matched, o marcar FP/FN.
  Facilita entender donde falla el student (no vio algo, vio de mas, caja desplazada).

- **Herramientas externas**  
  Integracion opcional con **FiftyOne**, **CVAT** o similar para explorar predicciones, filtrar por clase o por frame y comparar teacher/student de forma interactiva.

### 9.5. Performance (script de metricas de rendimiento)

- **Metricas detalladas**  
  No solo FPS: **latencia por frame** (min / media / p99 en ms), y si es posible **desglose por etapa**: preproceso, inferencia, postproceso (NMS, etc.). Ayuda a optimizar el cuello de botella.

- **Recursos en embedded**  
  En Raspberry Pi, UNIHIKER, ESP32-P4: uso de CPU (y si aplica GPU/NPU), RAM, y opcionalmente temperatura o throttling. Registrar en un CSV o JSON por run para comparar hardware.

- **Ventana de medicion**  
  Definir numero de frames o duracion del benchmark (ej. 100 frames o 30 s) y descartar warm-up (primeros N frames) para no sesgar FPS/latencia.

### 9.6. Resumen de mejoras

| Mejora | Donde aplica | Objetivo |
|--------|----------------|----------|
| Varios IoU (0.5, 0.75) | metrics_quality | Ver sensibilidad a localizacion |
| Umbral confianza en metricas | metrics_quality | Curvas P-R, trade-off confianza |
| Metricas por clase | metrics_quality | Multi-clase, reporte por clase |
| mAP | metrics_quality / pycocotools | Estandar, benchmarks |
| Analisis por frame / tamano | metrics_quality + export | Saber donde y en que falla |
| Config YAML/JSON | Todos los scripts | Reproducibilidad |
| Versionado y semilla | Pipeline | Reproducibilidad |
| Overlay teacher vs student | Script o modulo nuevo | Debug y comprension visual |
| Latencia y desglose por etapa | metrics_performance | Optimizacion |
| Recursos en embedded | metrics_performance | Comparar hardware |

Implementar estas mejoras en el orden que priorices (p. ej. primero varios IoU y config, luego analisis de errores y visualizacion, despues mAP y performance detallado) permite ir de la v1 educativa a un flujo de evaluacion mas profesional sin reescribir el nucleo.

---

## 10. Frameworks, librerias y herramientas profesionales

En la industria y en investigacion se usan estas herramientas para evaluacion, metricas y visualizacion. Nuestro nucleo v1 no las requiere; sirven como referencia y como paso siguiente (v2).

### Metricas y evaluacion

| Herramienta | Uso | Notas |
|-------------|-----|--------|
| **pycocotools** | Calcular mAP, Precision/Recall con multiples IoU, formato COCO | Estandar en deteccion; requiere anotaciones en formato COCO. |
| **COCO API / formato COCO** | Formato de anotaciones y predicciones (JSON con categories, images, annotations) | Benchmark COCO; interoperable con muchas herramientas. |
| **TensorFlow Object Detection API** | Metricas (mAP, etc.) y evaluacion en pipeline TF | Si ya usas modelos TF OD. |
| **torchmetrics** (Detection) | Metricas de deteccion en PyTorch (mAP, etc.) | Integrado en ecosistema PyTorch. |

### Formatos de anotacion / prediccion

- **COCO JSON**: el mas usado en benchmarks y papers (categories, images, annotations con bbox, category_id, etc.).
- **Pascal VOC XML**: clasico; un XML por imagen.
- **YOLO**: formato txt (clase, x_center, y_center, w, h normalizados); habitual en proyectos YOLO.

Nuestro JSON por frame (lista de frames con bbox absolutos) se puede **mapear a COCO** para usar pycocotools u otras herramientas que esperen COCO.

### Visualizacion y exploracion de resultados

| Herramienta | Uso | Notas |
|-------------|-----|--------|
| **FiftyOne** | Explorar datasets, visualizar predicciones vs ground truth, filtrar por metrica, comparar modelos | Muy usado para analisis visual y debugging. |
| **CVAT** | Anotacion y revision de predicciones (comparar modelo vs humano) | Open source; sirve para anotar y para revisar salidas. |
| **Label Studio** | Anotacion y revision, integracion con modelos | Alternativa a CVAT. |
| **Weights & Biases (W&B)** | Logs de metricas, artefactos, visualizaciones de evaluacion | Mas orientado a experimentos y ML ops. |

### Benchmarks y datasets de referencia

- **COCO** (Common Objects in Context): dataset y metricas estandar para deteccion/segmentacion; mAP@0.5, mAP@0.5:0.95, etc.
- **Open Images**: otro dataset publico grande; tiene su propio formato y herramientas de evaluacion.

En resumen: para **metricas estandar** lo mas habitual es **pycocotools + formato COCO**; para **ver donde falla el modelo** y comparar teacher/student, **FiftyOne** o un overlay propio; para **anotar o revisar**, **CVAT** o **Label Studio**. Nuestro flujo v1 es independiente de todo esto y se puede conectar luego exportando a COCO o importando en FiftyOne.
