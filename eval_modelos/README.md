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

En la version inicial vamos a trabajar con **tres scripts principales**:

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

En una **v2** agregaremos un cuarto componente separado:

4. **Script de Performance** (`metrics_performance_v1.py`)
   - Mide FPS, latencia, uso de CPU/RAM.
   - No mira calidad de deteccion, solo coste de ejecucion.
   - Se puede correr en cada hardware (PC, Raspberry, UNIHIKER, etc.).

---

## 3. Formato de la Matriz por Frame (Teacher y Student)

Tanto el teacher como el student van a producir archivos JSON con una **lista de frames evaluados**.

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

### 6.2. Entrada del script

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

Una vez que entendamos bien la v1 a mano, podremos:
- Mapear nuestras matrices a **formato COCO**.
- Usar **pycocotools** para calcular mAP y otras metricas estandar.
- Integrar con herramientas como **FiftyOne** para visualizacion.

---

## 7. Script de Performance (`metrics_performance_v1.py`) – FUTURO

Este script se encargara de medir **performance**, separado de la calidad:

- FPS (frames por segundo) del pipeline completo.
- Latencia por frame (ms/frame).
- Uso de CPU y memoria (si es relevante).
- Comparacion entre hardware:
  - PC
  - Raspberry Pi Zero 2W
  - UNIHIKER M10
  - ESP32-P4 (conceptualmente, con modelos tiny).

La idea es:
- Primero usar `metrics_quality_v1.py` para asegurarnos de que el modelo ve bien.
- Despues usar `metrics_performance_v1.py` para ver donde y que tan rapido podemos correrlo.

---

## 8. Resumen

Este sistema de evaluacion busca:
- Ser **modular** (teacher, student, metricas, performance separados).
- Ser **reproducible** (mismas matrices, mismos frames para todos los modelos).
- Ser **educativo**: primero v1 a mano, luego migrar a librerias profesionales.

Proximo paso sugerido:
- Definir en codigo (paso a paso) el script `gen_teacher_gt_v1.py` usando
  el modelo de MediaPipe que ya tienes (`7_ocv_scr_anonymize_v2py`) y generar
  el primer `teacher_faces_*.json` de prueba.

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

En la version inicial vamos a trabajar con **tres scripts principales**:

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

En una **v2** agregaremos un cuarto componente separado:

4. **Script de Performance** (`metrics_performance_v1.py`)
   - Mide FPS, latencia, uso de CPU/RAM.
   - No mira calidad de deteccion, solo coste de ejecucion.
   - Se puede correr en cada hardware (PC, Raspberry, UNIHIKER, etc.).

---

## 3. Formato de la Matriz por Frame (Teacher y Student)

Tanto el teacher como el student van a producir archivos JSON con una **lista de frames evaluados**.

Ejemplo de estructura (simplificada):

