# 🧠 Resumen General – Visión por Computadora (hasta ahora)

Este resumen recopila **todos los conceptos, técnicas y herramientas** que trabajamos hasta el momento, con foco en **entendimiento teórico y matemático**, no en copiar código.

---

## 1️⃣ Fundamentos de Imagen Digital

### Conceptos clave
- Una imagen es una **matriz NumPy**:
  - Grayscale → `H × W`
  - Color → `H × W × C`
- En OpenCV:
  - El acceso es `img[y, x, canal]`
  - Primero filas (alto), luego columnas (ancho)

### Idea fundamental
Una imagen no es algo visual sino **datos numéricos**.  
Todo lo que hacemos en visión por computadora son **operaciones matemáticas sobre matrices**.

---

## 2️⃣ Espacios de Color (BGR, RGB, HSV)

### Qué aprendimos
- OpenCV usa **BGR** por defecto
- HSV separa:
  - **Hue (H)** → color
  - **Saturation (S)** → pureza del color
  - **Value (V)** → brillo

### Por qué HSV es tan importante
- El color queda desacoplado de la iluminación
- Es más robusto para:
  - Detección de color
  - Segmentación
  - Tracking

---

## 3️⃣ Blurring (Suavizado)

### Métodos vistos
- Mean Blur
- Gaussian Blur
- Median Blur

### Fundamento matemático
Convolución con un kernel:

\[
I'(x,y) = \sum_{i,j} I(x+i, y+j) \cdot K(i,j)
\]

### Uso real
- Reducción de ruido
- Preprocesamiento para:
  - Thresholding
  - Edge detection
  - Contours

---

## 4️⃣ Thresholding (Umbralización)

### Tipos
- Manual:
  - Binary
  - Trunc
  - ToZero
- Automático:
  - Otsu
  - Adaptive Mean
  - Adaptive Gaussian

### Idea central
Convertir una imagen continua → **imagen binaria**

Paso clave antes de:
- Contornos
- OCR
- Morfología

---

## 5️⃣ Edge Detection (Detección de Bordes)

### Métodos trabajados
- Canny
- Laplacian

### Fundamento matemático
- Los bordes son zonas con **alto gradiente**
- Canny incluye:
  - Gradiente (Sobel)
  - Supresión de no-máximos
  - Umbrales dobles e histéresis

### Usos
- Detección de contornos
- Resaltar estructuras
- Preprocesamiento

---

## 6️⃣ Operaciones Morfológicas

### Técnicas
- Erosión
- Dilatación
- Base para opening y closing

### Base matemática
- Operaciones sobre conjuntos binarios usando un **elemento estructurante**

### Casos reales
- Eliminar ruido
- Unir regiones
- Mejorar máscaras

---

## 7️⃣ Contours

### Qué son
- Curvas cerradas que delimitan regiones con el mismo valor

### Puntos importantes
- No detectan objetos semánticos
- Necesitan:
  - Imagen binaria
  - Buen preprocesamiento

### Clave conceptual
**Contours ≠ Object Detection**  
Contours trabajan con **geometría**, no con significado.

---

## 8️⃣ Detección de Color

### Herramientas usadas
- OpenCV + NumPy
- HSV + rangos
- PIL para bounding box

### Concepto fuerte
Detectar color = segmentar por rangos en un espacio de color

### Limitaciones
- Sensible a iluminación
- Sensible a saturación
- Requiere calibración

---

## 9️⃣ MediaPipe – Detección de Caras

### Qué se entendió
- Coordenadas relativas `[0,1]` vs píxeles
- Bounding boxes
- Keypoints
- Confidence score

### Concepto clave
MediaPipe **no “ve” caras**, predice probabilidades y posiciones:

\[
P(\text{cara}), \hat{x}, \hat{y}, \hat{w}, \hat{h}
\]

---

## 🔟 Clasificación, Detección y Segmentación

| Tarea | Qué devuelve |
|-----|-------------|
| Clasificación | Qué hay |
| Detección | Qué + dónde |
| Segmentación | Qué + dónde + forma |

### Idea clave
- Clasificación: semántica
- Detección: semántica + localización
- Segmentación: semántica + píxel a píxel

---

## 1️⃣1️⃣ Evaluación de Modelos

### Métricas clásicas
- Precision
- Recall
- IoU
- mAP

### Métricas modernas
- COCO mAP @[.5:.95]
- FPS / Latencia
- Throughput
- Métricas energéticas (edge / mobile)

### Concepto importante
Un modelo no se evalúa solo por precisión, sino por **costo computacional**.

---

## 1️⃣2️⃣ Estructuración de Código

### Buenas prácticas vistas
- Separación lógica
- Tipado
- argparse
- Diseño orientado al aprendizaje

---

# 🚀 Propuesta de Temas para Continuar

## 🔹 Ruta A – Núcleo Teórico (recomendada)
Ideal para entender YOLO sin usarlo como caja negra.

### Módulo 2 – CNNs desde cero
- Convolución 2D
- Stride y padding
- Feature maps
- Pooling
- Por qué una CNN detecta patrones espaciales

### Módulo 3 – Object Detection clásico
- Sliding window
- Anchors
- Two-stage vs One-stage detectors
- Bounding box regression

---

## 🔹 Ruta B – Detectores Modernos

### Módulo 4 – YOLO (bien entendido)
- Grid cells
- Anchors
- Qué predice realmente
- Función de pérdida

### Módulo 5 – SSD / EfficientDet
- Multiscale detection
- Tradeoff precisión vs velocidad

---

## 🔹 Ruta C – Más allá del Bounding Box

### Módulo 6 – Segmentación
- Semantic segmentation
- Instance segmentation
- Mask R-CNN (conceptual)

### Módulo 7 – Tracking
- SORT / Deep SORT
- Detección + seguimiento

---

## 🎯 Recomendación final

Con el enfoque que estás teniendo:

👉 **Continuar con el Módulo 2: CNNs y detección desde la base**, con énfasis en teoría y matemática.

Eso te va a permitir entender **YOLO, SSD y cualquier detector moderno** sin depender de recetas.

---
