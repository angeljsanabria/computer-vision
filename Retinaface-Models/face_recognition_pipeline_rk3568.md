# Face Recognition Pipeline para RK3568 (RetinaFace RKNN + ArcFace)

## Objetivo

Construir un pipeline completo de reconocimiento facial sobre RK3568 usando:

- RetinaFace.rknn para detección facial
- Alignment estilo InsightFace/ArcFace
- ArcFace.rknn para embeddings
- Cosine similarity para matching
- Cámara USB como input
- Linux ARM64 sobre RK3568

El objetivo es evitar SDKs cerrados y mantener control total del pipeline.

---

# Arquitectura Final

```text
USB Camera
    ↓
RetinaFace.rknn
    ↓
bbox + landmarks
    ↓
Filtro de mejor rostro (opcional)
    ↓
Alignment InsightFace
    ↓
112x112 aligned face
    ↓
ArcFace.rknn
    ↓
Embedding
    ↓
Cosine similarity
    ↓
Identity match
```

---

# Componentes del Pipeline

## 1. Cámara USB

### Input

```python
cv2.VideoCapture("/dev/video0")
```

### Output

Frame RGB/BGR completo.

---

## 2. RetinaFace.rknn

### Responsabilidad

Detectar:

- bounding boxes faciales
- landmarks faciales

### Output esperado

```python
{
    "bbox": [x1, y1, x2, y2],
    "score": 0.98,
    "landmarks": [
        [left_eye_x, left_eye_y],
        [right_eye_x, right_eye_y],
        [nose_x, nose_y],
        [mouth_left_x, mouth_left_y],
        [mouth_right_x, mouth_right_y]
    ]
}
```

---

## 3. Selección de Mejor Rostro (Opcional pero recomendado)

### Objetivo

Si hay múltiples personas frente a la cámara:

- procesar SOLO la cara más relevante
- evitar inferencias innecesarias
- reducir uso NPU
- mejorar estabilidad

### Estrategia recomendada

Elegir:

- la cara más grande
- y con mejor confidence score

### Métrica recomendada

```text
face_score = area * confidence
```

Donde:

- area = (x2 - x1) * (y2 - y1)

### Algoritmo

```python
best_face = max(
    faces,
    key=lambda f: (
        ((f["bbox"][2] - f["bbox"][0]) *
         (f["bbox"][3] - f["bbox"][1]))
        * f["score"]
    )
)
```

### Resultado

Se prioriza:

- la cara más cercana a la cámara
- con mejor calidad de detección

---

## 4. Alignment InsightFace

### Objetivo

Normalizar geométricamente el rostro para ArcFace.

### Qué corrige

- rotación
- inclinación
- escala
- centrado
- affine transform

### Archivos necesarios de InsightFace

NO hace falta instalar todo InsightFace.

Solo necesitás portar:

### Funciones necesarias

- estimate_norm(...)
- norm_crop(...)

### Archivo de referencia

Buscar en InsightFace:

- face_align.py

o similares.

### Template facial estándar ArcFace

```python
arcface_dst = np.array([
 [38.2946, 51.6963],
 [73.5318, 51.5014],
 [56.0252, 71.7366],
 [41.5493, 92.3655],
 [70.7299, 92.2041]
], dtype=np.float32)
```

### Operaciones usadas

Matriz afín

```text
[
x
′
y
′
	​

]=A[
x
y
	​

]+b
```

### Código típico

```python
M = cv2.estimateAffinePartial2D(
    src_landmarks,
    arcface_dst
)[0]

aligned = cv2.warpAffine(
    image,
    M,
    (112,112)
)
```

---

## 5. ArcFace.rknn

### Responsabilidad

Generar embeddings faciales.

### Input esperado

```text
112x112 RGB aligned face
```

### Output

```python
embedding = [
    0.123,
   -0.551,
    ...
]
```

Normalmente:

- 128 floats
- 256 floats
- 512 floats

### Modelos recomendados

#### Para RK3568

**MobileFaceNet**

Prioriza:

- FPS
- temperatura
- bajo consumo

**ArcFace-r50**

Prioriza:

- accuracy
- embeddings más robustos

---

## 6. Similarity

### Método recomendado

Cosine similarity.

```text
cos(θ)=
∥A∥∥B∥
A⋅B
	​
```

### Código típico

```python
similarity = np.dot(a, b) / (
    np.linalg.norm(a) *
    np.linalg.norm(b)
)
```

---

## 7. Base de Embeddings

### Guardado recomendado

```python
np.save("juan.npy", embedding)
```

### Matching

```python
if similarity > 0.45:
    print("MATCH")
```

Threshold típico:

- 0.4 a 0.6

Debe calibrarse experimentalmente.

---

# RGA en RK3568

## Importante

RGA NO se usa automáticamente.

Ni:

- OpenCV
- numpy
- InsightFace

usan RGA por defecto.

## Qué usa CPU

Estas operaciones:

- cv2.resize()
- cv2.warpAffine()
- cv2.cvtColor()

usan CPU normalmente.

## Cuándo usar RGA

Solo si:

- múltiples cámaras
- muchos FPS
- muchas caras
- CPU alta

## Recomendación inicial

Primera versión:

- OpenCV alignment sobre CPU

Es suficientemente rápido en la mayoría de escenarios.

## Optimización futura

Más adelante:

- RGA resize
- RGA affine
- zero-copy
- tracking

---

# InspireFace (Opcional)

## Descarga RK3568

Desde:

<https://github.com/HyperInspire/InspireFace/releases>

Descargar:

- inspireface-linux-aarch64-rk356x-rk3588-1.2.3.zip

## Modelos InspireFace

Descargar:

```bash
bash command/download_models_general.sh Gundam_RK356X
```

## Importante

En este proyecto NO se recomienda depender completamente de InspireFace.

Se usará:

- RetinaFace.rknn propio
- Alignment estilo InsightFace
- ArcFace.rknn propio

porque brinda:

- mayor control
- mejor debugging
- flexibilidad
- optimización específica

---

# Pipeline Recomendado Final

```text
USB Camera
    ↓
RetinaFace.rknn
    ↓
bbox + landmarks
    ↓
selección mejor rostro
    ↓
InsightFace Alignment
    ↓
112x112 aligned face
    ↓
ArcFace.rknn
    ↓
embedding
    ↓
cosine similarity
    ↓
identity match
```

---

# Estado actual del proyecto

## Ya resuelto

- RetinaFace.rknn
- inferencia RKNN
- landmarks faciales

## Pendiente

- alignment InsightFace
- ArcFace.rknn
- cosine similarity
- base embeddings
- thresholds
- filtro mejor rostro
- optimización RGA opcional

## Prioridades reales

La calidad del sistema dependerá principalmente de:

- alignment consistente
- preprocessing consistente
- thresholds correctos
- calidad de cámara
- control de blur/pose

Más que del modelo exacto de embeddings.
