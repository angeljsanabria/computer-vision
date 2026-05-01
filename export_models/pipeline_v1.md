# Pipeline Profesional de Biometría Facial para Rockchip RK3568

Este documento detalla el flujo de trabajo (pipeline) utilizado en productos de nivel industrial (control de acceso, asistencia), optimizado para la NPU de la RK3568.

---

## 1. El Pipeline Propuesto

A diferencia de un script simple, un producto profesional implementa capas de validación para evitar errores de seguridad y falsos positivos.


| Etapa | Proceso | Nombre Técnico | Objetivo |
| :--- | :--- | :--- | :--- |
| **0** | **Disparo (Trigger)** | VAD (Video Activity Detection) | Detectar movimiento (Background Subtraction) para despertar la NPU. |
| **1** | **Detección** | **RetinaFace** / **SCRFD** | Localizar coordenadas del rostro y 5 puntos clave (Landmarks). |
| **2** | **Filtro de Calidad** | Face Quality Assessment | Descartar si hay mucho desenfoque (blur) o pose extrema (>30°). |
| **3** | **Anti-Fraude** | Liveness Detection | Verificar que es una persona real y no una foto/pantalla. |
| **4** | **Alineación** | Affine Transformation | Rotar y escalar la imagen para que los ojos queden siempre en la misma posición. |
| **5** | **Embedding** | Feature Extraction | Convertir la cara en un vector matemático (128 o 512 dimensiones). |
| **6** | **Comparación** | Vector Matching | Distancia Coseno contra DB usando FAISS para búsqueda 1:N ultra rápida. |

---

## 2. Los Modelos: Rockchip Zoo vs. Estado del Arte (2024)

Para la RK3568, el rendimiento depende de la **NPU**. Aquí están las mejores opciones actuales:

### A. Etapa de Detección (Face Detection)
*   **En el Zoo de Rockchip:** **RetinaFace** (MobileNet backbone).
    *   *Estado:* Muy estable y confiable. Es la opción segura para empezar.
*   **Opción Moderna (Mejor que el Zoo):** **SCRFD** (Sample and Computation Redistribution).
    *   *Ventaja:* Es el actual estado del arte para dispositivos Edge. Es más rápido que RetinaFace y detecta rostros con oclusiones (barbijos, lentes) con mayor precisión.
*   **Alternativa:** **YOLOv8-Face**. Si ya usas el ecosistema Ultralytics, es muy fácil de convertir a `.rknn`.

### B. Etapa de Reconocimiento (Face Recognition/Embeddings)
*   **En el Zoo de Rockchip:** **ArcFace** y **FaceNet**.
    *   *Estado:* ArcFace sigue siendo el estándar industrial. FaceNet ya se considera "antiguo" para productos nuevos.
*   **Opción Moderna (Mejor que el Zoo):** **AdaFace**.
    *   *Ventaja:* Lanzado recientemente, supera a ArcFace en "imágenes difíciles" (mala iluminación o baja resolución). Maneja mejor la calidad variable de las cámaras de seguridad.
*   **Eficiencia Extrema:** **MobileFaceNet**. Es la versión "mini". Ideal si necesitas procesar múltiples cámaras o dejar recursos de la NPU para otras tareas.

---

## 3. Implementación Sugerida para Producto Real

Si estás diseñando un producto hoy, esta es la "receta" recomendada para la RK3568:

1.  **Detector:** **SCRFD** (Cuantizado a **INT8**). Proporciona Landmarks muy precisos.
2.  **Alineador:** Script de Python usando `cv2.getAffineTransform` basado en los landmarks del paso 1.
3.  **Extractor de Identidad:** **ArcFace** o **AdaFace** (Backbone MobileNetV2, cuantizado a **INT8**).
4.  **Anti-Spoofing:** **Silent-Face-Anti-Spoofing** (Modelo pasivo, no requiere que el usuario se mueva).

---

## 4. Claves de Rendimiento en RK3568

*   **Uso de NPU:** Nunca uses modelos en FP16. La RK3568 brilla con **INT8**. Debes usar un dataset de calibración (unas 100-200 fotos de caras) al convertir con `rknn-toolkit2`.
*   **Pre-procesamiento:** Realizar el redimensionamiento de imagen (resize) y el cambio de espacio de color (BGR a RGB) dentro de la NPU si es posible, o usar `Pillow-simd` para no saturar la CPU.
*   **Memoria:** Para sistemas de acceso con miles de usuarios, no guardes las fotos en RAM. Guarda solo los **embeddings** (vectores) en una base de datos indexada (como SQLite con extensión vectorial o FAISS).

---
*Documento generado para desarrollo en sistemas embebidos Rockchip.*


# Migración y Optimización de Modelos para RK3568

Este documento detalla la estrategia de cambio de modelos para pasar de los ejemplos educativos del **Rockchip Model Zoo** a arquitecturas de vanguardia.

> **Nota Crítica:** Todos los modelos "Recomendados" requieren un proceso de conversión manual:  
> `PyTorch (.pt) / TensorFlow` —> `ONNX` —> `RKNN (vía RKNN-Toolkit2)`.

---

## Fase 1: Detección de Rostros (Face Detection)
El detector es el primer filtro. Un error aquí detiene todo el pipeline.


| Modelo Actual (Zoo) | Modelo Recomendado | ¿Por qué cambiar? |
| :--- | :--- | :--- |
| **RetinaFace** (MobileNet 0.25) | **SCRFD** (Efficient Face Det) | **Velocidad y Oclusiones:** SCRFD es hasta un 50% más eficiente en GFLOPs. Detecta rostros con barbijos, manos en la cara o ángulos extremos donde RetinaFace suele fallar. |
| **YOLOv3/v5-Face** | **YOLOv8-Face (Nano)** | **Precisión de Landmarks:** YOLOv8-Face ofrece puntos de referencia mucho más estables para el alineamiento, reduciendo el "temblor" de los cuadros en video real. |

---

## Fase 2: Reconocimiento Facial (Embeddings)
Aquí se define la seguridad del sistema y la capacidad de distinguir entre gemelos o personas parecidas.


| Modelo Actual (Zoo) | Modelo Recomendado | ¿Por qué cambiar? |
| :--- | :--- | :--- |
| **FaceNet** | **ArcFace (ResNet/MobileNet)** | **Obsolescencia:** FaceNet usa distancias euclidianas menos robustas. ArcFace es el estándar actual de la industria por su margen angular, mucho más preciso para 1:N. |
| **ArcFace** (Estándar) | **AdaFace** (2022/2023) | **Calidad Adaptativa:** AdaFace "entiende" si una imagen es de mala calidad (borrosa/oscura) y ajusta su peso en el entrenamiento. En la vida real, reduce drásticamente los falsos rechazos en pasillos mal iluminados. |
| **Cualquier modelo pesado** | **EdgeFace** (2023) | **Optimización Edge:** Diseñado específicamente para correr en NPUs limitadas. Ofrece precisión de servidor en un tamaño de modelo "mini". |

---

## Fase 3: Seguridad y Robustez (Liveness)
El Zoo de Rockchip no suele incluir protección contra ataques. Esta fase es obligatoria para un producto profesional.


| Lo que falta | Modelo a Integrar | Función |
| :--- | :--- | :--- |
| **Anti-Spoofing** | **Silent-Face-Anti-Spoofing** | Analiza la textura de la imagen para detectar si es una pantalla LCD o papel impreso. Sin esto, cualquier persona entra con una foto de Instagram. |
| **Filtro de Calidad** | **Face Quality Assessment** | Un modelo ligero que descarta frames antes de enviarlos a la NPU si el rostro no está de frente o está desenfocado, ahorrando batería y cómputo. |

---

## Checklist para la Conversión a .RKNN

Para que estos modelos externos funcionen en tu RK3568, debes seguir estos pasos en tu estación de trabajo (PC con Ubuntu):

1.  **Exportación ONNX:** Asegúrate de usar `opset=12` o superior. Desactiva operaciones dinámicas (como *Dynamic Shapes*) ya que la NPU requiere tamaños fijos (ej. 112x112).
2.  **Dataset de Calibración:** No conviertas a INT8 "en seco". Necesitas al menos 50-100 imágenes reales de rostros para que el Toolkit2 calcule los rangos de cuantización sin perder precisión.
3.  **Verificación de Capas:** Usa el comando `rknn.visualize()` para asegurarte de que ninguna capa se "caiga" a la CPU. Si una capa no es compatible, busca una implementación equivalente en el repositorio de Rockchip.
4.  **Optimización de Memoria:** Activa el `target_platform='rk3568'` en la configuración del Toolkit para que el compilador optimice los pesos para tu chip específico.

---
