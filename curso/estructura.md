# Curso de Computer Vision para Embedded Systems

## Objetivo General

Aprender Computer Vision desde fundamentos hasta implementación en dispositivos embebidos, con enfoque práctico en Raspberry Pi Zero 2W, UNIHIKER M10 y ESP32-P4, culminando con Edge Impulse, TensorFlow Lite y TensorFlow.

## Hardware Disponible

| Dispositivo | Procesador/Microcontrolador | Tipo | Uso en el Curso |
|-------------|----------------------------|------|-----------------|
| **UNIHIKER M10** | Allwinner H6 (ARM Cortex-A53) | Microprocesador (Linux embebido) | Fases iniciales: OpenCV, modelos completos |
| **Raspberry Pi Zero 2W** | Broadcom BCM2710A1 (ARM Cortex-A53 Quad-core) | Microprocesador (Linux embebido) | Fases iniciales: OpenCV, TensorFlow Lite |
| **ESP32-P4** | RISC-V Dual-core 32-bit | Microcontrolador | Fases avanzadas: TensorFlow Lite Micro, TinyML |
| **Sony IMX179** | Sensor de imagen CMOS | Cámara/Sensor | Captura de imágenes para todas las placas |

---

## Fase 0: Fundamentos (Ya Completado)

### Estado: ✅ DONE

**Temas cubiertos:**
- Fundamentos de imagen digital (matrices NumPy)
- Espacios de color (BGR, RGB, HSV)
- Blurring (Mean, Gaussian, Median)
- Thresholding (Manual y Automático)
- Edge Detection (Canny, Laplacian)
- Operaciones morfológicas
- Contours
- Detección de color
- MediaPipe básico
- Conceptos de Clasificación, Detección y Segmentación
- Métricas de evaluación (IoU, mAP, Precision, Recall)

---

## Fase 1: Deep Learning y CNNs (Fundamentos Teóricos)

### Objetivo
Entender cómo funcionan las redes neuronales convolucionales desde la base matemática, sin depender de frameworks como caja negra.

### 1.1 Fundamentos de Deep Learning
- **Neuronas y perceptrones**
  - Función de activación
  - Forward pass
  - Backpropagation (concepto)
- **Redes neuronales básicas**
  - Capas densas (fully connected)
  - Función de pérdida
  - Optimización (SGD, Adam)

### 1.2 Convolución 2D
- **Operación de convolución**
  - Kernel/filtro
  - Feature maps
  - Padding (valid, same)
  - Stride
- **Implementación manual** (NumPy)
  - Convolución 2D desde cero
  - Visualización de feature maps

### 1.3 Redes Neuronales Convolucionales (CNNs)
- **Arquitectura básica**
  - Convolutional layers
  - Pooling (Max, Average)
  - Flatten
  - Dense layers
- **Por qué funcionan**
  - Detección de patrones espaciales
  - Jerarquía de características
  - Invariantes (traslación, rotación parcial)

### 1.4 Arquitecturas Clásicas (Conceptual)
- LeNet-5
- AlexNet
- VGG
- ResNet (concepto de skip connections)

### 1.5 Transfer Learning
- **Concepto**
  - Pre-trained models
  - Fine-tuning
  - Feature extraction
- **Aplicación práctica**
  - Usar modelos pre-entrenados
  - Adaptar a tareas específicas

---

## Fase 2: Object Detection (Teoría y Práctica)

### Objetivo
Entender cómo funcionan los detectores modernos, desde sliding window hasta YOLO.

### 2.1 Object Detection Clásico
- **Sliding Window**
  - Concepto
  - Limitaciones computacionales
- **Region Proposal**
  - R-CNN (concepto)
  - Fast R-CNN
  - Faster R-CNN
- **Two-stage vs One-stage**
  - Tradeoffs precisión/velocidad

### 2.2 YOLO (You Only Look Once)
- **YOLO v1-v3 (conceptos)**
  - Grid cells
  - Anchors
  - Qué predice realmente
  - Función de pérdida
- **YOLO v8 (práctica)**
  - Instalación y uso
  - Entrenamiento básico
  - Inferencia

### 2.3 Otros Detectores
- **SSD (Single Shot Detector)**
  - Multiscale detection
  - Default boxes
- **EfficientDet**
  - Tradeoff precisión vs velocidad
  - Arquitectura eficiente

### 2.4 Evaluación de Detectores
- **Métricas específicas**
  - IoU para detección
  - Precision-Recall curve
  - mAP (mean Average Precision)
  - COCO mAP @[.5:.95]
- **Métricas para embedded**
  - FPS / Latencia
  - Throughput
  - Uso de memoria
  - Consumo energético

---

## Fase 3: Entrenamiento de Modelos

### Objetivo
Aprender a entrenar modelos desde cero y adaptar modelos pre-entrenados.

### 3.1 Preparación de Datos
- **Dataset**
  - Estructura de carpetas
  - Formatos (COCO, YOLO, Pascal VOC)
  - Anotación de imágenes
  - Herramientas (LabelImg, CVAT)
- **Data Augmentation**
  - Rotación, escalado, flip
  - Color jitter
  - Mixup, Cutout
  - Por qué es crucial

### 3.2 Entrenamiento Básico
- **Framework: PyTorch o TensorFlow**
  - Setup del entorno
  - DataLoader
  - Training loop
  - Validación
- **Hiperparámetros**
  - Learning rate
  - Batch size
  - Epochs
  - Early stopping

### 3.3 Entrenamiento de Clasificador
- **Proyecto práctico**
  - Dataset pequeño (ej: 3-5 clases)
  - Entrenar desde cero
  - Evaluar resultados

### 3.4 Entrenamiento de Detector
- **YOLO v8 custom**
  - Preparar dataset
  - Configurar entrenamiento
  - Monitorear métricas
  - Exportar modelo

### 3.5 Optimización de Modelos
- **Técnicas de optimización**
  - Quantization (INT8, FP16)
  - Pruning
  - Knowledge Distillation
  - Arquitecturas eficientes (MobileNet, EfficientNet)

---

## Fase 4: Embedded Systems - Hardware y Setup

### Objetivo
Conocer las capacidades y limitaciones de los dispositivos objetivo.

### Hardware Disponible

| Dispositivo | Procesador/Microcontrolador | Tipo | Uso en el Curso |
|-------------|----------------------------|------|-----------------|
| **UNIHIKER M10** | Allwinner H6 (ARM Cortex-A53) | Microprocesador (Linux embebido) | Fases iniciales: OpenCV, modelos completos |
| **Raspberry Pi Zero 2W** | Broadcom BCM2710A1 (ARM Cortex-A53 Quad-core) | Microprocesador (Linux embebido) | Fases iniciales: OpenCV, TensorFlow Lite |
| **ESP32-P4** | RISC-V Dual-core 32-bit | Microcontrolador | Fases avanzadas: TensorFlow Lite Micro, TinyML |
| **Sony IMX179** | Sensor de imagen CMOS | Cámara/Sensor | Captura de imágenes para todas las placas |

### 4.1 Raspberry Pi Zero 2W
- **Procesador**: Broadcom BCM2710A1 (ARM Cortex-A53 Quad-core)
- **Especificaciones**
  - CPU: ARM Cortex-A53 Quad-core @ 1.0 GHz
  - RAM: 512 MB
  - Interfaces: USB, GPIO, CSI (para cámara)
- **Setup**
  - Instalación de OS (Raspberry Pi OS)
  - OpenCV para ARM
  - Python environment
- **Cámara**
  - Configuración de módulo de cámara (Sony IMX179 compatible)
  - Captura de video/imagen

### 4.2 UNIHIKER M10
- **Procesador**: Allwinner H6 (ARM Cortex-A53)
- **Especificaciones**
  - CPU: ARM Cortex-A53
  - RAM: Variable según modelo
  - Pantalla integrada
  - Linux embebido
- **Setup**
  - OS y herramientas
  - OpenCV
  - Python environment

### 4.3 ESP32-P4
- **Microcontrolador**: RISC-V Dual-core 32-bit
- **Especificaciones**
  - CPU: RISC-V Dual-core @ 400 MHz
  - RAM: Limitada (~8 MB típico)
  - Flash: Variable según modelo
  - Limitaciones vs Raspberry Pi (sin OS, baremetal)
- **Setup**
  - MicroPython o C
  - TensorFlow Lite Micro
  - Consideraciones de memoria críticas

### 4.4 Sony IMX179 (Sensor de Cámara)
- **Tipo**: Sensor de imagen CMOS
- **Especificaciones**
  - Resolución: 8 MP
  - Interfaz: CSI (Camera Serial Interface)
- **Uso**
  - Compatible con Raspberry Pi Zero 2W
  - Compatible con UNIHIKER M10 (si tiene interfaz CSI)
  - Captura de imágenes para procesamiento

### 4.5 Comparación de Dispositivos
- **Tabla comparativa**
  - CPU performance
  - RAM disponible
  - GPU/Accelerator
  - Consumo energético
  - Casos de uso recomendados
  - Lenguajes soportados (Python vs C)

---

## Fase 5: OpenCV en Embedded

### Objetivo
Implementar pipelines de visión por computadora optimizados para recursos limitados.

### 5.1 Optimización de OpenCV
- **Compilación optimizada**
  - Flags de compilación para ARM
  - Deshabilitar módulos innecesarios
  - Optimizaciones NEON (si aplica)
- **Uso eficiente**
  - Reducir resolución
  - ROI (Region of Interest)
  - Operaciones en grayscale cuando sea posible

### 5.2 Pipeline Básico en Raspberry Pi
- **Proyecto práctico**
  - Captura de video
  - Procesamiento en tiempo real
  - Detección de color/objetos simples
  - Medición de FPS

### 5.3 Optimización de Performance
- **Técnicas**
  - Multithreading
  - Frame skipping
  - Procesamiento asíncrono
  - Reducción de operaciones costosas

---

## Fase 6: TensorFlow Lite

### Objetivo
Convertir y ejecutar modelos en dispositivos embebidos usando TensorFlow Lite.

### 6.1 Introducción a TensorFlow Lite
- **Qué es y por qué**
  - Formatos de modelo (.tflite)
  - Optimizaciones automáticas
  - Delegates (GPU, NNAPI, Coral)

### 6.2 Conversión de Modelos
- **Desde TensorFlow/Keras**
  - Conversión básica
  - Quantization post-training
  - Quantization aware training
- **Desde PyTorch**
  - ONNX como intermediario
  - Conversión a TFLite

### 6.3 Inferencia en Python (Raspberry Pi)
- **Setup**
  - Instalación de TensorFlow Lite
  - Cargar modelo .tflite
- **Implementación**
  - Preprocesamiento
  - Inferencia
  - Postprocesamiento
  - Medición de latencia

### 6.4 Optimización de Modelos TFLite
- **Técnicas**
  - Quantization INT8
  - Pruning
  - Model optimization toolkit
  - Benchmarking

### 6.5 Proyecto Práctico: Clasificador en Raspberry Pi
- **Implementación completa**
  - Modelo entrenado → TFLite
  - Pipeline de inferencia
  - Integración con cámara
  - Interfaz simple

---

## Fase 7: Edge Impulse

### Objetivo
Usar Edge Impulse para crear y desplegar modelos de visión optimizados para embedded.

### 7.1 Introducción a Edge Impulse
- **Plataforma**
  - Qué es y para qué sirve
  - Ventajas para embedded
  - Flujo de trabajo

### 7.2 Creación de Proyecto
- **Setup**
  - Cuenta y proyecto
  - Conectar dispositivo (Raspberry Pi)
  - Captura de datos

### 7.3 Entrenamiento en Edge Impulse
- **Dataset**
  - Subir imágenes
  - Anotación
  - Data augmentation
- **Modelo**
  - Selección de arquitectura
  - Hiperparámetros
  - Entrenamiento
  - Evaluación

### 7.4 Deployment
- **Exportación**
  - Modelo optimizado
  - C++ library
  - Python SDK
- **Integración**
  - Instalar en dispositivo
  - Ejecutar inferencia
  - Monitoreo

### 7.5 Proyecto Práctico Completo
- **Caso de uso real**
  - Definir problema
  - Capturar datos
  - Entrenar modelo
  - Desplegar en Raspberry Pi Zero 2W
  - Evaluar performance

---

## Fase 8: TensorFlow Lite Micro (ESP32-P4)

### Objetivo
Ejecutar modelos en microcontroladores con recursos muy limitados.

### 8.1 TensorFlow Lite Micro
- **Qué es**
  - Diferencias con TFLite estándar
  - Limitaciones
  - Casos de uso

### 8.2 Setup en ESP32-P4
- **Entorno de desarrollo**
  - ESP-IDF
  - Compilación de TFLite Micro
  - Flash del firmware

### 8.3 Modelo para Microcontrolador
- **Restricciones**
  - Tamaño de modelo (< 100KB típico)
  - Memoria RAM limitada
  - Operaciones soportadas
- **Optimización extrema**
  - Quantization INT8
  - Modelos ultra-ligeros
  - Arquitecturas específicas (MobileNet v1 tiny)

### 8.4 Inferencia en ESP32-P4
- **Implementación**
  - Cargar modelo
  - Preprocesamiento (limitado)
  - Inferencia
  - Postprocesamiento

### 8.5 Proyecto Práctico: Clasificación Simple
- **Caso de uso**
  - Sensor simple (ej: detección de presencia)
  - Modelo mínimo
  - Integración con hardware

---

## Fase 9: TensorFlow Completo en Linux Embebido

### Objetivo
Usar TensorFlow completo (no Lite) en dispositivos embebidos más potentes.

### 9.1 Dispositivos Objetivo
- **Hardware más potente**
  - NVIDIA Jetson Nano/Orin
  - Raspberry Pi 4/5 (con limitaciones)
  - Otros SBCs con GPU

### 9.2 Setup de TensorFlow
- **Instalación**
  - TensorFlow para ARM
  - Optimizaciones específicas
  - GPU support (si aplica)

### 9.3 Modelos Complejos
- **Ventajas vs TFLite**
  - Modelos más grandes
  - Operaciones completas
  - Fine-tuning en dispositivo (limitado)

### 9.4 Proyecto Práctico
- **Aplicación completa**
  - Modelo complejo
  - Pipeline de inferencia
  - Integración con sistema

---

## Fase 11: OCR (Optical Character Recognition) en Embedded

### Objetivo
Implementar sistemas de reconocimiento de texto optimizados para dispositivos embebidos, aplicando conceptos de preprocesamiento, detección y clasificación aprendidos en fases anteriores.

### 11.1 Fundamentos de OCR
- **Qué es OCR y qué resuelve**
  - Lectura automática de texto en imágenes
  - Casos de uso en embedded:
    - Lectura de displays digitales (termómetros, contadores)
    - Lectura de placas/etiquetas
    - Documentos escaneados
    - Lectura de códigos de barras/QR (extensión)
- **Qué NO resuelve**
  - No entiende el significado del texto (solo lo lee)
  - Requiere texto relativamente claro y legible
  - Dificultades con texto manuscrito (requiere modelos especializados)
  - Texto muy pequeño o muy borroso puede fallar
- **En qué parte del pipeline de visión se usa**
  - Preprocesamiento → Detección de texto → Reconocimiento → Post-procesamiento
  - Se integra con otros sistemas (detección de objetos, tracking)
- **Diferencias con conceptos similares**
  - vs Clasificación: OCR clasifica caracteres, no objetos completos
  - vs Object Detection: OCR detecta y reconoce texto, no objetos genéricos
  - vs Letter Recognition: OCR es más complejo (múltiples caracteres, palabras)

### 11.2 Pipeline de OCR
- **Dos etapas principales**
  1. **Text Detection (Detección de texto)**
     - Encontrar regiones donde hay texto en la imagen
     - Similar a object detection pero específico para texto
     - Output: bounding boxes de regiones de texto
  2. **Text Recognition (Reconocimiento de texto)**
     - Leer los caracteres dentro de cada región detectada
     - Clasificación de caracteres o secuencias
     - Output: texto legible (string)
- **Preprocesamiento crítico**
  - Binarización (thresholding)
  - Deskewing (corrección de inclinación)
  - Noise removal
  - Normalización de tamaño

### 11.3 Técnicas Clásicas vs Deep Learning
- **Métodos clásicos**
  - Tesseract OCR (basado en patrones y features clásicas)
  - Preprocesamiento manual intensivo
  - Limitaciones con fuentes no estándar
  - Ventajas: ligero, funciona sin GPU
- **Deep Learning para OCR**
  - **CRNN (Convolutional Recurrent Neural Networks)**
    - CNN para extraer features
    - RNN (LSTM) para secuencias de caracteres
    - CTC (Connectionist Temporal Classification) para alineación
  - **Attention mechanisms**
    - Modelos más modernos con atención
    - Mejor para texto complejo
  - **End-to-end models**
    - Detección + reconocimiento en un solo modelo
    - Más pesado pero más preciso

### 11.4 OCR en Embedded Systems
- **Consideraciones de recursos**
  - Modelos ligeros para embedded
  - Optimizaciones específicas (quantization, pruning)
  - Trade-offs precisión vs velocidad vs memoria
- **Herramientas disponibles**
  - **PaddlePaddle OCR** (ejemplo ARM Cortex-M)
    - Modelo optimizado para microcontroladores
    - Ejemplo disponible en ARM Edge AI
  - **EasyOCR**
    - Fácil de usar, pero más pesado
    - Mejor para Raspberry Pi que para ESP32
  - **Tesseract optimizado**
    - Versión clásica, muy ligera
    - Funciona en dispositivos limitados
  - **TensorFlow Lite para OCR**
    - Modelos CRNN convertidos a TFLite
    - Optimización para embedded

### 11.5 Implementación en Raspberry Pi Zero 2W / UNIHIKER
- **Setup y configuración**
  - Instalación de herramientas (Tesseract, EasyOCR, o modelos custom)
  - Preprocesamiento de imágenes
  - Consideraciones de performance
- **Pipeline completo**
  - Captura de imagen
  - Preprocesamiento (thresholding, deskewing)
  - Detección de regiones de texto
  - Reconocimiento de caracteres
  - Post-procesamiento (corrección, formateo)
- **Optimizaciones**
  - Reducir resolución de entrada
  - ROI (Region of Interest) para áreas específicas
  - Procesamiento asíncrono
  - Caching de resultados

### 11.6 OCR en ESP32-P4 (Opcional - Avanzado)
- **Limitaciones y desafíos**
  - Modelos ultra-ligeros necesarios
  - PaddlePaddle OCR en Cortex-M (ejemplo ARM)
  - Memoria muy limitada
  - Solo texto simple y claro
- **Casos de uso específicos**
  - Lectura de displays digitales simples
  - Lectura de números en contadores
  - Texto predecible y limitado

### 11.7 Proyecto Práctico: Sistema de Lectura de Displays
- **Caso de uso real**
  - Lectura de displays digitales (termómetros, contadores, medidores)
  - Captura con cámara
  - Procesamiento en tiempo real o batch
- **Implementación**
  - En Raspberry Pi Zero 2W o UNIHIKER
  - Pipeline completo de OCR
  - Optimización para tiempo real
  - Evaluación de precisión
  - Manejo de errores y casos edge

### 11.8 Variaciones según Hardware
- **Raspberry Pi Zero 2W / UNIHIKER**
  - Puede usar EasyOCR o Tesseract
  - Modelos CRNN con TensorFlow Lite
  - Tiempo real posible con optimizaciones
- **ESP32-P4**
  - Solo modelos ultra-ligeros (PaddlePaddle OCR)
  - Texto simple y predecible
  - Procesamiento más lento, posiblemente no tiempo real

---

## Fase 10: Proyectos Integrados y Optimización Avanzada

### Objetivo
Crear proyectos completos optimizados para producción en embedded.

### 10.1 Optimización Avanzada
- **Técnicas avanzadas**
  - Model quantization avanzada
  - Neural Architecture Search (NAS)
  - AutoML para embedded
  - Hardware-specific optimizations

### 10.2 Pipeline Completo
- **Arquitectura de sistema**
  - Captura → Preprocesamiento → Inferencia → Postprocesamiento → Acción
  - Gestión de memoria
  - Manejo de errores
  - Logging y monitoreo

### 10.3 Proyecto Final: Sistema Completo
- **Requisitos**
  - Múltiples dispositivos (opcional)
  - Modelo entrenado custom
  - Pipeline optimizado
  - Documentación completa
  - Benchmarking de performance

### 10.4 Mejores Prácticas
- **Producción**
  - Versionado de modelos
  - Testing
  - Deployment strategies
  - Mantenimiento

---

## Recursos y Referencias

### Documentación Oficial
- OpenCV: https://docs.opencv.org/
- TensorFlow Lite: https://www.tensorflow.org/lite
- Edge Impulse: https://docs.edgeimpulse.com/
- YOLO: https://docs.ultralytics.com/
- Tesseract OCR: https://github.com/tesseract-ocr/tesseract
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- PaddlePaddle OCR: https://github.com/PaddlePaddle/PaddleOCR

### Recursos ARM Edge AI
- ARM Edge AI Examples: https://developer.arm.com/edge-ai/example-applications
- OCR con PaddlePaddle en Cortex-M: [Learning Path ARM](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/avh_ppocr/)
- Letter Recognition en STM32: [Learning Path ARM](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/tflow_nn_stcube/)
- TinyML en Arm: [Learning Path ARM](https://learn.arm.com/learning-paths/embedded-and-microcontrollers/introduction-to-tinyml-on-arm/)

### Comunidades
- OpenCV Forum
- TensorFlow Community
- Edge Impulse Community
- Reddit: r/computervision, r/embedded, r/OCR

### Hardware
- Raspberry Pi Official Docs
- ESP32 Documentation
- UniHiker Documentation

---

## Notas Importantes

1. **Enfoque Práctico**: Cada fase debe incluir al menos un proyecto práctico.
2. **Iteración**: No es necesario completar todo antes de avanzar, se puede iterar.
3. **Adaptación**: Ajustar según necesidades específicas del proyecto final.
4. **Documentación**: Documentar cada proyecto y aprendizaje clave.
