Actúa como un Ingeniero de Software Experto en Visión Artificial. Necesito diseñar un plan de trabajo y los scripts necesarios para actualizar mi sistema de enrolamiento de rostros (generador de referencias), el cual actualmente genera embeddings (.npy) a partir de fotos fijas sin aplicar ninguna corrección.

El objetivo es generar dos alternativas de galería para realizar pruebas A/B de precisión y rendimiento en una placa edge RK3568:
1. `gallery.npy` (Multi-pose sintética de 3 embeddings por persona, sin alineación ArcFace).
2. `gallery_align.npy` (Mono-pose de 1 embedding por persona, con alineación estricta tipo ArcFace/InsightFace).

Armá un plan de trabajo detallado que incluya el diseño de los siguientes componentes:

### COMPONENTE 1: SCRIPT DE ENROLAMIENTO (OFFLINE / PC)
El script debe tomar una carpeta de fotos frontales de usuarios y procesar las dos opciones:
    
- LÓGICA PARA OPCIÓN 1 (gallery.npy - Multi-pose):
  1. Detectar el rostro y los 5 landmarks con RetinaFace.
  2. Calcular el ángulo original de los ojos. Rotar la imagen para alinear los ojos perfectamente a 0° (Fronteado). Este será el Embedding Central.
  3. A partir de esa imagen a 0°, aplicar una matriz de rotación geométrica (vía cv2.getRotationMatrix2D) para inclinar virtualmente el rostro a -5° (Giro Izquierda) y a +5° (Giro Derecha).
  4. Extraer los embeddings de las 3 variantes con MobileFaceNet y guardarlos en una matriz de (N*3, 128).

- LÓGICA PARA OPCIÓN 2 (gallery_align.npy - Mono-pose Aligned):
  1. Tomar la misma foto original.
  2. Aplicar la función de transformación de similitud estricta de InsightFace (estimate_norm_cp) utilizando los 5 landmarks para forzar el rostro a la cuadrícula fija estándar de ArcFace (112x112).
  3. Extraer 1 solo embedding por persona y guardarlo en una matriz de (N, 128).

### COMPONENTE 2: MODIFICACIONES EN RUNTIME (main_mov.py para RK3568)
Explica cómo debe adaptarse el orquestador en la placa para alternar entre ambas opciones mediante un flag (ej. `GALERIA_MODO = 'MULTIPOSE' | 'ALIGNED'`):

- Si está en 'MULTIPOSE': El runtime no alinea de forma compleja. Solo hace crop/resize por hardware (RGA). El Matcher hace producto punto contra (N*3, 128) y aplica `np.max()` por cada ID de usuario para buscar si matchea con cualquiera de las 3 poses.
- Si está en 'ALIGNED': El runtime debe aplicar la alineación matemática de InsightFace en cada frame antes de pasarlo a la NPU.

POR FAVOR, ENTREGÁ EL PLAN ESTRUCTURADO EN:
- Fase 1: Arquitectura del script de enrolamiento (Código Python para la generación sintética de las poses a 0°, -5° y +5°).
- Fase 2: Formato de almacenamiento de los archivos .npy y sus respectivos .json de metadatos para no perder la relación de qué vectores pertenecen a qué ID.
- Fase 3: Lógica del Matcher adaptada en main_mov.py para resolver el np.max() eficientemente en la opción Multi-pose.