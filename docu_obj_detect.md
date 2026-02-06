# Object detection

Object detection para Computer Vision es una tarea que 
localiza y clasifica uno o más objetos dentro de una 
imagen o video.

Hay varias tecnologias para esto
- mediapipe (manos y caras)
- OpenCv
- Yolov8
- Detectron2
- AWS Recoknition (servicio cloud)
- etc

En principio funciona con un
- Input: image
- Output: lista de obj detectados (Cuantos hay)
  - Bounding box (location)
  - Confidence score (0 a 1)
  - Obj Category (class name)

### Comparación con Clasificación y Segmentación
**Clasificación de Imágenes**:
La tarea consiste en decir qué hay en una imagen entera, sin indicar en qué lugar.

- Entrada: imagen completa
- Salida: una etiqueta

**Segmentación de Imágenes**

a. Segmentación Semántica
Asigna una clase a cada pixel.

No diferencia instancias individuales del mismo objeto.

b. Segmentación por Instancia

Como la semántica, pero también distingue entre objetos individuales.

Ejemplo: en una imagen con 3 personas, segmentación semántica etiqueta todos los píxeles como "persona", mientras que la segmentación por instancia los separa como "persona 1", "persona 2", etc.

### Metricas comunes
- Loss function
  - usada en el proceso de entrenamiento
  - Lower is better - Un valor menos de loss function es mejor!
- Evaluacion
  - IuU Intersection over union
    - IoU = (area of overlap)/(area of union)   - Va de 0 a 1.
    - Si el IoU > 0.5 o 0.75, se considera una detección correcta (según el umbral).
    - Dos bounding boxes, Indica el solapamiento entre la caja predicha y la caja real.
    - Mide la precision de deteccion
    - Higher is better - Mas cercano a 1 es mejor.
  - mAP mean avera precision
    - precision measure: cuan bien podemos interpretar el objeto encontrado?
      - qué porcentaje de tus detecciones son correctas.
    - recall measures: cuan efectivamente podemos encontrar objetos
      - qué porcentaje de objetos verdaderos detectaste.
    - higher is better
- MIN > 
- Object detection with Python FULL COURSE | Computer vision
https://www.youtube.com/watch?v=UL2cfTTqdNo&list=PLb49csYFtO2F13yGo4kNr3o3aJfGeFevK&ab_channel=Computervisionengineer
- 





