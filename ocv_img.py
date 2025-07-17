# ==============================================================================
# Módulo: OpenCV Básico
# Descripción: Carga, procesa y muestra una imagen con OpenCV
# ==============================================================================

# ---
# 📌 Introducción
# Computer Vision (CV): permite que las computadoras interpreten imágenes y videos usando inteligencia artificial.

# Una imagen es básicamente un arreglo de NumPy:
# - Tamaño 640x480: 640 columnas (ancho) y 480 filas (alto)
# - Cada píxel tiene 3 canales RGB → [120, 45, 255] representa azul fuerte
# - En OpenCV, el orden de color por defecto es **BGR**
# - Formato común: 8 bits por canal (0-255)

# OpenCV (Open Source Computer Vision Library)
# - Librería de código abierto para procesar imágenes y video en tiempo real.
# - Funcionalidades: leer, mostrar, filtrar colores, detectar bordes, movimiento, etc.
# - Instalación: `pip install opencv-python`

# ---
# 🎨 Color Spaces
# Un **espacio de color** define cómo se representan los colores numéricamente.
# OpenCV usa BGR por defecto, pero permite convertir a muchos otros con `cv2.cvtColor()`:
# - `cv2.COLOR_BGR2RGB`: Reordena los canales al formato más habitual en pantallas.
# - `cv2.COLOR_BGR2GRAY`: Reduce a un solo canal de luminancia (escala de grises).
#     - Útil para procesamiento más liviano y técnicas donde el color no es relevante.
#     - Por ejemplo: detección de bordes, detección de caras, umbralización.
# - `cv2.COLOR_BGR2HSV`: Muy útil para detectar colores (más robusto que en BGR).
#     - H: Hue (tono) → el color puro.
#     - S: Saturación → intensidad del color.
#     - V: Valor → brillo (luminosidad).
# ℹ️ Nota: Imágenes en escala de grises devuelven solo 2 dimensiones (H, W), sin canal explícito.

# Por ejemplo:
#     `img_gray.shape` devuelve solo 2 dimensiones → no tiene canal explícito como en (H, W, C)
#     pero sigue siendo una imagen con un único canal.

# ---
# 💧 Blurring (Desenfoque de imágenes)
# El desenfoque se utiliza para reducir el ruido, suavizar transiciones de color y preparar la imagen para otras tareas
# como la detección de bordes o máscaras.

# ✍️ Fundamento matemático:
# Se aplica una operación de *convolución* usando un **kernel** (matriz como 3x3, 5x5, etc).
# Cada píxel se reemplaza por un valor calculado a partir de sus vecinos. Ejemplo: promedio simple.
# La diferencia entre los métodos está en cómo se calcula ese nuevo valor:

# - `cv2.blur()` → Promedio simple (suavizado general).
# - `cv2.GaussianBlur()` → Promedio ponderado con distribución gaussiana (ideal para ruido fino).
# - `cv2.medianBlur()` → Usa la mediana de los píxeles vecinos (excelente para ruido "sal y pimienta").
# - `cv2.bilateralFilter()` (no incluido en este script) → Suaviza mientras preserva bordes (para filtros finos tipo “retoque facial”).

# Casos reales:
# - Mejora resultados de edge detection (`cv2.Canny`)
# - Limpia imágenes de cámaras ruidosas o comprimidas
# - Mejora máscaras HSV al eliminar ruido de color
# - Facilita el procesamiento en tareas como OCR o detección facial

# ---
# 🎯 Thresholding (Umbralización)
# El thresholding permite **segmentar una imagen** separando los píxeles de interés del fondo.
# Por ejemplo: "Todo lo que sea más claro que cierto umbral → blanco (255), el resto → negro (0)".
# Es fundamental para preparar imágenes antes de extraer contornos, reconocer texto, u objetos.

# ✍️ Fundamento matemático:
# Se aplica una función por píxel:
#     si pixel >= T: pixel = maxval
#     si pixel <  T: pixel = 0
# Donde `T` es el umbral definido y `maxval` es el valor máximo posible (por ejemplo 255).

# 📚 Tipos de umbral usados:
# - `cv2.THRESH_BINARY`           → blanco o negro según el umbral
# - `cv2.THRESH_BINARY_INV`       → igual al anterior pero invertido
# - `cv2.THRESH_TRUNC`            → los píxeles mayores al umbral se recortan al valor T
# - `cv2.THRESH_TOZERO`           → los píxeles menores a T se vuelven cero
# - `cv2.THRESH_TOZERO_INV`       → los mayores a T se vuelven cero

# 📈 Métodos automáticos:
# - `cv2.THRESH_OTSU`
#     → Calcula automáticamente el mejor umbral global (único valor T) separando fondo y objeto.
#     → ✅ Útil cuando la imagen tiene **histograma bimodal** (dos regiones bien diferenciadas de intensidad).
#     → Ej: segmentar letras oscuras sobre fondo claro en un escaneo limpio.

# - `cv2.ADAPTIVE_THRESH_MEAN_C`
#     → Divide la imagen en bloques y calcula el umbral local como el **promedio de vecinos**.
#     → ✅ Funciona bien en imágenes con **iluminación no uniforme** o sombras.
#     → Ej: documentos escaneados con zonas claras y oscuras.

# - `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`
#     → Igual que el anterior, pero pondera los vecinos usando una función gaussiana (más peso al centro).
#     → ✅ Más suave y preciso para transiciones graduales.
#     → Ej: fotos de documentos tomadas con celular, con brillo natural o degradado de luz.

# 💡 Casos reales:
# - Aislar texto para OCR (detección de caracteres)
# - Detectar formas o bordes antes de `cv2.findContours()`
# - Preparar máscaras binarias para segmentación
# - Separar regiones claras/oscura en imágenes médicas o industriales

# ---
# 🎯 Detección de Bordes (Edge Detection)
# La detección de bordes permite **resaltar los contornos** de objetos, texto o estructuras dentro de una imagen.
# Detecta **cambios bruscos de intensidad** (gradientes), lo cual es útil para segmentar, contar o analizar formas.

# ✍️ Fundamento matemático:
# - Se analizan los cambios de intensidad usando derivadas (gradientes).
# - En el caso de Canny, se calcula:
#     G = √(Gx² + Gy²)
#   donde Gx y Gy son derivadas en X e Y, respectivamente.
# - En Laplaciano, se aplica la segunda derivada:
#     ΔI = ∂²I/∂x² + ∂²I/∂y²

# 📚 Métodos utilizados:
# - `cv2.Canny(image, threshold1, threshold2)`
#     → Aplica:
#         1. Desenfoque (reduce ruido)
#         2. Cálculo de gradientes
#         3. Umbrales dobles para detectar bordes fuertes y débiles
#     → ✅ Muy usado en visión artificial, OCR, robótica y detección de movimiento.

# - `cv2.Laplacian(image_gray, cv2.CV_64F)`
#     → Usa la segunda derivada para encontrar zonas donde la intensidad cambia en todas direcciones.
#     → ✅ Detecta bordes sin importar su orientación.
#     → Requiere imagen en escala de grises.

# 📌 Parámetros importantes:
# - `threshold1` y `threshold2` → controlan qué tan fuerte debe ser un borde para ser detectado (en Canny).
# - `cv2.CV_64F` → evita pérdida de información en bordes negativos (Laplaciano).

# 💡 Casos reales:
# - Detectar bordes antes de aplicar `cv2.findContours()` o segmentar objetos.
# - Detectar zonas de texto para OCR.
# - Detección de movimiento o análisis de formas geométricas.
# - Navegación de robots, visión para drones, análisis médico y más.

# ✅ Recomendación:
# Siempre aplicar un filtro de suavizado (`cv2.GaussianBlur`) antes de detectar bordes, para reducir el impacto del ruido.


# ---
# 🧪 Funcionalidades implementadas
# ✅ Carga una imagen desde disco
# ✅ Muestra forma, tipo, tamaño y canales
# ✅ Redimensiona si es muy grande
# ✅ Dibuja texto o formas
# ✅ Modifica canales de color (azul, rojo)
# ✅ Recorta un área definida
# ✅ Convierte entre espacios de color (RGB, GRAY, HSV)
# ✅ Aplica blurring con distintas técnicas
# ✅ Muestra y exporta la imagen procesada
# ------------------------------------------------------------------------------

import cv2
import os

IMG = 'lily2.jpg'
IMG_OUT = 'out.jpg'
IMG_PATH = os.path.join('.', 'images', IMG) # '.' current dir
image = cv2.imread(IMG_PATH)

IMG2 = 'carta.png'
IMG_PATH2 = os.path.join('.', 'images', IMG2) # '.' current dir
image2= cv2.imread(IMG_PATH2)
# Leer imagen (por defecto en uint8; Para 16 bits agrego , cv2.IMREAD_UNCHANGED)
if image is None:
    print("No se pudo abrir la imagen")
    exit()

print(f'Type {type(image)}')
print(f'Shape {image.shape}')
print(f"Size: {image.size}")
print(f"Canales: {image.shape[2]}", )  # 3 = BGR
print(f"Dtype: {image.dtype}")  # Esto me indica los bits por canal: uint8, uint16 o float32
print("Pixel (0, 0):", image[0, 0])           # Arriba a la izquierda
print("Pixel (500, 500):", image[500, 500])   # Requiere que la imagen sea >= 501x501

# Process image
TO_RESIZE = False
if TO_RESIZE:
    h, w, _ = image.shape
    rSize = (640, 480)
    if h > rSize[0] and w > rSize[1]:
        img_rsized = cv2.resize(image, rSize)
        print(f'reSize {img_rsized.shape} smaller')
    else:
        print(f'reSize {image.shape} to larger')

ADD_TEXT = False
if ADD_TEXT:
    cv2.putText(image, "Pixel (10,100)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

ADD_CIRCLE = False
if ADD_CIRCLE:
    cv2.circle(image, (200, 200), 50, (255, 0, 0), -1)

ADD_LINE = True
if ADD_LINE:
    cv2.line(image, (100, 100), (400, 400), (255, 0, 0), 3)

ADD_RECTANGLE = False
if ADD_RECTANGLE:
    cv2.rectangle(image, (200, 200), (400, 400), (255, 0, 0), 3)

ADD_RECTANGLE_SOLID = True
if ADD_RECTANGLE_SOLID:
    cv2.rectangle(image, (200, 200), (400, 400), (255, 0, 0), -1)


UPD_BLUE = False
if UPD_BLUE:
    image[:, :, 0] = 255 # al maximo el B de BGR

UPD_JUST_RED = False
if UPD_JUST_RED:
    image[:, :, 0] = 0  # Azul a 0
    image[:, :, 1] = 0  # Verde a 0

TO_CROP = False
if TO_CROP:
    print(f"To crop original {image.shape}")
    crop_h = [110, 500]
    crop_w = [70, 1300]
    img_cropped = image[crop_w[0]:crop_w[1], crop_h[0]:crop_h[1]]    # es un numpy array, por se recorta asi

COLOR_SPACE = False
if COLOR_SPACE:
    print("Convert color space a otro color space")
    CONVERT_COLOR = False
    CONVERT_COLOR_GRAY = False
    CONVERT_COLOR_HSV = True
    if CONVERT_COLOR:
        print("Convert color space a otro color space")
        print("Cambio de BGR a RGB (el verde queda igual) - cv2.COLOR_BGR2RGB")
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if CONVERT_COLOR_GRAY:
        print("Cambio de BGR a escala grises - cv2.COLOR_BGR2GRAY")
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Esta me facilita tener la info de 3 canales ({image.shape}) en 1 canal ({img_gray.shape})")
    if CONVERT_COLOR_HSV:
        print("Cambio de BGR a escala hsv - cv2.COLOR_BGR2HSV")
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

UPD_BLUR = False
if UPD_BLUR:
    k_size = 50 # defino el kernel o vecindario de 50x50
    img_blur1a = cv2.blur(image, (k_size, k_size))

    k_size2 = 5
    img_blur1b = cv2.blur(image, (k_size2, k_size2))

UPD_BLUR2 = False
if UPD_BLUR2:
    k_size = 50 # defino el kernel o vecindario de 50x50
    img_blur2 = cv2.GaussianBlur(image, (7, 7), 7)

UPD_BLUR3 = False
if UPD_BLUR3:
    k_size = 50 # defino el kernel o vecindario de 50x50
    img_blur3 = cv2.medianBlur(image,  7)

UPD_THRESHOLD = False
if UPD_THRESHOLD:
    #primero paso a escala de grises
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, maxval = 35, 255
    ret, img_thr = cv2.threshold(img_gray, thresh, maxval, cv2.THRESH_BINARY)
    #cv2.THRESH_BINARY	Si el píxel > T, se vuelve maxval, sino 0
    # otros tipos: THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV
    # automaticos ADAPTIVE_THRESH_MEAN_C, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_OTSU
    # si el valor pasa de thresh toma el valor de maxval

UPD_THRESHOLD_AUTO = False
if UPD_THRESHOLD_AUTO:
    img_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Otsu (umbral automático)
    _, th_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptativos
    th_adapt_mean = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
    th_adapt_gauss = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)

UPD_EDGE_DETECTION1 = False
if UPD_EDGE_DETECTION1:
    img_edge1 = cv2.Canny(image, threshold1=50, threshold2=200)

UPD_EDGE_DETECTION2 = False
if UPD_EDGE_DETECTION2:
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_edge2 = cv2.Laplacian(img_gray, cv2.CV_64F)

SHOW = True
if SHOW:
    # Mostrar la imagen en una ventana
    if ADD_LINE or ADD_CIRCLE or ADD_TEXT or UPD_BLUE:
        cv2.imshow("Muestro imagen original", image)

    if TO_RESIZE:
        cv2.imshow(f"Resized {img_rsized.shape}", img_rsized)

    if TO_CROP:
        cv2.imshow(f"Recortada {img_cropped.shape}", img_cropped)

    if COLOR_SPACE:
        if CONVERT_COLOR:
            cv2.imshow("cv2.COLOR_BGR2RGB", img_rgb)
        if CONVERT_COLOR_GRAY:
            cv2.imshow("cv2.COLOR_BGR2RGB", img_gray)
            if CONVERT_COLOR_HSV:
                cv2.imshow("cv2.COLOR_BGR2HSV", img_hsv)

    if UPD_BLUR:
        cv2.imshow("Blurring grande - suavizado general", img_blur1a)
        cv2.imshow("Blurring pequeño - suavizado general", img_blur1b)

    if UPD_BLUR2:
        cv2.imshow("Gaussian Blur - ruido fino y bordes suaves", img_blur2)
    if UPD_BLUR3:
        cv2.imshow("Median Blur - ", img_blur3)

    if UPD_THRESHOLD:
        cv2.imshow("Thresholds", img_thr)

    if UPD_THRESHOLD_AUTO:
        cv2.imshow("Muestro imagen original", image2)
        cv2.imshow("Otsu", th_otsu)
        cv2.imshow("Adaptive Mean", th_adapt_mean)
        cv2.imshow("Adaptive Gaussian", th_adapt_gauss)

    if UPD_EDGE_DETECTION1:
        cv2.imshow("edge detection 1",img_edge1)

    if UPD_EDGE_DETECTION2:
        cv2.imshow("edge detection 2",img_edge2)

    # Guardo imagen procesada
    EXPORT = False
    if EXPORT:
        IMG_PATH_OUT = os.path.join('.', 'images', IMG_OUT)  # '.' current dir
        cv2.imwrite(IMG_PATH_OUT, image)

    # Esperar hasta que se presione una tecla por 10 segs
    cv2.waitKey(10*1000) # 0, indefinido.

    # Cerrar todas las ventanas
    cv2.destroyAllWindows()

'''
AGREGAR:
Thresholding (Umbralización).
Es para segmentar imagenes, es decir, separar pixeles de interes del fondo. Por ejemplo, dame todo lo que 
esta mas claro que cierto valor umbral. O todo lo que es mas oscuro que cierto valor umbral, pasalo a negro.
Puede servir para reconocimiento de texto, aislando las letras del fondo.
- DOcumentar todos los Threshold usados
'''
