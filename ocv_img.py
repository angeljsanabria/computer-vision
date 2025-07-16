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
# Leer imagen (por defecto en uint8; Para 16 bits agrego , cv2.IMREAD_UNCHANGED)
image = cv2.imread(IMG_PATH)
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

UPD_BLUR2 = True
if UPD_BLUR2:
    k_size = 50 # defino el kernel o vecindario de 50x50
    img_blur2 = cv2.GaussianBlur(image, (7, 7), 7)

UPD_BLUR3 = True
if UPD_BLUR3:
    k_size = 50 # defino el kernel o vecindario de 50x50
    img_blur3 = cv2.medianBlur(image,  7)


SHOW = True
if SHOW:
    # Mostrar la imagen en una ventana
    #cv2.imshow("Muestro imagen original", image)
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
Blurring
Cada vez que se aplica el blurring es como si se tomara un promedio de los pixeles vecinos (kernel, pequeña matriz
de 3x3 o 5x5, y se mueve por cada pixel de la imagen - convolucion -. Y este calculo va reemplazando el pixel central. 
La forma de 
como se definen los pixeles vecinos o el calculo es el tipo de blur que se aplica. Sirve para suavizado general, 
ruido fino, bodes suaves, eliminacion de ruidos con bordes nitidos.
'''
