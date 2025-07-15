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

# Por ejemplo:
#     `img_gray.shape` devuelve solo 2 dimensiones → no tiene canal explícito como en (H, W, C)
#     pero sigue siendo una imagen con un único canal.

# OpenCV (Open Source Computer Vision Library)
# - Librería de código abierto para procesar imágenes y video en tiempo real.
# - Funcionalidades: leer, mostrar, filtrar colores, detectar bordes, movimiento, etc.
# - Instalación: `pip install opencv-python`

# ---
# 🧪 Funcionalidades implementadas
# - Carga una imagen desde disco
# - Muestra forma, tipo y píxeles
# - Redimensiona si es muy grande y muestra
# - Dibuja texto y formas
# - Modifica canales de color
# - Guarda y muestra resultado
# - Recortar una imagen

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

COLOR_SPACE = True
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





SHOW = True
if SHOW:
    # Mostrar la imagen en una ventana
    cv2.imshow("Muestro image", image)
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
Color spaces se puede entender como las diferentes formas de representar los colores en la imagen 
aplico COLOR_BGR2RGB y bgr a escala de grises. 
La de grises es importante porque me pasa la info de 3 canales en 1 canal. 
Explicar porque hsv es importante o muy usado

Por que en:
        print("Cambio de BGR a escala grises - cv2.COLOR_BGR2GRAY")
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"Esta me facilita tener la info de 3 canales ({image.shape}) en 1 canal ({img_gray.shape})")

No me muestra que tiene un canal de color?  -> Esta me facilita tener la info de 3 canales ((723, 1440, 3)) en 1 canal ((723, 1440))

'''
