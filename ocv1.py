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

# Blanco y negro (escala de grises):
# - Un solo valor por píxel entre 0 (negro) y 255 (blanco)

# OpenCV (Open Source Computer Vision Library)
# - Librería de código abierto para procesar imágenes y video en tiempo real.
# - Funcionalidades: leer, mostrar, filtrar colores, detectar bordes, movimiento, etc.
# - Instalación: `pip install opencv-python`

# ---
# 🧪 Funcionalidades implementadas
# - Carga una imagen desde disco
# - Muestra forma, tipo y píxeles
# - Redimensiona si es muy grande
# - Dibuja texto y formas
# - Modifica canales de color
# - Guarda y muestra resultado

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
h, w, _ = image.shape
rsize = (640, 480)
if h > rsize[0] and w > rsize[1]:
    image = cv2.resize(image, rsize)
    print(f'reShape {image.shape}')

ADD_TEXT = False
if ADD_TEXT:
    cv2.putText(image, "Pixel (10,100)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

ADD_CIRCLE = False
if ADD_CIRCLE:
    cv2.circle(image, (200, 200), 50, (255, 0, 0), -1)

UPD_BLUE = False
if UPD_BLUE:
    image[:, :, 0] = 255 # al maximo el B de BGR

UPD_JUST_RED = True
if UPD_JUST_RED:
    image[:, :, 0] = 0  # Azul a 0
    image[:, :, 1] = 0  # Verde a 0

SHOW = False
if SHOW:
    # Mostrar la imagen en una ventana
    cv2.imshow("Muestro image", image)

    # Guardo imagen procesada
    EXPORT = False
    if EXPORT:
        IMG_PATH_OUT = os.path.join('.', 'images', IMG_OUT)  # '.' current dir
        cv2.imwrite(IMG_PATH_OUT, image)

    # Esperar hasta que se presione una tecla por 10 segs
    cv2.waitKey(10*1000) # 0, indefinido.

    # Cerrar todas las ventanas
    cv2.destroyAllWindows()

