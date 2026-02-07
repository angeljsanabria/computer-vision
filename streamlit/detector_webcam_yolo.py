"""
Script de deteccion de objetos usando YOLOv8 con interfaz web Streamlit desde webcam.

Descripcion:
-----------
Este script permite seleccionar una camara y detectar objetos en tiempo real usando YOLOv8n.
Muestra los resultados con bounding boxes y etiquetas sobre el video de la camara.

Ejecucion:
---------
1. Instalar dependencias (si no estan instaladas):
   pip install streamlit

2. Ejecutar la aplicacion:
   streamlit run streamlit/detector_webcam_yolo.py
   add: --server.port 8501

3. Ver en el navegador: http://localhost:8501
El script abrira una interfaz web donde podras seleccionar una camara y ver
la deteccion de objetos en tiempo real.
"""
import streamlit as st
import cv2
from ultralytics import YOLO

model = YOLO('../Yolo-Weights/yolov8n.pt')
st.title("Seleccion y Detección en webcam")

run = False

cams = []

cam_labels = {
    0: "iPhone (Continuity Camera)",
    1: "Webcam interna Mac",
    2: "Cam externa USB",
}

for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        cams.append((i, cam_labels.get(i, f"Cam {i}")))
    cap.release()

st.write("Listado de camaras")
cam_opciones = [f"{idx}: {label}" for idx, label in cams]
cam_seleccion = st.radio("Seleccioná una cámara", cam_opciones, index=None)


FRAME_WINDOW = st.image([])

if cam_seleccion:
    cam_idx = int(cam_seleccion.split(":")[0])
    cap = cv2.VideoCapture(cam_idx)
    run = True

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("No se pudo leer el frame.")
            break
        results = model(frame)
        annotated = results[0].plot()
        FRAME_WINDOW.image(annotated, channels="BGR")
else:
    st.info("Seleccioná una cámara para comenzar.")

# Cleanup
if 'cap' in locals():
    cap.release()
cv2.destroyAllWindows()
