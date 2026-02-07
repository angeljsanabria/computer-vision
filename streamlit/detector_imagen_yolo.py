"""
Script de deteccion de objetos usando YOLOv8 con interfaz web Streamlit.

Descripcion:
-----------
Este script permite cargar una imagen y detectar objetos usando el modelo YOLOv8n (nano).
Muestra los resultados con bounding boxes y etiquetas de los objetos detectados.

Ejecucion:
---------
1. Instalar dependencias (si no estan instaladas):
   pip install streamlit

2. Ejecutar la aplicacion:
   streamlit run streamlit/detector_imagen_yolo.py
   add: --server.port 8503

El script abrira una interfaz web en el navegador donde podras subir una imagen
y ver los objetos detectados en tiempo real.
"""
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("YOLOv8 Detector - A partir de una imagen")

uploaded_file = st.file_uploader("Cargá una imagen", type=["jpg", "png", "jpeg"])
model = YOLO('../Yolo-Weights/yolov8n.pt')

if uploaded_file:
    img = Image.open(uploaded_file)
    results = model(img)
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='Resultado', use_column_width=True)
