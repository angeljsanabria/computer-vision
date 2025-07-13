'''
pip install streamlit
'''
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("YOLOv8 Detector - A partir de una imagen")

uploaded_file = st.file_uploader("Cargá una imagen", type=["jpg", "png", "jpeg"])
model = YOLO('Yolo-Weights/yolov8n.pt')

if uploaded_file:
    img = Image.open(uploaded_file)
    results = model(img)
    res_plotted = results[0].plot()
    st.image(res_plotted, caption='Resultado', use_column_width=True)
