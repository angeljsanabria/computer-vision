"""
Script Teacher v1: genera pseudo-ground-truth para evaluacion de modelos.

Lee el JSON de especificacion de muestra (sampling_spec), abre el video en
esos frames, ejecuta MediaPipe Face Detection y guarda las detecciones en
formato lista por frame (frame, timestamp, detecciones con bbox absoluto).
Ese JSON lo usan el script de metricas y el student para comparar.
"""
import argparse
import json
import os
import sys

import cv2
import mediapipe as mp

# Permitir importar utils desde la raiz del proyecto
_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _raiz not in sys.path:
    sys.path.insert(0, _raiz)


from utils.image_utils import bbox_relativo_a_absoluto

video_path_override = "lyly.mp4"

SHOW_FRAME = True

def parse_args():
    return
    p = argparse.ArgumentParser(description="Genera pseudo-GT (teacher) desde sampling_spec + MediaPipe.")
    p.add_argument("--sampling_spec", type=str, required=True, help="Ruta al JSON de sampling (gen_sampling_spec_v1).")
    p.add_argument("--video", type=str, default=None, help="Override del path del video (si no usar el del spec).")
    p.add_argument("--out_json", type=str, default=None, help="Ruta del JSON de salida (default: data/teacher_faces_<nombre>.json).")
    p.add_argument("--model", type=int, choices=[0, 1], default=0, help="MediaPipe model_selection: 0 cercano, 1 lejano.")
    p.add_argument("--min_confidence", type=float, default=0.5, help="Umbral minimo de confianza (0.0-1.0).")
    return p.parse_args()


def run_teacher(spec_path: str, video_path_override: str | None, model_selection: int, min_conf: float) -> list:
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    video_path = video_path_override or spec["video_path"]
    samples = spec.get("samples", [])
    fps = spec.get("fps") or 30.0

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    if not samples:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    mp_face = mp.solutions.face_detection
    results_list = []


def every_n_frames(vPath: str, samples: list) -> list:

    if samples is None or len(samples) == 0: 
        print("Lista de frames vacia")
        return []
    
    video = cv2.VideoCapture(vPath)


    for s in samples:
        video.set(cv2.CAP_PROP_POS_FRAMES, s)
        ret, frame = video.read()  # Ir directo frame s
        
        if not ret:
            print("Error en muestreo")
            return # Raise error
            #break
        
        if SHOW_FRAME:
            cv2.imshow('Frame', frame)
            cv2.waitKey(2000)   


    
    video.release()
    cv2.destroyAllWindows()


    
    

    

def main():
    args = parse_args()


    with open('data/sampling_spec_lily.json', 'r', encoding='utf-8') as f:
        jdata = json.load(f)

    video_path = jdata["video_path"]
    strategy = jdata["strategy"]
    samples = jdata.get("samples", [])
    fps = jdata.get("fps") or 30.0
    strategy

    print("Video ", video_path)
    print("Fps: ", fps)
    print("Strategy: ", strategy)
    print("Samples: ", samples)

    if strategy == 'every_n_frames':
        every_n_frames(video_path, samples)



if __name__ == "__main__":
    main()
