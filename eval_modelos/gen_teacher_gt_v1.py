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


def parse_args():
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

    with mp_face.FaceDetection(model_selection=model_selection, min_detection_confidence=min_conf) as face_detection:
        for frame_idx in samples:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            timestamp = round(frame_idx / fps, 3) if fps > 0 else 0.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out = face_detection.process(frame_rgb)
            detecciones = []
            if out.detections:
                for det in out.detections:
                    conf = det.score[0]
                    if conf < min_conf:
                        continue
                    bbox_rel = det.location_data.relative_bounding_box
                    x, y, w, h = bbox_relativo_a_absoluto(bbox_rel, frame.shape)
                    detecciones.append({
                        "bbox": [x, y, w, h],
                        "class": "face",
                        "confidence": round(float(conf), 4),
                    })
            results_list.append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "detecciones": detecciones,
            })

    cap.release()
    return results_list


def main():
    args = parse_args()
    if not os.path.isfile(args.sampling_spec):
        print(f"Error: no existe el archivo de sampling: {args.sampling_spec}")
        sys.exit(1)

    data = run_teacher(
        args.sampling_spec,
        args.video,
        args.model,
        args.min_confidence,
    )

    if args.out_json:
        out_path = args.out_json
    else:
        dir_script = os.path.dirname(os.path.abspath(__file__))
        dir_data = os.path.join(dir_script, "data")
        with open(args.sampling_spec, "r", encoding="utf-8") as f:
            spec = json.load(f)
        nombre = os.path.splitext(os.path.basename(spec.get("video_path", "video")))[0]
        out_path = os.path.join(dir_data, f"teacher_faces_{nombre}.json")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Frames procesados: {len(data)}. Guardado: {out_path}")


if __name__ == "__main__":
    main()
