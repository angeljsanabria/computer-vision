"""
Genera un JSON de especificacion de muestra para un video: que frames evaluar.

Lee el video, aplica un tope maximo de muestras (MAX_SAMPLES) repartidas
uniformemente, y guarda en eval_modelos/data/sampling_spec_<nombre>.json
el path del video, metadata (fps, duracion, estrategia) y la lista de
indices de frame. Teacher y student usan ese JSON para procesar solo
esos frames. Si el archivo ya existe, lo actualiza.
"""
# source /Users/angel-dev/PycharmProjects/.venvOk/bin/activate
import cv2
import json
import os
from datetime import datetime, timezone


# Maximo de muestras
MAX_SAMPLES = 10  
# Estrategia de muestreo
STRATEGY = 'every_n_frames'



# Read video
VID = 'lily.mp4'
VID_PATH = os.path.join('..', 'videos', VID) # '.' current dir .. level down

video = cv2.VideoCapture(VID_PATH)
# Verificar apertura
if not video.isOpened():
    print("No se pudo abrir el video")
    exit()
else:
    print(f"Video abierto: {VID_PATH}")

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
video.release()

# Calcular lista de frames a muestrear (respetando MAX_SAMPLES)
M = min(MAX_SAMPLES, total_frames)
if M == 0:
    frames = []
elif M == 1:
    frames = [0]
else:
    # Repartir M indices uniformemente entre 0 y total_frames - 1
    frames = [int(round(i * (total_frames - 1) / (M - 1))) for i in range(M)]
    frames = [min(f, total_frames - 1) for f in frames]

# Ruta de salida: eval_modelos/data/sampling_spec_<nombre_video>.json
nombre_video = os.path.splitext(VID)[0]
dir_script = os.path.dirname(os.path.abspath(__file__))
dir_data = os.path.join(dir_script, "data")
os.makedirs(dir_data, exist_ok=True)
out_path = os.path.join(dir_data, f"sampling_spec_{nombre_video}.json")

# Si el archivo ya existe, informar y actualizar
existe = os.path.isfile(out_path)
if existe:
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            anterior = json.load(f)
        fecha_ant = anterior.get("fecha_actualizacion", "?")
        print(f"El archivo ya existe. Ultima actualizacion: {fecha_ant}. Actualizando...")
    except (json.JSONDecodeError, OSError):
        print("El archivo ya existe. Actualizando...")

# Encabezado + muestras, en ese orden
video_path_abs = os.path.abspath(VID_PATH)
fecha_actualizacion = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
fps_val = round(float(fps), 2) if fps else 0
duration_sec = round(total_frames / fps_val, 2) if fps_val > 0 else 0

spec = {
    "video_path": video_path_abs,
    "fecha_actualizacion": fecha_actualizacion,
    "strategy": STRATEGY,
    "total_frames": total_frames,
    "fps": fps_val,
    "duration_sec": duration_sec,
    "max_samples": MAX_SAMPLES,
    "num_samples_used": len(frames),
    "samples": frames,
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(spec, f, indent=2, ensure_ascii=False)

print(f"Total frames (video): {total_frames}")
print(f"Maximo de muestras: {MAX_SAMPLES}")
print(f"Cantidad de muestras usadas: {len(frames)}")
print(f"JSON {'actualizado' if existe else 'guardado'}: {out_path}")
