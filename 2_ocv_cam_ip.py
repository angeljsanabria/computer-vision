"""
    Prueba RTSP (Reolink: ruta Preview_01_sub o Preview_01_main).
    Edita las constantes y ejecuta:
    python 2_ocv_cam_ip.py
    'q' cierra + ctrl+c.
"""
import cv2

# Campos que forman la URL
USER_CAM = "angelcam"
PASS_CAM = "AngelCamara"
IP_CAM = "192.168.0.160"
RTSP_PORT_CAM = 554     # por defecto, pero depende de la config de la camara
RES_HIGH_CAM = "Preview_01_main"
RES_LOW_CAM = "Preview_01_sub"
STREAM_CAM = RES_LOW_CAM
MAX_FPS_ANALISIS = 2.0  # Limita en el analisis (y muestra) de los FPS (grab levanta y no analiza hasta retrieve)

# Ruta del stream en la camara (Reolink habitual); cambia main/sub o el sufijo si tu modelo difiere.
RTSP_URL = f"rtsp://{USER_CAM}:{PASS_CAM}@{IP_CAM}:{RTSP_PORT_CAM}/{STREAM_CAM}"

'''
    Mostrar logs de capturas
'''
LOG_CADA_CAPS = 2
def log_cada_10_frames(frame_count, t0, frame):
    if frame_count % LOG_CADA_CAPS != 0:
        return
    ticks = cv2.getTickCount() - t0
    dt = ticks / cv2.getTickFrequency()
    fps = frame_count / dt if dt > 0 else 0.0
    h, w = frame.shape[:2]
    print(f"[LOG] frame={frame_count} size={w}x{h} fps_aprox={fps:.2f}")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit("No abrio RTSP (datos, red o RTSP apagado en la camara).")

for _ in range(25):
    ok, frame = cap.read()
    if ok and frame is not None and frame.size:
        break
else:
    cap.release()
    raise SystemExit(f"Sin respuesta de frames. Revisar la url {RTSP_URL}")

win_name = "RTSP - q"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
frame_count = 0
t0 = cv2.getTickCount()
periodo = 1.0 / MAX_FPS_ANALISIS if MAX_FPS_ANALISIS > 0 else 0.0
next_due = cv2.getTickCount()
while True:
    # grab() avanza el stream; retrieve() decodifica solo cuando toca analizar/mostrar.
    if not cap.grab():
        break
    now_tick = cv2.getTickCount()
    if periodo > 0:
        now = now_tick / cv2.getTickFrequency()
        due = next_due / cv2.getTickFrequency()
    else:
        now, due = 0.0, 0.0
    if periodo > 0 and now < due:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue
    next_due = cv2.getTickCount() + int(periodo * cv2.getTickFrequency())

    ok, frame = cap.retrieve()
    if not ok or frame is None:
        break

    frame_count += 1
    log_cada_10_frames(frame_count, t0, frame)

    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
