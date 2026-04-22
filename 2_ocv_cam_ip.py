"""
Prueba RTSP (Reolink: ruta Preview_01_sub o Preview_01_main). Edita las constantes y ejecuta:
  python 2_ocv_cam_ip.py
'q' cierra.
"""
import cv2

USER_CAM = "angelcam"
PASS_CAM = "AngelCamara"
IP_CAM = "192.168.0.160"
RTSP_PORT_CAM = 554

# Ruta del stream en la camara (Reolink habitual); cambia main/sub o el sufijo si tu modelo difiere.
RTSP_URL = f"rtsp://{USER_CAM}:{PASS_CAM}@{IP_CAM}:{RTSP_PORT_CAM}/Preview_01_sub"

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit("No abrio RTSP (datos, red o RTSP apagado en la camara).")

for _ in range(25):
    ok, frame = cap.read()
    if ok and frame is not None and frame.size:
        break
else:
    cap.release()
    raise SystemExit("Sin frames; revisa usuario/clave y ruta (main vs sub).")

win_name = "RTSP - q"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        break
    cv2.imshow(win_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
