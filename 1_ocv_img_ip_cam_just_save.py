"""
Script minimo: solo descarga y guarda un JPEG de la camara IP (Reolink) por HTTP.

Usa el CGI cmd=Snap (una peticion GET), elige resolucion con SNAP_HTTP_URL
(RES_LOW o RES_FULL) y escribe el binario en FILE_PATH sin validar ni mostrar.

Requisitos: HTTP habilitado en la camara, credenciales correctas. La carpeta FILE_DIR debe existir.
Ejemplo: python "1_ocv_img_ip_cam_just_save.py"

Doc Reolink (Snap):
  https://support.reolink.com/hc/en-us/articles/360007011233-How-to-Capture-Live-JPEG-Image-of-Reolink-Cameras-via-Web-Browsers
"""
import requests
import os

# Campos que forman la URL
USER_CAM = "angelcam"
PASS_CAM = "AngelCamara"
IP_CAM = "192.168.0.160"
RES_HIGH_CAM = "width=2560&height=1920"
RES_LOW_CAM = "width=640&height=480"
FILE_DIR = "camara_snap"
FILE_NAME_IMG = "latest_camara_snap.jpg"
FILE_PATH = os.path.join(FILE_DIR, FILE_NAME_IMG)

# docu
# https://support.reolink.com/articles/360007011233-How-to-Capture-Live-JPEG-Image-of-Reolink-Cameras-via-Web-Browsers/?slug=stream&search=snapshot%20url

# ejemplo snap http url
# http://192.168.0.160/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa&user=angelcam&password=AngelCamara&width=640&height=480
SNAP_HTTP_URL_RES_FULL = f"http://{IP_CAM}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa&user={USER_CAM}&password={PASS_CAM}&{RES_HIGH_CAM}"
SNAP_HTTP_URL_RES_LOW = f"http://{IP_CAM}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa&user={USER_CAM}&password={PASS_CAM}&{RES_LOW_CAM}"

SNAP_HTTP_URL = SNAP_HTTP_URL_RES_LOW

try:
    # verify=False ignora errores de certificado en HTTPS
    response = requests.get(SNAP_HTTP_URL, timeout=100, verify=False)

    # Solo aceptamos 200, sino es falla
    if response.status_code == 200:
        with open(FILE_PATH, "wb") as f:
            f.write(response.content)
        print(f"Imagen guardada como {FILE_PATH}")
    else:
        print(f"Error del servidor: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"Error de conexión: {e}")

