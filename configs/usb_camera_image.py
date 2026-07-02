"""
Ajuste de imagen OpenCV solo para captura USB.
USE_DEFAULT: no modifica props de la camara.
USE_CUSTOM: cap.set brillo/contraste/saturacion (perfil SONY abajo).

Referencia V4L2 camara USB Sony IMX179 (/dev/video10, RK3568) no usada por OpenCV aqui:
  hue min=-180 max=180 default=0
  gamma min=100 max=500 default=300
  sharpness min=0 max=100 default=80
  auto_exposure menu 0-3 default=3
  focus_automatic_continuous default=1
"""
import os

import cv2

USB_CAMERA_IMAGE_MODE = os.getenv("USB_CAMERA_IMAGE_MODE", "USE_DEFAULT").upper()

BRIGHTNESS = 0  # SONY: min=-64 max=64 default=0
CONTRAST = 51  # SONY: min=0 max=100 default=51
SATURATION = 64  # SONY: min=0 max=100 default=64


def validar_usb_camera_image() -> None:
    if USB_CAMERA_IMAGE_MODE not in ("USE_DEFAULT", "USE_CUSTOM"):
        raise SystemExit(
            "CONFIG ERROR: USB_CAMERA_IMAGE_MODE debe ser USE_DEFAULT o USE_CUSTOM "
            f"(got {USB_CAMERA_IMAGE_MODE!r})."
        )


def aplicar_opencv(cap: cv2.VideoCapture) -> None:
    if USB_CAMERA_IMAGE_MODE != "USE_CUSTOM":
        return
    cap.set(cv2.CAP_PROP_BRIGHTNESS, BRIGHTNESS)
    cap.set(cv2.CAP_PROP_CONTRAST, CONTRAST)
    cap.set(cv2.CAP_PROP_SATURATION, SATURATION)
