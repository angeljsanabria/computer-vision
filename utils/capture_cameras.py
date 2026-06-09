import cv2
import threading
import time
import logging
import requests
import numpy as np
import settings as s

class CaptureCameras:
    def __init__(self):
        self.mode = s.MODO
        self.rtsp_url = s.IP_CAM_RTSP_URL
        self.snap_url = s.SNAP_HTTP_URL
        self.usb_index = s.USB_INDEX
        self.warmup_frames = s.WARMUP_FRAMES
        self.buffer_size = s.BUFFER_SIZE
        self.reintento_seg = s.REINTENTO_SEG
        self.http_timeout_s = s.HTTP_TIMEOUT_S

        self.latest_frame = None
        self.new_frame_available = False
        self.is_running = False
        self.lock = threading.Lock()
        self.thread = None
        self.cap = None

    def start(self):
        self.is_running = True
        if self.mode == "RTSP":
            self.thread = threading.Thread(target=self._rtsp_loop, name="H_RTSP")
        elif self.mode == "SNAP":
            self.thread = threading.Thread(target=self._snap_loop, name="H_Snap")
        elif self.mode == "USB":
            self.thread = threading.Thread(target=self._usb_loop, name="H_USB")
        else:
            logging.critical(f"Modo de captura no soportado: {self.mode}")
            self.is_running = False
            return self

        self.thread.daemon = True
        self.thread.start()
        return self

    def _configurar_buffer(self, cap: cv2.VideoCapture) -> None:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        except Exception:
            pass

    def _hard_reset_resources(self):
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                finally:
                    self.cap = None
            self.latest_frame = None
            self.new_frame_available = False

    def _rtsp_loop(self):
        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    if not self.cap.isOpened(): raise cv2.error("Fallo socket RTSP")
                    for _ in range(self.warmup_frames):
                        self.cap.grab()
                    self._configurar_buffer(self.cap)
                    logging.info("Stream RTSP estabilizado.")

                if not self.cap.grab(): raise cv2.error("Grab fallido")
                with self.lock:
                    ret, frame = self.cap.retrieve()
                    if ret:
                        self.latest_frame = frame
                        self.new_frame_available = True
            except (cv2.error, Exception) as e:
                logging.error(f"Error RTSP: {e}. Cooldown activo.")
                self._hard_reset_resources()
                time.sleep(self.reintento_seg)

    def _usb_loop(self):
        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.usb_index, cv2.CAP_V4L2)
                    if not self.cap.isOpened(): raise cv2.error("Bus USB busy/bloqueado")
                    for _ in range(self.warmup_frames):
                        self.cap.grab()
                    self._configurar_buffer(self.cap)
                    logging.info("Hardware USB estabilizado.")

                if not self.cap.grab(): raise cv2.error("Lectura USB fallida")
                with self.lock:
                    ret, frame = self.cap.retrieve()
                    if ret:
                        self.latest_frame = frame
                        self.new_frame_available = True
            except (cv2.error, Exception) as e:
                logging.error(f"Error USB: {e}. Cooldown activo.")
                self._hard_reset_resources()
                time.sleep(self.reintento_seg)

    def _snap_loop(self):
        while self.is_running:
            if self.new_frame_available:
                time.sleep(0.01)
                continue
            try:
                response = requests.get(self.snap_url, timeout=self.http_timeout_s)
                if response.status_code == 200:
                    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                            self.new_frame_available = True
                else:
                    raise requests.RequestException()
            except (requests.RequestException, Exception):
                logging.error("Error en Snapshot HTTP. Cooldown activo.")
                time.sleep(self.reintento_seg)

    def get_frame(self):
        with self.lock:
            if self.new_frame_available:
                self.new_frame_available = False
                return True, self.latest_frame
            return False, None

    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self._hard_reset_resources()