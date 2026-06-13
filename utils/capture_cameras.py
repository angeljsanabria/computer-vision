import cv2
import sys
import threading
import time
import logging
import requests
import numpy as np
from configs import settings as s


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
        self.max_fps = s.MAX_FPS

        if self.max_fps > 0:
            self._periodo_ticks = int(cv2.getTickFrequency() / self.max_fps)
        else:
            self._periodo_ticks = 0
        self._next_due_tick = cv2.getTickCount()
        self._frame_count = 0
        self._t0_tick = cv2.getTickCount()

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

    def _abrir_usb(self) -> cv2.VideoCapture | None:
        """Solo USB local. Linux: V4L2 primero; Windows: backend por defecto."""
        if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(self.usb_index, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap
            cap.release()
        cap = cv2.VideoCapture(self.usb_index)
        return cap if cap.isOpened() else None

    def _preparar_usb(self, cap: cv2.VideoCapture) -> bool:
        """Warmup USB: descarta lecturas hasta un frame valido."""
        for _ in range(self.warmup_frames):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return True
        return False

    def _publicar_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            self.latest_frame = frame
            self.new_frame_available = True
        self._frame_count += 1
        if self._frame_count % s.LOG_CADA_N_FRAMES == 0:
            dt = (cv2.getTickCount() - self._t0_tick) / cv2.getTickFrequency()
            fps = self._frame_count / dt if dt > 0 else 0.0
            h, w = frame.shape[:2]
            logging.info(
                f"[{self.mode}] frame={self._frame_count} "
                f"size={w}x{h} fps_aprox={fps:.2f}"
            )

    def _toca_capturar_stream(self) -> bool:
        """RTSP/USB: False = solo drenar con grab(); True = toca retrieve()."""
        if self._periodo_ticks <= 0:
            return True
        if cv2.getTickCount() < self._next_due_tick:
            return False
        self._next_due_tick = cv2.getTickCount() + self._periodo_ticks
        return True

    def _toca_capturar_snap(self) -> bool:
        """SNAP: False = esperar; True = hacer HTTP GET."""
        if self._periodo_ticks <= 0:
            return True
        if cv2.getTickCount() < self._next_due_tick:
            return False
        self._next_due_tick = cv2.getTickCount() + self._periodo_ticks
        return True

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
                    if not self.cap.isOpened():
                        raise cv2.error("Fallo socket RTSP")
                    for _ in range(self.warmup_frames):
                        self.cap.grab()
                    self._configurar_buffer(self.cap)
                    self._next_due_tick = cv2.getTickCount()  # reinicia reloj
                    logging.info("Stream RTSP estabilizado.")

                if not self.cap.grab():
                    raise cv2.error("Grab fallido")

                if not self._toca_capturar_stream():
                    continue

                ret, frame = self.cap.retrieve()
                if ret and frame is not None and frame.size > 0:
                    self._publicar_frame(frame)

            except (cv2.error, Exception) as e:
                logging.error(f"Error RTSP: {e}. Cooldown activo.")
                self._hard_reset_resources()
                time.sleep(self.reintento_seg)

    def _usb_loop(self):
        while self.is_running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = self._abrir_usb()
                    if self.cap is None:
                        raise cv2.error("No se pudo abrir camara USB")
                    if not self._preparar_usb(self.cap):
                        self.cap.release()
                        self.cap = None
                        raise cv2.error("Warmup USB fallido: sin frame valido")
                    self._configurar_buffer(self.cap)
                    self._next_due_tick = cv2.getTickCount()  # reinicia reloj
                    logging.info("Hardware USB estabilizado.")

                if not self.cap.grab():
                    raise cv2.error("Lectura USB fallida")

                if not self._toca_capturar_stream():
                    continue

                ret, frame = self.cap.retrieve()
                if ret and frame is not None and frame.size > 0:
                    self._publicar_frame(frame)

            except (cv2.error, Exception) as e:
                logging.error(f"Error USB: {e}. Cooldown activo.")
                self._hard_reset_resources()
                time.sleep(self.reintento_seg)

    def _snap_loop(self):
        while self.is_running:
            try:
                if not self._toca_capturar_snap():
                    time.sleep(0.001)
                    continue

                response = requests.get(self.snap_url, timeout=self.http_timeout_s)
                if response.status_code != 200:
                    raise requests.RequestException(f"HTTP {response.status_code}")

                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    self._publicar_frame(frame)

            except (requests.RequestException, Exception) as e:
                logging.error(f"Error Snap HTTP: {e}. Cooldown activo.")
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
