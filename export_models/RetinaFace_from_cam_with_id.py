"""
RetinaFace ONNX (USB) + MobileFaceNet ONNX: embedding de la mejor cara y comparacion
con un vector de referencia (.npy), p. ej. embeddings/angel1.npy.

Misma captura que RetinaFace_from_cam.py. Preproceso MobileFaceNet alineado con
face_embedding_from_image.py (RGB, ImageNet normalize, 112x112).

Constantes: MIN_SCORE_MEJOR_CARA_EMBEDDING, SIM_MIN_MATCH_VERIFICACION, FACE_CROP_MARGIN_FRAC.

Ejemplo:
  python export_models/RetinaFace_from_cam_with_id.py --display
  python export_models/RetinaFace_from_cam_with_id.py --display --ref-embedding embeddings/angel2.npy
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    raise SystemExit(
        "No se pudo importar onnxruntime.\n"
        f"  Python usado: {sys.executable}\n"
        "  Instala en ESE intérprete (con el venv activado): pip install onnxruntime\n"
        f"  Detalle: {e!r}"
    ) from e


def _project_root_with_utils(start_dir: Path) -> Path:
    cur = start_dir.resolve()
    for d in [cur, *cur.parents]:
        u = d / "utils"
        if u.is_dir() and (u / "aux_tools_retinaface.py").is_file():
            return d
    raise RuntimeError(
        "No se encontro carpeta 'utils' con aux_tools_retinaface.py. "
        f"Origen de busqueda: {start_dir}"
    )


ROOT = _project_root_with_utils(Path(__file__).resolve().parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.aux_tools_retinaface import (  # noqa: E402
    RETINAFACE_INPUT_HEIGHT,
    RETINAFACE_INPUT_WIDTH,
    RETINAFACE_LETTERBOX_FILL,
    retinaface_dets_desde_rknn_outputs,
)
from utils.image_utils import letterbox_bgr  # noqa: E402

# --- Misma politica de captura que RetinaFace_from_ip_cam.py (fuente = USB) ---
USAR_HILO_CAPTURA = True
TAMANO_BUFFER_CAMARA = 1
MAX_FPS_ANALISIS = 2.0
LOG_CADA_CAPS = 10
TIME_SAVE_DETECTION = 3 * 60
FILE_DIR = "camara_snap"
FILE_BASE_NAME_IMG = "latest_retinaface_usb"
REINTENTO_CAPTURA_SEG = 10

RETINAFACE_MEAN_BGR = np.array([104.0, 117.0, 123.0], dtype=np.float32)
RETINAFACE_SCORE_PRE_NMS = 0.02
RETINAFACE_SCORE_DETECCION = 0.2

# Mejor cara: score minimo RetinaFace para calcular embedding (mismo criterio que face_embedding_from_image).
MIN_SCORE_MEJOR_CARA_EMBEDDING = 0.90
# Similitud coseno minima (vectores L2=1) para considerar coincidencia con la referencia.
SIM_MIN_MATCH_VERIFICACION = 0.45
# Margen al recortar la caja antes de 112x112.
FACE_CROP_MARGIN_FRAC = 0.15

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
_MOBILEFACENET_HW = (112, 112)


def _preprocess_for_onnx(inp0_meta: ort.NodeArg, canvas_bgr: np.ndarray) -> np.ndarray:
    if canvas_bgr.dtype != np.uint8:
        canvas_bgr = canvas_bgr.astype(np.uint8)
    x32 = canvas_bgr.astype(np.float32)
    dims = inp0_meta.shape

    nchw = False
    nhwc = False
    if len(dims) == 4 and dims[1] == 3:
        nchw = True
    elif len(dims) == 4 and dims[-1] == 3:
        nhwc = True
    elif len(dims) == 4 and dims[1] == RETINAFACE_INPUT_WIDTH:
        nhwc = True
    else:
        nchw = True

    if nhwc:
        x32 -= RETINAFACE_MEAN_BGR.reshape(1, 1, 3)
        return np.expand_dims(x32, axis=0)

    if nchw:
        chw = np.transpose(x32, (2, 0, 1))
        feed = np.expand_dims(chw, axis=0)
        feed -= RETINAFACE_MEAN_BGR.reshape(1, 3, 1, 1)
        return feed

    raise RuntimeError("Forma de entrada ONNX no soportada: " + repr(dims))


def _bbox_crop_with_margin(
    det: np.ndarray, img_w: int, img_h: int, margin: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    mx = bw * margin
    my = bh * margin
    nx1 = int(np.floor(x1 - mx))
    ny1 = int(np.floor(y1 - my))
    nx2 = int(np.ceil(x2 + mx))
    ny2 = int(np.ceil(y2 + my))
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w - 1, nx2)
    ny2 = min(img_h - 1, ny2)
    if nx2 <= nx1 or ny2 <= ny1:
        return int(x1), int(y1), int(x2), int(y2)
    return nx1, ny1, nx2, ny2


def _crop_bgr_to_mobilefacenet_nchw(face_bgr: np.ndarray) -> np.ndarray:
    if face_bgr.size == 0:
        raise ValueError("recorte vacio")
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = cv2.resize(rgb, _MOBILEFACENET_HW, interpolation=cv2.INTER_LINEAR)
    norm = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
    chw = np.transpose(norm, (2, 0, 1))
    return np.expand_dims(chw.astype(np.float32), axis=0)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec.reshape(-1), ord=2))
    if n < 1e-12:
        return vec
    return (vec / n).astype(np.float32)


def configurar_buffer_camara(cap: cv2.VideoCapture) -> None:
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, TAMANO_BUFFER_CAMARA)
    except Exception:
        pass


def log_fps_analisis(frame_count: int, t0_tick: int, frame: np.ndarray) -> None:
    if frame_count % LOG_CADA_CAPS != 0:
        return
    ticks = cv2.getTickCount() - t0_tick
    dt = ticks / cv2.getTickFrequency()
    fps = frame_count / dt if dt > 0 else 0.0
    h, w = frame.shape[:2]
    print(f"[LOG] frame={frame_count} size={w}x{h} fps_aprox={fps:.2f}")


def construir_file_path_dia() -> str:
    dd_mm = time.strftime("%d_%m")
    file_name_img = f"{FILE_BASE_NAME_IMG}_{dd_mm}.jpg"
    return os.path.join(FILE_DIR, file_name_img)


def abrir_usb_con_calentamiento(device_index: int) -> cv2.VideoCapture | None:
    """Abre V4L/USB y descarta frames iniciales (misma idea que calentar RTSP)."""
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        cap.release()
        return None

    for _ in range(25):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size:
            configurar_buffer_camara(cap)
            return cap

    cap.release()
    return None


def esperar_primer_frame_grabber(
    grabber: "UltimoFrameCamara", timeout_seg: float = 2.0
) -> bool:
    t_ini = time.time()
    while (time.time() - t_ini) < timeout_seg:
        ok, _ = grabber.read_copy()
        if ok:
            return True
        time.sleep(0.05)
    return False


class UltimoFrameCamara:
    """
    Un solo hilo llama a cap.read(); el bucle principal copia el ultimo frame.
    Evita que mientras ONNX infiere se acumulen frames viejos en el buffer.
    """

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                with self._lock:
                    self._frame = None
                time.sleep(0.001)

    def read_copy(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join()
            self._thread = None


def main() -> None:
    _repo = ROOT
    default_onnx = _repo / "Retinaface-Models" / "RetinaFace_mobile320.onnx"
    default_mfn = _repo / "mobilenet_modelos" / "MobileFaceNet.onnx"
    default_ref = _repo / "embeddings" / "angel1.npy"

    parser = argparse.ArgumentParser(
        description="RetinaFace USB + MobileFaceNet: verificacion contra embedding .npy."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_onnx),
        help="Ruta al ONNX RetinaFace",
    )
    parser.add_argument(
        "--mobilefacenet-onnx",
        type=str,
        default=str(default_mfn),
        help="Ruta al ONNX MobileFaceNet (y .onnx.data si aplica)",
    )
    parser.add_argument(
        "--ref-embedding",
        type=str,
        default=str(default_ref),
        help="Embedding referencia .npy (mismo pipeline que face_embedding_from_image)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice de camara USB (0 por defecto).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Ventana OpenCV (q o ESC para salir).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar imagenes con detecciones.",
    )
    parser.add_argument(
        "--providers",
        type=str,
        default="CPUExecutionProvider",
        help="OnnxRuntime providers separados por coma.",
    )
    args = parser.parse_args()

    if not Path(args.model_path).is_file():
        raise SystemExit(f"No existe el ONNX RetinaFace: {args.model_path}")
    if not Path(args.mobilefacenet_onnx).is_file():
        raise SystemExit(f"No existe el ONNX MobileFaceNet: {args.mobilefacenet_onnx}")
    ref_path = Path(args.ref_embedding)
    if not ref_path.is_file():
        raise SystemExit(f"No existe el embedding referencia: {ref_path}")

    ref_vec = np.load(str(ref_path)).astype(np.float32).reshape(-1)
    if ref_vec.size != 128:
        raise SystemExit(
            f"Embedding referencia debe tener 128 valores, tiene {ref_vec.size}: {ref_path}"
        )
    ref_vec = _l2_normalize(ref_vec)

    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    session = ort.InferenceSession(args.model_path, providers=providers)
    inp0 = session.get_inputs()[0]
    input_name = inp0.name

    session_mfn = ort.InferenceSession(args.mobilefacenet_onnx, providers=providers)
    mfn_in = session_mfn.get_inputs()[0]
    mfn_in_name = mfn_in.name
    mfn_out_name = session_mfn.get_outputs()[0].name

    cap = abrir_usb_con_calentamiento(args.camera)
    while cap is None:
        print(
            f"[RETRY] No se pudo abrir/calentar USB cam {args.camera}. "
            f"Reintento en {REINTENTO_CAPTURA_SEG}s..."
        )
        time.sleep(REINTENTO_CAPTURA_SEG)
        cap = abrir_usb_con_calentamiento(args.camera)

    grabber: UltimoFrameCamara | None = None
    if USAR_HILO_CAPTURA:
        grabber = UltimoFrameCamara(cap)
        grabber.start()
        if not esperar_primer_frame_grabber(grabber):
            grabber.stop()
            cap.release()
            raise SystemExit(
                "Camara USB abierta pero el hilo no recibio el primer frame. "
                "Revisar permisos / indice --camera."
            )
        print("Captura en hilo auxiliar activa (menos latencia por buffer).")
    print(
        f"Referencia: {ref_path} | MobileFaceNet: {args.mobilefacenet_onnx} | "
        f"min_score_emb={MIN_SCORE_MEJOR_CARA_EMBEDDING} sim_match>={SIM_MIN_MATCH_VERIFICACION}"
    )

    if args.display:
        print("Camara USB lista. Modo display (q o ESC para salir).")
    else:
        print("Camara USB lista. Sin display (Ctrl+C para salir).")

    periodo_analisis_ticks = (
        int(cv2.getTickFrequency() / MAX_FPS_ANALISIS)
        if MAX_FPS_ANALISIS > 0
        else 0
    )
    next_due = cv2.getTickCount()
    frame_count = 0
    t0_tick = cv2.getTickCount()
    save_interval_ticks = int(TIME_SAVE_DETECTION * cv2.getTickFrequency())
    last_save_tick = cv2.getTickCount()

    try:
        while True:
            if periodo_analisis_ticks > 0:
                now_tick = cv2.getTickCount()
                if now_tick < next_due:
                    if args.display:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            break
                    time.sleep(0.001)
                    continue
                next_due = cv2.getTickCount() + periodo_analisis_ticks

            if grabber is not None:
                ok, frame = grabber.read_copy()
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                print(
                    f"[RETRY] Sin frame USB cam {args.camera}. "
                    f"Reintentando en {REINTENTO_CAPTURA_SEG}s..."
                )
                if grabber is not None:
                    grabber.stop()
                    grabber = None
                cap.release()

                while True:
                    if args.display:
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:
                            return
                    time.sleep(REINTENTO_CAPTURA_SEG)
                    cap = abrir_usb_con_calentamiento(args.camera)
                    if cap is not None:
                        if USAR_HILO_CAPTURA:
                            grabber = UltimoFrameCamara(cap)
                            grabber.start()
                            if esperar_primer_frame_grabber(grabber):
                                print("[RETRY] Camara USB reconectada. Captura en hilo reanudada.")
                                break
                            grabber.stop()
                            grabber = None
                            cap.release()
                            cap = None
                            print(
                                f"[RETRY] USB reabrio pero sin primer frame del hilo. "
                                f"Nuevo intento en {REINTENTO_CAPTURA_SEG}s..."
                            )
                        else:
                            print("[RETRY] Camara USB reconectada.")
                            break
                    print(
                        f"[RETRY] Fallo reapertura USB. Nuevo intento en {REINTENTO_CAPTURA_SEG}s..."
                    )
                continue

            frame_count += 1
            log_fps_analisis(frame_count, t0_tick, frame)

            img_height, img_width = frame.shape[:2]
            letterbox_img, lb_meta = letterbox_bgr(
                frame,
                (RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT),
                RETINAFACE_LETTERBOX_FILL,
            )
            tensor = _preprocess_for_onnx(inp0, letterbox_img)
            ort_outputs = session.run(None, {input_name: tensor})

            dets = retinaface_dets_desde_rknn_outputs(
                list(ort_outputs),
                img_width=img_width,
                img_height=img_height,
                aspect_ratio=lb_meta.aspect_ratio,
                offset_x=lb_meta.offset_x,
                offset_y=lb_meta.offset_y,
                score_deteccion=RETINAFACE_SCORE_DETECCION,
                score_pre_nms=RETINAFACE_SCORE_PRE_NMS,
            )

            n_faces = dets.shape[0]
            sim_display = "--"
            match_display = ""
            best_idx = -1
            if n_faces > 0:
                best_idx = int(np.argmax(dets[:, 4]))
                best = dets[best_idx]
                score_best = float(best[4])
                if score_best >= MIN_SCORE_MEJOR_CARA_EMBEDDING:
                    bx1, by1, bx2, by2 = _bbox_crop_with_margin(
                        best, img_width, img_height, FACE_CROP_MARGIN_FRAC
                    )
                    crop = frame[by1 : by2 + 1, bx1 : bx2 + 1].copy()
                    try:
                        feed_m = _crop_bgr_to_mobilefacenet_nchw(crop)
                        out_m = session_mfn.run(
                            [mfn_out_name], {mfn_in_name: feed_m}
                        )[0]
                        emb = _l2_normalize(
                            np.asarray(out_m, dtype=np.float32).reshape(-1)
                        )
                        sim_v = float(np.dot(ref_vec, emb))
                        sim_display = f"{sim_v:.3f}"
                        match_display = (
                            "MATCH"
                            if sim_v >= SIM_MIN_MATCH_VERIFICACION
                            else "NO_MATCH"
                        )
                    except Exception:
                        sim_display = "err"
                        match_display = "CROP"
                else:
                    sim_display = f"sc{score_best:.2f}"
                    match_display = "BAJO"

            if n_faces > 0:
                msg_parts = [f"face({float(row[4]):.2f})" for row in dets]
                extra = f" | ID sim={sim_display} {match_display}"
                print("Detecciones: " + ", ".join(msg_parts) + extra)

            for i, data in enumerate(dets):
                score_f = float(data[4])
                text = "{:.4f}".format(score_f)
                di = list(map(int, data))
                color = (0, 0, 255)
                thick = 2
                if i == best_idx and match_display == "MATCH":
                    color = (0, 200, 0)
                    thick = 3
                elif i == best_idx:
                    color = (0, 140, 255)
                    thick = 3
                cv2.rectangle(
                    frame, (di[0], di[1]), (di[2], di[3]), color, thick
                )
                cx, cy = di[0], di[1] + 12
                cv2.putText(
                    frame,
                    text,
                    (cx, cy),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5,
                    (255, 255, 255),
                )
                cv2.circle(frame, (di[5], di[6]), 1, (0, 0, 255), 5)
                cv2.circle(frame, (di[7], di[8]), 1, (0, 255, 255), 5)
                cv2.circle(frame, (di[9], di[10]), 1, (255, 0, 255), 5)
                cv2.circle(frame, (di[11], di[12]), 1, (0, 255, 0), 5)
                cv2.circle(frame, (di[13], di[14]), 1, (255, 0, 0), 5)

            if n_faces > 0:
                bar = f"{ref_path.name} sim={sim_display} {match_display}"
                cv2.rectangle(frame, (0, 0), (img_width, 36), (0, 0, 0), -1)
                cv2.putText(
                    frame,
                    bar,
                    (6, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            if n_faces > 0 and not args.no_save:
                now_save = cv2.getTickCount()
                if now_save - last_save_tick >= save_interval_ticks:
                    os.makedirs(FILE_DIR, exist_ok=True)
                    file_path = construir_file_path_dia()
                    if cv2.imwrite(file_path, frame):
                        print(f"[SAVE] {file_path}")
                        last_save_tick = now_save
                    else:
                        print(f"[SAVE] error al guardar: {file_path}")

            if args.display:
                cv2.imshow("RetinaFace + ID (USB)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        if grabber is not None:
            grabber.stop()
        cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
