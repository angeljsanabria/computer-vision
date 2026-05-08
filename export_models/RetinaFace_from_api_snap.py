"""
RetinaFace (.rknn) en placa Rockchip: inferencia RKNNLite desde snapshots HTTP
(cmd=Snap) de camara IP.

El postproceso sigue en utils (letterbox + retinaface_dets_desde_rknn_outputs).

python3 RetinaFace_lite_from_api_snap.py
python3 RetinaFace_lite_from_api_snap.py --display
python3 RetinaFace_lite_from_api_snap.py --no-save

"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

try:
    from rknnlite.api import RKNNLite
except ImportError as e:
    raise SystemExit(
        "Instala RKNN-Toolkit-Lite2 en la placa (rknnlite). "
        "Ej.: pip3 install --user rknn_toolkit_lite2-....whl"
    ) from e


def _project_root_with_utils(start_dir: Path) -> Path:
    """Sube desde start_dir hasta encontrar ./utils/ (RetinaFace + image_utils)."""
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

from utils.aux_tools_retinaface import (
    RETINAFACE_INPUT_HEIGHT,
    RETINAFACE_INPUT_WIDTH,
    RETINAFACE_LETTERBOX_FILL,
    retinaface_dets_desde_rknn_outputs,
)

from utils.image_utils import letterbox_bgr

# --- Misma camara / Snap que model_api_snap.py (ajustar IP y credenciales) ---
USER_CAM = "angelcam"
PASS_CAM = "AngelCamara"
IP_CAM = "192.168.0.160"
RES_HIGH_CAM = "width=2560&height=1920"
RES_LOW_CAM = "width=640&height=480"
SNAP_HTTP_URL_RES_FULL = (
    f"http://{IP_CAM}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa"
    f"&user={USER_CAM}&password={PASS_CAM}&{RES_HIGH_CAM}"
)
SNAP_HTTP_URL_RES_LOW = (
    f"http://{IP_CAM}/cgi-bin/api.cgi?cmd=Snap&channel=0&rs=aaa"
    f"&user={USER_CAM}&password={PASS_CAM}&{RES_LOW_CAM}"
)
SNAP_HTTP_URL = SNAP_HTTP_URL_RES_LOW

HTTP_TIMEOUT_S = 10
MAX_FPS_ANALISIS = 2.0
LOG_CADA_CAPS = 10
TIME_SAVE_DETECTION = 1 * 60
FILE_DIR = "camara_snap"
FILE_BASE_NAME_IMG = "latest_retinaface_snap"

RETINAFACE_SCORE_PRE_NMS = 0.02
RETINAFACE_SCORE_DETECCION = 0.2


def obtener_frame_snap(url: str) -> np.ndarray | None:
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT_S, verify=False)
    except requests.exceptions.RequestException as e:
        print(f"Error de conexion Snap: {e}")
        return None

    if response.status_code != 200:
        print(f"Error HTTP Snap: {response.status_code}")
        return None

    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return frame


def construir_file_path_dia() -> str:
    dd_mm = time.strftime("%d_%m")
    file_name_img = f"{FILE_BASE_NAME_IMG}_{dd_mm}.jpg"
    return os.path.join(FILE_DIR, file_name_img)


def log_fps_analisis(frame_count: int, t0_tick: int, frame: np.ndarray) -> None:
    if frame_count % LOG_CADA_CAPS != 0:
        return
    ticks = cv2.getTickCount() - t0_tick
    dt = ticks / cv2.getTickFrequency()
    fps = frame_count / dt if dt > 0 else 0.0
    h, w = frame.shape[:2]
    print(f"[LOG] frame={frame_count} size={w}x{h} fps_aprox={fps:.2f}")


def main() -> None:
    _wip = Path(__file__).resolve().parent
    _repo = _wip.parent
    default_rknn = _repo / "models" / "RetinaFace_mobile320_2.rknn"

    parser = argparse.ArgumentParser(
        description="RetinaFace RKNN desde Snap HTTP (mismo flujo que model_api_snap)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(default_rknn),
        help="Ruta al .rknn",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Ventana OpenCV (q o ESC para salir).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar imagen con detecciones.",
    )
    args = parser.parse_args()

    snap_url = SNAP_HTTP_URL

    if not Path(args.model_path).is_file():
        raise SystemExit(f"No existe el modelo: {args.model_path}")

    rknn = RKNNLite()
    print("--> load_rknn", args.model_path)
    if rknn.load_rknn(args.model_path) != 0:
        raise SystemExit("load_rknn failed")
    print("--> init_runtime")
    if rknn.init_runtime() != 0:
        rknn.release()
        raise SystemExit("init_runtime failed")

    if args.display:
        print("Snap API lista. Modo display (q o ESC para salir).")
    else:
        print("Snap API lista. Sin display (Ctrl+C para salir).")

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

            frame = obtener_frame_snap(snap_url)
            if frame is None:
                continue

            frame_count += 1
            log_fps_analisis(frame_count, t0_tick, frame)

            img_height, img_width = frame.shape[:2]
            letterbox_img, lb_meta = letterbox_bgr(
                frame,
                (RETINAFACE_INPUT_WIDTH, RETINAFACE_INPUT_HEIGHT),
                RETINAFACE_LETTERBOX_FILL,
            )
            infer_rgb = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(infer_rgb, axis=0)

            outputs = rknn.inference(inputs=[input_tensor])
            if not outputs:
                continue

            dets = retinaface_dets_desde_rknn_outputs(
                outputs,
                img_width=img_width,
                img_height=img_height,
                aspect_ratio=lb_meta.aspect_ratio,
                offset_x=lb_meta.offset_x,
                offset_y=lb_meta.offset_y,
                score_deteccion=RETINAFACE_SCORE_DETECCION,
                score_pre_nms=RETINAFACE_SCORE_PRE_NMS,
            )

            n_faces = dets.shape[0]
            if n_faces > 0:
                msg_parts = [f"face({float(row[4]):.2f})" for row in dets]
                print("Detecciones: " + ", ".join(msg_parts))

            for data in dets:
                text = "{:.4f}".format(float(data[4]))
                di = list(map(int, data))
                cv2.rectangle(
                    frame, (di[0], di[1]), (di[2], di[3]), (0, 0, 255), 2
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
                cv2.imshow("retinaface rknn snap api", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
    finally:
        rknn.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
