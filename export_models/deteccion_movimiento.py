"""
Deteccion de movimiento con camara USB (OpenCV MOG2).

Pensado para pipeline ligero (p. ej. RK3568): frame pequeno, ~2 FPS, sin sombras.
Tras detectar movimiento se puede llamar a un modelo pesado (RetinaFace, etc.).

Ejemplo:
  python export_models/deteccion_movimiento.py
  python export_models/deteccion_movimiento.py --camera 0 --movimiento_pixeles 800
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MOG2 + camara USB: movimiento para disparar etapa pesada."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Indice VideoCapture (USB suele ser 0)",
    )
    parser.add_argument("--ancho", type=int, default=320, help="Ancho procesamiento")
    parser.add_argument("--alto", type=int, default=240, help="Alto procesamiento")
    parser.add_argument("--history", type=int, default=20, help="Historial MOG2")
    parser.add_argument(
        "--var_threshold",
        type=int,
        default=40,
        help="varThreshold MOG2",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Frames de calibracion de fondo",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="FPS objetivo (aprox.). 2 -> periodo ~0.5 s",
    )
    parser.add_argument(
        "--movimiento_pixeles",
        type=int,
        default=1000,
        help="Umbral cv2.countNonZero(mask) para avisar movimiento",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(
            "No se abrio la camara indice {}. Prueba otro --camera.".format(args.camera)
        )

    procesar_wh = (args.ancho, args.alto)
    periodo_s = 1.0 / max(args.fps, 0.1)

    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=False,
    )

    print("Calibrando fondo...")
    for _ in range(args.warmup):
        ret, frame = cap.read()
        if ret:
            fgbg.apply(
                cv2.resize(frame, procesar_wh, interpolation=cv2.INTER_AREA),
                learningRate=0.5,
            )
        else:
            print("Warm-up: frame no leido.")
            break

    print("Sistema listo. Ctrl+C para salir.")

    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Fin de captura.")
                break

            small_frame = cv2.resize(
                frame, procesar_wh, interpolation=cv2.INTER_AREA
            )
            mask = fgbg.apply(small_frame)
            pixel_count = int(cv2.countNonZero(mask))

            if pixel_count > args.movimiento_pixeles:
                print(
                    "Movimiento ({:d} pix). Aqui modelo pesado (RetinaFace, etc.).".format(
                        pixel_count
                    )
                )

            elapsed = time.time() - t0
            espera = periodo_s - elapsed
            if espera > 0:
                time.sleep(espera)
    except KeyboardInterrupt:
        print("Interrupcion por teclado.")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
    sys.exit(0)
