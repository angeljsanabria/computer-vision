"""
Script de detección y anonimización de rostros usando MediaPipe y OpenCV.

Este script permite:
- Detectar caras en imágenes o en video desde la cámara.
- Dibujar un bounding box sobre cada cara detectada.
- Aplicar desenfoque (blur) sobre las caras para anonimizar.
- Mostrar información detallada de la detección, como puntuación y keypoints.

Uso:
------
Para procesar una imagen:
    python ocv_scr_anonymize.py --source image --img_name foto.jpg --details --box

Para usar la cámara (por ejemplo, la secundaria):
    python ocv_scr_anonymize.py --source camera --cam_index 1 --details --box
    python ocv_scr_anonymize.py --source camera --cam_index 1 --box --notblur

Opciones:
---------
--details   : Muestra información detallada de cada cara detectada.
--box       : Dibuja un bounding box sobre la cara y aplica blur.
--notblur   : No aplica desenfoque (aunque se haya pasado --box).
"""

import cv2
import os
import mediapipe as mp
import argparse
import numpy as np
from fontTools.misc.cython import returns

# keypoints face
relative_keypoints_dict = { 0 : "Ojo derecho", 1 : "Ojo izquierdo", 2 : "Nariz", 3 : "Boca",
                            4 : "Oreja derecha", 5 : "Oreja izquierda" }

# image const
IMG_OUT = 'out.jpg'
IMG_FOLDER = 'images'


def parse_args():
    parser = argparse.ArgumentParser(description="Detección de caras con MediaPipe")
    parser.add_argument("--source", type=str, choices=["image", "camera"], default="image",
                        help="Fuente de entrada: 'image' o 'camera'")
    parser.add_argument("--model", type=int, choices=[0, 1], default=0,
                        help="Modelo de MediaPipe: 0 para cerca, 1 para lejos")
    parser.add_argument("--img_name", type=str, default=None,
                        help="Nombre del archivo de imagen (requerido si source='image')")
    parser.add_argument("--cam_index", type=int, default=0,
                        help="Índice de la cámara (por defecto 0)")
    parser.add_argument("--details", action="store_true",
                        help="Activa prints de detalles")
    parser.add_argument("--box", action="store_true",
                        help="add bounding boxes y blur")
    parser.add_argument("--notblur", action="store_true",
                        help="Not apply blur")
    return parser.parse_args()

def _load_img(img_name: str | None = None):
    if img_name is None:
        # read image default
        IMG = 'lily2.jpg'
    else:
        IMG = img_name

    IMG_PATH = os.path.join('.', IMG_FOLDER, IMG) # '.' current dir
    return cv2.imread(IMG_PATH)


def _print_detail(detected):
    if detected is not None:
        print(f"Se detectaron {len(detected)} caras.")
        cara = 0
        for detection in detected:
            cara = cara + 1
            print(f"\r\n####### CARA {cara} ####### ")
            print("Confidence:", detection.score[0])
            bbox = detection.location_data.relative_bounding_box
            print("Bounding Box:")
            print(" - x:", bbox.xmin)
            print(" - y:", bbox.ymin)
            print(" - width:", bbox.width)
            print(" - height:", bbox.height)

            print("Keypoints:")
            for i, kp in enumerate(detection.location_data.relative_keypoints):
                print(f"#{i} Point {relative_keypoints_dict[i]}: ({kp.x}, {kp.y})")
                '''
                # Índice    Parte del rostro
                    0       Ojo derecho
                    1       Ojo izquierdo
                    2       Nariz
                    3       Boca
                    4       Oreja derecha
                    5       Oreja izquierda 
                '''
    else:
        print("No se detectaron caras.")

# mediapipe no devuelve coordenadas absolutas (en píxeles)
# sino coordenadas relativas, es decir, normalizadas en el rango [0.0, 1.0]
# por eso hay que multiplicarlas por el h, w de las dimensiones de la imagen (shape).
# Por que usa coordenadas relativas?
#   Independencia de resolución: Funciona igual en imágenes de 100×100 o 1000×1000.
#   Facilita escalado: Podés reescalar la imagen sin romper las posiciones de los puntos clave.
#   Interoperabilidad con modelos: Muchos modelos usan esta convención en visión por computadora.
def _process_bbox(_bbox,
                  img_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    #: mediapipe.framework.formats.location_data_pb2.RelativeBoundingBox,
    H, W, _ = img_shape
    x = int(_bbox.xmin * W)
    y = int(_bbox.ymin * H)
    w = int(_bbox.width * W)
    h = int(_bbox.height * H)
    return x, y, w, h




def main():
    args = parse_args()

    # detect face
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=args.model, min_detection_confidence=0.5)

    PRINT_DETALLE_DETECCION = args.details
    ADD_RECTANGLE_DETECCION = args.box
    ADD_BLUR = not args.notblur

    print(args)

    # Inicializo el modelo de deteccion de caras de mp
    #   model_selection 0 or 1:
    #       Si 0, detecta caras cerca de la camara (como 2 metros).
    #       Si 1, es mas lejos (como 5 metros)
    with mp_face_detection.FaceDetection(model_selection = 0, min_detection_confidence=0.5) as face_detection:

        if args.source == "image":
            img = _load_img(args.img_name)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Mediapipe trabaja en RGB, por eso transformo el color space
            out = face_detection.process(img_rgb)

            if PRINT_DETALLE_DETECCION:
                _print_detail(out.detections)

            if out.detections is not None:
                for detection in out.detections:
                    location_data = detection.location_data
                    bbox = location_data.relative_bounding_box

                    x, y, w, h = _process_bbox(bbox, img.shape)
                    if ADD_RECTANGLE_DETECCION:
                        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 10)

                    if ADD_BLUR:
                        # repesentacion en numpy de la img:
                        #   img[alto, ancho, canales] == img[filas, columnas, canales] == img[y, x, canales]
                        img[y : y+h, x : x+w, :] = cv2.blur(img[y : y+h, x : x+w, :], (20, 20), )  # Mas intenso (50, 50)

            # show image
            cv2.imshow('face', img)
            # Esperar hasta que se presione una tecla por 10 segs
            cv2.waitKey(10 * 1000)  # 0, indefinido.

            # Cerrar todas las ventanas
            cv2.destroyAllWindows()

        elif args.source == "camera":
            cap = cv2.VideoCapture(args.cam_index)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                out = face_detection.process(img_rgb)

                if PRINT_DETALLE_DETECCION:
                    _print_detail(out.detections)

                if out.detections:
                    for detection in out.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x, y, w, h = _process_bbox(bbox, frame.shape)

                        if ADD_RECTANGLE_DETECCION:
                            frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                        if ADD_BLUR:
                            # repesentacion en numpy de la img:
                            #   img[alto, ancho, canales] == img[filas, columnas, canales] == img[y, x, canales]
                            frame[y : y+h, x : x+w, :] = cv2.blur(frame[y : y+h, x : x+w, :], (20, 20), )  # Mas intenso (50, 50)

                cv2.imshow("Camera", frame)
                if cv2.waitKey(5) & 0xFF == 27:  # Escape
                    break

            cap.release()
            cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
