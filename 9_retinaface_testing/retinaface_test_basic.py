"""
Prueba basica de deteccion de caras con RetinaFace (serengil/retinaface).
Ejecutar desde la raiz del proyecto con el venv que tenga retina-face instalado:
  source .venvRetinaFace/bin/activate
  python 9_retinaface_testing/retinaface_test_basic.py

RetinaFace usa TensorFlow; la API espera ruta de imagen. Para video se guarda
cada frame en un archivo temporal y se llama al detector.

Parametros (cambiar en el script):
- IMAGE: True = imagen (images/lily2.jpg), False = video (videos/lily.mp4).
- SHOW: True = ventana con resultado, False = solo guardar.
"""
import os
import sys
import cv2
import tempfile

# Ruta al directorio del script y a la raiz del proyecto
_DIR_SCRIPT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR_SCRIPT, ".."))

IMAGE = False
SHOW = True


def _draw_detections(frame, resp):
    """Dibuja cajas y landmarks sobre frame. resp = dict de RetinaFace.detect_faces."""
    if not resp or not isinstance(resp, dict):
        return frame
    out = frame.copy()
    for key, data in resp.items():
        if not isinstance(data, dict):
            continue
        area = data.get("facial_area")
        score = data.get("score", 0)
        landmarks = data.get("landmarks", {})
        if area is None or len(area) != 4:
            continue
        x1, y1, x2, y2 = [int(x) for x in area]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{score:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for name, (lx, ly) in landmarks.items():
            cv2.circle(out, (int(lx), int(ly)), 2, (0, 0, 255), -1)
    return out


def main():
    try:
        from retinaface import RetinaFace
    except ImportError:
        print("Instala retina-face: pip install retina-face")
        print("Recomendado: usar un venv dedicado (ej. .venvRetinaFace) con requirements-retinaface.txt")
        sys.exit(1)

    source = os.path.join(_ROOT, "images", "lily2.jpg") if IMAGE else os.path.join(_ROOT, "videos", "lily.mp4")
    if not os.path.isfile(source):
        print(f"No se encuentra: {source}")
        sys.exit(1)

    if IMAGE:
        resp = RetinaFace.detect_faces(source)
        frame = cv2.imread(source)
        if frame is None:
            print("No se pudo leer la imagen")
            sys.exit(1)
        out = _draw_detections(frame, resp)
        if SHOW:
            cv2.imshow("RetinaFace", out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            save_path = os.path.join(_ROOT, "runs", "retinaface_test.jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, out)
            print("Guardado:", save_path)
        print("Detecciones:", len(resp) if isinstance(resp, dict) else 0)
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("No se pudo abrir el video:", source)
            sys.exit(1)
        win = "RetinaFace"
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            temp_path = f.name
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(temp_path, frame)
                resp = RetinaFace.detect_faces(temp_path)
                out = _draw_detections(frame, resp)
                cv2.imshow(win, out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            if SHOW:
                cv2.destroyAllWindows()
        print("Fin video")

    print("Listo.")


if __name__ == "__main__":
    main()
