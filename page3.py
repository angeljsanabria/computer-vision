import cv2

# Etiquetas personalizadas según el índice
cam_labels = {
    0: "iPhone (Continuity Camera)",
    1: "Webcam interna Mac",
    2: "Cam externa USB",
}

def testear_camaras(max_devices=5):
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if not cap.isOpened():
            print(f"Cam {i}: ❌ No disponible")
            continue

        label = cam_labels.get(i, f"Cam {i}")
        print(f"{label}: ✅ MOSTRANDO video... Presioná 'q' para cerrar.")

        while True:
            success, frame = cap.read()
            if not success:
                print(f"{label}: ⚠️ No se pudo leer el frame.")
                break
            cv2.imshow(label, frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

testear_camaras(3)

