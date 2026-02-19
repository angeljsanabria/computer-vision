"""
Utilidad para visualizar camaras en pantalla completa - Ideal para presentaciones.

Descripcion:
-----------
Este script detecta automaticamente que camaras estan disponibles en el sistema
y permite navegar entre ellas presionando teclas. Es ideal para:
- Mostrar tu celular en pantalla completa durante videollamadas (Meet, Zoom, Teams)
- Detectar que indice tiene tu camara cuando no lo sabes
- Verificar que camaras funcionan correctamente
- Presentaciones donde necesitas mostrar contenido desde una camara externa

Ejecucion:
---------
python utils/util_cam_viewer.py

Controles:
---------
- 'n' o 'N': Cambiar a la siguiente camara disponible
- 'q' o 'Q': Cerrar y salir del script
- 'p' o 'P': Cambiar a la camara anterior (previous)
- 'r' o 'R': Rotar la imagen 90 grados a la izquierda

Parametros:
----------
- Modifica detectar_camaras_disponibles(5) para cambiar el numero maximo de
  camaras a detectar
"""
import sys
from pathlib import Path

# Anadir raiz del proyecto al path para que "from utils..." funcione al ejecutar desde finals/
_raiz = Path(__file__).resolve().parent.parent
if str(_raiz) not in sys.path:
    sys.path.insert(0, str(_raiz))

import cv2
import numpy as np
from utils.image_utils import rotar_frame

def detectar_camaras_disponibles(max_devices=5):
    """
    Detecta que camaras estan disponibles en el sistema.
    
    Returns:
        list: Lista de indices de camaras disponibles
    """
    camaras_disponibles = []
    
    for i in range(max_devices):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Intentar leer un frame para verificar que realmente funciona
            ret, _ = cap.read()
            if ret:
                camaras_disponibles.append(i)
        cap.release()
    
    return camaras_disponibles

def obtener_info_camara(cam_idx):
    """
    Obtiene informacion tecnica de una camara.
    
    Returns:
        dict: Diccionario con informacion de la camara
    """
    cap = cv2.VideoCapture(cam_idx)
    info = {}
    
    if cap.isOpened():
        # Informacion basica
        info['ancho'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info['alto'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        
        # Backend usado
        backend = int(cap.get(cv2.CAP_PROP_BACKEND))
        backend_names = {
            0: "CAP_ANY",
            200: "CAP_V4L2",      # Linux
            700: "CAP_DSHOW",     # Windows DirectShow
            800: "CAP_MSMF",      # Windows Media Foundation
            1200: "CAP_AVFOUNDATION"  # macOS
        }
        info['backend'] = backend_names.get(backend, f"Unknown ({backend})")
        
        # Formato de video (FOURCC)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        info['fourcc'] = fourcc if fourcc.isprintable() else "N/A"
        
        # Propiedades adicionales (pueden no estar disponibles)
        info['brillo'] = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        info['contraste'] = cap.get(cv2.CAP_PROP_CONTRAST)
        info['saturacion'] = cap.get(cv2.CAP_PROP_SATURATION)
        
    cap.release()
    return info

def ajustar_frame_manteniendo_aspect_ratio(frame, max_ancho, max_alto):
    """
    Ajusta el frame manteniendo el aspect ratio original.
    Agrega barras negras (letterboxing/pillarboxing) si es necesario.
    
    Nota sobre rendimiento:
    - cv2.copyMakeBorder(): Mas rapido (optimizado en C++), menos codigo
    - numpy manual: Mas control, mas legible para entender el proceso
    
    Args:
        frame: Frame de video a ajustar
        max_ancho: Ancho maximo de la ventana
        max_alto: Alto maximo de la ventana
    
    Returns:
        Frame ajustado con barras negras si es necesario
    """
    h, w = frame.shape[:2]
    
    # Calcular el factor de escala para que quepa en la ventana
    escala_ancho = max_ancho / w
    escala_alto = max_alto / h
    escala = min(escala_ancho, escala_alto)  # Usar la escala mas pequena para que quepa
    
    # Nuevas dimensiones manteniendo aspect ratio
    nuevo_ancho = int(w * escala)
    nuevo_alto = int(h * escala)
    
    # Redimensionar manteniendo aspect ratio
    frame_redimensionado = cv2.resize(frame, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
    
    # Crear imagen negra del tamano de la ventana usando numpy
    frame_final = np.zeros((max_alto, max_ancho, 3), dtype=np.uint8)
    
    # Calcular posicion para centrar la imagen
    y_offset = (max_alto - nuevo_alto) // 2
    x_offset = (max_ancho - nuevo_ancho) // 2
    
    # Colocar la imagen redimensionada en el centro usando indexacion numpy
    frame_final[y_offset:y_offset + nuevo_alto, x_offset:x_offset + nuevo_ancho] = frame_redimensionado
    
    return frame_final

def mostrar_info_camaras(camaras_disponibles):
    """Muestra informacion de las camaras detectadas."""
    print("=" * 60)
    print("CAMARAS DETECTADAS")
    print("=" * 60)
    if camaras_disponibles:
        for idx, cam_idx in enumerate(camaras_disponibles):
            info = obtener_info_camara(cam_idx)
            print(f"  [{idx}] Camara indice {cam_idx}")
            print(f"      Resolucion: {info.get('ancho', 'N/A')}x{info.get('alto', 'N/A')}")
            print(f"      FPS: {info.get('fps', 'N/A'):.2f}" if info.get('fps', 0) > 0 else "      FPS: N/A")
            print(f"      Backend: {info.get('backend', 'N/A')}")
            if info.get('fourcc') and info.get('fourcc') != 'N/A':
                print(f"      Formato: {info.get('fourcc', 'N/A')}")
            print()
        print("=" * 60)
        print(f"Total: {len(camaras_disponibles)} camara(s) disponible(s)")
        print("=" * 60)
    else:
        print("No se encontraron camaras disponibles")
        print("=" * 60)

def testear_camaras():
    """Funcion principal que permite navegar entre camaras disponibles."""
    # Detectar camaras disponibles
    camaras_disponibles = detectar_camaras_disponibles(5)
    
    if not camaras_disponibles:
        print("No se encontraron camaras disponibles.")
        return
    
    # Mostrar informacion
    mostrar_info_camaras(camaras_disponibles)
    
    # Indice actual en la lista de camaras disponibles
    camara_actual_idx = 0
    cap = None
    rotacion_actual = 0  # 0, 90, 180, 270 grados
    
    print("\nControles:")
    print("  'n' o 'N': Siguiente camara")
    print("  'p' o 'P': Camara anterior")
    print("  'r' o 'R': Rotar imagen 90 grados a la izquierda")
    print("  'q' o 'Q': Salir")
    print("\nMostrando camara...")
    
    # Nombre fijo de la ventana para reutilizarla (redimensionable)
    window_name = "Camaras Disponibles"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Establecer tamaño inicial de la ventana (puede ser redimensionada arrastrando)
    cv2.resizeWindow(window_name, 800, 600)
    
    while True:
        # Obtener el indice real de la camara actual
        camara_real_idx = camaras_disponibles[camara_actual_idx]
        
        # Abrir la camara si no esta abierta o si cambio
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(camara_real_idx)
            if not cap.isOpened():
                print(f"Error: No se pudo abrir la camara {camara_real_idx}")
                break
            print(f"\n[Camara {camara_actual_idx + 1}/{len(camaras_disponibles)}] "
                  f"Indice {camara_real_idx} - Presiona 'n' para siguiente, 'q' para salir")
        
        # Leer frame
        success, frame = cap.read()
        if not success:
            print(f"Error: No se pudo leer el frame de la camara {camara_real_idx}")
            break
        
        # Aplicar rotacion si es necesario
        if rotacion_actual != 0:
            frame = rotar_frame(frame, rotacion_actual)
        
        # Obtener tamano actual de la ventana
        try:
            ventana_ancho = int(cv2.getWindowImageRect(window_name)[2])
            ventana_alto = int(cv2.getWindowImageRect(window_name)[3])
        except:
            # Si no se puede obtener, usar tamano por defecto
            ventana_ancho = 800
            ventana_alto = 600
        
        # Ajustar frame manteniendo aspect ratio (evita deformacion)
        frame_ajustado = ajustar_frame_manteniendo_aspect_ratio(frame, ventana_ancho, ventana_alto)
        
        # Actualizar el titulo de la ventana (usando setWindowTitle para cambiar el titulo sin crear nueva ventana)
        cv2.setWindowTitle(window_name, f"Camara {camara_actual_idx + 1}/{len(camaras_disponibles)} -- Presiona 'n' para la siguiente -- o Q para salir")
        cv2.imshow(window_name, frame_ajustado)
        
        # Leer tecla presionada
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('n') or key == ord('N'):
            # Siguiente camara
            camara_actual_idx = (camara_actual_idx + 1) % len(camaras_disponibles)
            rotacion_actual = 0  # Resetear rotacion al cambiar de camara
            if cap is not None:
                cap.release()
            cap = None  # Forzar reapertura
        elif key == ord('r') or key == ord('R'):
            # Rotar 90 grados a la izquierda
            rotacion_actual = (rotacion_actual - 90) % 360
            print(f"Rotacion: {rotacion_actual} grados")
            
        elif key == ord('p') or key == ord('P'):
            # Camara anterior
            camara_actual_idx = (camara_actual_idx - 1) % len(camaras_disponibles)
            rotacion_actual = 0  # Resetear rotacion al cambiar de camara
            if cap is not None:
                cap.release()
            cap = None  # Forzar reapertura
    
    # Limpiar recursos
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\nCerrado correctamente.")

if __name__ == "__main__":
    testear_camaras()

