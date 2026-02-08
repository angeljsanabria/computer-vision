"""
Utilidades para deteccion y manejo de camaras.

Este modulo proporciona funciones para detectar camaras disponibles
y obtener informacion tecnica de las mismas.
"""
import cv2


def detectar_camaras_disponibles(max_devices=5):
    """
    Detecta que camaras estan disponibles en el sistema.
    
    Args:
        max_devices: Numero maximo de dispositivos a verificar (default: 5)
    
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
    
    Args:
        cam_idx: Indice de la camara
    
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


def mostrar_info_camaras(camaras_disponibles):
    """
    Muestra informacion de las camaras detectadas en consola.
    
    Args:
        camaras_disponibles: Lista de indices de camaras disponibles
    """
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
