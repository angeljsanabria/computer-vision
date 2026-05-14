import numpy as np

# Simulamos base de datos cargada en memoria al iniciar el programa
# db_embeddings.shape = (100, 128)
db_embeddings = np.load('rostros_registrados.npy')
db_names = ["Persona A", "Persona B", ...]

def identificar_rostro(embedding_detectado):
    # 1. Normalizar el vector detectado (Similitud de Coseno)
    embedding_detectado /= np.linalg.norm(embedding_detectado)
    
    # 2. COMPARACIÓN MATRICIAL (Aquí está el truco profesional)
    # Multiplicamos el vector (1, 128) por la matriz (128, 100)
    # Esto nos da 100 distancias en una sola operación de CPU/NPU
    similitudes = np.dot(db_embeddings, embedding_detectado)
    
    # 3. Obtener el índice del valor más alto
    idx_max = np.argmax(similitudes)
    score = similitudes[idx_max]
    
    # 4. Umbral de confianza (Threshold)
    # En MobileFaceNet, un score > 0.4 - 0.6 suele ser positivo
    if score > 0.5:
        return db_names[idx_max], score
    return "Desconocido", score

# Averigurar si efectivamente usa la NPU para la multiplicacion

