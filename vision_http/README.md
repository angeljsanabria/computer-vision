# vision_http

API HTTP de estado del pipeline de vision. **FastAPI** expone los endpoints; **uvicorn** sirve la app en un **hilo dedicado** dentro del mismo proceso que `main_mov.py`. No hace falta lanzar uvicorn por separado.

El pipeline publica un snapshot en memoria (`vision_store`) al final de cada frame; la API solo lo lee.

## Dependencias

```bash
# fastapi uvicorn
pip install -r requirements.txt
```

Poner la variable de entorno `ENABLE_ENDPOINT=true` (ver `requirements.txt`).
O modificar el `configs/settings.py` correspondiente.

## Configuracion

Variables de entorno (definidas en `configs/settings.py`):

| Variable | Default | Descripcion |
|----------|---------|-------------|
| `ENABLE_ENDPOINT` | `true` | Activa la API al arrancar `main_mov.py` |
| `HTTP_API_HOST` | `0.0.0.0` | Bind del servidor |
| `HTTP_API_PORT` | `8008` | Puerto HTTP |

## Endpoints

### `GET /health`

Comprobacion basica de que el servidor responde.

**Respuesta 200:**

```json
{"ok": true}
```

### `GET /api/v1/vision-status`

Estado publico del pipeline (ultimo frame procesado).

**Servidor caido o inaccesible:** este endpoint no devuelve JSON de error. Si `main_mov.py` no esta corriendo, `ENABLE_ENDPOINT=false`, el puerto esta mal o hay corte de red, el cliente obtiene **timeout** o **error de conexion** (p. ej. `curl: (7) Failed to connect`, `curl: (28) Operation timed out`, `ConnectionRefusedError` en Python). El consumidor debe configurar su propio timeout y tratar esa respuesta como "API no disponible".

**Respuesta 200 — sin caras / IDLE:**

```json
{
  "status": "NO_FACE_DETECTION",
  "person_id": null,
  "name": null,
  "face_count": 0,
  "refresh_remaining_s": 0,
  "updated_at": "2026-07-03T15:04:05.123456+00:00"
}
```

**Respuesta 200 — caras detectadas, sin identidad:**

```json
{
  "status": "FACES_DETECTED",
  "person_id": null,
  "name": null,
  "face_count": 1,
  "refresh_remaining_s": 0,
  "updated_at": "2026-07-03T15:04:10.654321+00:00"
}
```

**Respuesta 200 — cara reconocida (incluye retencion por timer):**

```json
{
  "status": "FACE_RECOGNIZED",
  "person_id": "001",
  "name": "Juan Perez",
  "face_count": 1,
  "refresh_remaining_s": 42,
  "updated_at": "2026-07-03T15:04:15.987654+00:00"
}
```

Valores posibles de `status`:

- `NO_FACE_DETECTION` — no face session / no faces
- `FACES_DETECTED` — faces visible, identity not confirmed
- `FACE_RECOGNIZED` — identity confirmed or retained by refresh timer

## Contrato API (referencia para consumidores)

Copia estable del JSON que devuelve la API. Las claves son **fijas**; no se agregan campos extra (`stale`, `error`, `fsm_state`, etc.).

### Base

| Item | Valor |
|------|-------|
| Protocolo | HTTP |
| Formato | JSON (`Content-Type: application/json`) |
| Version ruta | `/api/v1/` |
| Host/puerto | Configurable (`HTTP_API_HOST`, `HTTP_API_PORT`; default `0.0.0.0:8008`) |
| Autenticacion | Ninguna (v1) |
| Metodo | Solo `GET` |

### `GET /health`

| Campo | Tipo | Siempre | Descripcion |
|-------|------|---------|-------------|
| `ok` | `boolean` | si | `true` si el servidor HTTP responde |

```json
{
  "ok": true
}
```

### `GET /api/v1/vision-status`

Respuesta **200** con objeto plano. Siempre incluye **exactamente** estas 6 claves:

| Campo | Tipo JSON | Nullable | Descripcion |
|-------|-----------|----------|-------------|
| `status` | `string` (enum) | no | Estado de negocio publico |
| `person_id` | `string` | si | ID en galeria; `null` si no hay identidad confirmada |
| `name` | `string` | si | Display name for the person; `null` if no confirmed identity |
| `face_count` | `integer` | no | Cantidad de caras detectadas en el ultimo frame (`>= 0`) |
| `refresh_remaining_s` | `integer` | no | Segundos restantes de retencion del MATCH; entero truncado (`>= 0`) |
| `updated_at` | `string` | no | Timestamp UTC ISO 8601 del ultimo snapshot publicado |

#### Enum `status`

| Valor | Significado |
|-------|-------------|
| `NO_FACE_DETECTION` | No faces in frame, FSM in IDLE, or initial state |
| `FACES_DETECTED` | Faces visible; identity not yet confirmed |
| `FACE_RECOGNIZED` | Identity confirmed or retained by refresh timer |

#### Reglas por `status`

| `status` | `person_id` | `name` | `face_count` | `refresh_remaining_s` |
|----------|-------------|----------|--------------|------------------------|
| `NO_FACE_DETECTION` | `null` | `null` | `0` | `0` |
| `FACES_DETECTED` | `null` | `null` | `>= 1` | `0` |
| `FACE_RECOGNIZED` | `string` | `string` | `>= 0` | `>= 0` (segundos de retencion activos) |

Notas:

- `FACE_RECOGNIZED` solo mientras la FSM esta en `FACE_RECOGNIZED` (retencion activa).
- Durante retencion, `face_count` puede ser `0` si ese frame no corrio RetinaFace (cooldown); la identidad y el timer siguen vigentes.
- `refresh_remaining_s` es `int()` del timer interno (trunca decimales, nunca negativo).
- En `FACE_RECOGNIZED`, `refresh_remaining_s > 0` indica retencion del ultimo MATCH aunque el rostro actual no re-matchee o no haya deteccion en ese frame.
- `updated_at` ejemplo: `"2026-07-03T15:04:05.123456+00:00"` (UTC, con offset `+00:00`).

#### Ejemplos completos (copiar/pegar)

**Sin deteccion:**

```json
{
  "status": "NO_FACE_DETECTION",
  "person_id": null,
  "name": null,
  "face_count": 0,
  "refresh_remaining_s": 0,
  "updated_at": "2026-07-03T15:04:05.123456+00:00"
}
```

**Caras sin identidad:**

```json
{
  "status": "FACES_DETECTED",
  "person_id": null,
  "name": null,
  "face_count": 2,
  "refresh_remaining_s": 0,
  "updated_at": "2026-07-03T15:04:10.654321+00:00"
}
```

**Identidad confirmada o retenida:**

```json
{
  "status": "FACE_RECOGNIZED",
  "person_id": "001",
  "name": "Juan Perez",
  "face_count": 1,
  "refresh_remaining_s": 42,
  "updated_at": "2026-07-03T15:04:15.987654+00:00"
}
```

#### Errores y disponibilidad

| Situacion | Comportamiento esperado del cliente |
|-----------|-------------------------------------|
| Servidor apagado / puerto cerrado | Error de conexion o timeout; **no** hay JSON |
| `ENABLE_ENDPOINT=false` | Mismo caso: conexion rechazada |
| Respuesta HTTP != 200 | Tratar como fallo (no es parte del contrato v1 normal) |
| Campo desconocido en JSON | Ignorar (v1 no envia extras; reservado para versiones futuras) |

Recomendacion: timeout de cliente 2–5 s y polling cada 500 ms–1 s segun latencia aceptable.

#### Esquema JSON (JSON Schema draft 2020-12)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.local/computer-vision/vision-status/v1",
  "title": "VisionStatusResponse",
  "type": "object",
  "additionalProperties": false,
  "required": [
    "status",
    "person_id",
    "name",
    "face_count",
    "refresh_remaining_s",
    "updated_at"
  ],
  "properties": {
    "status": {
      "type": "string",
      "enum": [
        "NO_FACE_DETECTION",
        "FACES_DETECTED",
        "FACE_RECOGNIZED"
      ]
    },
    "person_id": {
      "type": ["string", "null"]
    },
    "name": {
      "type": ["string", "null"]
    },
    "face_count": {
      "type": "integer",
      "minimum": 0
    },
    "refresh_remaining_s": {
      "type": "integer",
      "minimum": 0
    },
    "updated_at": {
      "type": "string",
      "format": "date-time"
    }
  }
}
```

#### Tipos de referencia (implementacion cliente)

Python (stdlib + typing):

```python
from dataclasses import dataclass
from typing import Literal

VisionStatus = Literal[
    "NO_FACE_DETECTION",
    "FACES_DETECTED",
    "FACE_RECOGNIZED",
]

@dataclass
class VisionStatusResponse:
    status: VisionStatus
    person_id: str | None
    name: str | None
    face_count: int
    refresh_remaining_s: int
    updated_at: str
```

TypeScript:

```typescript
type VisionStatus =
  | "NO_FACE_DETECTION"
  | "FACES_DETECTED"
  | "FACE_RECOGNIZED";

interface VisionStatusResponse {
  status: VisionStatus;
  person_id: string | null;
  name: string | null;
  face_count: number;
  refresh_remaining_s: number;
  updated_at: string; // ISO 8601 UTC
}
```

## Testing con curl

Con el pipeline corriendo y la API activa:

```bash
curl -s http://127.0.0.1:8008/health
curl -s http://127.0.0.1:8008/api/v1/vision-status | jq
```

Desde otra maquina (host = IP de la placa):

```bash
# Si la ip es: 192.168.1.50
curl -s http://192.168.1.50:8008/api/v1/vision-status
```

PowerShell:

```powershell
curl http://127.0.0.1:8008/health
curl http://127.0.0.1:8008/api/v1/vision-status
```

## Deploy

### Modelo de ejecucion

```
main_mov.py  (proceso unico)
├── hilo principal  → pipeline (camara, MOG2, FSM, inferencia)
└── hilo H_VisionAPI → uvicorn + FastAPI (vision_http)
```

**No** ejecutar `uvicorn` manualmente antes del script. `main_mov.py` llama a `start_api_thread()` tras `validar_todo()` si `ENABLE_ENDPOINT=true`.

### Cierre (Ctrl+C, salida del bucle, excepcion)

Al terminar el bucle, el `finally` de `main_mov.py` llama a `stop_api_thread()` **antes** de liberar modelos RKNN y la camara:

1. `server.should_exit = True` — apagado ordenado de uvicorn/FastAPI
2. `thread.join(timeout=5s)` — espera a que suelte el puerto
3. Luego `_release_runtime()` (RetinaFace, MobileFaceNet) y `capture.stop()`

No hace falta matar uvicorn a mano ni un segundo comando al reiniciar. Tras un cierre limpio el puerto queda libre para volver a ejecutar `python WIP/main_mov.py`.

Si el hilo no termina en 5 s, se loguea un warning; el proceso sigue liberando el resto del hardware.

### Arranque manual (RK3568 o PC)

Desde la raiz del repo:

```bash
export ENABLE_ENDPOINT=true
export HTTP_API_HOST=0.0.0.0
export HTTP_API_PORT=8008
# resto de vars del pipeline (CONFIG_MODO, INFERENCE_BACKEND, etc.)

python WIP/main_mov.py
```

La API queda disponible en cuanto arranca el proceso; el JSON refleja el estado real cuando el bucle procesa frames.

### systemd (resumen)

Un solo servicio que ejecuta `main_mov.py`. No crear unidad aparte para uvicorn. Exponer el puerto en firewall si aplica.

### Docker / docker-compose

Mismo principio: **un contenedor, un proceso** (`main_mov.py`). Publicar el puerto de la API.

Ejemplo orientativo (ajustar imagen y variables al entorno RK3568):

```yaml
# docker-compose.yml (ejemplo)
services:
  vision:
    build: .
    restart: unless-stopped
    environment:
      ENABLE_ENDPOINT: "true"
      HTTP_API_HOST: "0.0.0.0"
      HTTP_API_PORT: "8008"
      CONFIG_MODO: "USB"              # RTSP | SNAP | USB (settings.py)
      USB_DEVICE_INDEX: "0"           # indice OpenCV; default 0 en settings.py
      INFERENCE_BACKEND: "rk3568"
      LOG_MODE: "prod"
    ports:
      - "8008:8008"
    # Solo si CONFIG_MODO=USB en Linux: pasar el nodo V4L2 que corresponda al indice.
    # USB_DEVICE_INDEX=0 -> /dev/video0, 1 -> /dev/video1, etc.
    # devices:
    #   - /dev/video0:/dev/video0
    # volumes: modelos, gallery, etc.
```

```dockerfile
# Dockerfile (ejemplo minimo)
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install fastapi uvicorn
COPY . .
CMD ["python", "WIP/main_mov.py"]
```

**Nota:** no usar `CMD uvicorn vision_http.routes:app ...` salvo que se desacople la API del pipeline (no es el diseno actual). El store en memoria solo se actualiza desde `main_mov.py`.

## Estructura del modulo

| Archivo | Rol |
|---------|-----|
| `types.py` | Snapshot JSON y estados publicos |
| `store.py` | Store thread-safe en memoria |
| `derive.py` | Deriva estado publico desde FSM + detecciones |
| `routes.py` | App FastAPI y rutas |
| `server.py` | Arranque y cierre uvicorn en hilo dedicado |
