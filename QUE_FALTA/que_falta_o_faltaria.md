

## Problema cuando no encuentra la camara:

Watchdog de captura (soft), no del proceso
Estado explícito: OPENING → STREAMING → DEGRADED → RECOVERING.

Si N segundos sin frame válido (p. ej. 5–15 s, no 10 min):
release() del VideoCapture
reabrir el mismo dispositivo
warmup corto
Contador de recuperaciones + log con razón (open_fail, grab_fail, stale_frame).
El bucle de inferencia sigue vivo; solo se reinicia el hilo/recurso de captura.
Eso es lo estándar en sistemas edge (kioscos, ANPR, control de acceso).