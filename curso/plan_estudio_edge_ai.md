# Plan de Estudio: Lead Edge AI / Embedded Systems Engineer

Este documento contiene un plan de estudio estructurado y un plan de trabajo enfocado en la arquitectura, la resiliencia y la integracion de sistemas en el mundo real, diseñado para prepararte para el rol de Lead Edge AI / Embedded Systems Engineer.

## Modulo 1: Fundamentos de Sistemas Embebidos y Linux Avanzado (Semanas 1-4)

El objetivo es dominar el sistema operativo base y la creacion de servicios resilientes para produccion.

*   **Linux Systemd Avanzado:**
    *   Creacion de unit files (`.service`, `.timer`).
    *   Gestion de dependencias (`Requires`, `After`).
    *   Implementacion de Watchdogs de hardware y software para auto-recuperacion.
    *   Integracion con `journald` para logging centralizado y rotacion de logs.
*   **Python para Produccion:**
    *   Tipado estatico (`mypy`), linters (`ruff`, `flake8`) y testing (`pytest`).
    *   Manejo avanzado de concurrencia y asincronismo (`asyncio`, `multiprocessing`) para no bloquear los flujos de video.
    *   Diseño de arquitecturas tolerantes a fallos (patrones de reintentos, circuit breakers).
*   **Docker en el Edge (Recursos Restringidos):**
    *   Construcciones multi-etapa (multi-stage builds) para reducir drasticamente el tamaño de la imagen.
    *   Limitacion de recursos en contenedores (CPU shares, memory limits).
    *   Redes de contenedores y orquestacion ligera (Docker Compose) en dispositivos ARM.

## Modulo 2: Ecosistema de Hardware Edge y Yocto Project (Semanas 5-8)

Cubre la capa mas cercana al hardware, desde la seleccion hasta la compilacion del sistema operativo.

*   **Plataformas Hardware:**
    *   Arquitectura de NVIDIA Jetson (Nano, Orin NX) y Raspberry Pi Compute Module (CM4/CM5).
    *   Conceptos de diseño termico, consumo de energia (power envelopes) y ciclos de trabajo (duty cycles).
*   **Yocto Project:**
    *   Conceptos basicos: Poky, BitBake, capas (layers) y recetas (recipes).
    *   Creacion de una distribucion Linux minima y personalizada para una Raspberry Pi o Jetson.
    *   Integracion de paquetes personalizados (Python, Docker) en la imagen de Yocto.

## Modulo 3: AWS IoT y Gestion de Flotas a Escala (Semanas 9-12)

El despliegue y monitoreo remoto es el corazon de este puesto.

*   **AWS IoT Core:**
    *   Aprovisionamiento de dispositivos (Fleet Provisioning) y gestion de certificados X.509.
    *   Uso de Device Shadows (estado reportado vs. estado deseado) para configuracion remota.
    *   Pipelines de telemetria (MQTT a AWS Kinesis o DynamoDB).
*   **AWS IoT Greengrass V2:**
    *   Arquitectura de Greengrass y despliegue de componentes personalizados (procesos Python y contenedores Docker).
    *   Comunicacion IPC (Inter-Process Communication) entre componentes locales.
*   **Actualizaciones OTA (Over-The-Air):**
    *   Estrategias de actualizacion segura (A/B partition updates).
    *   Mecanismos de rollback automatico ante fallos de conectividad o de inicio de servicios.

## Modulo 4: Pipelines de Video e Integraciones Retail (Semanas 13-15)

Preparacion para redes inestables y camaras que se desconectan en entornos reales.

*   **Protocolos de Video y Camaras IP:**
    *   Ingesta de video mediante RTSP (Real Time Streaming Protocol).
    *   Uso del protocolo ONVIF para descubrimiento automatico de camaras en la red local, control PTZ y gestion de perfiles.
*   **Procesamiento de Video Resiliente:**
    *   Uso de GStreamer y OpenCV (en Python/C++) para capturar frames.
    *   Logica de reconexion automatica, buffers conscientes de la latencia y descarte de frames corruptos.
*   **Integracion POS (Point of Sale):**
    *   Conceptos de integracion de datos transaccionales (sockets TCP/UDP, APIs REST, parseo de tramas serie/USB).
    *   Correlacion de marcas de tiempo (timestamps) entre eventos de video y transacciones de caja.

## Modulo 5: Optimizacion de IA en el Edge (Semanas 16-18)

Hacer que modelos existentes corran rapido y eficientemente en hardware restringido.

*   **Frameworks de Inferencia:**
    *   NVIDIA TensorRT: Conversion de modelos, optimizacion de grafos y ejecucion.
    *   ONNX Runtime: Configuracion de execution providers (CPU, CUDA, TensorRT).
*   **Tecnicas de Optimizacion:**
    *   Cuantizacion de modelos (de FP32 a FP16 o INT8) midiendo la perdida de precision.
    *   Afinidad de CPU/GPU y gestion de memoria unificada en dispositivos Jetson.
*   **Profiling (Medicion de rendimiento):**
    *   Uso de herramientas como `tegrastats` y `jtop` (Jetson Stats) para monitorear cuellos de botella termicos y de memoria.

---

## Proyecto Integrador Final: Edge Retail Vision Node (Semanas 19-24)

Construye este sistema para demostrar tu capacidad de integracion en un portafolio:

1.  **Hardware:** Usa una Raspberry Pi 4/5 o una Jetson Nano.
2.  **OS:** Compila una imagen minima con Yocto Project que incluya Docker y AWS Greengrass.
3.  **Video:** Escribe un servicio en Python (gestionado por systemd con watchdog) que descubra una camara IP simulada o real via ONVIF y consuma su flujo RTSP.
4.  **IA:** Implementa un modelo ligero de deteccion de objetos (ej. YOLOv8 formato ONNX/TensorRT) que procese el video buscando personas o productos.
5.  **Integracion POS:** Crea un script mock que emita datos de transacciones de ventas por un socket TCP. El sistema debe emparejar las detecciones visuales con las transacciones en tiempo real.
6.  **Nube:** Todo el sistema debe ser un componente de AWS Greengrass. La telemetria (temperatura del CPU, uso de RAM, conteo de detecciones) y los eventos de fraude/ventas deben enviarse a AWS IoT Core via MQTT.
7.  **Documentacion:** Escribe un "Runbook" detallado explicando la arquitectura, como desplegarlo y una seccion de "Troubleshooting".

---

## Recursos Recomendados

**Libros:**
*   *Mastering Embedded Linux Programming* (Chris Simmonds) - Esencial para Yocto y Linux embebido.
*   *Designing Data-Intensive Applications* (Martin Kleppmann) - Para entender como diseñar sistemas tolerantes a fallos.

**Cursos y Documentacion:**
*   **Yocto:** Tutoriales oficiales de Yocto Project y el canal de YouTube "LiveEmbedded".
*   **AWS:** Documentacion oficial de AWS IoT Core y Greengrass V2.
*   **NVIDIA:** Cursos gratuitos del "NVIDIA Deep Learning Institute" (DLI) enfocados en Jetson y TensorRT.

## Consejos para la Entrevista

1.  **Enfocate en los fallos:** Diseña pensando en que pasa cuando se corta el internet, la camara se apaga o la memoria se corrompe.
2.  **Mentalidad de Operaciones:** Estructura bien los logs (`journalctl`), el monitoreo remoto y las alertas para facilitar el soporte.
3.  **Conoce los limites del hardware:** Entiende las diferencias de procesamiento y disipacion de calor en entornos reales.
