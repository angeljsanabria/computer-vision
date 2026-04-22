# Curso de Computer Vision para Embedded Systems (curso_v2)

## Descripción

Este curso está diseñado para desarrolladores de sistemas embebidos que quieren aprender Computer Vision desde fundamentos hasta implementación en dispositivos embebidos.

**curso_v2** prioriza el hardware **MYIR MYD-LR3568-GK-B** (Rockchip RK3568, NPU, Edge IoT industrial). Los temas generales del CV siguen siendo validos; el despliegue y la inferencia acelerada se orientan a **RKNN** y al pipeline documentado en [edge_iot_myd](../edge_iot_myd/README.md).

## Hardware Disponible

| Dispositivo | Procesador/Microcontrolador | Tipo | Uso en el Curso |
|-------------|----------------------------|------|-----------------|
| **MYD-LR3568-GK-B** | Rockchip RK3568J (Cortex-A55), NPU ~1 TOPS, Mali-G52 | IPC Linux embebido (Debian/MYIR) | **Plataforma principal v2**: vision Edge IoT, RKNN, doble Ethernet, CAN/4G |
| **UNIHIKER M10** | Allwinner H6 (ARM Cortex-A53) | Microprocesador (Linux embebido) | Referencia / comparacion |
| **Raspberry Pi Zero 2W** | Broadcom BCM2710A1 (ARM Cortex-A53 Quad-core) | Microprocesador (Linux embebido) | Referencia / comparacion |
| **ESP32-P4** | RISC-V Dual-core 32-bit | Microcontrolador | Fases avanzadas: TensorFlow Lite Micro, TinyML |
| **Sony IMX179** | Sensor de imagen CMOS | Cámara/Sensor | Captura de imágenes para todas las placas |

## Objetivo Final

Aprender a implementar sistemas de visión por computadora en el hardware disponible usando herramientas como:
- **RKNN / RKNN Lite** (NPU Rockchip en MYD-LR3568-GK-B) y ciclo **ONNX → .rknn** (ver [edge_iot_myd](../edge_iot_myd/README.md))
- **TensorFlow Lite** / **ONNX Runtime** como referencia o fallback en CPU
- **Edge Impulse** (cuando encaje en el flujo pedagogico)
- **TensorFlow** (entrenamiento o prototipos en PC)

## Estructura del Curso

El curso está dividido en **11 fases** progresivas:

1. **Fase 0**: Fundamentos (✅ Completado)
2. **Fase 1**: Deep Learning y CNNs
3. **Fase 2**: Object Detection
4. **Fase 3**: Entrenamiento de Modelos
5. **Fase 4**: Embedded Systems - Hardware y Setup
6. **Fase 5**: OpenCV en Embedded
7. **Fase 6**: TensorFlow Lite
8. **Fase 7**: Edge Impulse
9. **Fase 8**: TensorFlow Lite Micro (ESP32-P4)
10. **Fase 9**: TensorFlow Completo en Linux Embebido
11. **Fase 11**: OCR (Optical Character Recognition) en Embedded
12. **Fase 10**: Proyectos Integrados y Optimización Avanzada

## Archivos

- **`estructura.md`**: Estructura completa del curso con todos los temas detallados
- **`progreso.md`**: Tabla de seguimiento con estados (TODO/WIP/DONE)
- **`README.md`**: Este archivo
- **[`../edge_iot_myd/`](../edge_iot_myd/README.md)**: Proyecto Edge IoT MYD (manifiestos, Docker RKNN-Toolkit2, scripts demo)
- **[`../Docu/MYD-LR3568_BSP_versiones.md`](../Docu/MYD-LR3568_BSP_versiones.md)**: Checklist BSP / versiones RKNN
- **`10_img_en_C/`** (en raiz del repo): Clase "Primera aproximacion a imagen en C"; ver `10_img_en_C/CLASE_primera_aproximacion_imagen_C.md` y `10_img_en_C/README.md`.

## Cómo Usar Este Curso

1. **Revisar la estructura**: Lee `estructura.md` para entender el camino completo
2. **Seguir el progreso**: Usa `progreso.md` para marcar temas completados
3. **Actualizar estados**: Cambia los estados en `progreso.md`:
   - 🔲 TODO → 🔄 WIP (cuando empieces)
   - 🔄 WIP → ✅ DONE (cuando termines)

## Flujo de Desarrollo

**IMPORTANTE**: El código sigue un flujo de desarrollo específico:

1. **Desarrollo en Computadora**: Todo el código se desarrolla y prueba PRIMERO en una computadora (PC)
   - Permite debuggear fácilmente
   - Permite probar conceptos sin limitaciones de hardware
   - El código debe funcionar correctamente en computadora antes de portar

2. **Port a Dispositivos Embebidos**: Una vez que funciona y se entiende en computadora, se adapta/porta a:
   - **MYD-LR3568-GK-B** (objetivo principal v2: RKNN en NPU, modelos versionados en `/opt/edge-models/...`)
   - UNIHIKER M10
   - Raspberry Pi Zero 2W
   - ESP32-P4

**Regla**: No saltar directamente a código para embedded sin haberlo probado y entendido primero en computadora.

## Enfoque del Curso

- **Teoría primero**: Entender el "por qué" antes del "cómo"
- **Práctica constante**: Cada fase incluye proyectos prácticos
- **Desarrollo iterativo**: Computadora primero, luego embedded
- **Enfoque embedded**: Siempre pensando en limitaciones de recursos
- **Iterativo**: Puedes avanzar y volver según necesidades

## Prerequisitos

- ✅ Conocimientos básicos de OpenCV (completado en Fase 0)
- ✅ Python intermedio
- ✅ Conocimientos básicos de C (para ESP32)
- ✅ Familiaridad con sistemas embebidos (recomendado)

## Clase en C (opcional): primera aproximacion a imagen

En **`10_img_en_C/`** esta la clase "Primera aproximacion a imagen en C". La leccion se sigue en el chat (te guio); los ejemplos se codearan en esa carpeta paso a paso. Ver `10_img_en_C/CLASE_primera_aproximacion_imagen_C.md` y `10_img_en_C/README.md`.

## Proximo Paso

**Iniciar Fase 1**: Fundamentos de Deep Learning y CNNs

---

## Notas

- Este curso es iterativo - podemos ajustar la estructura según tus necesidades
- Cada fase puede tener sub-proyectos que documentaremos
- El objetivo es llegar a implementaciones reales en hardware
