# 10_img_en_C – Primera aproximacion a imagen en C

Carpeta del curso: tema "imagen como datos en C". Aqui se codearan los ejemplos (practica paso a paso); la guia es la clase.

## Contenido

- **`CLASE_primera_aproximacion_imagen_C.md`**: leccion completa (marco conceptual, fundamento, intuicion, practica, conexion, senales de comprension, perspectiva profesional). La practica se implementa en esta carpeta, paso a paso.
- **Ejemplos en C**: se van agregando segun avancemos la clase (cargar imagen, dimensiones, pixel, struct opcional, etc.). Se compilan con gcc en Mac/PC; se usa `stb_image.h` (descarga indicada en la clase).

## Listado del tema (que vemos en este bloque)

1. **Marco conceptual**: imagen en C = entrada de datos; buffer de bytes; archivo/camara como fuente; no casteo, si layout y (opcional) struct con metadata + puntero.
2. **Fundamento**: orden en memoria (fila, pixel, canal); formula del indice del pixel; papel de stb_image.
3. **Intuicion**: imagen como vector lineal; puntero al primer byte; relacion con "entrada del modelo" (TFLite, etc.).
4. **Practica**: programa que carga imagen (stb_image), imprime ancho/alto/canales y pixel (0,0); pasos 1–5 de la clase; codear en esta carpeta.
5. **Conexion**: buffers en proyectos reales; TFLite Micro en Fase 8.
6. **Senales de comprension**: como verificar que lo entendiste.
7. **Perspectiva profesional**: como se usa en proyectos reales.
8. **Mock de camara**: simular en la Mac la misma estructura que usaras en ESP32 con camara USB (struct Imagen + `camera_get_frame()` que en mock usa stb_image; en ESP32 usara el driver USB). Practica: struct + mock + main que "lee frame" igual que con camara real.

## Uso

1. Seguir la clase con el profesor/asistente (explicacion y guia en el chat).
2. Codear los ejemplos aqui (`10_img_en_C/`) paso a paso.
3. Compilar con gcc; descargar `stb_image.h` segun la clase.

## Siguiente paso en el curso

Despues de este tema, continuar con **Fase 1**: Fundamentos de Deep Learning y CNNs (`curso/progreso.md`, `curso/estructura.md`).
