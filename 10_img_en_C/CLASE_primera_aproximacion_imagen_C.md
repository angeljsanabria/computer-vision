# Clase: Primera aproximacion a imagen en C

Tema del curso: primera aproximacion a "imagen como datos" en C, sin OpenCV ni hardware. Se ejecuta en la Mac (o PC) con gcc. Objetivo: entender que en C una imagen en memoria es un buffer de bytes y como se relaciona con lo que ya viste en Python/NumPy, como base para mas adelante (Fase 8, ESP32-P4).

---

## 1. Marco conceptual (el que y el por que)

### Que problema resuelve esto

En Python usas OpenCV o NumPy: la imagen es un array (matriz de numeros). En C no hay eso por defecto: la imagen es un **bloque de memoria** (bytes) que alguien tiene que llenar leyendo el archivo (JPG, PNG, etc.). Esta clase te pone en ese punto: **como llega una imagen "a memoria" en C** y que forma tiene ese bloque (ancho, alto, canales, orden de los bytes). Sin eso, mas adelante no tendrias claro que es lo que alimentas a un modelo o lo que procesas en un microcontrolador.

### Que NO resuelve

- No hace vision por computadora (no hay filtros, deteccion ni redes). Solo **carga** un archivo y te da los bytes.
- No sustituye OpenCV en Python para desarrollar algoritmos; eso se sigue haciendo en PC con Python. Aqui el objetivo es **conceptos y C**, no un proyecto completo de vision.
- No usa hardware embebido; es 100 % local (gcc en Mac/PC) para que puedas compilar y probar sin placas.

### En que parte del pipeline esta

Es el paso **mas anterior** del pipeline: **entrada de datos**. Antes de cualquier procesamiento (blur, deteccion, inferencia), la imagen tiene que estar en memoria. En Python eso lo hace `cv2.imread()` o PIL; en C necesitas una libreria que decodifique JPG/PNG y te devuelva un puntero a bytes. Esta clase se queda ahi: obtener ese puntero y saber que significan los numeros (ancho, alto, canales, primer pixel).

### Diferencia con conceptos parecidos

- **Python + OpenCV**: `img = cv2.imread("foto.jpg")` te da un array NumPy; la imagen ya esta en memoria y puedes hacer `img[y, x]` o operaciones vectorizadas. En C no hay array "magico"; es un `unsigned char*` y tu sabes que los primeros `ancho*alto*canales` bytes son los pixeles (por ejemplo fila por fila, R G B R G B...).
- **ESP32-P4 / camara**: en el microcontrolador la imagen puede venir de un sensor (camara) o de un buffer; ese buffer tiene la misma idea (bytes en orden). Aqui practicamos con un archivo en disco para no depender de hardware.

---

## 2. Fundamento (el como por debajo)

No hay formulas de vision aqui; lo que hay es **como se representa la imagen en memoria**.

- **Archivo (JPG/PNG)**: es un archivo binario con cabeceras y datos comprimidos (o no). Para obtener pixeles en C hace falta **decodificar** ese formato (JPEG, PNG, etc.). No se hace a mano en esta clase; se usa una libreria.

- **Imagen en memoria (despues de cargar)**:
  - Secuencia de bytes: `unsigned char*` (o `uint8_t*`).
  - Orden tipico: **fila por fila**, y dentro de cada fila **pixel por pixel**; cada pixel tiene **canales** (ej. 3 = R, G, B en ese orden).
  - Ejemplo concreto: imagen 2x2, 3 canales. 12 bytes en total. Orden:  
    `[R00,G00,B00, R01,G01,B01, R10,G10,B10, R11,G11,B11]`.  
    El pixel (0,0) son los bytes 0,1,2; el (1,0) son 3,4,5; el (0,1) son 6,7,8; el (1,1) son 9,10,11.  
    Formula de posicion del primer byte del pixel (x,y): `(y * ancho + x) * canales`.  
    Asi dejas de ver "un bloque magico" y ves "una tabla de numeros" como en NumPy, pero en C con un solo puntero.

- **Libreria (stb_image)**: una cabecera de dominio publico que implementa decodificadores de JPG, PNG, BMP, etc. La funcion que usamos devuelve ese puntero y escribe en variables el ancho, alto y numero de canales. No hay matematica de vision; solo **lectura de archivo → buffer lineal de bytes**.

---

## 3. Intuicion visual / mental

- Imagina la imagen como una **tabla** de filas y columnas (como en Python). En C no tienes "filas" como tipo; tienes un **vector largo** donde la fila 0 va primero, luego la fila 1, etc. El indice lineal es `fila*ancho + columna` (y si hay 3 canales, cada pixel ocupa 3 posiciones).
- El **puntero** que te devuelve la libreria apunta al primer byte (esquina superior izquierda, primer canal del primer pixel). Avanzar de a 1 byte cambia de canal; avanzar de a `canales` bytes cambia de pixel en la misma fila; avanzar de a `ancho*canales` bytes cambia de fila.
- Cuando mas adelante hables de "entrada del modelo" o "buffer para TFLite", sera el mismo concepto: un bloque de bytes con dimensiones conocidas (ancho, alto, canales).

---

## 4. Practica minima (no recetas)

**Objetivo:** escribir un programa en C que cargue una imagen desde un archivo (usando stb_image), imprima ancho, alto y canales, e imprima los valores del pixel (0,0). El codigo se construye por pasos; los ejemplos se codearan en esta carpeta `10_img_en_C`.

### Paso 1: entorno y cabecera

- Crear un solo archivo `.c` en `10_img_en_C/`.
- Necesitas la cabecera **stb_image**. Es una sola cabecera de dominio publico; se descarga de https://github.com/nothings/stb (archivo `stb_image.h`) y se coloca en el mismo directorio (o en una ruta que indiques con `-I` al compilar).
- En **un solo** `.c` debes definir `STB_IMAGE_IMPLEMENTATION` antes de incluir `stb_image.h`; asi la implementacion se compila en tu unidad de traduccion. Luego `#include "stb_image.h"` y los includes estandar que necesites (`stdio.h`, `stdlib.h`).

Concepto: la libreria no es "magia"; es quien decodifica JPG/PNG y te devuelve un puntero a bytes. Tu programa solo usa esa API.

### Paso 2: main y ruta del archivo

- En `main(int argc, char **argv)` decides la ruta de la imagen: si `argc > 1`, usas `argv[1]`; si no, un nombre por defecto (por ejemplo `"test.jpg"`). Eso te permite ejecutar `./mi_programa imagen.png` o `./mi_programa` y que siga funcionando.

Concepto: mismo comportamiento que un script de Python que recibe la ruta por argumento o usa una por defecto.

### Paso 3: cargar la imagen

- La API de stb_image que necesitas es:  
  `unsigned char *datos = stbi_load(ruta, &ancho, &alto, &canales, 0);`  
  `ruta` es el path; `ancho`, `alto`, `canales` son punteros a `int` donde la libreria escribe las dimensiones; el ultimo `0` significa "quiero todos los canales tal cual" (3 o 4).
- Si `datos` es `NULL`, la carga fallo (archivo no existe, formato no soportado, etc.). En ese caso imprime un mensaje de error (por ejemplo con `fprintf(stderr, ...)`) y devuelve un codigo de salida distinto de 0 (por ejemplo 1). No sigas usando `datos`.

Concepto: en C no hay excepciones; se comprueba el puntero. Si es NULL, no dereferenciar.

### Paso 4: imprimir dimensiones y pixel (0,0)

- Con `ancho`, `alto`, `canales` ya puedes imprimir (por ejemplo con `printf`) las dimensiones.
- El pixel (0,0) empieza en `datos[0]`. Si hay al menos 3 canales, `datos[0]` es R, `datos[1]` es G, `datos[2]` es B; si hay 4, `datos[3]` es A. Imprime esos valores (por ejemplo como "R=... G=... B=..." con formato `%u` para unsigned).

Concepto: confirmas que el buffer es "fila por fila, pixel por pixel, canal por canal" como en la seccion de fundamento.

### Paso 5: liberar memoria y salir

- stb_image asigna memoria internamente. Para liberarla se usa `stbi_image_free(datos)`. Llamala antes de salir del programa (en la rama donde la carga fue correcta).
- Devuelve 0 si todo fue bien.

Concepto: en C quien pide memoria debe liberarla; la libreria te da la funcion para eso.

### Compilacion y ejecucion (Mac con gcc)

- Compilar: `gcc -o leer_imagen leer_imagen.c -lm` (o el nombre que des a tu `.c`). Si `stb_image.h` esta en otro directorio: `gcc -I/ruta/al/dir -o leer_imagen leer_imagen.c -lm`.
- Ejecutar: `./leer_imagen` o `./leer_imagen ruta/a/imagen.png`.
- Debes usar una imagen real (JPG/PNG/BMP) que exista en esa ruta; si no, `stbi_load` devolvera NULL y veras tu mensaje de error.

El codigo es **verificacion**: comprueba que entendiste buffer, dimensiones y posicion del pixel (0,0). Si algo no cuadra, vuelve al apartado de fundamento (orden en memoria).

---

## 5. Casos reales y conexion con lo moderno

- En proyectos reales en C/C++ (motores de juego, aplicaciones embebidas, pipelines de video) la imagen suele llegar como buffer (archivo, camara, red). Ese buffer tiene la misma estructura: ancho, alto, canales, bytes en orden. OpenCV en C++ abstrae eso en `cv::Mat`; en C puro trabajas con el puntero.
- En **TensorFlow Lite Micro** (Fase 8, ESP32-P4) la entrada del modelo es un buffer de bytes (por ejemplo imagen en escala de grises o RGB). Saber que la imagen es "un bloque de bytes con ancho, alto, canales" te permite entender que es lo que estas pasando al interprete.
- Las librerias modernas (OpenCV, PIL, stb_image) te evitan escribir decodificadores JPG/PNG; el concepto de "imagen = buffer + dimensiones" es el mismo.

---

## 6. Senales de que lo entendiste

- Puedes explicar sin codigo: "En C la imagen en memoria es un puntero a bytes; las primeras posiciones son el primer pixel (por ejemplo R, G, B), y el orden es fila por fila."
- Si te dan ancho=10, alto=5, canales=3, sabes que el pixel (2,1) empieza en el byte `(1*10+2)*3` = 36.
- Si `stbi_load` devuelve NULL, sabes que no debes usar el puntero y que debes informar error y liberar nada que hayas pedido tu (la libreria no asigna si falla).
- Entiendes que este programa no hace vision por computadora; solo pone la imagen en memoria para que mas adelante puedas pensar en "que hago con ese buffer" (inferencia, filtros, etc.).

---

## 7. Como lo haria un ingeniero senior

En un proyecto real en C/C++ suelen usar OpenCV (C++) o una libreria de decodificacion ya integrada (stb_image, libpng, libjpeg). El buffer resultante se pasa a la siguiente etapa (preprocesado, modelo, codificador). Se documentan formato (orden de canales, tipo de dato) y quien es responsable de liberar la memoria. Aqui hacemos "a mano" con stb_image para fijar el concepto; en industria se reutiliza el mismo concepto con las herramientas del proyecto.

---

## 8. Mock de camara: simular la estructura para ESP32 (camara USB)

En el curso usamos **ESP32-P4** (Fase 8) y en tu setup conectas una **camara por USB**. En el microcontrolador el driver de la camara te dara frames con la misma idea: **buffer de bytes + ancho + alto + formato**. Para practicar en la Mac sin hardware, podemos **mockear** esa fuente: una funcion que "obtiene un frame" y rellena la misma estructura; en el mock la fuente es un archivo (stb_image), en el ESP32 sera la API de la camara USB.

### Por que sirve en C para el ESP32

- En el ESP32 (o ESP32-P4) no tendras `stbi_load` leyendo un JPG del disco; tendras la camara USB que entrega un frame (buffer + dimensiones). El **tipo de dato** con el que trabajas es el mismo: struct con metadata + puntero al buffer.
- Si en la Mac escribes el pipeline contra una funcion `camera_get_frame(...)` que devuelve esa estructura, cuando portes al ESP32 solo cambias la **implementacion** de `camera_get_frame` (llamada al driver USB en lugar de cargar archivo); el resto del codigo (procesar frame, imprimir, o mas adelante pasar al modelo) sigue igual.

### Estructura comun (frame / imagen)

La misma que ya vimos; sirve tanto para "imagen cargada de archivo" como para "frame de camara":

- `ancho`, `alto`, `canales` (o `formato`: RGB, YUV, etc.).
- `datos`: puntero al primer byte del buffer.
- Quien asigna y libera: en el mock lo hace stb_image (tu llamas `stbi_image_free`); en el ESP32 lo suele gestionar el driver (te da un buffer que puede ser estatico o que debes devolver con `release_frame` segun el SDK).

### Mock en la Mac

- **Interfaz**: una funcion, por ejemplo `int camera_get_frame(Imagen *out, const char *ruta_archivo);`. En la Mac (mock): dentro de la funcion usas `stbi_load(ruta_archivo, &out->ancho, &out->alto, &out->canales, 0)` y guardas el puntero en `out->datos`. Devuelves 0 si OK, -1 si fallo. Asi simulas "el frame viene de algun lado"; en el mock ese "algun lado" es un archivo.
- **Uso en main**: llamas `camera_get_frame(&frame, "test.jpg")`; si devuelve 0, usas `frame.ancho`, `frame.alto`, `frame.datos` igual que cuando en el ESP32 llames al driver y te devuelva el frame. El resto del codigo (imprimir dimensiones, acceder a un pixel, o mas adelante preprocesar para un modelo) es identico.
- **Liberacion**: en el mock, cuando termines de usar el frame, llamas `stbi_image_free(frame.datos)`. En el ESP32 el driver te dira si debes llamar algo como `camera_release_frame()`.

### En el ESP32 con camara USB (real)

- La camara USB (UVC u otro protocolo) tendra un driver o SDK (por ejemplo ESP-IDF con componente USB host + UVC). Ese driver te dara algo del estilo: "aqui tienes un buffer, ancho X, alto Y, formato Z". Tu rellenaras el mismo `Imagen` (o el struct que use el SDK) y tu codigo de procesamiento no cambia.
- Hardware del curso: en estructura tenes **Sony IMX179** (CSI) para Raspberry Pi / UniHiker; para **ESP32-P4** indicaste **camara por USB**, asi que la fuente del frame en el ESP sera ese dispositivo USB, no CSI. La estructura del frame (buffer + metadata) es la misma; solo cambia quien llena el buffer.

### Practica sugerida (despues del programa "leer imagen")

1. Definir el struct `Imagen` (ancho, alto, canales, datos).
2. Implementar `camera_get_frame(Imagen *out, const char *ruta)` en mock: dentro, `stbi_load` y asignar a `out`; devolver 0 o -1.
3. En `main`: declarar `Imagen frame`; llamar `camera_get_frame(&frame, "test.jpg")`; si OK, imprimir dimensiones y pixel (0,0); al final `stbi_image_free(frame.datos)`.
4. Comprobar que mentalmente "esto mismo en el ESP32 seria una llamada al driver de la camara USB que rellena el mismo struct".

Asi tienes una primera aproximacion que sirve en C para el ESP32 y un mock simple que simula leer la camara usando la misma estructura que manejaras con la camara USB.

---

**Siguiente paso en el curso:** Fase 1 (Fundamentos de Deep Learning y CNNs). Esta clase es una aproximacion opcional en C antes de seguir con deteccion y embedded.
