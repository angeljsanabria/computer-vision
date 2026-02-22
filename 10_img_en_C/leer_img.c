/*
Ejercicio de clase: leer una imagen y mostrar sus dimensiones y un pixel.
descargar stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
en mac: curl -L -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
compilar con: 
gcc -o leer_imagen leer_imagen.c -lm
gcc -o leer_img leer_img.c -I. -lstdc++ -lm
*/
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    /* Paso 2 y siguientes: ruta, cargar, imprimir, liberar */
    return 0;
}