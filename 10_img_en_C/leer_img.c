/*
Ejercicio de clase: leer una imagen y mostrar sus dimensiones y un pixel.
descargar stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
en mac: curl -L -o stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
compilar con: 
gcc -o leer_imagen leer_img.c -lm
gcc -o leer_img leer_img.c -I. -lstdc++ -lm
sin imagen:
gcc -o leer_imagen leer_img.c -lm && ./leer_imagen 
con imagen (va como argumento al ejecutar el programa):
gcc -o leer_imagen leer_img.c -lm && ./leer_imagen ../images/lily2.jpg
*/
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    char* pathImg = NULL;
    printf("###########################\n");
    printf("Inicio leer imagenes %d, argv: %s\n", argc, argv[0]);
    if (argc >= 2) {
        printf("Nombre de la imagen: %s\n", argv[1]);
        pathImg = argv[1];
    } 
        
    if(pathImg == NULL) {
        printf("No se proporciono nombre de la imagen\n");
        return 1;
    }

    /* Paso 2 y siguientes: ruta, cargar, imprimir, liberar */
    int alto, ancho, canales;

    int stRead = stbi_info(pathImg, &ancho, &alto, &canales);

    if(stRead == 0){
        printf("No se pudo leer la imagen \n");
        return 1;
    }
    
    printf("\r\n\r\nAncho: %d, Alto: %d, Canales: %d\r\n", ancho, alto, canales);
    printf("Size en bytes %u\n", (ancho * alto * canales));

    unsigned char* img = stbi_load(pathImg, &ancho, &alto, &canales, 0);
    if (img == NULL){
        printf("Error: No se pudo cargar la IMG de %s\n", pathImg);
        return 1;
    }
    
    printf("La imagen se cargo correctamente.\n");


    stbi_image_free(img);

    return 0;
}