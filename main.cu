#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Definición de los 3 filtros (máscaras 3x3)
float filtro_blur[9] = {
    1/9.0f, 1/9.0f, 1/9.0f,
    1/9.0f, 1/9.0f, 1/9.0f,
    1/9.0f, 1/9.0f, 1/9.0f
};

float filtro_sharpen[9] = {
     0, -1,  0,
    -1,  5, -1,
     0, -1,  0
};

float filtro_edge[9] = {
    -1, -1, -1,
    -1,  8, -1,
    -1, -1, -1
};

// Estructura simple para la imagen
typedef struct {
    int ancho;
    int alto;
    unsigned char *datos;
} Imagen;

// Función para leer PGM (Formato P5 binario)
Imagen* leerPGM(const char *nombreArchivo) {
    FILE *f = fopen(nombreArchivo, "rb");
    if (!f) { printf("Error al abrir imagen\n"); exit(1); }

    char formato[3];
    fscanf(f, "%2s", formato); // Debe ser "P5"
    
    Imagen *img = (Imagen*)malloc(sizeof(Imagen));
    
    // Saltar comentarios y leer dimensiones
    int c = fgetc(f);
    while (c == '#' || c == '\n' || c == '\r') {
        while(c != '\n' && c != '\r') c = fgetc(f);
        c = fgetc(f);
    }
    ungetc(c, f);
    
    fscanf(f, "%d %d", &img->ancho, &img->alto);
    fgetc(f); // Saltar el espacio/salto de línea
    fgetc(f); // Saltar el valor máximo (usualmente 255)

    img->datos = (unsigned char*)malloc(img->ancho * img->alto);
    fread(img->datos, 1, img->ancho * img->alto, f);
    fclose(f);
    return img;
}

// Función para guardar PGM
void guardarPGM(const char *nombreArchivo, Imagen *img) {
    FILE *f = fopen(nombreArchivo, "wb");
    fprintf(f, "P5\n%d %d\n255\n", img->ancho, img->alto);
    fwrite(img->datos, 1, img->ancho * img->alto, f);
    fclose(f);
}
// Función auxiliar para obtener el valor del píxel manejando los bordes (clamp)
unsigned char obtenerPixel(Imagen *img, int x, int y) {
    // Clamp en X
    if (x < 0) x = 0;
    if (x >= img->ancho) x = img->ancho - 1;
    // Clamp en Y
    if (y < 0) y = 0;
    if (y >= img->alto) y = img->alto - 1;
    
    return img->datos[y * img->ancho + x];
}

// Convolución en CPU
void convolucionCPU(Imagen *origen, Imagen *destino, float *filtro) {
    for (int y = 0; y < origen->alto; y++) {
        for (int x = 0; x < origen->ancho; x++) {
            float suma = 0.0f;
            
            // Aplicar la máscara de 3x3
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    unsigned char pixel = obtenerPixel(origen, x + kx, y + ky);
                    // El índice del filtro va de 0 a 8
                    int filtro_idx = (ky + 1) * 3 + (kx + 1);
                    suma += pixel * filtro[filtro_idx];
                }
            }
            
            // Guardar resultado (clamp entre 0 y 255)
            if (suma < 0) suma = 0;
            if (suma > 255) suma = 255;
            destino->datos[y * origen->ancho + x] = (unsigned char)suma;
        }
    }
}

int main() {
    // Leer imagen de entrada
    Imagen *original = leerPGM("entrada.pgm");
    if (!original) return 1;
    
    printf("Imagen cargada: %dx%d píxeles\n", original->ancho, original->alto);
    
    // Crear imagen destino (misma dimensión)
    Imagen *destino = (Imagen*)malloc(sizeof(Imagen));
    destino->ancho = original->ancho;
    destino->alto = original->alto;
    destino->datos = (unsigned char*)malloc(original->ancho * original->alto);
    
    // Aplicar los 3 filtros en CPU
    printf("Aplicando filtro BLUR en CPU...\n");
    convolucionCPU(original, destino, filtro_blur);
    guardarPGM("salida_blur_cpu.pgm", destino);
    
    printf("Aplicando filtro SHARPEN en CPU...\n");
    convolucionCPU(original, destino, filtro_sharpen);
    guardarPGM("salida_sharpen_cpu.pgm", destino);
    
    printf("Aplicando filtro EDGE en CPU...\n");
    convolucionCPU(original, destino, filtro_edge);
    guardarPGM("salida_edge_cpu.pgm", destino);
    
    printf("¡Listo! Revisa las 3 imágenes de salida.\n");
    
    // Liberar memoria
    free(original->datos);
    free(original);
    free(destino->datos);
    free(destino);
    
    return 0;
}