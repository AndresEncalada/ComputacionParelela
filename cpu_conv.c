/*
 * cpu_conv.c - Funciones de imagen, mascaras y convolucion 2D secuencial en CPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

// stb_image: cabecera unica para leer JPG, PNG, BMP, etc. sin dependencias externas
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"

#define NUM_REPS 5  // 5 repeticiones medidas (+ 1 previa de calentamiento)

// Estructura basica para manejar la imagen en un arreglo 1D
// El pixel (fila, col) se ubica en datos[fila * ancho + col]
typedef struct {
    int ancho;
    int alto;
    unsigned char *datos;
} Imagen;

Imagen* crearImagen(int ancho, int alto) {
    Imagen *img = (Imagen*)malloc(sizeof(Imagen));
    img->ancho = ancho;
    img->alto = alto;
    img->datos = (unsigned char*)calloc((size_t)ancho * alto, 1);
    return img;
}

void liberarImagen(Imagen *img) {
    free(img->datos);
    free(img);
}

// Lee una imagen en formato PGM binario (P5)
Imagen* leerPGM(const char *ruta) {
    FILE *f = fopen(ruta, "rb");
    if (!f) return NULL;

    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fclose(f);
        return NULL;
    }

    // Saltar espacios y lineas de comentarios que empiezan con '#'
    int c = fgetc(f);
    while (c == '\n' || c == '\r' || c == ' ' || c == '\t') c = fgetc(f);
    while (c == '#') {
        while (c != '\n' && c != EOF) c = fgetc(f);
        c = fgetc(f);
    }
    ungetc(c, f);

    Imagen *img = (Imagen*)malloc(sizeof(Imagen));
    int maxval;
    if (fscanf(f, "%d %d %d", &img->ancho, &img->alto, &maxval) != 3) {
        free(img);
        fclose(f);
        return NULL;
    }
    fgetc(f); // Saltar el salto de linea antes de los bytes de la imagen

    img->datos = (unsigned char*)malloc((size_t)img->ancho * img->alto);
    size_t leidos = fread(img->datos, 1, (size_t)img->ancho * img->alto, f);
    fclose(f);
    if (leidos != (size_t)img->ancho * img->alto) {
        liberarImagen(img);
        return NULL;
    }
    return img;
}

// Carga cualquier imagen (JPG, PNG, BMP o PGM) y la devuelve en escala de grises.
// stb_image convierte automaticamente de color a gris con la formula estandar.
Imagen* leerImagen(const char *ruta) {
    // Si la extension es .pgm usamos nuestro lector propio (garantiza compatibilidad P5)
    int n = (int)strlen(ruta);
    if (n >= 4) {
        const char *e = ruta + n - 4;
        if ((e[0]=='.') &&
            (e[1]=='p'||e[1]=='P') &&
            (e[2]=='g'||e[2]=='G') &&
            (e[3]=='m'||e[3]=='M'))
            return leerPGM(ruta);
    }

    // Para JPG/PNG/BMP: stbi_load con desired_channels=1 entrega gris directamente
    int ancho, alto, canales_orig;
    unsigned char *raw = stbi_load(ruta, &ancho, &alto, &canales_orig, 1);
    if (!raw) {
        fprintf(stderr, "No se pudo leer la imagen '%s': %s\n", ruta, stbi_failure_reason());
        return NULL;
    }

    Imagen *img = crearImagen(ancho, alto);
    memcpy(img->datos, raw, (size_t)ancho * alto);
    stbi_image_free(raw);
    return img;
}

void guardarPGM(const char *ruta, Imagen *img) {
    FILE *f = fopen(ruta, "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", img->ancho, img->alto);
    fwrite(img->datos, 1, (size_t)img->ancho * img->alto, f);
    fclose(f);
}

// Si hay imagen base, toma un recorte cuadrado del centro y lo escala a (ancho x alto)
// usando vecino mas cercano. Si no hay imagen base, llena un gradiente diagonal.
Imagen* prepararImagen(Imagen *base, int ancho, int alto) {
    Imagen *img = crearImagen(ancho, alto);
    if (base && base->ancho > 0 && base->alto > 0) {
        // Lado del cuadrado centrado dentro de la imagen original
        int lado = (base->ancho < base->alto) ? base->ancho : base->alto;
        int ox = (base->ancho - lado) / 2;
        int oy = (base->alto  - lado) / 2;

        for (int y = 0; y < alto; y++) {
            int src_y = oy + (int)((long long)y * lado / alto);
            for (int x = 0; x < ancho; x++) {
                int src_x = ox + (int)((long long)x * lado / ancho);
                img->datos[y * ancho + x] = base->datos[src_y * base->ancho + src_x];
            }
        }
    } else {
        for (int y = 0; y < alto; y++)
            for (int x = 0; x < ancho; x++)
                img->datos[y * ancho + x] = (unsigned char)((x + y) * 255 / (ancho + alto));
    }
    return img;
}

// --- Definicion de mascaras ---

// Filtro de media 3x3: promedia los 9 vecinos para suavizar la imagen
float MEDIA_3[9] = {
    1/9.0f, 1/9.0f, 1/9.0f,
    1/9.0f, 1/9.0f, 1/9.0f,
    1/9.0f, 1/9.0f, 1/9.0f
};

// Filtro de nitidez 3x3: refuerza el pixel central y resta los lados
float NITIDEZ_3[9] = {
     0, -1,  0,
    -1,  5, -1,
     0, -1,  0
};

// Filtro Laplaciano 3x3: detecta cambios bruscos de intensidad (bordes)
float LAPLACIANO_3[9] = {
     0,  1,  0,
     1, -4,  1,
     0,  1,  0
};

// Llena una mascara de media de tamano k x k donde cada valor es 1/(k*k)
void mascaraMedia(float *m, int k) {
    float v = 1.0f / (k * k);
    for (int i = 0; i < k * k; i++) m[i] = v;
}

// --- Funciones de convolucion ---

// Aplica la convolucion y guarda el resultado en float sin recortar a [0, 255].
// Esto permite comparar los decimales exactos contra la GPU antes de guardar la imagen.
void convolucionCPU_Crudo(Imagen *entrada, float *salida_cruda, float *mascara, int tam) {
    int mitad = tam / 2; // Desplazamiento desde el centro de la mascara

    for (int fila = 0; fila < entrada->alto; fila++) {
        for (int col = 0; col < entrada->ancho; col++) {
            float suma = 0.0f;

            // Recorrer la vecindad de tam x tam alrededor de (fila, col)
            for (int ky = 0; ky < tam; ky++) {
                for (int kx = 0; kx < tam; kx++) {
                    int y = fila + ky - mitad;
                    int x = col  + kx - mitad;

                    // Zero-padding: si el vecino cae fuera de la imagen, vale 0
                    unsigned char pixel = 0;
                    if (x >= 0 && x < entrada->ancho &&
                        y >= 0 && y < entrada->alto)
                        pixel = entrada->datos[y * entrada->ancho + x];

                    suma += (float)pixel * mascara[ky * tam + kx];
                }
            }
            salida_cruda[fila * entrada->ancho + col] = suma;
        }
    }
}

// Limita los valores al rango [0, 255] (clamp) y los pasa a unsigned char para el PGM
void convertirCrudoAPGM(const float *salida_cruda, Imagen *salida) {
    int n = salida->ancho * salida->alto;
    for (int i = 0; i < n; i++) {
        float v = salida_cruda[i];
        if (v < 0.0f)   v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        salida->datos[i] = (unsigned char)v;
    }
}

void convolucionCPU(Imagen *entrada, Imagen *salida, float *mascara, int tam) {
    float *crudo = (float*)malloc((size_t)entrada->ancho * entrada->alto * sizeof(float));
    convolucionCPU_Crudo(entrada, crudo, mascara, tam);
    convertirCrudoAPGM(crudo, salida);
    free(crudo);
}

// --- Medicion de tiempos ---

// Funcion auxiliar para ordenar el arreglo de tiempos con qsort
static int cmpd(const void *a, const void *b) {
    double da = *(double*)a, db = *(double*)b;
    return (da > db) - (da < db);
}

// Hace 1 corrida de calentamiento y luego mide NUM_REPS repeticiones.
// Devuelve la mediana en milisegundos.
double medirTiempo(Imagen *ent, float *sal_cruda, float *mask, int tam,
                   double *tiempos_out, double *vmin, double *vmax) {
    double t[NUM_REPS];
    convolucionCPU_Crudo(ent, sal_cruda, mask, tam); // Calentamiento no medido

    for (int r = 0; r < NUM_REPS; r++) {
        clock_t ini = clock();
        convolucionCPU_Crudo(ent, sal_cruda, mask, tam);
        t[r] = (double)(clock() - ini) / CLOCKS_PER_SEC * 1000.0;
        if (tiempos_out) tiempos_out[r] = t[r];
    }

    // Ordenamos los 5 tiempos para sacar min, max y la mediana (posicion central)
    qsort(t, NUM_REPS, sizeof(double), cmpd);
    *vmin = t[0];
    *vmax = t[NUM_REPS - 1];
    return t[NUM_REPS / 2];
}

// --- Validacion con matriz pequena 5x5 ---

void validarManual() {
    printf("-- Validacion manual (Matriz 5x5) --\n");

    /*
     * Matriz de prueba 5x5:
     *   10 20 30 40 50
     *   20 30 40 50 60
     *   30 40 50 60 70
     *   40 50 60 70 80
     *   50 60 70 80 90
     *
     * Calculo a mano con filtro de media 3x3 y zero-padding:
     *   Centro (2,2): (30+40+50 + 40+50+60 + 50+60+70) / 9 = 450 / 9 = 50.0
     *   Esquina (0,0): (0+0+0 + 0+10+20 + 0+20+30) / 9 = 80 / 9 = 8.8889 (en entero: 8)
     */
    unsigned char datos[25] = {
        10,20,30,40,50,
        20,30,40,50,60,
        30,40,50,60,70,
        40,50,60,70,80,
        50,60,70,80,90
    };
    Imagen ent = {5, 5, datos};
    Imagen *sal = crearImagen(5, 5);
    float crudo[25];

    float m[9];
    mascaraMedia(m, 3);
    convolucionCPU_Crudo(&ent, crudo, m, 3);
    convertirCrudoAPGM(crudo, sal);

    printf("  Pixel central (2,2): esperado=50.0000 (50) | obtenido=%.4f (%d) -> %s\n",
           crudo[12], sal->datos[12],
           (fabsf(crudo[12] - 50.0f) < 1e-4f && sal->datos[12] == 50) ? "OK" : "FALLA");
    printf("  Pixel esquina (0,0): esperado=8.8889  (8)  | obtenido=%.4f (%d) -> %s\n\n",
           crudo[0], sal->datos[0],
           (fabsf(crudo[0] - 80.0f/9.0f) < 1e-4f && sal->datos[0] == 8) ? "OK" : "FALLA");

    liberarImagen(sal);
}
