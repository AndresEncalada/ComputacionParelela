/*
 * cpu_conv.c - Convolución 2D en CPU
 *
 * Compilar:
 *   nvcc -O2 -o cpu_conv.exe cpu_conv.c
 *
 * Uso:
 *   cpu_conv.exe [imagen.pgm]
 *   (sin argumento genera imágenes sintéticas)
 *
 * Salida:
 *   resultados/resultados_cpu.csv  - tiempos de ejecución
 *   resultados/*.pgm               - imágenes filtradas
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NUM_REPS 5  /* repeticiones medidas por experimento (+ 1 calentamiento) */

/* ─────────────────────────────────────────────────────────────────────────── */
/* Imagen                                                                      */
/* ─────────────────────────────────────────────────────────────────────────── */

/* pixel(fila, col) = datos[fila * ancho + col] */
typedef struct { int ancho, alto; unsigned char *datos; } Imagen;

Imagen* crearImagen(int ancho, int alto) {
    Imagen *img = (Imagen*)malloc(sizeof(Imagen));
    img->ancho = ancho; img->alto = alto;
    img->datos = (unsigned char*)calloc(ancho * alto, 1);
    return img;
}

void liberarImagen(Imagen *img) { free(img->datos); free(img); }

/* Lee PGM binario (P5). Retorna NULL si falla. */
Imagen* leerPGM(const char *ruta) {
    FILE *f = fopen(ruta, "rb");
    if (!f) return NULL;

    char magic[3];
    fscanf(f, "%2s", magic);
    if (strcmp(magic, "P5") != 0) { fclose(f); return NULL; }

    /* Saltar comentarios */
    int c = fgetc(f);
    while (c == '\n' || c == '\r' || c == ' ') c = fgetc(f);
    while (c == '#') {
        while (c != '\n' && c != EOF) c = fgetc(f);
        c = fgetc(f);
    }
    ungetc(c, f);

    Imagen *img = (Imagen*)malloc(sizeof(Imagen));
    int maxval;
    fscanf(f, "%d %d %d", &img->ancho, &img->alto, &maxval);
    fgetc(f); /* byte separador entre cabecera y datos binarios */

    img->datos = (unsigned char*)malloc(img->ancho * img->alto);
    fread(img->datos, 1, img->ancho * img->alto, f);
    fclose(f);
    return img;
}

void guardarPGM(const char *ruta, Imagen *img) {
    FILE *f = fopen(ruta, "wb");
    if (!f) return;
    fprintf(f, "P5\n%d %d\n255\n", img->ancho, img->alto);
    fwrite(img->datos, 1, img->ancho * img->alto, f);
    fclose(f);
}

/* Recorte centrado de 'base' al tamaño (ancho x alto).
 * Si base es NULL o muy pequeña, genera un gradiente sintético. */
Imagen* prepararImagen(Imagen *base, int ancho, int alto) {
    Imagen *img = crearImagen(ancho, alto);
    if (base && base->ancho >= ancho && base->alto >= alto) {
        int ox = (base->ancho - ancho) / 2;
        int oy = (base->alto  - alto)  / 2;
        for (int y = 0; y < alto; y++)
            memcpy(&img->datos[y * ancho],
                   &base->datos[(y + oy) * base->ancho + ox], ancho);
    } else {
        /* Gradiente diagonal: útil para ver el efecto de los filtros */
        for (int y = 0; y < alto; y++)
            for (int x = 0; x < ancho; x++)
                img->datos[y * ancho + x] = (unsigned char)((x + y) * 255 / (ancho + alto));
    }
    return img;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Máscaras                                                                    */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Media 3x3: todos los coeficientes = 1/9 → suaviza la imagen */
float MEDIA_3[9] = {
    1/9.0f, 1/9.0f, 1/9.0f,
    1/9.0f, 1/9.0f, 1/9.0f,
    1/9.0f, 1/9.0f, 1/9.0f
};

/* Nitidez 3x3: amplifica el centro y resta los vecinos → realza bordes */
float NITIDEZ_3[9] = {
     0, -1,  0,
    -1,  5, -1,
     0, -1,  0
};

/* Laplaciano 3x3: resalta cambios bruscos de intensidad */
float LAPLACIANO_3[9] = {
     0,  1,  0,
     1, -4,  1,
     0,  1,  0
};

/* Genera una máscara de media k×k: cada coeficiente = 1/(k*k) */
void mascaraMedia(float *m, int k) {
    float v = 1.0f / (k * k);
    for (int i = 0; i < k * k; i++) m[i] = v;
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Convolución                                                                 */
/* ─────────────────────────────────────────────────────────────────────────── */

/*
 * Aplica la máscara (tam × tam) sobre 'entrada' y escribe en 'salida'.
 *
 * Manejo de bordes: ZERO-PADDING.
 *   Píxeles fuera de la imagen se tratan como 0.
 *   Esto hace que CPU y GPU (Parte 2) sean comparables directamente.
 */
void convolucionCPU(Imagen *entrada, Imagen *salida, float *mascara, int tam) {
    int mitad = tam / 2;

    for (int fila = 0; fila < entrada->alto; fila++) {
        for (int col = 0; col < entrada->ancho; col++) {
            double suma = 0.0;

            for (int ky = 0; ky < tam; ky++) {
                for (int kx = 0; kx < tam; kx++) {
                    int y = fila + ky - mitad;
                    int x = col  + kx - mitad;

                    /* Zero-padding: pixel fuera de imagen vale 0 */
                    unsigned char pixel = 0;
                    if (x >= 0 && x < entrada->ancho &&
                        y >= 0 && y < entrada->alto)
                        pixel = entrada->datos[y * entrada->ancho + x];

                    suma += pixel * mascara[ky * tam + kx];
                }
            }

            /* Clamp [0, 255] */
            if (suma < 0)   suma = 0;
            if (suma > 255) suma = 255;
            salida->datos[fila * entrada->ancho + col] = (unsigned char)suma;
        }
    }
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Cronometraje                                                                */
/* ─────────────────────────────────────────────────────────────────────────── */

static int cmpd(const void *a, const void *b) {
    double da = *(double*)a, db = *(double*)b;
    return (da > db) - (da < db);
}

/* Ejecuta 1 calentamiento + NUM_REPS medidas. Retorna la mediana en ms. */
double medirTiempo(Imagen *ent, Imagen *sal, float *mask, int tam,
                   double *vmin, double *vmax) {
    double t[NUM_REPS];
    convolucionCPU(ent, sal, mask, tam); /* calentamiento */
    for (int r = 0; r < NUM_REPS; r++) {
        clock_t ini = clock();
        convolucionCPU(ent, sal, mask, tam);
        t[r] = (double)(clock() - ini) / CLOCKS_PER_SEC * 1000.0;
    }
    qsort(t, NUM_REPS, sizeof(double), cmpd);
    *vmin = t[0]; *vmax = t[NUM_REPS - 1];
    return t[NUM_REPS / 2]; /* mediana */
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Validación manual                                                           */
/* ─────────────────────────────────────────────────────────────────────────── */

/* Verifica la convolución sobre una matriz 5×5 con resultados calculados a mano. */
void validarManual() {
    printf("-- Validacion manual --\n");

    /*
     * Entrada 5x5:
     *   10 20 30 40 50
     *   20 30 40 50 60
     *   30 40 50 60 70
     *   40 50 60 70 80
     *   50 60 70 80 90
     *
     * Con media 3x3 y zero-padding:
     *   Pixel central (2,2): vecinos = {30,40,50,40,50,60,50,60,70}
     *                        suma=450, 450/9=50  → esperado: 50
     *
     *   Pixel esquina (0,0): vecinos con padding = {0,0,0,0,10,20,0,20,30}
     *                        suma=80, 80/9≈8.88  → esperado: 8
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

    float m[9]; mascaraMedia(m, 3);
    convolucionCPU(&ent, sal, m, 3);

    printf("  Pixel central (2,2): esperado=50  obtenido=%d  -> %s\n",
           sal->datos[12], sal->datos[12] == 50 ? "OK" : "FALLA");
    printf("  Pixel esquina (0,0): esperado=8   obtenido=%d  -> %s\n\n",
           sal->datos[0],  sal->datos[0]  == 8  ? "OK" : "FALLA");

    liberarImagen(sal);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* Helper para un experimento                                                  */
/* ─────────────────────────────────────────────────────────────────────────── */

void experimento(Imagen *img, float *mask, int tam, const char *filtro,
                 const char *archivo, FILE *csv) {
    Imagen *sal = crearImagen(img->ancho, img->alto);
    double vmin, vmax;
    double med = medirTiempo(img, sal, mask, tam, &vmin, &vmax);

    printf("  %-12s mascara=%2dx%-2d  res=%dx%d  mediana=%.0f ms  [%.0f, %.0f]\n",
           filtro, tam, tam, img->ancho, img->alto, med, vmin, vmax);
    guardarPGM(archivo, sal);
    fprintf(csv, "CPU,%s,%dx%d,%dx%d,%.1f,%.1f,%.1f\n",
            filtro, img->ancho, img->alto, tam, tam, med, vmin, vmax);

    liberarImagen(sal);
}

/* ─────────────────────────────────────────────────────────────────────────── */
/* main                                                                        */
/* ─────────────────────────────────────────────────────────────────────────── */

int main(int argc, char *argv[]) {
#ifdef _WIN32
    system("if not exist resultados mkdir resultados");
#else
    system("mkdir -p resultados");
#endif

    Imagen *base = (argc >= 2) ? leerPGM(argv[1]) : NULL;
    if (base) printf("Imagen base: %dx%d px\n", base->ancho, base->alto);
    else      printf("Sin imagen base; usando gradiente sintetico.\n");

    /* Máscaras de media variables */
    float m3[9], m9[81], m21[441];
    mascaraMedia(m3, 3); mascaraMedia(m9, 9); mascaraMedia(m21, 21);

    /* Validación antes de los experimentos */
    validarManual();

    FILE *csv = fopen("resultados/resultados_cpu.csv", "w");
    fprintf(csv, "implementacion,filtro,resolucion,mascara,mediana_ms,min_ms,max_ms\n");

    /* Filtros 3x3 sobre 4000x4000 */
    printf("[Exp 1] Filtros 3x3 sobre 4000x4000\n");
    Imagen *i4k = prepararImagen(base, 4000, 4000);
    experimento(i4k, m3,           3, "media",      "resultados/media_4k.pgm",      csv);
    experimento(i4k, NITIDEZ_3,    3, "nitidez",    "resultados/nitidez_4k.pgm",    csv);
    experimento(i4k, LAPLACIANO_3, 3, "laplaciano", "resultados/laplaciano_4k.pgm", csv);
    liberarImagen(i4k);

    /* Media 9x9 en tres resoluciones */
    printf("[Exp 2] Resolucion con media 9x9\n");
    Imagen *i2k = prepararImagen(base, 2000, 2000);
    experimento(i2k, m9, 9, "media", "resultados/media9_2k.pgm", csv);
    liberarImagen(i2k);
    i4k = prepararImagen(base, 4000, 4000);
    experimento(i4k, m9, 9, "media", "resultados/media9_4k.pgm", csv);
    liberarImagen(i4k);
    Imagen *i6k = prepararImagen(base, 6000, 6000);
    experimento(i6k, m9, 9, "media", "resultados/media9_6k.pgm", csv);
    liberarImagen(i6k);

    /* Tres tamaños de máscara sobre 4000x4000 */
    printf("[Exp 3] Tamano de mascara sobre 4000x4000\n");
    i4k = prepararImagen(base, 4000, 4000);
    experimento(i4k, m3,  3,  "media", "resultados/media3_4k.pgm",   csv);
    experimento(i4k, m9,  9,  "media", "resultados/media9_4k_b.pgm", csv);
    experimento(i4k, m21, 21, "media", "resultados/media21_4k.pgm",  csv);
    liberarImagen(i4k);

    fclose(csv);
    if (base) liberarImagen(base);
    printf("Listo. Ver resultados/resultados_cpu.csv\n");
    return 0;
}
