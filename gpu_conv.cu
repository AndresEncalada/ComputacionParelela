/*
 * gpu_conv.cu - Kernel CUDA 1D y medicion de tiempos en GPU
 */

#include <cuda_runtime.h>

#define HILOS_BASE 256  // Tamano de bloque fijo para los primeros 3 experimentos
#define TOLERANCIA 1e-4f

// Macro para verificar si alguna llamada de CUDA dio error
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

// --- Kernel en CUDA ---

// Configuracion 1D (bloques e hilos solo en X).
// Cada hilo calcula 1 pixel de salida leyendo desde memoria global.
__global__ void kernelConvolucionGPU(const unsigned char *entrada,
                                     float *salida_cruda,
                                     const float *mascara,
                                     int ancho, int alto, int tam) {
    // 1. Calcular el indice global 1D del hilo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Control de bordes: si hay mas hilos que pixeles en la imagen, no hacen nada
    if (idx >= ancho * alto) return;

    // 2. Convertir el indice 1D a coordenadas 2D (fila y columna)
    int fila  = idx / ancho;
    int col   = idx % ancho;

    int mitad = tam / 2;
    float suma = 0.0f;

    // 3. Recorrer la mascara centrada en (fila, col)
    for (int ky = 0; ky < tam; ky++) {
        for (int kx = 0; kx < tam; kx++) {
            int y = fila + ky - mitad;
            int x = col  + kx - mitad;

            // Zero-padding: fuera de los limites de la imagen se asume 0
            unsigned char pixel = 0;
            if (x >= 0 && x < ancho && y >= 0 && y < alto)
                pixel = entrada[y * ancho + x];

            suma += (float)pixel * mascara[ky * tam + kx];
        }
    }

    // Guardar el valor en float antes de truncar a [0, 255]
    salida_cruda[idx] = suma;
}

// --- Validacion numerica ---

// Calcula el error absoluto maximo y el error absoluto medio (MAE) entre CPU y GPU
void compararCPUvsGPU(const float *cpu_crudo, const float *gpu_crudo, int n,
                      double *max_err, double *mae) {
    double max_e = 0.0, suma_e = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs((double)cpu_crudo[i] - (double)gpu_crudo[i]);
        if (diff > max_e) max_e = diff;
        suma_e += diff;
    }
    *max_err = max_e;
    *mae     = suma_e / n;
}

// Prueba la matriz pequena de 5x5 en GPU y la compara con el resultado de CPU
void validarManualCPU_GPU() {
    validarManual();

    unsigned char datos[25] = {
        10,20,30,40,50,
        20,30,40,50,60,
        30,40,50,60,70,
        40,50,60,70,80,
        50,60,70,80,90
    };
    Imagen ent = {5, 5, datos};
    float crudo_cpu[25], crudo_gpu[25];
    float m[9];
    mascaraMedia(m, 3);

    convolucionCPU_Crudo(&ent, crudo_cpu, m, 3);

    unsigned char *d_ent = NULL;
    float *d_sal = NULL, *d_mask = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_ent,  25 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc((void**)&d_sal,  25 * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_mask, 9  * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_ent,  datos, 25 * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, m,     9  * sizeof(float),         cudaMemcpyHostToDevice));

    // Lanzamos 32 hilos para 25 pixeles (los 7 hilos sobrantes prueban el if idx >= ancho*alto)
    kernelConvolucionGPU<<<1, 32>>>(d_ent, d_sal, d_mask, 5, 5, 3);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(crudo_gpu, d_sal, 25 * sizeof(float), cudaMemcpyDeviceToHost));

    double max_err = 0.0, mae = 0.0;
    compararCPUvsGPU(crudo_cpu, crudo_gpu, 25, &max_err, &mae);
    printf("  [GPU 5x5 vs Manual/CPU] Central=%.4f | Esquina=%.4f | MaxErr=%.2e | MAE=%.2e -> %s\n\n",
           crudo_gpu[12], crudo_gpu[0], max_err, mae, (max_err <= TOLERANCIA) ? "OK" : "FALLA");

    CUDA_CHECK(cudaFree(d_ent));
    CUDA_CHECK(cudaFree(d_sal));
    CUDA_CHECK(cudaFree(d_mask));
}

// --- Medicion de tiempos en GPU ---

// Mide con cudaEvent_t los 3 pasos por separado:
// 1) Copia H2D (CPU a GPU), 2) Kernel, 3) Copia D2H (GPU a CPU), y la suma de los tres.
// La reserva de memoria con cudaMalloc queda fuera de la medicion.
void medirTiempoGPU(Imagen *ent, float *h_salida_cruda, float *mask, int tam,
                    int hilos_bloque, const char *filtro, FILE *csv_crudo,
                    double *med_h2d, double *med_ker, double *med_d2h,
                    double *med_tot, double *iqr_tot, double *std_tot,
                    double *iqr_ker, double *std_ker) {
    int n = ent->ancho * ent->alto;
    size_t bytes_img  = (size_t)n * sizeof(unsigned char);
    size_t bytes_sal  = (size_t)n * sizeof(float);
    size_t bytes_mask = (size_t)tam * tam * sizeof(float);

    // Cantidad de bloques redondeando hacia arriba para cubrir todos los pixeles
    int bloques = (n + hilos_bloque - 1) / hilos_bloque;

    unsigned char *d_ent = NULL;
    float *d_sal = NULL, *d_mask = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_ent,  bytes_img));
    CUDA_CHECK(cudaMalloc((void**)&d_sal,  bytes_sal));
    CUDA_CHECK(cudaMalloc((void**)&d_mask, bytes_mask));

    cudaEvent_t ini, fin;
    CUDA_CHECK(cudaEventCreate(&ini));
    CUDA_CHECK(cudaEventCreate(&fin));

    // Ejecucion de calentamiento (no se mide)
    CUDA_CHECK(cudaMemcpy(d_ent,  ent->datos, bytes_img,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, mask,       bytes_mask, cudaMemcpyHostToDevice));
    kernelConvolucionGPU<<<bloques, hilos_bloque>>>(d_ent, d_sal, d_mask, ent->ancho, ent->alto, tam);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_salida_cruda, d_sal, bytes_sal, cudaMemcpyDeviceToHost));

    double t_h2d[NUM_REPS], t_ker[NUM_REPS], t_d2h[NUM_REPS], t_tot[NUM_REPS];
    char cfg[64];
    snprintf(cfg, sizeof(cfg), "%d_hilos_%d_bloques", hilos_bloque, bloques);

    for (int r = 0; r < NUM_REPS; r++) {
        float ms_h2d = 0.0f, ms_ker = 0.0f, ms_d2h = 0.0f;

        // 1. Medir transferencia CPU -> GPU (imagen y mascara)
        CUDA_CHECK(cudaEventRecord(ini, 0));
        CUDA_CHECK(cudaMemcpy(d_ent,  ent->datos, bytes_img,  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mask, mask,       bytes_mask, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(fin, 0));
        CUDA_CHECK(cudaEventSynchronize(fin));
        CUDA_CHECK(cudaEventElapsedTime(&ms_h2d, ini, fin));

        // 2. Medir solo la ejecucion del kernel
        CUDA_CHECK(cudaEventRecord(ini, 0));
        kernelConvolucionGPU<<<bloques, hilos_bloque>>>(d_ent, d_sal, d_mask, ent->ancho, ent->alto, tam);
        CUDA_CHECK(cudaEventRecord(fin, 0));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventSynchronize(fin));
        CUDA_CHECK(cudaEventElapsedTime(&ms_ker, ini, fin));

        // 3. Medir transferencia GPU -> CPU (arreglo de salida)
        CUDA_CHECK(cudaEventRecord(ini, 0));
        CUDA_CHECK(cudaMemcpy(h_salida_cruda, d_sal, bytes_sal, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(fin, 0));
        CUDA_CHECK(cudaEventSynchronize(fin));
        CUDA_CHECK(cudaEventElapsedTime(&ms_d2h, ini, fin));

        t_h2d[r] = ms_h2d;
        t_ker[r] = ms_ker;
        t_d2h[r] = ms_d2h;
        t_tot[r] = ms_h2d + ms_ker + ms_d2h;

        if (csv_crudo) {
            fprintf(csv_crudo, "GPU_H2D,%s,%dx%d,%dx%d,%s,%d,%.4f,ms\n",
                    filtro, ent->ancho, ent->alto, tam, tam, cfg, r + 1, t_h2d[r]);
            fprintf(csv_crudo, "GPU_Kernel,%s,%dx%d,%dx%d,%s,%d,%.4f,ms\n",
                    filtro, ent->ancho, ent->alto, tam, tam, cfg, r + 1, t_ker[r]);
            fprintf(csv_crudo, "GPU_D2H,%s,%dx%d,%dx%d,%s,%d,%.4f,ms\n",
                    filtro, ent->ancho, ent->alto, tam, tam, cfg, r + 1, t_d2h[r]);
            fprintf(csv_crudo, "GPU_Conjunto,%s,%dx%d,%dx%d,%s,%d,%.4f,ms\n",
                    filtro, ent->ancho, ent->alto, tam, tam, cfg, r + 1, t_tot[r]);
        }
    }

    // Calcular desviacion estandar del tiempo total y del kernel
    double suma_tot = 0.0, suma_ker = 0.0;
    for (int i = 0; i < NUM_REPS; i++) {
        suma_tot += t_tot[i];
        suma_ker += t_ker[i];
    }
    double media_tot = suma_tot / NUM_REPS, media_ker = suma_ker / NUM_REPS;
    double var_tot = 0.0, var_ker = 0.0;
    for (int i = 0; i < NUM_REPS; i++) {
        var_tot += (t_tot[i] - media_tot) * (t_tot[i] - media_tot);
        var_ker += (t_ker[i] - media_ker) * (t_ker[i] - media_ker);
    }
    *std_tot = sqrt(var_tot / (NUM_REPS - 1));
    if (std_ker) *std_ker = sqrt(var_ker / (NUM_REPS - 1));

    // Ordenar las 5 repeticiones para obtener mediana (indice 2) y rango intercuartilico (Q3 - Q1)
    qsort(t_h2d, NUM_REPS, sizeof(double), cmpd);
    qsort(t_ker, NUM_REPS, sizeof(double), cmpd);
    qsort(t_d2h, NUM_REPS, sizeof(double), cmpd);
    qsort(t_tot, NUM_REPS, sizeof(double), cmpd);

    *med_h2d = t_h2d[NUM_REPS / 2];
    *med_ker = t_ker[NUM_REPS / 2];
    *med_d2h = t_d2h[NUM_REPS / 2];
    *med_tot = t_tot[NUM_REPS / 2];
    *iqr_tot = t_tot[3] - t_tot[1];
    if (iqr_ker) *iqr_ker = t_ker[3] - t_ker[1];

    CUDA_CHECK(cudaEventDestroy(ini));
    CUDA_CHECK(cudaEventDestroy(fin));
    CUDA_CHECK(cudaFree(d_ent));
    CUDA_CHECK(cudaFree(d_sal));
    CUDA_CHECK(cudaFree(d_mask));
}
