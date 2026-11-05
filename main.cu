/*
 * main.cu - Programa principal que ejecuta CPU (cpu_conv.c) y GPU (gpu_conv.cu)
 *
 * Compilar:
 *   nvcc -O2 -o main main.cu
 *
 * Uso:
 *   ./main [imagen.jpg]
 */

#include "cpu_conv.c"
#include "gpu_conv.cu"

// Corre un experimento en CPU y en GPU usando la misma imagen en memoria,
// compara los arreglos en float para sacar el error y guarda todo en el CSV.
void experimentoUnificado(Imagen *img, float *mask, int tam, const char *filtro,
                          int hilos_bloque, const char *archivo_base,
                          FILE *csv_crudo, FILE *csv_resumen) {
    int n = img->ancho * img->alto;
    Imagen *sal_cpu = crearImagen(img->ancho, img->alto);
    Imagen *sal_gpu = crearImagen(img->ancho, img->alto);
    float *crudo_cpu = (float*)malloc((size_t)n * sizeof(float));
    float *crudo_gpu = (float*)malloc((size_t)n * sizeof(float));

    // 1. Medir en CPU (1 calentamiento + 5 repeticiones)
    double t_cpu[NUM_REPS], vmin_c, vmax_c;
    double med_cpu = medirTiempo(img, crudo_cpu, mask, tam, t_cpu, &vmin_c, &vmax_c);

    for (int r = 0; r < NUM_REPS; r++) {
        fprintf(csv_crudo, "CPU,%s,%dx%d,%dx%d,1_hilo_secuencial,%d,%.4f,ms\n",
                filtro, img->ancho, img->alto, tam, tam, r + 1, t_cpu[r]);
    }

    // Calcular desviacion estandar e IQR de CPU
    double suma_c = 0.0;
    for (int i = 0; i < NUM_REPS; i++) suma_c += t_cpu[i];
    double media_c = suma_c / NUM_REPS, var_c = 0.0;
    for (int i = 0; i < NUM_REPS; i++) var_c += (t_cpu[i] - media_c) * (t_cpu[i] - media_c);
    double std_cpu = sqrt(var_c / (NUM_REPS - 1));

    qsort(t_cpu, NUM_REPS, sizeof(double), cmpd);
    double iqr_cpu = t_cpu[3] - t_cpu[1];

    // 2. Medir en GPU por separado: H2D, Kernel, D2H y tiempo conjunto
    double med_h2d, med_ker, med_d2h, med_tot, iqr_tot, std_tot, iqr_ker, std_ker;
    medirTiempoGPU(img, crudo_gpu, mask, tam, hilos_bloque, filtro, csv_crudo,
                   &med_h2d, &med_ker, &med_d2h, &med_tot, &iqr_tot, &std_tot,
                   &iqr_ker, &std_ker);

    // 3. Comparar el resultado en float de CPU contra el de GPU (antes del clamp a 0..255)
    double max_err = 0.0, mae = 0.0;
    compararCPUvsGPU(crudo_cpu, crudo_gpu, n, &max_err, &mae);

    // 4. Pasar de float a [0, 255] y guardar las imagenes de salida
    if (archivo_base) {
        convertirCrudoAPGM(crudo_cpu, sal_cpu);
        convertirCrudoAPGM(crudo_gpu, sal_gpu);
        char ruta_cpu[256], ruta_gpu[256];
        snprintf(ruta_cpu, sizeof(ruta_cpu), "resultados/%s_cpu.jpg", archivo_base);
        snprintf(ruta_gpu, sizeof(ruta_gpu), "resultados/%s_gpu.jpg", archivo_base);
        guardarImagen(ruta_cpu, sal_cpu);
        guardarImagen(ruta_gpu, sal_gpu);
    }

    // Calcular aceleracion solo del kernel y aceleracion contando transferencias
    double speedup_computo = (med_ker > 0.0) ? (med_cpu / med_ker) : 0.0;
    double speedup_total   = (med_tot > 0.0) ? (med_cpu / med_tot) : 0.0;

    printf("  %-10s %2dx%-2d | %4dx%-4d | blq=%3d | CPU=%6.1f ms | GPU_Ker=%6.3f ms | GPU_Tot=%6.2f ms | SpKer=%6.1fx | SpTot=%5.1fx | MaxErr=%.1e [%s]\n",
           filtro, tam, tam, img->ancho, img->alto, hilos_bloque,
           med_cpu, med_ker, med_tot, speedup_computo, speedup_total,
           max_err, (max_err <= TOLERANCIA) ? "OK" : "FALLA");

    fprintf(csv_resumen, "%s,%dx%d,%dx%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.6e,%.6e\n",
            filtro, img->ancho, img->alto, tam, tam, hilos_bloque,
            med_cpu, iqr_cpu, std_cpu,
            med_h2d, med_ker, iqr_ker, std_ker, med_d2h, med_tot, iqr_tot, std_tot,
            speedup_computo, speedup_total, max_err, mae);

    free(crudo_cpu);
    free(crudo_gpu);
    liberarImagen(sal_cpu);
    liberarImagen(sal_gpu);
}

int main(int argc, char *argv[]) {
#ifdef _WIN32
    system("if not exist resultados mkdir resultados");
#else
    system("mkdir -p resultados");
#endif

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU detectada: %s | Compute Capability: %d.%d | VRAM: %.0f MB\n",
           prop.name, prop.major, prop.minor, (double)prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Nota: cudaMalloc y cudaFree estan excluidos de la medicion de tiempos.\n\n");

    Imagen *base = (argc >= 2) ? leerImagen(argv[1]) : leerImagen("entrada.pgm");
    if (base) printf("Imagen base leida: %dx%d px\n\n", base->ancho, base->alto);
    else      printf("Sin imagen base; usando gradiente sintetico.\n\n");

    // Crear las mascaras de media para 3x3, 9x9 y 21x21
    float m3[9], m9[81], m21[441];
    mascaraMedia(m3, 3);
    mascaraMedia(m9, 9);
    mascaraMedia(m21, 21);

    // Prueba manual con matriz pequena 5x5 en CPU y GPU
    validarManualCPU_GPU();

    FILE *csv_crudo   = fopen("resultados/resultados.csv", "w");
    FILE *csv_resumen = fopen("resultados/resumen.csv", "w");
    fprintf(csv_crudo, "implementacion,filtro,resolucion,mascara,config_hilos_bloques,repeticion,tiempos,unidades\n");
    fprintf(csv_resumen, "filtro,resolucion,mascara,hilos_bloque,cpu_med_ms,cpu_iqr_ms,cpu_std_ms,gpu_h2d_ms,gpu_kernel_ms,gpu_ker_iqr_ms,gpu_ker_std_ms,gpu_d2h_ms,gpu_total_ms,gpu_tot_iqr_ms,gpu_tot_std_ms,aceleracion_computo,aceleracion_transferencias,error_abs_max,mae\n");

    // Prueba con tamano que no es multiplo de 256 hilos por bloque (2050x2050)
    printf("[Validacion Bordes] Dimension no multiplo de bloque (2050x2050)\n");
    Imagen *i_irreg = prepararImagen(base, 2050, 2050);
    experimentoUnificado(i_irreg, m3, 3, "media", HILOS_BASE, "borde_2050", csv_crudo, csv_resumen);
    liberarImagen(i_irreg);

    // Se genera la imagen 4000x4000 una sola vez para usarla en los experimentos 1, 3 y 4
    Imagen *i4k = prepararImagen(base, 4000, 4000);

    // Experimento 1: Comparar los 3 filtros 3x3 sobre 4000x4000
    printf("\n[Exp 1] Filtros 3x3 sobre 4000x4000 (256 hilos/bloque)\n");
    experimentoUnificado(i4k, m3,           3, "media",      HILOS_BASE, "media_4k",      csv_crudo, csv_resumen);
    experimentoUnificado(i4k, NITIDEZ_3,    3, "nitidez",    HILOS_BASE, "nitidez_4k",    csv_crudo, csv_resumen);
    experimentoUnificado(i4k, LAPLACIANO_3, 3, "laplaciano", HILOS_BASE, "laplaciano_4k", csv_crudo, csv_resumen);

    // Experimento 2: Cambiar la resolucion (2000, 4000 y 6000) dejando fija la media 9x9
    printf("\n[Exp 2] Resolucion con media 9x9 (256 hilos/bloque)\n");
    Imagen *i2k = prepararImagen(base, 2000, 2000);
    experimentoUnificado(i2k, m9, 9, "media", HILOS_BASE, "media9_2k", csv_crudo, csv_resumen);
    liberarImagen(i2k);

    experimentoUnificado(i4k, m9, 9, "media", HILOS_BASE, "media9_4k", csv_crudo, csv_resumen);

    Imagen *i6k = prepararImagen(base, 6000, 6000);
    experimentoUnificado(i6k, m9, 9, "media", HILOS_BASE, "media9_6k", csv_crudo, csv_resumen);
    liberarImagen(i6k);

    // Experimento 3: Cambiar el tamano de la mascara (3x3, 9x9, 21x21) sobre 4000x4000
    printf("\n[Exp 3] Tamano de mascara sobre 4000x4000 (256 hilos/bloque)\n");
    experimentoUnificado(i4k, m3,  3,  "media", HILOS_BASE, "media3_4k",   csv_crudo, csv_resumen);
    experimentoUnificado(i4k, m9,  9,  "media", HILOS_BASE, "media9_4k_b", csv_crudo, csv_resumen);
    experimentoUnificado(i4k, m21, 21, "media", HILOS_BASE, "media21_4k",  csv_crudo, csv_resumen);

    // Experimento 4: Probar 128, 256 y 512 hilos por bloque con media 9x9 sobre 4000x4000
    printf("\n[Exp 4] Configuracion de hilos por bloque (128, 256, 512) sobre 4000x4000\n");
    experimentoUnificado(i4k, m9, 9, "media", 128, NULL, csv_crudo, csv_resumen);
    experimentoUnificado(i4k, m9, 9, "media", 256, NULL, csv_crudo, csv_resumen);
    experimentoUnificado(i4k, m9, 9, "media", 512, NULL, csv_crudo, csv_resumen);

    liberarImagen(i4k);
    fclose(csv_crudo);
    fclose(csv_resumen);
    if (base) liberarImagen(base);

    printf("\nListo. Resultados guardados en resultados/resultados.csv y resultados/resumen.csv\n");
    return 0;
}
