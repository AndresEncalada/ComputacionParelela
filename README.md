# Convolución Paralela de Imágenes

Convolución 2D sobre imágenes en escala de grises implementada en C (CPU) y CUDA C (GPU).

## Estructura del código

- `cpu_conv.c`: Funciones de lectura/escritura PGM, generación de máscaras y convolución secuencial en CPU.
- `gpu_conv.cu`: Kernel CUDA 1D (en X), validación numérica (CPU vs GPU) y medición de tiempos con `cudaEvent_t`.
- `main.cu`: Programa principal que integra ambos módulos, ejecuta las validaciones y corre los 4 experimentos en CPU y GPU.

## Filtros

| Filtro | Máscara | Efecto |
|--------|---------|--------|
| Media | k×k, todos los coeficientes = 1/(k²) | Suavizado |
| Nitidez | 3×3 `[0,-1,0; -1,5,-1; 0,-1,0]` | Realce de bordes |
| Laplaciano | 3×3 `[0,1,0; 1,-4,1; 0,1,0]` | Detección de bordes |

Manejo de bordes: zero-padding (los píxeles fuera de la imagen se tratan como 0).

## Requisitos

- CUDA Toolkit (probado con nvcc)
- GPU NVIDIA

## Compilación

```bash
nvcc -O2 -o main main.cu
```

## Uso

```bash
# Ejecución por defecto (usa entrada.pgm o genera gradiente sintético si no existe)
./main

# Pasando una imagen específica
./main imagen.pgm
```

Las imágenes resultantes y los archivos de resultados se guardan en `resultados/`:
- `resultados/resultados.csv`: Tiempos crudos de las 5 repeticiones (CPU, H2D, Kernel, D2H y Conjunto).
- `resultados/resumen.csv`: Medianas, IQR, desviación estándar, aceleración y error numérico (MAE y error máximo).
- `resultados/*_cpu.pgm` y `resultados/*_gpu.pgm`: Imágenes filtradas en CPU y GPU.

## Formato de imagen

Las imágenes de entrada deben estar en escala de grises en formato PGM (P5 binario). Para convertir desde PNG o JPG:

```bash
magick entrada.jpg -colorspace Gray salida.pgm
```
