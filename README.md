# Convolución Paralela de Imágenes

Convolución 2D sobre imágenes en escala de grises implementada en C (CPU) y CUDA C (GPU).

## Filtros

| Filtro | Máscara | Efecto |
|--------|---------|--------|
| Media | k×k, todos los coeficientes = 1/(k²) | Suavizado |
| Nitidez | 3×3 `[0,-1,0; -1,5,-1; 0,-1,0]` | Realce de bordes |
| Laplaciano | 3×3 `[0,1,0; 1,-4,1; 0,1,0]` | Detección de bordes |

Manejo de bordes: zero-padding (los píxeles fuera de la imagen se tratan como 0).

## Requisitos

- CUDA Toolkit (probado con nvcc)
- GPU NVIDIA (para la versión GPU)

## Compilación

```bash
# Versión CPU
nvcc -O2 -o cpu_conv cpu_conv.c

# Versión GPU
nvcc -O2 -o gpu_conv gpu_conv.cu
```

## Uso

```bash
# Con una imagen
cpu_conv imagen.pgm

# Sin imagen (genera un gradiente sintético)
cpu_conv
```

Las imágenes resultantes y los tiempos de ejecución se guardan en `resultados/`.

## Formato de imagen

Las imágenes de entrada deben estar en escala de grises en formato PGM (P5 binario). Para convertir desde PNG o JPG:

```bash
magick entrada.jpg -colorspace Gray salida.pgm
```
