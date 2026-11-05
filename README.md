# Convolución Paralela de Imágenes

Convolución 2D sobre imágenes en escala de grises implementada en C (CPU) y CUDA C (GPU).

## Estructura del código

| Archivo | Descripción |
|---------|-------------|
| `cpu_conv.c` | Lectura/escritura de imágenes, generación de máscaras y convolución secuencial en CPU |
| `gpu_conv.cu` | Kernel CUDA 1D, validación numérica CPU vs GPU y medición de tiempos con `cudaEvent_t` |
| `main.cu` | Programa principal: integra ambos módulos, ejecuta validaciones y los 4 experimentos |
| `stb_image.h` | Cabecera de dominio público para leer JPG/PNG/BMP (sin dependencias externas) |
| `stb_image_write.h` | Cabecera de dominio público para escribir JPG (sin dependencias externas) |

## Filtros implementados

| Filtro | Máscara | Efecto |
|--------|---------|--------|
| Media | k×k, todos los coeficientes = 1/(k²) | Suavizado |
| Nitidez | 3×3 `[0,-1,0; -1,5,-1; 0,-1,0]` | Realce de bordes |
| Laplaciano | 3×3 `[0,1,0; 1,-4,1; 0,1,0]` | Detección de bordes |

Manejo de bordes: zero-padding (los píxeles fuera de la imagen se tratan como 0).

## Dependencias

- **CUDA Toolkit** con `nvcc` (probado con CUDA 12.x)
- **GPU NVIDIA** con Compute Capability ≥ 3.5
- `stb_image.h` y `stb_image_write.h` — incluidas en el repositorio, no requieren instalación

## Compilación

```bash
nvcc -O2 -diag-suppress 550 -o main main.cu
```

La bandera `-diag-suppress 550` silencia advertencias de variables sin usar generadas internamente por stb_image.

## Datos de entrada

El programa acepta cualquier imagen en formato JPG, PNG o BMP y la convierte automáticamente a escala de grises.

```bash
# Con una imagen propia:
./main foto.jpg

# Sin imagen: el programa genera un gradiente sintético de prueba
./main
```

Para reproducir exactamente los experimentos del informe, se recomienda una imagen de al menos 6000×6000 píxeles (la más grande que usa el Experimento 2).

## Pruebas y validaciones

El programa ejecuta dos validaciones automáticas al arrancar, antes de los experimentos:

1. **Validación manual 5×5 (CPU y GPU):** aplica un filtro de media 3×3 sobre una matriz conocida y comprueba dos píxeles calculados a mano:
   - Centro (2,2): suma de vecinos = 450 → 450/9 = **50.0000**
   - Esquina (0,0): con zero-padding = 80/9 = **8.8889** → byte **8**
   El mismo cálculo se ejecuta en GPU con 32 hilos (25 píxeles útiles + 7 que prueban el guard de límites). Ambos deben imprimir `OK`.

2. **Validación de bordes 2050×2050:** dimensión no múltipla de 256 hilos/bloque — verifica que el guard `idx ≥ ancho×alto` funciona correctamente y que CPU y GPU producen el mismo resultado numérico.

## Experimentos

| # | Variable | Configuración fija | Valores explorados |
|---|----------|-------------------|-------------------|
| 1 | Tipo de filtro | 4000×4000, 256 hilos | media, nitidez, laplaciano (3×3) |
| 2 | Resolución | media 9×9, 256 hilos | 2000², 4000², 6000² |
| 3 | Tamaño de máscara | 4000×4000, 256 hilos | 3×3, 9×9, 21×21 |
| 4 | Hilos por bloque | media 9×9, 4000×4000 | 128, 256, 512 |

Cada configuración ejecuta 1 calentamiento + 5 repeticiones medidas. Se reporta la mediana, IQR y desviación estándar.

## Resultados

Los resultados se guardan en `resultados/`:

| Archivo | Contenido |
|---------|-----------|
| `resultados.csv` | Tiempos crudos de las 5 repeticiones por configuración (CPU, H2D, Kernel, D2H, Total) |
| `resumen.csv` | Medianas, IQR, std, aceleración de cómputo, aceleración con transferencias y error numérico máximo |
| `*_cpu.jpg` | Imágenes filtradas producidas por la implementación CPU |
| `*_gpu.jpg` | Imágenes filtradas producidas por la implementación GPU |

### Generar gráficas

Los CSV pueden abrirse directamente en Excel, LibreOffice Calc o cualquier herramienta de análisis de datos. Para graficar en Python revise el archivo `grafico.ipynb`.
