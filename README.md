# Convolución Paralela de Imágenes (Python Secuencial y PyCUDA)

Convolución 2D sobre imágenes en escala de grises implementada en **Python secuencial puro** (CPU) y **PyCUDA** (GPU), continuando la Práctica 01 en C/CUDA C (conservada en la rama `master`, commit `eea17b5`).

## Estructura del código

| Archivo | Descripción |
|---------|-------------|
| `src/cpu_conv.py` | Lectura/escritura de imágenes, generación de máscaras y convolución 2D secuencial píxel a píxel en Python puro (límite configurable de 30 min por configuración) |
| `src/gpu_conv.py` | Kernel CUDA 1D compilado con `pycuda.compiler.SourceModule`, validación numérica CPU vs GPU y medición de tiempos con `pycuda.driver.Event` |
| `src/main.py` | Programa principal: integra ambos módulos, registra inicialización de contexto y compilación del kernel, ejecuta validaciones y los 4 experimentos |
| `analisis/grafico.ipynb` | Notebook en Python para generar las gráficas comparativas |
| `requirements.txt` | Dependencias de Python y versiones probadas |

> La implementación original de referencia en C secuencial y CUDA C (`src/cpu_conv.c`, `src/gpu_conv.cu`, `src/main.cu`) se conserva intacta en la rama `master` (commit `eea17b5`).

## Filtros implementados

| Filtro | Máscara | Efecto |
|--------|---------|--------|
| Media | k×k, todos los coeficientes = 1/(k²) | Suavizado |
| Nitidez | 3×3 `[0,-1,0; -1,5,-1; 0,-1,0]` | Realce de bordes |
| Laplaciano | 3×3 `[0,1,0; 1,-4,1; 0,1,0]` | Detección de bordes |

Manejo de bordes: zero-padding (los píxeles fuera de la imagen se tratan como 0).

## Dependencias y entorno

- **Python** 3.14+ (probado con 3.14.10)
- **CUDA Toolkit** con `nvcc` (probado con CUDA 13.4) y **GPU NVIDIA**
- Paquetes en `requirements.txt`:
  - `numpy==2.5.4`
  - `pillow==13.0.0` (solo para lectura/escritura de imágenes JPG/PNG/BMP)
  - `pycuda==2026.1`

### Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Ejecución

El programa acepta cualquier imagen en formato JPG, PNG, BMP o PGM y la convierte automáticamente a escala de grises. Si no se pasa ruta, usa `entrada.jpg` por defecto (o un gradiente sintético si no existe).

```bash
# Con la imagen del repositorio:
.venv/bin/python src/main.py entrada.jpg

# Solo validación manual 5x5 en CPU:
.venv/bin/python src/cpu_conv.py
```

## Pruebas, validaciones y mediciones

1. **Validación manual 5×5 (CPU y GPU):** aplica un filtro de media 3×3 sobre una matriz conocida y comprueba centro `(2,2)` = **50.0000** y esquina `(0,0)` = **8.8889** (byte **8**) con 32 hilos en GPU.
2. **Validación de bordes 2050×2050:** dimensión no múltipla de 256 hilos/bloque para verificar el guard `idx >= ancho * alto`.
3. **Medición de rendimiento:**
   - **Preparación:** inicialización del contexto CUDA (`pycuda.autoinit`) y compilación/carga del kernel (`SourceModule`) se registran por separado al inicio.
   - **Reserva de memoria:** `cuda.mem_alloc` y `free` quedan **excluidos** del tiempo conjunto (`H2D + Kernel + D2H`).
   - **Repeticiones:** 1 calentamiento + 5 repeticiones medidas (mediana, IQR y desviación estándar).
   - **Límite CPU:** 30 minutos (`1800 s`) por configuración; si se supera, se marca como `no_completada` sin calcular aceleraciones y continúa la ejecución en GPU.

## Experimentos

| # | Variable | Configuración fija | Valores explorados | Versiones |
|---|----------|-------------------|-------------------|-----------|
| 1 | Tipo de filtro | 4000×4000, 256 hilos | media, nitidez, laplaciano (3×3) | CPU y GPU |
| 2 | Resolución | media 9×9, 256 hilos | 2000², 4000², 6000² | CPU y GPU |
| 3 | Tamaño de máscara | 4000×4000, 256 hilos | 3×3, 9×9, 21×21 | CPU y GPU |
| 4 | Hilos por bloque | media 9×9, 4000×4000 | 128, 256, 512 | Solo GPU |

## Resultados

Los resultados se guardan en `resultados/`:
- `resultados.csv`: tiempos crudos por repetición (`CPU`, `GPU_H2D`, `GPU_Kernel`, `GPU_D2H`, `GPU_Conjunto`) incluyendo la columna `estado` (`completada` / `no_completada`).
- `resumen.csv`: medianas, IQR, desviación estándar, aceleración de cómputo, aceleración con transferencias y error numérico máximo/MAE.
- `*_cpu.jpg` y `*_gpu.jpg`: imágenes filtradas de salida.
