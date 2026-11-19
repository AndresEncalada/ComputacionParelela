# Resultados — Comparación de Convolución 2D: C, CUDA C, Python y PyCUDA

## Entorno de ejecución (mismo equipo para las 4 versiones)

- **CPU:** Intel(R) Core(TM) 7 240H (16 hilos)
- **RAM:** 15 GiB
- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8151 MiB VRAM, CC 12.0)
- **Sistema Operativo:** Fedora Linux 44
- **Driver / CUDA:** Driver `615.78.08` · CUDA Toolkit `13.4` (`nvcc 13.4.92`)
- **Python / Librerías:** Python `3.14.10` · `numpy==2.5.4` · `pillow==13.0.0` · `pycuda==2026.1`
- **Imagen base:** `entrada.jpg` (6000×4000 px)
- **Límite Python secuencial:** 30 minutos (`1800 s`) por configuración (`no_completada` sin cálculo de aceleración si se supera)
- **Reserva de memoria GPU:** `cudaMalloc`/`cudaFree` y `mem_alloc`/`free` excluidos del tiempo conjunto

---

## Validación manual 5×5 (media 3×3)

| Píxel | Esperado | C / Python CPU | CUDA C / PyCUDA GPU | Estado |
|---|---|---|---|---|
| Central (2,2) | 50.0000 (50) | 50.0000 (50) | 50.0000 | OK |
| Esquina (0,0) | 8.8889 (8) | 8.8889 (8) | 8.8889 | OK |

---

## Validación de bordes (2050×2050, media 3×3, 256 hilos)

| Versión | CPU (ms) | GPU H2D (ms) | GPU Kernel (ms) | GPU D2H (ms) | GPU Total (ms) | Acel. Kernel | Acel. Total | Error máx. | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C / CUDA C | 81.19 | 0.434 | 0.157 | 1.557 | 2.155 | 516.31× | 37.67× | 4.58e-05 | 3.96e-06 |
| Python / PyCUDA | 10854.10 | 0.469 | 0.194 | 1.558 | 2.220 | 56018.25× | 4888.32× | 3.90e-05 | 5.79e-06 |

---

## Exp 1 — Tipo de filtro (3×3, 4000×4000, 256 hilos)

| Filtro | C CPU (ms) | CUDA C Ker (ms) | CUDA C Tot (ms) | Python CPU (ms) | PyCUDA Ker (ms) | PyCUDA Tot (ms) | Acel. CUDA C (Ker / Tot) | Acel. PyCUDA (Ker / Tot) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Media | 319.44 | 0.719 | 8.28 | 42955.48 | 0.770 | 9.11 | 444.18× / 38.57× | 55755.06× / 4715.64× |
| Nitidez | 325.62 | 0.717 | 8.45 | 24355.15 | 0.718 | 6.35 | 454.07× / 38.56× | 33941.24× / 3835.41× |
| Laplaciano | 326.34 | 0.719 | 8.10 | 24349.59 | 0.717 | 6.45 | 453.81× / 40.30× | 33963.79× / 3776.49× |

---

## Exp 2 — Resolución (media 9×9, 256 hilos)

| Resolución | C CPU (ms) | CUDA C Ker (ms) | CUDA C Tot (ms) | Python CPU (ms) | PyCUDA Ker (ms) | PyCUDA Tot (ms) | Acel. CUDA C (Ker / Tot) | Acel. PyCUDA (Ker / Tot) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000×2000 | 634.60 | 0.790 | 2.66 | 50546.22 | 0.673 | 2.23 | 802.95× / 238.88× | 75060.33× / 22707.69× |
| 4000×4000 | 2495.85 | 3.419 | 11.02 | 217254.83 | 2.716 | 8.52 | 730.01× / 226.39× | 80000.16× / 25490.97× |
| 6000×6000 | 5581.12 | 8.283 | 25.43 | `no_completada` (>30 min) | 6.086 | 19.29 | 673.83× / 219.44× | N/A |

---

## Exp 3 — Tamaño de máscara (media, 4000×4000, 256 hilos)

| Máscara | C CPU (ms) | CUDA C Ker (ms) | CUDA C Tot (ms) | Python CPU (ms) | PyCUDA Ker (ms) | PyCUDA Tot (ms) | Acel. CUDA C (Ker / Tot) | Acel. PyCUDA (Ker / Tot) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3×3 | 319.60 | 0.717 | 8.72 | 24274.60 | 0.721 | 6.26 | 445.83× / 36.66× | 33675.81× / 3878.21× |
| 9×9 | 2531.98 | 3.439 | 10.97 | 186354.84 | 2.725 | 8.63 | 736.20× / 230.79× | 68395.33× / 21601.57× |
| 21×21 | 15869.68 | 17.608 | 25.82 | `no_completada` (>30 min) | 13.683 | 19.89 | 901.29× / 614.62× | N/A |

---

## Exp 4 — Hilos por bloque (media 9×9, 4000×4000, CUDA C vs PyCUDA)

| Hilos / bloque | CUDA C Ker (ms) | CUDA C Tot (ms) | CUDA C Ker IQR (ms) | PyCUDA Ker (ms) | PyCUDA Tot (ms) | PyCUDA Ker IQR (ms) |
|---|---:|---:|---:|---:|---:|---:|
| 128 | 3.446 | 11.59 | 0.189 | 2.702 | 8.55 | 0.007 |
| 256 | 3.372 | 11.16 | 0.086 | 2.694 | 8.22 | 0.020 |
| 512 | 3.404 | 11.23 | 0.035 | 2.744 | 8.37 | 0.026 |
