# Resultados — Convolución 2D CPU (Python) vs GPU (PyCUDA)

GPU: NVIDIA GeForce RTX 3050 Laptop GPU · CC 8.6 · 4096 MB
Imagen base: `entrada.jpg` (6000×4000 px)
Límite CPU: 120 s por convolución

## Validación manual 5×5 (media 3×3)

| Píxel | Esperado | CPU | GPU | Estado |
|---|---|---|---|---|
| Central (2,2) | 50.0000 | 50.0000 | 50.0000 | OK |
| Esquina (0,0) | 8.8889 | 8.8889 | 8.8889 | OK |

## Validación de bordes (2050×2050, media 3×3, 256 hilos)

| CPU (ms) | GPU kernel (ms) | GPU total (ms) | Acel. kernel | Acel. total | Error máx. | Estado |
|---|---|---|---|---|---|---|
| 8647.05 | 0.440 | 2.94 | 19638.10× | 2944.70× | 3.90e-05 | OK |

## Exp 1 — Filtros 3×3 (4000×4000, 256 hilos)

| Filtro | CPU (ms) | GPU H2D (ms) | GPU kernel (ms) | GPU D2H (ms) | GPU total (ms) | Acel. kernel | Acel. total | Error máx. |
|---|---|---|---|---|---|---|---|---|
| Media | 33190.59 | 2.063 | 1.587 | 7.723 | 11.47 | 20911.41× | 2893.59× | 4.07e-05 |
| Nitidez | 32894.28 | 2.087 | 1.590 | 7.974 | 11.71 | 20685.10× | 2808.03× | 0 |
| Laplaciano | 32935.08 | 1.710 | 1.575 | 7.314 | 10.59 | 20914.45× | 3109.12× | 0 |

## Exp 2 — Resolución (media 9×9, 256 hilos)

| Resolución | CPU (ms) | GPU H2D (ms) | GPU kernel (ms) | GPU D2H (ms) | GPU total (ms) | Acel. kernel | Acel. total | Error máx. |
|---|---|---|---|---|---|---|---|---|
| 2000×2000 | 93404.69 | 0.567 | 2.454 | 2.696 | 5.72 | 38069.91× | 16343.85× | 2.08e-04 |
| 4000×4000 | >120 s | 1.752 | 9.546 | 7.562 | 18.86 | >12571.07× | >6362.97× | 2.67e-04* |
| 6000×6000 | >120 s | 4.757 | 21.393 | 18.507 | 44.05 | >5609.25× | >2724.28× | 1.99e-04* |

## Exp 3 — Tamaño de máscara (media, 4000×4000, 256 hilos)

| Máscara | CPU (ms) | GPU H2D (ms) | GPU kernel (ms) | GPU D2H (ms) | GPU total (ms) | Acel. kernel | Acel. total | Error máx. |
|---|---|---|---|---|---|---|---|---|
| 3×3 | 35706.16 | 2.039 | 1.599 | 8.578 | 12.22 | 22326.17× | 2922.89× | 4.07e-05 |
| 9×9 | >120 s | 1.793 | 9.551 | 9.466 | 20.77 | >12564.75× | >5778.84× | 2.72e-04* |
| 21×21 | >120 s | 1.986 | 46.718 | 7.665 | 56.24 | >2568.61× | >2133.61× | 3.83e-04* |

## Exp 4 — Hilos por bloque (media 9×9, 4000×4000)

| Hilos/bloque | CPU (ms) | GPU H2D (ms) | GPU kernel (ms) | GPU D2H (ms) | GPU total (ms) | Acel. kernel | Acel. total | Error máx. |
|---|---|---|---|---|---|---|---|---|
| 128 | >120 s | 1.855 | 9.570 | 7.674 | 19.23 | >12538.79× | >6240.28× | 2.72e-04* |
| 256 | >120 s | 1.920 | 9.562 | 7.776 | 19.37 | >12549.53× | >6193.89× | 2.72e-04* |
| 512 | >120 s | 2.248 | 9.547 | 9.719 | 21.80 | >12569.72× | >5505.83× | 2.72e-04* |

\* Error calculado sobre los píxeles procesados por la CPU antes del corte.
