# Convolución Paralela de Imágenes

Convolución 2D sobre imágenes en escala de grises con **Python secuencial** (CPU) y **PyCUDA** (GPU).

## Estructura del repositorio

```
.
├── experimentos/          Práctica: implementación CPU/GPU, mediciones y análisis
│   ├── src/               cpu_conv.py, gpu_conv.py, main.py
│   ├── analisis/          grafico.ipynb (gráficas comparativas)
│   ├── resultados/        CSV de tiempos e imágenes filtradas
│   ├── entrada.jpg        Imagen base de los experimentos
│   ├── RESULTADOS.md      Análisis de resultados
│   └── requirements.txt
└── requirements.txt       Dependencias del proyecto
```

| Parte | Documentación | Cómo ejecutar |
|-------|---------------|---------------|
| Experimentos | [experimentos/README.md](experimentos/README.md) | `cd experimentos && python src/main.py entrada.jpg` |

> La implementación original en C secuencial y CUDA C se conserva en la rama `master` (commit `eea17b5`).

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
