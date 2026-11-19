# Convolución Paralela de Imágenes

Convolución 2D sobre imágenes en escala de grises con **Python secuencial** (CPU) y **PyCUDA** (GPU), y una aplicación web para aplicar los filtros de forma interactiva.

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
├── app/                   Aplicación web FastAPI (interfaz lista, lógica pendiente)
│   ├── main.py            Rutas y API
│   ├── filtros.py         Catálogo de filtros
│   ├── servicios/         Integración con la convolución
│   ├── templates/, static/
│   ├── scripts/           Generación de miniaturas
│   └── requirements.txt
└── requirements.txt       Instala ambas partes
```

| Parte | Documentación | Cómo ejecutar |
|-------|---------------|---------------|
| Experimentos | [experimentos/README.md](experimentos/README.md) | `cd experimentos && python src/main.py entrada.jpg` |
| App web | [app/README.md](app/README.md) | `uvicorn app.main:app --reload` (desde la raíz) |

> La implementación original en C secuencial y CUDA C se conserva en la rama `master` (commit `eea17b5`).

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # todo
.venv/bin/pip install -r experimentos/requirements.txt   # solo experimentos (requiere CUDA)
.venv/bin/pip install -r app/requirements.txt            # solo app web
```
