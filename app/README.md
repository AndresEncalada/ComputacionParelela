# Aplicación web (FastAPI)

Interfaz web para cargar una imagen, elegir el filtro en un carrusel y, para la media, el tamaño de máscara k×k (3–21, impar). Muestra la imagen original, la filtrada y un botón de descarga.

Se ejecuta **desde la raíz del repositorio** (`app` es un paquete de Python):

```bash
pip install -r app/requirements.txt
uvicorn app.main:app --reload   # http://localhost:8000
```

| Archivo | Descripción |
|---------|-------------|
| `main.py` | Rutas: `GET /` (interfaz), `GET /api/filtros` (catálogo), `POST /api/procesar` (multipart: `imagen`, `filtro`, `mascara`; devuelve PNG) |
| `filtros.py` | Catálogo de filtros y límites de máscara (fuente única para frontend y validación) |
| `servicios/procesamiento.py` | `aplicar_filtro()`: punto de integración con `experimentos/src/gpu_conv.py` (**pendiente**, devuelve 501) |
| `templates/`, `static/` | HTML (Jinja2), CSS y JS de la interfaz |
| `scripts/generar_miniaturas.py` | Genera las miniaturas del carrusel recortando el centro de `experimentos/resultados/*_gpu.jpg` |

Mientras `aplicar_filtro()` no esté implementada, la interfaz muestra una vista previa simulada con filtros CSS y deshabilita la descarga.

Regenerar las miniaturas del carrusel:

```bash
python app/scripts/generar_miniaturas.py
```
