# Convolución Paralela de Imágenes

Convolución 2D sobre imágenes en escala de grises con **Python secuencial** (CPU) y **PyCUDA** (GPU), y una aplicación web para aplicar los filtros de forma interactiva.

## Estructura del repositorio

```
.
├── experimentos-pycuda/   Práctica: implementación CPU/GPU, mediciones y análisis
│   ├── src/               cpu_conv.py, gpu_conv.py, main.py
│   ├── analisis/          grafico.ipynb (gráficas comparativas)
│   ├── resultados/        CSV de tiempos e imágenes filtradas
│   ├── entrada.jpg        Imagen base de los experimentos
│   ├── RESULTADOS.md      Análisis de resultados
│   └── requirements.txt
├── app/                   Servicio web FastAPI + PyCUDA (interfaz y API)
│   ├── main.py            Rutas y API
│   ├── filtros.py         Catálogo de filtros
│   ├── servicios/         Motor GPU (contexto, cola, memoria) y procesamiento
│   ├── directo.py         Mismo procesamiento sin HTTP
│   ├── templates/, static/
│   ├── scripts/           Generación de miniaturas
│   └── requirements.txt
├── pruebas/               Validación: matriz/bordes, equivalencia, entradas inválidas, aislamiento
├── Dockerfile, .dockerignore
└── requirements.txt       Instala ambas partes
```

| Parte | Documentación | Cómo ejecutar |
|-------|---------------|---------------|
| Experimentos | [experimentos-pycuda/README.md](experimentos-pycuda/README.md) | `cd experimentos-pycuda && python src/main.py entrada.jpg` |
| App web | [app/README.md](app/README.md) | `uvicorn app.main:app --workers 1` (desde la raíz) o Docker |
| Pruebas | [pruebas/README.md](pruebas/README.md) | `python -m pytest pruebas -v` |

> La implementación original en C secuencial y CUDA C se conserva en la rama `master` (commit `eea17b5`).

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt          # todo
.venv/bin/pip install -r experimentos-pycuda/requirements.txt   # solo experimentos (requiere CUDA)
.venv/bin/pip install -r app/requirements.txt            # solo app web
```

## Docker

### Requisitos del anfitrión

| Componente | Dónde vive | Notas |
|------------|------------|-------|
| Controlador NVIDIA (`libcuda`, `nvidia-smi`) | **Anfitrión** | No va en la imagen. Debe soportar CUDA ≥ 13.4 (la versión "CUDA Version" de `nvidia-smi` en el anfitrión debe ser ≥ 13.4) |
| Runtime de GPU para contenedores | **Anfitrión** | Linux: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) configurado con `nvidia-ctk runtime configure --runtime=docker`. Windows: Docker Desktop con backend WSL2 (incluye el soporte de GPU) |
| CUDA Toolkit 13.4.3 (nvcc, cabeceras, bibliotecas) | **Imagen** | Base `nvidia/cuda:13.4.3-devel-ubuntu24.04`. Se necesita `devel` porque PyCUDA compila el kernel con nvcc al arrancar |
| Python 3.12, PyCUDA, numpy, FastAPI… | **Imagen** | Versiones fijadas en `app/requirements.txt` |

### Comandos

```bash
# 1. Comprobar que Docker puede usar la GPU (con la misma imagen base)
docker run --rm --gpus all nvidia/cuda:13.4.3-devel-ubuntu24.04 nvidia-smi

# 2. Construir la imagen (desde la raíz del repositorio)
docker build -t app-filtros:1.0 .

# 3. Iniciar el servicio (uvicorn, 1 proceso, sin --reload)
docker run -d --name app-filtros --gpus all -p 8000:8000 app-filtros:1.0
#    con límites distintos:  -e MAX_COLA=2 -e MAX_MB=50

# 4. Verificar GPU dentro del contenedor y estado del servicio
docker exec app-filtros nvidia-smi
curl http://localhost:8000/api/salud
docker logs app-filtros            # muestra init_contexto, compilacion_kernel y cada solicitud

# 5. Consumir el servicio
curl -F imagen=@experimentos-pycuda/entrada.jpg -F filtro=media -F mascara=9 \
     -D cabeceras.txt -o resultado.png http://localhost:8000/api/procesar

# 6. PyCUDA directo dentro del contenedor (sin HTTP), con imágenes montadas
docker run --rm --gpus all -v "$PWD/experimentos-pycuda:/datos:ro" app-filtros:1.0 \
     python -m app.directo /datos/entrada.jpg --filtro media --mascara 9 --reps 5

# 7. Pruebas de validación contra el contenedor (desde el anfitrión)
URL_SERVICIO=http://localhost:8000 python -m pytest pruebas -v
```

> En **Git Bash** sobre Windows, antepón `MSYS_NO_PATHCONV=1` a los comandos `docker run -v ...` para que no se conviertan las rutas del contenedor.

### Trazabilidad de una construcción

```bash
git rev-parse HEAD                                              # commit del código
docker image inspect app-filtros:1.0 --format '{{.Id}}'         # ID de la imagen construida
docker image inspect nvidia/cuda:13.4.3-devel-ubuntu24.04 --format '{{index .RepoDigests 0}}'  # digest de la base
docker run --rm --gpus all app-filtros:1.0 python -c "import pycuda, numpy, platform; print(platform.python_version(), pycuda.VERSION_TEXT, numpy.__version__)"
```

Para verificar que la imagen se construye desde una copia limpia:

```bash
git clone <url-del-repositorio> copia-limpia && cd copia-limpia
git checkout <commit>
docker build --no-cache -t app-filtros:verificacion .
```
