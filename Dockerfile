# Servicio web de convolucion con PyCUDA.
#
# Imagen base: CUDA 13.4.3 "devel" sobre Ubuntu 24.04. Se usa devel (no runtime) porque
# PyCUDA compila el kernel con nvcc en tiempo de ejecucion (SourceModule).
# El controlador NVIDIA NO va en la imagen: lo aporta el anfitrion a traves del runtime
# de contenedores de NVIDIA (docker run --gpus all). El controlador del anfitrion debe
# soportar CUDA >= 13.4 (ver README).
FROM nvidia/cuda:13.4.3-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_ROOT=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}

# Python 3.12 de Ubuntu 24.04 + compilador para construir PyCUDA
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv python3-dev build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

WORKDIR /srv

# Dependencias primero (capa reutilizable). PyCUDA se compila contra la libcuda "stub"
# de la imagen; en ejecucion usa la libcuda real del controlador del anfitrion.
COPY app/requirements.txt app/requirements.txt
RUN LIBRARY_PATH=/usr/local/cuda/lib64/stubs pip install --no-cache-dir -r app/requirements.txt

# Kernel y mascaras de la practica 02 + aplicacion
COPY experimentos-pycuda/src experimentos-pycuda/src
COPY app app

# Usuario sin privilegios (PyCUDA guarda su cache de compilacion en $HOME/.cache)
RUN useradd --create-home --uid 10001 servicio
USER servicio

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/salud').status == 200 else 1)"

# Servidor de produccion: sin --reload y con 1 solo proceso (un contexto CUDA)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
