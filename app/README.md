# Aplicación web (FastAPI + PyCUDA)

Servicio web que aplica los filtros de convolución de la práctica 02 en la GPU. La interfaz permite cargar una imagen, elegir el filtro (y el tamaño 3×3, 9×9 o 21×21 para la media) y muestra la original, el resultado, los tiempos y un botón de descarga.

```
Navegador ──POST /api/procesar──► FastAPI (main.py) ──► procesamiento.py ──► motor_gpu.py ──► GPU
    ▲                                                     decodificar/codificar   kernel de la práctica 02
    └──────────── PNG + cabeceras X-Tiempo-* ◄────────────┘
```

## Estructura

| Archivo | Responsabilidad |
|---------|-----------------|
| `main.py` | Capa HTTP: rutas, validación de la solicitud, traducción de errores a códigos HTTP, cabeceras de tiempo y registro |
| `servicios/procesamiento.py` | Procesamiento sin HTTP: validar y decodificar → GPU → recorte [0, 255] → PNG/NPY. Lo usan el servicio y `directo.py` |
| `servicios/motor_gpu.py` | Contexto CUDA, compilación del kernel, cola de solicitudes, memoria, transferencias y tiempos con eventos CUDA |
| `practica02.py` | Único punto de importación del kernel (`experimentos-pycuda/src/kernel_conv.py`) y de las máscaras (`cpu_conv.py`) |
| `filtros.py` | Catálogo de filtros, validación de parámetros y construcción de máscaras |
| `config.py` | Límites configurables por variables de entorno |
| `directo.py` | Ejecución directa del mismo procesamiento (sin HTTP) |
| `templates/`, `static/` | Interfaz (HTML, CSS, JS) |

## Ejecución local (sin Docker)

Requiere GPU NVIDIA, CUDA Toolkit con `nvcc` y las dependencias de `requirements.txt`. Desde la raíz del repositorio:

```bash
pip install -r app/requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1   # http://localhost:8000
```

Ejecución directa, sin HTTP (1 calentamiento + N repeticiones, una línea JSON con tiempos por repetición):

```bash
python -m app.directo experimentos-pycuda/entrada.jpg --filtro media --mascara 9 --reps 5 --salida resultado.png
```

## API

### `POST /api/procesar`

Solicitud `multipart/form-data`:

| Campo | Tipo | Obligatorio | Valores |
|-------|------|-------------|---------|
| `imagen` | archivo | sí | JPG, PNG, BMP o PGM (se valida el formato real por contenido, no por la extensión) |
| `filtro` | texto | sí | `media`, `nitidez`, `laplaciano` |
| `mascara` | entero | no | Solo para `media`: `3`, `9` o `21` (por defecto 9), los tamaños de la práctica 02. Los otros filtros siempre usan 3×3 |
| `salida` | texto | no | `png` (por defecto) o `npy` (float32 sin recortar, para validación numérica) |

Ejemplo:

```bash
curl -F imagen=@experimentos-pycuda/entrada.jpg -F filtro=media -F mascara=9 \
     -D cabeceras.txt -o resultado.png http://localhost:8000/api/procesar
```

**Respuesta correcta (200):** el cuerpo es la imagen filtrada en escala de grises (`image/png`). Cabeceras:

| Cabecera | Contenido |
|----------|-----------|
| `X-Request-ID` | Identificador único de la solicitud (UUID4 hex) |
| `X-Imagen`, `X-Filtro`, `X-Mascara` | Dimensiones procesadas, filtro y máscara aplicados |
| `X-Hilos-Bloque`, `X-Bloques` | Configuración de lanzamiento del kernel (1D en X) |
| `Content-Disposition` | Nombre único del resultado: `<original>_<filtro><k>_<id8>.png` |
| `X-Tiempo-Cola-ms` | Espera hasta que la GPU quedó libre (reloj monotónico) |
| `X-Tiempo-Decodificacion-ms` | Validación + decodificación + conversión a grises (CPU) |
| `X-Tiempo-Reserva-ms` | `cuda.mem_alloc` de entrada, salida y máscara |
| `X-Tiempo-H2D-ms` | Copia CPU→GPU de imagen y máscara (eventos CUDA) |
| `X-Tiempo-Kernel-ms` | Ejecución del kernel (eventos CUDA) |
| `X-Tiempo-D2H-ms` | Copia GPU→CPU del resultado float32 (eventos CUDA) |
| `X-Tiempo-Liberacion-ms` | `free()` de la memoria de la GPU |
| `X-Tiempo-GPU-Total-ms` | Todo lo hecho en el hilo de la GPU (reserva + H2D + kernel + D2H + liberación + preparación de arreglos) |
| `X-Tiempo-Codificacion-ms` | Recorte a [0, 255] y codificación PNG (o NPY) |
| `X-Tiempo-Servidor-ms` | Procesamiento completo en el servidor **sin** la espera en cola (no incluye recibir la carga ni enviar la respuesta) |
| `Server-Timing` | Las mismas etapas en formato estándar (visibles en las herramientas del navegador) |

**Errores:** cuerpo JSON `{"detail": "<mensaje>", "codigo": "<código>", "request_id": "<id>"}`.

| HTTP | `codigo` | Causa |
|------|----------|-------|
| 413 | `archivo_demasiado_grande` | El archivo supera `MAX_MB` |
| 413 | `dimensiones_excedidas` | Lado > `MAX_LADO` o píxeles > `MAX_MEGAPIXELES` |
| 415 | `formato_no_valido` | El contenido no es JPG, PNG, BMP ni PGM |
| 422 | `parametro_invalido` | Filtro desconocido, máscara no permitida, salida no válida o campo faltante |
| 422 | `imagen_corrupta` / `archivo_vacio` | La imagen no se puede decodificar o está vacía |
| 503 | `cola_llena` | Hay `MAX_COLA` solicitudes en espera. Incluye `Retry-After: 2` |
| 503 | `gpu_no_disponible` | No hay GPU o CUDA no se pudo inicializar (con diagnóstico) |
| 504 | `tiempo_agotado` | La solicitud superó `TIMEOUT_S` |
| 507 | `memoria_gpu_insuficiente` | `cuda.mem_alloc` falló por falta de memoria |
| 500 | `error_gpu` / `error_interno` | Error de CUDA durante el procesamiento u otro error inesperado |

### `GET /api/salud`

Devuelve 200 si la GPU está lista y 503 si no lo está. Incluye la GPU (nombre, compute capability, memoria), las versiones (CUDA, PyCUDA, numpy, Python), los tiempos de arranque (contexto, compilación, primera ejecución), el estado de la cola y los límites.

### `GET /api/filtros`

Catálogo de filtros, rango de la máscara y límites (lo usa la interfaz).

## Configuración (variables de entorno)

| Variable | Por defecto | Uso |
|----------|-------------|-----|
| `MAX_MB` | 25 | Tamaño máximo del archivo |
| `MAX_LADO` | 8000 | Ancho o alto máximo (px) |
| `MAX_MEGAPIXELES` | 40 | Píxeles máximos (6000×6000 = 36 MP) |
| `MAX_COLA` | 4 | Solicitudes en espera además de la que está en la GPU |
| `HILOS_BLOQUE` | 256 | Hilos por bloque del kernel |
| `TIMEOUT_S` | 120 | Espera máxima por solicitud |
| `LOG_ARCHIVO` | — | Ruta para guardar el registro además de mostrarlo en la salida estándar |
| `RETARDO_PRUEBA_MS` | 0 | **Solo para pruebas**: añade un retardo por solicitud para forzar la cola llena. No usar al medir |

Límites de memoria: con 40 MP, la GPU necesita 40 MB (entrada uint8) + 160 MB (salida float32) por solicitud, lo que cabe en una GPU de 4 GB. En el host, cada solicitud usa unos 400 MB en el peor caso (RGB decodificado + grises + float32 + PNG), y como máximo hay `MAX_COLA + 1` solicitudes en memoria.

## Integración con PyCUDA

- **Contexto CUDA:** lo crea un único hilo trabajador (`gpu-worker`) con `cuda.init()` + `Device(0).make_context()` al arrancar el servidor, antes de aceptar solicitudes. No se usa `pycuda.autoinit` porque ataría el contexto al hilo principal. Todas las llamadas CUDA ocurren en ese hilo y el contexto se libera (`pop` + `detach`) al apagar.
- **Kernel:** se compila una sola vez con `SourceModule` (nvcc) desde `experimentos-pycuda/src/kernel_conv.py`, el mismo código de la práctica 02 (1 píxel por hilo, distribución 1D en X, zero-padding, salida float32).
- **Arranque en frío:** el tiempo de creación del contexto, el de compilación y el de una primera ejecución (64×64) se registran por separado en el log y en `/api/salud`. No forman parte de los tiempos por solicitud.
- **Memoria:** en cada solicitud se hace `mem_alloc` de entrada (uint8), salida (float32) y máscara, y `free()` en un bloque `finally`, de modo que la memoria se libera aunque haya error. Las transferencias se hacen desde memoria paginable (arreglos numpy normales).
- **Sincronización y tiempos:** se usa `Event.record()` antes y después de H2D, del kernel y de D2H, y `Event.synchronize()` antes de consultar `time_since`. El resto de etapas se mide con `time.perf_counter` (reloj monotónico).
- **Atención de solicitudes:** cola FIFO acotada (`MAX_COLA`) y una sola solicitud a la vez en la GPU. Si la cola está llena, se rechaza de inmediato con 503. La decodificación y la codificación corren en el pool de hilos del servidor, en paralelo con la GPU. Se usa **un solo proceso** de uvicorn (`--workers 1`) porque cada proceso crearía su propio contexto CUDA.
- **Errores:** si no hay GPU, nvcc o PyCUDA, el servicio arranca igual, pero `/api/salud` y `/api/procesar` responden 503 con el diagnóstico. **No se recurre a la CPU.** Los errores de memoria se convierten en 507.

## Registro (logs)

Todo el registro, incluido el de uvicorn, usa el mismo formato de columnas ([registro.py](registro.py)): `fecha y hora (ms) | nivel | origen | mensaje`. Al arrancar se escribe un bloque con la GPU, las versiones, los tiempos de arranque en frío y los límites. Después se escribe una línea por solicitud, con su ID corto:

```
2026-11-20 06:42:37.291 | INFO  | arranque  | GPU ............ NVIDIA GeForce RTX 3050 Laptop GPU (CC 8.6, 4096 MB)
2026-11-20 06:42:37.291 | INFO  | arranque  | Arranque ....... contexto 281.13 ms | compilacion kernel 766.71 ms | primera ejecucion 5.90 ms
2026-11-20 06:42:38.629 | INFO  | solicitud | [3ca0099d] 200 OK     media      21x21   6000x4000 | cola    0.10 ms | h2d   3.28 ms | kernel   83.48 ms | d2h  21.37 ms | servidor   724.72 ms
2026-11-20 06:42:38.755 | WARN  | solicitud | [0be22e5e] 422 parametro_invalido       sepia        3x3 | 0.52 ms | Filtro desconocido: 'sepia'. ...
2026-11-20 06:42:38.755 | INFO  | http      | POST /api/procesar          -> 422  (172.17.0.1:49356)
```

Las solicitudes rechazadas se registran como `WARN` y los fallos del servidor como `ERROR`. Para conservar el registro durante los experimentos-pycuda:

```bash
docker logs app-filtros > registros/servicio.log                       # volcado del contenedor
docker run ... -e LOG_ARCHIVO=/registros/servicio.log -v "$PWD/registros:/registros" app-filtros:1.0   # archivo continuo
```

En `python -m app.directo` el registro sale por stderr y las líneas JSON de tiempos por stdout, para poder separarlos (`2> arranque.log > tiempos.jsonl`).

## Formato del CSV de mediciones (acordado)

Las mediciones del servicio y de la ejecución directa se registran en formato largo, con **una fila por métrica**, igual que `resultados.csv` de la práctica 02:

```
request_id,modo,filtro,resolucion,mascara,hilos_bloque,bloques,repeticion,concurrencia,ronda,metrica,tiempo,unidades,estado,codigo_error,entorno,gpu,commit
```

| Columna | Valores |
|---------|---------|
| `request_id` | `X-Request-ID` (servicio) o `directo-<n>` |
| `modo` | `directo_anfitrion`, `directo_contenedor`, `servicio` |
| `resolucion`, `mascara` | `4000x4000`, `9x9` |
| `repeticion` | `0` = calentamiento (no entra en estadísticas), `1..5` = medidas |
| `concurrencia`, `ronda` | Clientes simultáneos (1 o 2) y ronda (1..5); `1` y vacío fuera de esa prueba |
| `metrica` | `cola`, `decodificacion`, `reserva`, `h2d`, `kernel`, `d2h`, `liberacion`, `gpu_total`, `codificacion`, `servidor` (cabeceras `X-Tiempo-*`) y `cliente_total` (medido en el cliente con `time.perf_counter`) |
| `unidades` | `ms` |
| `estado` | `completada`, `no_completada`, `rechazada` (503 cola llena), `error` |
| `codigo_error` | Campo `codigo` del JSON de error, o vacío |
| `entorno` | `anfitrion` o `contenedor` + sistema operativo |
| `gpu`, `commit` | Nombre de la GPU (`/api/salud`) y `git rev-parse --short HEAD` |

## Solicitudes y archivos

- Cada solicitud recibe un UUID (`X-Request-ID`) que aparece en las cabeceras, en el cuerpo de los errores y en el registro.
- El procesamiento se hace **en memoria**: no se escriben archivos intermedios ni de resultado en el servidor, así que una solicitud no puede sobrescribir ni recibir la imagen de otra. La carga se guarda en un `SpooledTemporaryFile` de Starlette (en disco solo si supera 1 MB), que se cierra y elimina al terminar cada solicitud (`imagen.close()` en `finally`).
- El nombre de descarga es único: `<original>_<filtro><k>_<id8>.png`.

## Pruebas

Ver [pruebas/README.md](../pruebas/README.md).

Para regenerar las miniaturas del carrusel:

```bash
python app/scripts/generar_miniaturas.py
```
