"""
generar_miniaturas.py - Recorta un trozo central de las imagenes de resultados
y genera las miniaturas circulares del carrusel de filtros.

Uso (desde la raiz del repositorio):
    python app/scripts/generar_miniaturas.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

RAIZ = Path(__file__).resolve().parents[2]
RESULTADOS = RAIZ / "experimentos-pycuda" / "resultados"
DESTINO = RAIZ / "app" / "static" / "img" / "filtros"

# id del filtro -> imagen de resultados de origen
ORIGENES = {
    "media": RESULTADOS / "media21_4k_gpu.jpg",
    "nitidez": RESULTADOS / "nitidez_4k_gpu.jpg",
    "laplaciano": RESULTADOS / "laplaciano_4k_gpu.jpg",
}

LADO_RECORTE = 0.42   # fraccion del lado de la imagen que se recorta
CENTRO = (0.56, 0.58)  # centro del recorte (x, y) relativo; enfoca al sujeto
LADO_MINIATURA = 192   # px (se muestra a ~72 px, sirve para pantallas HiDPI)

# El laplaciano deja casi todo en negro; solo para la miniatura se estira el contraste.
REALZAR = {"laplaciano"}


def recortar_central(ruta: Path) -> Image.Image:
    img = Image.open(ruta)
    ancho, alto = img.size
    lado = int(min(ancho, alto) * LADO_RECORTE)
    cx, cy = int(ancho * CENTRO[0]), int(alto * CENTRO[1])
    izq = max(0, min(ancho - lado, cx - lado // 2))
    sup = max(0, min(alto - lado, cy - lado // 2))
    recorte = img.crop((izq, sup, izq + lado, sup + lado))
    return recorte.resize((LADO_MINIATURA, LADO_MINIATURA), Image.LANCZOS)


def main() -> None:
    DESTINO.mkdir(parents=True, exist_ok=True)
    for filtro, origen in ORIGENES.items():
        salida = DESTINO / f"{filtro}.jpg"
        mini = recortar_central(origen)
        if filtro in REALZAR:
            mini = ImageOps.autocontrast(mini, cutoff=1)
        mini.save(salida, quality=90)
        print(f"{origen.name} -> {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
