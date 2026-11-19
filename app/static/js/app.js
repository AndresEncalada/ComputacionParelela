// app.js - Interaccion de la interfaz. La logica de convolucion vive en el
// backend (POST /api/procesar); aqui solo se arma la peticion y se muestra
// la respuesta. Mientras el backend devuelva 501 se muestra una vista previa
// simulada con filtros CSS para poder revisar el diseno.
(() => {
  "use strict";

  const CONFIG = window.CONFIG;
  const $ = (id) => document.getElementById(id);

  const ui = {
    input: $("input-imagen"),
    zona: $("zona-carga"),
    zonaVacia: $("zona-vacia"),
    vistaPrevia: $("vista-previa"),
    nombre: $("nombre-archivo"),
    info: $("info-archivo"),
    btnCambiar: $("btn-cambiar"),
    carrusel: $("carrusel"),
    flechaIzq: $("flecha-izq"),
    flechaDer: $("flecha-der"),
    descripcion: $("descripcion-filtro"),
    controlMascara: $("control-mascara"),
    inputMascara: $("input-mascara"),
    valorMascara: $("valor-mascara"),
    chips: document.querySelectorAll(".chip[data-k]"),
    btnAplicar: $("btn-aplicar"),
    estado: $("estado"),
    resultados: $("resultados"),
    etiquetas: $("etiquetas"),
    avisoSimulado: $("aviso-simulado"),
    imgOriginal: $("img-original"),
    imgResultado: $("img-resultado"),
    btnDescargar: $("btn-descargar"),
  };

  const estado = {
    archivo: null,
    urlOriginal: null,
    urlResultado: null,
    filtro: CONFIG.filtros[0].id,
    mascara: CONFIG.mascara.defecto,
  };

  const filtroActual = () => CONFIG.filtros.find((f) => f.id === estado.filtro);

  // ---------- Carga de imagen ----------
  function cargarArchivo(archivo) {
    if (!archivo || (!archivo.type.startsWith("image/") && !archivo.name.endsWith(".pgm"))) {
      mostrarEstado("Selecciona un archivo de imagen válido.", true);
      return;
    }
    if (estado.urlOriginal) URL.revokeObjectURL(estado.urlOriginal);
    estado.archivo = archivo;
    estado.urlOriginal = URL.createObjectURL(archivo);

    ui.vistaPrevia.onload = () => {
      const { naturalWidth: w, naturalHeight: h } = ui.vistaPrevia;
      ui.info.textContent = `${w} × ${h} px · ${formatearBytes(archivo.size)}`;
      // Proporcion del marco limitada al rango de Instagram (4:5 a 1.91:1)
      const proporcion = Math.min(1.91, Math.max(0.8, w / h));
      document.documentElement.style.setProperty("--proporcion", proporcion.toFixed(3));
    };
    ui.vistaPrevia.src = estado.urlOriginal;
    ui.vistaPrevia.hidden = false;
    ui.zonaVacia.hidden = true;
    ui.zona.classList.add("zona-carga--cargada");
    ui.nombre.textContent = archivo.name;
    ui.btnCambiar.hidden = false;
    ui.btnAplicar.disabled = false;
    ui.resultados.hidden = true;
    mostrarEstado("");
  }

  ui.input.addEventListener("change", () => cargarArchivo(ui.input.files[0]));
  ui.btnCambiar.addEventListener("click", () => ui.input.click());
  ui.zona.addEventListener("click", (e) => {
    if (estado.archivo) e.preventDefault(); // con imagen cargada se usa "Cambiar"
  });

  ["dragenter", "dragover"].forEach((ev) =>
    ui.zona.addEventListener(ev, (e) => {
      e.preventDefault();
      ui.zona.classList.add("zona-carga--arrastrando");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    ui.zona.addEventListener(ev, (e) => {
      e.preventDefault();
      ui.zona.classList.remove("zona-carga--arrastrando");
    })
  );
  ui.zona.addEventListener("drop", (e) => cargarArchivo(e.dataTransfer.files[0]));

  // ---------- Carrusel de filtros ----------
  function seleccionarFiltro(boton) {
    ui.carrusel.querySelectorAll(".filtro").forEach((b) => {
      const activo = b === boton;
      b.classList.toggle("filtro--activo", activo);
      b.setAttribute("aria-checked", String(activo));
    });
    estado.filtro = boton.dataset.filtro;
    ui.descripcion.textContent = boton.dataset.descripcion;
    ui.controlMascara.hidden = boton.dataset.mascaraVariable !== "true";
    boton.scrollIntoView({ behavior: "smooth", inline: "nearest", block: "nearest" });
  }

  ui.carrusel.addEventListener("click", (e) => {
    const boton = e.target.closest(".filtro");
    if (boton) seleccionarFiltro(boton);
  });

  const desplazar = (dir) => ui.carrusel.scrollBy({ left: dir * 190, behavior: "smooth" });
  ui.flechaIzq.addEventListener("click", () => desplazar(-1));
  ui.flechaDer.addEventListener("click", () => desplazar(1));

  function actualizarFlechas() {
    const { scrollLeft, scrollWidth, clientWidth } = ui.carrusel;
    ui.flechaIzq.disabled = scrollLeft <= 2;
    ui.flechaDer.disabled = scrollLeft + clientWidth >= scrollWidth - 2;
  }
  ui.carrusel.addEventListener("scroll", actualizarFlechas, { passive: true });
  window.addEventListener("resize", actualizarFlechas);

  // ---------- Tamano de mascara ----------
  function fijarMascara(k) {
    estado.mascara = k;
    ui.inputMascara.value = k;
    ui.valorMascara.textContent = `${k} × ${k}`;
    ui.chips.forEach((c) => c.classList.toggle("chip--activo", Number(c.dataset.k) === k));
  }
  ui.inputMascara.addEventListener("input", () => fijarMascara(Number(ui.inputMascara.value)));
  ui.chips.forEach((c) => c.addEventListener("click", () => fijarMascara(Number(c.dataset.k))));

  // ---------- Procesar ----------
  async function procesar() {
    if (!estado.archivo) return;
    const filtro = filtroActual();
    const mascara = filtro.mascara_variable ? estado.mascara : 3;

    const datos = new FormData();
    datos.append("imagen", estado.archivo);
    datos.append("filtro", filtro.id);
    datos.append("mascara", String(mascara));

    ocupado(true);
    mostrarEstado("Procesando…");
    const inicio = performance.now();

    try {
      const resp = await fetch("/api/procesar", { method: "POST", body: datos });
      const ms = performance.now() - inicio;

      if (resp.ok) {
        const blob = await resp.blob();
        mostrarResultado({ filtro, mascara, url: URL.createObjectURL(blob), ms, simulado: false });
        mostrarEstado("");
      } else if (resp.status === 501) {
        mostrarResultado({ filtro, mascara, url: estado.urlOriginal, ms: null, simulado: true });
        mostrarEstado("");
      } else {
        const err = await resp.json().catch(() => ({}));
        mostrarEstado(err.detail || `Error ${resp.status}`, true);
      }
    } catch {
      mostrarEstado("No se pudo contactar con el servidor.", true);
    } finally {
      ocupado(false);
    }
  }
  ui.btnAplicar.addEventListener("click", procesar);

  function mostrarResultado({ filtro, mascara, url, ms, simulado }) {
    if (estado.urlResultado && estado.urlResultado !== estado.urlOriginal) {
      URL.revokeObjectURL(estado.urlResultado);
    }
    estado.urlResultado = url;

    ui.imgOriginal.src = estado.urlOriginal;
    ui.imgResultado.src = url;
    ui.imgResultado.className = simulado ? `sim-${filtro.id}` : "";
    ui.imgResultado.style.setProperty("--sim-blur", `${Math.max(1, mascara / 4)}px`);

    const chips = [filtro.nombre, `Máscara ${mascara}×${mascara}`];
    if (ms !== null) chips.push(`${ms.toFixed(0)} ms`);
    ui.etiquetas.replaceChildren(
      ...chips.map((t) => Object.assign(document.createElement("span"), { textContent: t }))
    );

    const base = estado.archivo.name.replace(/\.[^.]+$/, "");
    ui.btnDescargar.href = simulado ? "#" : url;
    ui.btnDescargar.download = `${base}_${filtro.id}${filtro.mascara_variable ? mascara : ""}.png`;
    ui.btnDescargar.setAttribute("aria-disabled", String(simulado));
    ui.avisoSimulado.hidden = !simulado;

    ui.resultados.hidden = false;
    ui.resultados.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ---------- Utilidades ----------
  function ocupado(si) {
    ui.btnAplicar.disabled = si;
    ui.btnAplicar.classList.toggle("boton--cargando", si);
  }

  function mostrarEstado(texto, error = false) {
    ui.estado.textContent = texto;
    ui.estado.classList.toggle("estado--error", error);
  }

  function formatearBytes(b) {
    if (b < 1024) return `${b} B`;
    if (b < 1024 ** 2) return `${(b / 1024).toFixed(0)} KB`;
    return `${(b / 1024 ** 2).toFixed(1)} MB`;
  }

  // ---------- Inicio ----------
  fijarMascara(estado.mascara);
  actualizarFlechas();
})();
