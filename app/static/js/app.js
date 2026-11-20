// app.js - Interaccion de la interfaz. Consume el servicio HTTP:
//   GET  /api/salud     -> estado de la GPU (se consulta al cargar)
//   POST /api/procesar  -> imagen filtrada (PNG) + tiempos en cabeceras X-Tiempo-*
(() => {
  "use strict";

  const CONFIG = window.CONFIG;
  const LIMITES = CONFIG.limites || { max_mb: 25 };
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
    valorMascara: $("valor-mascara"),
    opcionesMascara: document.querySelectorAll(".segmentado__opcion[data-k]"),
    btnAplicar: $("btn-aplicar"),
    estado: $("estado"),
    avisoGpu: $("aviso-gpu"),
    resultados: $("resultados"),
    etiquetas: $("etiquetas"),
    imgOriginal: $("img-original"),
    imgResultado: $("img-resultado"),
    btnDescargar: $("btn-descargar"),
    barraTiempos: $("barra-tiempos"),
    listaTiempos: $("lista-tiempos"),
  };

  const estado = {
    archivo: null,
    urlOriginal: null,
    urlResultado: null,
    filtro: CONFIG.filtros[0].id,
    mascara: CONFIG.mascara.defecto,
    enCurso: false,        // evita solicitudes duplicadas mientras se procesa
    gpuDisponible: true,
  };

  const filtroActual = () => CONFIG.filtros.find((f) => f.id === estado.filtro);

  // ---------- Estado de la GPU ----------
  async function consultarSalud() {
    try {
      const resp = await fetch("/api/salud");
      const datos = await resp.json();
      estado.gpuDisponible = resp.ok;
      if (!resp.ok) {
        ui.avisoGpu.textContent = `GPU no disponible: ${datos.diagnostico || "sin diagnóstico"}`;
        ui.avisoGpu.hidden = false;
      }
    } catch {
      ui.avisoGpu.textContent = "No se pudo contactar con el servidor.";
      ui.avisoGpu.hidden = false;
      estado.gpuDisponible = false;
    }
    actualizarBoton();
  }

  // ---------- Carga de imagen ----------
  function cargarArchivo(archivo) {
    if (!archivo) return;
    if (archivo.size > LIMITES.max_mb * 1024 * 1024) {
      mostrarEstado(`El archivo pesa ${formatearBytes(archivo.size)}; el límite es ${LIMITES.max_mb} MB.`, true);
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
    // El navegador no muestra PGM/BMP raros: el servidor valida el formato real igualmente
    ui.vistaPrevia.onerror = () => {
      ui.info.textContent = `${formatearBytes(archivo.size)} · vista previa no disponible`;
    };
    ui.vistaPrevia.src = estado.urlOriginal;
    ui.vistaPrevia.hidden = false;
    ui.zonaVacia.hidden = true;
    ui.zona.classList.add("zona-carga--cargada");
    ui.nombre.textContent = archivo.name;
    ui.btnCambiar.hidden = false;
    ui.resultados.hidden = true;
    mostrarEstado("");
    actualizarBoton();
  }

  ui.input.addEventListener("change", () => cargarArchivo(ui.input.files[0]));
  ui.btnCambiar.addEventListener("click", () => {
    ui.input.value = ""; // permite volver a elegir el mismo archivo
    ui.input.click();
  });
  ui.zona.addEventListener("click", (e) => {
    // Con imagen cargada, un clic en la foto no abre el selector (se usa "Cambiar").
    // El clic sintetico de input.click() tambien sube hasta aqui: no hay que cancelarlo.
    if (estado.archivo && e.target !== ui.input) e.preventDefault();
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
    ui.valorMascara.textContent = `${k} × ${k}`;
    ui.opcionesMascara.forEach((o) => o.setAttribute("aria-checked", String(Number(o.dataset.k) === k)));
  }
  ui.opcionesMascara.forEach((o) => o.addEventListener("click", () => fijarMascara(Number(o.dataset.k))));

  // ---------- Procesar ----------
  async function procesar() {
    if (!estado.archivo || estado.enCurso) return;
    const filtro = filtroActual();
    const mascara = filtro.mascara_variable ? estado.mascara : 3;

    const datos = new FormData();
    datos.append("imagen", estado.archivo);
    datos.append("filtro", filtro.id);
    datos.append("mascara", String(mascara));

    ocupado(true);
    mostrarEstado("Procesando en la GPU…");
    const inicio = performance.now();

    try {
      const resp = await fetch("/api/procesar", { method: "POST", body: datos });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        const id = err.request_id ? ` (solicitud ${err.request_id.slice(0, 8)})` : "";
        mostrarEstado(`${err.detail || `Error ${resp.status}`}${id}`, true);
        return;
      }
      const blob = await resp.blob(); // tiempo total del cliente: hasta tener la imagen completa
      const totalCliente = performance.now() - inicio;
      mostrarResultado({ filtro, mascara, blob, resp, totalCliente });
      mostrarEstado("");
    } catch {
      mostrarEstado("No se pudo contactar con el servidor.", true);
    } finally {
      ocupado(false);
    }
  }
  ui.btnAplicar.addEventListener("click", procesar);

  function leerTiempos(resp) {
    const t = (nombre) => Number(resp.headers.get(`X-Tiempo-${nombre}-ms`));
    return {
      cola: t("Cola"), decodificacion: t("Decodificacion"), h2d: t("H2D"), kernel: t("Kernel"),
      d2h: t("D2H"), codificacion: t("Codificacion"), servidor: t("Servidor"),
    };
  }

  function mostrarResultado({ filtro, mascara, blob, resp, totalCliente }) {
    if (estado.urlResultado) URL.revokeObjectURL(estado.urlResultado);
    estado.urlResultado = URL.createObjectURL(blob);

    ui.imgOriginal.src = estado.urlOriginal;
    ui.imgResultado.src = estado.urlResultado;

    const requestId = resp.headers.get("X-Request-ID") || "";
    const chips = [filtro.nombre, `Máscara ${mascara}×${mascara}`, `${resp.headers.get("X-Imagen") || ""} px`];
    if (requestId) chips.push(`ID ${requestId.slice(0, 8)}`);
    ui.etiquetas.replaceChildren(
      ...chips.map((t) => Object.assign(document.createElement("span"), { textContent: t }))
    );

    const base = estado.archivo.name.replace(/\.[^.]+$/, "");
    const sufijoK = filtro.mascara_variable ? mascara : "";
    ui.btnDescargar.href = estado.urlResultado;
    ui.btnDescargar.download = `${base}_${filtro.id}${sufijoK}_${requestId.slice(0, 8)}.png`;

    pintarTiempos(leerTiempos(resp), totalCliente);
    ui.resultados.hidden = false;
    ui.resultados.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // Cada fila: [clave, etiqueta, alcance, clase de color]
  const FILAS = [
    ["cliente", "Total en el cliente", "Desde enviar la solicitud hasta recibir la imagen completa (incluye red).", "", true],
    ["servidor", "Procesamiento del servidor", "Decodificación + GPU + codificación PNG. No incluye la espera en cola.", "", true],
    ["cola", "Espera en cola", "Tiempo hasta que la GPU quedó libre para esta solicitud.", ""],
    ["decodificacion", "Decodificación", "Validar y convertir la imagen a escala de grises (CPU).", "t-dec"],
    ["h2d", "CPU → GPU", "Copia de imagen y máscara (eventos CUDA, memoria paginable).", "t-h2d"],
    ["kernel", "Kernel de convolución", "Solo la ejecución en la GPU (eventos CUDA).", "t-kernel"],
    ["d2h", "GPU → CPU", "Copia del resultado float32 (eventos CUDA).", "t-d2h"],
    ["codificacion", "Codificación PNG", "Recorte a [0, 255] y compresión del resultado (CPU).", "t-cod"],
  ];

  function pintarTiempos(t, totalCliente) {
    const valores = { ...t, cliente: totalCliente };
    ui.listaTiempos.replaceChildren(
      ...FILAS.flatMap(([clave, etiqueta, alcance, clase, principal]) => {
        const dt = document.createElement("dt");
        if (principal) dt.classList.add("principal");
        const punto = Object.assign(document.createElement("span"), { className: `punto ${clase}` });
        if (!clase) punto.style.visibility = "hidden";
        const texto = document.createElement("div");
        texto.append(etiqueta, Object.assign(document.createElement("small"), { textContent: alcance }));
        dt.append(punto, texto);
        const dd = Object.assign(document.createElement("dd"), { textContent: formatearMs(valores[clave]) });
        if (principal) dd.classList.add("principal");
        return [dt, dd];
      })
    );

    // Barra proporcional de las etapas del servidor
    const etapas = FILAS.filter((f) => f[3]);
    const suma = etapas.reduce((s, f) => s + (valores[f[0]] || 0), 0) || 1;
    ui.barraTiempos.replaceChildren(
      ...etapas.map(([clave, etiqueta, , clase]) => {
        const s = document.createElement("span");
        s.className = clase;
        s.style.flexGrow = String((valores[clave] || 0) / suma);
        s.title = `${etiqueta}: ${formatearMs(valores[clave])}`;
        return s;
      })
    );
  }

  // ---------- Utilidades ----------
  function actualizarBoton() {
    ui.btnAplicar.disabled = !estado.archivo || estado.enCurso || !estado.gpuDisponible;
  }

  function ocupado(si) {
    estado.enCurso = si;
    ui.btnAplicar.classList.toggle("boton--cargando", si);
    ui.btnAplicar.setAttribute("aria-busy", String(si));
    actualizarBoton();
  }

  function mostrarEstado(texto, error = false) {
    ui.estado.textContent = texto;
    ui.estado.classList.toggle("estado--error", error);
  }

  function formatearMs(ms) {
    if (!Number.isFinite(ms)) return "—";
    return `${ms < 10 ? ms.toFixed(3) : ms.toFixed(1)} ms`;
  }

  function formatearBytes(b) {
    if (b < 1024) return `${b} B`;
    if (b < 1024 ** 2) return `${(b / 1024).toFixed(0)} KB`;
    return `${(b / 1024 ** 2).toFixed(1)} MB`;
  }

  // ---------- Inicio ----------
  fijarMascara(estado.mascara);
  actualizarFlechas();
  consultarSalud();
})();
