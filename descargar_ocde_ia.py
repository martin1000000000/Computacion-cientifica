"""
==============================================================================
Descargador DINÁMICO de datos de IA desde la API SDMX de la OCDE
==============================================================================
Cada vez que se ejecuta:
  1. Consulta el catálogo de la OCDE para descubrir TODOS los dataflows
  2. Filtra los que estén relacionados con IA / tecnología / innovación
  3. Descarga SOLO los que no se han descargado antes
  4. Si ya descargó todos, prueba con nuevas agencias/temas

Así, cada ejecución se conecta a APIs DISTINTAS.

Uso:
    python descargar_ocde_ia.py

Limite de la OCDE: ~60 descargas/hora. El script espera entre peticiones.
==============================================================================
"""

import json
import time
import sys
import re
import logging
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET

import requests
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────

DATA_DIR = Path("./Data/ocde_ia")
DATA_DIR.mkdir(parents=True, exist_ok=True)

ESTADO_PATH = DATA_DIR / "estado_descargas.json"

# Tiempo de espera entre peticiones (segundos)
ESPERA_ENTRE_PETICIONES = 5

# Máximo de descargas de datos por ejecución (para no pasarse del rate limit)
MAX_DESCARGAS_POR_EJECUCION = 15

MAX_REINTENTOS = 3
TIMEOUT = 60

# Control de salida para evitar archivos gigantes y errores por espacio.
MAX_FILAS_JSON_LIMPIO = 1_000_000
GUARDAR_JSON_LIMPIO_COMPRIMIDO = True

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-OECD/1.0",
    "Accept": "application/vnd.sdmx.data+json;version=2.0.0",
}

# ─────────────────────────────────────────────────────────────
# AGENCIAS DE LA OCDE QUE TIENEN DATOS DE IA / TECNOLOGÍA
# Cada ejecución explora una agencia distinta del catálogo
# ─────────────────────────────────────────────────────────────

AGENCIAS = [
    "OECD.STI.PIE",   # Patentes, indicadores económicos
    "OECD.STI.STP",   # Ciencia, tecnología, política de innovación (MSTI)
    "OECD.STI.DEG",   # Economía digital, ICT, uso de IA en empresas
    "OECD.ELS.SAE",   # Empleo, trabajo, habilidades
    "OECD.EDU.IMEP",  # Educación
    "OECD.SDD.TPS",   # Productividad
    "OECD.CFE.SME",   # PyMEs, emprendimiento
    "OECD.GOV.GOV",   # Gobierno digital
    "OECD.TAD.TNC",   # Comercio, inversión
    "OECD.SDD.NAD",   # Cuentas nacionales
]

# Palabras clave para filtrar dataflows relacionados con IA/tech
KEYWORDS_IA = [
    "artificial intelligence", "ai ", " ai,", " ai.", "(ai)",
    "machine learning", "deep learning",
    "ict", "digital", "patent", "brevet",
    "innovation", "innovacion",
    "research", "r&d", "gerd", "berd",
    "technology", "technologie",
    "robot", "automation",
    "broadband", "internet",
    "startup", "venture capital",
    "skills", "stem", "competenc",
    "scientist", "researcher",
    "software", "data",
    "telecommunication", "telecom",
    "e-commerce", "ecommerce",
    "cybersecurity", "cyber",
]

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

import io

# Forzar UTF-8 en stdout para Windows (evita UnicodeEncodeError con emojis)
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "descargas.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ESTADO (seguimiento de qué se descargó)
# ─────────────────────────────────────────────────────────────

def cargar_estado() -> dict:
    if ESTADO_PATH.exists():
        return json.loads(ESTADO_PATH.read_text(encoding="utf-8"))
    return {
        "descargas": {},
        "agencias_exploradas": [],
        "dataflows_descubiertos": {},
        "ultima_ejecucion": None,
        "ejecuciones": 0,
    }


def guardar_estado(estado: dict):
    estado["ultima_ejecucion"] = datetime.now().isoformat()
    ESTADO_PATH.write_text(
        json.dumps(estado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ─────────────────────────────────────────────────────────────
# PASO 1: DESCUBRIR DATAFLOWS DE UNA AGENCIA
# ─────────────────────────────────────────────────────────────

def descubrir_dataflows_agencia(agencia: str) -> list[dict]:
    """
    Consulta el catálogo SDMX de una agencia para obtener TODOS
    sus dataflows disponibles. Retorna lista de dicts con info.
    """
    url = f"https://sdmx.oecd.org/public/rest/dataflow/{agencia}"
    log.info(f"🔍 Explorando catálogo de: {agencia}")
    log.info(f"   URL: {url}")

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": HEADERS["User-Agent"]},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            log.warning(f"   ⚠️  HTTP {resp.status_code} al explorar {agencia}")
            return []
    except Exception as e:
        log.warning(f"   ❌ Error conectando a {agencia}: {e}")
        return []

    # Parsear XML del catálogo
    dataflows = []
    try:
        root = ET.fromstring(resp.content)
        # Namespaces SDMX
        ns = {
            "mes": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message",
            "str": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/structure",
            "com": "http://www.sdmx.org/resources/sdmxml/schemas/v2_1/common",
        }

        for df in root.findall(".//str:Dataflow", ns):
            df_id = df.get("id", "")
            version = df.get("version", "1.0")

            # Obtener nombre en inglés
            nombre_en = ""
            for name in df.findall("com:Name", ns):
                lang = name.get("{http://www.w3.org/XML/1998/namespace}lang", "")
                if lang == "en":
                    nombre_en = name.text or ""
                    break
            if not nombre_en:
                name_el = df.find("com:Name", ns)
                nombre_en = name_el.text if name_el is not None else df_id

            # Obtener descripción en inglés
            desc_en = ""
            for desc in df.findall("com:Description", ns):
                lang = desc.get("{http://www.w3.org/XML/1998/namespace}lang", "")
                if lang == "en":
                    desc_en = desc.text or ""
                    break

            dataflows.append({
                "dataflow_id": df_id,
                "agencia": agencia,
                "version": version,
                "nombre": nombre_en,
                "descripcion": desc_en[:300],
            })

    except ET.ParseError as e:
        log.warning(f"   ⚠️  Error parseando XML: {e}")

    log.info(f"   📋 Encontrados: {len(dataflows)} dataflows en {agencia}")
    return dataflows


def filtrar_por_ia(dataflows: list[dict]) -> list[dict]:
    """Filtra dataflows que contengan keywords de IA/tech."""
    filtrados = []
    for df in dataflows:
        texto = f"{df['nombre']} {df['descripcion']}".lower()
        if any(kw in texto for kw in KEYWORDS_IA):
            filtrados.append(df)
    return filtrados


# ─────────────────────────────────────────────────────────────
# PASO 2: CONSTRUIR URL Y DESCARGAR DATOS
# ─────────────────────────────────────────────────────────────

def construir_url_datos(df_info: dict, start: str = "2015", end: str = "2024") -> str:
    """
    Construye la URL de datos para un dataflow descubierto.
    Usa 'all' como filtro para obtener todos los datos disponibles.
    """
    agencia = df_info["agencia"]
    dataflow_id = df_info["dataflow_id"]
    version = df_info["version"]

    url = (
        f"https://sdmx.oecd.org/public/rest/data/"
        f"{agencia},{dataflow_id},{version}/all"
        f"?startPeriod={start}&endPeriod={end}"
        f"&dimensionAtObservation=AllDimensions"
    )
    return url


def descargar_con_reintentos(url: str, nombre: str) -> dict | None:
    """Descarga JSON con reintentos y backoff exponencial."""
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            log.info(f"   Intento {intento}/{MAX_REINTENTOS}...")
            resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)

            if resp.status_code == 200:
                log.info(f"   ✅ OK — {len(resp.content):,} bytes")
                return resp.json()
            elif resp.status_code == 404:
                log.warning(f"   ⚠️  404 — No hay datos para: {nombre}")
                return None
            elif resp.status_code == 413:
                log.warning(f"   ⚠️  413 — Datos demasiado grandes, intentando rango menor...")
                return None
            elif resp.status_code == 429:
                espera = 60 * intento
                log.warning(f"   ⏳ 429 Rate Limit — esperando {espera}s")
                time.sleep(espera)
            elif resp.status_code == 406:
                log.warning(f"   ⚠️  406 — Formato no soportado para: {nombre}")
                return None
            else:
                log.warning(f"   ❌ HTTP {resp.status_code}")
                if intento < MAX_REINTENTOS:
                    time.sleep(10 * intento)
        except requests.exceptions.Timeout:
            log.warning(f"   ⏰ Timeout en intento {intento}")
            if intento < MAX_REINTENTOS:
                time.sleep(15 * intento)
        except requests.exceptions.ConnectionError as e:
            log.warning(f"   🔌 Error de conexión: {e}")
            if intento < MAX_REINTENTOS:
                time.sleep(15 * intento)
        except json.JSONDecodeError:
            log.warning(f"   ⚠️  Respuesta no es JSON válido")
            return None
        except Exception as e:
            log.error(f"   💥 Error inesperado: {e}")
            return None

    log.error(f"   ❌ Falló después de {MAX_REINTENTOS} intentos: {nombre}")
    return None


def parsear_sdmx_json(payload: dict) -> pd.DataFrame:
    """Convierte JSON SDMX en DataFrame plano."""
    data = payload.get("data", {})

    if "structure" in data:
        obs_dims = data["structure"]["dimensions"]["observation"]
    elif "structures" in data:
        obs_dims = data["structures"][0]["dimensions"]["observation"]
    else:
        log.warning("   ⚠️  Estructura SDMX no reconocida")
        return pd.DataFrame()

    dim_ids = [d["id"] for d in obs_dims]
    dim_values = [d["values"] for d in obs_dims]

    datasets = data.get("dataSets", [])
    if not datasets:
        return pd.DataFrame()

    observations = datasets[0].get("observations", {})
    if not observations:
        return pd.DataFrame()

    records = []
    for obs_key, obs_val in observations.items():
        idx = [int(i) for i in obs_key.split(":")]
        rec = {}
        for i, dim_id in enumerate(dim_ids):
            if i < len(idx) and idx[i] < len(dim_values[i]):
                selected = dim_values[i][idx[i]]
                rec[dim_id] = selected.get("id")
                rec[f"{dim_id}_label"] = selected.get("name")
        rec["OBS_VALUE"] = obs_val[0] if isinstance(obs_val, list) and obs_val else None
        records.append(rec)

    df = pd.DataFrame(records)
    df["OBS_VALUE"] = pd.to_numeric(df["OBS_VALUE"], errors="coerce")
    return df


def guardar_json_limpio_seguro(df: pd.DataFrame, safe_id: str) -> tuple[str | None, str]:
    """
    Guarda JSON limpio de forma segura para evitar caídas por archivos muy grandes
    o falta de espacio en disco.
    """
    n_filas = len(df)

    if n_filas > MAX_FILAS_JSON_LIMPIO:
        log.warning(
            "   ⚠️  JSON limpio omitido: "
            f"{n_filas:,} filas supera el límite ({MAX_FILAS_JSON_LIMPIO:,})."
        )
        return None, "omitido_tamano"

    try:
        if GUARDAR_JSON_LIMPIO_COMPRIMIDO:
            json_path = DATA_DIR / f"{safe_id}_clean.json.gz"
            df.to_json(
                path_or_buf=str(json_path),
                orient="records",
                force_ascii=False,
                compression="gzip",
            )
        else:
            json_path = DATA_DIR / f"{safe_id}_clean.json"
            df.to_json(
                path_or_buf=str(json_path),
                orient="records",
                force_ascii=False,
            )

        log.info(f"   📄 JSON → {json_path.name}")
        return json_path.name, "ok"

    except OSError as e:
        if e.errno == 28:
            log.error("   ❌ Sin espacio al guardar JSON limpio. Se mantiene solo CSV.")
            return None, "sin_espacio"

        log.warning(f"   ⚠️  OSError guardando JSON limpio: {e}")
        return None, "error_os"

    except Exception as e:
        log.warning(f"   ⚠️  No se pudo guardar JSON limpio: {e}")
        return None, "error"


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("🚀 DESCARGADOR DINÁMICO DE DATOS IA — API OCDE (SDMX)")
    log.info("   Cada ejecución descubre y descarga APIs DISTINTAS")
    log.info("=" * 70)

    estado = cargar_estado()
    estado["ejecuciones"] = estado.get("ejecuciones", 0) + 1
    n_ejecucion = estado["ejecuciones"]
    ya_descargados = estado.get("descargas", {})
    agencias_exploradas = estado.get("agencias_exploradas", [])
    todos_descubiertos = estado.get("dataflows_descubiertos", {})

    log.info(f"📊 Ejecución #{n_ejecucion}")
    log.info(f"   Ya descargados: {len(ya_descargados)} datasets")
    log.info(f"   Agencias exploradas: {len(agencias_exploradas)}/{len(AGENCIAS)}")
    log.info("")

    # ─── FASE 1: Descubrir dataflows de agencias no exploradas ───
    agencias_nuevas = [a for a in AGENCIAS if a not in agencias_exploradas]

    if agencias_nuevas:
        # Explorar 2 agencias nuevas por ejecución (para no abusar)
        agencias_a_explorar = agencias_nuevas[:2]

        for agencia in agencias_a_explorar:
            dataflows = descubrir_dataflows_agencia(agencia)
            time.sleep(ESPERA_ENTRE_PETICIONES)

            # Filtrar por IA/tech
            relevantes = filtrar_por_ia(dataflows)
            log.info(f"   🧠 Relacionados con IA/tech: {len(relevantes)} de {len(dataflows)}")

            for df_info in relevantes:
                clave = f"{df_info['agencia']}|{df_info['dataflow_id']}"
                if clave not in todos_descubiertos:
                    todos_descubiertos[clave] = df_info

            agencias_exploradas.append(agencia)
            log.info("")

        estado["agencias_exploradas"] = agencias_exploradas
        estado["dataflows_descubiertos"] = todos_descubiertos
        guardar_estado(estado)

    # ─── FASE 2: Identificar cuáles NO se han descargado aún ────
    pendientes = []
    for clave, df_info in todos_descubiertos.items():
        if clave not in ya_descargados:
            pendientes.append((clave, df_info))

    if not pendientes:
        # Si ya descargamos todo lo descubierto, explorar más agencias
        if agencias_nuevas:
            log.info("ℹ️  Todos los dataflows descubiertos ya fueron descargados.")
            log.info("   Pero hay más agencias por explorar. Ejecuta de nuevo.")
        else:
            log.info("🎉 ¡Todo explorado y descargado!")
            log.info(f"   Total datasets: {len(ya_descargados)}")
            log.info(f"   Total agencias: {len(agencias_exploradas)}")
            log.info("   Para re-descargar todo, borra estado_descargas.json")
        guardar_estado(estado)
        return

    # Limitar descargas por ejecución
    a_descargar = pendientes[:MAX_DESCARGAS_POR_EJECUCION]

    log.info("─" * 70)
    log.info(f"📥 DESCARGANDO {len(a_descargar)} de {len(pendientes)} pendientes")
    log.info(f"   (máximo {MAX_DESCARGAS_POR_EJECUCION} por ejecución)")
    log.info("─" * 70)

    for i, (clave, df_info) in enumerate(a_descargar, 1):
        log.info("")
        log.info(f"{'═' * 70}")
        log.info(f"📡 [{i}/{len(a_descargar)}] {df_info['nombre']}")
        log.info(f"   Agencia:  {df_info['agencia']}")
        log.info(f"   Dataflow: {df_info['dataflow_id']}")

        url = construir_url_datos(df_info)
        log.info(f"   URL: {url[:130]}...")

        payload = descargar_con_reintentos(url, df_info["nombre"])

        # Crear un ID seguro para el nombre de archivo
        safe_id = re.sub(r'[^a-zA-Z0-9_]', '_', df_info["dataflow_id"]).lower()
        safe_id = re.sub(r'_+', '_', safe_id).strip('_')

        if payload is None:
            ya_descargados[clave] = {
                "nombre": df_info["nombre"],
                "agencia": df_info["agencia"],
                "dataflow": df_info["dataflow_id"],
                "fecha": datetime.now().isoformat(),
                "estado": "sin_datos",
            }
            guardar_estado(estado)
        else:
            # Guardar JSON crudo
            raw_path = DATA_DIR / f"{safe_id}_raw.json"
            raw_name = None
            try:
                raw_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                raw_name = raw_path.name
                log.info(f"   💾 JSON crudo → {raw_name}")
            except OSError as e:
                if e.errno == 28:
                    log.error("   ❌ Sin espacio al guardar JSON crudo. Se continúa.")
                else:
                    log.warning(f"   ⚠️  No se pudo guardar JSON crudo: {e}")

            # Parsear a DataFrame
            df = parsear_sdmx_json(payload)

            if df.empty:
                log.warning(f"   ⚠️  DataFrame vacío")
                ya_descargados[clave] = {
                    "nombre": df_info["nombre"],
                    "agencia": df_info["agencia"],
                    "dataflow": df_info["dataflow_id"],
                    "fecha": datetime.now().isoformat(),
                    "estado": "vacio",
                }
                if raw_name:
                    ya_descargados[clave]["archivo_raw"] = raw_name
            else:
                # CSV limpio
                csv_path = DATA_DIR / f"{safe_id}_clean.csv"
                try:
                    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                    log.info(f"   📊 CSV  → {csv_path.name} ({len(df):,} filas)")
                except OSError as e:
                    if e.errno == 28:
                        log.error("   ❌ Sin espacio al guardar CSV. Dataset marcado con error.")
                    else:
                        log.warning(f"   ⚠️  No se pudo guardar CSV: {e}")

                    ya_descargados[clave] = {
                        "nombre": df_info["nombre"],
                        "agencia": df_info["agencia"],
                        "dataflow": df_info["dataflow_id"],
                        "fecha": datetime.now().isoformat(),
                        "estado": "error_guardado",
                        "registros": len(df),
                        "columnas": list(df.columns),
                    }
                    if raw_name:
                        ya_descargados[clave]["archivo_raw"] = raw_name
                    guardar_estado(estado)
                    continue

                # JSON limpio (con límites y compresión)
                archivo_json, json_estado = guardar_json_limpio_seguro(df, safe_id)

                ya_descargados[clave] = {
                    "nombre": df_info["nombre"],
                    "agencia": df_info["agencia"],
                    "dataflow": df_info["dataflow_id"],
                    "fecha": datetime.now().isoformat(),
                    "estado": "ok",
                    "registros": len(df),
                    "columnas": list(df.columns),
                    "archivo_csv": csv_path.name,
                    "json_estado": json_estado,
                }
                if archivo_json:
                    ya_descargados[clave]["archivo_json"] = archivo_json
                if raw_name:
                    ya_descargados[clave]["archivo_raw"] = raw_name

            guardar_estado(estado)

        # Esperar entre peticiones
        if i < len(a_descargar):
            log.info(f"   ⏳ Esperando {ESPERA_ENTRE_PETICIONES}s...")
            time.sleep(ESPERA_ENTRE_PETICIONES)

    # ─── RESUMEN ──────────────────────────────────────────────
    exitosos = sum(1 for d in ya_descargados.values() if d.get("estado") == "ok")
    total_reg = sum(d.get("registros", 0) for d in ya_descargados.values())
    quedan = len(pendientes) - len(a_descargar)

    log.info("")
    log.info("=" * 70)
    log.info("📊 RESUMEN")
    log.info("=" * 70)
    log.info(f"   Ejecución:             #{n_ejecucion}")
    log.info(f"   Agencias exploradas:   {len(agencias_exploradas)}/{len(AGENCIAS)}")
    log.info(f"   Dataflows descubiertos:{len(todos_descubiertos)}")
    log.info(f"   Datasets descargados:  {exitosos} con datos")
    log.info(f"   Total registros:       {total_reg:,}")
    log.info(f"   Pendientes:            {quedan}")
    log.info(f"   📂 Datos en: {DATA_DIR.resolve()}")

    if quedan > 0:
        log.info("")
        log.info(f"   ▶ Ejecuta de nuevo para descargar {quedan} datasets más")
    elif agencias_nuevas and len(agencias_nuevas) > 2:
        log.info("")
        log.info(f"   ▶ Ejecuta de nuevo para explorar más agencias")

    log.info("")
    log.info("🎉 ¡Listo!")
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────

def limpiar_estado():
    """Borra el estado para empezar desde cero."""
    if ESTADO_PATH.exists():
        ESTADO_PATH.unlink()
        log.info("🗑️  Estado borrado. La próxima ejecución empieza desde cero.")


def ver_resumen():
    """Muestra un resumen de lo que se ha descargado hasta ahora."""
    estado = cargar_estado()
    descargas = estado.get("descargas", {})

    print(f"\n{'=' * 60}")
    print(f"📊 RESUMEN DE DESCARGAS")
    print(f"{'=' * 60}")
    print(f"Ejecuciones: {estado.get('ejecuciones', 0)}")
    print(f"Agencias exploradas: {len(estado.get('agencias_exploradas', []))}")
    print(f"Dataflows descubiertos: {len(estado.get('dataflows_descubiertos', {}))}")
    print(f"Total descargas: {len(descargas)}")
    print()

    for clave, info in descargas.items():
        icono = "✅" if info.get("estado") == "ok" else "⚠️"
        regs = info.get("registros", 0)
        print(f"  {icono} [{info.get('agencia', '?')}] {info.get('nombre', clave)}")
        if regs:
            print(f"     → {regs:,} registros | {info.get('archivo_csv', 'N/A')}")
    print()


if __name__ == "__main__":
    # Para ver qué se ha descargado sin descargar nada:
    # ver_resumen()

    # Para empezar desde cero:
    # limpiar_estado()

    main()
