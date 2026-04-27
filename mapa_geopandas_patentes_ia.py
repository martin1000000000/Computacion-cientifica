"""Mapa coroplético: Patentes de IA por país (OCDE, 2022).

Genera un mapa mundial usando GeoPandas que muestra la cantidad de
patentes de inteligencia artificial registradas por cada país en 2022,
según datos de la OCDE.

El mapa visualiza la concentración geográfica de la innovación en IA,
evidenciando que un puñado de países (EE.UU., China, Japón, Corea del Sur
y Alemania) concentra la enorme mayoría de patentes de IA a nivel mundial.

Salidas:
  - Data/ocde_ia/mapa_patentes_ia_2022.png
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ──────────────────────────── RUTAS ────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "Data"
INPUT_JSON = DATA_DIR / "oecd_ai_patents_2020_clean.json"
OUTPUT_PNG = DATA_DIR / "ocde_ia" / "mapa_patentes_ia_2022.png"

# ──────────────────────────── CONSTANTES ────────────────────────────
YEAR_FILTER = "2022"

# Nombres de países en español para el hover / anotaciones
ISO3_ES: dict[str, str] = {
    "USA": "Estados Unidos",
    "CHN": "China",
    "JPN": "Japón",
    "KOR": "Corea del Sur",
    "DEU": "Alemania",
    "TWN": "Taiwán",
    "FRA": "Francia",
    "CAN": "Canadá",
    "GBR": "Reino Unido",
    "SWE": "Suecia",
    "NLD": "Países Bajos",
    "CHE": "Suiza",
    "IND": "India",
    "ISR": "Israel",
    "FIN": "Finlandia",
    "IRL": "Irlanda",
    "ITA": "Italia",
    "SGP": "Singapur",
    "AUS": "Australia",
    "ESP": "España",
    "BRA": "Brasil",
    "DNK": "Dinamarca",
    "NOR": "Noruega",
    "BEL": "Bélgica",
    "AUT": "Austria",
    "RUS": "Rusia",
    "SAU": "Arabia Saudita",
    "TUR": "Turquía",
    "ARE": "Emiratos Árabes",
    "POL": "Polonia",
    "HUN": "Hungría",
    "PRT": "Portugal",
    "NZL": "Nueva Zelanda",
    "LUX": "Luxemburgo",
    "GRC": "Grecia",
    "THA": "Tailandia",
    "ZAF": "Sudáfrica",
    "MEX": "México",
    "CHL": "Chile",
}


def load_patent_data() -> pd.DataFrame:
    """Carga y filtra los datos de patentes de IA del JSON limpio."""
    with open(INPUT_JSON, encoding="utf-8") as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    df = df[df["year"] == YEAR_FILTER].copy()
    # Agregar duplicados (si existen) por país
    df = df.groupby("iso3", as_index=False).agg(
        {"country": "first", "ai_patents_value": "sum"}
    )
    df = df.sort_values("ai_patents_value", ascending=False).reset_index(drop=True)
    df["country_es"] = df["iso3"].map(ISO3_ES).fillna(df["country"])
    return df


def build_map(df_patents: pd.DataFrame) -> None:
    """Construye y guarda el mapa coroplético con GeoPandas."""

    # ── Cargar geometría mundial (Natural Earth 110m) ──
    NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(NE_URL)
    # Usar ISO_A3_EH que tiene mejor cobertura que ISO_A3
    world = world.rename(columns={"ISO_A3_EH": "iso3"})
    # Corregir códigos faltantes
    fixes = {"France": "FRA", "Norway": "NOR", "Kosovo": "XKX", "Somaliland": "SOM"}
    for name, code in fixes.items():
        world.loc[world["NAME"] == name, "iso3"] = code

    # ── Merge ──
    world = world.merge(df_patents[["iso3", "ai_patents_value", "country_es"]],
                        on="iso3", how="left")

    # ── Escala de color logarítmica (las diferencias son enormes) ──
    vmin = 1.0
    vmax = float(df_patents["ai_patents_value"].max())
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # ── Paleta personalizada ──
    colors_list = ["#f0f0f0", "#fef3c7", "#fcd34d", "#f59e0b",
                   "#ea580c", "#dc2626", "#991b1b"]
    cmap = mcolors.LinearSegmentedColormap.from_list("ia_patents", colors_list, N=256)
    cmap.set_bad(color="#e5e7eb")  # países sin datos → gris claro

    # ── Figura ──
    fig, ax = plt.subplots(1, 1, figsize=(18, 10), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    # Pintar todos los países con borde sutil
    world.plot(ax=ax, color="#334155", edgecolor="#475569", linewidth=0.3)

    # Pintar países con datos
    has_data = world["ai_patents_value"].notna()
    world[has_data].plot(
        ax=ax,
        column="ai_patents_value",
        cmap=cmap,
        norm=norm,
        edgecolor="#475569",
        linewidth=0.4,
        legend=False,
    )

    # ── Barra de color ──
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.65)
    cbar.set_label("Patentes de IA (escala logarítmica)",
                   fontsize=12, color="white", labelpad=10)
    cbar.ax.tick_params(colors="white", labelsize=9)

    # ── Anotar Top 10 países ──
    # Centroides para posicionar etiquetas
    centroids = world[has_data].copy()
    centroids["centroid"] = centroids.geometry.representative_point()
    centroids = centroids.sort_values("ai_patents_value", ascending=False).head(10)

    # Offsets manuales para evitar solapamiento (dx, dy en coordenadas)
    label_offsets: dict[str, tuple[float, float]] = {
        "USA": (-25, 8),
        "CHN": (12, -12),
        "JPN": (18, 5),
        "KOR": (15, -8),
        "DEU": (-5, 10),
        "TWN": (15, -3),
        "FRA": (-15, -10),
        "GBR": (-18, 8),
        "CAN": (-20, 15),
        "SWE": (5, 12),
    }

    for _, row in centroids.iterrows():
        iso = row["iso3"]
        cx = row["centroid"].x
        cy = row["centroid"].y
        dx, dy = label_offsets.get(iso, (8, 5))
        label = f"{row['country_es']}\n{row['ai_patents_value']:,.0f}"

        ax.annotate(
            label,
            xy=(cx, cy),
            xytext=(cx + dx, cy + dy),
            fontsize=7.5,
            color="white",
            fontweight="bold",
            ha="center",
            va="center",
            arrowprops=dict(
                arrowstyle="-",
                color="#94a3b8",
                lw=0.7,
            ),
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#1e293b",
                edgecolor="#64748b",
                alpha=0.85,
            ),
        )

    # ── Título y subtítulo ──
    ax.set_title(
        "Patentes de Inteligencia Artificial por País — 2022",
        fontsize=20,
        fontweight="bold",
        color="white",
        pad=20,
    )
    ax.text(
        0.5, 1.02,
        "Fuente: OCDE · Familias de patentes IP5 relacionadas con IA",
        transform=ax.transAxes,
        fontsize=10,
        color="#94a3b8",
        ha="center",
        va="bottom",
    )

    # ── Nota metodológica ──
    ax.text(
        0.01, 0.02,
        ("Nota: Las patentes se cuentan por familias IP5 (5 principales oficinas mundiales).\n"
         "Un valor fraccionario indica co‑invención entre países.\n"
         "Los países sin datos aparecen en gris oscuro."),
        transform=ax.transAxes,
        fontsize=8,
        color="#94a3b8",
        va="bottom",
        fontstyle="italic",
    )

    # ── Limpiar ejes ──
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")

    plt.tight_layout()

    # ── Guardar ──
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Mapa guardado en: {OUTPUT_PNG}")


def main() -> None:
    df = load_patent_data()
    print(f"Países con datos ({YEAR_FILTER}): {len(df)}")
    print(f"Top 5:\n{df[['country_es', 'ai_patents_value']].head().to_string(index=False)}")
    print()
    build_map(df)


if __name__ == "__main__":
    main()
