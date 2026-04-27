"""Mapa coropletico: uso estimado de modelos IA por pais (HuggingFace).

Genera un mapa mundial con GeoPandas usando el CSV agregado por pais.

Salidas:
- Data/huggingface_ia/huggingface_uso_modelos_mapa_geopandas.png
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
INPUT_CSV = BASE_DIR / "Data" / "huggingface_ia" / "huggingface_uso_estimado_por_pais.csv"
OUTPUT_PNG = BASE_DIR / "Data" / "huggingface_ia" / "huggingface_uso_modelos_mapa_geopandas.png"

NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

COLOR_SCALE = [
    "#f8fafc",
    "#dbeafe",
    "#93c5fd",
    "#60a5fa",
    "#2563eb",
    "#1e40af",
    "#172554",
]


def load_usage_data() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"No existe el archivo: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"country_iso3", "estimated_downloads"}
    missing = required - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Faltan columnas requeridas: {missing_list}")

    df["estimated_downloads"] = pd.to_numeric(
        df["estimated_downloads"], errors="coerce"
    ).fillna(0.0)
    df = df[df["estimated_downloads"] > 0].copy()
    return df


def build_map(df: pd.DataFrame) -> None:
    if df.empty:
        print("[WARN] No hay datos para el mapa.")
        return

    world = gpd.read_file(NE_URL)
    world = world.rename(columns={"ISO_A3_EH": "iso3"})

    fixes = {
        "France": "FRA",
        "Norway": "NOR",
        "Kosovo": "XKX",
        "Somaliland": "SOM",
    }
    for name, code in fixes.items():
        world.loc[world["NAME"] == name, "iso3"] = code

    world = world.merge(
        df[["country_iso3", "estimated_downloads"]],
        left_on="iso3",
        right_on="country_iso3",
        how="left",
    )

    vmin = 1.0
    vmax = float(df["estimated_downloads"].max())
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list("hf_usage", COLOR_SCALE, N=256)
    cmap.set_bad(color="#e5e7eb")

    fig, ax = plt.subplots(1, 1, figsize=(18, 10), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    world.plot(ax=ax, color="#334155", edgecolor="#475569", linewidth=0.3)

    has_data = world["estimated_downloads"].notna()
    world[has_data].plot(
        ax=ax,
        column="estimated_downloads",
        cmap=cmap,
        norm=norm,
        edgecolor="#475569",
        linewidth=0.4,
        legend=False,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.65)
    cbar.set_label("Descargas estimadas (escala log)", fontsize=12, color="white")
    cbar.ax.tick_params(colors="white", labelsize=9)

    ax.set_title(
        "Uso estimado de modelos IA por pais (HuggingFace)",
        fontsize=20,
        fontweight="bold",
        color="white",
        pad=20,
    )
    ax.text(
        0.5,
        1.02,
        "Fuente: HuggingFace (estimacion por idioma/owner)",
        transform=ax.transAxes,
        fontsize=10,
        color="#94a3b8",
        ha="center",
        va="bottom",
    )

    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")
    plt.tight_layout()

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[OK] Mapa guardado en: {OUTPUT_PNG}")


def main() -> None:
    df = load_usage_data()
    print(f"Paises con datos: {len(df)}")
    print(df[["country_iso3", "estimated_downloads"]].head().to_string(index=False))
    build_map(df)


if __name__ == "__main__":
    main()
