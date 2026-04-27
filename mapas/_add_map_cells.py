"""Agrega celdas del mapa GeoPandas al notebook prueba_02.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "prueba_02.ipynb"

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Celdas actuales: {len(nb['cells'])}")

# Celdas nuevas a agregar
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## Mapa GeoPandas: Patentes de IA por Pa\u00eds (OCDE, 2022)\n",
            "\n",
            "Mapa coropl\u00e9tico que muestra la cantidad de patentes de Inteligencia Artificial\n",
            "registradas por cada pa\u00eds en 2022, seg\u00fan datos de la OCDE (familias IP5).\n",
            "\n",
            "**Hallazgo clave:** Solo 4 pa\u00edses (EE.UU., China, Jap\u00f3n y Corea del Sur)\n",
            "concentran m\u00e1s del 80% de todas las patentes de IA a nivel mundial."
        ]
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "import geopandas as gpd\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib.colors as mcolors\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "\n",
            "# ---- Rutas ----\n",
            "DATA_DIR = Path(\"Data\")\n",
            "INPUT_JSON = DATA_DIR / \"oecd_ai_patents_2020_clean.json\"\n",
            "OUTPUT_PNG = DATA_DIR / \"ocde_ia\" / \"mapa_patentes_ia_2022.png\"\n",
            "YEAR_FILTER = \"2022\"\n",
            "\n",
            "# Nombres en espa\u00f1ol\n",
            "ISO3_ES = {\n",
            "    \"USA\": \"Estados Unidos\", \"CHN\": \"China\", \"JPN\": \"Jap\u00f3n\",\n",
            "    \"KOR\": \"Corea del Sur\", \"DEU\": \"Alemania\", \"TWN\": \"Taiw\u00e1n\",\n",
            "    \"FRA\": \"Francia\", \"CAN\": \"Canad\u00e1\", \"GBR\": \"Reino Unido\",\n",
            "    \"SWE\": \"Suecia\", \"NLD\": \"Pa\u00edses Bajos\", \"CHE\": \"Suiza\",\n",
            "    \"IND\": \"India\", \"ISR\": \"Israel\", \"FIN\": \"Finlandia\",\n",
            "    \"IRL\": \"Irlanda\", \"ITA\": \"Italia\", \"SGP\": \"Singapur\",\n",
            "    \"AUS\": \"Australia\", \"ESP\": \"Espa\u00f1a\", \"BRA\": \"Brasil\",\n",
            "    \"DNK\": \"Dinamarca\", \"NOR\": \"Noruega\", \"BEL\": \"B\u00e9lgica\",\n",
            "    \"AUT\": \"Austria\", \"RUS\": \"Rusia\", \"SAU\": \"Arabia Saudita\",\n",
            "    \"TUR\": \"Turqu\u00eda\", \"ARE\": \"Emiratos \u00c1rabes\", \"POL\": \"Polonia\",\n",
            "    \"HUN\": \"Hungr\u00eda\", \"PRT\": \"Portugal\", \"NZL\": \"Nueva Zelanda\",\n",
            "    \"LUX\": \"Luxemburgo\", \"GRC\": \"Grecia\", \"THA\": \"Tailandia\",\n",
            "    \"ZAF\": \"Sud\u00e1frica\", \"MEX\": \"M\u00e9xico\", \"CHL\": \"Chile\",\n",
            "}\n",
            "\n",
            "# ---- Cargar datos ----\n",
            "with open(INPUT_JSON, encoding=\"utf-8\") as f:\n",
            "    raw = json.load(f)\n",
            "\n",
            "df = pd.DataFrame(raw)\n",
            "df = df[df[\"year\"] == YEAR_FILTER].copy()\n",
            "df = df.groupby(\"iso3\", as_index=False).agg({\"country\": \"first\", \"ai_patents_value\": \"sum\"})\n",
            "df = df.sort_values(\"ai_patents_value\", ascending=False).reset_index(drop=True)\n",
            "df[\"country_es\"] = df[\"iso3\"].map(ISO3_ES).fillna(df[\"country\"])\n",
            "\n",
            "print(f\"Pa\u00edses con datos ({YEAR_FILTER}): {len(df)}\")\n",
            "print(f\"\\nTop 5:\")\n",
            "df[[\"country_es\", \"ai_patents_value\"]].head()"
        ],
        "outputs": [],
        "execution_count": None
    },
    {
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# ---- Construir mapa con GeoPandas ----\n",
            "NE_URL = \"https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip\"\n",
            "world = gpd.read_file(NE_URL)\n",
            "world = world.rename(columns={\"ISO_A3_EH\": \"iso3\"})\n",
            "\n",
            "# Corregir codigos faltantes\n",
            "for name, code in {\"France\": \"FRA\", \"Norway\": \"NOR\"}.items():\n",
            "    world.loc[world[\"NAME\"] == name, \"iso3\"] = code\n",
            "\n",
            "# Merge con datos de patentes\n",
            "world = world.merge(df[[\"iso3\", \"ai_patents_value\", \"country_es\"]], on=\"iso3\", how=\"left\")\n",
            "\n",
            "# ---- Escala logaritmica y paleta ----\n",
            "vmin, vmax = 1.0, float(df[\"ai_patents_value\"].max())\n",
            "norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)\n",
            "colors_list = [\"#f0f0f0\", \"#fef3c7\", \"#fcd34d\", \"#f59e0b\", \"#ea580c\", \"#dc2626\", \"#991b1b\"]\n",
            "cmap = mcolors.LinearSegmentedColormap.from_list(\"ia_patents\", colors_list, N=256)\n",
            "cmap.set_bad(color=\"#e5e7eb\")\n",
            "\n",
            "# ---- Dibujar ----\n",
            "fig, ax = plt.subplots(1, 1, figsize=(18, 10), facecolor=\"#0f172a\")\n",
            "ax.set_facecolor(\"#1e293b\")\n",
            "\n",
            "# Todos los paises (fondo gris)\n",
            "world.plot(ax=ax, color=\"#334155\", edgecolor=\"#475569\", linewidth=0.3)\n",
            "\n",
            "# Paises con datos (coloreados)\n",
            "has_data = world[\"ai_patents_value\"].notna()\n",
            "world[has_data].plot(ax=ax, column=\"ai_patents_value\", cmap=cmap, norm=norm,\n",
            "                    edgecolor=\"#475569\", linewidth=0.4, legend=False)\n",
            "\n",
            "# ---- Barra de color ----\n",
            "sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)\n",
            "sm.set_array([])\n",
            "cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.65)\n",
            "cbar.set_label(\"Patentes de IA (escala logar\u00edtmica)\", fontsize=12, color=\"white\", labelpad=10)\n",
            "cbar.ax.tick_params(colors=\"white\", labelsize=9)\n",
            "\n",
            "# ---- Anotar Top 10 ----\n",
            "centroids = world[has_data].copy()\n",
            "centroids[\"centroid\"] = centroids.geometry.representative_point()\n",
            "centroids = centroids.sort_values(\"ai_patents_value\", ascending=False).head(10)\n",
            "\n",
            "label_offsets = {\n",
            "    \"USA\": (-25, 8), \"CHN\": (12, -12), \"JPN\": (18, 5), \"KOR\": (15, -8),\n",
            "    \"DEU\": (-5, 10), \"TWN\": (15, -3), \"FRA\": (-15, -10),\n",
            "    \"GBR\": (-18, 8), \"CAN\": (-20, 15), \"SWE\": (5, 12),\n",
            "}\n",
            "\n",
            "for _, row in centroids.iterrows():\n",
            "    iso = row[\"iso3\"]\n",
            "    cx, cy = row[\"centroid\"].x, row[\"centroid\"].y\n",
            "    dx, dy = label_offsets.get(iso, (8, 5))\n",
            "    label = f\"{row['country_es']}\\n{row['ai_patents_value']:,.0f}\"\n",
            "    ax.annotate(label, xy=(cx, cy), xytext=(cx + dx, cy + dy),\n",
            "                fontsize=7.5, color=\"white\", fontweight=\"bold\", ha=\"center\", va=\"center\",\n",
            "                arrowprops=dict(arrowstyle=\"-\", color=\"#94a3b8\", lw=0.7),\n",
            "                bbox=dict(boxstyle=\"round,pad=0.3\", facecolor=\"#1e293b\",\n",
            "                          edgecolor=\"#64748b\", alpha=0.85))\n",
            "\n",
            "# ---- Titulo y notas ----\n",
            "ax.set_title(\"Patentes de Inteligencia Artificial por Pa\u00eds \\u2014 2022\",\n",
            "             fontsize=20, fontweight=\"bold\", color=\"white\", pad=20)\n",
            "ax.text(0.5, 1.02, \"Fuente: OCDE \\u00b7 Familias de patentes IP5 relacionadas con IA\",\n",
            "        transform=ax.transAxes, fontsize=10, color=\"#94a3b8\", ha=\"center\", va=\"bottom\")\n",
            "ax.text(0.01, 0.02,\n",
            "        \"Nota: Las patentes se cuentan por familias IP5 (5 principales oficinas mundiales).\\n\"\n",
            "        \"Un valor fraccionario indica co-invenci\u00f3n entre pa\u00edses.\\n\"\n",
            "        \"Los pa\u00edses sin datos aparecen en gris oscuro.\",\n",
            "        transform=ax.transAxes, fontsize=8, color=\"#94a3b8\", va=\"bottom\", fontstyle=\"italic\")\n",
            "\n",
            "ax.set_xlim(-180, 180)\n",
            "ax.set_ylim(-60, 85)\n",
            "ax.axis(\"off\")\n",
            "plt.tight_layout()\n",
            "\n",
            "# Guardar\n",
            "OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)\n",
            "fig.savefig(OUTPUT_PNG, dpi=200, bbox_inches=\"tight\", facecolor=fig.get_facecolor())\n",
            "plt.show()\n",
            "print(f\"Mapa guardado en: {OUTPUT_PNG}\")"
        ],
        "outputs": [],
        "execution_count": None
    }
]

# Agregar las celdas al final
nb["cells"].extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Celdas agregadas. Total ahora: {len(nb['cells'])}")
