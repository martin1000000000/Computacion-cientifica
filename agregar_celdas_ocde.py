"""
Script para agregar celdas al final de prueba_02.ipynb
con el código que lee JSON de la OCDE y genera gráficos.
"""
import json

# Leer el notebook existente
with open('prueba_02.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Nuevas celdas a agregar
nuevas_celdas = []

# ─── Celda Markdown: Titulo de sección ───────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "## 📊 Indicadores de IA por País — Datos de la OCDE\n",
        "\n",
        "**Fuente:** [OECD.AI Policy Observatory](https://oecd.ai/en/data) / [OECD Data Explorer](https://data-explorer.oecd.org)  \n",
        "**API de referencia:** [OECD SDMX REST API](https://sdmx.oecd.org)  \n",
        "**Datasets:** ICT Access & Usage by Businesses, MSTI (GERD), OECD Patent Statistics, OECD.AI  \n",
        "\n",
        "En esta sección se cargan datos de indicadores relacionados con la **Inteligencia Artificial** desde un archivo JSON\n",
        "con información recopilada de la OCDE, y se generan gráficos comparativos segmentados por país.\n",
        "\n",
        "> ⚠️ El archivo `Data/ocde_ia_indicadores.json` contiene los datasets preprocesados\n",
        "> obtenidos de los reportes publicados por la OCDE."
    ]
})

# ─── Celda Code: Cargar JSON ─────────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import json\n",
        "import numpy as np\n",
        "import matplotlib.ticker as mtick\n",
        "\n",
        "# ─── Cargar datos desde el JSON de la OCDE ────────────────────────────\n",
        "with open('./Data/ocde_ia_indicadores.json', 'r', encoding='utf-8') as f:\n",
        "    ocde_data = json.load(f)\n",
        "\n",
        "# Mostrar metadatos\n",
        "meta = ocde_data['metadata']\n",
        "print(f\"📡 Fuente: {meta['fuente']}\")\n",
        "print(f\"🔗 URL:    {meta['url_fuente']}\")\n",
        "print(f\"📅 Fecha:  {meta['fecha_recopilacion']}\")\n",
        "print(f\"\\n📋 Datasets disponibles:\")\n",
        "for key, ds in ocde_data['datasets'].items():\n",
        "    print(f\"   • {key}: {ds['titulo']} ({ds['periodo']})\")"
    ]
})

# ─── Celda Markdown: Gráfico 1 ───────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📊 Gráfico 1: Empresas que utilizan tecnologías de IA por país"
    ]
})

# ─── Celda Code: Gráfico 1 ───────────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Paleta de colores moderna\n",
        "COLORES = [\n",
        "    '#2563eb', '#dc2626', '#16a34a', '#f59e0b', '#8b5cf6',\n",
        "    '#ec4899', '#06b6d4', '#f97316', '#6366f1', '#14b8a6',\n",
        "    '#e11d48', '#0891b2', '#7c3aed', '#059669', '#d97706'\n",
        "]\n",
        "\n",
        "# ─── Leer dataset desde JSON ─────────────────────────────────────────\n",
        "ds1 = ocde_data['datasets']['empresas_usando_ia']\n",
        "df1 = pd.DataFrame(ds1['datos'])\n",
        "df1 = df1.sort_values('valor')\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 8))\n",
        "\n",
        "bars = ax.barh(range(len(df1)), df1['valor'].values,\n",
        "               color=COLORES[:len(df1)], edgecolor='white',\n",
        "               height=0.7, alpha=0.85)\n",
        "ax.set_yticks(range(len(df1)))\n",
        "ax.set_yticklabels(df1['pais'].values, fontsize=10)\n",
        "ax.set_xlabel(ds1['unidad'], fontsize=12)\n",
        "ax.set_title(f\"{ds1['titulo']}\\n({ds1['fuente_especifica']}, {ds1['periodo']})\",\n",
        "             fontsize=14, fontweight='bold', pad=15)\n",
        "\n",
        "for i, (bar, val) in enumerate(zip(bars, df1['valor'].values)):\n",
        "    ax.text(val + 0.3, i, f'{val:.1f}%', va='center',\n",
        "            fontsize=9, fontweight='bold')\n",
        "\n",
        "ax.spines['top'].set_visible(False)\n",
        "ax.spines['right'].set_visible(False)\n",
        "ax.set_xlim(0, df1['valor'].max() + 3)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ─── Celda Markdown: Gráfico 2 ───────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📊 Gráfico 2: Gasto en I+D como porcentaje del PIB"
    ]
})

# ─── Celda Code: Gráfico 2 ───────────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ─── Leer dataset desde JSON ─────────────────────────────────────────\n",
        "ds2 = ocde_data['datasets']['gasto_id_pib']\n",
        "df2 = pd.DataFrame(ds2['datos'])\n",
        "df2 = df2.sort_values('valor', ascending=False)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(14, 8))\n",
        "\n",
        "# Colores: rojo para promedio OCDE, azul para el resto\n",
        "colores_rd = ['#e11d48' if p == 'OECD Average' else '#2563eb'\n",
        "              for p in df2['pais'].values]\n",
        "\n",
        "bars = ax.bar(range(len(df2)), df2['valor'].values,\n",
        "              color=colores_rd, edgecolor='white',\n",
        "              width=0.75, alpha=0.85)\n",
        "ax.set_xticks(range(len(df2)))\n",
        "ax.set_xticklabels(df2['pais'].values, rotation=45, ha='right', fontsize=9)\n",
        "ax.set_ylabel(ds2['unidad'], fontsize=12)\n",
        "ax.set_title(f\"{ds2['titulo']}\\n({ds2['fuente_especifica']}, {ds2['periodo']})\",\n",
        "             fontsize=14, fontweight='bold', pad=15)\n",
        "\n",
        "# Línea promedio OCDE\n",
        "avg = df2[df2['pais'] == 'OECD Average']['valor'].values[0]\n",
        "ax.axhline(y=avg, color='#e11d48', linestyle='--', linewidth=1.5,\n",
        "           alpha=0.7, label=f'Promedio OCDE ({avg}%)')\n",
        "ax.legend(fontsize=10, loc='upper right')\n",
        "\n",
        "for i, val in enumerate(df2['valor'].values):\n",
        "    ax.text(i, val + 0.05, f'{val:.2f}', ha='center', va='bottom',\n",
        "            fontsize=8, fontweight='bold')\n",
        "\n",
        "ax.spines['top'].set_visible(False)\n",
        "ax.spines['right'].set_visible(False)\n",
        "ax.set_ylim(0, df2['valor'].max() + 0.8)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ─── Celda Markdown: Gráfico 3 ───────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📊 Gráfico 3: Publicaciones científicas sobre IA"
    ]
})

# ─── Celda Code: Gráfico 3 ───────────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ─── Leer dataset desde JSON ─────────────────────────────────────────\n",
        "ds3 = ocde_data['datasets']['publicaciones_ia']\n",
        "df3 = pd.DataFrame(ds3['datos'])\n",
        "df3 = df3.sort_values('valor')\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 8))\n",
        "\n",
        "gradient_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df3)))\n",
        "\n",
        "bars = ax.barh(range(len(df3)), df3['valor'].values,\n",
        "               color=gradient_colors, edgecolor='white',\n",
        "               height=0.7, alpha=0.85)\n",
        "ax.set_yticks(range(len(df3)))\n",
        "ax.set_yticklabels(df3['pais'].values, fontsize=10)\n",
        "ax.set_xlabel(ds3['unidad'], fontsize=12)\n",
        "ax.set_title(f\"{ds3['titulo']}\\n({ds3['fuente_especifica']}, {ds3['periodo']})\",\n",
        "             fontsize=14, fontweight='bold', pad=15)\n",
        "\n",
        "for i, val in enumerate(df3['valor'].values):\n",
        "    ax.text(val + 1, i, f'{val:.1f}k', va='center',\n",
        "            fontsize=9, fontweight='bold')\n",
        "\n",
        "ax.spines['top'].set_visible(False)\n",
        "ax.spines['right'].set_visible(False)\n",
        "ax.set_xlim(0, df3['valor'].max() + 20)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ─── Celda Markdown: Gráfico 4 ───────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📊 Gráfico 4: Patentes de IA por país"
    ]
})

# ─── Celda Code: Gráfico 4 ───────────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ─── Leer dataset desde JSON ─────────────────────────────────────────\n",
        "ds4 = ocde_data['datasets']['patentes_ia']\n",
        "df4 = pd.DataFrame(ds4['datos'])\n",
        "df4 = df4.sort_values('valor', ascending=False)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(14, 7))\n",
        "\n",
        "bars = ax.bar(range(len(df4)), df4['valor'].values,\n",
        "              color=plt.cm.plasma(np.linspace(0.15, 0.85, len(df4))),\n",
        "              edgecolor='white', width=0.75, alpha=0.85)\n",
        "ax.set_xticks(range(len(df4)))\n",
        "ax.set_xticklabels(df4['pais'].values, rotation=45, ha='right', fontsize=9)\n",
        "ax.set_ylabel(ds4['unidad'], fontsize=12)\n",
        "ax.set_title(f\"{ds4['titulo']}\\n({ds4['fuente_especifica']}, {ds4['periodo']})\",\n",
        "             fontsize=14, fontweight='bold', pad=15)\n",
        "\n",
        "for i, val in enumerate(df4['valor'].values):\n",
        "    ax.text(i, val + 300, f'{val:,}', ha='center', va='bottom',\n",
        "            fontsize=8, fontweight='bold')\n",
        "\n",
        "ax.spines['top'].set_visible(False)\n",
        "ax.spines['right'].set_visible(False)\n",
        "ax.set_ylim(0, df4['valor'].max() * 1.15)\n",
        "ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: f'{x:,.0f}'))\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ─── Celda Markdown: Gráfico 5 ───────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📊 Gráfico 5: Inversión privada en IA"
    ]
})

# ─── Celda Code: Gráfico 5 ───────────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ─── Leer dataset desde JSON ─────────────────────────────────────────\n",
        "ds5 = ocde_data['datasets']['inversion_privada_ia']\n",
        "df5 = pd.DataFrame(ds5['datos'])\n",
        "df5 = df5.sort_values('valor', ascending=False)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(14, 7))\n",
        "\n",
        "bars = ax.bar(range(len(df5)), df5['valor'].values,\n",
        "              color=plt.cm.magma(np.linspace(0.2, 0.8, len(df5))),\n",
        "              edgecolor='white', width=0.75, alpha=0.85)\n",
        "ax.set_xticks(range(len(df5)))\n",
        "ax.set_xticklabels(df5['pais'].values, rotation=45, ha='right', fontsize=9)\n",
        "ax.set_ylabel(ds5['unidad'], fontsize=12)\n",
        "ax.set_title(f\"{ds5['titulo']}\\n({ds5['fuente_especifica']}, {ds5['periodo']})\",\n",
        "             fontsize=14, fontweight='bold', pad=15)\n",
        "\n",
        "for i, val in enumerate(df5['valor'].values):\n",
        "    ax.text(i, val + 0.5, f'${val:.1f}B', ha='center', va='bottom',\n",
        "            fontsize=8, fontweight='bold')\n",
        "\n",
        "ax.spines['top'].set_visible(False)\n",
        "ax.spines['right'].set_visible(False)\n",
        "ax.set_ylim(0, df5['valor'].max() * 1.15)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ─── Celda Markdown: Gráfico 6 ───────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### 📊 Gráfico 6: Relación entre Gasto en I+D y Adopción de IA\n",
        "\n",
        "Scatter plot que muestra la correlación entre el gasto en I+D (% del PIB) y el porcentaje de empresas que adoptan IA en cada país."
    ]
})

# ─── Celda Code: Gráfico 6 ───────────────────────────────────────────────────
nuevas_celdas.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# ─── Leer dataset desde JSON ─────────────────────────────────────────\n",
        "ds6 = ocde_data['datasets']['rd_vs_adopcion_ia']\n",
        "df6 = pd.DataFrame(ds6['datos'])\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(12, 8))\n",
        "\n",
        "for i, row in df6.iterrows():\n",
        "    color = COLORES[i % len(COLORES)]\n",
        "    ax.scatter(row['gasto_id'], row['adopcion_ia'], s=120,\n",
        "               color=color, edgecolors='white', linewidth=1.5, zorder=5)\n",
        "    ax.annotate(row['pais'], (row['gasto_id'], row['adopcion_ia']),\n",
        "                textcoords='offset points', xytext=(8, 5),\n",
        "                fontsize=8, fontweight='bold')\n",
        "\n",
        "# Línea de tendencia\n",
        "x_vals = df6['gasto_id'].values\n",
        "y_vals = df6['adopcion_ia'].values\n",
        "z = np.polyfit(x_vals, y_vals, 1)\n",
        "p = np.poly1d(z)\n",
        "r2 = np.corrcoef(x_vals, y_vals)[0, 1] ** 2\n",
        "x_line = np.linspace(x_vals.min() - 0.2, x_vals.max() + 0.2, 100)\n",
        "ax.plot(x_line, p(x_line), '--', color='gray', alpha=0.5, linewidth=1.5,\n",
        "        label=f'Tendencia lineal (R² ≈ {r2:.2f})')\n",
        "\n",
        "ax.set_xlabel(ds6['variables'][0], fontsize=12)\n",
        "ax.set_ylabel(ds6['variables'][1], fontsize=12)\n",
        "ax.set_title(f\"{ds6['titulo']}\\n({ds6['fuente_especifica']}, {ds6['periodo']})\",\n",
        "             fontsize=14, fontweight='bold', pad=15)\n",
        "ax.legend(fontsize=10)\n",
        "ax.spines['top'].set_visible(False)\n",
        "ax.spines['right'].set_visible(False)\n",
        "ax.grid(True, alpha=0.3)\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# ─── Agregar celdas al notebook ───────────────────────────────────────────────
nb['cells'].extend(nuevas_celdas)

# Guardar notebook
with open('prueba_02.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"✅ Se agregaron {len(nuevas_celdas)} celdas al final de prueba_02.ipynb")
print("   Incluye:")
print("   • 1 celda markdown de introducción")
print("   • 1 celda para cargar el JSON")
print("   • 6 gráficos (cada uno con markdown + código)")
print("\n   Todos leen datos desde: Data/ocde_ia_indicadores.json")
