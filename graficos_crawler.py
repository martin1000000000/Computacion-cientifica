"""Genera graficos OCDE desde los CSV del crawler.

Este script toma los CSV limpios producidos por
`crawler/descargar_ocde_ia.py` en `Data/ocde_ia/` y recrea 6 graficos:

1. Empresas con instrumentos tech-focused (proxy IA)
2. Gasto en I+D apoyado por politica industrial
3. Publicaciones IA (proxy bibliometrico)
4. Patentes IA
5. Inversion en instrumentos IA/tecnologia (proxy)
6. Relacion I+D vs tech-focused

Uso:
    python graficos_crawler.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

BASE_DIR = Path(__file__).resolve().parent
DATA_OECD = BASE_DIR / "Data" / "ocde_ia"
OUT_OECD = BASE_DIR / "graficos_api_ocde"
OUT_OECD.mkdir(parents=True, exist_ok=True)
OUT_OECD_DATA = OUT_OECD / "data"
OUT_OECD_DATA.mkdir(parents=True, exist_ok=True)

if sns is not None:
    sns.set_theme(style="whitegrid")
else:
    plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.labelcolor"] = "#2f2f2f"
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9

SOURCE_NOTE = "Fuente: OCDE (SDMX), procesamiento propio"


def read_oecd_csv(filename: str, usecols: list[str] | None = None) -> pd.DataFrame:
    path = DATA_OECD / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el archivo: {path}. Ejecuta crawler/descargar_ocde_ia.py primero."
        )
    return pd.read_csv(path, low_memory=False, usecols=usecols)


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def filter_positive(df: pd.DataFrame, value_col: str = "OBS_VALUE") -> pd.DataFrame:
    out = df.copy()
    out[value_col] = to_numeric(out[value_col])
    return out[out[value_col].notna() & (out[value_col] > 0)].copy()


def latest_year(df: pd.DataFrame, year_col: str = "TIME_PERIOD") -> pd.DataFrame:
    if year_col not in df.columns:
        return df.copy()

    years = to_numeric(df[year_col]).dropna()
    if years.empty:
        return df.copy()

    return df[to_numeric(df[year_col]) == years.max()].copy()


def latest_year_value(df: pd.DataFrame, year_col: str = "TIME_PERIOD") -> int | None:
    if year_col not in df.columns:
        return None

    years = to_numeric(df[year_col]).dropna()
    if years.empty:
        return None

    return int(years.max())


def remove_aggregate_areas(df: pd.DataFrame, area_col: str = "REF_AREA_label") -> pd.DataFrame:
    if area_col not in df.columns:
        return df.copy()

    area_lower = df[area_col].fillna("").astype(str).str.lower()
    blocked_pattern = "oecd|world|european union|euro area|non-oecd economies|not applicable"
    return df.loc[~area_lower.str.contains(blocked_pattern, regex=True)].copy()


def minmax_robust(
    series: pd.Series,
    q_low: float = 0.05,
    q_high: float = 0.95,
) -> pd.Series:
    values = to_numeric(series).fillna(0)
    low = values.quantile(q_low)
    high = values.quantile(q_high)
    clipped = values.clip(lower=low, upper=high)

    if high <= low:
        return pd.Series(np.zeros(len(clipped)), index=clipped.index)

    return (clipped - low) / (high - low)


def build_summary(chart_code: str, agg: pd.DataFrame) -> pd.DataFrame:
    summary = agg.reset_index(drop=True).copy()
    summary.insert(0, "rank", summary.index + 1)
    summary.insert(0, "chart", chart_code)
    summary = summary.rename(
        columns={
            "REF_AREA_label": "pais",
            "OBS_VALUE": "valor_agregado",
        }
    )
    return summary


def count_unique_by_country(
    df: pd.DataFrame,
    country_col: str = "REF_AREA_label",
    id_col: str = "UNIQUE_ID",
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[country_col, "OBS_VALUE"])

    agg = (
        df.groupby(country_col, as_index=False)[id_col]
        .nunique()
        .rename(columns={id_col: "OBS_VALUE"})
    )
    return agg


def format_compact(value: float) -> str:
    if value is None or np.isnan(value):
        return ""
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_val >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{value / 1_000:.1f}K"
    if abs_val >= 100:
        return f"{value:,.0f}"
    if abs_val >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def save_chart_data(
    df: pd.DataFrame,
    filename: str,
    chart_code: str | None = None,
    year: int | None = None,
) -> None:
    if df.empty:
        return

    out = df.copy()
    insert_at = 0
    if chart_code and "chart" not in out.columns:
        out.insert(insert_at, "chart", chart_code)
        insert_at += 1
    if year and "year" not in out.columns:
        out.insert(insert_at, "year", year)

    out_path = OUT_OECD_DATA / filename
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Guardado: {out_path}")


def top_country_bar(
    df: pd.DataFrame,
    title: str,
    output_name: str,
    country_col: str = "REF_AREA_label",
    value_col: str = "OBS_VALUE",
    top_n: int = 15,
    color: str = "#2f5597",
    x_label: str = "Valor agregado",
) -> pd.DataFrame:
    if df.empty:
        print(f"[WARN] No hay datos para {output_name}")
        return pd.DataFrame(columns=[country_col, value_col])

    agg = (
        df.groupby(country_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
        .head(top_n)
    )

    plot_df = agg.sort_values(value_col, ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(plot_df[country_col], plot_df[value_col], color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pais")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(axis="x", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_max = plot_df[value_col].max()
    for idx, value in enumerate(plot_df[value_col]):
        ax.text(
            value + (x_max * 0.01),
            idx,
            format_compact(value),
            va="center",
            ha="left",
            fontsize=8,
            color="#2f2f2f",
        )

    fig.text(0.01, 0.01, SOURCE_NOTE, ha="left", fontsize=8, color="#666666")
    fig.tight_layout()

    out_path = OUT_OECD / output_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Guardado: {out_path}")
    return agg


def generar_graficos_ocde() -> pd.DataFrame:
    summaries: list[pd.DataFrame] = []

    # Grafico 01: Empresas tech-focused como proxy de IA.
    df_emp = read_oecd_csv(
        "dsd_industrial_policy_df_grantax_clean.csv",
        usecols=["UNIQUE_ID", "REF_AREA_label", "TIME_PERIOD", "TECH_FOCUSED_label"],
    )
    emp_year = latest_year_value(df_emp)
    df_emp = remove_aggregate_areas(latest_year(df_emp))
    emp_mask = df_emp["TECH_FOCUSED_label"].astype(str).str.lower().eq("yes")
    df_emp_plot = df_emp.loc[emp_mask].copy()
    if df_emp_plot.empty:
        df_emp_plot = df_emp.copy()

    agg = count_unique_by_country(df_emp_plot)
    agg = top_country_bar(
        agg,
        title=f"Grafico 01: Instrumentos tech-focused (QuIS{', ' + str(emp_year) if emp_year else ''})",
        output_name="ocde_grafico_01_empresas_ia.png",
        color="#4e79a7",
        x_label="Instrumentos",
    )
    if not agg.empty:
        save_chart_data(
            agg,
            "ocde_grafico_01_empresas_ia_datos.csv",
            chart_code="grafico_01_empresas_ia",
            year=emp_year,
        )
        summaries.append(build_summary("grafico_01_empresas_ia", agg))

    # Grafico 02: Gasto en I+D apoyado por politica industrial.
    df_rd = read_oecd_csv(
        "dsd_industrial_policy_df_fin_clean.csv",
        usecols=["UNIQUE_ID", "REF_AREA_label", "TIME_PERIOD", "RD_label"],
    )
    rd_year = latest_year_value(df_rd)
    df_rd = remove_aggregate_areas(latest_year(df_rd))
    rd_mask = df_rd["RD_label"].astype(str).str.lower().eq("yes")
    df_rd_plot = df_rd.loc[rd_mask].copy()
    if df_rd_plot.empty:
        df_rd_plot = df_rd.copy()

    agg = count_unique_by_country(df_rd_plot)
    agg = top_country_bar(
        agg,
        title=(
            f"Grafico 02: Instrumentos con apoyo a I+D (QuIS"
            f"{', ' + str(rd_year) if rd_year else ''})"
        ),
        output_name="ocde_grafico_02_gasto_id.png",
        color="#f28e2b",
        x_label="Instrumentos",
    )
    if not agg.empty:
        save_chart_data(
            agg,
            "ocde_grafico_02_gasto_id_datos.csv",
            chart_code="grafico_02_gasto_id",
            year=rd_year,
        )
        summaries.append(build_summary("grafico_02_gasto_id", agg))

    # Grafico 03: Publicaciones IA por areas bibliometricas.
    df_pub = read_oecd_csv(
        "dsd_biblio_df_biblio_clean.csv",
        usecols=[
            "REF_AREA_label",
            "TIME_PERIOD",
            "OBS_VALUE",
            "ASJC_label",
            "MEASURE_label",
            "UNIT_MEASURE_label",
        ],
    )
    pub_year = latest_year_value(df_pub)
    df_pub = remove_aggregate_areas(latest_year(filter_positive(df_pub)))
    df_pub = df_pub[
        df_pub["MEASURE_label"].astype(str).str.lower().eq(
            "fractional counts of scientific publications"
        )
    ]
    df_pub = df_pub[
        df_pub["UNIT_MEASURE_label"].astype(str).str.lower().eq("scientific publications")
    ]
    asjc = df_pub["ASJC_label"].fillna("").astype(str).str.lower()
    kw_pub = "artificial|machine|deep|neural|language|vision|data|comput"
    df_pub_plot = df_pub[asjc.str.contains(kw_pub, regex=True)].copy()
    if df_pub_plot.empty:
        df_pub_plot = df_pub.copy()

    agg = top_country_bar(
        df_pub_plot,
        title=f"Grafico 03: Publicaciones IA (ASJC{', ' + str(pub_year) if pub_year else ''})",
        output_name="ocde_grafico_03_publicaciones_ia.png",
        color="#59a14f",
        x_label="Publicaciones",
    )
    if not agg.empty:
        save_chart_data(
            agg,
            "ocde_grafico_03_publicaciones_ia_datos.csv",
            chart_code="grafico_03_publicaciones_ia",
            year=pub_year,
        )
        summaries.append(build_summary("grafico_03_publicaciones_ia", agg))

    # Grafico 04: Patentes IA por tecnologia patentada.
    df_pat = read_oecd_csv(
        "dsd_patents_df_patents_oecdspecific_clean.csv",
        usecols=[
            "REF_AREA_label",
            "TIME_PERIOD",
            "OBS_VALUE",
            "OECD_TECHNOLOGY_PATENT_label",
            "MEASURE_label",
            "UNIT_MEASURE_label",
        ],
    )
    pat_year = latest_year_value(df_pat)
    df_pat = remove_aggregate_areas(latest_year(filter_positive(df_pat)))
    df_pat = df_pat[
        df_pat["MEASURE_label"].astype(str).str.lower().eq("patent families")
    ]
    df_pat = df_pat[
        df_pat["UNIT_MEASURE_label"].astype(str).str.lower().eq("sets of patents")
    ]
    tech = df_pat["OECD_TECHNOLOGY_PATENT_label"].fillna("").astype(str).str.lower()
    kw_pat = "artificial|machine|comput|digital|data|ict|software"
    df_pat_plot = df_pat[tech.str.contains(kw_pat, regex=True)].copy()
    if df_pat_plot.empty:
        df_pat_plot = df_pat.copy()

    agg = top_country_bar(
        df_pat_plot,
        title=f"Grafico 04: Patentes IA (familias{', ' + str(pat_year) if pat_year else ''})",
        output_name="ocde_grafico_04_patentes_ia.png",
        color="#e15759",
        x_label="Familias de patentes",
    )
    if not agg.empty:
        save_chart_data(
            agg,
            "ocde_grafico_04_patentes_ia_datos.csv",
            chart_code="grafico_04_patentes_ia",
            year=pat_year,
        )
        summaries.append(build_summary("grafico_04_patentes_ia", agg))

    # Grafico 05: Instrumentos de inversion como proxy IA/tecnologia.
    df_inv = read_oecd_csv(
        "dsd_industrial_policy_df_fin_clean.csv",
        usecols=["UNIQUE_ID", "REF_AREA_label", "TIME_PERIOD", "INSTRUMENT_TYPE_label"],
    )
    inv_year = latest_year_value(df_inv)
    df_inv = remove_aggregate_areas(latest_year(df_inv))
    instr = df_inv["INSTRUMENT_TYPE_label"].fillna("").astype(str).str.lower()
    kw_inv = "venture|equity|loan|capital|investment|fund"
    df_inv_plot = df_inv[instr.str.contains(kw_inv, regex=True)].copy()
    if df_inv_plot.empty:
        df_inv_plot = df_inv.copy()

    agg = count_unique_by_country(df_inv_plot)
    agg = top_country_bar(
        agg,
        title=f"Grafico 05: Instrumentos de inversion (QuIS{', ' + str(inv_year) if inv_year else ''})",
        output_name="ocde_grafico_05_inversion_ia.png",
        color="#76b7b2",
        x_label="Instrumentos",
    )
    if not agg.empty:
        save_chart_data(
            agg,
            "ocde_grafico_05_inversion_ia_datos.csv",
            chart_code="grafico_05_inversion_ia",
            year=inv_year,
        )
        summaries.append(build_summary("grafico_05_inversion_ia", agg))

    # Grafico 06: Relacion I+D vs tech-focused con escala robusta.
    df_fin_sc = read_oecd_csv(
        "dsd_industrial_policy_df_fin_clean.csv",
        usecols=[
            "UNIQUE_ID",
            "REF_AREA_label",
            "TIME_PERIOD",
            "RD_label",
            "TECH_FOCUSED_label",
        ],
    )
    df_gr_sc = read_oecd_csv(
        "dsd_industrial_policy_df_grantax_clean.csv",
        usecols=[
            "UNIQUE_ID",
            "REF_AREA_label",
            "TIME_PERIOD",
            "RD_label",
            "TECH_FOCUSED_label",
        ],
    )

    df_fin_sc = latest_year(df_fin_sc)
    df_gr_sc = latest_year(df_gr_sc)
    df_fin_sc = df_fin_sc.assign(SOURCE="fin")
    df_gr_sc = df_gr_sc.assign(SOURCE="grantax")
    df_sc = pd.concat([df_fin_sc, df_gr_sc], ignore_index=True)
    df_sc = remove_aggregate_areas(df_sc)
    df_sc["INSTRUMENT_KEY"] = df_sc["UNIQUE_ID"].astype(str) + "|" + df_sc["SOURCE"]
    sc_year = latest_year_value(df_sc)

    rd_country = (
        df_sc[df_sc["RD_label"].astype(str).str.lower().eq("yes")]
        .groupby("REF_AREA_label", as_index=False)["INSTRUMENT_KEY"]
        .nunique()
        .rename(columns={"INSTRUMENT_KEY": "rd_value"})
    )

    ia_country = (
        df_sc[df_sc["TECH_FOCUSED_label"].astype(str).str.lower().eq("yes")]
        .groupby("REF_AREA_label", as_index=False)["INSTRUMENT_KEY"]
        .nunique()
        .rename(columns={"INSTRUMENT_KEY": "ia_proxy_value"})
    )

    df_scatter = rd_country.merge(ia_country, on="REF_AREA_label", how="outer").fillna(0)
    df_scatter = df_scatter[
        (df_scatter["rd_value"] > 0) | (df_scatter["ia_proxy_value"] > 0)
    ].copy()
    df_scatter["rd_log"] = np.log1p(df_scatter["rd_value"])
    df_scatter["ia_log"] = np.log1p(df_scatter["ia_proxy_value"])
    df_scatter["rd_norm"] = minmax_robust(df_scatter["rd_log"])
    df_scatter["ia_norm"] = minmax_robust(df_scatter["ia_log"])
    df_scatter["size_norm"] = minmax_robust(
        np.log1p(df_scatter["rd_value"] + df_scatter["ia_proxy_value"])
    )

    fig, ax = plt.subplots(figsize=(11, 7))
    if sns is not None:
        sns.scatterplot(
            data=df_scatter,
            x="rd_norm",
            y="ia_norm",
            size="size_norm",
            sizes=(80, 700),
            alpha=0.75,
            color="#2f5597",
            legend=False,
            ax=ax,
        )
    else:
        sizes = 80 + (620 * df_scatter["size_norm"].fillna(0))
        ax.scatter(
            df_scatter["rd_norm"],
            df_scatter["ia_norm"],
            s=sizes,
            alpha=0.75,
            color="#2f5597",
            edgecolors="white",
            linewidths=0.8,
        )

    if len(df_scatter) >= 2:
        slope, intercept = np.polyfit(df_scatter["rd_norm"], df_scatter["ia_norm"], 1)
        x_line = np.linspace(0, 1, 200)
        ax.plot(x_line, slope * x_line + intercept, color="#d1495b", linewidth=2)

    for _, row in df_scatter.nlargest(8, "ia_norm").iterrows():
        ax.annotate(
            row["REF_AREA_label"],
            (row["rd_norm"], row["ia_norm"]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=8,
        )

    corr = df_scatter["rd_log"].corr(df_scatter["ia_log"])
    ax.set_title(
        "Grafico 06: Instrumentos I+D vs tech-focused"
        f"{f' ({sc_year})' if sc_year else ''}"
    )
    ax.set_xlabel("Instrumentos I+D (normalizado)")
    ax.set_ylabel("Instrumentos tech-focused (normalizado)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_scatter = OUT_OECD / "ocde_grafico_06_rd_vs_ia.png"
    fig.savefig(out_scatter, dpi=300, bbox_inches="tight")
    plt.close(fig)

    df_scatter_sorted = df_scatter.sort_values(["ia_norm", "rd_norm"], ascending=False)
    save_chart_data(
        df_scatter_sorted,
        "ocde_grafico_06_rd_vs_ia_datos.csv",
        chart_code="grafico_06_rd_vs_ia",
        year=sc_year,
    )
    print(f"[OK] Guardado: {out_scatter}")
    print(f"Correlacion logaritmica (r): {corr:.3f}")

    resumen = (
        pd.concat(summaries, ignore_index=True)
        if summaries
        else pd.DataFrame(columns=["chart", "rank", "pais", "valor_agregado"])
    )
    resumen_path = OUT_OECD_DATA / "ocde_graficos_resumen.csv"
    resumen.to_csv(resumen_path, index=False, encoding="utf-8-sig")

    print(f"[OK] Guardado: {resumen_path}")
    print(f"Graficos generados en: {OUT_OECD.resolve()}")
    return resumen


if __name__ == "__main__":
    generar_graficos_ocde()
