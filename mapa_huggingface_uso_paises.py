"""Genera un mapa mundial de uso estimado de modelos de IA por pais.

Metodologia de inferencia geografica:
1) Se usa card_languages como proxy principal de pais.
2) Si no hay idioma, se usa el owner del modelo como fallback.

Salidas:
- Data/huggingface_ia/huggingface_uso_estimado_por_pais.csv
- Data/huggingface_ia/huggingface_uso_modelos_mapa.html
- Data/huggingface_ia/huggingface_uso_modelos_mapa.png (opcional, requiere kaleido)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import pandas as pd
import plotly.express as px

Method = Literal["language_proxy", "owner_fallback", "unmapped"]


@dataclass(frozen=True)
class OutputPaths:
    input_csv: Path = Path("Data/huggingface_ia/huggingface_models_ia.csv")
    output_csv: Path = Path("Data/huggingface_ia/huggingface_uso_estimado_por_pais.csv")
    output_html: Path = Path("Data/huggingface_ia/huggingface_uso_modelos_mapa.html")
    output_png: Path = Path("Data/huggingface_ia/huggingface_uso_modelos_mapa.png")


PATHS: Final[OutputPaths] = OutputPaths()
REQUIRED_COLUMNS: Final[set[str]] = {"id", "downloads"}
OPTIONAL_COLUMNS: Final[set[str]] = {"card_languages"}

COLOR_SCALE: Final[list[tuple[float, str]]] = [
    (0.0, "#fff8e7"),
    (0.25, "#ffd36b"),
    (0.5, "#ff9d3f"),
    (0.75, "#f26a3d"),
    (1.0, "#b6362d"),
]

ISO3_TO_COUNTRY_NAME: Final[dict[str, str]] = {
    "AFG": "Afganistan",
    "ALB": "Albania",
    "ARM": "Armenia",
    "AZE": "Azerbaiyan",
    "BEL": "Belgica",
    "BGD": "Bangladesh",
    "BGR": "Bulgaria",
    "BRA": "Brasil",
    "CHN": "China",
    "CZE": "Chequia",
    "DEU": "Alemania",
    "DNK": "Dinamarca",
    "ESP": "Espana",
    "EST": "Estonia",
    "ETH": "Etiopia",
    "FIN": "Finlandia",
    "FRA": "Francia",
    "GBR": "Reino Unido",
    "GEO": "Georgia",
    "GRC": "Grecia",
    "HRV": "Croacia",
    "HUN": "Hungria",
    "IDN": "Indonesia",
    "IND": "India",
    "IRL": "Irlanda",
    "IRN": "Iran",
    "ISL": "Islandia",
    "ISR": "Israel",
    "ITA": "Italia",
    "JPN": "Japon",
    "KAZ": "Kazajistan",
    "KEN": "Kenia",
    "KHM": "Camboya",
    "KOR": "Corea del Sur",
    "LAO": "Laos",
    "LKA": "Sri Lanka",
    "LTU": "Lituania",
    "LUX": "Luxemburgo",
    "LVA": "Letonia",
    "MLT": "Malta",
    "MMR": "Myanmar",
    "MNG": "Mongolia",
    "MYS": "Malasia",
    "NGA": "Nigeria",
    "NLD": "Paises Bajos",
    "NOR": "Noruega",
    "NPL": "Nepal",
    "PAK": "Pakistan",
    "POL": "Polonia",
    "ROU": "Rumania",
    "RUS": "Rusia",
    "RWA": "Ruanda",
    "SAU": "Arabia Saudita",
    "SRB": "Serbia",
    "SVK": "Eslovaquia",
    "SVN": "Eslovenia",
    "SWE": "Suecia",
    "THA": "Tailandia",
    "TUR": "Turquia",
    "UKR": "Ucrania",
    "USA": "Estados Unidos",
    "UZB": "Uzbekistan",
    "VNM": "Vietnam",
    "ZAF": "Sudafrica",
}

LANG_TO_ISO3: Final[dict[str, str]] = {
    "af": "ZAF",
    "am": "ETH",
    "ar": "SAU",
    "az": "AZE",
    "bg": "BGR",
    "bn": "BGD",
    "ca": "ESP",
    "cs": "CZE",
    "cy": "GBR",
    "da": "DNK",
    "de": "DEU",
    "el": "GRC",
    "en": "USA",
    "es": "ESP",
    "et": "EST",
    "eu": "ESP",
    "fa": "IRN",
    "fi": "FIN",
    "fr": "FRA",
    "ga": "IRL",
    "gl": "ESP",
    "ha": "NGA",
    "he": "ISR",
    "hi": "IND",
    "hr": "HRV",
    "hu": "HUN",
    "hy": "ARM",
    "id": "IDN",
    "ig": "NGA",
    "is": "ISL",
    "it": "ITA",
    "ja": "JPN",
    "ka": "GEO",
    "kk": "KAZ",
    "km": "KHM",
    "ko": "KOR",
    "lb": "LUX",
    "lo": "LAO",
    "lt": "LTU",
    "lv": "LVA",
    "mn": "MNG",
    "ms": "MYS",
    "mt": "MLT",
    "my": "MMR",
    "ne": "NPL",
    "nl": "NLD",
    "no": "NOR",
    "pl": "POL",
    "ps": "AFG",
    "pt": "BRA",
    "ro": "ROU",
    "ru": "RUS",
    "rw": "RWA",
    "si": "LKA",
    "sk": "SVK",
    "sl": "SVN",
    "sq": "ALB",
    "sr": "SRB",
    "sv": "SWE",
    "sw": "KEN",
    "ta": "IND",
    "te": "IND",
    "th": "THA",
    "tr": "TUR",
    "uk": "UKR",
    "ur": "PAK",
    "uz": "UZB",
    "vi": "VNM",
    "xh": "ZAF",
    "yo": "NGA",
    "zh": "CHN",
    "zu": "ZAF",
}

OWNER_TO_ISO3: Final[dict[str, str]] = {
    "baai": "CHN",
    "deepseek-ai": "CHN",
    "eleutherai": "USA",
    "facebook": "USA",
    "facebookai": "USA",
    "google": "USA",
    "meta-llama": "USA",
    "mistralai": "FRA",
    "moonshotai": "CHN",
    "openai": "USA",
    "openai-community": "USA",
    "prosusai": "NLD",
    "pyannote": "BEL",
    "qwen": "CHN",
    "salesforce": "USA",
    "stabilityai": "GBR",
    "zai-org": "CHN",
}


def split_semicolon_values(value: object) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [token.strip().lower() for token in text.split(";") if token.strip()]


def extract_owner(model_id: str) -> str:
    if not model_id or "/" not in model_id:
        return ""
    return model_id.split("/", 1)[0].strip().lower()


def infer_countries(model_id: str, card_languages: object) -> tuple[list[str], Method]:
    iso3_from_languages = [
        LANG_TO_ISO3[lang]
        for lang in split_semicolon_values(card_languages)
        if lang in LANG_TO_ISO3
    ]

    unique_iso3 = list(dict.fromkeys(iso3_from_languages))
    if unique_iso3:
        return unique_iso3, "language_proxy"

    owner_iso3 = OWNER_TO_ISO3.get(extract_owner(model_id))
    if owner_iso3:
        return [owner_iso3], "owner_fallback"

    return [], "unmapped"


def load_source_dataframe(paths: OutputPaths) -> pd.DataFrame:
    if not paths.input_csv.exists():
        raise FileNotFoundError(f"No existe el archivo de entrada: {paths.input_csv}")

    df = pd.read_csv(paths.input_csv)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"El CSV no contiene columnas requeridas: {missing_list}")

    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["downloads"] = pd.to_numeric(df["downloads"], errors="coerce").fillna(0.0)
    return df


def aggregate_usage_by_country(df: pd.DataFrame) -> tuple[pd.DataFrame, Counter[str]]:
    usage_totals: defaultdict[str, float] = defaultdict(float)
    usage_models: defaultdict[str, set[str]] = defaultdict(set)
    method_counter: Counter[str] = Counter(
        {
            "language_proxy": 0,
            "owner_fallback": 0,
            "unmapped": 0,
        }
    )

    for model_id, downloads, card_languages in df[["id", "downloads", "card_languages"]].itertuples(
        index=False,
        name=None,
    ):
        countries, method = infer_countries(str(model_id), card_languages)
        method_counter[method] += 1

        if not countries or float(downloads) <= 0.0:
            continue

        downloads_share = float(downloads) / len(countries)
        for country_iso3 in countries:
            usage_totals[country_iso3] += downloads_share
            usage_models[country_iso3].add(str(model_id))

    if not usage_totals:
        return pd.DataFrame(), method_counter

    out = pd.DataFrame(
        {
            "country_iso3": list(usage_totals.keys()),
            "estimated_downloads": list(usage_totals.values()),
            "models_contributing": [len(usage_models[c]) for c in usage_totals.keys()],
        }
    )

    out = out.sort_values("estimated_downloads", ascending=False).reset_index(drop=True)
    total_downloads = float(out["estimated_downloads"].sum())
    out["estimated_downloads_pct"] = (out["estimated_downloads"] / total_downloads) * 100.0
    return out, method_counter


def prepare_plot_dataframe(df_country: pd.DataFrame) -> pd.DataFrame:
    df_plot = df_country.copy()
    df_plot["country_name"] = df_plot["country_iso3"].map(ISO3_TO_COUNTRY_NAME).fillna(df_plot["country_iso3"])
    return df_plot


def build_map_figure(df_plot: pd.DataFrame):
    fig = px.choropleth(
        df_plot,
        locations="country_iso3",
        color="estimated_downloads",
        hover_name="country_name",
        hover_data={
            "country_iso3": False,
            "estimated_downloads": False,
            "estimated_downloads_pct": False,
            "models_contributing": False,
            "country_name": False,
        },
        color_continuous_scale=COLOR_SCALE,
        projection="natural earth",
        title="Uso estimado de modelos de IA por pais",
    )

    fig.update_traces(
        marker_line_color="#7f8ea3",
        marker_line_width=0.4,
        customdata=df_plot[
            [
                "country_name",
                "country_iso3",
                "estimated_downloads",
                "estimated_downloads_pct",
                "models_contributing",
            ]
        ].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})"
            "<br>Descargas estimadas: %{customdata[2]:,.0f}"
            "<br>Participacion estimada: %{customdata[3]:.2f}%"
            "<br>Modelos aportantes: %{customdata[4]:,.0f}"
            "<extra></extra>"
        ),
    )

    fig.update_geos(
        showcountries=True,
        countrycolor="#9ba8b8",
        showcoastlines=False,
        showland=True,
        landcolor="#f5f3ea",
        showocean=True,
        oceancolor="#dfe8f2",
        bgcolor="#f7f9fc",
        showframe=False,
    )

    fig.update_layout(
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="#f7f9fc",
        plot_bgcolor="#f7f9fc",
        font={"family": "Segoe UI, Arial, sans-serif", "size": 13, "color": "#1f2937"},
        title={"x": 0.5, "xanchor": "center", "y": 0.95, "yanchor": "top"},
        coloraxis_colorbar={"title": "Descargas estimadas", "tickformat": ",.0f", "len": 0.72},
    )

    fig.add_annotation(
        text=(
            "Nota: el pais se infiere por idioma declarado en el modelo; "
            "cuando falta, se usa el owner como fallback."
        ),
        xref="paper",
        yref="paper",
        x=0.0,
        y=0.0,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font={"size": 11, "color": "#4b5563"},
        align="left",
    )
    return fig


def save_outputs(df_country: pd.DataFrame, paths: OutputPaths) -> None:
    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_country.to_csv(paths.output_csv, index=False, encoding="utf-8")

    fig = build_map_figure(prepare_plot_dataframe(df_country))
    fig.write_html(paths.output_html, include_plotlyjs="cdn")

    try:
        fig.write_image(paths.output_png, width=1400, height=800, scale=2)
    except Exception:
        # Requiere kaleido. El HTML siempre queda generado.
        pass


def print_summary(paths: OutputPaths, method_counter: Counter[str]) -> None:
    print("[OK] Mapa generado")
    print(f"- CSV:  {paths.output_csv}")
    print(f"- HTML: {paths.output_html}")
    if paths.output_png.exists():
        print(f"- PNG:  {paths.output_png}")
    else:
        print("- PNG:  no generado (instala kaleido para exportar imagen)")

    print("\nModelos procesados por metodo:")
    print(f"- language_proxy: {method_counter.get('language_proxy', 0)}")
    print(f"- owner_fallback: {method_counter.get('owner_fallback', 0)}")
    print(f"- unmapped: {method_counter.get('unmapped', 0)}")


def main(paths: OutputPaths = PATHS) -> None:
    df = load_source_dataframe(paths)
    df_country, method_counter = aggregate_usage_by_country(df)

    if df_country.empty:
        raise RuntimeError("No fue posible inferir paises a partir de card_languages/owner.")

    save_outputs(df_country, paths)
    print_summary(paths, method_counter)


if __name__ == "__main__":
    main()
