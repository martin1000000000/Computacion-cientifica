"""
Crawler de metadata de modelos de IA usando la API oficial de Hugging Face.

Fuentes oficiales:
- https://huggingface.co/docs/hub/main/api
- https://oecd.ai/en/huggingface

Objetivo:
- Descargar metadata cruda de modelos de IA reales desde Hugging Face.
- Guardar JSON crudo por pagina y datasets combinados en CSV/JSON.
- Enriquecer los modelos con metadata adicional util para graficos:
  popularidad, adopcion acumulada, arquitectura, licencia, idiomas,
  cantidad de archivos, cantidad de Spaces que usan el modelo, etc.

Uso:
    python crawler_huggingface_ia.py
    python crawler_huggingface_ia.py --tasks text-generation image-text-to-text
    python crawler_huggingface_ia.py --page-size 50 --pages-per-task 2
    python crawler_huggingface_ia.py --detail-model-limit 150
    python crawler_huggingface_ia.py --show-summary
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://huggingface.co/api/models"
DATA_DIR = Path("Data/huggingface_ia")
RAW_DIR = DATA_DIR / "raw"
DETAILS_RAW_DIR = RAW_DIR / "details"
LOG_PATH = DATA_DIR / "crawler.log"
MODELS_JSON_PATH = DATA_DIR / "huggingface_models_ia.json"
MODELS_CSV_PATH = DATA_DIR / "huggingface_models_ia.csv"
SUMMARY_CSV_PATH = DATA_DIR / "huggingface_task_summary.csv"
META_JSON_PATH = DATA_DIR / "huggingface_run_metadata.json"

DEFAULT_TIMEOUT = (10, 60)
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_PAGE_SIZE = 100
DEFAULT_PAGES_PER_TASK = 2
DEFAULT_DETAIL_MODEL_LIMIT = 120
MAX_RETRIES = 4

DEFAULT_TASKS = [
    "text-generation",
    "text-classification",
    "image-text-to-text",
    "text-to-image",
    "automatic-speech-recognition",
]

LIST_EXPAND_FIELDS = [
    "author",
    "baseModels",
    "pipeline_tag",
    "downloads",
    "downloadsAllTime",
    "evalResults",
    "inference",
    "inferenceProviderMapping",
    "likes",
    "createdAt",
    "lastModified",
    "library_name",
    "private",
    "gated",
    "disabled",
    "tags",
    "cardData",
    "config",
    "gguf",
    "resourceGroup",
    "transformersInfo",
    "safetensors",
    "sha",
    "siblings",
    "model-index",
    "spaces",
    "widgetData",
    "xetEnabled",
    "trendingScore",
]

MODEL_COLUMNS = [
    "id",
    "author",
    "pipeline_tag",
    "matched_task_filters",
    "downloads",
    "downloadsAllTime",
    "likes",
    "trendingScore",
    "createdAt",
    "created_year",
    "lastModified",
    "last_modified_year",
    "library_name",
    "private",
    "gated",
    "disabled",
    "modelId",
    "sha",
    "used_storage_bytes",
    "used_storage_gb",
    "inference_status",
    "inference_provider_count",
    "inference_providers",
    "eval_result_count",
    "gguf_enabled",
    "resource_group",
    "xet_enabled",
    "license_tag",
    "card_license",
    "card_base_model",
    "card_library_name",
    "base_models",
    "base_model_count",
    "card_languages",
    "card_tags",
    "tag_count",
    "tags",
    "dataset_tags",
    "arxiv_refs",
    "base_model_tags",
    "architecture",
    "model_type",
    "auto_model",
    "processor",
    "parameter_count_total",
    "parameter_count_billions",
    "parameter_types",
    "repo_file_count",
    "space_count",
    "space_examples",
    "widget_example_count",
    "model_index_entry_count",
    "has_readme",
    "has_safetensors_file",
    "detail_fetched",
]

SUMMARY_COLUMNS = [
    "task_filter",
    "model_count",
    "avg_downloads",
    "median_downloads",
    "avg_downloads_all_time",
    "median_downloads_all_time",
    "avg_likes",
    "median_likes",
]


def setup_logging() -> logging.Logger:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DETAILS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("crawler_huggingface_ia")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


log = setup_logging()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawler de modelos de IA desde la API oficial de Hugging Face."
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Tareas a consultar. Si se omite, usa un set recomendado.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=DEFAULT_PAGE_SIZE,
        help="Cantidad de modelos por pagina.",
    )
    parser.add_argument(
        "--pages-per-task",
        type=int,
        default=DEFAULT_PAGES_PER_TASK,
        help="Cuantas paginas descargar por cada tarea.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Espera entre peticiones consecutivas.",
    )
    parser.add_argument(
        "--sort",
        default="downloads",
        help="Campo de orden para la API. Ejemplo: downloads, likes, lastModified.",
    )
    parser.add_argument(
        "--detail-model-limit",
        type=int,
        default=DEFAULT_DETAIL_MODEL_LIMIT,
        help="Maximo de modelos unicos a enriquecer con el endpoint de detalle. Usa 0 para enriquecer todos.",
    )
    parser.add_argument(
        "--skip-detail-fetch",
        action="store_true",
        help="No hace la fase de enriquecimiento por modelo.",
    )
    parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Muestra el resumen del ultimo dataset guardado.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.page_size < 1:
        raise ValueError("--page-size debe ser mayor que cero.")
    if args.pages_per_task < 1:
        raise ValueError("--pages-per-task debe ser mayor o igual a 1.")
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds no puede ser negativo.")


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "ComputacionCientifica-UACh/1.0 (crawler_huggingface_ia.py)",
            "Accept": "application/json",
        }
    )
    return session


def fetch_with_retries(
    session: requests.Session,
    url: str,
    params: dict[str, Any] | None,
    timeout: tuple[int, int],
    delay_seconds: float,
) -> requests.Response:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response

            if response.status_code in {429, 500, 502, 503, 504}:
                wait_time = max(delay_seconds, attempt * delay_seconds)
                log.warning(
                    "HTTP %s en %s. Esperando %.1fs antes de reintentar (%s/%s).",
                    response.status_code,
                    url,
                    wait_time,
                    attempt,
                    MAX_RETRIES,
                )
                time.sleep(wait_time)
                continue

            response.raise_for_status()
        except (requests.Timeout, requests.ConnectionError) as exc:
            wait_time = max(delay_seconds, attempt * delay_seconds)
            log.warning(
                "Problema de red (%s). Esperando %.1fs antes de reintentar (%s/%s).",
                exc.__class__.__name__,
                wait_time,
                attempt,
                MAX_RETRIES,
            )
            time.sleep(wait_time)

    raise RuntimeError(f"No fue posible obtener datos desde {url}.")


def parse_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None

    match = re.search(r"<([^>]+)>;\s*rel=\"next\"", link_header)
    if not match:
        return None
    return match.group(1)


def sanitize_name(value: str) -> str:
    value = value.replace("/", "_")
    value = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)
    return value.strip("_").lower()


def extract_prefixed_tags(tags: list[str], prefix: str) -> list[str]:
    return [tag[len(prefix):] for tag in tags if tag.startswith(prefix)]


def ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def join_unique(values: list[Any]) -> list[str]:
    seen: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def extract_year(date_value: Any) -> str:
    if not date_value or not isinstance(date_value, str):
        return ""
    match = re.match(r"(\d{4})-", date_value)
    if match:
        return match.group(1)
    return ""


def extract_card_data(item: dict[str, Any]) -> dict[str, Any]:
    card = item.get("cardData") or {}
    return {
        "card_license": card.get("license", ""),
        "card_base_model": "; ".join(join_unique(ensure_list(card.get("base_model")))),
        "card_library_name": card.get("library_name", ""),
        "card_languages": join_unique(ensure_list(card.get("language"))),
        "card_tags": join_unique(ensure_list(card.get("tags"))),
    }


def extract_config_data(item: dict[str, Any]) -> dict[str, Any]:
    config = item.get("config") or {}
    architectures = ensure_list(config.get("architectures"))
    return {
        "architecture": "; ".join(join_unique(architectures)),
        "model_type": config.get("model_type", ""),
    }


def extract_transformers_info(item: dict[str, Any]) -> dict[str, Any]:
    info = item.get("transformersInfo") or {}
    return {
        "auto_model": info.get("auto_model", ""),
        "processor": info.get("processor", ""),
    }


def extract_safetensors_info(item: dict[str, Any]) -> dict[str, Any]:
    data = item.get("safetensors") or {}
    parameters = data.get("parameters") or {}
    total = data.get("total")
    param_types = join_unique(list(parameters.keys()))

    parameter_count_total = total if isinstance(total, (int, float)) else ""
    parameter_count_billions = ""
    if isinstance(total, (int, float)):
        parameter_count_billions = round(total / 1_000_000_000, 3)

    return {
        "parameter_count_total": parameter_count_total,
        "parameter_count_billions": parameter_count_billions,
        "parameter_types": "; ".join(param_types),
    }


def extract_repo_metrics(item: dict[str, Any]) -> dict[str, Any]:
    siblings = item.get("siblings") or []
    sibling_files = [sib.get("rfilename", "") for sib in siblings if isinstance(sib, dict)]
    lower_files = [name.lower() for name in sibling_files]

    spaces = ensure_list(item.get("spaces"))
    widget_data = ensure_list(item.get("widgetData"))
    model_index = ensure_list(item.get("model-index"))

    return {
        "repo_file_count": len(sibling_files),
        "space_count": len(spaces),
        "space_examples": "; ".join(join_unique(spaces[:5])),
        "widget_example_count": len(widget_data),
        "model_index_entry_count": len(model_index),
        "has_readme": any(name == "readme.md" for name in lower_files),
        "has_safetensors_file": any(name.endswith(".safetensors") for name in lower_files),
    }


def flatten_base_models(value: Any) -> list[str]:
    collected: list[str] = []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        if isinstance(value.get("id"), str):
            collected.append(value["id"])
        for nested in ensure_list(value.get("models")):
            collected.extend(flatten_base_models(nested))
        return collected

    if isinstance(value, list):
        for item in value:
            collected.extend(flatten_base_models(item))
        return collected

    return collected


def extract_platform_metrics(item: dict[str, Any]) -> dict[str, Any]:
    provider_mapping = item.get("inferenceProviderMapping") or {}
    if isinstance(provider_mapping, dict):
        provider_names = join_unique(list(provider_mapping.keys()))
    else:
        provider_names = []

    base_models = join_unique(flatten_base_models(item.get("baseModels")))
    if not base_models:
        base_models = join_unique(ensure_list((item.get("cardData") or {}).get("base_model")))
    used_storage = item.get("usedStorage")
    used_storage_bytes = used_storage if isinstance(used_storage, (int, float)) else ""
    used_storage_gb = ""
    if isinstance(used_storage, (int, float)):
        used_storage_gb = round(used_storage / 1_000_000_000, 3)

    eval_results = item.get("evalResults")
    eval_result_count = 0
    if isinstance(eval_results, list):
        eval_result_count = len(eval_results)
    elif isinstance(eval_results, dict):
        eval_result_count = len(eval_results)

    gguf_value = item.get("gguf")
    gguf_enabled = ""
    if gguf_value is not None:
        gguf_enabled = bool(gguf_value)

    return {
        "sha": item.get("sha", ""),
        "used_storage_bytes": used_storage_bytes,
        "used_storage_gb": used_storage_gb,
        "inference_status": item.get("inference", ""),
        "inference_provider_count": len(provider_names),
        "inference_providers": "; ".join(provider_names),
        "eval_result_count": eval_result_count,
        "gguf_enabled": gguf_enabled,
        "resource_group": item.get("resourceGroup", ""),
        "xet_enabled": item.get("xetEnabled", ""),
        "base_models": "; ".join(base_models),
        "base_model_count": len(base_models),
    }


def normalize_model(item: dict[str, Any], task_filter: str) -> dict[str, Any]:
    tags = item.get("tags", []) or []

    model = {
        "id": item.get("id"),
        "author": item.get("author"),
        "pipeline_tag": item.get("pipeline_tag"),
        "matched_task_filters": [task_filter],
        "downloads": item.get("downloads"),
        "downloadsAllTime": item.get("downloadsAllTime"),
        "likes": item.get("likes"),
        "trendingScore": item.get("trendingScore"),
        "createdAt": item.get("createdAt"),
        "created_year": extract_year(item.get("createdAt")),
        "lastModified": item.get("lastModified"),
        "last_modified_year": extract_year(item.get("lastModified")),
        "library_name": item.get("library_name"),
        "private": item.get("private"),
        "gated": item.get("gated"),
        "disabled": item.get("disabled"),
        "modelId": item.get("modelId"),
        "license_tag": next((tag.split(":", 1)[1] for tag in tags if tag.startswith("license:")), ""),
        "tag_count": len(tags),
        "tags": tags,
        "dataset_tags": extract_prefixed_tags(tags, "dataset:"),
        "arxiv_refs": extract_prefixed_tags(tags, "arxiv:"),
        "base_model_tags": extract_prefixed_tags(tags, "base_model:"),
        "detail_fetched": False,
    }

    model.update(extract_card_data(item))
    model.update(extract_config_data(item))
    model.update(extract_transformers_info(item))
    model.update(extract_safetensors_info(item))
    model.update(extract_repo_metrics(item))
    model.update(extract_platform_metrics(item))
    return model


def merge_list_field(existing: dict[str, Any], incoming: dict[str, Any], field_name: str) -> list[str]:
    merged = list(existing.get(field_name, []))
    for value in incoming.get(field_name, []):
        if value not in merged:
            merged.append(value)
    return merged


def merge_models(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)

    matched = list(existing.get("matched_task_filters", []))
    for task in incoming.get("matched_task_filters", []):
        if task not in matched:
            matched.append(task)
    merged["matched_task_filters"] = matched

    scalar_fields = [
        "author",
        "pipeline_tag",
        "downloads",
        "downloadsAllTime",
        "likes",
        "trendingScore",
        "createdAt",
        "created_year",
        "lastModified",
        "last_modified_year",
        "library_name",
        "private",
        "gated",
        "disabled",
        "modelId",
        "sha",
        "used_storage_bytes",
        "used_storage_gb",
        "inference_status",
        "inference_provider_count",
        "inference_providers",
        "eval_result_count",
        "gguf_enabled",
        "resource_group",
        "xet_enabled",
        "license_tag",
        "card_license",
        "card_base_model",
        "card_library_name",
        "base_models",
        "base_model_count",
        "architecture",
        "model_type",
        "auto_model",
        "processor",
        "parameter_count_total",
        "parameter_count_billions",
        "parameter_types",
        "repo_file_count",
        "space_count",
        "space_examples",
        "widget_example_count",
        "model_index_entry_count",
        "has_readme",
        "has_safetensors_file",
        "detail_fetched",
    ]

    for field in scalar_fields:
        value = incoming.get(field)
        if value not in (None, "", []):
            merged[field] = value

    for list_field in [
        "tags",
        "dataset_tags",
        "arxiv_refs",
        "base_model_tags",
        "card_languages",
        "card_tags",
    ]:
        merged[list_field] = merge_list_field(existing, incoming, list_field)

    if incoming.get("tag_count") not in (None, ""):
        merged["tag_count"] = max(int(existing.get("tag_count", 0)), int(incoming.get("tag_count", 0)))

    return merged


def save_raw_page(task_filter: str, page_number: int, payload: Any) -> str:
    file_name = f"{sanitize_name(task_filter)}_page_{page_number:03d}.json"
    path = RAW_DIR / file_name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_name


def save_raw_detail(model_id: str, payload: Any) -> str:
    file_name = f"{sanitize_name(model_id)}.json"
    path = DETAILS_RAW_DIR / file_name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return file_name


def save_models_json(models: list[dict[str, Any]]) -> None:
    payload = []
    for model in models:
        item = dict(model)
        item["matched_task_filters"] = "; ".join(model.get("matched_task_filters", []))
        payload.append(item)

    MODELS_JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_models_csv(models: list[dict[str, Any]]) -> None:
    with MODELS_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=MODEL_COLUMNS)
        writer.writeheader()

        for model in models:
            row = dict(model)
            row["matched_task_filters"] = "; ".join(model.get("matched_task_filters", []))
            row["tags"] = "; ".join(model.get("tags", []))
            row["dataset_tags"] = "; ".join(model.get("dataset_tags", []))
            row["arxiv_refs"] = "; ".join(model.get("arxiv_refs", []))
            row["base_model_tags"] = "; ".join(model.get("base_model_tags", []))
            row["card_languages"] = "; ".join(model.get("card_languages", []))
            row["card_tags"] = "; ".join(model.get("card_tags", []))
            writer.writerow({column: row.get(column, "") for column in MODEL_COLUMNS})


def build_task_summary(models: list[dict[str, Any]], tasks: list[str]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []

    for task in tasks:
        task_models = [model for model in models if task in model.get("matched_task_filters", [])]
        downloads = [model["downloads"] for model in task_models if isinstance(model.get("downloads"), (int, float))]
        downloads_all_time = [
            model["downloadsAllTime"]
            for model in task_models
            if isinstance(model.get("downloadsAllTime"), (int, float))
        ]
        likes = [model["likes"] for model in task_models if isinstance(model.get("likes"), (int, float))]

        summary_rows.append(
            {
                "task_filter": task,
                "model_count": len(task_models),
                "avg_downloads": round(statistics.mean(downloads), 2) if downloads else "",
                "median_downloads": round(statistics.median(downloads), 2) if downloads else "",
                "avg_downloads_all_time": round(statistics.mean(downloads_all_time), 2) if downloads_all_time else "",
                "median_downloads_all_time": round(statistics.median(downloads_all_time), 2) if downloads_all_time else "",
                "avg_likes": round(statistics.mean(likes), 2) if likes else "",
                "median_likes": round(statistics.median(likes), 2) if likes else "",
            }
        )

    return summary_rows


def save_summary_csv(summary_rows: list[dict[str, Any]]) -> None:
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def save_metadata(metadata: dict[str, Any]) -> None:
    META_JSON_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_metadata() -> dict[str, Any] | None:
    if not META_JSON_PATH.exists():
        return None
    return json.loads(META_JSON_PATH.read_text(encoding="utf-8"))


def print_summary() -> None:
    metadata = load_metadata()
    if not metadata:
        log.info("No hay metadata guardada aun en %s", META_JSON_PATH.resolve())
        return

    log.info("")
    log.info("=" * 72)
    log.info("RESUMEN DEL CRAWLER HUGGING FACE IA")
    log.info("=" * 72)
    log.info("Fecha de ejecucion: %s", metadata.get("run_at"))
    log.info("Tareas: %s", ", ".join(metadata.get("tasks", [])))
    log.info("Orden: %s", metadata.get("sort"))
    log.info("Modelos unicos: %s", metadata.get("model_count"))
    log.info("Modelos enriquecidos: %s", metadata.get("detail_models_fetched"))
    log.info("CSV modelos: %s", MODELS_CSV_PATH.resolve())
    log.info("JSON modelos: %s", MODELS_JSON_PATH.resolve())
    log.info("Resumen por tarea: %s", SUMMARY_CSV_PATH.resolve())
    log.info("Raw paginas: %s", RAW_DIR.resolve())
    log.info("Raw detalles: %s", DETAILS_RAW_DIR.resolve())
    log.info("")

    for task, info in metadata.get("task_runs", {}).items():
        log.info(
            "[%s] paginas=%s | modelos vistos=%s | raw=%s",
            task,
            info.get("pages"),
            info.get("models_seen"),
            ", ".join(info.get("raw_files", [])),
        )


def crawl_task(
    session: requests.Session,
    task_filter: str,
    page_size: int,
    pages_per_task: int,
    sort: str,
    delay_seconds: float,
    models_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    params = {
        "limit": page_size,
        "sort": sort,
        "direction": -1,
        "filter": task_filter,
        "expand[]": LIST_EXPAND_FIELDS,
    }

    next_url = BASE_URL
    next_params: dict[str, Any] | None = params
    page_number = 0
    raw_files: list[str] = []
    models_seen = 0

    while page_number < pages_per_task and next_url:
        page_number += 1

        response = fetch_with_retries(
            session=session,
            url=next_url,
            params=next_params,
            timeout=DEFAULT_TIMEOUT,
            delay_seconds=delay_seconds,
        )

        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError(f"Respuesta inesperada para la tarea {task_filter}.")

        raw_file = save_raw_page(task_filter, page_number, payload)
        raw_files.append(raw_file)

        log.info(
            "[%s] pagina %s | modelos recibidos: %s",
            task_filter,
            page_number,
            len(payload),
        )

        for item in payload:
            normalized = normalize_model(item, task_filter=task_filter)
            model_id = normalized["id"]
            if not model_id:
                continue

            if model_id in models_by_id:
                models_by_id[model_id] = merge_models(models_by_id[model_id], normalized)
            else:
                models_by_id[model_id] = normalized
            models_seen += 1

        next_url = parse_next_link(response.headers.get("Link"))
        next_params = None

        if page_number < pages_per_task and next_url:
            log.info("[%s] esperando %.1fs antes de la siguiente pagina.", task_filter, delay_seconds)
            time.sleep(delay_seconds)

    return {
        "pages": page_number,
        "models_seen": models_seen,
        "raw_files": raw_files,
    }


def fetch_model_details(
    session: requests.Session,
    model_id: str,
    delay_seconds: float,
) -> tuple[dict[str, Any], str]:
    url = f"{BASE_URL}/{model_id}"
    response = fetch_with_retries(
        session=session,
        url=url,
        params=None,
        timeout=DEFAULT_TIMEOUT,
        delay_seconds=delay_seconds,
    )

    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Respuesta inesperada en detalle para {model_id}.")

    raw_file = save_raw_detail(model_id, payload)
    detail_model = normalize_model(payload, task_filter="")
    detail_model["matched_task_filters"] = []
    detail_model["detail_fetched"] = True
    return detail_model, raw_file


def enrich_models_with_details(
    session: requests.Session,
    models_by_id: dict[str, dict[str, Any]],
    detail_model_limit: int,
    delay_seconds: float,
) -> tuple[int, list[str]]:
    models = list(models_by_id.values())
    models = sorted(
        models,
        key=lambda item: (
            item.get("downloads") if isinstance(item.get("downloads"), (int, float)) else -1,
            item.get("likes") if isinstance(item.get("likes"), (int, float)) else -1,
            item.get("id") or "",
        ),
        reverse=True,
    )

    if detail_model_limit > 0:
        models = models[:detail_model_limit]

    raw_detail_files: list[str] = []

    for index, model in enumerate(models, start=1):
        model_id = model.get("id")
        if not model_id:
            continue

        detail_model, raw_file = fetch_model_details(
            session=session,
            model_id=model_id,
            delay_seconds=delay_seconds,
        )
        detail_model["matched_task_filters"] = list(models_by_id[model_id].get("matched_task_filters", []))
        models_by_id[model_id] = merge_models(models_by_id[model_id], detail_model)
        raw_detail_files.append(raw_file)

        log.info(
            "[detalle %s/%s] %s | downloadsAllTime=%s | files=%s | spaces=%s",
            index,
            len(models),
            model_id,
            models_by_id[model_id].get("downloadsAllTime"),
            models_by_id[model_id].get("repo_file_count"),
            models_by_id[model_id].get("space_count"),
        )

        if index < len(models):
            time.sleep(delay_seconds)

    return len(models), raw_detail_files


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.show_summary:
        print_summary()
        return

    tasks = args.tasks or DEFAULT_TASKS
    session = build_session()
    models_by_id: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] = {
        "run_at": datetime.now().isoformat(),
        "tasks": tasks,
        "page_size": args.page_size,
        "pages_per_task": args.pages_per_task,
        "delay_seconds": args.delay_seconds,
        "sort": args.sort,
        "detail_model_limit": args.detail_model_limit,
        "task_runs": {},
    }

    log.info("=" * 72)
    log.info("CRAWLER DE MODELOS DE IA - HUGGING FACE API")
    log.info("=" * 72)
    log.info("Tareas activas: %s", ", ".join(tasks))
    log.info("Page size: %s | Pages per task: %s", args.page_size, args.pages_per_task)
    log.info("Orden: %s desc", args.sort)
    log.info("Espera entre peticiones: %.1fs", args.delay_seconds)
    log.info("Limite de enriquecimiento por detalle: %s", args.detail_model_limit)

    for index, task_filter in enumerate(tasks):
        log.info("-" * 72)
        log.info("[%s] iniciando descarga...", task_filter)

        task_info = crawl_task(
            session=session,
            task_filter=task_filter,
            page_size=args.page_size,
            pages_per_task=args.pages_per_task,
            sort=args.sort,
            delay_seconds=args.delay_seconds,
            models_by_id=models_by_id,
        )
        metadata["task_runs"][task_filter] = task_info

        if index < len(tasks) - 1:
            log.info("Esperando %.1fs antes de la siguiente tarea.", args.delay_seconds)
            time.sleep(args.delay_seconds)

    detail_models_fetched = 0
    detail_raw_files: list[str] = []
    if not args.skip_detail_fetch:
        log.info("-" * 72)
        log.info("Iniciando enriquecimiento por modelo...")
        detail_models_fetched, detail_raw_files = enrich_models_with_details(
            session=session,
            models_by_id=models_by_id,
            detail_model_limit=args.detail_model_limit,
            delay_seconds=args.delay_seconds,
        )

    models = sorted(
        models_by_id.values(),
        key=lambda item: (
            item.get("downloads") if isinstance(item.get("downloads"), (int, float)) else -1,
            item.get("likes") if isinstance(item.get("likes"), (int, float)) else -1,
            item.get("id") or "",
        ),
        reverse=True,
    )

    save_models_json(models)
    save_models_csv(models)

    summary_rows = build_task_summary(models, tasks=tasks)
    save_summary_csv(summary_rows)

    metadata["model_count"] = len(models)
    metadata["detail_models_fetched"] = detail_models_fetched
    metadata["detail_raw_files"] = detail_raw_files
    save_metadata(metadata)
    print_summary()


if __name__ == "__main__":
    main()
