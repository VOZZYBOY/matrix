
# tenant_chain_data.py

"""Utility functions for tenant/chain-specific JSON data storage.

This module is responsible for:
1. Building the on-disk path where JSON data for a particular tenant/chain lives.
2. Ensuring the parent directory exists.
3. Downloading the data from MatrixCRM API with pagination when the file is missing.
4. Returning the absolute path to the cached file so that other modules (e.g. rag_setup) can load it.

The API endpoint being called returns paginated data.  Pagination parameters are assumed to be
`Page` (1-based) and `PageSize`.  The defaults used here work with current backend implementation;
update them if the backend changes.

Environment variables that influence behaviour:
    MATRIXCRM_BASE_URL   – base URL of backend (default: "https://back.matrixcrm.ru")
    MATRIXCRM_PAGE_SIZE  – items requested per page (default: 200)
    MATRIXCRM_TIMEOUT    – request timeout seconds (default: 30)

If an API token is required, pass it into `download_chain_data` – it will be sent as
an `Authorization: Bearer <token>` header.
"""

from __future__ import annotations

import os
import json
import logging
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

BASE_STORAGE_DIR = "base"  
DEFAULT_PAGE_SIZE = int(os.getenv("MATRIXCRM_PAGE_SIZE", "200"))
DEFAULT_TIMEOUT = int(os.getenv("MATRIXCRM_TIMEOUT", "600"))  # 10 minutes default
BASE_URL = os.getenv("MATRIXCRM_BASE_URL", "https://back.matrixcrm.ru")
_ENDPOINT = f"{BASE_URL.rstrip('/')}/api/v1/AI/servicesByFilters"

def _ensure_tenant_dir(tenant_id: str) -> str:
    tenant_dir = os.path.join(BASE_STORAGE_DIR, tenant_id)
    if not os.path.isdir(tenant_dir):
        os.makedirs(tenant_dir, exist_ok=True)
        logger.info("[Tenant %s] Создана директория %s", tenant_id, tenant_dir)
    return tenant_dir


def build_file_path(tenant_id: str, chain_id: str) -> str:
    """Return the expected JSON file path for given tenant/chain."""
    tenant_dir = _ensure_tenant_dir(tenant_id)
    return os.path.join(tenant_dir, f"{tenant_id}_{chain_id}.json")


def _request_page(chain_id: str, page: int, page_size: int, api_token: Optional[str]) -> List[dict]:
    """Request a single page; returns list (possibly empty) of items."""
    params = {
        "ChainId": chain_id,
        "Page": page,
        "PageSize": page_size,
    }
    headers = {"Accept": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    try:
        r = requests.get(_ENDPOINT, params=params, headers=headers, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            if "items" in data:
                return data["items"]  # ignore hasMore, loop will discover emptiness
            # Other common wrappers
            if "data" in data:
                return data["data"]
            if "services" in data:
                return data["services"]
            # If dict but not wrapped list → treat as single item
            return [data]
        if isinstance(data, list):
            return data
        logger.warning("Неожиданный формат ответа API: %s", type(data))
        return []
    except Exception as exc:
        logger.error("Ошибка запроса страницы %s API MatrixCRM: %s", page, exc, exc_info=True)
        return []


def download_chain_data(tenant_id: str, chain_id: str, *, api_token: Optional[str] = None, force: bool = False) -> str:
    """Download data for tenant/chain if missing (or `force`), returns path to JSON file.

    The full dataset from all pages is stored as a single JSON array.
    """
    file_path = build_file_path(tenant_id, chain_id)

    if os.path.isfile(file_path) and not force:
        logger.info("[Tenant %s / Chain %s] Используем кэшированные данные (%s)", tenant_id, chain_id, file_path)
        return file_path

    logger.info("[Tenant %s / Chain %s] Скачиваем данные с пагинацией...", tenant_id, chain_id)
    all_items: List[dict] = []
    page = 1
    page_size = DEFAULT_PAGE_SIZE

    while True:
        items = _request_page(chain_id, page, page_size, api_token)
        if items:
            logger.info("[Tenant %s / Chain %s] Загружена страница %s, элементов: %s", tenant_id, chain_id, page, len(items))
        if not items:
            break
        all_items.extend(items)
        logger.debug(
            "Получена страница %s (%s элементов) для chain_id=%s", page, len(items), chain_id
        )
        if len(items) < page_size:
            break
        page += 1

    if not all_items:
        logger.warning("[Tenant %s / Chain %s] Получен пустой список услуг. Файл не будет создан.", tenant_id, chain_id)
        return file_path  

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_items, f, ensure_ascii=False, indent=2)
        logger.info("[Tenant %s / Chain %s] Сохранено %s записей в %s", tenant_id, chain_id, len(all_items), file_path)
    except Exception as exc:
        logger.error("Не удалось записать файл %s: %s", file_path, exc, exc_info=True)

    return file_path


def discover_all_json_files(base_dir: str = BASE_STORAGE_DIR) -> List[str]:
    """Recursively collect all *.json files inside the base storage directory."""
    json_files: List[str] = []
    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.lower().endswith(".json"):
                json_files.append(os.path.join(root, fname))
    return json_files
