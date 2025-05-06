# tenant_config_manager.py

import os
import json
import logging
from typing import Dict, Any, Optional, List
# Добавляем импорт Document, если будем его использовать напрямую
# Или просто ожидаем словари с нужной структурой
# from langchain_core.documents import Documen

logger = logging.getLogger(__name__)
CONFIG_DIR = "tenant_configs"

try:
    os.makedirs(CONFIG_DIR, exist_ok=True)
except OSError as e:
    logger.error(f"Не удалось создать директорию для конфигураций тенантов '{CONFIG_DIR}': {e}")

def get_config_path(tenant_id: str) -> str:
    """Возвращает путь к файлу конфигурации для тенанта."""
    safe_filename = "".join(c for c in tenant_id if c.isalnum() or c in ('.', '-', '_')).rstrip()
    if not safe_filename:
        raise ValueError(f"Не удалось сгенерировать безопасное имя файла из tenant_id: {tenant_id}")
    return os.path.join(CONFIG_DIR, f"{safe_filename}.json")

def save_tenant_settings(tenant_id: str, settings: Dict[str, Any]) -> bool:
    """Сохраняет настройки тенанта в JSON файл."""
    if not tenant_id:
        logger.error("Попытка сохранить настройки для пустого tenant_id")
        return False
    if not isinstance(settings, dict):
        logger.error(f"Настройки для tenant_id '{tenant_id}' должны быть словарем, получено: {type(settings)}")
        return False

    try:
        config_path = get_config_path(tenant_id)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, ensure_ascii=False, indent=4)
        logger.info(f"Настройки для тенанта '{tenant_id}' сохранены в '{config_path}'")
        return True
    except Exception as e:
        logger.error(f"Ошибка сохранения настроек для тенанта '{tenant_id}' в файл '{config_path}': {e}", exc_info=True)
        return False

def load_tenant_settings(tenant_id: str) -> Dict[str, Any]:
    """Загружает все настройки тенанта из JSON файла."""
    if not tenant_id:
        logger.warning("Попытка загрузить настройки для пустого tenant_id")
        return {}

    try:
        config_path = get_config_path(tenant_id)
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                if not isinstance(settings, dict):
                     logger.warning(f"Файл конфигурации '{config_path}' для тенанта '{tenant_id}' не содержит JSON объект (словарь).")
                     return {}
                logger.debug(f"Настройки для тенанта '{tenant_id}' загружены из '{config_path}'")
                return settings
        else:
            logger.debug(f"Файл конфигурации для тенанта '{tenant_id}' не найден ('{config_path}'). Возвращены пустые настройки.")
            return {}
    except Exception as e:
        logger.error(f"Ошибка загрузки настроек для тенанта '{tenant_id}' из файла '{config_path}': {e}", exc_info=True)
        return {}

def get_prompt_addition(tenant_id: str) -> str:
    """Загружает только дополнение к промпту для тенанта."""
    settings = load_tenant_settings(tenant_id)
    prompt_addition = settings.get("prompt_addition", "")
    if not isinstance(prompt_addition, str):
        logger.warning(f"Значение 'prompt_addition' для tenant_id '{tenant_id}' не является строкой. Игнорируется.")
        return ""
    return prompt_addition.strip()

# --- Функции для clinic_info_docs --- #

def load_tenant_clinic_info(tenant_id: str) -> List[Dict[str, Any]]:
    """Загружает список документов clinic_info для тенанта."""
    settings = load_tenant_settings(tenant_id)
    clinic_info_docs_data = settings.get("clinic_info_docs", [])

    if not isinstance(clinic_info_docs_data, list):
        logger.warning(f"Значение 'clinic_info_docs' для tenant_id '{tenant_id}' не является списком. Возвращен пустой список.")
        return []

    # Простая валидация структуры документов (опционально)
    valid_docs = []
    for i, doc_data in enumerate(clinic_info_docs_data):
        if isinstance(doc_data, dict) and \
           isinstance(doc_data.get("page_content"), str) and \
           isinstance(doc_data.get("metadata"), dict):
            valid_docs.append(doc_data)
        else:
            logger.warning(f"Некорректная структура документа #{i+1} в clinic_info_docs для tenant_id '{tenant_id}'. Пропуск.")

    logger.debug(f"Загружено {len(valid_docs)} документов clinic_info для тенанта '{tenant_id}'.")
    return valid_docs

def save_tenant_clinic_info(tenant_id: str, clinic_info_docs: List[Dict[str, Any]]) -> bool:
    """Сохраняет или перезаписывает список документов clinic_info для тенанта."""
    if not isinstance(clinic_info_docs, list):
        logger.error(f"clinic_info_docs для tenant_id '{tenant_id}' должен быть списком словарей.")
        return False

    current_settings = load_tenant_settings(tenant_id)
    current_settings["clinic_info_docs"] = clinic_info_docs
    return save_tenant_settings(tenant_id, current_settings)

# +++ Функции для получения списка тенантов и времени модификации +++
def list_tenants() -> List[str]:
    """Возвращает список ID тенантов, найденных в директории конфигураций."""
    tenant_ids = []
    try:
        for filename in os.listdir(CONFIG_DIR):
            if filename.endswith(".json"):
                tenant_id, _ = os.path.splitext(filename)
                # Дополнительная проверка, что имя файла не пустое после удаления расширения
                if tenant_id:
                     tenant_ids.append(tenant_id)
    except FileNotFoundError:
        logger.warning(f"Директория конфигураций '{CONFIG_DIR}' не найдена.")
    except Exception as e:
        logger.error(f"Ошибка при сканировании директории '{CONFIG_DIR}': {e}", exc_info=True)
    return tenant_ids

def get_settings_file_mtime(tenant_id: str) -> Optional[float]:
    """Возвращает время последней модификации файла настроек (timestamp)."""
    if not tenant_id:
        return None
    try:
        config_path = get_config_path(tenant_id)
        if os.path.exists(config_path):
            return os.path.getmtime(config_path)
    except Exception as e:
        logger.error(f"Ошибка получения времени модификации для '{config_path}': {e}", exc_info=True)
    return None

def get_clinic_info_file_mtime(tenant_id: str) -> Optional[float]:
    """
    Возвращает время последней модификации файла настроек, так как clinic_info
    хранится в том же файле.
    """
    # Так как clinic_info хранится в том же файле, что и остальные настройки,
    # просто возвращаем время модификации основного файла настроек.
    return get_settings_file_mtime(tenant_id) 