# clinic_index.py
from typing import Dict, Any, Optional, List
import logging
import re

logger = logging.getLogger(__name__)


def normalize_text(text: Optional[str], keep_spaces: bool = False) -> str:
    """
    Приводит строку к нижнему регистру, удаляет дефисы и опционально пробелы.
    Безопасно обрабатывает None, возвращая пустую строку.
    """
    if not text:
        return ""
    normalized = text.lower().replace("-", "")
    if keep_spaces:
        normalized = re.sub(r'\s+', ' ', normalized).strip()
    else:
        normalized = normalized.replace(" ", "")
    return normalized


TENANT_INDEXES: Dict[str, Dict[str, Dict[str, str]]] = {}

ENTITY_KEYS = [
    ("serviceName", "serviceId", True),
    ("employeeFullName", "employeeId", True),
    ("filialName", "filialId", False),
    ("categoryName", "categoryId", True),
]

def build_indexes_for_tenant(tenant_id: str, raw_data: List[Dict[str, Any]]):
    """
    Строит индексы имя <-> id для всех сущностей по сырым данным тенанта.
    Использует normalize_text.
    """
    if not raw_data:
        logger.warning(f"Нет данных для построения индексов для тенанта {tenant_id}")
        return
    indexes = {}
    for name_key, id_key, keep_spaces_flag in ENTITY_KEYS:
        name_to_id = {}
        id_to_name = {}
        for item in raw_data:
            name = item.get(name_key)
            id_ = item.get(id_key)
            if name and id_:
                normalized_name = normalize_text(name, keep_spaces=keep_spaces_flag)
                if normalized_name not in name_to_id:
                    name_to_id[normalized_name] = id_
                if id_ not in id_to_name:
                    id_to_name[id_] = name
        indexes[f"{name_key}_to_id"] = name_to_id
        indexes[f"{id_key}_to_name"] = id_to_name
    TENANT_INDEXES[tenant_id] = indexes
    logger.info(f"Построены индексы для тенанта {tenant_id} с использованием normalize_text.")

def get_id_by_name(tenant_id: str, entity: str, name: str) -> Optional[str]:
    """
    Получить id по имени для сущности (service, employee, filial, category) и tenant_id.
    Использует normalize_text.
    entity: 'service', 'employee', 'filial', 'category'
    """
    keep_spaces_for_entity = False
    if entity == "service" or entity == "employee" or entity == "category":
        keep_spaces_for_entity = True
    
    normalized_name_to_search = normalize_text(name, keep_spaces=keep_spaces_for_entity)
    
    name_key_for_index = ""
    if entity == "service": name_key_for_index = "serviceName"
    elif entity == "employee": name_key_for_index = "employeeFullName"
    elif entity == "filial": name_key_for_index = "filialName"
    elif entity == "category": name_key_for_index = "categoryName"
    else:
        logger.error(f"Неизвестный тип сущности '{entity}' при поиске ID по имени для тенанта {tenant_id}")
        return None

    index_map_key = f"{name_key_for_index}_to_id"
    
    found_id = TENANT_INDEXES.get(tenant_id, {}).get(index_map_key, {}).get(normalized_name_to_search)
    if not found_id:
        logger.warning(f"ID не найден для тенанта '{tenant_id}', сущности '{entity}', нормализованного имени '{normalized_name_to_search}' (исходное: '{name}'). Проверьте ENTITY_KEYS и keep_spaces флаги.")
    return found_id

def get_name_by_id(tenant_id: str, entity: str, id_: str) -> Optional[str]:
    id_key_for_index = ""
    if entity == "service": id_key_for_index = "serviceId"
    elif entity == "employee": id_key_for_index = "employeeId"
    elif entity == "filial": id_key_for_index = "filialId"
    elif entity == "category": id_key_for_index = "categoryId"
    else:
        logger.error(f"Неизвестный тип сущности '{entity}' при поиске имени по ID для тенанта {tenant_id}")
        return None
        
    index_map_key = f"{id_key_for_index}_to_name"
    return TENANT_INDEXES.get(tenant_id, {}).get(index_map_key, {}).get(id_)


