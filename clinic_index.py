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
    Сначала ищет точное совпадение, затем частичное (если имя из запроса является подстрокой полного имени).
    entity: 'service', 'employee', 'filial', 'category'
    """
    keep_spaces_for_entity = False
    # Для услуг, сотрудников и категорий сохраняем пробелы при нормализации, т.к. они могут быть частью названия
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
    name_to_id_map = TENANT_INDEXES.get(tenant_id, {}).get(index_map_key, {})

    if not name_to_id_map:
        logger.warning(f"Карта '{index_map_key}' не найдена или пуста для тенанта '{tenant_id}'.")
        return None

    # 1. Поиск точного совпадения
    exact_match_id = name_to_id_map.get(normalized_name_to_search)
    if exact_match_id:
        logger.info(f"Найдено точное совпадение ID ('{exact_match_id}') для тенанта '{tenant_id}', сущности '{entity}', нормализованного имени '{normalized_name_to_search}' (исходное: '{name}').")
        return exact_match_id

    # 2. Поиск частичного совпадения (если normalized_name_to_search является подстрокой ключа в карте)
    #    Это полезно, если пользователь ввел "Соня Сеферова", а в базе "Соня Сеферова Магамедовна"
    #    Или для услуг, например, ввел "Лазерная эпиляция", а в базе "Лазерная эпиляция бикини".
    partial_matches = []
    for indexed_norm_name, item_id in name_to_id_map.items():
        if normalized_name_to_search in indexed_norm_name:
            partial_matches.append(item_id)
            logger.debug(f"Найдено частичное совпадение: запрос '{normalized_name_to_search}' содержится в '{indexed_norm_name}' (ID: {item_id})")

    if len(partial_matches) == 1:
        logger.info(f"Найдено ОДНО частичное совпадение ID ('{partial_matches[0]}') для тенанта '{tenant_id}', сущности '{entity}', по запросу '{normalized_name_to_search}' (исходное: '{name}').")
        return partial_matches[0]
    
    if len(partial_matches) > 1:
        # Если частичных совпадений несколько, это неоднозначность.
        # Пока возвращаем None и логируем, чтобы избежать неправильного выбора.
        # В будущем можно вернуть список или специальный флаг.
        original_names_found = []
        id_to_name_map_key = f"{ENTITY_KEYS[[e[0] for e in ENTITY_KEYS].index(name_key_for_index)][1]}_to_name"
        id_to_name_map = TENANT_INDEXES.get(tenant_id, {}).get(id_to_name_map_key, {})
        for pid in partial_matches:
            original_names_found.append(id_to_name_map.get(pid, f"(ID: {pid})"))
        
        logger.warning(f"Найдено НЕСКОЛЬКО ({len(partial_matches)}) частичных совпадений для тенанта '{tenant_id}', сущности '{entity}', по запросу '{normalized_name_to_search}' (исходное: '{name}'). Совпадения: {original_names_found}. Возвращаем None из-за неоднозначности.")
        return None # Неоднозначность

    logger.warning(f"ID не найден (ни точное, ни частичное совпадение) для тенанта '{tenant_id}', сущности '{entity}', нормализованного имени '{normalized_name_to_search}' (исходное: '{name}').")
    return None

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


