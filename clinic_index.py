# clinic_index.py
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


TENANT_INDEXES: Dict[str, Dict[str, Dict[str, str]]] = {}

ENTITY_KEYS = [
    ("serviceName", "serviceId"),
    ("employeeFullName", "employeeId"),
    ("filialName", "filialId"),
    ("categoryName", "categoryId"),
]

def build_indexes_for_tenant(tenant_id: str, raw_data: List[Dict[str, Any]]):
    """
    Строит индексы имя <-> id для всех сущностей по сырым данным тенанта.
    """
    if not raw_data:
        logger.warning(f"Нет данных для построения индексов для тенанта {tenant_id}")
        return
    indexes = {}
    for name_key, id_key in ENTITY_KEYS:
        name_to_id = {}
        id_to_name = {}
        for item in raw_data:
            name = item.get(name_key)
            id_ = item.get(id_key)
            if name and id_:
                name_to_id[name.strip()] = id_
                id_to_name[id_] = name.strip()
        indexes[f"{name_key}_to_id"] = name_to_id
        indexes[f"{id_key}_to_name"] = id_to_name
    TENANT_INDEXES[tenant_id] = indexes
    logger.info(f"Построены индексы для тенанта {tenant_id}: {list(indexes.keys())}")

def get_id_by_name(tenant_id: str, entity: str, name: str) -> Optional[str]:
    """
    Получить id по имени для сущности (service, employee, filial, category) и tenant_id.
    entity: 'service', 'employee', 'filial', 'category'
    """
    key = f"{entity}Name_to_id"
    return TENANT_INDEXES.get(tenant_id, {}).get(key, {}).get(name.strip())

def get_name_by_id(tenant_id: str, entity: str, id_: str) -> Optional[str]:
    key = f"{entity}Id_to_name"
    return TENANT_INDEXES.get(tenant_id, {}).get(key, {}).get(id_)


