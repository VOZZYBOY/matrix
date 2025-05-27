# clinic_index.py
from typing import Dict, Any, Optional, List, Tuple
import logging
import re

logger = logging.getLogger(__name__)


def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def normalize_employee_name(name: str) -> str:
    """
    Специальная нормализация для ФИО сотрудников.
    Обрабатывает различные форматы ФИО (фамилия-имя-отчество, имя-фамилия-отчество).
    """
    if not name:
        return ""
    
    # Базовая нормализация
    normalized = name.lower().replace("-", "")
    
    # Убираем множественные пробелы и разбиваем на слова
    words = re.sub(r'\s+', ' ', normalized).strip().split()
    
    # Если меньше 2 слов, возвращаем как есть
    if len(words) < 2:
        return " ".join(words)
    
    # Сортируем слова для унификации порядка
    # Это позволит найти "Соня Сеферова Магамедовна" по запросу "Сеферова Соня Магамедовна"
    words.sort()
    
    return " ".join(words)


def normalize_text(text: Optional[str], keep_spaces: bool = False, sort_words: bool = False) -> str:
    """
    Приводит строку к нижнему регистру, удаляет дефисы и опционально пробелы.
    Опционально сортирует слова в строке.
    Безопасно обрабатывает None, возвращая пустую строку.
    """
    if not text:
        return ""
    # Сначала базовая нормализация (нижний регистр, замена дефисов)
    normalized = text.lower().replace("-", "")
    
    # Обработка пробелов
    if keep_spaces or sort_words: # Если нужно сортировать слова, пробелы между ними важны на этом этапе
        normalized = re.sub(r'\s+', ' ', normalized).strip() # Заменяем множество пробелов на один, убираем по краям
    else:
        normalized = re.sub(r'\s+', '', normalized) # Удаляем все пробелы

    # Сортировка слов, если флаг установлен
    if sort_words:
        words = normalized.split()
        words.sort()
        final_normalized = " ".join(words)
        if not keep_spaces: # Если изначально не просили сохранять пробелы, но сортировали (что требует их временного сохранения)
            final_normalized = re.sub(r'\s+', '', final_normalized) # Теперь удаляем пробелы после сортировки
        return final_normalized
    
    return normalized


TENANT_INDEXES: Dict[str, Dict[str, Dict[str, str]]] = {}

ENTITY_KEYS = [
    # (name_key, id_key, keep_spaces_flag, sort_words_flag)
    ("serviceName", "serviceId", True, False),
    ("employeeFullName", "employeeId", True, True),  # <--- sort_words=True для имен сотрудников
    ("filialName", "filialId", True, False),  # <--- Исправлено! Теперь keep_spaces=True для филиалов
    ("categoryName", "categoryId", True, False),
]

# --- Индекс соответствия serviceId -> categoryId для быстрого поиска ---
SERVICEID_TO_CATEGORYID_INDEX: Dict[str, Dict[str, str]] = {}


def build_indexes_for_tenant(tenant_id: str, raw_data: List[Dict[str, Any]]):
    """
    Строит индексы имя <-> id для всех сущностей по сырым данным тенанта.
    Использует normalize_text.
    Дополнительно строит индекс serviceId -> categoryId для быстрого поиска категории по услуге.
    
    Новая версия поддерживает многозначный индекс name_to_ids для обработки случаев,
    когда разные ID могут иметь одинаковое нормализованное имя (например, одинаковые услуги в разных филиалах).
    """
    if not raw_data:
        logger.warning(f"Нет данных для построения индексов для тенанта {tenant_id}")
        return
    indexes = {}
    for name_key, id_key, keep_spaces_flag, sort_words_flag in ENTITY_KEYS:
        # Обычный индекс для обратной совместимости
        name_to_id = {}
        # Многозначный индекс (новый)
        name_to_ids = {}
        id_to_name = {}
        for item in raw_data:
            name = item.get(name_key)
            id_ = item.get(id_key)
            if name and id_:
                normalized_name = normalize_text(name, keep_spaces=keep_spaces_flag, sort_words=sort_words_flag)
                
                # Добавляем в многозначный индекс name_to_ids
                if normalized_name not in name_to_ids:
                    name_to_ids[normalized_name] = [id_]
                elif id_ not in name_to_ids[normalized_name]:
                    name_to_ids[normalized_name].append(id_)
                    logger.info(
                        f"Tenant '{tenant_id}': Добавлен дополнительный ID '{id_}' для нормализованного имени '{normalized_name}' "
                        f"(из оригинала: '{name}') в многозначном индексе. Всего ID для этого имени: {len(name_to_ids[normalized_name])}"
                    )
                
                # Логика для обычного name_to_id (сохраняем для обратной совместимости)
                if normalized_name not in name_to_id:
                    name_to_id[normalized_name] = id_
                elif name_to_id[normalized_name] != id_: 
                     logger.warning(
                        f"Tenant '{tenant_id}': Обнаружена коллизия нормализованных имен для сущности '{name_key}'. "
                        f"Нормализованное имя '{normalized_name}' (из оригинала: '{name}') пытается сопоставиться с ID '{id_}', "
                        f"но уже сопоставлено с ID '{name_to_id[normalized_name]}' в обычном индексе. "
                        f"Сохраняется первое сопоставление (с ID '{name_to_id[normalized_name]}'), "
                        f"но все ID доступны в многозначном индексе."
                    )

                # Логика для id_to_name
                if id_ not in id_to_name:
                    id_to_name[id_] = name # Сохраняем оригинальное имя
                elif id_to_name[id_] != name: 
                    # Особое внимание для филиалов, так как это текущая проблема
                    if id_key == "filialId":
                        logger.warning(
                            f"Tenant '{tenant_id}': Обнаружено несоответствие данных для filialId '{id_}'. "
                            f"Этот ID уже сопоставлен с оригинальным именем филиала '{id_to_name[id_]}', "
                            f"но другая запись пытается сопоставить его с именем '{name}'. "
                            f"Для функции get_name_by_id будет сохранено первое сопоставленное имя ('{id_to_name[id_]}'). "
                            f"Это НАСТОЯТЕЛЬНО УКАЗЫВАЕТ на то, что в исходном файле данных один и тот же filialId используется для разных названий филиалов."
                        )
                    else: # Общее предупреждение для других сущностей
                        logger.warning(
                            f"Tenant '{tenant_id}': Дублирующийся ID '{id_}' для сущности '{name_key}'. "
                            f"Существующее оригинальное имя: '{id_to_name[id_]}', новое оригинальное имя: '{name}'. "
                            f"Сохраняется первое ('{id_to_name[id_]}'). Это может указывать на несоответствие данных."
                        )
        
        indexes[f"{name_key}_to_id"] = name_to_id  # Обычный индекс (однозначный, для обратной совместимости)
        indexes[f"{name_key}_to_ids"] = name_to_ids  # Новый многозначный индекс
        indexes[f"{id_key}_to_name"] = id_to_name
    TENANT_INDEXES[tenant_id] = indexes
    
    
    serviceid_to_categoryid = {}
    for item in raw_data:
        service_id = item.get("serviceId")
        category_id = item.get("categoryId")
        if service_id and category_id:
            serviceid_to_categoryid[service_id] = category_id
    SERVICEID_TO_CATEGORYID_INDEX[tenant_id] = serviceid_to_categoryid
    logger.info(f"Построены индексы для тенанта {tenant_id} с использованием normalize_text и индекс serviceId->categoryId. Проверьте предупреждения выше на возможные несоответствия данных.")


def get_id_by_name(tenant_id: str, entity: str, name: str) -> Optional[str]:
    """
    Получить id по имени для сущности (service, employee, filial, category) и tenant_id.
    Использует normalize_text.
    Сначала ищет точное совпадение, затем частичное (если имя из запроса является подстрокой полного имени).
    entity: 'service', 'employee', 'filial', 'category'
    """
    keep_spaces_for_entity = False
    sort_words_for_entity = False
    
    if entity == "service" or entity == "category" or entity == "filial":
        keep_spaces_for_entity = True
    elif entity == "employee":
        keep_spaces_for_entity = True
        sort_words_for_entity = True 
    
    normalized_name_to_search = normalize_text(name, keep_spaces=keep_spaces_for_entity, sort_words=sort_words_for_entity)
    
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
    partial_matches_ids = []
    # Собираем ID и нормализованные имена для последующего нечеткого поиска, если понадобится
    candidate_norm_names_for_fuzzy: List[Tuple[str, str]] = [] 

    for indexed_norm_name, item_id in name_to_id_map.items():
        candidate_norm_names_for_fuzzy.append((indexed_norm_name, item_id))
        if normalized_name_to_search in indexed_norm_name:
            partial_matches_ids.append(item_id)
            logger.debug(f"Найдено частичное совпадение (подстрока): запрос '{normalized_name_to_search}' содержится в '{indexed_norm_name}' (ID: {item_id})")

    if len(partial_matches_ids) == 1:
        logger.info(f"Найдено ОДНО частичное совпадение (подстрока) ID ('{partial_matches_ids[0]}') для тенанта '{tenant_id}', сущности '{entity}', по запросу '{normalized_name_to_search}' (исходное: '{name}').")
        return partial_matches_ids[0]
    
    if len(partial_matches_ids) > 1:
        # Если частичных совпадений несколько, это неоднозначность для данного этапа.
        # Логируем, но не возвращаем, так как нечеткий поиск может дать лучший результат.
        logger.warning(f"Найдено НЕСКОЛЬКО ({len(partial_matches_ids)}) частичных совпадений (подстрока) для тенанта '{tenant_id}', сущности '{entity}', по запросу '{normalized_name_to_search}' (исходное: '{name}'). Переходим к нечеткому поиску.")
        # Не возвращаем None здесь, а продолжаем к нечеткому поиску по всем кандидатам.

    # 3. Нечеткий поиск с использованием расстояния Левенштейна
    if candidate_norm_names_for_fuzzy:
        fuzzy_matches = []
        # Динамический порог: 1 для коротких строк (<=5 символов), иначе 2.
        # Для филиалов (entity == 'filial') всегда 1, так как их названия обычно короткие и точные.
        # Для более длинных названий услуг или сотрудников можно допустить 2 ошибки.
        threshold = 1
        if entity == 'filial':
            threshold = 1
        elif len(normalized_name_to_search) > 7: # Увеличил порог длины для большего threshold
            threshold = 2
        elif len(normalized_name_to_search) > 4: # Промежуточный порог
             threshold = 1 # Оставляем 1 для средней длины
        # Для очень коротких (<=4) останется 1 по умолчанию из threshold = 1

        logger.info(f"Нечеткий поиск для '{normalized_name_to_search}' (длина {len(normalized_name_to_search)}) с порогом {threshold}...")
        
        # ---> НАЧАЛО НОВОГО ЛОГИРОВАНИЯ <---
        if entity == "employee" and candidate_norm_names_for_fuzzy:
            sample_candidates = [cand[0] for cand in candidate_norm_names_for_fuzzy[:10]] # Берем первые 10 нормализованных имен
            logger.info(f"Пример нормализованных кандидатов для '{entity}' перед нечетким поиском ({len(candidate_norm_names_for_fuzzy)} всего): {sample_candidates}")
        # ---> КОНЕЦ НОВОГО ЛОГИРОВАНИЯ <---

        for norm_name_candidate, item_id_candidate in candidate_norm_names_for_fuzzy:
            dist = levenshtein_distance(normalized_name_to_search, norm_name_candidate)
            if dist <= threshold:
                fuzzy_matches.append({'id': item_id_candidate, 'name': norm_name_candidate, 'dist': dist})
                logger.debug(f"Кандидат для нечеткого поиска: '{norm_name_candidate}' (ID: {item_id_candidate}), расстояние: {dist}")

        if fuzzy_matches:
            # Сортируем по расстоянию, затем по длине имени (предпочитаем более короткие при равном расстоянии)
            fuzzy_matches.sort(key=lambda x: (x['dist'], len(x['name'])))
            
            # Если лучший результат имеет расстояние 0, и он один такой, это почти как точное совпадение.
            if fuzzy_matches[0]['dist'] == 0:
                # Убедимся, что он один с dist 0
                zero_dist_matches = [m for m in fuzzy_matches if m['dist'] == 0]
                if len(zero_dist_matches) == 1:
                    logger.info(f"Найдено ОДНО точное совпадение через нечеткий поиск (dist 0): ID ('{zero_dist_matches[0]['id']}') для '{normalized_name_to_search}'.")
                    return zero_dist_matches[0]['id']
                else: # Несколько с dist 0 - это странно, но возможно если нормализация дала одинаковые строки для разных ID
                    logger.warning(f"Найдено НЕСКОЛЬКО ({len(zero_dist_matches)}) совпадений с расстоянием 0 через нечеткий поиск для '{normalized_name_to_search}'. Это неоднозначность. ID: {[m['id'] for m in zero_dist_matches]}")
                    return None # Неоднозначность

            # Если есть совпадения с расстоянием > 0, но в пределах порога
            # и если первое из них (лучшее) имеет уникальное расстояние среди всех
            # или если все с минимальным расстоянием указывают на один и тот же ID (маловероятно, но для полноты)
            best_fuzzy_match = fuzzy_matches[0]
            # Проверим, есть ли другие матчи с таким же минимальным расстоянием
            all_best_dist_matches = [m for m in fuzzy_matches if m['dist'] == best_fuzzy_match['dist']]

            if len(all_best_dist_matches) == 1:
                logger.info(f"Найдено ОДНО лучшее нечеткое совпадение: ID ('{best_fuzzy_match['id']}') для '{normalized_name_to_search}' (кандидат: '{best_fuzzy_match['name']}', расстояние: {best_fuzzy_match['dist']}).")
                return best_fuzzy_match['id']
            else:
                # Если несколько совпадений с одинаковым минимальным расстоянием Левенштейна
                # Получаем оригинальные имена для этих совпадений
                id_to_name_map_key = f"{ENTITY_KEYS[[e[0] for e in ENTITY_KEYS].index(name_key_for_index)][1]}_to_name"
                id_to_name_map = TENANT_INDEXES.get(tenant_id, {}).get(id_to_name_map_key, {})
                original_names_of_ambiguous_matches = [
                    id_to_name_map.get(m['id'], f"ID:{m['id']}") for m in all_best_dist_matches
                ]
                logger.warning(f"Найдено НЕСКОЛЬКО ({len(all_best_dist_matches)}) нечетких совпадений с одинаковым лучшим расстоянием ({best_fuzzy_match['dist']}) для '{normalized_name_to_search}'. Оригинальные имена кандидатов: {original_names_of_ambiguous_matches}. Это неоднозначность.")
                return None # Неоднозначность

    # Если ничего не найдено ни одним из методов
    logger.warning(f"ID не найден (ни точное, ни частичное, ни нечеткое совпадение) для тенанта '{tenant_id}', сущности '{entity}', нормализованного имени '{normalized_name_to_search}' (исходное: '{name}').")
    return None

def get_name_by_id(tenant_id: str, entity: str, id_: str) -> Optional[str]:
    """
    Получить оригинальное имя сущности по её ID.
    """
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

def get_all_ids_by_name(tenant_id: str, entity: str, name: str) -> List[str]:
    """
    Получает все возможные ID для данного имени сущности, используя многозначный индекс.
    Полезно для случаев, когда одно и то же нормализованное имя может соответствовать
    нескольким ID (например, одинаковые услуги в разных филиалах).
    
    Args:
        tenant_id: ID тенанта
        entity: Тип сущности ('service', 'employee', 'filial', 'category')
        name: Имя сущности для поиска
        
    Returns:
        Список ID, соответствующих данному имени, или пустой список, если ничего не найдено
    """
    if not tenant_id or not entity or not name:
        logger.warning(f"Один из обязательных параметров пуст: tenant_id={tenant_id}, entity={entity}, name={name}")
        return []
    
    name_key_for_index = ""
    if entity == "service": name_key_for_index = "serviceName"
    elif entity == "employee": name_key_for_index = "employeeFullName"
    elif entity == "filial": name_key_for_index = "filialName"
    elif entity == "category": name_key_for_index = "categoryName"
    else:
        logger.error(f"Неизвестный тип сущности '{entity}' при поиске всех ID по имени для тенанта {tenant_id}")
        return []
    
    # Определение флагов для нормализации на основе типа сущности
    keep_spaces_flag = True  # По умолчанию
    sort_words_flag = False  # По умолчанию
    
    for nk, ik, ks, sw in ENTITY_KEYS:
        if nk == name_key_for_index:
            keep_spaces_flag = ks
            sort_words_flag = sw
            break
    
    normalized_name_to_search = normalize_text(name, keep_spaces=keep_spaces_flag, sort_words=sort_words_flag)
    
    # Пытаемся найти в многозначном индексе
    name_to_ids_map_key = f"{name_key_for_index}_to_ids"
    name_to_ids_map = TENANT_INDEXES.get(tenant_id, {}).get(name_to_ids_map_key, {})
    
    if normalized_name_to_search in name_to_ids_map:
        return name_to_ids_map[normalized_name_to_search]
    
    # Если не нашли по точному совпадению, можно добавить нечеткий поиск здесь
    # (по аналогии с get_id_by_name, но возвращая все подходящие ID)
    
    logger.warning(f"Не найдено ID для тенанта '{tenant_id}', сущности '{entity}', нормализованного имени '{normalized_name_to_search}' (исходное: '{name}') в многозначном индексе.")
    return []

def get_category_id_by_service_id(tenant_id: str, service_id: str) -> Optional[str]:
    """
    Быстро получить categoryId по serviceId для конкретного тенанта.
    """
    return SERVICEID_TO_CATEGORYID_INDEX.get(tenant_id, {}).get(service_id)

def get_category_id_by_service_name(tenant_id: str, service_name: str, filial_name: Optional[str] = None) -> Optional[str]:
    """
    Получить ID категории по названию услуги и опциональному филиалу.
    Сначала получает service_id, затем находит categoryId через индекс.
    """
    try:
        # Сначала получаем service_id
        service_id = get_id_by_name(tenant_id, "service", service_name)
        if not service_id:
            logger.warning(f"Не найден service_id для услуги '{service_name}' в тенанте '{tenant_id}'")
            return None
        
        # Затем получаем category_id через индекс
        category_id = get_category_id_by_service_id(tenant_id, service_id)
        if not category_id:
            logger.warning(f"Не найден category_id для service_id '{service_id}' в тенанте '{tenant_id}'")
            return None
            
        return category_id
    except Exception as e:
        logger.error(f"Ошибка при получении category_id по service_name '{service_name}': {e}")
        return None

def get_service_id_by_name(tenant_id: str, service_name: str, filial_name: Optional[str] = None) -> Optional[str]:
    """
    Получить ID услуги по её названию и опциональному филиалу.
    """
    try:
        service_id = get_id_by_name(tenant_id, "service", service_name)
        if not service_id:
            logger.warning(f"Не найден service_id для услуги '{service_name}' в тенанте '{tenant_id}'")
            return None
        return service_id
    except Exception as e:
        logger.error(f"Ошибка при получении service_id по названию '{service_name}': {e}")
        return None

def get_default_tenant_id() -> str:
    """
    Возвращает ID тенанта по умолчанию.
    В данной реализации предполагаем, что используется первый доступный тенант.
    """
    if TENANT_INDEXES:
        # Берём первый доступный tenant_id из индекса
        first_tenant = next(iter(TENANT_INDEXES.keys()), None)
        if first_tenant:
            logger.info(f"Возвращен default tenant_id: {first_tenant}")
            return first_tenant
    
    # Fallback - возвращаем известный tenant_id из константы
    default_tenant = "medyumed.2023-04-24"
    logger.warning(f"Не найдено тенантов в индексе, возвращен fallback tenant_id: {default_tenant}")
    return default_tenant


