#clinic_index.py
from typing import Dict, Any, Optional, List, Tuple
import logging
import re
from fuzzywuzzy import fuzz, process

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
    
    normalized = text.lower().replace("-", "")
    
    if keep_spaces or sort_words:
        normalized = re.sub(r'\s+', ' ', normalized).strip()
    else:
        normalized = re.sub(r'\s+', '', normalized)

    # Сортировка слов, если флаг установлен
    if sort_words:
        words = normalized.split()
        words.sort()
        final_normalized = " ".join(words)
        if not keep_spaces:
            final_normalized = re.sub(r'\s+', '', final_normalized)
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
                
                # Логика для обычного name_to_id (сохраняем для обратной совместимости)
                if normalized_name not in name_to_id:
                    name_to_id[normalized_name] = id_
                elif name_to_id[normalized_name] != id_: 
                     logger.debug(
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
                        logger.debug(
                            f"Tenant '{tenant_id}': Обнаружено несоответствие данных для filialId '{id_}'. "
                            f"Этот ID уже сопоставлен с оригинальным именем филиала '{id_to_name[id_]}', "
                            f"но другая запись пытается сопоставить его с именем '{name}'. "
                            f"Для функции get_name_by_id будет сохранено первое сопоставленное имя ('{id_to_name[id_]}'). "
                            f"Это НАСТОЯТЕЛЬНО УКАЗЫВАЕТ на то, что в исходном файле данных один и тот же filialId используется для разных названий филиалов."
                        )
                    else: # Общее предупреждение для других сущностей
                        logger.debug(
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
    logger.info(f"Построены индексы для тенанта {tenant_id}")


def _select_best_id_from_multiple_matches(
    tenant_id: str, 
    entity: str, 
    name_key_for_index: str, 
    candidate_ids: List[str], 
    normalized_search_name: str, 
    original_search_name: str
) -> Optional[str]:
    """
    Выбирает лучший ID из множественных совпадений с ГАРАНТИРОВАННОЙ стабильностью результата.
    
    Стратегия:
    1. Предпочтение по точности совпадения (наименьшее расстояние Левенштейна)
    2. Предпочтение по подстроке (если запрос является подстрокой названия)
    3. Предпочтение по длине названия (самое короткое название)
    4. СТРОГО стабильная сортировка по ID (для обеспечения повторяемости)
    
    Args:
        tenant_id: ID тенанта
        entity: Тип сущности 
        name_key_for_index: Ключ для доступа к индексу имён
        candidate_ids: Список ID-кандидатов
        normalized_search_name: Нормализованное имя для поиска
        original_search_name: Оригинальное имя для поиска
        
    Returns:
        Лучший ID или None, если не удалось определить
    """
    if not candidate_ids:
        return None
    
    if len(candidate_ids) == 1:
        return candidate_ids[0]
    
    # Сортируем candidate_ids для стабильности
    candidate_ids = sorted(candidate_ids)
    
    try:
        # Получаем карты индексов
        name_to_id_map_key = f"{name_key_for_index}_to_id"
        id_key = ENTITY_KEYS[[e[0] for e in ENTITY_KEYS].index(name_key_for_index)][1]
        id_to_name_map_key = f"{id_key}_to_name"
        
        name_to_id_map = TENANT_INDEXES.get(tenant_id, {}).get(name_to_id_map_key, {})
        id_to_name_map = TENANT_INDEXES.get(tenant_id, {}).get(id_to_name_map_key, {})
        
        if not name_to_id_map or not id_to_name_map:
            logger.warning(f"Не удалось получить индексы для выбора лучшего ID")
            return candidate_ids[0]  # Возвращаем первый по алфавиту
        
        # Анализируем каждый кандидат
        candidates_analysis = []
        
        for candidate_id in candidate_ids:
            original_name = id_to_name_map.get(candidate_id)
            if not original_name:
                continue
                
            # Определяем флаги нормализации для данного типа сущности
            keep_spaces_flag = True
            sort_words_flag = False
            for nk, ik, ks, sw in ENTITY_KEYS:
                if nk == name_key_for_index:
                    keep_spaces_flag = ks
                    sort_words_flag = sw
                    break
            
            normalized_candidate_name = normalize_text(original_name, keep_spaces=keep_spaces_flag, sort_words=sort_words_flag)
            
            # 1. Точность совпадения (расстояние Левенштейна)
            levenshtein_dist = levenshtein_distance(normalized_search_name, normalized_candidate_name)
            
            # 2. Длина названия (предпочитаем более короткие)
            name_length = len(original_name)
            
            # 3. Проверяем, является ли поисковый запрос подстрокой названия кандидата
            is_substring = normalized_search_name in normalized_candidate_name
            
            candidates_analysis.append({
                'id': candidate_id,
                'original_name': original_name,
                'normalized_name': normalized_candidate_name,
                'levenshtein_distance': levenshtein_dist,
                'name_length': name_length,
                'is_substring': is_substring
            })
        
        if not candidates_analysis:
            return candidate_ids[0]
        
        # СТАБИЛЬНАЯ сортировка по строгим критериям:
        # 1. По наименьшему расстоянию Левенштейна (точность)
        # 2. По подстроке (True идёт первым)
        # 3. По наименьшей длине названия
        # 4. По ID (СТРОГАЯ стабильная сортировка для повторяемости)
        candidates_analysis.sort(key=lambda x: (
            x['levenshtein_distance'],      # Сначала наименьшее расстояние
            not x['is_substring'],          # False (подстрока) идёт первым
            x['name_length'],               # Затем по длине
            x['id']                         # Наконец, стабильная сортировка по ID
        ))
        
        best_candidate = candidates_analysis[0]
        return best_candidate['id']
        
    except Exception as e:
        logger.error(f"Ошибка при выборе лучшего ID из множественных совпадений: {e}")
        # В случае ошибки возвращаем первый ID по алфавиту для стабильности
        return sorted(candidate_ids)[0]


def get_id_by_name(tenant_id: str, entity: str, name: str) -> Optional[str]:

    """
    Получить id по имени для сущности (service, employee, filial, category) и tenant_id.
    Использует normalize_text с гарантированной стабильностью результата.
    Сначала ищет точное совпадение, затем частичное (если имя из запроса является подстрокой полного имени).
    entity: 'service', 'employee', 'filial', 'category'
    """
    if not tenant_id or not entity or not name:
        logger.warning(f"Один из обязательных параметров пуст: tenant_id={tenant_id}, entity={entity}, name={name}")
        return None
        
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

    # Используем многозначный индекс для более стабильного поиска
    name_to_ids_map_key = f"{name_key_for_index}_to_ids"
    name_to_ids_map = TENANT_INDEXES.get(tenant_id, {}).get(name_to_ids_map_key, {})

    if not name_to_ids_map:
        logger.warning(f"Карта '{name_to_ids_map_key}' не найдена или пуста для тенанта '{tenant_id}'.")
        return None

    # 1. Поиск точного совпадения в многозначном индексе
    exact_match_ids = name_to_ids_map.get(normalized_name_to_search, [])
    if exact_match_ids:
        if len(exact_match_ids) == 1:
            return exact_match_ids[0]
        else:
            # Множественные точные совпадения - используем стабильную стратегию выбора
            best_id = _select_best_id_from_multiple_matches(
                tenant_id, entity, name_key_for_index, exact_match_ids, normalized_name_to_search, name
            )
            if best_id:
                return best_id

    # 2. Поиск частичного совпадения (если normalized_name_to_search является подстрокой ключа в карте)
    partial_matches_ids = []

    for indexed_norm_name, item_ids in name_to_ids_map.items():
        if normalized_name_to_search in indexed_norm_name:
            partial_matches_ids.extend(item_ids)

    if partial_matches_ids:
        # Убираем дубликаты и сортируем для стабильности
        unique_partial_ids = sorted(list(set(partial_matches_ids)))
        
        if len(unique_partial_ids) == 1:
            return unique_partial_ids[0]
        else:
            # Если частичных совпадений несколько, применяем стабильную стратегию выбора
            best_id = _select_best_id_from_multiple_matches(
                tenant_id, entity, name_key_for_index, unique_partial_ids, normalized_name_to_search, name
            )
            
            if best_id:
                return best_id

    # 3. Универсальный нечеткий поиск с использованием fuzzywuzzy для лучшей производительности
    if name_to_ids_map:
        # Создаем список всех нормализованных имен для поиска
        all_normalized_names = list(name_to_ids_map.keys())
        
        # Используем fuzzywuzzy для быстрого нечеткого поиска
        # ratio - обычное сравнение строк
        # partial_ratio - частичное совпадение (один текст содержится в другом)
        # token_sort_ratio - сравнение с сортировкой слов
        # token_set_ratio - сравнение множеств слов
        
        # Настраиваем пороги в зависимости от типа сущности и длины запроса
        if entity == 'filial':
            min_ratio = 80  # Высокий порог для филиалов
        elif entity == 'employee':
            min_ratio = 75  # Средний порог для сотрудников
        elif len(normalized_name_to_search) <= 10:
            min_ratio = 85  # Высокий порог для коротких запросов
        else:
            min_ratio = 70  # Нижний порог для длинных запросов
        
        # Извлекаем ключевые слова из поискового запроса для взвешенного поиска
        search_words = set(normalized_name_to_search.split())
        
        # Создаем взвешенные оценки для каждого кандидата
        weighted_candidates = []
        
        for candidate_name in all_normalized_names:
            candidate_words = set(candidate_name.split())
            
            # Базовая оценка fuzzywuzzy
            base_score = fuzz.token_set_ratio(normalized_name_to_search, candidate_name)
            
            # Пропускаем кандидатов с очень низким базовым score
            if base_score < min_ratio:
                continue
            
            # Бонус за совпадение ключевых слов
            common_words = search_words.intersection(candidate_words)
            word_match_bonus = len(common_words) * 10  # +10 баллов за каждое совпадающее слово
            
            # Дополнительный бонус за точные числовые совпадения (например, "1200")
            number_bonus = 0
            search_numbers = re.findall(r'\d+', normalized_name_to_search)
            candidate_numbers = re.findall(r'\d+', candidate_name)
            for num in search_numbers:
                if num in candidate_numbers:
                    number_bonus += 15  # +15 баллов за каждое совпадающее число
            
            # Штраф за избыточные слова в кандидате (предпочитаем более короткие названия)
            extra_words_penalty = max(0, len(candidate_words) - len(search_words)) * 2
            
            # Итоговая взвешенная оценка
            final_score = base_score + word_match_bonus + number_bonus - extra_words_penalty
            
            weighted_candidates.append({
                'name': candidate_name,
                'base_score': base_score,
                'word_bonus': word_match_bonus,
                'number_bonus': number_bonus,
                'penalty': extra_words_penalty,
                'final_score': final_score,
                'ids': name_to_ids_map[candidate_name]
            })
        
        if weighted_candidates:
            # Сортируем по итоговой оценке (убывание), затем по длине названия, затем по ID
            weighted_candidates.sort(key=lambda x: (-x['final_score'], len(x['name']), sorted(x['ids'])[0]))
            
            # Берем лучшего кандидата
            best_candidate = weighted_candidates[0]
            candidate_ids = best_candidate['ids']
            
            if len(candidate_ids) == 1:
                return candidate_ids[0]
            else:
                # Если у лучшего кандидата несколько ID, применяем стратегию выбора
                best_id = _select_best_id_from_multiple_matches(
                    tenant_id, entity, name_key_for_index, candidate_ids, normalized_name_to_search, name
                )
                if best_id:
                    return best_id

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
    
    
    keep_spaces_flag = True  
    sort_words_flag = False 
    
    for nk, ik, ks, sw in ENTITY_KEYS:
        if nk == name_key_for_index:
            keep_spaces_flag = ks
            sort_words_flag = sw
            break
    
    normalized_name_to_search = normalize_text(name, keep_spaces=keep_spaces_flag, sort_words=sort_words_flag)
    

    name_to_ids_map_key = f"{name_key_for_index}_to_ids"
    name_to_ids_map = TENANT_INDEXES.get(tenant_id, {}).get(name_to_ids_map_key, {})
    
    if normalized_name_to_search in name_to_ids_map:
        return name_to_ids_map[normalized_name_to_search]
    
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
        service_id = get_id_by_name(tenant_id, "service", service_name)
        if not service_id:
            return None
        
        category_id = get_category_id_by_service_id(tenant_id, service_id)
        if not category_id:
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
            return None
        return service_id
    except Exception as e:
        logger.error(f"Ошибка при получении service_id по названию '{service_name}': {e}")
        return None

def get_service_id_by_name_and_filial(tenant_id: str, service_name: str, filial_name: Optional[str] = None) -> Optional[str]:
    """
    Получить ID услуги по её названию с учетом филиала для избежания неоднозначности.
    Если filial_name указан, ищет услугу только в этом филиале.
    Если filial_name не указан, использует стандартный поиск с возможной неоднозначностью.
    
    Args:
        tenant_id: ID тенанта
        service_name: Название услуги
        filial_name: Название филиала (опционально)
        
    Returns:
        ID услуги или None, если не найдено
    """
    if not tenant_id or not service_name:
        return None
    
    if not filial_name:
        # Если филиал не указан, используем стандартный поиск
        return get_id_by_name(tenant_id, "service", service_name)
    
    # Если филиал указан, сначала получаем его ID
    filial_id = get_id_by_name(tenant_id, "filial", filial_name)
    if not filial_id:
        return None
    
    # Получаем все возможные ID услуги
    service_ids = get_all_ids_by_name(tenant_id, "service", service_name)
    if not service_ids:
        return None
    
    # Если только один ID услуги, возвращаем его
    if len(service_ids) == 1:
        return service_ids[0]
    
    # Если несколько ID, фильтруем по филиалу через сырые данные
    try:
        # Получаем доступ к индексам тенанта
        tenant_indexes = TENANT_INDEXES.get(tenant_id, {})
        if not tenant_indexes:
            logger.error(f"Не найдены индексы для тенанта '{tenant_id}'")
            return None
        
        # Ищем в сырых данных (это немного хак, но эффективно)
        # Загружаем данные из файла для точной фильтрации
        import json
        data_file = f"/home/erik/matrixai/base/{tenant_id}.json"
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Файл данных не найден: {data_file}")
            return None
        
        # Фильтруем услуги по филиалу
        matching_services = []
        for record in raw_data:
            if (record.get('filialId') == filial_id and 
                record.get('serviceId') in service_ids):
                matching_services.append(record['serviceId'])
        
        if not matching_services:
            return None
        
        # Убираем дубликаты и выбираем первый для стабильности
        unique_matching_services = sorted(list(set(matching_services)))
        selected_service_id = unique_matching_services[0]
        
        return selected_service_id
        
    except Exception as e:
        logger.error(f"Ошибка при поиске услуги '{service_name}' в филиале '{filial_name}': {e}")
        # Возвращаем первый ID как fallback
        return sorted(service_ids)[0]


def get_default_tenant_id() -> str:
    """
    Возвращает ID тенанта по умолчанию.
    В данной реализации предполагаем, что используется первый доступный тенант.
    """
    if TENANT_INDEXES:
        first_tenant = next(iter(TENANT_INDEXES.keys()), None)
        if first_tenant:
            return first_tenant
    
    return "medyumed.2023-04-24"


