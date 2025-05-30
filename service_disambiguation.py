from typing import Dict, List, Optional, Tuple, Set
import logging
import os
from clinic_index import get_id_by_name, get_name_by_id, normalize_text, get_category_id_by_service_id

logger = logging.getLogger(__name__)

def find_similar_services(tenant_id: str, query: str, _clinic_data: List[Dict]) -> List[Dict]:
    """
    Находит услуги, похожие на указанный запрос.
    Использует как подстроку, так и расстояние Левенштейна для нечеткого поиска.
    
    Args:
        tenant_id: ID тенанта
        query: поисковый запрос (часть названия услуги)
        _clinic_data: данные клиники
        
    Returns:
        список словарей, содержащих информацию о найденных услугах
    """
    normalized_query = normalize_text(query, keep_spaces=True).lower()
    
    # Собираем уникальные услуги, соответствующие запросу
    similar_services: Dict[str, Dict] = {}  # serviceId -> service_info
    
    # ЭТАП 1: Точное соответствие подстроки
    for item in _clinic_data:
        service_name = item.get("serviceName")
        service_id = item.get("serviceId")
        category_id = item.get("categoryId")
        category_name = item.get("categoryName")
        filial_id = item.get("filialId")
        filial_name = item.get("filialName")
        price = item.get("price")
        
        if not service_name or not service_id:
            continue
            
        normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
        
        # 1. Поиск по подстроке
        if normalized_query in normalized_service_name:
            # Если serviceId уже есть в словаре, добавляем филиал
            if service_id in similar_services:
                if filial_id and filial_name:
                    similar_services[service_id]["filials"][filial_id] = filial_name
            else:
                # Иначе создаем новую запись
                similar_services[service_id] = {
                    "serviceId": service_id,
                    "serviceName": service_name,
                    "categoryId": category_id,
                    "categoryName": category_name,
                    "filials": {filial_id: filial_name} if filial_id and filial_name else {},
                    "price": price
                }
    
    # Если уже нашли что-то по точному соответствию, возвращаем
    if similar_services:
        logger.info(f"Найдено {len(similar_services)} услуг по подстроке для запроса '{query}'")
        return list(similar_services.values())
    
    # ЭТАП 2: Нечеткий поиск с использованием расстояния Левенштейна
    # Используем только если не нашли точные совпадения
    logger.info(f"Не найдено точных совпадений для '{query}', переходим к нечеткому поиску")
    
    # Собираем все уникальные услуги для нечеткого поиска
    all_services: Dict[str, Dict] = {}  # normalized_service_name -> {id, original_name}
    for item in _clinic_data:
        service_name = item.get("serviceName")
        service_id = item.get("serviceId")
        if not service_name or not service_id:
            continue
        
        normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
        if normalized_service_name not in all_services:
            all_services[normalized_service_name] = {
                "id": service_id, 
                "original_name": service_name
            }
    
    # Определяем порог для нечеткого поиска
    threshold = 1 if len(normalized_query) <= 7 else 2
    
    # Ищем похожие услуги с помощью расстояния Левенштейна
    fuzzy_matches = []
    for norm_name, service_info in all_services.items():
        from clinic_index import levenshtein_distance
        dist = levenshtein_distance(normalized_query, norm_name)
        if dist <= threshold:
            fuzzy_matches.append({
                'id': service_info["id"],
                'name': service_info["original_name"],
                'dist': dist
            })            # Если нашли нечеткие совпадения
    if fuzzy_matches:
        # Сортируем по расстоянию (меньше - лучше)
        fuzzy_matches.sort(key=lambda x: x['dist'])
        
        # Берем все ID с минимальным расстоянием
        min_dist = fuzzy_matches[0]['dist']
        best_matches = [match for match in fuzzy_matches if match['dist'] == min_dist]
        
        logger.info(f"Найдено {len(best_matches)} нечетких совпадений для '{query}' с расстоянием {min_dist}")
        
        # Если у нас несколько совпадений с одинаковым минимальным расстоянием,
        # попробуем найти то, которое лучше соответствует запросу по ключевым словам
        if len(best_matches) > 1:
            # Разбиваем запрос на ключевые слова
            query_words = normalize_text(query, keep_spaces=True).lower().split()
            
            # Проверяем, есть ли в исходном запросе ключевые слова, которые могут помочь выбрать
            # между похожими услугами (например "колени" vs "голени")
            keywords_matches = []
            for match in best_matches:
                keywords_score = 0
                service_name = match['name'].lower()
                
                # Проверяем вхождение каждого слова из запроса в название услуги
                for word in query_words:
                    if word in service_name:
                        keywords_score += 1
                
                keywords_matches.append({
                    'id': match['id'],
                    'name': match['name'],
                    'score': keywords_score
                })
            
            # Сортируем по количеству совпадений ключевых слов (больше - лучше)
            keywords_matches.sort(key=lambda x: x['score'], reverse=True)
            
            # Если есть услуга с более высоким score по ключевым словам, выбираем её
            if keywords_matches[0]['score'] > 0:
                logger.info(f"Выбрана услуга '{keywords_matches[0]['name']}' как наиболее соответствующая запросу '{query}' по ключевым словам (score: {keywords_matches[0]['score']})")
                best_matches = [m for m in best_matches if m['id'] == keywords_matches[0]['id']]
                
                # Если после фильтрации по ключевым словам все еще несколько вариантов,
                # попробуем еще один метод - точное соответствие ключевых слов
                if len(best_matches) > 1:
                    # Составляем список точных ключевых слов из запроса
                    query_keywords = []
                    if "колен" in query.lower():
                        query_keywords.append("колен")
                    if "голен" in query.lower():
                        query_keywords.append("голен")
                    if "ног" in query.lower():
                        query_keywords.append("ног")
                    if "рук" in query.lower():
                        query_keywords.append("рук")
                    
                    # Если нашли хотя бы одно ключевое слово
                    if query_keywords:
                        for keyword in query_keywords:
                            exact_matches = [m for m in best_matches if keyword in m['name'].lower()]
                            if exact_matches:
                                logger.info(f"Выбрана услуга '{exact_matches[0]['name']}' благодаря точному совпадению ключевого слова '{keyword}'")
                                best_matches = exact_matches
                                break
        
        # Собираем данные для найденных услуг
        for match in best_matches:
            service_id = match['id']
            
            # Если этой услуги еще нет в результатах, добавляем
            if service_id not in similar_services:
                # Ищем первое вхождение этой услуги в _clinic_data для получения дополнительной информации
                for item in _clinic_data:
                    if item.get("serviceId") == service_id:
                        similar_services[service_id] = {
                            "serviceId": service_id,
                            "serviceName": item.get("serviceName"),
                            "categoryId": item.get("categoryId"),
                            "categoryName": item.get("categoryName"),
                            "filials": {item.get("filialId"): item.get("filialName")} 
                                      if item.get("filialId") and item.get("filialName") else {},
                            "price": item.get("price")
                        }
                        break
            
            # Добавляем информацию о филиалах для этой услуги
            if service_id in similar_services:
                for item in _clinic_data:
                    if item.get("serviceId") == service_id:
                        filial_id = item.get("filialId")
                        filial_name = item.get("filialName")
                        if filial_id and filial_name:
                            similar_services[service_id]["filials"][filial_id] = filial_name
    
    return list(similar_services.values())

def suggest_services(tenant_id: str, query: str, _clinic_data: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Формирует сообщение с предложениями услуг для уточнения пользовательского запроса.
    
    Args:
        tenant_id: ID тенанта
        query: поисковый запрос (часть названия услуги)
        _clinic_data: данные клиники
        
    Returns:
        tuple: (сообщение для пользователя, список найденных услуг)
    """
    similar_services = find_similar_services(tenant_id, query, _clinic_data)
    
    if not similar_services:
        return f"Не найдено услуг, соответствующих запросу '{query}'.", []
    
    if len(similar_services) == 1:
        service = similar_services[0]
        filials_str = ", ".join(service["filials"].values())
        return f"Найдена услуга '{service['serviceName']}' (категория: {service['categoryName']}). Доступна в филиалах: {filials_str}.", similar_services
    
    # Если услуг много и есть похожие названия (например, "Ultraformer 1500" для разных частей тела),
    # то группируем их для лучшего представления
    if len(similar_services) > 3:
        grouped_services = group_similar_service_names(similar_services)
        if len(grouped_services) < len(similar_services):  # Если группировка помогла сократить количество отображаемых элементов
            return format_grouped_services_message(query, grouped_services), similar_services
    
    # Сортируем услуги по названию для консистентного отображения
    similar_services.sort(key=lambda s: normalize_text(s["serviceName"], keep_spaces=True))
    
    message_parts = [f"Найдено {len(similar_services)} услуг, соответствующих запросу '{query}'. Пожалуйста, уточните:"]
    
    for i, service in enumerate(similar_services):
        filials_str = ", ".join(service["filials"].values())
        price_str = f" - {service['price']} руб." if service.get('price') is not None else ""
        message_parts.append(f"{i+1}. {service['serviceName']}{price_str} (категория: {service['categoryName']}, доступна в: {filials_str})")
    
    return "\n".join(message_parts), similar_services

def verify_service_in_filial(tenant_id: str, service_id: str, filial_id: str, _clinic_data: List[Dict]) -> bool:
    """
    Проверяет, доступна ли услуга в указанном филиале.
    
    Args:
        tenant_id: ID тенанта
        service_id: ID услуги
        filial_id: ID филиала
        _clinic_data: данные клиники
        
    Returns:
        bool: True, если услуга доступна в филиале, иначе False
    """
    for item in _clinic_data:
        if item.get("serviceId") == service_id and item.get("filialId") == filial_id:
            return True
    return False

def get_service_details(tenant_id: str, service_id: str, filial_id: Optional[str], _clinic_data: List[Dict]) -> Optional[Dict]:
    """
    Получает детали услуги с указанным ID в указанном филиале или в любом филиале, если filial_id=None.
    
    Args:
        tenant_id: ID тенанта
        service_id: ID услуги
        filial_id: ID филиала или None
        _clinic_data: данные клиники
        
    Returns:
        Dict или None: информация об услуге
    """
    for item in _clinic_data:
        if item.get("serviceId") == service_id:
            if filial_id is None or item.get("filialId") == filial_id:
                return {
                    "serviceId": service_id,
                    "serviceName": item.get("serviceName", ""),
                    "categoryId": item.get("categoryId", ""),
                    "categoryName": item.get("categoryName", ""),
                    "filialId": item.get("filialId", ""),
                    "filialName": item.get("filialName", ""),
                    "price": item.get("price", 0),
                    "serviceDescription": item.get("serviceDescription", "")
                }
    return None

def get_filials_for_service(tenant_id: str, service_id: str, _clinic_data: List[Dict]) -> List[Dict]:
    """
    Получает список филиалов, где доступна указанная услуга.
    
    Args:
        tenant_id: ID тенанта
        service_id: ID услуги
        _clinic_data: данные клиники
        
    Returns:
        список словарей с информацией о филиалах
    """
    filials: Dict[str, Dict] = {}  # filialId -> filial_info
    
    for item in _clinic_data:
        if item.get("serviceId") == service_id:
            filial_id = item.get("filialId")
            filial_name = item.get("filialName")
            if filial_id and filial_name and filial_id not in filials:
                filials[filial_id] = {
                    "filialId": filial_id,
                    "filialName": filial_name,
                    "price": item.get("price")
                }
    
    return list(filials.values())

def select_service_and_validate_filial(tenant_id: str, service_name: str, filial_name: str, _clinic_data: List[Dict]) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    Выбирает услугу по названию и проверяет её наличие в указанном филиале.
    
    Args:
        tenant_id: ID тенанта
        service_name: название услуги
        filial_name: название филиала
        _clinic_data: данные клиники
        
    Returns:
        tuple: (ID услуги, ID филиала, ID категории, сообщение)
    """
    service_id = get_id_by_name(tenant_id, 'service', service_name)
    if not service_id:
        # Если услуга не найдена, пробуем найти похожие
        message, similar_services = suggest_services(tenant_id, service_name, _clinic_data)
        return None, None, None, message
    
    filial_id = get_id_by_name(tenant_id, 'filial', filial_name)
    if not filial_id:
        filials = get_filials_for_service(tenant_id, service_id, _clinic_data)
        filial_names = [f["filialName"] for f in filials]
        return service_id, None, None, f"Филиал '{filial_name}' не найден. Услуга '{service_name}' доступна в следующих филиалах: {', '.join(filial_names)}."
    
    # Проверяем, доступна ли услуга в этом филиале
    if not verify_service_in_filial(tenant_id, service_id, filial_id, _clinic_data):
        # Если нет, смотрим, в каких филиалах она доступна
        filials = get_filials_for_service(tenant_id, service_id, _clinic_data)
        filial_names = [f["filialName"] for f in filials]
        return service_id, filial_id, None, f"Услуга '{service_name}' не доступна в филиале '{filial_name}'. Она доступна в следующих филиалах: {', '.join(filial_names)}."
    
    # Получаем ID категории для этой услуги
    category_id = get_category_id_by_service_id(tenant_id, service_id)
    if not category_id:
        # Если категория не найдена через быструю функцию, ищем в clinic_data
        service_details = get_service_details(tenant_id, service_id, filial_id, _clinic_data)
        if service_details:
            category_id = service_details.get("categoryId")
    
    # Все проверки пройдены
    display_service_name = get_name_by_id(tenant_id, 'service', service_id) or service_name
    display_filial_name = get_name_by_id(tenant_id, 'filial', filial_id) or filial_name
    
    if display_service_name != service_name or display_filial_name != filial_name:
        clarification = ""
        if display_service_name != service_name:
            clarification += f" (уточнено название услуги до '{display_service_name}')"
        if display_filial_name != filial_name:
            clarification += f" (уточнено название филиала до '{display_filial_name}')"
        message = f"Услуга доступна в указанном филиале{clarification}."
    else:
        message = "Услуга доступна в указанном филиале."
    
    return service_id, filial_id, category_id, message

def verify_service_for_employee(tenant_id: str, service_id: str, employee_id: str, _clinic_data: List[Dict]) -> bool:
    """
    Проверяет, выполняет ли сотрудник указанную услугу.
    
    Args:
        tenant_id: ID тенанта
        service_id: ID услуги
        employee_id: ID сотрудника
        _clinic_data: данные клиники
        
    Returns:
        bool: True, если сотрудник выполняет указанную услугу, иначе False
    """
    for item in _clinic_data:
        if item.get("serviceId") == service_id and item.get("employeeId") == employee_id:
            return True
    return False

async def verify_service_employee_filial_compatibility(tenant_id: str, service_id: str, employee_id: str, filial_id: str, api_token: str) -> Tuple[bool, str]:
    """
    Проверяет совместимость услуги, сотрудника и филиала через актуальный API.
    Использует рабочую комбинацию filial_id + service_id вместо employee_id + service_id.
    
    Args:
        tenant_id: ID тенанта
        service_id: ID услуги
        employee_id: ID сотрудника
        filial_id: ID филиала
        api_token: Bearer-токен для авторизации
        
    Returns:
        Tuple[bool, str]: (успех, сообщение)
    """
    # Получаем названия для сообщений
    service_name = get_name_by_id(tenant_id, 'service', service_id) or f"ID:{service_id}"
    employee_name = get_name_by_id(tenant_id, 'employee', employee_id) or f"ID:{employee_id}"
    filial_name = get_name_by_id(tenant_id, 'filial', filial_id) or f"ID:{filial_id}"
    
    try:
        from client_data_service import get_multiple_data_from_api
        
        # ШАГИ ПРОВЕРКИ:
        # 1. Проверяем, доступна ли услуга в указанном филиале (filial_id + service_id)
        # 2. Если да, то проверяем, работает ли указанный сотрудник среди тех, кто оказывает эту услугу в этом филиале
        
        logger.info(f"Проверяем совместимость: service_id={service_id}, employee_id={employee_id}, filial_id={filial_id}")
        
        # Шаг 1: Получаем всех сотрудников, которые оказывают данную услугу в данном филиале
        api_data = await get_multiple_data_from_api(
            api_token=api_token,
            service_id=service_id,
            filial_id=filial_id,
            tenant_id=tenant_id
        )
        
        if not api_data:
            # API не вернуло данных - значит услуга недоступна в этом филиале
            logger.warning(f"Услуга {service_id} недоступна в филиале {filial_id}")
            return False, f"Услуга '{service_name}' недоступна в филиале '{filial_name}' согласно актуальным данным."
        
        # Шаг 2: Проверяем, есть ли указанный сотрудник среди тех, кто оказывает эту услугу
        found_employee = False
        available_employees = []
        
        for item in api_data:
            item_employee_id = item.get('employeeId')
            item_employee_name = item.get('employeeFullName') or item.get('employeeName')
            
            if item_employee_name:
                available_employees.append(item_employee_name)
            
            if item_employee_id == employee_id:
                found_employee = True
                # Обновляем названия из актуальных данных API если они есть
                if item.get('serviceName'):
                    service_name = item.get('serviceName')
                if item_employee_name:
                    employee_name = item_employee_name
                if item.get('filialName'):
                    filial_name = item.get('filialName')
                
                logger.info(f"Найден сотрудник {employee_id} для услуги {service_id} в филиале {filial_id}")
        
        if found_employee:
            return True, f"Услуга '{service_name}' доступна у сотрудника '{employee_name}' в филиале '{filial_name}'."
        else:
            # Услуга доступна в филиале, но указанный сотрудник её не оказывает
            if available_employees:
                # Удаляем дубликаты и сортируем
                unique_employees = sorted(set(available_employees))
                employee_list = ', '.join(unique_employees)
                logger.warning(f"Сотрудник {employee_id} не найден среди {len(unique_employees)} доступных сотрудников")
                return False, f"Сотрудник '{employee_name}' не оказывает услугу '{service_name}' в филиале '{filial_name}'. Доступные сотрудники: {employee_list}."
            else:
                logger.warning(f"Нет доступных сотрудников для услуги {service_id} в филиале {filial_id}")
                return False, f"Услуга '{service_name}' недоступна в филиале '{filial_name}' или отсутствуют сотрудники для её оказания."
                
    except Exception as e:
        logger.error(f"Ошибка при проверке совместимости через API: {e}", exc_info=True)
        return False, f"Ошибка при проверке совместимости услуги, сотрудника и филиала: {str(e)}"

def validate_services_for_filial(tenant_id: str, service_ids: List[str], filial_id: str, _clinic_data: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """
    Проверяет доступность списка услуг в указанном филиале.
    
    Args:
        tenant_id: ID тенанта
        service_ids: Список ID услуг
        filial_id: ID филиала
        _clinic_data: данные клиники
        
    Returns:
        Tuple[List[str], List[Dict]]: (список валидных ID, список недоступных услуг с информацией)
    """
    valid_service_ids = []
    invalid_services = []
    
    for service_id in service_ids:
        if not service_id:  # Пропускаем пустые ID
            continue
            
        service_name = get_name_by_id(tenant_id, 'service', service_id) or f"ID:{service_id}"
        
        if verify_service_in_filial(tenant_id, service_id, filial_id, _clinic_data):
            valid_service_ids.append(service_id)
        else:
            # Получаем список филиалов, где доступна услуга
            filials = get_filials_for_service(tenant_id, service_id, _clinic_data)
            filial_names = [f["filialName"] for f in filials] if filials else []
            
            invalid_services.append({
                "serviceId": service_id,
                "serviceName": service_name,
                "availableFilials": filial_names
            })
    
    return valid_service_ids, invalid_services

def find_similar_services_in_filial(tenant_id: str, query: str, filial_id: str, _clinic_data: List[Dict]) -> List[Dict]:
    """
    Находит услуги, содержащие указанную подстроку в названии, доступные в конкретном филиале.
    
    Args:
        tenant_id: ID тенанта
        query: поисковый запрос (часть названия услуги)
        filial_id: ID филиала
        _clinic_data: данные клиники
        
    Returns:
        список словарей, содержащих информацию о найденных услугах
    """
    normalized_query = normalize_text(query, keep_spaces=True).lower()
    
    # Собираем уникальные услуги, соответствующие запросу и доступные в филиале
    similar_services: Dict[str, Dict] = {}  # serviceId -> service_info
    
    for item in _clinic_data:
        if item.get("filialId") != filial_id:
            continue
            
        service_name = item.get("serviceName")
        service_id = item.get("serviceId")
        category_id = item.get("categoryId")
        category_name = item.get("categoryName")
        price = item.get("price")
        
        if not service_name or not service_id:
            continue
            
        normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
        
        if normalized_query in normalized_service_name:
            if service_id not in similar_services:
                similar_services[service_id] = {
                    "serviceId": service_id,
                    "serviceName": service_name,
                    "categoryId": category_id,
                    "categoryName": category_name,
                    "price": price,
                    "filialId": filial_id
                }
    
    # Преобразуем словарь в список и сортируем по релевантности (длине названия)
    result = list(similar_services.values())
    result.sort(key=lambda x: len(x["serviceName"]))
    
    return result

def suggest_services_in_filial(tenant_id: str, query: str, filial_id: str, _clinic_data: List[Dict]) -> Tuple[str, List[Dict]]:
    """
    Формирует сообщение с предложениями услуг для уточнения пользовательского запроса с учетом филиала.
    
    Args:
        tenant_id: ID тенанта
        query: поисковый запрос (часть названия услуги)
        filial_id: ID филиала
        _clinic_data: данные клиники
        
    Returns:
        tuple: (сообщение для пользователя, список найденных услуг)
    """
    similar_services = find_similar_services_in_filial(tenant_id, query, filial_id, _clinic_data)
    filial_name = get_name_by_id(tenant_id, 'filial', filial_id) or f"ID:{filial_id}"
    
    if not similar_services:
        # Если в указанном филиале нет похожих услуг, поищем во всех филиалах
        general_services = find_similar_services(tenant_id, query, _clinic_data)
        if not general_services:
            return f"Не найдено услуг, соответствующих запросу '{query}' ни в одном филиале.", []
        
        # Сгруппируем услуги по филиалам
        services_by_filial: Dict[str, List[Dict]] = {}
        for service in general_services:
            for f_id, f_name in service.get("filials", {}).items():
                if f_id not in services_by_filial:
                    services_by_filial[f_id] = []
                services_by_filial[f_id].append({
                    "serviceId": service["serviceId"],
                    "serviceName": service["serviceName"],
                    "categoryName": service["categoryName"],
                    "price": service.get("price")
                })
        
        message_parts = [f"Услуга '{query}' не найдена в филиале '{filial_name}', но доступна в других филиалах:"]
        
        for fid, services in services_by_filial.items():
            f_name = get_name_by_id(tenant_id, 'filial', fid) or f"ID:{fid}"
            message_parts.append(f"\nВ филиале '{f_name}':")
            
            for i, service in enumerate(services[:3]): # Ограничиваем до 3 услуг на филиал
                price_str = f" - {service['price']} руб." if service.get('price') is not None else ""
                message_parts.append(f"- {service['serviceName']}{price_str} (категория: {service['categoryName']})")
            
            if len(services) > 3:
                message_parts.append(f"  ... и еще {len(services) - 3} услуг")
        
        return "\n".join(message_parts), general_services
    
    if len(similar_services) == 1:
        service = similar_services[0]
        price_str = f" (цена: {service['price']} руб.)" if service.get('price') is not None else ""
        return f"В филиале '{filial_name}' найдена услуга '{service['serviceName']}'{price_str} (категория: {service['categoryName']}).", similar_services
    
    # Если услуг много и есть похожие названия (например, "Ultraformer 1500" для разных частей тела),
    # то группируем их для лучшего представления
    if len(similar_services) > 3:
        grouped_services = group_similar_service_names(similar_services)
        if len(grouped_services) < len(similar_services):  # Если группировка помогла сократить количество отображаемых элементов
            message = format_grouped_services_message(query, grouped_services)
            message = message.replace("Найдено", f"В филиале '{filial_name}' найдено")
            return message, similar_services
    
    # Сортируем услуги по названию для консистентного отображения
    similar_services.sort(key=lambda s: normalize_text(s["serviceName"], keep_spaces=True))
    
    message_parts = [f"В филиале '{filial_name}' найдено {len(similar_services)} услуг, соответствующих запросу '{query}'. Пожалуйста, уточните:"]
    
    for i, service in enumerate(similar_services):
        price_str = f" - {service['price']} руб." if service.get('price') is not None else ""
        message_parts.append(f"{i+1}. {service['serviceName']}{price_str} (категория: {service['categoryName']})")
    
    return "\n".join(message_parts), similar_services

def group_similar_service_names(services: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Улучшенная функция группировки услуг с похожими названиями.
    Специально адаптирована для услуг Ultraformer и подобных процедур с множеством вариантов параметров.
    
    Args:
        services: список словарей с информацией об услугах
        
    Returns:
        Dict[str, List[Dict]]: сгруппированные услуги по логическим категориям
    """
    if not services:
        return {}
        
    # Проверяем, являются ли услуги услугами Ultraformer
    is_ultraformer = False
    for service in services:
        service_name = service.get("serviceName", "").lower()
        if "ultraformer" in service_name or "ультрафомер" in service_name:
            is_ultraformer = True
            break
    
    # Специальная группировка для Ultraformer
    if is_ultraformer:
        import re
        groups = {}
        
        # Вспомогательная функция для извлечения числа линий/точек
        def extract_number_param(name: str) -> str:
            line_matches = re.findall(r'(\d+)\s*(?:лини|точ)', name.lower())
            return line_matches[0] if line_matches else "0"
        
        # Вспомогательная функция для определения части тела
        body_part_keywords = {
            "лиц": "лицо",
            "шея": "шея",
            "шеи": "шея",
            "колен": "колени",
            "голен": "голени",
            "рук": "руки",
            "ног": "ноги",
            "бедр": "бедра",
            "ягодиц": "ягодицы",
            "живот": "живот",
            "спин": "спина",
            "декольт": "декольте",
            "тел": "тело"
        }
        
        def extract_body_part(name: str) -> str:
            name_lower = name.lower()
            for key, value in body_part_keywords.items():
                if key in name_lower:
                    return value
            return "другое"
        
        # Группируем услуги Ultraformer по числу линий и части тела
        for service in services:
            service_name = service.get("serviceName", "")
            if not service_name:
                continue
                
            # Определяем параметры группировки
            number_param = extract_number_param(service_name)
            body_part = extract_body_part(service_name)
            
            # Формируем ключ группы
            if "ultraformer" in service_name.lower():
                group_key = f"Ultraformer {number_param} линий - {body_part}"
            elif "ультрафомер" in service_name.lower():
                group_key = f"Ультрафомер {number_param} линий - {body_part}"
            else:
                group_key = f"Процедура {number_param} линий - {body_part}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(service)
            
    else:
        # Стандартная группировка для других услуг по общему префиксу
        groups: Dict[str, List[Dict]] = {}
        
        for service in services:
            service_name = service.get("serviceName", "")
            if not service_name:
                continue
                
            # Пытаемся найти общий префикс с существующими группами
            matched = False
            for prefix in list(groups.keys()):  # Используем list() для безопасного изменения словаря
                # Улучшенная проверка сходства:
                # 1. Общий префикс не менее 60% длины короткой строки И длиной не менее 5 символов
                # 2. Одна строка является префиксом другой при длине не менее 3 символов
                common_prefix = os.path.commonprefix([prefix, service_name])
                
                if ((len(common_prefix) >= 0.6 * min(len(prefix), len(service_name))) and len(common_prefix) >= 5) or \
                   (len(prefix) >= 3 and service_name.startswith(prefix)) or \
                   (len(service_name) >= 3 and prefix.startswith(service_name)):
                    groups[prefix].append(service)
                    matched = True
                    break
                    
            if not matched:
                # Создаем новую группу
                groups[service_name] = [service]
            
    return groups

def format_grouped_services_message(query: str, grouped_services: Dict[str, List[Dict]]) -> str:
    """
    Улучшенная функция форматирования сообщения с группированными услугами.
    Адаптирована для наглядного отображения услуг Ultraformer и других групп.
    
    Args:
        query: исходный запрос пользователя
        grouped_services: сгруппированные услуги по префиксам или категориям
        
    Returns:
        str: форматированное сообщение
    """
    if not grouped_services:
        return f"Не найдено услуг, соответствующих запросу '{query}'."
    
    total_services = sum(len(group) for group in grouped_services.values())
    if total_services == 1:
        # Если всего одна услуга, просто вернем ее название
        for group in grouped_services.values():
            if group:
                service = group[0]
                price_str = f" (цена: {service.get('price')} руб.)" if service.get('price') is not None else ""
                return f"Найдена услуга '{service.get('serviceName')}'{price_str} (категория: {service.get('categoryName', 'Неизвестная')})."
    
    # Определяем, является ли это запросом на Ultraformer
    is_ultraformer = "ultraformer" in query.lower() or "ультрафомер" in query.lower()
    
    # Формируем заголовок сообщения
    if is_ultraformer:
        message_parts = [f"Найдено {total_services} вариантов услуги Ultraformer для запроса '{query}'. Выберите нужный вариант:"]
    else:
        message_parts = [f"Найдено {total_services} услуг, соответствующих запросу '{query}'. Выберите нужную:"]
    
    # Сортируем группы для лучшего отображения
    sorted_groups = []
    
    # Для Ultraformer сортируем по числу линий и части тела
    if is_ultraformer:
        import re
        
        # Вспомогательная функция для извлечения числа линий для сортировки
        def extract_lines_number(group_name):
            matches = re.findall(r'(\d+)\s*лини', group_name.lower())
            return int(matches[0]) if matches else 0
        
        # Словарь приоритета частей тела для сортировки
        body_parts_priority = {
            "лицо": 1,
            "шея": 2, 
            "декольте": 3,
            "руки": 4,
            "ноги": 5,
            "колени": 6,
            "бедра": 7,
            "ягодицы": 8,
            "живот": 9,
            "спина": 10,
            "тело": 11,
            "другое": 99
        }
        
        # Вспомогательная функция для определения приоритета части тела
        def get_body_part_priority(group_name):
            for part, priority in body_parts_priority.items():
                if part in group_name.lower():
                    return priority
            return 99
        
        # Сортируем группы Ultraformer по числу линий и части тела
        for group_name, services in grouped_services.items():
            sorted_groups.append((group_name, services, extract_lines_number(group_name), get_body_part_priority(group_name)))
        
        # Сортируем: сначала по количеству линий (по убыванию), затем по приоритету части тела
        sorted_groups.sort(key=lambda x: (-x[2], x[3]))
    else:
        # Обычная сортировка по длине названия группы и количеству услуг
        for group_name, services in grouped_services.items():
            sorted_groups.append((group_name, services, len(group_name), len(services)))
        
        # Сортируем стандартные группы по возрастанию длины названия
        sorted_groups.sort(key=lambda x: (x[2], -x[3]))
    
    # Формируем сообщение с отсортированными группами
    group_index = 1
    for group_name, services, _, _ in sorted_groups:
        if len(services) == 1:
            # Если в группе одна услуга, показываем ее отдельно
            service = services[0]
            price_str = f" - {service.get('price')} руб." if service.get('price') is not None else ""
            category_str = f" (категория: {service.get('categoryName', 'Неизвестная')})" if service.get('categoryName') else ""
            message_parts.append(f"{group_index}. {service.get('serviceName')}{price_str}{category_str}")
            group_index += 1
        else:
            # Если в группе несколько услуг, создаем подгруппу
            message_parts.append(f"{group_index}. Группа '{group_name}' ({len(services)} услуг):")
            
            # Сортируем услуги в группе по цене
            sorted_services = sorted(services, key=lambda s: float(s.get('price', 0)) if s.get('price') is not None else 0)
            
            # Показываем максимум 5 услуг в группе
            max_services_to_show = min(5, len(services))
            for i, service in enumerate(sorted_services[:max_services_to_show], 1):
                price_str = f" - {service.get('price')} руб." if service.get('price') is not None else ""
                message_parts.append(f"   {group_index}.{i}. {service.get('serviceName')}{price_str}")
            
            # Если в группе больше 5 услуг, добавляем информацию об остальных
            if len(services) > max_services_to_show:
                message_parts.append(f"   ... и еще {len(services) - max_services_to_show} услуг в этой группе")
            
            group_index += 1
    
    # Добавляем инструкцию для пользователя
    message_parts.append("\nПожалуйста, уточните нужную услугу, указав ее полное название или номер группы/услуги из списка.")
    
    return "\n".join(message_parts)
