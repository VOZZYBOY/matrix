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
    
    
    similar_services: Dict[str, Dict] = {}  
    
    
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
        
       
        if normalized_query in normalized_service_name:
            
            if service_id in similar_services:
                if filial_id and filial_name:
                    similar_services[service_id]["filials"][filial_id] = filial_name
            else:
             
                similar_services[service_id] = {
                    "serviceId": service_id,
                    "serviceName": service_name,
                    "categoryId": category_id,
                    "categoryName": category_name,
                    "filials": {filial_id: filial_name} if filial_id and filial_name else {},
                    "price": price
                }
    
  
    if similar_services:
        logger.info(f"Найдено {len(similar_services)} услуг по подстроке для запроса '{query}'")
        return list(similar_services.values())
    
  
    logger.info(f"Не найдено точных совпадений для '{query}', переходим к нечеткому поиску")
    
    
    all_services: Dict[str, Dict] = {}  
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
    
    
    threshold = 1 if len(normalized_query) <= 7 else 2
    
    
    fuzzy_matches = []
    for norm_name, service_info in all_services.items():
        from clinic_index import levenshtein_distance
        dist = levenshtein_distance(normalized_query, norm_name)
        if dist <= threshold:
            fuzzy_matches.append({
                'id': service_info["id"],
                'name': service_info["original_name"],
                'dist': dist
            })           
    if fuzzy_matches:
       
        fuzzy_matches.sort(key=lambda x: x['dist'])
        
       
        min_dist = fuzzy_matches[0]['dist']
        best_matches = [match for match in fuzzy_matches if match['dist'] == min_dist]
        
        logger.info(f"Найдено {len(best_matches)} нечетких совпадений для '{query}' с расстоянием {min_dist}")
        
        
        if len(best_matches) > 1:
           
            query_words = normalize_text(query, keep_spaces=True).lower().split()
            
          
            keywords_matches = []
            for match in best_matches:
                keywords_score = 0
                service_name = match['name'].lower()
                
                
                for word in query_words:
                    if word in service_name:
                        keywords_score += 1
                
                keywords_matches.append({
                    'id': match['id'],
                    'name': match['name'],
                    'score': keywords_score
                })
            
          
            keywords_matches.sort(key=lambda x: x['score'], reverse=True)
            
            
            if keywords_matches[0]['score'] > 0:
                logger.info(f"Выбрана услуга '{keywords_matches[0]['name']}' как наиболее соответствующая запросу '{query}' по ключевым словам (score: {keywords_matches[0]['score']})")
                best_matches = [m for m in best_matches if m['id'] == keywords_matches[0]['id']]
                
             
                if len(best_matches) > 1:
                 
                    query_keywords = []
                    if "колен" in query.lower():
                        query_keywords.append("колен")
                    if "голен" in query.lower():
                        query_keywords.append("голен")
                    if "ног" in query.lower():
                        query_keywords.append("ног")
                    if "рук" in query.lower():
                        query_keywords.append("рук")
                    
                   
                    if query_keywords:
                        for keyword in query_keywords:
                            exact_matches = [m for m in best_matches if keyword in m['name'].lower()]
                            if exact_matches:
                                logger.info(f"Выбрана услуга '{exact_matches[0]['name']}' благодаря точному совпадению ключевого слова '{keyword}'")
                                best_matches = exact_matches
                                break
        
      
        for match in best_matches:
            service_id = match['id']
            
            
            if service_id not in similar_services:
               
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
    
    similar_services.sort(key=lambda s: normalize_text(s["serviceName"], keep_spaces=True))
    
    message_parts = [f"Найдено {len(similar_services)} услуг, соответствующих запросу '{query}'. Пожалуйста, уточните:"]
    
    for i, service in enumerate(similar_services):
        filials_str = ", ".join(service["filials"].values())
        price_str = f" - {service['price']} руб." if service.get('price') is not None else ""
        message_parts.append(f"{i+1}. {service['serviceName']}{price_str} (категория: {service['categoryName']}, доступна в: {filials_str})")
    
    return "\n".join(message_parts), similar_services

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
        
        general_services = find_similar_services(tenant_id, query, _clinic_data)
        if not general_services:
            return f"Не найдено услуг, соответствующих запросу '{query}' ни в одном филиале.", []
        
        
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
            
            for i, service in enumerate(services[:3]): 
                price_str = f" - {service['price']} руб." if service.get('price') is not None else ""
                message_parts.append(f"- {service['serviceName']}{price_str} (категория: {service['categoryName']})")
            
            if len(services) > 3:
                message_parts.append(f"  ... и еще {len(services) - 3} услуг")
        
        return "\n".join(message_parts), general_services
    
    if len(similar_services) == 1:
        service = similar_services[0]
        price_str = f" (цена: {service['price']} руб.)" if service.get('price') is not None else ""
        return f"В филиале '{filial_name}' найдена услуга '{service['serviceName']}'{price_str} (категория: {service['categoryName']}).", similar_services
    
    similar_services.sort(key=lambda s: normalize_text(s["serviceName"], keep_spaces=True))
    
    message_parts = [f"В филиале '{filial_name}' найдено {len(similar_services)} услуг, соответствующих запросу '{query}'. Пожалуйста, уточните:"]
    
    for i, service in enumerate(similar_services):
        price_str = f" - {service['price']} руб." if service.get('price') is not None else ""
        message_parts.append(f"{i+1}. {service['serviceName']}{price_str} (категория: {service['categoryName']})")
    
    return "\n".join(message_parts), similar_services

def group_similar_service_names(services: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Улучшенная функция группировки услуг с похожими названиями.
    # Сортируем услуги по названию для консистентного отображения (агрегация отключена) услуг Ultraformer и других групп.
    
    Args:
        services: список словарей с информацией об услугах
        
    Returns:
        Dict[str, List[Dict]]: сгруппированные услуги по логическим категориям
    """
    if not services:
        return {}
        
    groups: Dict[str, List[Dict]] = {}
    
    for service in services:
        service_name = service.get("serviceName", "")
        if not service_name:
            continue
                
        groups[service_name] = [service]
            
    return groups

def format_grouped_services_message(query: str, grouped_services: Dict[str, List[Dict]]) -> str:
    """
    Улучшенная функция форматирования сообщения с группированными услугами.
    # Сортируем услуги по названию для консистентного отображения (агрегация отключена) услуг Ultraformer и других групп.
    
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
      
        for group in grouped_services.values():
            if group:
                service = group[0]
                price_str = f" (цена: {service.get('price')} руб.)" if service.get('price') is not None else ""
                return f"Найдена услуга '{service.get('serviceName')}'{price_str} (категория: {service.get('categoryName', 'Неизвестная')})."
    
   
    is_ultraformer = "ultraformer" in query.lower() or "ультрафомер" in query.lower()
    
    
    if is_ultraformer:
        message_parts = [f"Найдено {total_services} вариантов услуги Ultraformer для запроса '{query}'. Выберите нужный вариант:"]
    else:
        message_parts = [f"Найдено {total_services} услуг, соответствующих запросу '{query}'. Выберите нужную:"]
    
    
    sorted_groups = []
    
    
    if is_ultraformer:
        import re
        
       
        def extract_lines_number(group_name):
            matches = re.findall(r'(\d+)\s*лини', group_name.lower())
            return int(matches[0]) if matches else 0
        
        
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
        
        
        def get_body_part_priority(group_name):
            for part, priority in body_parts_priority.items():
                if part in group_name.lower():
                    return priority
            return 99
        
        
        for group_name, services in grouped_services.items():
            sorted_groups.append((group_name, services, extract_lines_number(group_name), get_body_part_priority(group_name)))
        
        
        sorted_groups.sort(key=lambda x: (-x[2], x[3]))
    else:
        
        for group_name, services in grouped_services.items():
            sorted_groups.append((group_name, services, len(group_name), len(services)))
        
        
        sorted_groups.sort(key=lambda x: (x[2], -x[3]))
    
    
    group_index = 1
    for group_name, services, _, _ in sorted_groups:
        if len(services) == 1:
            
            service = services[0]
            price_str = f" - {service.get('price')} руб." if service.get('price') is not None else ""
            category_str = f" (категория: {service.get('categoryName', 'Неизвестная')})" if service.get('categoryName') else ""
            message_parts.append(f"{group_index}. {service.get('serviceName')}{price_str}{category_str}")
            group_index += 1
        else:
            
            message_parts.append(f"{group_index}. Группа '{group_name}' ({len(services)} услуг):")
            
            
            sorted_services = sorted(services, key=lambda s: float(s.get('price', 0)) if s.get('price') is not None else 0)
            
            
            max_services_to_show = min(5, len(services))
            for i, service in enumerate(sorted_services[:max_services_to_show], 1):
                price_str = f" - {service.get('price')} руб." if service.get('price') is not None else ""
                message_parts.append(f"   {group_index}.{i}. {service.get('serviceName')}{price_str}")
            
            
            if len(services) > max_services_to_show:
                message_parts.append(f"   ... и еще {len(services) - max_services_to_show} услуг в этой группе")
            
            group_index += 1
    
    
    message_parts.append("\nПожалуйста, уточните нужную услугу, указав ее полное название или номер группы/услуги из списка.")
    
    return "\n".join(message_parts)
