# clinic_functions.py

import logging
import re
from typing import Optional, List, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field
from client_data_service import get_free_times_of_employee_by_services, add_record
from clinic_index import get_name_by_id, normalize_text, get_id_by_name
import asyncio
from datetime import datetime

_clinic_data: List[Dict[str, Any]] = []
_tenant_id_for_clinic_data: Optional[str] = None
logger = logging.getLogger(__name__)

def get_readable_count_form(count, forms: Tuple[str, str, str]) -> str:
    """
    Возвращает правильную форму слова для заданного числа.
    forms: кортеж из трех строк (например, ("яблоко", "яблока", "яблок"))
    """
    if count % 10 == 1 and count % 100 != 11:
        return forms[0]
    elif 2 <= count % 10 <= 4 and (count % 100 < 10 or count % 100 >= 20):
        return forms[1]
    else:
        return forms[2]
# --- КОНЕЦ ДОБАВЛЕНИЯ ---

def set_clinic_data(data: List[Dict[str, Any]], tenant_id: Optional[str] = None):
    """
    Устанавливает данные клиники для использования функциями в этом модуле.
    Вызывается один раз при инициализации основного скрипта.
    """
    global _clinic_data, _tenant_id_for_clinic_data
    _clinic_data = data
    _tenant_id_for_clinic_data = tenant_id
    if _clinic_data:
        logger.info(f"[Funcs] Данные клиники ({len(_clinic_data)} записей) установлены для функций тенанта '{_tenant_id_for_clinic_data if _tenant_id_for_clinic_data else 'Неизвестно'}'.")
    else:
        logger.warning(f"[Funcs] Установлены пустые данные клиники для тенанта '{_tenant_id_for_clinic_data if _tenant_id_for_clinic_data else 'Неизвестно'}'.")



# --- Вспомогательная функция для получения оригинального названия филиала ---
def get_original_filial_name(normalized_name: str) -> Optional[str]:
    """Находит оригинальное название филиала по его нормализованному имени."""
    if not normalized_name or not _clinic_data: return None
    # Оптимизация: создаем карту нормализованных имен один раз, если она нужна часто
    # Но для редких вызовов можно итерировать
    for item in _clinic_data:
        original_name = item.get("filialName")
        if original_name and normalize_text(original_name) == normalized_name:
            return original_name
    return None # Или можно вернуть normalized_name.capitalize() как фоллбэк

# --- Определения Классов Функций (с использованием normalize_text) ---

class FindEmployees(BaseModel):
    """
    Модель для поиска сотрудников по различным критериям.
    
    Режимы работы:
    1. Получение списка сотрудников в филиале (указан только filial_name)
    2. Получение списка услуг сотрудника в филиале (указаны employee_name и filial_name)
    3. Поиск сотрудников, выполняющих услугу в филиале (указаны service_name и filial_name)
    4. Общий поиск по всем критериям
    """
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")
    page_number: int = Field(default=1, description="Номер страницы результатов")
    page_size: int = Field(default=10, description="Количество результатов на странице")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[FindEmployees] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."
        
        logger.info(f"[FC Proc] Поиск сотрудников (Сотрудник: '{self.employee_name}', Услуга: '{self.service_name}', Филиал: '{self.filial_name}'), Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        # Шаг 1: Получаем ID для filial_name и service_name из запроса
        target_filial_id: Optional[str] = None
        display_filial_name_query = self.filial_name
        if self.filial_name:
            target_filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
            if not target_filial_id:
                return f"Филиал с названием, похожим на '{self.filial_name}', не найден."
            logger.info(f"Найден ID филиала '{target_filial_id}' для имени '{self.filial_name}'.")

        # Нормализуем запросы для поиска
        norm_emp_name = normalize_text(self.employee_name, keep_spaces=True) if self.employee_name else ""
        norm_service_name = normalize_text(self.service_name, keep_spaces=True) if self.service_name else ""

        filtered_data = []
        for item in _clinic_data:
            item_emp_name_raw = item.get('employeeFullName')
            item_service_name_raw = item.get('serviceName')

            item_filial_id_raw = item.get('filialId')

            norm_item_emp = normalize_text(item_emp_name_raw, keep_spaces=True) if item_emp_name_raw else ""
            norm_item_service = normalize_text(item_service_name_raw, keep_spaces=True) if item_service_name_raw else ""


            emp_match = (not norm_emp_name or (norm_item_emp and norm_emp_name in norm_item_emp))
            
            if target_filial_id: # Если искали конкретный филиал
                filial_match = (item_filial_id_raw == target_filial_id)
            else: # Если филиал не был указан, то совпадение по филиалу всегда True
                filial_match = True
            
            service_match = (not norm_service_name or (norm_item_service and norm_service_name in norm_item_service))

            if emp_match and service_match and filial_match:
                filtered_data.append(item)

        if not filtered_data:
             search_criteria = []
             if self.employee_name: search_criteria.append(f"имя содержит '{self.employee_name}'")
             if self.service_name: search_criteria.append(f"услуга содержит '{self.service_name}'")
             if self.filial_name: search_criteria.append(f"филиал '{self.filial_name}'")
             criteria_str = ", ".join(search_criteria) if search_criteria else "указанным критериям"
             return f"Сотрудники, соответствующие {criteria_str}, не найдены."

        employees_info: Dict[str, Dict[str, Any]] = {}
        for item in filtered_data:
            e_id = item.get('employeeId')
            if not e_id: continue

            if e_id not in employees_info:
                employees_info[e_id] = {
                    'name': item.get('employeeFullName'),
                    'services': set(),
                    'filials': set()
                }

            s_name = item.get('serviceName')
            f_name = item.get('filialName')

            # Добавляем оригинальные имена в сеты
            if s_name and (not norm_service_name or norm_service_name in normalize_text(s_name, keep_spaces=True)):
                 employees_info[e_id]['services'].add(s_name)
            
            # Отображаем филиал, если он был найден по ID или если фильтра по филиалу не было
            if f_name:
                if target_filial_id and item.get('filialId') == target_filial_id: # Если фильтровали по филиалу и это он
                     employees_info[e_id]['filials'].add(f_name)
                elif not target_filial_id: # Если фильтра по филиалу не было, добавляем все филиалы сотрудника
                     employees_info[e_id]['filials'].add(f_name)

        response_parts = []
        limit = 5
        count = 0
        found_count = 0
        # Сортируем по нормализованному имени для консистентности
        sorted_employees = sorted(employees_info.items(), key=lambda x: normalize_text(x[1].get('name', ''), keep_spaces=True))
        
        # Применяем пагинацию к отсортированному списку ID сотрудников
        start_idx = (self.page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        paginated_employee_ids = [emp_id for emp_id, _ in sorted_employees[start_idx:end_idx]]

        for emp_id in paginated_employee_ids:
            emp_data = employees_info[emp_id]
            name = emp_data.get('name')
            if not name: continue

            # Получаем и сортируем оригинальные имена услуг и филиалов
            services = sorted(list(emp_data.get('services', set())), key=lambda s: normalize_text(s, keep_spaces=True))
            filials = sorted(list(emp_data.get('filials', set())), key=normalize_text)

            # Пропускаем, если обязательные фильтры не дали результатов для этого сотрудника
            if norm_service_name and not services: continue
            if target_filial_id and not filials: # Если искали филиал и его нет у сотрудника после всех фильтров
                continue

            found_count += 1
            if count < limit:
                service_str = f"   Услуги: {', '.join(services)}" if services else ""
                filial_str = f"   Филиалы: {', '.join(filials)}" if filials else ""
                emp_info = f"- {name}" # Выводим оригинальное имя
                if filial_str: emp_info += f"\n{filial_str}"
                if service_str: emp_info += f"\n{service_str}"
                response_parts.append(emp_info)
                count += 1

        if found_count == 0:
             return "Сотрудники найдены по части критериев, но ни один не соответствует всем условиям."

        final_response = ["Найдены следующие сотрудники:"] + response_parts
        if found_count > limit:
             final_response.append(f"\n... (и еще {found_count - limit} сотрудник(ов). Уточните запрос.)")

        return "\n".join(final_response)


class GetServicePrice(BaseModel):
    """Модель для получения цены на конкретную услугу."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос цены происходит в процессе записи на прием")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[GetServicePrice] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос цены (Услуга: {self.service_name}, Филиал: {self.filial_name}, В процессе записи: {self.in_booking_process}), Tenant: {_tenant_id_for_clinic_data}")

        # Поиск услуг напрямую по названию без преобразования в ID
        normalized_query = normalize_text(self.service_name, keep_spaces=True).lower()
        display_service_name = self.service_name
        
        # Соберем все подходящие услуги по названию
        matching_services = []
        
        # Сначала ищем точные совпадения
        exact_matches = []
        for item in _clinic_data:
            service_name = item.get("serviceName")
            if not service_name:
                continue
                
            normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
            
            # Точное совпадение
            if normalized_service_name == normalized_query:
                exact_matches.append(item)
        
        # Если есть точные совпадения, используем их
        if exact_matches:
            matching_services = exact_matches
            logger.info(f"[GetServicePrice] Найдено {len(matching_services)} точных совпадений для услуги '{self.service_name}'")
            display_service_name = exact_matches[0].get("serviceName") or self.service_name
        else:
            # Ищем услуги, содержащие запрос как подстроку
            substring_matches = []
            for item in _clinic_data:
                service_name = item.get("serviceName")
                if not service_name:
                    continue
                    
                normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
                
                # Совпадение по подстроке
                if normalized_query in normalized_service_name:
                    substring_matches.append(item)
            
            # Если есть совпадения по подстроке, используем их
            if substring_matches:
                matching_services = substring_matches
                logger.info(f"[GetServicePrice] Найдено {len(matching_services)} совпадений по подстроке для услуги '{self.service_name}'")
                display_service_name = substring_matches[0].get("serviceName") or self.service_name
            else:
                # Если нет ни точного соответствия, ни по подстроке, используем нечеткий поиск
                # Используем отдельную функцию из service_disambiguation для специализированного поиска
                from service_disambiguation import find_similar_services
                similar_services = find_similar_services(_tenant_id_for_clinic_data, self.service_name, _clinic_data)
                
                if similar_services:
                    # Преобразуем результаты из find_similar_services в формат элементов из _clinic_data
                    service_ids = [s['serviceId'] for s in similar_services]
                    matching_services = [item for item in _clinic_data if item.get('serviceId') in service_ids]
                    logger.info(f"[GetServicePrice] С помощью нечеткого поиска найдено {len(matching_services)} похожих услуг для '{self.service_name}'")
        
        # Если нет совпадений вообще, сообщаем об ошибке
        if not matching_services:
            logger.warning(f"[GetServicePrice] Услуга '{self.service_name}' не найдена ни по точному соответствию, ни по подстроке, ни по нечеткому поиску")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        
        # Проверяем, содержит ли название запрашиваемой услуги специфические слова для фильтрации
        # Расширенный список ключевых слов для частей тела и областей применения
        keyword_filters = {
            "колен": "колени",
            "голен": "голени", 
            "бедр": "бедра",
            "рук": "руки",
            "лиц": "лицо",
            "шеи": "шея",
            "шея": "шея",
            "ягодиц": "ягодицы",
            "спин": "спина",
            "живот": "живот",
            "декольт": "декольте",
            "подбород": "подбородок",
            "щек": "щеки",
            "лоб": "лоб", 
            "скул": "скулы",
            "шрам": "шрамы",
            "глаз": "глаза",
            "висок": "виски",
            "груд": "грудь",
            "плеч": "плечи",
            "тел": "тело",
            "зон": "зона"
        }
        
        # Расширенный набор фильтров для Ultraformer и подобных процедур
        ultraformer_filters = [
            # Ищем число линий в запросе (например, 1500)
            r"(\d+)\s*лини",
            # Ищем число точек в запросе
            r"(\d+)\s*точ",
            # Ищем зоны обработки
            r"(\d+)\s*зон"
        ]
        
        # Специализированные словари для обработки запросов Ultraformer
        ultraformer_specific_parts = {
            'лоб': 'лоб',
            'щеки': 'щеки',
            'щек': 'щеки',
            'скул': 'скулы',
            'скулы': 'скулы',
            'подборо': 'подбородок',
            'шея': 'шея',
            'декольте': 'декольте',
            'колен': 'колени',
            'бедр': 'бедра',
            'ягодиц': 'ягодицы',
            'живот': 'живот'
        }
        
        filtered_services = matching_services
        used_filters = []
        
        query_lower = self.service_name.lower()
        is_ultraformer = "ultraformer" in query_lower or "ультрафомер" in query_lower
        
        # Специализированная логика для Ultraformer
        if is_ultraformer:
            import re
            logger.info(f"[GetServicePrice] Обнаружен запрос для Ultraformer: '{self.service_name}'")
            
            # Шаг 1: Поиск числовых параметров (линий)
            ultraformer_numbers = []
            for pattern in ultraformer_filters:
                matches = re.findall(pattern, query_lower)
                if matches:
                    for number_value in matches:
                        ultraformer_numbers.append(number_value)
                        logger.info(f"[GetServicePrice] Обнаружен параметр '{number_value}' в запросе Ultraformer")
            
            # Шаг 2: Поиск специфичных частей тела для Ultraformer
            ultraformer_parts = []
            for part_key, part_name in ultraformer_specific_parts.items():
                if part_key in query_lower:
                    ultraformer_parts.append(part_name)
                    logger.info(f"[GetServicePrice] Обнаружена часть тела '{part_name}' в запросе Ultraformer")
            
            # Шаг 3: Фильтрация по найденным параметрам
            if ultraformer_numbers:
                number_filtered = []
                for service in filtered_services:
                    service_name = service.get('serviceName', '').lower()
                    has_all_numbers = True
                    for num in ultraformer_numbers:
                        # Проверка, содержит ли название услуги числовой параметр
                        if not re.search(fr"{num}\s*(?:лини|точ|зон)", service_name):
                            has_all_numbers = False
                            break
                    if has_all_numbers:
                        number_filtered.append(service)
                
                if number_filtered:
                    filtered_services = number_filtered
                    logger.info(f"[GetServicePrice] Фильтрация по числовым параметрам {ultraformer_numbers}: выбрано {len(filtered_services)} вариантов")
            
            # Шаг 4: Фильтрация по частям тела
            if ultraformer_parts:
                part_filtered = []
                for service in filtered_services:
                    service_name = service.get('serviceName', '').lower()
                    for part in ultraformer_parts:
                        part_key = part.lower()
                        if part_key in service_name:
                            part_filtered.append(service)
                            break
                
                if part_filtered:
                    filtered_services = part_filtered
                    logger.info(f"[GetServicePrice] Фильтрация по частям тела {ultraformer_parts}: выбрано {len(filtered_services)} вариантов")
        else:
            # Для не-Ultraformer используем обычную фильтрацию по ключевым словам
            for keyword, display_word in keyword_filters.items():
                if keyword in self.service_name.lower():
                    keyword_services = [s for s in filtered_services if keyword in s.get('serviceName', '').lower()]
                    if keyword_services:
                        filtered_services = keyword_services
                        used_filters.append(display_word)
                        logger.info(f"[GetServicePrice] Фильтрация по '{keyword}': выбрано {len(filtered_services)} вариантов")
        
        # Если у нас больше одной похожей услуги
        if len(filtered_services) > 1:
            # Пытаемся сгруппировать услуги с одинаковыми характеристиками
            try:
                # Группируем услуги по нормализованному имени (без учета регистра)
                service_groups = {}
                for service in filtered_services:
                    service_name = service.get('serviceName', '')
                    norm_name = normalize_text(service_name, keep_spaces=True).lower()
                    
                    if norm_name not in service_groups:
                        service_groups[norm_name] = []
                    service_groups[norm_name].append(service)
                
                # Если все услуги имеют одно и то же нормализованное имя, но разные ID
                if len(service_groups) == 1:
                    # Это случай, когда одна и та же услуга имеет разные ID в разных филиалах
                    # Выбираем первую услугу из группы, так как они все эквивалентны
                    first_group = next(iter(service_groups.values()))
                    filtered_services = [first_group[0]]
                    logger.info(f"[GetServicePrice] Обнаружены одинаковые услуги с разными ID. Выбрана первая из {len(first_group)}")
            except Exception as e:
                logger.warning(f"[GetServicePrice] Ошибка при группировке услуг: {e}")
            
            # Если после группировки всё ещё осталось более одной услуги
            if len(filtered_services) > 1:
                # В процессе записи автоматически выбираем наиболее подходящую услугу
                if self.in_booking_process or len(filtered_services) <= 5:  # Автоматический выбор, если мало вариантов
                    # Улучшенный алгоритм выбора наиболее релевантной услуги на основе взвешенной оценки
                    import re
                    from difflib import SequenceMatcher
                    
                    query = self.service_name.lower()
                    best_score = -1
                    best_service = filtered_services[0]
                    
                    for service in filtered_services:
                        service_name = service.get('serviceName', '').lower()
                        score = 0
                        
                        # Оценка 1: Совпадение слов
                        query_words = query.split()
                        word_match_score = sum(1 for word in query_words if word in service_name)
                        score += word_match_score * 5  # Вес для совпадения слов
                        
                        # Оценка 2: Процент совпадения строки (нечеткое совпадение)
                        similarity_ratio = SequenceMatcher(None, query, service_name).ratio()
                        score += similarity_ratio * 20  # Вес для общего совпадения строки
                        
                        # Оценка 3: Точное совпадение чисел (например, "1500" линий)
                        numbers_in_query = re.findall(r'\d+', query)
                        numbers_in_service = re.findall(r'\d+', service_name)
                        
                        # Увеличенный вес для точного совпадения чисел
                        for num in numbers_in_query:
                            if num in numbers_in_service:
                                score += 15  # Увеличенный вес для совпадения чисел
                        
                        # Оценка 4: Совпадение и контекст числовых параметров
                        for num_q in numbers_in_query:
                            for num_s in numbers_in_service:
                                if num_q == num_s:
                                    # Проверяем контекст числа в запросе и услуге
                                    # Например, "1500 линий" должно соответствовать "1500 линий", а не просто "1500"
                                    q_context = re.findall(fr"{num_q}\s*([а-яА-Яa-zA-Z]+)", query)
                                    s_context = re.findall(fr"{num_s}\s*([а-яА-Яa-zA-Z]+)", service_name)
                                    
                                    if q_context and s_context and q_context[0].lower() == s_context[0].lower():
                                        score += 10  # Дополнительные очки за совпадение контекста числа
                        
                        # Оценка 5: Учет ключевых слов для частей тела с повышенным приоритетом
                        is_ultraformer = "ultraformer" in query or "ультрафомер" in query
                        
                        if is_ultraformer:
                            # Для Ultraformer даем больший вес совпадениям частей тела
                            for keyword, display_word in ultraformer_specific_parts.items():
                                if keyword in query and keyword in service_name:
                                    score += 12  # Высокий вес для специфичных частей тела Ultraformer
                        else:
                            # Для обычных услуг используем стандартные ключевые слова
                            for keyword in keyword_filters:
                                if keyword in query and keyword in service_name:
                                    score += 8  # Вес для совпадения ключевых слов
                        
                        # Оценка 6: Штраф за лишние слова/параметры, которых нет в запросе
                        service_words = set(service_name.split())
                        query_words_set = set(query_words)
                        extra_words = len(service_words - query_words_set)
                        score -= extra_words * 2  # Небольшой штраф за лишние слова в названии услуги
                        
                        logger.debug(f"[GetServicePrice] Оценка для '{service_name}': {score}")
                        
                        if score > best_score:
                            best_score = score
                            best_service = service
                            
                    logger.info(f"[GetServicePrice] Автоматически выбрана услуга '{best_service.get('serviceName')}' среди {len(filtered_services)} вариантов (счет: {best_score})")
                    filtered_services = [best_service]
                    display_service_name = best_service.get('serviceName') or self.service_name
                else:
                    # При обычном запросе цены предлагаем выбор пользователю с улучшенной группировкой
                    response_parts = [f"Для запроса '{self.service_name}' найдено {len(filtered_services)} похожих услуг:"]
                    
                    # Если много услуг, используем улучшенную группировку для лучшего отображения
                    if len(filtered_services) > 5:
                        # Шаг 1: Определяем основные параметры группировки
                        query_lower = self.service_name.lower()
                        is_ultraformer = "ultraformer" in query_lower or "ультрафомер" in query_lower
                        
                        # Определяем параметры для группировки
                        grouping_params = {}
                        
                        # Для Ultraformer используем специальную группировку
                        if is_ultraformer:
                            import re
                            
                            # Группировка по числу линий/точек
                            by_lines = {}
                            for service in filtered_services:
                                sname = service.get('serviceName', '').lower()
                                line_matches = re.findall(r'(\d+)\s*лини', sname)
                                lines_key = line_matches[0] if line_matches else "неизвестно"
                                
                                if lines_key not in by_lines:
                                    by_lines[lines_key] = []
                                by_lines[lines_key].append(service)
                            
                            # Группировка по частям тела внутри групп по линиям
                            result_groups = {}
                            
                            for lines_key, services in by_lines.items():
                                # Для каждой группы линий группируем по части тела
                                body_parts = {}
                                
                                for service in services:
                                    sname = service.get('serviceName', '').lower()
                                    body_part = "другое"
                                    
                                    # Определяем часть тела - сначала по специфическим для Ultraformer
                                    for keyword, display_label in ultraformer_specific_parts.items():
                                        if keyword in sname:
                                            body_part = display_label
                                            break
                                    
                                    # Если не нашли в специфических, проверяем общие
                                    if body_part == "другое":
                                        for keyword, display_label in keyword_filters.items():
                                            if keyword in sname:
                                                body_part = display_label
                                                break
                                    
                                    if body_part not in body_parts:
                                        body_parts[body_part] = []
                                    body_parts[body_part].append(service)
                                
                                # Добавляем в итоговый словарь
                                for body_part, body_services in body_parts.items():
                                    key = f"Ultraformer {lines_key} линий - {body_part}"
                                    result_groups[key] = body_services
                            
                            # Добавляем группы в ответ
                            for group_name, services in sorted(result_groups.items()):
                                response_parts.append(f"\n- {group_name} ({len(services)} услуг):")
                                for i, service in enumerate(services[:3], 1):  # Показываем только 3 первых услуги
                                    price_str = f" - {service.get('price')} руб." if service.get('price') is not None else ""
                                    response_parts.append(f"  {i}. {service.get('serviceName')}{price_str}")
                                if len(services) > 3:
                                    response_parts.append(f"  ...и еще {len(services)-3} вариантов")
                        else:
                            # Стандартная группировка по частям тела для других услуг
                            by_body_part = {}
                            
                            for service in filtered_services:
                                sname = service.get('serviceName', '').lower()
                                body_part = "другое"
                                
                                # Определяем часть тела из названия
                                for keyword, display_label in keyword_filters.items():
                                    if keyword in sname:
                                        body_part = display_label
                                        break
                                
                                if body_part not in by_body_part:
                                    by_body_part[body_part] = []
                                by_body_part[body_part].append(service)
                            
                            # Показываем услуги сгруппированными
                            for body_part, services in by_body_part.items():
                                response_parts.append(f"\n- Для {body_part} ({len(services)} услуг):")
                                for i, service in enumerate(services[:5], 1):  # Ограничиваем 5 услугами на группу
                                    price_str = f" - {service.get('price')} руб." if service.get('price') is not None else ""
                                    response_parts.append(f"  {i}. {service.get('serviceName')}{price_str}")
                                if len(services) > 5:
                                    response_parts.append(f"  ...и еще {len(services)-5} услуг")
                    else:
                        # Если услуг немного, просто перечисляем их
                        for i, service in enumerate(filtered_services, 1):
                            price_str = f" - {service.get('price')} руб." if service.get('price') is not None else ""
                            response_parts.append(f"{i}. {service.get('serviceName')}{price_str}")
                    
                    response_parts.append("\nПожалуйста, уточните, какую именно услугу вы имели в виду, указав её название или номер из списка.")
                    return "\n".join(response_parts)
        
        # Если осталась только одна услуга после фильтрации
        if len(filtered_services) == 1:
            display_service_name = filtered_services[0].get('serviceName') or self.service_name
            logger.info(f"[GetServicePrice] Используем '{display_service_name}' в качестве основной услуги для отображения")
        
        # Получаем информацию о филиале, если указан
        target_filial_name = self.filial_name
        display_filial_name = target_filial_name
        
        # Ищем филиал по имени, используя функцию get_id_by_name
        filial_id = None
        if target_filial_name:
            # Используем системную функцию, которая правильно нормализует название
            if _tenant_id_for_clinic_data:
                filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', target_filial_name)
                
            if filial_id:
                logger.info(f"[GetServicePrice] Найден филиал '{target_filial_name}' с ID '{filial_id}'")
                # Получаем точное имя филиала из индекса для отображения
                exact_filial_name = get_name_by_id(_tenant_id_for_clinic_data, 'filial', filial_id)
                if exact_filial_name:
                    display_filial_name = exact_filial_name
            else:
                # Если не нашли через индекс, пробуем старый способ прямого поиска по названию
                normalized_filial_query = normalize_text(target_filial_name, keep_spaces=True).lower()
                filial_found = False
                
                for item in _clinic_data:
                    filial_name = item.get("filialName")
                    if not filial_name:
                        continue
                    
                    normalized_filial_name = normalize_text(filial_name, keep_spaces=True).lower()
                    
                    if normalized_filial_name == normalized_filial_query or normalized_filial_query in normalized_filial_name:
                        filial_found = True
                        display_filial_name = filial_name
                        # Запомним ID филиала для дальнейшего сравнения
                        filial_id = item.get("filialId")
                        break
                        
                if not filial_found:
                    logger.warning(f"[GetServicePrice] Филиал '{self.filial_name}' не найден при поиске цены на '{self.service_name}'")
                    return f"Филиал с названием, похожим на '{self.filial_name}', не найден при поиске цены на '{display_service_name}'."
                
            logger.info(f"[GetServicePrice] Найден филиал '{display_filial_name}'")
        
        # Собираем цены для найденных услуг
        candidate_prices = []
        logger.info(f"[GetServicePrice] Начинаем поиск цен для {len(filtered_services)} услуг" + 
                   (f" в филиале '{display_filial_name}' (ID: {filial_id})" if filial_id else ""))
        
        for item in filtered_services:
            service_name = item.get('serviceName')
            price_raw = item.get('price')
            item_filial_name = item.get('filialName')
            item_filial_id = item.get('filialId')
            
            logger.debug(f"[GetServicePrice] Проверяем услугу '{service_name}' в филиале '{item_filial_name}' (ID: {item_filial_id}), цена: {price_raw}")
            
            if price_raw is None or price_raw == '':
                logger.debug(f"[GetServicePrice] Для услуги '{service_name}' не найдена цена")
                continue
                
            try:
                price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                logger.debug(f"[GetServicePrice] Найдена цена {price} для услуги '{service_name}'")
            except (ValueError, TypeError):
                logger.warning(f"[GetServicePrice] Не удалось преобразовать цену '{price_raw}' для услуги '{service_name}'")
                continue
                
            # Если запрошен конкретный филиал, и он не совпадает с текущим, пропускаем
            if target_filial_name:
                if not item_filial_name:
                    continue
                
                # Если у нас есть ID филиала, сравниваем по нему (точное совпадение)
                if filial_id:
                    item_filial_id = item.get("filialId")
                    if item_filial_id != filial_id:
                        logger.debug(f"[GetServicePrice] ID филиала '{item_filial_id}' не соответствует запрошенному ID '{filial_id}'")
                        continue
                else:
                    # Запасной вариант - сравнение по нормализованным названиям
                    normalized_filial = normalize_text(item_filial_name, keep_spaces=True).lower()
                    normalized_target_filial = normalize_text(target_filial_name, keep_spaces=True).lower()
                    
                    if normalized_filial != normalized_target_filial and normalized_target_filial not in normalized_filial:
                        logger.debug(f"[GetServicePrice] Филиал '{item_filial_name}' не соответствует запрошенному '{target_filial_name}'")
                        continue
            
            candidate_prices.append({
                'price': price,
                'filial_name': item_filial_name or "(не указан)"
            })
            logger.info(f"[GetServicePrice] Добавлена цена {price} руб. для услуги '{service_name}' в филиале '{item_filial_name or '(не указан)'}'")

        if not candidate_prices:
            filial_context = f" в филиале '{display_filial_name}'" if target_filial_name else ""
            logger.warning(f"[GetServicePrice] Не найдены цены для услуги '{display_service_name}'{filial_context}")
            
            # Добавим подробную диагностику
            if target_filial_name and filial_id:
                logger.warning(f"[GetServicePrice] Диагностика: Филиал с ID '{filial_id}' найден, но услуга в нем не найдена")
                
                # Поищем эту услугу в других филиалах для диагностики
                service_in_other_filials = []
                for item in _clinic_data:
                    if item.get('serviceId') == filtered_services[0].get('serviceId') and item.get('price') is not None:
                        other_filial_name = item.get('filialName', 'Неизвестный филиал')
                        other_filial_id = item.get('filialId', 'Нет ID')
                        if other_filial_id != filial_id:
                            service_in_other_filials.append(f"'{other_filial_name}' (ID: {other_filial_id})")
                
                if service_in_other_filials:
                    logger.info(f"[GetServicePrice] Услуга '{display_service_name}' найдена в других филиалах: {', '.join(service_in_other_filials)}")
            
            return f"Цена на услугу '{display_service_name}'{filial_context} не найдена."

        # Если был запрос для конкретного филиала и нашлись цены
        if target_filial_name:
            if len(candidate_prices) == 1:
                price_info = candidate_prices[0]
                # Если исходное название услуги отличается от найденного, уточним это в ответе
                is_similar_service = normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True)
                service_clarification = ""
                
                if is_similar_service:
                    if self.in_booking_process:
                        service_clarification = f" (автоматически подобрана похожая услуга: '{display_service_name}')"
                    else:
                        service_clarification = f" (найдена похожая услуга: '{display_service_name}')"
                
                filial_clarification = f" (филиал '{display_filial_name}')" 
                logger.info(f"[GetServicePrice] Найдена цена {price_info['price']} руб. для услуги '{display_service_name}' в филиале '{display_filial_name}'")
                return f"Цена на '{self.service_name}'{service_clarification}{filial_clarification} составляет {price_info['price']:.0f} руб."
            elif len(candidate_prices) > 1:
                # Улучшенный алгоритм выбора цены при наличии нескольких вариантов
                
                # Группируем цены по точному названию услуги
                prices_by_exact_name = {}
                for cp in candidate_prices:
                    exact_name = cp.get('exact_service_name', '')
                    if not exact_name:
                        continue
                        
                    if exact_name not in prices_by_exact_name:
                        prices_by_exact_name[exact_name] = []
                    prices_by_exact_name[exact_name].append(cp['price'])
                
                # Если все цены относятся к одной точной услуге, вернем среднюю
                if len(prices_by_exact_name) == 1:
                    exact_name = next(iter(prices_by_exact_name.keys()))
                    prices = prices_by_exact_name[exact_name]
                    avg_price = sum(prices) / len(prices)
                    logger.info(f"[GetServicePrice] Для услуги '{exact_name}' найдено {len(prices)} одинаковых цен, средняя: {avg_price:.0f} руб.")
                    
                    is_similar_service = normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True)
                    service_clarification = ""
                    
                    if is_similar_service:
                        if self.in_booking_process:
                            service_clarification = f" (автоматически подобрана услуга: '{display_service_name}')"
                        else:
                            service_clarification = f" (найдена похожая услуга: '{display_service_name}')"
                        
                    filial_clarification = f" (филиал '{display_filial_name}')"
                    return f"Цена на '{self.service_name}'{service_clarification}{filial_clarification} составляет {avg_price:.0f} руб."
                else:
                    # Если есть разные услуги с разными ценами, но у нас включен режим booking_process
                    if self.in_booking_process:
                        # Берем цену услуги с наилучшим соответствием имени запроса
                        import difflib
                        
                        query = self.service_name.lower()
                        best_match = None
                        best_similarity = -1
                        
                        for cp in candidate_prices:
                            service_name = cp.get('exact_service_name', '').lower()
                            similarity = difflib.SequenceMatcher(None, query, service_name).ratio()
                            
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_match = cp
                        
                        if best_match:
                            logger.info(f"[GetServicePrice] Автоматически выбрана цена {best_match['price']} для наиболее похожей услуги '{best_match.get('exact_service_name')}' (сходство: {best_similarity:.2f})")
                            
                            service_clarification = f" (автоматически подобрана похожая услуга: '{best_match.get('exact_service_name')}')"
                            filial_clarification = f" (филиал '{display_filial_name}')"
                            return f"Цена на '{self.service_name}'{service_clarification}{filial_clarification} составляет {best_match['price']:.0f} руб."
                    
                    # В обычном режиме - возвращаем диапазон цен
                    all_prices = [cp['price'] for cp in candidate_prices]
                    min_price = min(all_prices)
                    max_price = max(all_prices)
                    
                    if min_price == max_price:
                        logger.info(f"[GetServicePrice] Все {len(all_prices)} цен совпадают: {min_price:.0f} руб.")
                        price_msg = f"{min_price:.0f} руб."
                    else:
                        logger.info(f"[GetServicePrice] Диапазон цен: от {min_price:.0f} до {max_price:.0f} руб.")
                        price_msg = f"от {min_price:.0f} до {max_price:.0f} руб."
                    
                    is_similar_service = normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True)
                    service_clarification = f" (найдено {len(candidate_prices)} вариантов услуги)"
                    
                    if is_similar_service:
                        service_clarification = f" (найдено {len(candidate_prices)} вариантов похожей услуги: '{display_service_name}')"
                        
                    filial_clarification = f" (филиал '{display_filial_name}')"
                    return f"Цена на '{self.service_name}'{service_clarification}{filial_clarification} составляет {price_msg}"
            else: # Не нашли цен для запрошенного филиала
                logger.warning(f"[GetServicePrice] Цена для услуги '{display_service_name}' в филиале '{display_filial_name}' не найдена")
                
                # Если мы пытались найти похожую услугу и не нашли цену
                if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True):
                    return f"Цена на услугу '{self.service_name}' (искали как '{display_service_name}') в филиале '{display_filial_name}' не найдена."
                else:
                    return f"Цена на услугу '{display_service_name}' в филиале '{display_filial_name}' не найдена."

        # Если филиал не был указан, а цены есть (могут быть разные для разных филиалов)
        # Группируем по филиалам, чтобы избежать дублирования цен в одном филиале
        prices_by_filial: Dict[str, float] = {}
        for cp in candidate_prices:
            filial_name = cp['filial_name'] or "(не указан)"
            normalized_filial_name = normalize_text(filial_name, keep_spaces=True).lower()
            
            # Берем первую попавшуюся цену для филиала, если вдруг их несколько
            if normalized_filial_name not in prices_by_filial:
                prices_by_filial[normalized_filial_name] = {
                    'price': cp['price'],
                    'display_name': filial_name
                }
                logger.debug(f"[GetServicePrice] Добавлена цена {cp['price']} руб. для филиала {filial_name}")
        
        unique_prices_with_filials = [
            {'price': info['price'], 'filial_name': info['display_name']}
            for _, info in prices_by_filial.items()
        ]
        unique_prices_with_filials.sort(key=lambda x: (x['price'], normalize_text(x['filial_name'])))
        logger.info(f"[GetServicePrice] Найдено {len(unique_prices_with_filials)} уникальных цен для услуги '{display_service_name}'")

        service_clarification = f" (уточнено до '{display_service_name}')" if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True) else ""
        if self.in_booking_process and normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True):
            service_clarification = f" (автоматически подобрана похожая услуга: '{display_service_name}')"

        if len(unique_prices_with_filials) == 1:
            price_info = unique_prices_with_filials[0]
            filial_text = f" в филиале {price_info['filial_name']}" if price_info['filial_name'] != "(не указан)" else ""
            logger.info(f"[GetServicePrice] Возвращается цена {price_info['price']} руб. для услуги '{display_service_name}'{filial_text}")
            return f"Цена на '{self.service_name}'{service_clarification}{filial_text} составляет {price_info['price']:.0f} руб."
        else:
            logger.info(f"[GetServicePrice] Возвращается {len(unique_prices_with_filials)} цен для услуги '{display_service_name}' в разных филиалах")
            response_parts = [f"Цена на '{self.service_name}'{service_clarification} различается в зависимости от филиала:"]
            limit = 5
            for i, price_info in enumerate(unique_prices_with_filials):
                if i >= limit: break
                filial_text = f" (в филиале {price_info['filial_name']}" if price_info['filial_name'] != "(не указан)" else " (филиал не указан)"
                response_parts.append(f"- {price_info['price']:.0f} руб.{filial_text})")
            if len(unique_prices_with_filials) > limit:
                response_parts.append(f"... и еще в {len(unique_prices_with_filials) - limit} местах.")
            return "\n".join(response_parts)


class ListFilials(BaseModel):
    """Модель для получения списка филиалов."""
    # Нет аргументов

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info("[FC Proc] Запрос списка филиалов")

        # Собираем оригинальные имена филиалов
        filials: Set[str] = set(filter(None, (item.get('filialName') for item in _clinic_data)))

        if not filials:
            return "Информация о филиалах не найдена."
        # Сортируем оригинальные имена по их нормализованным версиям
        return "Доступные филиалы клиники:\n*   " + "\n*   ".join(sorted(list(filials), key=normalize_text))


class GetEmployeeServices(BaseModel):
    """Модель для получения списка услуг конкретного сотрудника."""
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество услуг на странице")

    def process(self) -> str:
        if not _clinic_data:
            return "Ошибка: База данных клиники пуста."
        logger.info(f"[FC Proc] Запрос услуг сотрудника: {self.employee_name}, Page: {self.page_number}, Size: {self.page_size}")

        if not _tenant_id_for_clinic_data:
            logger.error(f"[GetEmployeeServices] _tenant_id_for_clinic_data не установлен. Невозможно найти сотрудника '{self.employee_name}'.")
            return "Системная ошибка: не удалось определить идентификатор клиники для поиска сотрудника."

        employee_id_found = get_id_by_name(_tenant_id_for_clinic_data, 'employee', self.employee_name)

        if not employee_id_found:
            logger.warning(f"[GetEmployeeServices] Сотрудник с именем '{self.employee_name}' (ID не найден) не найден для тенанта {_tenant_id_for_clinic_data}.")
            return f"Сотрудник с именем, похожим на '{self.employee_name}', не найден."
        
        actual_employee_name_from_db = get_name_by_id(_tenant_id_for_clinic_data, 'employee', employee_id_found) or self.employee_name
        logger.info(f"[GetEmployeeServices] Найден сотрудник: '{actual_employee_name_from_db}' (ID: '{employee_id_found}') для тенанта {_tenant_id_for_clinic_data}.")

        # Сначала соберем все уникальные услуги (serviceName) для данного сотрудника
        all_services_for_employee: Set[str] = set()
        employee_services: Dict[str, List[str]] = {} # Услуги по филиалам
        filials_of_employee: Set[str] = set() # Филиалы, где работает сотрудник
        
        for item in _clinic_data:
            if item.get('employeeId') == employee_id_found:
                service_name = item.get('serviceName')
                filial_name = item.get('filialName')
                category_name = item.get('categoryName')

                if service_name and filial_name and category_name:
                    full_service_info = f"{service_name} (Категория: {category_name})"
                    if filial_name not in employee_services:
                        employee_services[filial_name] = []
                    if full_service_info not in employee_services[filial_name]:
                        employee_services[filial_name].append(full_service_info)
                    filials_of_employee.add(filial_name)
        
        if not actual_employee_name_from_db: # Если после цикла имя так и не нашлось (маловероятно, если ID есть)
            actual_employee_name_from_db = self.employee_name

        if not employee_services:
            return f"Не найдено услуг для сотрудника '{actual_employee_name_from_db}' (ID: {employee_id_found}). Возможно, он не оказывает услуг или данные неполны."

        response_parts = [f"Услуги сотрудника '{actual_employee_name_from_db}' (ID: {employee_id_found}):"]
        if len(filials_of_employee) > 1:
            response_parts.append(f"(Работает в филиалах: {', '.join(sorted(list(filials_of_employee)))})\n")
        
        for filial, services in sorted(employee_services.items()):
            response_parts.append(f"\nФилиал: {filial}")
            for i, service_info in enumerate(sorted(services)):
                response_parts.append(f"  {i+1}. {service_info}")
        
        return "\n".join(response_parts)


class CheckServiceInFilial(BaseModel):
    """Модель для проверки наличия услуги в филиале."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос происходит в процессе записи на прием")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[CheckServiceInFilial] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Проверка услуги '{self.service_name}' в филиале '{self.filial_name}', В процессе записи: {self.in_booking_process}, Tenant: {_tenant_id_for_clinic_data}")

        # Поиск услуг напрямую по названию без преобразования в ID
        normalized_query = normalize_text(self.service_name, keep_spaces=True).lower()
        normalized_filial_query = normalize_text(self.filial_name, keep_spaces=True).lower()
        display_service_name = self.service_name
        display_filial_name = self.filial_name
        
        # Соберем все подходящие услуги по названию
        matching_services = []
        
        # Сначала ищем точные совпадения
        exact_matches = []
        for item in _clinic_data:
            service_name = item.get("serviceName")
            if not service_name:
                continue
                
            normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
            
            # Точное совпадение
            if normalized_service_name == normalized_query:
                exact_matches.append(item)
        
        # Если есть точные совпадения, используем их
        if exact_matches:
            matching_services = exact_matches
            logger.info(f"[CheckServiceInFilial] Найдено {len(matching_services)} точных совпадений для услуги '{self.service_name}'")
            display_service_name = exact_matches[0].get("serviceName") or self.service_name
        else:
            # Ищем услуги, содержащие запрос как подстроку
            substring_matches = []
            for item in _clinic_data:
                service_name = item.get("serviceName")
                if not service_name:
                    continue
                    
                normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
                
                # Совпадение по подстроке
                if normalized_query in normalized_service_name:
                    substring_matches.append(item)
            
            # Если есть совпадения по подстроке, используем их
            if substring_matches:
                matching_services = substring_matches
                logger.info(f"[CheckServiceInFilial] Найдено {len(matching_services)} совпадений по подстроке для услуги '{self.service_name}'")
                display_service_name = substring_matches[0].get("serviceName") or self.service_name
            else:
                # Если нет ни точного соответствия, ни по подстроке, используем нечеткий поиск
                from service_disambiguation import find_similar_services
                similar_services = find_similar_services(_tenant_id_for_clinic_data, self.service_name, _clinic_data)
                
                if similar_services:
                    # Преобразуем результаты из find_similar_services в формат элементов из _clinic_data
                    service_ids = [s['serviceId'] for s in similar_services]
                    matching_services = [item for item in _clinic_data if item.get('serviceId') in service_ids]
                    logger.info(f"[CheckServiceInFilial] С помощью нечеткого поиска найдено {len(matching_services)} похожих услуг для '{self.service_name}'")
        
        # Если нет совпадений вообще, сообщаем об ошибке
        if not matching_services:
            logger.warning(f"[CheckServiceInFilial] Услуга '{self.service_name}' не найдена ни по точному соответствию, ни по подстроке, ни по нечеткому поиску")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        
        # Проверяем, содержит ли название запрашиваемой услуги специфические слова для фильтрации
        keyword_filters = {
            "колен": "колени",
            "голен": "голени",
            "бедр": "бедра",
            "рук": "руки"
        }
        
        filtered_services = matching_services
        used_filters = []
        
        # Применяем все подходящие фильтры к услугам
        for keyword, display_word in keyword_filters.items():
            if keyword in self.service_name.lower():
                keyword_services = [s for s in filtered_services if keyword in s.get('serviceName', '').lower()]
                if keyword_services:
                    filtered_services = keyword_services
                    used_filters.append(display_word)
                    logger.info(f"[CheckServiceInFilial] Фильтрация по '{keyword}': выбрано {len(filtered_services)} вариантов")
                
                # Если у нас больше одной похожей услуги
                if len(filtered_services) > 1:
                    # В процессе записи автоматически выбираем наиболее подходящую услугу
                    if self.in_booking_process:
                        # Выбираем услугу с наибольшим количеством совпадающих ключевых слов
                        max_keyword_matches = 0
                        best_service = filtered_services[0]
                        
                        for service in filtered_services:
                            service_name = service.get('serviceName', '').lower()
                            query_words = self.service_name.lower().split()
                            
                            # Подсчитываем количество совпадающих слов
                            matches = sum(1 for word in query_words if word in service_name)
                            
                            if matches > max_keyword_matches:
                                max_keyword_matches = matches
                                best_service = service
                                
                        logger.info(f"[CheckServiceInFilial] В процессе записи автоматически выбрана услуга '{best_service.get('serviceName')}' среди {len(filtered_services)} похожих вариантов")
                        filtered_services = [best_service]
                        display_service_name = best_service.get('serviceName') or self.service_name
                    else:
                        # При обычном запросе предлагаем выбор пользователю
                        logger.info(f"[CheckServiceInFilial] Найдено {len(filtered_services)} похожих услуг для выбора пользователем")
                        response_parts = [f"Для запроса '{self.service_name}' найдено {len(filtered_services)} похожих услуг:"]
                        
                        # Добавляем все похожие услуги в список для выбора
                        for i, service in enumerate(filtered_services, 1):
                            response_parts.append(f"{i}. {service.get('serviceName')}")
                        
                        response_parts.append("\nПожалуйста, уточните, какую именно услугу вы хотите проверить в филиале '{self.filial_name}', указав её название или номер из списка.")
                        return "\n".join(response_parts)
                
                # Если осталась только одна услуга после фильтрации
                if len(filtered_services) == 1:
                    display_service_name = filtered_services[0].get('serviceName') or self.service_name
                    logger.info(f"[CheckServiceInFilial] Используем '{display_service_name}' в качестве основной услуги для отображения")
                elif len(filtered_services) == 0:
                    logger.warning(f"[CheckServiceInFilial] Услуга '{self.service_name}' не найдена после фильтрации")
                    return f"Услуга с названием, похожим на '{self.service_name}', не найдена в базе."

        # Поиск филиала напрямую по названию (без ID)
        normalized_filial_query = normalize_text(self.filial_name, keep_spaces=True).lower()
        filial_found = False
        display_filial_name = self.filial_name
        
        for item in _clinic_data:
            filial_name = item.get("filialName")
            if not filial_name:
                continue
            
            normalized_filial_name = normalize_text(filial_name, keep_spaces=True).lower()
            
            if normalized_filial_name == normalized_filial_query or normalized_filial_query in normalized_filial_name:
                filial_found = True
                display_filial_name = filial_name
                break
                
        if not filial_found:
            logger.warning(f"[CheckServiceInFilial] Филиал '{self.filial_name}' не найден")
            return f"Филиал с названием, похожим на '{self.filial_name}', не найден в базе."

        # Проверяем наличие услуги в филиале
        service_found_in_filial = False
        matching_services_in_filial = []
        
        for service in filtered_services:
            service_name = service.get('serviceName')
            filial_name = service.get('filialName')
            
            if not filial_name or not service_name:
                continue
                
            normalized_filial_name = normalize_text(filial_name, keep_spaces=True).lower()
            
            if normalized_filial_name == normalized_filial_query or normalized_filial_query in normalized_filial_name:
                service_found_in_filial = True
                matching_services_in_filial.append(service_name)
        
        # Удаляем дубликаты из списка услуг
        matching_services_in_filial = list(set(matching_services_in_filial))
        
        service_clarification = f" (уточнено до '{display_service_name}')" if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True) else ""
        if self.in_booking_process and normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True):
            service_clarification = f" (автоматически подобрана похожая услуга: '{display_service_name}')"
        filial_clarification = f" (уточнено до '{display_filial_name}')" if normalize_text(display_filial_name, keep_spaces=True) != normalize_text(self.filial_name, keep_spaces=True) else ""

        if service_found_in_filial:
            if len(matching_services_in_filial) > 1:
                services_list = ", ".join(matching_services_in_filial)
                return f"Да, услуги, подобные '{self.service_name}', доступны в филиале '{self.filial_name}'{filial_clarification}: {services_list}"
            else:
                return f"Да, услуга '{self.service_name}'{service_clarification} доступна в филиале '{self.filial_name}'{filial_clarification}."
        else:
            return f"Нет, услуга '{self.service_name}'{service_clarification} не найдена в филиале '{self.filial_name}'{filial_clarification}."


class CompareServicePriceInFilials(BaseModel):
    """Модель для сравнения цен на услугу в нескольких филиалах."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(min_length=2, description="Список из ДВУХ или БОЛЕЕ филиалов")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос происходит в процессе записи на прием")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[CompareServicePriceInFilials] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."
        if not self.filial_names or len(self.filial_names) < 2:
            return "Ошибка: Нужно указать как минимум два названия филиала для сравнения."

        logger.info(f"[FC Proc] Сравнение цен на '{self.service_name}' в филиалах: {self.filial_names}, В процессе записи: {self.in_booking_process}, Tenant: {_tenant_id_for_clinic_data}")

        # Поиск услуг напрямую по названию без преобразования в ID
        normalized_query = normalize_text(self.service_name, keep_spaces=True).lower()
        display_service_name = self.service_name
        
        # Соберем все подходящие услуги по названию
        matching_services = []
        
        # Сначала ищем точные совпадения
        exact_matches = []
        for item in _clinic_data:
            service_name = item.get("serviceName")
            if not service_name:
                continue
                
            normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
            
            # Точное совпадение
            if normalized_service_name == normalized_query:
                exact_matches.append(item)
        
        # Если есть точные совпадения, используем их
        if exact_matches:
            matching_services = exact_matches
            logger.info(f"[CompareServicePriceInFilials] Найдено {len(matching_services)} точных совпадений для услуги '{self.service_name}'")
            display_service_name = exact_matches[0].get("serviceName") or self.service_name
        else:
            # Ищем услуги, содержащие запрос как подстроку
            substring_matches = []
            for item in _clinic_data:
                service_name = item.get("serviceName")
                if not service_name:
                    continue
                    
                normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
                
                # Совпадение по подстроке
                if normalized_query in normalized_service_name:
                    substring_matches.append(item)
            
            # Если есть совпадения по подстроке, используем их
            if substring_matches:
                matching_services = substring_matches
                logger.info(f"[CompareServicePriceInFilials] Найдено {len(matching_services)} совпадений по подстроке для услуги '{self.service_name}'")
                display_service_name = substring_matches[0].get("serviceName") or self.service_name
            else:
                # Если нет ни точного соответствия, ни по подстроке, используем нечеткий поиск
                from service_disambiguation import find_similar_services
                similar_services = find_similar_services(_tenant_id_for_clinic_data, self.service_name, _clinic_data)
                
                if similar_services:
                    # Преобразуем результаты из find_similar_services в формат элементов из _clinic_data
                    service_ids = [s['serviceId'] for s in similar_services]
                    matching_services = [item for item in _clinic_data if item.get('serviceId') in service_ids]
                    logger.info(f"[CompareServicePriceInFilials] С помощью нечеткого поиска найдено {len(matching_services)} похожих услуг для '{self.service_name}'")
        
        # Если нет совпадений вообще, сообщаем об ошибке
        if not matching_services:
            logger.warning(f"[CompareServicePriceInFilials] Услуга '{self.service_name}' не найдена ни по точному соответствию, ни по подстроке, ни по нечеткому поиску")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        
        # Проверяем, содержит ли название запрашиваемой услуги специфические слова для фильтрации
        keyword_filters = {
            "колен": "колени",
            "голен": "голени",
            "бедр": "бедра",
            "рук": "руки"
        }
        
        filtered_services = matching_services
        used_filters = []
        
        # Применяем все подходящие фильтры к услугам
        for keyword, display_word in keyword_filters.items():
            if keyword in self.service_name.lower():
                keyword_services = [s for s in filtered_services if keyword in s.get('serviceName', '').lower()]
                if keyword_services:
                    filtered_services = keyword_services
                    used_filters.append(display_word)
                    logger.info(f"[CompareServicePriceInFilials] Фильтрация по '{keyword}': выбрано {len(filtered_services)} вариантов")
        
        # Если у нас больше одной похожей услуги
        if len(filtered_services) > 1:
            # В процессе записи автоматически выбираем наиболее подходящую услугу
            if self.in_booking_process:
                # Выбираем услугу с наибольшим количеством совпадающих ключевых слов
                max_keyword_matches = 0
                best_service = filtered_services[0]
                
                for service in filtered_services:
                    service_name = service.get('serviceName', '').lower()
                    query_words = self.service_name.lower().split()
                    
                    # Подсчитываем количество совпадающих слов
                    matches = sum(1 for word in query_words if word in service_name)
                    
                    if matches > max_keyword_matches:
                        max_keyword_matches = matches
                        best_service = service
                        
                logger.info(f"[CompareServicePriceInFilials] В процессе записи автоматически выбрана услуга '{best_service.get('serviceName')}' среди {len(filtered_services)} похожих вариантов")
                filtered_services = [best_service]
                display_service_name = best_service.get('serviceName') or self.service_name
            else:
                # При обычном запросе предлагаем выбор пользователю
                logger.info(f"[CompareServicePriceInFilials] Найдено {len(filtered_services)} похожих услуг для выбора пользователем")
                response_parts = [f"Для запроса '{self.service_name}' найдено {len(filtered_services)} похожих услуг:"]
                
                # Добавляем все похожие услуги в список для выбора
                for i, service in enumerate(filtered_services, 1):
                    response_parts.append(f"{i}. {service.get('serviceName')}")
                
                response_parts.append("\nПожалуйста, уточните, цены на какую именно услугу вы хотите сравнить в филиалах, указав её название или номер из списка.")
                return "\n".join(response_parts)
        
        # Если осталась только одна услуга после фильтрации
        if len(filtered_services) == 1:
            display_service_name = filtered_services[0].get('serviceName') or self.service_name
            logger.info(f"[CompareServicePriceInFilials] Используем '{display_service_name}' в качестве основной услуги для отображения")
        elif len(filtered_services) == 0:
            logger.warning(f"[CompareServicePriceInFilials] Услуга '{self.service_name}' не найдена после фильтрации")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."

        # Словарь для хранения {input_filial_name: display_name} или {input_filial_name: None} если не найден
        filial_resolution_map: Dict[str, Optional[str]] = {}
        unique_input_filial_names = sorted(list(set(f_name.strip() for f_name in self.filial_names if f_name.strip())), key=normalize_text)
        
        if len(unique_input_filial_names) < 2:
            return "Ошибка: Нужно указать как минимум два УНИКАЛЬНЫХ и непустых названия филиала для сравнения."

        invalid_filial_names_input: List[str] = []
        valid_filials: Dict[str, str] = {}  # {normalized_filial_name: display_name}

        # Поиск филиалов напрямую по названию
        for filial_name_input in unique_input_filial_names:
            normalized_filial_query = normalize_text(filial_name_input, keep_spaces=True).lower()
            filial_found = False
            
            # Ищем филиал по точному совпадению или подстроке
            for item in _clinic_data:
                filial_name = item.get("filialName")
                if not filial_name:
                    continue
                
                normalized_filial_name = normalize_text(filial_name, keep_spaces=True).lower()
                
                if normalized_filial_name == normalized_filial_query or normalized_filial_query in normalized_filial_name:
                    filial_found = True
                    filial_resolution_map[filial_name_input] = filial_name
                    valid_filials[normalized_filial_name] = filial_name
                    break
            
            if not filial_found:
                filial_resolution_map[filial_name_input] = None
                invalid_filial_names_input.append(filial_name_input)

        if invalid_filial_names_input:
            all_db_filial_names_set: Set[str] = set()
            for item_data in _clinic_data:
                fn = item_data.get('filialName')
                if fn: all_db_filial_names_set.add(fn)
            existing_filials_str = ", ".join(sorted(list(all_db_filial_names_set), key=normalize_text)) if all_db_filial_names_set else "(нет данных)"
            
            # Вынесем .join() в отдельную переменную
            joined_invalid_names = ", ".join(invalid_filial_names_input)
            return f"Следующие запрошенные филиалы не найдены: {joined_invalid_names}. Доступные филиалы в базе: {existing_filials_str}."
        
        # Проверка наличия как минимум двух различных филиалов
        if len(valid_filials) < 2:
            return "Нужно как минимум два РАЗЛИЧНЫХ корректных филиала для сравнения (после разрешения имен)."

        # Собираем цены для выбранных филиалов и услуги
        prices_by_filial: Dict[str, float] = {}  # {normalized_filial_name: price}
        
        # Для каждой услуги из filtered_services находим её цены в запрошенных филиалах
        for service_item in filtered_services:
            service_name = service_item.get('serviceName')
            
            for item in _clinic_data:
                if item.get('serviceId') != service_item.get('serviceId'):
                    continue
                    
                item_filial_name = item.get('filialName')
                if not item_filial_name:
                    continue
                    
                normalized_item_filial = normalize_text(item_filial_name, keep_spaces=True).lower()
                
                # Проверяем, соответствует ли филиал одному из запрошенных
                if normalized_item_filial in valid_filials:
                    price_raw = item.get('price')
                    if price_raw is not None and price_raw != '':
                        try:
                            price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                            # Если для этого филиала цена еще не записана, записываем
                            # (Одна услуга должна иметь одну цену в одном филиале)
                            if normalized_item_filial not in prices_by_filial:
                                prices_by_filial[normalized_item_filial] = price
                        except (ValueError, TypeError):
                            logger.warning(f"[CompareServicePriceInFilials] Не удалось сконвертировать цену '{price_raw}' для услуги '{service_name}' в филиале '{item_filial_name}'")
                            continue
        
        service_clarification = f" (уточнено до '{display_service_name}')" if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True) else ""
        if self.in_booking_process and normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True):
            service_clarification = f" (автоматически подобрана похожая услуга: '{display_service_name}')"
        
        response_parts = [f"Сравнение цен на услугу '{self.service_name}'{service_clarification}:"]
        
        found_prices_count = 0
        results_for_display: List[Tuple[str, Optional[float]]] = []
        
        # Преобразуем словарь {normalized_filial_name: price} в список кортежей (display_name, price)
        for normalized_name, price in prices_by_filial.items():
            display_name = valid_filials.get(normalized_name, "Неизвестный филиал")
            results_for_display.append((display_name, price))
        
        # Добавляем филиалы без цен
        for normalized_name, display_name in valid_filials.items():
            if normalized_name not in prices_by_filial:
                results_for_display.append((display_name, None))
        
        # Сортируем по имени филиала для консистентного вывода
        results_for_display.sort(key=lambda x: normalize_text(x[0]))

        for display_name, price_val in results_for_display:
            if price_val is not None:
                response_parts.append(f"- {display_name}: {price_val:.0f} руб.")
                found_prices_count += 1
            else:
                response_parts.append(f"- {display_name}: Цена не найдена.")
        
        if found_prices_count == 0:
            # Это сообщение должно учитывать, что мы уже проверили валидность филиалов
            valid_filial_names = ", ".join(sorted([name for name in valid_filials.values()], key=normalize_text))
            return f"Цена на услугу '{display_service_name}' не найдена ни в одном из запрошенных (и существующих) филиалов: {valid_filial_names}."
        
        # Ищем самую низкую цену только среди тех, где она была найдена
        actual_prices_list = [p for _, p in results_for_display if p is not None]
        if found_prices_count >= 2 and actual_prices_list:
            min_price = min(actual_prices_list) 
            cheapest_filials = sorted([name for name, p_val in results_for_display if p_val == min_price], key=normalize_text)
            if cheapest_filials: 
                response_parts.append(f"\nСамая низкая цена ({min_price:.0f} руб.) в: {", ".join(cheapest_filials)}.")
        elif found_prices_count == 1:
            response_parts.append("\nНедостаточно данных для сравнения (цена найдена только в одном из запрошенных филиалов).")
        # Если found_prices_count == 0, то предыдущий if уже сработал.

        return "\n".join(response_parts)


class FindServiceLocations(BaseModel):
    """Модель для поиска филиалов, где доступна услуга."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос происходит в процессе записи на прием")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[FindServiceLocations] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."
        
        logger.info(f"[FC Proc] Поиск филиалов для услуги: {self.service_name}, В процессе записи: {self.in_booking_process}")

        # Поиск услуг напрямую по названию без преобразования в ID
        normalized_query = normalize_text(self.service_name, keep_spaces=True).lower()
        display_service_name = self.service_name
        
        # Соберем все подходящие услуги по названию
        matching_services = []
        
        # Сначала ищем точные совпадения
        exact_matches = []
        for item in _clinic_data:
            service_name = item.get("serviceName")
            if not service_name:
                continue
                
            normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
            
            # Точное совпадение
            if normalized_service_name == normalized_query:
                exact_matches.append(item)
        
        # Если есть точные совпадения, используем их
        if exact_matches:
            matching_services = exact_matches
            logger.info(f"[FindServiceLocations] Найдено {len(matching_services)} точных совпадений для услуги '{self.service_name}'")
            display_service_name = exact_matches[0].get("serviceName") or self.service_name
        else:
            # Ищем услуги, содержащие запрос как подстроку
            substring_matches = []
            for item in _clinic_data:
                service_name = item.get("serviceName")
                if not service_name:
                    continue
                    
                normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
                
                # Совпадение по подстроке
                if normalized_query in normalized_service_name:
                    substring_matches.append(item)
            
            # Если есть совпадения по подстроке, используем их
            if substring_matches:
                matching_services = substring_matches
                logger.info(f"[FindServiceLocations] Найдено {len(matching_services)} совпадений по подстроке для услуги '{self.service_name}'")
                display_service_name = substring_matches[0].get("serviceName") or self.service_name
            else:
                # Если нет ни точного соответствия, ни по подстроке, используем нечеткий поиск
                from service_disambiguation import find_similar_services
                similar_services = find_similar_services(_tenant_id_for_clinic_data, self.service_name, _clinic_data)
                
                if similar_services:
                    # Преобразуем результаты из find_similar_services в формат элементов из _clinic_data
                    service_ids = [s['serviceId'] for s in similar_services]
                    matching_services = [item for item in _clinic_data if item.get('serviceId') in service_ids]
                    logger.info(f"[FindServiceLocations] С помощью нечеткого поиска найдено {len(matching_services)} похожих услуг для '{self.service_name}'")
        
        # Если нет совпадений вообще, сообщаем об ошибке
        if not matching_services:
            logger.warning(f"[FindServiceLocations] Услуга '{self.service_name}' не найдена ни по точному соответствию, ни по подстроке, ни по нечеткому поиску")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        
        # Проверяем, содержит ли название запрашиваемой услуги специфические слова для фильтрации
        keyword_filters = {
            "колен": "колени",
            "голен": "голени",
            "бедр": "бедра",
            "рук": "руки"
        }
        
        filtered_services = matching_services
        used_filters = []
        
        # Применяем все подходящие фильтры к услугам
        for keyword, display_word in keyword_filters.items():
            if keyword in self.service_name.lower():
                keyword_services = [s for s in filtered_services if keyword in s.get('serviceName', '').lower()]
                if keyword_services:
                    filtered_services = keyword_services
                    used_filters.append(display_word)
                    logger.info(f"[FindServiceLocations] Фильтрация по '{keyword}': выбрано {len(filtered_services)} вариантов")
        
        # Если у нас больше одной похожей услуги
        if len(filtered_services) > 1:
            # В процессе записи автоматически выбираем наиболее подходящую услугу
            if self.in_booking_process:
                # Выбираем услугу с наибольшим количеством совпадающих ключевых слов
                max_keyword_matches = 0
                best_service = filtered_services[0]
                
                for service in filtered_services:
                    service_name = service.get('serviceName', '').lower()
                    query_words = self.service_name.lower().split()
                    
                    # Подсчитываем количество совпадающих слов
                    matches = sum(1 for word in query_words if word in service_name)
                    
                    if matches > max_keyword_matches:
                        max_keyword_matches = matches
                        best_service = service
                        
                logger.info(f"[FindServiceLocations] В процессе записи автоматически выбрана услуга '{best_service.get('serviceName')}' среди {len(filtered_services)} похожих вариантов")
                filtered_services = [best_service]
                display_service_name = best_service.get('serviceName') or self.service_name
            else:
                # При обычном запросе предлагаем выбор пользователю
                logger.info(f"[FindServiceLocations] Найдено {len(filtered_services)} похожих услуг для выбора пользователем")
                response_parts = [f"Для запроса '{self.service_name}' найдено {len(filtered_services)} похожих услуг:"]
                
                # Добавляем все похожие услуги в список для выбора
                for i, service in enumerate(filtered_services, 1):
                    response_parts.append(f"{i}. {service.get('serviceName')}")
                
                response_parts.append("\nПожалуйста, уточните, расположение какой именно услуги вы хотите найти, указав её название или номер из списка.")
                return "\n".join(response_parts)
        
        # Если осталась только одна услуга после фильтрации
        if len(filtered_services) == 1:
            display_service_name = filtered_services[0].get('serviceName') or self.service_name
            logger.info(f"[FindServiceLocations] Используем '{display_service_name}' в качестве основной услуги для отображения")
        elif len(filtered_services) == 0:
            logger.warning(f"[FindServiceLocations] Услуга '{self.service_name}' не найдена после фильтрации")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."

        # Собираем все филиалы, где доступна указанная услуга или похожие услуги
        locations: Dict[str, str] = {}  # norm_filial_name -> original_filial_name
        
        for service in filtered_services:
            service_name = service.get('serviceName')
            
            # Для каждой услуги находим все филиалы, где она доступна
            for item in _clinic_data:
                item_service_name = item.get('serviceName')
                filial_name = item.get('filialName')
                
                # Пропускаем записи без filialName или serviceName
                if not item_service_name or not filial_name:
                    continue
                
                # Проверяем, соответствует ли услуга одной из найденных
                if item_service_name == service_name:
                    norm_filial_name = normalize_text(filial_name)
                    if norm_filial_name not in locations:
                        locations[norm_filial_name] = filial_name

        # Формируем результат
        if not locations:
            return f"Услуга '{display_service_name}' найдена, но информация о филиалах отсутствует."
        else:
            sorted_original_filials = sorted(locations.values(), key=normalize_text)
            
            # Если исходное название услуги отличается от найденного, уточним это в ответе
            service_clarification = ""
            if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True):
                if self.in_booking_process:
                    service_clarification = f" (автоматически подобрана похожая услуга: '{display_service_name}')"
                else:
                    service_clarification = f" (найдена похожая услуга: '{display_service_name}')"
            
            result_intro = f"Услуга '{self.service_name}'{service_clarification} доступна в филиалах:"
            return f"{result_intro}\n*   " + "\n*   ".join(sorted_original_filials)


class FindSpecialistsByServiceOrCategoryAndFilial(BaseModel):
    """Модель для поиска специалистов по услуге/категории и филиалу."""
    query_term: str = Field(description="Название услуги ИЛИ категории")
    filial_name: str = Field(description="Точное название филиала")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=15, description="Количество специалистов на странице")

    def process(self) -> str:
        if not self.query_term or not self.filial_name: return "Ошибка: Укажите услугу/категорию и филиал."
        if not _clinic_data: return "Ошибка: Данные клиники не загружены."
        if not _tenant_id_for_clinic_data:
            logger.error("[FindSpecialistsByServiceOrCategoryAndFilial] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Поиск специалистов (Запрос: '{self.query_term}', Филиал: '{self.filial_name}'), Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
        if not filial_id:
            return f"Филиал с названием, похожим на '{self.filial_name}', не найден."
        
        original_filial_display_name = get_name_by_id(_tenant_id_for_clinic_data, 'filial', filial_id) or self.filial_name

        service_id_match = get_id_by_name(_tenant_id_for_clinic_data, 'service', self.query_term)
        category_id_match = None
        query_type = ""
        resolved_query_term_name = self.query_term

        if service_id_match:
            query_type = "service"
            logger.info(f"Запрос '{self.query_term}' распознан как услуга с ID: {service_id_match}")
            resolved_query_term_name = get_name_by_id(_tenant_id_for_clinic_data, 'service', service_id_match) or self.query_term
        else:
            category_id_match = get_id_by_name(_tenant_id_for_clinic_data, 'category', self.query_term)
            if category_id_match:
                query_type = "category"
                logger.info(f"Запрос '{self.query_term}' распознан как категория с ID: {category_id_match}")
                resolved_query_term_name = get_name_by_id(_tenant_id_for_clinic_data, 'category', category_id_match) or self.query_term
            else:
                return f"Термин '{self.query_term}' не распознан ни как услуга, ни как категория."

        matching_employees: Dict[str, str] = {}
        for item in _clinic_data:
            if item.get("filialId") != filial_id:
                continue

            emp_id = item.get("employeeId")
            emp_name_raw = item.get("employeeFullName")
            if not emp_id or not emp_name_raw: continue

            match_found = False
            if query_type == "service" and item.get("serviceId") == service_id_match:
                match_found = True
            elif query_type == "category" and item.get("categoryId") == category_id_match:
                match_found = True
            
            if match_found:
                if emp_id not in matching_employees:
                    matching_employees[emp_id] = emp_name_raw # Сохраняем оригинальное имя

        if not matching_employees:
            return f"В филиале '{original_filial_display_name}' не найдено специалистов для '{resolved_query_term_name}'."
        else:
            sorted_employee_names = sorted(list(matching_employees.values()), key=lambda n: normalize_text(n, keep_spaces=True, sort_words=True))
            total_specialists = len(sorted_employee_names)

            start_index = (self.page_number - 1) * self.page_size
            end_index = start_index + self.page_size
            paginated_specialist_names = sorted_employee_names[start_index:end_index]

            if not paginated_specialist_names and self.page_number > 1:
                return f"В филиале '{original_filial_display_name}' для '{resolved_query_term_name}' больше нет специалистов для отображения (страница {self.page_number})."
            
            page_info = f" (страница {self.page_number} из {(total_specialists + self.page_size - 1) // self.page_size if self.page_size > 0 else 1})" if total_specialists > self.page_size else ""
            
            clarified_query_term = f" (уточнено до '{resolved_query_term_name}')" if normalize_text(resolved_query_term_name, keep_spaces=True) != normalize_text(self.query_term, keep_spaces=True) else ""
            clarified_filial_name = f" (уточнено до '{original_filial_display_name}')" if normalize_text(original_filial_display_name) != normalize_text(self.filial_name) else ""

            response_intro = f"В филиале '{self.filial_name}'{clarified_filial_name} для '{self.query_term}'{clarified_query_term} найдены специалисты{page_info}:"
            response_list = "\n- " + "\n- ".join(paginated_specialist_names)

            more_info = ""
            if end_index < total_specialists:
                remaining = total_specialists - end_index
                more_info = f"\n... и еще {remaining} {get_readable_count_form(remaining, ('специалист', 'специалиста', 'специалистов'))}."

            return f"{response_intro}{response_list}{more_info}"


class ListServicesInCategory(BaseModel):
    """Модель для получения списка услуг в конкретной категории."""
    category_name: str = Field(description="Точное название категории")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество услуг на странице")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[ListServicesInCategory] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос услуг в категории: {self.category_name}, Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        category_id = get_id_by_name(_tenant_id_for_clinic_data, 'category', self.category_name)
        if not category_id:
            return f"Категория с названием, похожим на '{self.category_name}', не найдена."

        display_category_name = get_name_by_id(_tenant_id_for_clinic_data, 'category', category_id) or self.category_name

        services_in_category: Set[str] = set()
        for item in _clinic_data:
            if item.get('categoryId') == category_id:
                srv_name_raw = item.get('serviceName')
                if srv_name_raw: services_in_category.add(srv_name_raw)

        if not services_in_category:
            return f"В категории '{display_category_name}' не найдено конкретных услуг."
        else:
            sorted_services = sorted(list(services_in_category), key=lambda s: normalize_text(s, keep_spaces=True))
            total_services = len(sorted_services)

            start_index = (self.page_number - 1) * self.page_size
            end_index = start_index + self.page_size
            paginated_services = sorted_services[start_index:end_index]

            if not paginated_services and self.page_number > 1:
                return f"В категории '{display_category_name}' больше нет услуг для отображения (страница {self.page_number})."

            clarification = f" (уточнено до '{display_category_name}')" if normalize_text(display_category_name, keep_spaces=True) != normalize_text(self.category_name, keep_spaces=True) else ""
            page_info = f" (страница {self.page_number} из {(total_services + self.page_size - 1) // self.page_size if self.page_size > 0 else 1})" if total_services > self.page_size else ""
            
            response_intro = f"В категорию '{self.category_name}'{clarification} входят следующие услуги{page_info}:"
            response_list = "\n- " + "\n- ".join(paginated_services)
            
            more_info = ""
            if end_index < total_services:
                remaining = total_services - end_index
                more_info = f"\n... и еще {remaining} {get_readable_count_form(remaining, ('услуга', 'услуги', 'услуг'))}."
            
            return f"{response_intro}{response_list}{more_info}"


class ListServicesInFilial(BaseModel):
    """Модель для получения списка всех услуг в конкретном филиале."""
    filial_name: str = Field(description="Точное название филиала")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=30, description="Количество услуг на странице")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[ListServicesInFilial] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос всех услуг в филиале: {self.filial_name}, Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
        if not filial_id:
            all_filials_db_orig: Set[str] = set()
            if _clinic_data:
                for item_data in _clinic_data:
                    f_name_val = item_data.get('filialName')
                    if f_name_val: all_filials_db_orig.add(f_name_val)
            suggestion = f"Доступные филиалы: {', '.join(sorted(list(all_filials_db_orig), key=normalize_text))}." if all_filials_db_orig else "Список филиалов в базе пуст."
            return f"Филиал '{self.filial_name}' не найден. {suggestion}".strip()

        display_filial_name = get_name_by_id(_tenant_id_for_clinic_data, 'filial', filial_id) or self.filial_name

        services_in_filial: Set[str] = set()
        for item in _clinic_data:
            if item.get('filialId') == filial_id:
                srv_name_raw = item.get('serviceName')
                if srv_name_raw: services_in_filial.add(srv_name_raw)

        if not services_in_filial:
            return f"В филиале '{display_filial_name}' не найдено информации об услугах."
        else:
            sorted_services = sorted(list(services_in_filial), key=lambda s: normalize_text(s, keep_spaces=True))
            
            start_index = (self.page_number - 1) * self.page_size
            end_index = start_index + self.page_size
            output_services = sorted_services[start_index:end_index]

            if not output_services and self.page_number > 1:
                return f"В филиале '{display_filial_name}' больше нет услуг для отображения (страница {self.page_number})."

            total_services = len(sorted_services)
            more_services_info = ""
            if end_index < total_services:
                remaining_services = total_services - end_index
                more_services_info = f"... и еще {remaining_services} услуг."
            
            return (f"В филиале '{display_filial_name}' доступны услуги (страница {self.page_number}):\\n* "
                   + "\\n* ".join(output_services) + f"\\n{more_services_info}".strip())


class FindServicesInPriceRange(BaseModel):
    """Модель для поиска услуг в заданном ценовом диапазоне."""
    min_price: float = Field(description="Минимальная цена")
    max_price: float = Field(description="Максимальная цена")
    category_name: Optional[str] = Field(default=None, description="Опционально: категория")
    filial_name: Optional[str] = Field(default=None, description="Опционально: филиал")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество услуг на странице")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[FindServicesInPriceRange] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Поиск услуг в диапазоне цен: {self.min_price}-{self.max_price}, Категория: {self.category_name}, Филиал: {self.filial_name}, Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        target_category_id: Optional[str] = None
        display_category_name_query = self.category_name
        if self.category_name:
            target_category_id = get_id_by_name(_tenant_id_for_clinic_data, 'category', self.category_name)
            if not target_category_id:
                return f"Категория '{self.category_name}' не найдена."
            display_category_name_query = get_name_by_id(_tenant_id_for_clinic_data, 'category', target_category_id) or self.category_name

        target_filial_id: Optional[str] = None
        display_filial_name_query = self.filial_name
        if self.filial_name:
            target_filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
            if not target_filial_id:
                return f"Филиал '{self.filial_name}' не найден."
            display_filial_name_query = get_name_by_id(_tenant_id_for_clinic_data, 'filial', target_filial_id) or self.filial_name

        found_services: List[Dict[str, Any]] = [] # Список словарей для каждой найденной услуги
        processed_service_ids: Set[str] = set() # Для избежания дублирования услуг с разными ценами в одном филиале, если такое возможно

        for item in _clinic_data:
            service_id = item.get('serviceId')
            if not service_id or service_id in processed_service_ids: continue

            if target_category_id and item.get('categoryId') != target_category_id:
                continue
            if target_filial_id and item.get('filialId') != target_filial_id:
                continue
            
            price_raw = item.get('price')
            if price_raw is None or price_raw == '': continue
            try:
                price = float(str(price_raw).replace(' ', '').replace(',', '.'))
            except (ValueError, TypeError):
                continue

            if self.min_price <= price <= self.max_price:
                service_name_raw = item.get('serviceName')
                filial_name_raw = item.get('filialName', "(не указан)") # Берем из данных, если есть
                category_name_raw = item.get('categoryName', "(не указана)") # Берем из данных, если есть
                
                # Если фильтр по филиалу был, но в данных он отсутствует, то имя филиала будет из запроса (display_filial_name_query)
                # Аналогично для категории.
                effective_filial_name = display_filial_name_query if target_filial_id else filial_name_raw
                effective_category_name = display_category_name_query if target_category_id else category_name_raw

                found_services.append({
                    'name': service_name_raw,
                    'price': price,
                    'filial': effective_filial_name,
                    'category': effective_category_name,
                    'id': service_id # Для сортировки и потенциальной дедупликации, если нужно
                })
                processed_service_ids.add(service_id)
        
        if not found_services:
            return f"Услуги в ценовом диапазоне {self.min_price}-{self.max_price} руб. (с учетом фильтров) не найдены."

        # Сортируем по цене, затем по имени услуги
        found_services.sort(key=lambda x: (x['price'], normalize_text(x['name'], keep_spaces=True)))
        total_found = len(found_services)

        start_index = (self.page_number - 1) * self.page_size
        end_index = start_index + self.page_size
        paginated_services = found_services[start_index:end_index]

        if not paginated_services and self.page_number > 1:
            return f"Больше нет услуг в указанном ценовом диапазоне для отображения (страница {self.page_number})."

        response_parts = []
        price_range_str = f"от {self.min_price:.0f} до {self.max_price:.0f} руб."
        filters_applied_list = []
        if display_category_name_query: filters_applied_list.append(f"категория '{display_category_name_query}'")
        if display_filial_name_query: filters_applied_list.append(f"филиал '{display_filial_name_query}'")
        filters_str = f" (фильтры: { ', '.join(filters_applied_list)})" if filters_applied_list else ""
        
        page_info = f" (страница {self.page_number} из {(total_found + self.page_size - 1) // self.page_size if self.page_size > 0 else 1})" if total_found > self.page_size else ""

        response_parts.append(f"Найдены услуги в диапазоне {price_range_str}{filters_str}{page_info}:")
        
        for service in paginated_services:
            detail = f"- {service['name']} - {service['price']:.0f} руб."
            location_info = []
            if not target_filial_id and service['filial'] != "(не указан)": # Показываем филиал, если не было фильтра по нему
                location_info.append(f"Филиал: {service['filial']}")
            if not target_category_id and service['category'] != "(не указана)": # Показываем категорию, если не было фильтра по ней
                location_info.append(f"Категория: {service['category']}")
            if location_info:
                detail += f" ({ '; '.join(location_info) })"
            response_parts.append(detail)

        if end_index < total_found:
            remaining = total_found - end_index
            response_parts.append(f"... и еще {remaining} {get_readable_count_form(remaining, ('услуга', 'услуги', 'услуг'))}. Введите 'дальше', чтобы увидеть их.")
        
        return "\n".join(response_parts)


class ListAllCategories(BaseModel):
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=30, description="Количество категорий на странице")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[ListAllCategories] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос всех категорий, Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        all_categories: Set[str] = set()
        for item in _clinic_data:
            cat_name = item.get('categoryName')
            if cat_name: all_categories.add(cat_name)
        
        if not all_categories:
            return "Категории услуг не найдены в базе данных."
        else:
            sorted_categories = sorted(list(all_categories), key=lambda c: normalize_text(c, keep_spaces=True))
            total_categories = len(sorted_categories)

            start_index = (self.page_number - 1) * self.page_size
            end_index = start_index + self.page_size
            paginated_categories = sorted_categories[start_index:end_index]

            if not paginated_categories and self.page_number > 1:
                return f"Больше нет категорий для отображения (страница {self.page_number})."

            page_info = f" (страница {self.page_number} из {(total_categories + self.page_size - 1) // self.page_size if self.page_size > 0 else 1})" if total_categories > self.page_size else ""
            
            response_intro = f"Доступные категории услуг{page_info}:"
            response_list = "\n- " + "\n- ".join(paginated_categories)
            
            more_info = ""
            if end_index < total_categories:
                remaining = total_categories - end_index
                more_info = f"\n... и еще {remaining} {get_readable_count_form(remaining, ('категория', 'категории', 'категорий'))}."

            return f"{response_intro}{response_list}{more_info}"

class ListEmployeeFilials(BaseModel):
    """Модель для получения списка филиалов конкретного сотрудника."""
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")

    def process(self) -> str:
        """Возвращает список всех филиалов, где работает сотрудник."""
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[ListEmployeeFilials] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос филиалов сотрудника: {self.employee_name}, Tenant: {_tenant_id_for_clinic_data}")

        employee_id = get_id_by_name(_tenant_id_for_clinic_data, 'employee', self.employee_name)
        if not employee_id:
            return f"Сотрудник с именем, похожим на '{self.employee_name}', не найден."

        display_employee_name = get_name_by_id(_tenant_id_for_clinic_data, 'employee', employee_id) or self.employee_name

        found_filial_ids: Set[str] = set()
        for item in _clinic_data:
            if item.get('employeeId') == employee_id:
                filial_id_from_item = item.get('filialId')
                if filial_id_from_item: found_filial_ids.add(filial_id_from_item)

        if not found_filial_ids:
            return f"Для сотрудника '{display_employee_name}' не найдено информации о филиалах."
        else:
            name_clarification = ""
            if normalize_text(display_employee_name, keep_spaces=True) != normalize_text(self.employee_name, keep_spaces=True):
                name_clarification = f" (найдено по запросу '{self.employee_name}')"

            # Преобразуем ID филиалов в их имена для вывода
            filial_names_for_output: List[str] = []
            for f_id in found_filial_ids:
                f_name = get_name_by_id(_tenant_id_for_clinic_data, 'filial', f_id)
                if f_name: filial_names_for_output.append(f_name)
                else: filial_names_for_output.append(f"(ID: {f_id})") # На случай, если имя не найдено
            
            sorted_filials = sorted(filial_names_for_output, key=normalize_text)

            if len(sorted_filials) == 1:
                 return f"Сотрудник {display_employee_name}{name_clarification} работает в филиале: {sorted_filials[0]}."
            else:
                 return f"Сотрудник {display_employee_name}{name_clarification} работает в следующих филиалах:\n*   " + "\n*   ".join(sorted_filials)

class GetFreeSlotsArgs(BaseModel):
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    employee_name: str = Field(description="ФИО сотрудника (точно или частично)")
    service_names: List[str] = Field(description="Список названий услуг (точно)")
    date_time: Optional[str] = Field(default=None, description="Дата для поиска слотов (формат YYYY-MM-DD)")
    filial_name: str = Field(description="Название филиала (точно)")
    lang_id: str = Field(default="ru", description="Язык ответа")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")

class GetFreeSlots(BaseModel):
    tenant_id: str
    employee_id: str
    service_ids: List[str]
    date_time: str
    filial_id: str
    lang_id: str = "ru"
    api_token: Optional[str] = None

    async def process(self, **kwargs) -> str:
        if not self.date_time:
            self.date_time = datetime.now().strftime("%Y.%m.%d")
            logger.info(f"Дата не указана, используется текущая: {self.date_time}")
        
        # Проверка корректности ID услуг и сотрудника
        if not self.employee_id:
            return "Не удалось определить ID сотрудника. Пожалуйста, проверьте правильность имени сотрудника."
        
        if not self.service_ids or not all(self.service_ids):
            return "Не удалось определить ID услуг. Пожалуйста, проверьте правильность названий услуг."
        
        if not self.filial_id:
            return "Не удалось определить ID филиала. Пожалуйста, проверьте правильность названия филиала."
        
        # Проверка соответствия услуг и филиала
        import service_disambiguation
        valid_services_for_filial = []
        invalid_services = []
        
        # МОДИФИКАЦИЯ: Пропускаем локальную проверку доступности услуги в филиале
        # и позволяем API решить, доступна ли услуга
        
        # Сохраняем оригинальный код в комментарии для ссылки
        # for service_id in self.service_ids:
        #     if service_disambiguation.verify_service_in_filial(self.tenant_id, service_id, self.filial_id, _clinic_data):
        #         valid_services_for_filial.append(service_id)
        #     else:
        #         service_name = get_name_by_id(self.tenant_id, 'service', service_id) or f"ID:{service_id}"
        #         invalid_services.append(service_name)
        
        # if invalid_services:
        #     filial_name = get_name_by_id(self.tenant_id, 'filial', self.filial_id) or f"ID:{self.filial_id}"
        #     invalid_list = ", ".join(invalid_services)
        #     return f"Следующие услуги недоступны в филиале '{filial_name}': {invalid_list}. Пожалуйста, выберите другие услуги или другой филиал."
        
        # МОДИФИКАЦИЯ: Считаем все услуги валидными и передаем их в API
        valid_services_for_filial = self.service_ids
        
        logger.info(f"Запрос свободных слотов: TenantID={self.tenant_id}, EmployeeID={self.employee_id}, ServiceIDs={self.service_ids}, Date={self.date_time}, FilialID={self.filial_id}")
        try:
            # Добавляем более подробное логирование перед вызовом API
            logger.info(f"Выполняем вызов API get_free_times_of_employee_by_services с параметрами: employeeId={self.employee_id}, serviceId={self.service_ids}, dateTime={self.date_time}, filialId={self.filial_id}")
            
            response_data = await get_free_times_of_employee_by_services(
                tenant_id=self.tenant_id,
                employee_id=self.employee_id,
                service_ids=self.service_ids,
                date_time=self.date_time,
                filial_id=self.filial_id,
                api_token=self.api_token
            )

            if response_data and response_data.get('code') == 200:
                api_data_content = response_data.get('data')

                if isinstance(api_data_content, dict) and 'workDates' in api_data_content:
                    all_formatted_slots = []
                    employee_name_original = api_data_content.get('name', self.employee_id)
                    
                    filial_name_original = get_name_by_id(self.tenant_id, 'filial', self.filial_id) or self.filial_id

                    for work_day in api_data_content.get('workDates', []):
                        date_str = work_day.get('date')
                        time_slots_for_day = work_day.get('timeSlots', [])
                        
                        if date_str and time_slots_for_day:
                            try:
                                formatted_date_str = datetime.strptime(date_str, "%Y.%m.%d").strftime("%d.%m.%Y")
                            except ValueError:
                                formatted_date_str = date_str 
                            
                            day_slots_str = f"На {formatted_date_str}: {', '.join(time_slots_for_day)}"
                            all_formatted_slots.append(day_slots_str)
                    
                    if not all_formatted_slots:
                        logger.info(f"Не найдено слотов в 'workDates' для сотрудника {employee_name_original} ({self.employee_id}) на {self.date_time} в филиале {filial_name_original} ({self.filial_id}).")
                        return f"К сожалению, у сотрудника {employee_name_original} нет свободных слотов на указанную дату в филиале {filial_name_original} по выбранным услугам."
                    
                    response_message = f"Доступные слоты для сотрудника {employee_name_original} в филиале {filial_name_original}:\n" + "\n".join(all_formatted_slots)
                    return response_message

                elif isinstance(api_data_content, list):
                    logger.info("API вернуло 'data' как список (старый формат). Обработка...")
                    processed_slots = []
                    if not api_data_content: 
                         logger.info(f"API вернуло пустой список в 'data' для сотрудника {self.employee_id} на {self.date_time} в филиале {self.filial_id}.")
                         return f"К сожалению, свободных слотов не найдено."

                    for slot_info in api_data_content:
                        if isinstance(slot_info, dict) and 'time' in slot_info:
                            processed_slots.append(slot_info['time'])
                    
                    if not processed_slots:
                        logger.info(f"Не найдено ключей 'time' в элементах списка 'data' от API для {self.employee_id}.")
                        return "Свободные слоты не найдены (не удалось обработать ответ API)."
                    
                   
                    employee_name_display = get_name_by_id(self.tenant_id, 'employee', self.employee_id) or self.employee_id
                    filial_name_display = get_name_by_id(self.tenant_id, 'filial', self.filial_id) or self.filial_id
                    date_display = self.date_time 

                    return f"Для сотрудника {employee_name_display} в филиале {filial_name_display} на {date_display} доступны следующие слоты: {', '.join(processed_slots)}."

                else: # Неожиданный формат data
                    logger.warning(f"Ожидался список или словарь с 'workDates' для 'data' от API, но получен {type(api_data_content)}. Содержимое: {str(api_data_content)[:500]}. Слоты не будут обработаны.")
                   
                    employee_name_from_data = "неизвестного сотрудника"
                    if isinstance(api_data_content, dict) and 'name' in api_data_content:
                        employee_name_from_data = api_data_content['name']
                    
                    filial_name_display = get_name_by_id(self.tenant_id, 'filial', self.filial_id) or self.filial_id
                    return f"Для сотрудника {employee_name_from_data} в филиале {filial_name_display} на {self.date_time} не найдено свободных слотов, или формат данных от API не распознан."

            elif response_data:
                error_msg = response_data.get('message', 'Неизвестная ошибка от API')
                logger.error(f"Ошибка от API при запросе свободных слотов (код {response_data.get('code')}): {error_msg}. Параметры: emp={self.employee_id}, fil={self.filial_id}, date={self.date_time}")
                return f"Не удалось получить свободные слоты: {error_msg}"
            else:
                logger.error(f"Нет ответа от API при запросе свободных слотов для emp={self.employee_id}, fil={self.filial_id}, date={self.date_time}")
                return "Не удалось связаться с системой записи для получения свободных слотов."

        except Exception as e:
            logger.error(f"Исключение в GetFreeSlots.process для emp={self.employee_id}, fil={self.filial_id}, date={self.date_time}: {e}", exc_info=True)
            return f"Произошла внутренняя ошибка при поиске свободных слотов: {e}"

class BookAppointment(BaseModel):
    tenant_id: str
    phone_number: str
    service_id: str
    employee_id: str
    filial_id: str
    category_id: str
    service_original_name: str
    date_of_record: str
    start_time: str
    end_time: str
    duration_of_time: int
    lang_id: str = "ru"
    api_token: Optional[str] = None
    price: float = 0
    sale_price: float = 0
    complex_service_id: str = ""
    color_code_record: str = ""
    total_price: float = 0
    traffic_channel: int = Field(default=0, description="Канал трафика (опционально)")
    traffic_channel_id: str = Field(default="", description="ID канала трафика (опционально)")

    async def process(self, **kwargs) -> str:
        try:
            if not self.service_id:
                return f"ID услуги не предоставлен."
            if not self.employee_id:
                return f"ID сотрудника не предоставлен."
            if not self.filial_id:
                return f"ID филиала не предоставлен."
            if not self.category_id:
                return f"ID категории не предоставлен."

            result = await add_record(
                tenant_id=self.tenant_id,
                phone_number=self.phone_number,
                service_id=self.service_id,
                employee_id=self.employee_id,
                filial_id=self.filial_id,
                category_id=self.category_id,
                service_original_name=self.service_original_name,
                date_of_record=self.date_of_record,
                start_time=self.start_time,
                end_time=self.end_time,
                duration_of_time=self.duration_of_time,
                lang_id=self.lang_id,
                api_token=self.api_token,
                price=self.price,
                sale_price=self.sale_price,
                complex_service_id=self.complex_service_id,
                color_code_record=self.color_code_record,
                total_price=self.total_price,
                traffic_channel=self.traffic_channel,
                traffic_channel_id=self.traffic_channel_id
            )
            if result.get('code') == 200:
                return "Запись успешно создана!"
            else:
                return f"Ошибка при создании записи: {result.get('message', 'Неизвестная ошибка')}"
        except Exception as e:
            logger.error(f"Ошибка в BookAppointment.process для tenant '{self.tenant_id}', service_id '{self.service_id}': {e}", exc_info=True)
            return f"Ошибка при создании записи: {type(e).__name__} - {e}"

class BookAppointmentAIPayload(BaseModel):
    lang_id: str = "ru"
    client_phone_number: str
    # services_payload будет List[ServiceDetailItemFromLLM] приходить из matrixai.py
    # но для вызова add_record мы его конвертируем в List[Dict[str, Any]]
    services_payload: List[Any] # Используем List[Any] для гибкости, конвертация ниже
    filial_id: str
    date_of_record: str
    start_time: str
    end_time: str
    duration_of_time: int
    to_employee_id: str
    total_price: float
    api_token: Optional[str] = None
    color_code_record: Optional[str] = None
    traffic_channel: Optional[int] = None
    traffic_channel_id: Optional[str] = None
    tenant_id: str # <--- ДОБАВЛЕНО ПОЛЕ tenant_id

    async def process(self, **kwargs) -> str:
        try:
            # Конвертируем List[ServiceDetailItemFromLLM] в List[Dict[str, Any]]
            # если services_payload содержит Pydantic модели
            services_as_dicts = []
            for item in self.services_payload:
                if hasattr(item, 'model_dump'): # Проверяем, является ли элемент Pydantic моделью
                    services_as_dicts.append(item.model_dump())
                elif isinstance(item, dict): # Если это уже словарь
                    services_as_dicts.append(item)
                else:
                    # Обработка неожиданного типа элемента, если необходимо
                    logger.error(f"Неожиданный тип элемента в services_payload: {type(item)}")
                    # Можно либо пропустить, либо вызвать ошибку
                    raise ValueError(f"Элемент в services_payload должен быть Pydantic моделью или словарем, получен: {type(item)}")

            result = await add_record(
                api_token=self.api_token,
                tenant_id=self.tenant_id,  # <--- Используем self.tenant_id
                client_phone_number=self.client_phone_number,
                services_payload=services_as_dicts, # <--- Передаем конвертированный список словарей
                filial_id=self.filial_id,
                date_of_record=self.date_of_record,
                start_time=self.start_time,
                end_time=self.end_time,
                duration_of_time=self.duration_of_time,
                to_employee_id=self.to_employee_id,
                total_price=self.total_price,
                lang_id=self.lang_id,
                color_code_record=self.color_code_record,
                traffic_channel=self.traffic_channel,
                traffic_channel_id=self.traffic_channel_id
            )
            
            if result.get('code') == 200:
                return "Запись успешно создана!"
            else:
                return f"Ошибка при создании записи: {result.get('message', 'Неизвестная ошибка')}"
        except Exception as e:
            logger.error(f"Ошибка в BookAppointmentAIPayload.process: {e}", exc_info=True)
            return f"Ошибка при создании записи (AI Payload): {type(e).__name__} - {e}"

