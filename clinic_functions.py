#clinic_functions.py

import logging
import re
from typing import Optional, List, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field
from client_data_service import (
    get_free_times_of_employee_by_services,
    add_record,
    get_multiple_data_from_api,
    update_record_time,
    get_cancel_reasons,
    cancel_record,
)
from clinic_index import get_name_by_id, normalize_text, get_id_by_name, get_service_id_by_name
import asyncio
from datetime import datetime, timedelta

_clinic_data: List[Dict[str, Any]] = []
_tenant_id_for_clinic_data: Optional[str] = None
logger = logging.getLogger(__name__)

def smart_name_match(query_name: str, db_name: str) -> bool:
    """
    Умное сравнение имен, учитывающее разный порядок слов и различия в капитализации.
    
    Args:
        query_name: Имя из запроса (например, "Сеферова Соня Магамедовна")
        db_name: Имя из базы данных (например, "Соня Сеферова Магамедовна")
    
    Returns:
        bool: True, если имена соответствуют
    """
    if not query_name or not db_name:
        return False
        
    
    normalized_query = normalize_text(query_name, keep_spaces=False, sort_words=True)
    normalized_db = normalize_text(db_name, keep_spaces=False, sort_words=True)
    
    
    if normalized_query == normalized_db:
        return True
    
    
    query_words = set(normalize_text(query_name, keep_spaces=True).lower().split())
    db_words = set(normalize_text(db_name, keep_spaces=True).lower().split())
    
    
    query_words = {word for word in query_words if len(word) > 1}
    db_words = {word for word in db_words if len(word) > 1}
    
   
    if query_words and query_words.issubset(db_words):
        return True
    
    
    if query_words and db_words:
        intersection = query_words.intersection(db_words)
        coverage = len(intersection) / len(query_words)
        return coverage >= 0.7
    
    return False

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

def get_original_filial_name(normalized_name: str) -> Optional[str]:
    """Находит оригинальное название филиала по его нормализованному имени."""
    if not normalized_name or not _clinic_data: 
        return None
    for item in _clinic_data:
        original_name = item.get("filialName")
        if original_name and normalize_text(original_name) == normalized_name:
            return original_name
    return None


class FindEmployees(BaseModel):
    """
    Модель для поиска сотрудников по различным критериям (унифицированный API с 6 режимами).
    
    6 режимов работы в зависимости от переданных параметров:
    1. Только filial_name → Все сотрудники в филиале
    2. Только service_name → Все сотрудники, оказывающие услугу (независимо от филиала)
    3. Только employee_name → Все услуги сотрудника (независимо от филиала)
    4. filial_name + employee_name → Услуги сотрудника в конкретном филиале
    5. filial_name + service_name → Сотрудники, оказывающие услугу в конкретном филиале
    6. service_name + employee_name → Филиалы, где сотрудник оказывает конкретную услугу
    
    ВАЖНО: Возвращает категории услуг вместо конкретных услуг для лучшей навигации.
    """
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")
    page_number: int = Field(default=1, description="Номер страницы результатов")
    page_size: int = Field(default=15, description="Количество результатов на странице")

    async def process(self, tenant_id: str, api_token: str) -> str: # <-- Added tenant_id and api_token
        logger.info(f"[FindEmployees Proc] Поиск сотрудников (Сотрудник: '{self.employee_name}', Услуга: '{self.service_name}', Филиал: '{self.filial_name}'), Tenant: {tenant_id}, Page: {self.page_number}, Size: {self.page_size}")

        # Get IDs for filters
        employee_id_query = get_id_by_name(tenant_id, 'employee', self.employee_name) if self.employee_name else None
        service_id_query = get_id_by_name(tenant_id, 'service', self.service_name) if self.service_name else None
        filial_id_query = get_id_by_name(tenant_id, 'filial', self.filial_name) if self.filial_name else None

        if self.employee_name and not employee_id_query:
            return f"Сотрудник с именем '{self.employee_name}' не найден."
        if self.service_name and not service_id_query:
             return f"Услуга с названием '{self.service_name}' не найдена."
        if self.filial_name and not filial_id_query:
             return f"Филиал с названием '{self.filial_name}' не найден."

        from client_data_service import get_multiple_data_from_api
        try:
            api_data = await get_multiple_data_from_api(
                api_token=api_token,
                filial_id=filial_id_query,
                employee_id=employee_id_query,
                service_id=service_id_query,
                tenant_id=tenant_id 
            )
        except Exception as e:
            logger.error(f"[FindEmployees Proc] Ошибка вызова get_multiple_data_from_api: {e}", exc_info=True)
            return f"Ошибка при получении данных из актуальной базы: {str(e)}"

        if not api_data:
            not_found_message = "Данные не найдены"
            filters_used_parts = []
            if self.employee_name: filters_used_parts.append(f"для сотрудника '{self.employee_name}'")
            if self.service_name: filters_used_parts.append(f"по услуге '{self.service_name}'")
            if self.filial_name: filters_used_parts.append(f"в филиале '{self.filial_name}'")

            if filters_used_parts:
                not_found_message = f"В актуальной базе данных не найдено информации, соответствующей запросу: {', '.join(filters_used_parts)}."
            else:
                not_found_message = "В актуальной базе данных не найдено информации, соответствующей вашему запросу."
            
        
            if (self.employee_name and employee_id_query and not self.service_name):
                 if self.filial_name and filial_id_query: # Mode 4
                      not_found_message = f"Сотрудник '{self.employee_name}' не оказывает услуг в филиале '{self.filial_name}' согласно данным из API."
                 else: 
                      not_found_message = f"Для сотрудника '{self.employee_name}' не найдено услуг в данных из API."

            logger.info(f"[FindEmployees Proc] API вернул пустой список или нерелевантные данные. Ответ: {not_found_message}")
            return not_found_message

        is_mode_3_employee_services = bool(self.employee_name and not self.service_name and not self.filial_name and employee_id_query)
        is_mode_4_employee_services_in_filial = bool(self.employee_name and self.filial_name and not self.service_name and employee_id_query and filial_id_query)
        is_mode_5_service_employees_in_filial = bool(self.filial_name and self.service_name and not self.employee_name and filial_id_query and service_id_query)
        is_mode_6_employee_service_filials = bool(self.service_name and self.employee_name and not self.filial_name and service_id_query and employee_id_query)

        if is_mode_3_employee_services or is_mode_4_employee_services_in_filial:
            employee_name_display = get_name_by_id(tenant_id, 'employee', employee_id_query) or self.employee_name
            filial_name_display_mode4 = None
            if is_mode_4_employee_services_in_filial and filial_id_query:
                filial_name_display_mode4 = get_name_by_id(tenant_id, 'filial', filial_id_query) or self.filial_name
            services_by_filial: Dict[str, List[str]] = {}
            
            for item in api_data:
                service_name_item = item.get('serviceName')
                filial_id_item = item.get('filialId')
                
                if is_mode_4_employee_services_in_filial and filial_id_item != filial_id_query:
                    continue
                
                # Get filial name by ID
                filial_name_item = get_name_by_id(tenant_id, 'filial', filial_id_item) if filial_id_item else None
                
                if service_name_item and filial_name_item:
                    if filial_name_item not in services_by_filial:
                        services_by_filial[filial_name_item] = []
                    if service_name_item not in services_by_filial[filial_name_item]:
                        services_by_filial[filial_name_item].append(service_name_item)

            # Sort services within each filial
            for filial_name in services_by_filial:
                services_by_filial[filial_name].sort(key=normalize_text)

            # Create display items for pagination
            display_items = []
            sorted_filial_names = sorted(services_by_filial.keys(), key=normalize_text)
            
            if is_mode_4_employee_services_in_filial:
                # Mode 4: just list services directly
                # Find the correct filial name key in services_by_filial (it comes from API)
                target_services = []
                for filial_name_from_api in services_by_filial:
                    # Compare normalized names to handle potential differences
                    if (normalize_text(filial_name_from_api) == normalize_text(filial_name_display_mode4) or
                        filial_name_from_api == filial_name_display_mode4):
                        target_services.extend(services_by_filial[filial_name_from_api])
                        break
                
                for service_name in target_services:
                    display_items.append({'type': 'service', 'name': service_name})
            else:
                # Mode 3: group by filials with headers
                for filial_name in sorted_filial_names:
                    display_items.append({'type': 'filial_header', 'name': filial_name})
                    for service_name in services_by_filial[filial_name]:
                        display_items.append({'type': 'service', 'name': service_name})

            if not display_items:
                if is_mode_4_employee_services_in_filial:
                    return f"Сотрудник '{employee_name_display}' не оказывает услуг в филиале '{filial_name_display_mode4}'."
                return f"Не найдено услуг для сотрудника '{employee_name_display}'."

            # Pagination
            total_items = len(display_items)
            start_idx = (self.page_number - 1) * self.page_size
            end_idx = start_idx + self.page_size
            paginated_items = display_items[start_idx:end_idx]

            if not paginated_items and self.page_number > 1:
                max_pages = (total_items + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
                return f"Страница {self.page_number} не найдена. Доступно страниц: {max_pages} для услуг сотрудника '{employee_name_display}'."

            # Format response
            response_parts = []
            if is_mode_4_employee_services_in_filial:
                response_parts.append(f"Сотрудник {employee_name_display} в филиале '{filial_name_display_mode4}' оказывает следующие услуги:")
            else:
                response_parts.append(f"Сотрудник {employee_name_display} оказывает следующие услуги:")

            for item in paginated_items:
                if item['type'] == 'filial_header':
                    response_parts.append(f"\nВ филиале \"{item['name']}\":")
                elif item['type'] == 'service':
                    indent = "  - " if is_mode_4_employee_services_in_filial else "    - "
                    response_parts.append(f"{indent}{item['name']}")

            # Add pagination info for services
            total_services = sum(1 for item in display_items if item['type'] == 'service')
            shown_services = sum(1 for item in paginated_items if item['type'] == 'service')
            
            if total_items > self.page_size or self.page_number > 1:
                max_pages = (total_items + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
                pagination_note = f"Показано услуг на этой странице: {shown_services} (всего найдено услуг: {total_services}). Страница {self.page_number} из {max_pages}."
                if end_idx < total_items:
                    pagination_note += f" Для следующей страницы укажите page_number={self.page_number + 1}."
                response_parts.append(f"\n{pagination_note}")

            return "\n".join(response_parts)

        # Mode 5: service + filial -> employees who do this service in this filial
        if is_mode_5_service_employees_in_filial:
            service_name_display = get_name_by_id(tenant_id, 'service', service_id_query) or self.service_name
            filial_name_display = get_name_by_id(tenant_id, 'filial', filial_id_query) or self.filial_name
            
            employees_set = set()
            for item in api_data:
                emp_name = item.get('employeeFullname')
                if emp_name:
                    employees_set.add(emp_name)
            
            employees_list = sorted(list(employees_set))
            
            if not employees_list:
                return f"В филиале '{filial_name_display}' не найдено сотрудников, оказывающих услугу '{service_name_display}'."
            
            # Pagination for employees
            total_employees = len(employees_list)
            start_idx = (self.page_number - 1) * self.page_size
            end_idx = start_idx + self.page_size
            paginated_employees = employees_list[start_idx:end_idx]
            
            if not paginated_employees and self.page_number > 1:
                max_pages = (total_employees + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
                return f"Страница {self.page_number} не найдена. Доступно страниц: {max_pages} для сотрудников, оказывающих услугу '{service_name_display}' в филиале '{filial_name_display}'."
            
            response_parts = []
            response_parts.append(f"В филиале '{filial_name_display}' услугу '{service_name_display}' оказывают следующие сотрудники:")
            
            for emp_name in paginated_employees:
                response_parts.append(f"  - {emp_name}")
            
            # Add pagination info
            if total_employees > self.page_size or self.page_number > 1:
                max_pages = (total_employees + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
                pagination_note = f"Показано сотрудников на этой странице: {len(paginated_employees)} (всего найдено: {total_employees}). Страница {self.page_number} из {max_pages}."
                if end_idx < total_employees:
                    pagination_note += f" Для следующей страницы укажите page_number={self.page_number + 1}."
                response_parts.append(f"\n{pagination_note}")
            
            return "\n".join(response_parts)

        # Mode 6: service + employee -> filials where this employee does this service
        if is_mode_6_employee_service_filials:
            service_name_display = get_name_by_id(tenant_id, 'service', service_id_query) or self.service_name
            employee_name_display = get_name_by_id(tenant_id, 'employee', employee_id_query) or self.employee_name
            
            filials_set = set()
            for item in api_data:
                filial_id_item = item.get('filialId')
                if filial_id_item:
                    filial_name_item = get_name_by_id(tenant_id, 'filial', filial_id_item)
                    if filial_name_item:
                        filials_set.add(filial_name_item)
            
            filials_list = sorted(list(filials_set))
            
            if not filials_list:
                return f"Сотрудник '{employee_name_display}' не оказывает услугу '{service_name_display}' ни в одном филиале."
            
            # Pagination for filials
            total_filials = len(filials_list)
            start_idx = (self.page_number - 1) * self.page_size
            end_idx = start_idx + self.page_size
            paginated_filials = filials_list[start_idx:end_idx]
            
            if not paginated_filials and self.page_number > 1:
                max_pages = (total_filials + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
                return f"Страница {self.page_number} не найдена. Доступно страниц: {max_pages} для филиалов, где сотрудник '{employee_name_display}' оказывает услугу '{service_name_display}'."
            
            response_parts = []
            response_parts.append(f"Сотрудник '{employee_name_display}' оказывает услугу '{service_name_display}' в следующих филиалах:")
            
            for filial_name in paginated_filials:
                response_parts.append(f"  - {filial_name}")
            
            # Add pagination info
            if total_filials > self.page_size or self.page_number > 1:
                max_pages = (total_filials + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
                pagination_note = f"Показано филиалов на этой странице: {len(paginated_filials)} (всего найдено: {total_filials}). Страница {self.page_number} из {max_pages}."
                if end_idx < total_filials:
                    pagination_note += f" Для следующей страницы укажите page_number={self.page_number + 1}."
                response_parts.append(f"\n{pagination_note}")
            
            return "\n".join(response_parts)

        # Process the data received from the API for other modes (listing employees)
        # The API returns a list of dictionaries, where each dict seems to be a service provided by an employee in a filial.
        # We need to aggregate this by employee.
        employees_info: Dict[str, Dict[str, Any]] = {}
        
        for item in api_data:
            emp_id = item.get('employeeId')
            emp_name = item.get('employeeFullname')  # Fixed: API returns 'employeeFullname' (lowercase 'n')
            service_name = item.get('serviceName')
            filial_id = item.get('filialId')
            filial_name = item.get('filialName')
            category_name = item.get('categoryName') # Assuming API returns category name

            if not emp_id or not emp_name:
                continue

            if emp_id not in employees_info:
                employees_info[emp_id] = {
                    'id': emp_id,
                    'name': emp_name,
                    'filials': {}, # filial_id -> filial_name
                    'services_by_filial': {}, # filial_id -> {service_name -> category_name}
                }

            if filial_id and filial_name:
                employees_info[emp_id]['filials'][filial_id] = filial_name
                if filial_id not in employees_info[emp_id]['services_by_filial']:
                    employees_info[emp_id]['services_by_filial'][filial_id] = {}
                if service_name:
                    # Associate service with category under the specific filial
                    employees_info[emp_id]['services_by_filial'][filial_id][service_name] = category_name or "Без категории"

        # Aggregate employees into a list, maintaining the dict structure for each employee
        employee_list_data = list(employees_info.values())
        
        # Sort employees by name
        sorted_employees = sorted(
            employee_list_data,
            key=lambda x: normalize_text(x.get('name', ''), keep_spaces=True) # Use normalized name for sorting
        )
        
        total_found = len(sorted_employees)
        start_idx = (self.page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        paginated_employees = sorted_employees[start_idx:end_idx]

        if not paginated_employees and self.page_number > 1:
            max_pages = (total_found + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
            return f"Страница {self.page_number} не найдена. Доступно страниц: {max_pages}. Всего найдено сотрудников: {total_found}"

        response_parts = []
        
        # --- Улучшенная вводная часть ---
        intro_parts = []
        if self.filial_name: intro_parts.append(f"В филиале '{self.filial_name}'")
        intro_parts.append("найдены следующие сотрудники:")

        response_parts.append(" ".join(intro_parts))
        # --- Конец улучшенной вводной части ---

        
        # --- Форматирование каждого сотрудника ---
        for emp_data in paginated_employees:
            emp_name = emp_data['name']
            
            # Collect all categories for the employee based on the fetched data
            all_categories_for_employee = set()
            for filial_services in emp_data['services_by_filial'].values():
                 all_categories_for_employee.update(filial_services.values())
            
            sorted_categories = sorted(list(all_categories_for_employee), key=normalize_text)
            
            if sorted_categories:
                categories_str = ", ".join(sorted_categories)
                emp_info_line = f"- {emp_name} (Категории: {categories_str})"
            else:
                emp_info_line = f"- {emp_name}"
            
            response_parts.append(emp_info_line)
        # --- Конец форматирования каждого сотрудника ---

        # Add pagination info
        if total_found > self.page_size:
             max_pages = (total_found + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
             response_parts.append(f"\nПоказано {len(paginated_employees)} из {total_found} сотрудников (страница {self.page_number}/{max_pages}). Используйте page_number={self.page_number + 1} для следующей страницы.")

        return "\n".join(response_parts)


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

        # Используем fuzzy search из clinic_index для поиска услуги
        service_id = get_service_id_by_name(_tenant_id_for_clinic_data, self.service_name)
        
        if not service_id:
            logger.warning(f"[GetServicePrice] Услуга '{self.service_name}' не найдена через fuzzy search")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        
        logger.info(f"[GetServicePrice] Найдена услуга через fuzzy search: service_id={service_id}")
        
        # Нормализуем запрос филиала для фильтрации цен
        normalized_filial_query = normalize_text(self.filial_name, keep_spaces=True).lower() if self.filial_name else None
        
        # Находим все записи с этим service_id
        matching_services = []
        for item in _clinic_data:
            if item.get("serviceId") == service_id:
                matching_services.append(item)
        
        # Применяем фильтрацию по ключевым словам
        keyword_filters = {
            "колен": "колени",
            "голен": "голени", 
            "бедр": "бедра",
            "рук": "руки",
            "лиц": "лицо",
            "шеи": "шея",
            "ягодиц": "ягодицы",
            "спин": "спина",
            "живот": "живот"
        }
        
        filtered_services = matching_services
        for keyword, display_word in keyword_filters.items():
            if keyword in self.service_name.lower():
                keyword_services = [s for s in filtered_services if keyword in s.get('serviceName', '').lower()]
                if keyword_services:
                    filtered_services = keyword_services
                    logger.info(f"[GetServicePrice] Фильтрация по '{keyword}': выбрано {len(filtered_services)} вариантов")
        
        # Обработка множественных результатов
        if len(filtered_services) > 1:
            if self.in_booking_process:
                # В процессе записи выбираем наиболее подходящую услугу
                best_service = self._select_best_service(filtered_services)
                filtered_services = [best_service]
                logger.info(f"[GetServicePrice] Автоматически выбрана услуга '{best_service.get('serviceName')}'")
            else:
                # При обычном запросе предлагаем выбор
                return self._format_service_selection_for_price(filtered_services)
        
        # Собираем цены
        prices_by_filial = {}  # normalized_filial_name -> (price, display_name)
        selected_service_names = {s.get('serviceName') for s in filtered_services}
        
        for item in _clinic_data:
            item_service_name = item.get('serviceName')
            item_filial_name = item.get('filialName')
            
            if (item_service_name in selected_service_names and 
                item_filial_name):
                
                normalized_filial = normalize_text(item_filial_name, keep_spaces=True).lower()
                
                # Если указан конкретный филиал, собираем цены только для него
                if normalized_filial_query and normalized_filial != normalized_filial_query and normalized_filial_query not in normalized_filial:
                    continue
                
                price_raw = item.get('price')
                if price_raw is not None and price_raw != '':
                    try:
                        price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                        if normalized_filial not in prices_by_filial:
                            prices_by_filial[normalized_filial] = (price, item_filial_name)
                    except (ValueError, TypeError):
                        logger.warning(f"[GetServicePrice] Некорректная цена '{price_raw}'")
                        continue
        
        # Формируем результат
        display_service_name = filtered_services[0].get('serviceName') or self.service_name
        service_clarification = ""
        
        if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True):
            if self.in_booking_process:
                service_clarification = f" (автоматически подобрана похожая услуга: '{display_service_name}')"
            else:
                service_clarification = f" (уточнено до '{display_service_name}')"
        
        if not prices_by_filial:
            if self.filial_name:
                return f"Цена на услугу '{display_service_name}' в филиале '{self.filial_name}' не найдена."
            else:
                return f"Цена на услугу '{display_service_name}' не найдена ни в одном филиале."
        
        # Если указан конкретный филиал
        if self.filial_name:
            if len(prices_by_filial) == 1:
                price, filial_display_name = list(prices_by_filial.values())[0]
                return f"Цена на услугу '{self.service_name}'{service_clarification} в филиале '{filial_display_name}': {price:.0f} руб."
            else:
                # Показываем все цены в найденных филиалах
                sorted_prices = sorted(prices_by_filial.items(), key=lambda x: x[1][1])
                response_parts = [f"Цена на услугу '{self.service_name}'{service_clarification} в подходящих филиалах:"]
                for _, (price, filial_display_name) in sorted_prices:
                    response_parts.append(f"- {filial_display_name}: {price:.0f} руб.")
                return "\n".join(response_parts)
        
        # Если филиал не указан, показываем цены во всех доступных филиалах
        sorted_prices = sorted(prices_by_filial.items(), key=lambda x: normalize_text(x[1][1]))
        
        if len(sorted_prices) == 1:
            price, filial_display_name = list(sorted_prices)[0][1]
            return f"Цена на услугу '{self.service_name}'{service_clarification}: {price:.0f} руб. (филиал: {filial_display_name})"
        else:
            response_parts = [f"Цена на услугу '{self.service_name}'{service_clarification}:"]
            
            for _, (price, filial_display_name) in sorted_prices:
                response_parts.append(f"- {filial_display_name}: {price:.0f} руб.")
            
            # Находим самую низкую цену
            all_prices = [price for _, (price, _) in sorted_prices]
            min_price = min(all_prices)
            cheapest_filials = [filial_name for _, (price, filial_name) in sorted_prices if price == min_price]
            
            if len(cheapest_filials) == 1:
                response_parts.append(f"\nСамая низкая цена ({min_price:.0f} руб.) в филиале: {cheapest_filials[0]}.")
            else:
                cheapest_names = ", ".join(sorted(cheapest_filials, key=normalize_text))
                response_parts.append(f"\nСамая низкая цена ({min_price:.0f} руб.) в филиалах: {cheapest_names}.")
            
            return "\n".join(response_parts)

    def _select_best_service(self, services):
        """Выбирает наиболее подходящую услугу из списка."""
        from difflib import SequenceMatcher
        
        query_lower = self.service_name.lower()
        best_score = -1
        best_service = services[0]
        
        for service in services:
            service_name = service.get('serviceName', '').lower()
            score = 0
            
            # Совпадение слов
            query_words = query_lower.split()
            word_matches = sum(1 for word in query_words if word in service_name)
            score += word_matches * 5
            
            # Общее совпадение строки
            similarity = SequenceMatcher(None, query_lower, service_name).ratio()
            score += similarity * 10
            
            if score > best_score:
                best_score = score
                best_service = service
                
        return best_service
    
    def _format_service_selection_for_price(self, services):
        """Форматирует список услуг для выбора при запросе цены."""
        response_parts = [f"Для запроса '{self.service_name}' найдено {len(services)} похожих услуг:"]
        
        for i, service in enumerate(services, 1):
            response_parts.append(f"{i}. {service.get('serviceName')}")
        
        response_parts.append("\nПожалуйста, уточните, цену на какую именно услугу вы хотите узнать, указав её название или номер из списка.")
        return "\n".join(response_parts)


class CheckServiceInFilial(BaseModel):
    """Модель для проверки наличия услуги в филиале."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос происходит в процессе записи на прием")

    async def process(self, tenant_id: str, api_token: str) -> str: # <-- Added tenant_id and api_token
        logger.info(f"[CheckServiceInFilial Proc] Проверка услуги '{self.service_name}' в филиале '{self.filial_name}', Tenant: {tenant_id}")

        # Get IDs first
        service_id_query = get_id_by_name(tenant_id, 'service', self.service_name)
        filial_id_query = get_id_by_name(tenant_id, 'filial', self.filial_name)

        # If names were provided but IDs not found
        if not service_id_query:
            return f"Услуга с названием '{self.service_name}' не найдена."
        if not filial_id_query:
            return f"Филиал с названием '{self.filial_name}' не найден."

        # Call the new API endpoint with both serviceId and filialId
        from client_data_service import get_multiple_data_from_api
        api_data = await get_multiple_data_from_api(
             api_token=api_token,
             service_id=service_id_query,
             filial_id=filial_id_query,
             tenant_id=tenant_id # Pass for logging
        )

        # If api_data is not empty, it means the combination exists
        if api_data:
            # Assuming the API returns at least one entry if the combination is valid
            # Get actual names from API data if available, fallback to original query names
            actual_service_name = api_data[0].get('serviceName', self.service_name)
            actual_filial_name = api_data[0].get('filialName', self.filial_name)
            logger.info(f"[CheckServiceInFilial Proc] Услуга '{actual_service_name}' (ID: {service_id_query}) найдена в филиале '{actual_filial_name}' (ID: {filial_id_query}).")
            return f"Услуга '{actual_service_name}' доступна в филиале '{actual_filial_name}'."
        else:
            # If api_data is empty, the combination does not exist in the API data.
            logger.info(f"[CheckServiceInFilial Proc] Услуга '{self.service_name}' (ID: {service_id_query}) не найдена в филиале '{self.filial_name}' (ID: {filial_id_query}) через актуальную базу данных. Пытаемся найти, где она доступна.")
            
            # If not available in the requested filial, try to find where it *is* available
            # Call the API again, this time only filtering by serviceId
            api_data_for_service = await get_multiple_data_from_api(
                 api_token=api_token,
                 service_id=service_id_query,
                 tenant_id=tenant_id # Pass for logging
            )

            available_filials = set() # Use a set to store unique filial names
            service_name_from_api = self.service_name

            if api_data_for_service:
                 for item in api_data_for_service:
                      item_filial_name = item.get('filialName')
                      if item_filial_name:
                           available_filials.add(item_filial_name)
                      # Use the service name from the first item in the list if available
                      if not service_name_from_api and item.get('serviceName'):
                           service_name_from_api = item.get('serviceName')

            if available_filials:
                 filial_list_str = ', '.join(sorted(list(available_filials), key=normalize_text))
                 return f"Услуга '{service_name_from_api}' недоступна в филиале '{self.filial_name}'. Она доступна в следующих филиалах: {filial_list_str}."
            else:
                 # Service ID was found, but no filial data returned from the API
                 # This could indicate data inconsistency or that the service is simply not offered anywhere according to the API
                 return f"Услуга '{service_name_from_api}' недоступна в филиале '{self.filial_name}'. Данные о ее доступности в других филиалах отсутствуют в актуальной базе данных."


class FindServiceLocations(BaseModel):
    """Модель для поиска филиалов, где доступна услуга."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос происходит в процессе записи на прием")

    async def process(self, tenant_id: str, api_token: str) -> str: # <-- Added tenant_id and api_token
        logger.info(f"[FindServiceLocations Proc] Поиск филиалов для услуги '{self.service_name}', Tenant: {tenant_id}")

        # Get service ID first
        service_id_query = get_id_by_name(tenant_id, 'service', self.service_name)
        if not service_id_query:
             return f"Услуга с названием '{self.service_name}' не найдена."

        # Call the new API endpoint filtering by serviceId
        from client_data_service import get_multiple_data_from_api
        api_data = await get_multiple_data_from_api(
             api_token=api_token,
             service_id=service_id_query,
             tenant_id=tenant_id # Pass for logging
        )

        if not api_data:
            # Service ID was found, but API returned no data for it.
            # This implies the service is not associated with any employee/filial in the live data.
            return f"Услуга '{self.service_name}' найдена, но данные о филиалах, где она доступна, отсутствуют в актуальной базе данных."

        # Extract unique filial names from the API response
        available_filials = set() # Use a set for unique names
        actual_service_name = self.service_name # Default to query name

        for item in api_data:
            filial_name = item.get('filialName')
            if filial_name:
                available_filials.add(filial_name)
            # Capture the actual service name from the API data if available
            if not actual_service_name and item.get('serviceName'):
                actual_service_name = item.get('serviceName')

        if available_filials:
             sorted_filial_names = sorted(list(available_filials), key=normalize_text)
             filial_list_str = ', '.join(sorted_filial_names)
             # Use the service name captured from API data if available
             return f"Услуга '{actual_service_name}' доступна в следующих филиалах: {filial_list_str}."
        else:
             # Should not happen if api_data was not empty, but as a fallback
             return f"Услуга '{actual_service_name}' найдена, но данные о филиалах, где она доступна, отсутствуют в актуальной базе данных."


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

        # Нормализованные запросы для поиска
        normalized_service_query = normalize_text(self.service_name, keep_spaces=True).lower()
        normalized_filial_queries = [normalize_text(name, keep_spaces=True).lower() for name in self.filial_names]
        
        # Найденные услуги по приоритету совпадения
        exact_matches = []
        substring_matches = []
        
        # Поиск услуг напрямую в данных
        for item in _clinic_data:
            service_name = item.get("serviceName")
            if not service_name:
                continue
            
            normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
            
            # Точное совпадение
            if normalized_service_name == normalized_service_query:
                exact_matches.append(item)
            # Совпадение по подстроке
            elif normalized_service_query in normalized_service_name:
                substring_matches.append(item)
        
        # Выбираем лучшие совпадения
        matching_services = exact_matches if exact_matches else substring_matches
        
        if not matching_services:
            logger.warning(f"[CompareServicePriceInFilials] Услуга '{self.service_name}' не найдена")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        
        # Если много совпадений, выбираем лучшую в процессе записи
        if len(matching_services) > 1 and self.in_booking_process:
            best_service = self._select_best_service(matching_services)
            matching_services = [best_service]
        
        # Собираем цены по филиалам
        prices_by_filial = {}  # normalized_filial_name -> (price, display_name, found)
        selected_service_names = {s.get('serviceName') for s in matching_services}
        
        # Инициализируем все запрошенные филиалы
        for i, filial_name in enumerate(self.filial_names):
            normalized = normalized_filial_queries[i]
            prices_by_filial[normalized] = (None, filial_name, False)
        
        # Ищем цены в данных
        for item in _clinic_data:
            item_service_name = item.get('serviceName')
            item_filial_name = item.get('filialName')
            
            if (item_service_name in selected_service_names and item_filial_name):
                normalized_filial = normalize_text(item_filial_name, keep_spaces=True).lower()
                
                # Проверяем, соответствует ли один из запрошенных филиалов
                for norm_query in normalized_filial_queries:
                    if (normalized_filial == norm_query or 
                        norm_query in normalized_filial):
                        
                        price_raw = item.get('price')
                        if price_raw is not None and price_raw != '':
                            try:
                                price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                                prices_by_filial[norm_query] = (price, item_filial_name, True)
                                break
                            except (ValueError, TypeError):
                                logger.warning(f"[CompareServicePriceInFilials] Некорректная цена '{price_raw}'")
                                continue
        
        # Формируем результат
        display_service_name = matching_services[0].get('serviceName') or self.service_name
        service_clarification = ""
        
        if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True):
            if self.in_booking_process:
                service_clarification = f" (автоматически подобрана похожая услуга: '{display_service_name}')"
            else:
                service_clarification = f" (уточнено до '{display_service_name}')"
        
        response_parts = [f"Сравнение цен на услугу '{self.service_name}'{service_clarification}:"]
        
        found_prices = []
        missing_filials = []
        
        for norm_query, (price, display_name, found) in prices_by_filial.items():
            if found and price is not None:
                response_parts.append(f"- {display_name}: {price:.0f} руб.")
                found_prices.append((price, display_name))
            else:
                missing_filials.append(display_name)
        
        if missing_filials:
            response_parts.append(f"\nЦена не найдена в филиалах: {', '.join(missing_filials)}")
        
        if found_prices:
            if len(found_prices) > 1:
                min_price = min(found_prices, key=lambda x: x[0])
                max_price = max(found_prices, key=lambda x: x[0])
                
                if min_price[0] != max_price[0]:
                    response_parts.append(f"\nСамая низкая цена: {min_price[0]:.0f} руб. в '{min_price[1]}'")
                    response_parts.append(f"Самая высокая цена: {max_price[0]:.0f} руб. в '{max_price[1]}'")
                    savings = max_price[0] - min_price[0]
                    response_parts.append(f"Экономия: {savings:.0f} руб.")
        
        return "\n".join(response_parts)
    
    def _select_best_service(self, services):
        """Выбирает наиболее подходящую услугу из списка."""
        from difflib import SequenceMatcher
        
        query_lower = self.service_name.lower()
        best_score = -1
        best_service = services[0]
        
        for service in services:
            service_name = service.get('serviceName', '').lower()
            score = 0
            
            # Совпадение слов
            query_words = query_lower.split()
            word_matches = sum(1 for word in query_words if word in service_name)
            score += word_matches * 5
            
            # Общее совпадение строки
            similarity = SequenceMatcher(None, query_lower, service_name).ratio()
            score += similarity * 10
            
            if score > best_score:
                best_score = score
                best_service = service
                
        return best_service




class ListFilials(BaseModel):
    """Модель для получения списка всех филиалов."""
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество филиалов на странице")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[ListFilials] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос списка филиалов, Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        all_filials_orig: Set[str] = set()
        for item_data in _clinic_data:
            f_name_val = item_data.get('filialName')
            if f_name_val: 
                all_filials_orig.add(f_name_val)

        if not all_filials_orig:
            return "Список филиалов в базе пуст."

        sorted_filials = sorted(list(all_filials_orig), key=normalize_text)
        total_filials = len(sorted_filials)
        
        start_index = (self.page_number - 1) * self.page_size
        end_index = start_index + self.page_size
        paginated_filials = sorted_filials[start_index:end_index]

        if not paginated_filials and self.page_number > 1:
            return f"Больше нет филиалов для отображения (страница {self.page_number})."

        page_info = f" (страница {self.page_number} из {(total_filials + self.page_size - 1) // self.page_size if self.page_size > 0 else 1})" if total_filials > self.page_size else ""
        
        response_intro = f"Доступные филиалы{page_info}:"
        response_list = "\n- " + "\n- ".join(paginated_filials)
        
        more_info = ""
        if end_index < total_filials:
            remaining = total_filials - end_index
            more_info = f"\n... и еще {remaining} {get_readable_count_form(remaining, ('филиал', 'филиала', 'филиалов'))}."
        
        return f"{response_intro}{response_list}{more_info}"


class GetSlots(BaseModel):
    """Модель для получения доступных слотов времени."""
    employee_name: str = Field(description="ФИО сотрудника")
    service_name: str = Field(description="Название услуги")
    filial_name: str = Field(description="Название филиала")
    from_date: Optional[str] = Field(default=None, description="Дата начала поиска (YYYY-MM-DD)")
    to_date: Optional[str] = Field(default=None, description="Дата окончания поиска (YYYY-MM-DD)")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[GetSlots] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Получение слотов (Сотрудник: {self.employee_name}, Услуга: {self.service_name}, Филиал: {self.filial_name}), Tenant: {_tenant_id_for_clinic_data}")

        # Получаем ID через системные функции
        employee_id = get_id_by_name(_tenant_id_for_clinic_data, 'employee', self.employee_name)
        service_id = get_id_by_name(_tenant_id_for_clinic_data, 'service', self.service_name)
        filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)

        if not employee_id:
            return f"Сотрудник '{self.employee_name}' не найден."
        if not service_id:
            return f"Услуга '{self.service_name}' не найдена."
        if not filial_id:
            return f"Филиал '{self.filial_name}' не найден."

        # Проверяем, что сотрудник выполняет эту услугу в данном филиале
        employee_provides_service = False
        for item in _clinic_data:
            if (item.get('employeeId') == employee_id and 
                item.get('serviceId') == service_id and 
                item.get('filialId') == filial_id):
                employee_provides_service = True
                break

        if not employee_provides_service:
            return f"Сотрудник '{self.employee_name}' не выполняет услугу '{self.service_name}' в филиале '{self.filial_name}'."

        try:
            slots_data = asyncio.run(get_free_times_of_employee_by_services(
                _tenant_id_for_clinic_data, employee_id, [service_id], 
                self.from_date, self.to_date,
                filial_id, self.lang_id, api_token=self.api_token
            ))
            response_parts = self._format_slots(slots_data)
            return '\n'.join(response_parts)
        except Exception as e:
            logger.error(f"[GetSlots] Ошибка при получении слотов: {e}")
            return f"Ошибка при получении доступных слотов: {str(e)}"

    def _format_slots(self, slots_data):
        if not slots_data or not slots_data.get('result'):
            return ["Нет доступных слотов для записи."]
        
        result = slots_data['result']
        response_parts = ["Доступные слоты для записи:"]
        
        for date_slot in result:
            date = date_slot.get('date', 'Неизвестная дата')
            slots = date_slot.get('slots', [])
            
            if slots:
                slot_times = [slot.get('time', 'Неизвестное время') for slot in slots]
                response_parts.append(f"\n{date}: {', '.join(slot_times)}")
        
        return response_parts


class BookAppointment(BaseModel):
    """Модель для записи на прием."""
    employee_name: str = Field(description="ФИО сотрудника")
    service_name: str = Field(description="Название услуги")
    filial_name: str = Field(description="Название филиала")
    client_name: str = Field(description="ФИО клиента")
    client_phone: str = Field(description="Телефон клиента")
    appointment_date: str = Field(description="Дата записи (YYYY-MM-DD)")
    appointment_time: str = Field(description="Время записи (HH:MM)")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[BookAppointment] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Бронирование (Сотрудник: {self.employee_name}, Услуга: {self.service_name}, Филиал: {self.filial_name}, Клиент: {self.client_name}), Tenant: {_tenant_id_for_clinic_data}")

        # Получаем ID через системные функции
        employee_id = get_id_by_name(_tenant_id_for_clinic_data, 'employee', self.employee_name)
        service_id = get_id_by_name(_tenant_id_for_clinic_data, 'service', self.service_name)
        filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)

        if not employee_id:
            return f"Сотрудник '{self.employee_name}' не найден."
        if not service_id:
            return f"Услуга '{self.service_name}' не найдена."
        if not filial_id:
            return f"Филиал '{self.filial_name}' не найден."

        try:
            booking_result = asyncio.run(add_record(
                _tenant_id_for_clinic_data, employee_id, service_id, filial_id,
                self.client_name, self.client_phone, 
                self.date_of_record, self.start_time, self.end_time, self.duration_of_time,
                self.lang_id, api_token=self.api_token
            ))
            if booking_result.get('success'):
                return f"Запись успешно создана!"
            else:
                error_msg = booking_result.get('error', 'Неизвестная ошибка')
                return f"Ошибка при создании записи: {error_msg}"
        except Exception as e:
            logger.error(f"[BookAppointment] Ошибка при записи: {e}")
            return f"Ошибка при создании записи: {str(e)}"


# Добавим остальные функции, которые не требуют изменений...
class ListServicesInFilial(BaseModel):
    """Модель для получения списка всех услуг в конкретном филиале."""
    filial_name: str = Field(description="Точное название филиала")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=30, description="Количество услуг на странице")

    async def process(self, tenant_id: str, api_token: str) -> str:
        logger.info(f"[ListServicesInFilial Proc] Получение списка услуг в филиале '{self.filial_name}', Tenant: {tenant_id}, Page: {self.page_number}, Size: {self.page_size}")

        # Get filial ID first (mandatory)
        filial_id_query = get_id_by_name(tenant_id, 'filial', self.filial_name)
        if not filial_id_query:
            return f"Филиал с названием '{self.filial_name}' не найден."
        
        actual_filial_name = get_name_by_id(tenant_id, 'filial', filial_id_query) or self.filial_name

        # Try new API endpoint first for getting services by categories
        from client_data_service import get_filial_services_by_categories
        categories_data = await get_filial_services_by_categories(
            api_token=api_token,
            filial_id=filial_id_query,
            tenant_id=tenant_id
        )

        if categories_data:
            # Use new structured data format
            return self._format_services_by_categories(categories_data, actual_filial_name)
        else:
            # Fallback to old API endpoint
            logger.info(f"[ListServicesInFilial] Новый API недоступен, используем старый метод для филиала '{actual_filial_name}'")
            return await self._process_with_old_api(tenant_id, api_token, filial_id_query, actual_filial_name)

    def _format_services_by_categories(self, categories_data: List[Dict[str, Any]], filial_name: str) -> str:
        """Форматирует данные из нового API endpoint с категориями и услугами."""
        if not categories_data:
            return f"В филиале '{filial_name}' не найдено услуг."

        # Prepare items for pagination (categories and their services)
        paginatable_items = []
        
        # Sort categories by name
        sorted_categories = sorted(categories_data, key=lambda x: normalize_text(x.get('categoryName', '')))
        
        for category_data in sorted_categories:
            category_name = category_data.get('categoryName', 'Без категории')
            services = category_data.get('services', [])
            
            # Add category header
            paginatable_items.append(('category', category_name, None, None, None))
            
            # Sort services within category
            sorted_services = sorted(services, key=lambda x: normalize_text(x.get('serviceName', '')))
            
            for service in sorted_services:
                service_name = service.get('serviceName', 'Неизвестная услуга')
                price = service.get('price')
                duration = service.get('duration')
                description = service.get('description', '').strip()
                
                paginatable_items.append(('service', service_name, price, duration, description))

        # Apply pagination
        total_items = len(paginatable_items)
        start_idx = (self.page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        paginated_items = paginatable_items[start_idx:end_idx]

        if not paginated_items and self.page_number > 1:
            max_pages = (total_items + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
            return f"Страница {self.page_number} не найдена. Доступно страниц: {max_pages}. Всего пунктов в филиале '{filial_name}': {total_items}"

        # Format response
        response_parts = []
        page_info = f" (страница {self.page_number} из {(total_items + self.page_size - 1) // self.page_size if total_items > 0 and self.page_size > 0 else 1})" if total_items > self.page_size else ""
        
        response_parts.append(f"Услуги в филиале '{filial_name}'{page_info}:")

        for item_type, name, price, duration, description in paginated_items:
            if item_type == 'category':
                response_parts.append(f"\n📋 {name}:")
            elif item_type == 'service':
                service_info = f"  • {name}"
                if price is not None:
                    try:
                        service_info += f" - {int(price):,} ₽".replace(',', ' ')
                    except (ValueError, TypeError):
                        service_info += f" - {price} ₽"
                if duration is not None:
                    service_info += f" ({duration} мин)"
                response_parts.append(service_info)
                
                if description:
                    response_parts.append(f"    💡 {description}")

        # Add pagination info
        if end_idx < total_items:
            response_parts.append(f"\n... показано {len(paginated_items)} из {total_items} пунктов. Используйте page_number={self.page_number + 1} для следующей страницы.")

        response_parts.append(f"\n💰 Для уточнения актуальных цен используйте функцию GetServicePrice.")

        return "\n".join(response_parts)

    async def _process_with_old_api(self, tenant_id: str, api_token: str, filial_id_query: str, actual_filial_name: str) -> str:
        """Fallback метод, использующий старый API endpoint."""
        from client_data_service import get_multiple_data_from_api
        
        api_data = await get_multiple_data_from_api(
             api_token=api_token,
             filial_id=filial_id_query,
             tenant_id=tenant_id
        )

        if not api_data:
            return f"В филиале '{actual_filial_name}' не найдено услуг или данные о них отсутствуют в актуальной базе данных."

        # Group services by category from the API response
        categories_with_services: Dict[str, Set[str]] = {}
        
        for item in api_data:
            service_name = item.get('serviceName')
            category_name = item.get('categoryName')

            if service_name and category_name:
                if category_name not in categories_with_services:
                    categories_with_services[category_name] = set()
                categories_with_services[category_name].add(service_name)

        if not categories_with_services:
            return f"В филиале '{actual_filial_name}' не найдено услуг или данные о них отсутствуют в актуальной базе данных (после группировки)."

        # Prepare items for pagination, including category headers
        paginatable_items = []
        sorted_categories = sorted(categories_with_services.keys(), key=normalize_text)

        for category in sorted_categories:
             paginatable_items.append(('category', category, None, None, None))
             services_in_category = sorted(list(categories_with_services[category]), key=normalize_text)
             for service in services_in_category:
                 paginatable_items.append(('service', service, None, None, None))

        # Apply pagination
        total_paginatable_items = len(paginatable_items)
        start_idx = (self.page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        paginated_items = paginatable_items[start_idx:end_idx]

        if not paginated_items and self.page_number > 1:
            max_pages = (total_paginatable_items + self.page_size - 1) // self.page_size if total_paginatable_items > 0 and self.page_size > 0 else 1
            return f"Страница {self.page_number} не найдена. Доступно страниц: {max_pages}. Всего пунктов в филиале '{actual_filial_name}': {total_paginatable_items}"

        response_parts = []
        page_info = f" (страница {self.page_number} из {(total_paginatable_items + self.page_size - 1) // self.page_size if total_paginatable_items > 0 and self.page_size > 0 else 1})" if total_paginatable_items > self.page_size else ""

        response_parts.append(f"Услуги в филиале '{actual_filial_name}'{page_info}:")

        for item_type, name, price, duration, description in paginated_items:
            if item_type == 'category':
                response_parts.append(f"\n📋 {name}:")
            elif item_type == 'service':
                response_parts.append(f"  • {name}")

        # Add pagination info
        if end_idx < total_paginatable_items:
            response_parts.append(f"\n... показано {len(paginated_items)} из {total_paginatable_items} пунктов. Используйте page_number={self.page_number + 1} для следующей страницы.")

        response_parts.append(f"\n💰 Для уточнения актуальных цен используйте функцию GetServicePrice.")

        return "\n".join(response_parts)
        start_idx = (self.page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        paginated_items = paginatable_items[start_idx:end_idx]

        if not paginated_items and self.page_number > 1:
             max_pages = (total_paginatable_items + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
             return f"Страница {self.page_number} не найдена. Доступно страниц: {max_pages}. Всего пунктов (категории+услуги) в филиале '{actual_filial_name}': {total_paginatable_items}"

        response_parts = []
        page_info = f" (страница {self.page_number} из {(total_paginatable_items + self.page_size - 1) // self.page_size if total_paginatable_items > 0 and self.page_size > 0 else 1})" if total_paginatable_items > self.page_size else ""

        response_parts.append(f"Услуги в филиале '{actual_filial_name}'{page_info}:")

        current_category = None
        for category, service in paginated_items:
            if service is None:
                response_parts.append(f"\n📋 {category}:")
                current_category = category
            elif current_category is not None:
                response_parts.append(f"  • {service}")
            elif service is not None:
                response_parts.append(f"• {service}")

        # Add pagination info
        if end_idx < total_paginatable_items:
             response_parts.append(f"\n... показано {len(paginated_items)} из {total_paginatable_items} пунктов (категории + услуги). Используйте page_number={self.page_number + 1} для следующей страницы.")

        response_parts.append(f"\n💰 Для получения подробной информации об услуге используйте функцию GetServicePrice.")

        return "\n".join(response_parts)



# Добавим все остальные функции (ListCategories, ListServicesInCategory, FindServicesInPriceRange и т.д.)
# Они остаются без изменений, так как уже работают с прямым поиском или используют ID там, где это необходимо

class ListCategories(BaseModel):
    """Модель для получения списка всех категорий услуг."""
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество категорий на странице")

    async def process(self, tenant_id: str, api_token: str) -> str:
        logger.info(f"[ListCategories Proc] Запрос списка категорий, Tenant: {tenant_id}, Page: {self.page_number}, Size: {self.page_size}")

        # Получаем актуальные данные через API
        try:
            api_data = await get_multiple_data_from_api(
                api_token=api_token,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.error(f"Ошибка при вызове get_multiple_data_from_api для получения списка категорий: {e}", exc_info=True)
            return f"Ошибка при получении актуальных данных для списка категорий."

        if not api_data:
            return "Не найдено актуальных данных для получения списка категорий."

        all_categories_orig: Set[str] = set()
        for item_data in api_data:
            c_name_val = item_data.get('categoryName')
            if c_name_val: 
                all_categories_orig.add(c_name_val)

        if not all_categories_orig:
            return "Список категорий в базе пуст."

        sorted_categories = sorted(list(all_categories_orig), key=normalize_text)
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


class GetEmployeeServices(BaseModel):
    """Модель для получения списка категорий услуг конкретного сотрудника, опционально в конкретном филиале.
    
    ВАЖНО: Теперь возвращает категории услуг вместо конкретных услуг для соответствия категориально-ориентированному подходу.
    Для получения конкретных услуг в категории используйте GetServicesByCategory.
    """
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала (опционально, для фильтрации категорий по филиалу)")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество категорий на странице")

    async def process(self, tenant_id: str, api_token: str) -> str: # <-- Added tenant_id and api_token
        logger.info(f"[GetEmployeeServices Proc] Получение услуг сотрудника '{self.employee_name}' в филиале '{self.filial_name}' (Tenant: {tenant_id}, Page: {self.page_number}, Size: {self.page_size})")

        # Get employee ID first
        employee_id_query = get_id_by_name(tenant_id, 'employee', self.employee_name)
        if not employee_id_query:
             return f"Сотрудник с именем '{self.employee_name}' не найден."
             
        # Get filial ID if provided
        filial_id_query = get_id_by_name(tenant_id, 'filial', self.filial_name) if self.filial_name else None
        # If filial name was provided but ID not found, inform the user
        if self.filial_name and not filial_id_query:
             return f"Филиал с названием '{self.filial_name}' не найден."

        # Call the new API endpoint with employeeId and optional filialId
        from client_data_service import get_multiple_data_from_api
        api_data = await get_multiple_data_from_api(
             api_token=api_token,
             employee_id=employee_id_query,
             filial_id=filial_id_query, # Pass filialId if found (optional filter)
             tenant_id=tenant_id # Pass for logging
        )

        if not api_data:
            filial_info = f" в филиале '{self.filial_name}'" if self.filial_name else ""
            return f"Не найдено услуг для сотрудника '{self.employee_name}'{filial_info} через актуальную базу данных."

        # Process data to extract unique services and their categories
        # The API returns a list where each item is employee+service+filial.
        # We need to list unique services provided by the employee (optionally filtered by filial).
        services_info: Dict[str, Dict[str, Any]] = {}
        
        for item in api_data:
            service_id = item.get('serviceId')
            service_name = item.get('serviceName')
            category_name = item.get('categoryName') # Assuming API provides category name
            item_filial_id = item.get('filialId')

            if not service_id or not service_name or not item_filial_id: # Ensure essential data is present
                continue
            
            # Apply filial filter if specified in the original request
            if filial_id_query and item_filial_id != filial_id_query:
                continue

            # Add service if not already added (ensuring uniqueness across filials if no filial filter)
            if service_id not in services_info:
                 services_info[service_id] = {
                      'name': service_name,
                      'category': category_name # Store category name if available
                 }

        if not services_info:
             filial_info = f" в филиале '{self.filial_name}'" if self.filial_name else ""
             return f"Не найдено услуг для сотрудника '{self.employee_name}'{filial_info} через актуальную базу данных."

        # Extract service names and sort them
        service_list = sorted(services_info.values(), key=lambda x: normalize_text(x.get('name', ''), keep_spaces=True))

        # Apply pagination
        total_services = len(service_list)
        start_idx = (self.page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        paginated_services = service_list[start_idx:end_idx]

        if not paginated_services and self.page_number > 1:
             max_pages = (total_services + self.page_size - 1) // self.page_size if self.page_size > 0 else 1
             return f"Страница {self.page_number} услуг не найдена. Доступно страниц: {max_pages}. Всего услуг у сотрудника '{self.employee_name}': {total_services}"

        response_parts = []
        filial_part = f" в филиале '{self.filial_name}'" if self.filial_name else ""
        page_info = f" (страница {self.page_number} из {(total_services + self.page_size - 1) // self.page_size if self.page_size > 0 else 1})" if total_services > self.page_size else ""

        # Group services by category for display (optional, based on required output format)
        # If the requirement is just a list of services, the following grouping is not needed.
        # Assuming we just need a list of services for the employee (optionally in a filial).
        
        response_parts.append(f"Услуги сотрудника '{self.employee_name}'{filial_part}{page_info}:")

        for service_data in paginated_services:
             service_name = service_data['name']
             category_name = service_data.get('category')
             if category_name:
                  response_parts.append(f"- {service_name} (Категория: {category_name})")
             else:
                  response_parts.append(f"- {service_name}")

        # Add pagination info
        if end_idx < total_services:
            response_parts.append(f"\n... показано {len(paginated_services)} из {total_services} услуг. Используйте page_number={self.page_number + 1} для следующей страницы.")
            
        response_parts.append(f"\nДля получения подробной информации об услуге используйте функцию GetServicePrice или FindServiceLocations.")

        return "\n".join(response_parts)


class ListServicesInCategory(BaseModel):
    """Модель для получения списка услуг в конкретной категории."""
    category_name: str = Field(description="Точное название категории")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество услуг на странице")

    async def process(self, tenant_id: str, api_token: str) -> str:
        logger.info(f"[ListServicesInCategory Proc] Запрос услуг в категории: {self.category_name}, Tenant: {tenant_id}, Page: {self.page_number}, Size: {self.page_size}")

        # Get category ID first
        category_id = get_id_by_name(tenant_id, 'category', self.category_name)
        if not category_id:
            return f"Категория с названием, похожим на '{self.category_name}', не найдена."

        display_category_name = get_name_by_id(tenant_id, 'category', category_id) or self.category_name

        # Call the new API endpoint filtering by categoryId (using service_id parameter with category logic)
        from client_data_service import get_multiple_data_from_api
        try:
            # Get all data and filter by category on our side since API doesn't have category_id parameter
            api_data = await get_multiple_data_from_api(
                api_token=api_token,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.error(f"Ошибка при вызове get_multiple_data_from_api для категории {self.category_name}: {e}", exc_info=True)
            return f"Ошибка при получении данных услуг для категории '{self.category_name}'."

        if not api_data:
            return f"Не найдено данных в актуальной базе для анализа категории '{display_category_name}'."

        # Filter services by category from API response
        services_in_category: Set[str] = set()
        for item in api_data:
            item_category_id = item.get('categoryId')
            if item_category_id == category_id:
                service_name = item.get('serviceName')
                if service_name:
                    services_in_category.add(service_name)

        if not services_in_category:
            return f"В категории '{display_category_name}' не найдено услуг в актуальной базе данных."
        
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


class FindServicesInPriceRange(BaseModel):
    """Модель для поиска услуг в заданном ценовом диапазоне."""
    min_price: float = Field(description="Минимальная цена")
    max_price: float = Field(description="Максимальная цена")
    category_name: Optional[str] = Field(default=None, description="Опционально: категория")
    filial_name: Optional[str] = Field(default=None, description="Опционально: филиал")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество услуг на странице")

    async def process(self, tenant_id: str, api_token: str) -> str:
        logger.info(f"[FindServicesInPriceRange Proc] Поиск услуг в диапазоне цен: {self.min_price}-{self.max_price}, Категория: {self.category_name}, Филиал: {self.filial_name}, Tenant: {tenant_id}, Page: {self.page_number}, Size: {self.page_size}")

        # Получаем актуальные данные через API
        try:
            api_data = await get_multiple_data_from_api(
                api_token=api_token,
                tenant_id=tenant_id,
                filial_id=None  # Получаем данные по всем филиалам, потом фильтруем
            )
        except Exception as e:
            logger.error(f"Ошибка при вызове get_multiple_data_from_api для поиска услуг в диапазоне цен: {e}", exc_info=True)
            return f"Ошибка при получении актуальных данных для поиска услуг в диапазоне цен {self.min_price}-{self.max_price}."

        if not api_data:
            return f"Не найдено актуальных данных для поиска услуг в диапазоне цен {self.min_price}-{self.max_price}."

        target_category_id: Optional[str] = None
        display_category_name_query = self.category_name
        if self.category_name:
            target_category_id = get_id_by_name(tenant_id, 'category', self.category_name)
            if not target_category_id:
                return f"Категория '{self.category_name}' не найдена."
            display_category_name_query = get_name_by_id(tenant_id, 'category', target_category_id) or self.category_name

        target_filial_id: Optional[str] = None
        display_filial_name_query = self.filial_name
        if self.filial_name:
            target_filial_id = get_id_by_name(tenant_id, 'filial', self.filial_name)
            if not target_filial_id:
                return f"Филиал '{self.filial_name}' не найден."
            display_filial_name_query = get_name_by_id(tenant_id, 'filial', target_filial_id) or self.filial_name

        found_services: List[Dict[str, Any]] = [] # Список словарей для каждой найденной услуги
        processed_service_ids: Set[str] = set() # Для избежания дублирования услуг с разными ценами в одном филиале, если такое возможно

        for item in api_data:
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
                filial_name_raw = item.get('filialName', "(не указан)") 
                category_name_raw = item.get('categoryName', "(не указана)") 
                
                effective_filial_name = display_filial_name_query if target_filial_id else filial_name_raw
                effective_category_name = display_category_name_query if target_category_id else category_name_raw

                found_services.append({
                    'name': service_name_raw,
                    'price': price,
                    'filial': effective_filial_name,
                    'category': effective_category_name,
                    'id': service_id 
                })
                processed_service_ids.add(service_id)
        
        if not found_services:
            return f"Услуги в ценовом диапазоне {self.min_price}-{self.max_price} руб. (с учетом фильтров) не найдены."

        
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
            if not target_filial_id and service['filial'] != "(не указан)": 
                location_info.append(f"Филиал: {service['filial']}")
            if not target_category_id and service['category'] != "(не указана)": 
                location_info.append(f"Категория: {service['category']}")
            if location_info:
                detail += f" ({ '; '.join(location_info) })"
            response_parts.append(detail)

        if end_index < total_found:
            remaining = total_found - end_index
            response_parts.append(f"... и еще {remaining} {get_readable_count_form(remaining, ('услуга', 'услуги', 'услуг'))}. Введите 'дальше', чтобы увидеть их.")
        
        return "\n".join(response_parts)


class FindSpecialistsByServiceOrCategoryAndFilial(BaseModel):
    """Модель для поиска специалистов по услуге/категории и филиалу."""
    query_term: str = Field(description="Название услуги ИЛИ категории")
    filial_name: str = Field(description="Точное название филиала")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=15, description="Количество специалистов на странице")

    async def process(self, tenant_id: str, api_token: str) -> str:
        if not self.query_term or not self.filial_name: 
            return "Ошибка: Укажите услугу/категорию и филиал."

        logger.info(f"[FindSpecialistsByServiceOrCategoryAndFilial Proc] Поиск специалистов (Запрос: '{self.query_term}', Филиал: '{self.filial_name}'), Tenant: {tenant_id}, Page: {self.page_number}, Size: {self.page_size}")

        filial_id = get_id_by_name(tenant_id, 'filial', self.filial_name)
        if not filial_id:
            return f"Филиал с названием, похожим на '{self.filial_name}', не найден."
        
        original_filial_display_name = get_name_by_id(tenant_id, 'filial', filial_id) or self.filial_name

        service_id_match = get_id_by_name(tenant_id, 'service', self.query_term)
        category_id_match = None
        query_type = ""
        resolved_query_term_name = self.query_term

        if service_id_match:
            query_type = "service"
            logger.info(f"Запрос '{self.query_term}' распознан как услуга с ID: {service_id_match}")
            resolved_query_term_name = get_name_by_id(tenant_id, 'service', service_id_match) or self.query_term
        else:
            category_id_match = get_id_by_name(tenant_id, 'category', self.query_term)
            if category_id_match:
                query_type = "category"
                logger.info(f"Запрос '{self.query_term}' распознан как категория с ID: {category_id_match}")
                resolved_query_term_name = get_name_by_id(tenant_id, 'category', category_id_match) or self.query_term
            else:
                return f"Термин '{self.query_term}' не распознан ни как услуга, ни как категория."

        # Получаем актуальные данные о специалистах через API
        from client_data_service import get_multiple_data_from_api
        try:
            # Фильтруем по филиалу и услуге/категории
            api_data = await get_multiple_data_from_api(
                api_token=api_token,
                filial_id=filial_id,
                service_id=service_id_match if query_type == "service" else None,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.error(f"Ошибка при вызове get_multiple_data_from_api для поиска специалистов: {e}", exc_info=True)
            return f"Ошибка при получении актуальных данных специалистов для '{self.query_term}' в филиале '{self.filial_name}'."

        if not api_data:
            return f"В филиале '{original_filial_display_name}' не найдено актуальных данных для анализа специалистов по '{resolved_query_term_name}'."

        matching_employees: Dict[str, str] = {}
        
        for item in api_data:
            # Дополнительная фильтрация по категории если нужно (API фильтрует только по service_id)
            if query_type == "category" and item.get("categoryId") != category_id_match:
                continue

            emp_id = item.get("employeeId")
            emp_name_raw = item.get("employeeFullName")
            if not emp_id or not emp_name_raw: 
                continue

            # Добавляем сотрудника в результаты
            if emp_id not in matching_employees:
                matching_employees[emp_id] = emp_name_raw

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


class ListEmployeeFilials(BaseModel):
    """Модель для получения списка филиалов конкретного сотрудника."""
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")

    async def process(self, tenant_id: str, api_token: str) -> str: # <-- Added tenant_id and api_token
        logger.info(f"[ListEmployeeFilials Proc] Получение филиалов для сотрудника '{self.employee_name}', Tenant: {tenant_id}")

        # Get employee ID first
        employee_id_query = get_id_by_name(tenant_id, 'employee', self.employee_name)
        if not employee_id_query:
            return f"Сотрудник с именем '{self.employee_name}' не найден."

        # Call the new API endpoint filtering by employeeId
        from client_data_service import get_multiple_data_from_api
        try:
            api_data = await get_multiple_data_from_api(
                 api_token=api_token,
                 employee_id=employee_id_query,
                 tenant_id=tenant_id # Pass for logging
            )
        except Exception as e:
            logger.error(f"Ошибка при вызове get_multiple_data_from_api для сотрудника {self.employee_name}: {e}", exc_info=True)
            return f"Ошибка при получении данных о филиалах для сотрудника '{self.employee_name}'."

        if not api_data:
            # Employee ID was found, but API returned no data for them.
            # This implies the employee is not associated with any service/filial in the live data.
            return f"Для сотрудника '{self.employee_name}' не найдено связанных филиалов в актуальной базе данных."

        # Extract unique filial names from the API response
        employee_filials = set() # Use a set for unique names
        actual_employee_name = self.employee_name # Default to query name

        for item in api_data:
            filial_name = item.get('filialName')
            if filial_name:
                employee_filials.add(filial_name)
            # Capture the actual employee name from the API data if available
            if not actual_employee_name and item.get('employeeFullName'):
                actual_employee_name = item.get('employeeFullName')

        if not employee_filials:
             # This case should technically be covered by api_data check, but good to be explicit.
             return f"Для сотрудника '{actual_employee_name}' не найдено связанных филиалов в актуальной базе данных."

        # Format the result
        sorted_filials = sorted(list(employee_filials), key=normalize_text)
        filial_list_str = ', '.join(sorted_filials)

        return f"Сотрудник '{actual_employee_name}' ведет прием в следующих филиалах: {filial_list_str}."


class GetFreeSlotsArgs(BaseModel):
    """Аргументы для GetFreeSlots."""
    tenant_id: str
    employee_id: str
    service_ids: List[str]
    date_time: str
    filial_id: str
    lang_id: str = "ru"
    api_token: Optional[str] = None


class GetFreeSlots(BaseModel):
    """Класс для получения свободных слотов сотрудника."""
    tenant_id: str
    employee_id: str
    service_ids: List[str]
    date_time: str
    filial_id: str
    lang_id: str = "ru"
    api_token: Optional[str] = None

    async def process(self, **kwargs) -> str:
        # Устанавливаем начальную дату для поиска
        if not self.date_time:
            search_date = datetime.now()
            self.date_time = search_date.strftime("%Y.%m.%d")
            logger.info(f"Дата не указана, используется текущая: {self.date_time}")
        else:
            try:
                search_date = datetime.strptime(self.date_time, "%Y.%m.%d")
            except ValueError:
                try:
                    search_date = datetime.strptime(self.date_time, "%d.%m.%Y")
                    self.date_time = search_date.strftime("%Y.%m.%d")
                except ValueError:
                    search_date = datetime.now()
                    self.date_time = search_date.strftime("%Y.%m.%d")
                    logger.warning(f"Неверный формат даты, используется текущая: {self.date_time}")
        
        # Проверка корректности ID услуг и сотрудника
        if not self.employee_id:
            return "Не удалось определить ID сотрудника. Пожалуйста, проверьте правильность имени сотрудника."
        
        if not self.service_ids or not all(self.service_ids):
            return "Не удалось определить ID услуг. Пожалуйста, проверьте правильность названий услуг."
        
        if not self.filial_id:
            return "Не удалось определить ID филиала. Пожалуйста, проверьте правильность названия филиала."
        
        # Получаем имена для отображения
        employee_name_display = get_name_by_id(self.tenant_id, 'employee', self.employee_id) or self.employee_id
        filial_name_display = get_name_by_id(self.tenant_id, 'filial', self.filial_id) or self.filial_id
        
        # Умный поиск слотов с итерацией по датам
        max_iterations = 7
        date_increment = 7  # дней
        original_date = search_date
        searched_dates = []
        
        logger.info(f"Начинаем умный поиск слотов: TenantID={self.tenant_id}, EmployeeID={self.employee_id}, ServiceIDs={self.service_ids}, StartDate={self.date_time}, FilialID={self.filial_id}")
        
        for iteration in range(max_iterations):
            current_date_str = search_date.strftime("%Y-%m-%d")
            searched_dates.append(search_date.strftime("%d.%m.%Y"))
            
            logger.info(f"Поиск слотов: итерация {iteration + 1}/{max_iterations}, дата: {current_date_str}")
            
            try:
                response_data = await get_free_times_of_employee_by_services(
                    tenant_id=self.tenant_id,
                    employee_id=self.employee_id,
                    service_ids=self.service_ids,
                    date_time=current_date_str,
                    filial_id=self.filial_id,
                    api_token=self.api_token
                )

                if response_data and response_data.get('code') == 200:
                    api_data_content = response_data.get('data')

                    # Обработка нового формата с workDates
                    if isinstance(api_data_content, dict) and 'workDates' in api_data_content:
                        all_formatted_slots = []
                        employee_name_from_api = api_data_content.get('name', employee_name_display)

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
                        
                        if all_formatted_slots:
                            # Найдены слоты! Возвращаем результат
                            if iteration > 0:
                                date_range_info = f"(поиск проводился с {searched_dates[0]} по {searched_dates[-1]})"
                            else:
                                date_range_info = f"на запрашиваемую дату {searched_dates[0]}"
                            
                            response_message = f"Доступные слоты для сотрудника {employee_name_from_api} в филиале {filial_name_display} {date_range_info}:\n" + "\n".join(all_formatted_slots)
                            logger.info(f"Найдены слоты на итерации {iteration + 1}, дата: {current_date_str}")
                            return response_message

                    # Обработка старого формата (список)
                    elif isinstance(api_data_content, list) and api_data_content:
                        processed_slots = []
                        for slot_info in api_data_content:
                            if isinstance(slot_info, dict) and 'time' in slot_info:
                                processed_slots.append(slot_info['time'])
                        
                        if processed_slots:
                            if iteration > 0:
                                date_range_info = f"(поиск проводился с {searched_dates[0]} по {searched_dates[-1]})"
                            else:
                                date_range_info = f"на запрашиваемую дату {searched_dates[0]}"
                            
                            logger.info(f"Найдены слоты на итерации {iteration + 1}, дата: {current_date_str}")
                            return f"Для сотрудника {employee_name_display} в филиале {filial_name_display} {date_range_info} доступны следующие слоты: {', '.join(processed_slots)}."
                
                # Ошибка API - прерываем поиск
                elif response_data and response_data.get('code') != 200:
                    error_msg = response_data.get('message', 'Неизвестная ошибка от API')
                    logger.error(f"Ошибка от API при запросе свободных слотов (код {response_data.get('code')}): {error_msg}")
                    return f"Не удалось получить свободные слоты: {error_msg}"

            except Exception as e:
                logger.error(f"Исключение в GetFreeSlots.process на итерации {iteration + 1}, дата {current_date_str}: {e}", exc_info=True)
                
                # Если это таймаут, даем более понятное сообщение
                if "ReadTimeout" in str(e) or "timeout" in str(e).lower():
                    return f"Извините, система записи временно медленно отвечает. Попробуйте повторить запрос через несколько минут или обратитесь к администратору."
                else:
                    return f"Произошла внутренняя ошибка при поиске свободных слотов: {e}"
            
            # Переходим к следующей дате (+7 дней)
            search_date += timedelta(days=date_increment)
        
        # Если после всех итераций слоты не найдены
        date_range_str = f"с {searched_dates[0]} по {searched_dates[-1]}"
        logger.info(f"Слоты не найдены после {max_iterations} итераций поиска в диапазоне {date_range_str}")
        
        return f"К сожалению, у сотрудника {employee_name_display} нет свободных слотов в филиале {filial_name_display} в период {date_range_str} по выбранным услугам. Попробуйте выбрать другую дату или другого специалиста."


class BookAppointmentAIPayload(BaseModel):
    """Класс для создания записи с AI payload."""
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




class RescheduleAppointment(BaseModel):
    """Перенос существующей записи клиента на новое время."""
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (опционально, для логов)")
    record_id: str = Field(description="ID переносимой записи")
    date_of_record: str = Field(description="Новая дата (YYYY-MM-DD)")
    start_time: str = Field(description="Новое время начала (HH:MM)")
    end_time: str = Field(description="Новое время окончания (HH:MM)")
    api_token: Optional[str] = None

    async def process(self, tenant_id: Optional[str] = None, api_token: Optional[str] = None) -> str:
        # Берём tenant_id и токен из аргументов или из свойств экземпляра
        tenant_to_use = tenant_id or self.tenant_id
        token_to_use = api_token or self.api_token

        try:
            result = await update_record_time(
                record_id=self.record_id,
                date_of_record=self.date_of_record,
                start_time=self.start_time,
                end_time=self.end_time,
                tenant_id=tenant_to_use,
                api_token=token_to_use,
            )
            code = result.get("code")
            if code == 200:
                return (
                    f"Запись успешно перенесена на {self.date_of_record} "
                    f"{self.start_time}-{self.end_time}."
                )
            return f"Не удалось перенести запись: {result.get('message', 'Неизвестная ошибка')}"
        except Exception as e:
            logger.error(f"Ошибка в RescheduleAppointment.process: {e}", exc_info=True)
            return f"Ошибка при переносе записи: {type(e).__name__} - {e}"


class CancelAppointment(BaseModel):
    """Отмена существующей записи клиента."""
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (опционально, для логов)")
    record_id: str = Field(description="ID отменяемой записи")
    chain_id: str = Field(description="Chain ID, которому принадлежит запись")
    canceling_reason: Optional[str] = Field(default=None, description="ID или текст причины отмены (если не указан, система выберет автоматически)")
    api_token: Optional[str] = None

    async def process(self, tenant_id: Optional[str] = None, api_token: Optional[str] = None) -> str:
        tenant_to_use = tenant_id or self.tenant_id
        token_to_use = api_token or self.api_token

        if not token_to_use:
            return "Критическая ошибка: отсутствует API токен для отмены записи."

        try:
            reason_to_use = None
            reasons_list = await get_cancel_reasons(chain_id=self.chain_id, tenant_id=tenant_to_use, api_token=token_to_use)
            if not reasons_list:
                return "Не удалось получить список причин отмены."

            # Строим словарь id->name
            id_name_map = {str(item.get("id")): item.get("name", "") for item in reasons_list if isinstance(item, dict)}

            # Если LLM передала текст причины (не число) — пытаемся найти наиболее близкую
            if self.canceling_reason and not self.canceling_reason.isdigit():
                import difflib
                names = list(id_name_map.values())
                best_match = difflib.get_close_matches(self.canceling_reason.lower(), [n.lower() for n in names], n=1, cutoff=0.3)
                if best_match:
                    # Получаем id по имени
                    for rid, rname in id_name_map.items():
                        if rname.lower() == best_match[0]:
                            reason_to_use = rid
                            logger.info(f"[CancelAppointment] По тексту '{self.canceling_reason}' выбран reason id={rid} ('{rname}')")
                            break
            elif self.canceling_reason and self.canceling_reason.isdigit():
                reason_to_use = self.canceling_reason

            # Если всё ещё не выбрано — берём первую причину
            if not reason_to_use:
                first_id = next(iter(id_name_map.keys()))
                first_name = id_name_map[first_id]
                reason_to_use = first_id
                logger.info(f"[CancelAppointment] По умолчанию выбран первый reason id={first_id} ('{first_name}')")

            result = await cancel_record(
                record_id=self.record_id,
                chain_id=self.chain_id,
                canceling_reason=reason_to_use,
                tenant_id=tenant_to_use,
                api_token=token_to_use,
            )
            code = result.get("code")
            if code == 200:
                return "Запись успешно отменена."
            return f"Не удалось отменить запись: {result.get('message', 'Неизвестная ошибка')}"
        except Exception as e:
            logger.error(f"Ошибка в CancelAppointment.process: {e}", exc_info=True)
            return f"Ошибка при отмене записи: {type(e).__name__} - {e}"

class GetServicesByCategory(BaseModel):
    """
    Модель для получения услуг по категории с обязательным указанием филиала.
    Используется после того, как пользователь выбрал категорию из результатов FindCategoriesByQuery.
    """
    category_name: str = Field(description="Точное название категории")
    filial_name: str = Field(description="ОБЯЗАТЕЛЬНОЕ точное название филиала")
    page_number: int = Field(default=1, description="Номер страницы результатов")
    page_size: int = Field(default=20, description="Количество услуг на странице")
    include_prices: bool = Field(default=True, description="Включать ли цены в результат")

    async def process(self, tenant_id: str, api_token: str) -> str:
        logger.info(f"[GetServicesByCategory Proc] Получение услуг по категории: '{self.category_name}' в филиале '{self.filial_name}', Tenant: {tenant_id}, Page: {self.page_number}, Size: {self.page_size}")

        # Проверяем существование категории
        category_id = get_id_by_name(tenant_id, 'category', self.category_name)
        if not category_id:
            return f"Категория с названием, похожим на '{self.category_name}', не найдена."

        # Проверяем существование филиала
        filial_id = get_id_by_name(tenant_id, 'filial', self.filial_name)
        if not filial_id:
            return f"Филиал с названием, похожим на '{self.filial_name}', не найден."

        # Получаем точные названия для отображения
        display_category_name = get_name_by_id(tenant_id, 'category', category_id) or self.category_name
        display_filial_name = get_name_by_id(tenant_id, 'filial', filial_id) or self.filial_name

        # Получаем актуальные данные через API с фильтрацией по филиалу
        from client_data_service import get_multiple_data_from_api
        try:
            api_data = await get_multiple_data_from_api(
                api_token=api_token,
                filial_id=filial_id,
                tenant_id=tenant_id
            )
        except Exception as e:
            logger.error(f"Ошибка при вызове get_multiple_data_from_api для категории {self.category_name} в филиале {self.filial_name}: {e}", exc_info=True)
            return f"Ошибка при получении актуальных данных для категории '{self.category_name}' в филиале '{self.filial_name}'."

        if not api_data:
            return f"В филиале '{display_filial_name}' не найдено актуальных данных для анализа доступности услуг."

        # Собираем услуги из указанной категории в указанном филиале из API
        services_in_category_and_filial = {}  # service_name -> {'price': float or None, 'count': int}
        
        for item in api_data:
            # Фильтруем по категории (так как API фильтрует только по филиалу)
            if item.get('categoryId') != category_id:
                continue
                
            service_name = item.get('serviceName')
            if not service_name:
                continue
            
            # Получаем цену из статических данных если требуется
            price = None
            if self.include_prices:
                # Цены берем из статических данных для стабильности
                service_id = item.get('serviceId')
                if service_id and _clinic_data:
                    for static_item in _clinic_data:
                        if (static_item.get('serviceId') == service_id and 
                            static_item.get('filialId') == filial_id):
                            price_raw = static_item.get('price')
                            if price_raw is not None and price_raw != '':
                                try:
                                    price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                                    break
                                except (ValueError, TypeError):
                                    pass
            
            if service_name not in services_in_category_and_filial:
                services_in_category_and_filial[service_name] = {'price': price, 'count': 0}
            else:
                # Если цена еще не установлена, устанавливаем её
                if services_in_category_and_filial[service_name]['price'] is None and price is not None:
                    services_in_category_and_filial[service_name]['price'] = price
            
            services_in_category_and_filial[service_name]['count'] += 1

        if not services_in_category_and_filial:
            category_clarification = f" (уточнено до '{display_category_name}')" if normalize_text(display_category_name, keep_spaces=True) != normalize_text(self.category_name, keep_spaces=True) else ""
            filial_clarification = f" (уточнено до '{display_filial_name}')" if normalize_text(display_filial_name, keep_spaces=True) != normalize_text(self.filial_name, keep_spaces=True) else ""
            
            return f"В категории '{self.category_name}'{category_clarification} в филиале '{self.filial_name}'{filial_clarification} не найдено услуг."

        # Сортируем услуги по названию
        sorted_services = sorted(
            services_in_category_and_filial.items(),
            key=lambda x: normalize_text(x[0], keep_spaces=True)
        )

        # Применяем пагинацию
        total_found = len(sorted_services)
        start_idx = (self.page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        paginated_services = sorted_services[start_idx:end_idx]

        if not paginated_services and self.page_number > 1:
            max_pages = (total_found + self.page_size - 1) // self.page_size
            return f"Страница {self.page_number} не найдена. Доступно страниц: {max_pages}. Всего найдено услуг: {total_found}"

        # Формируем ответ
        response_parts = []
        
        # Добавляем уточнения если названия были изменены
        category_clarification = f" (уточнено до '{display_category_name}')" if normalize_text(display_category_name, keep_spaces=True) != normalize_text(self.category_name, keep_spaces=True) else ""
        filial_clarification = f" (уточнено до '{display_filial_name}')" if normalize_text(display_filial_name, keep_spaces=True) != normalize_text(self.filial_name, keep_spaces=True) else ""
        
        page_info = f" (страница {self.page_number} из {(total_found + self.page_size - 1) // self.page_size})" if total_found > self.page_size else ""
        
        response_parts.append(f"Услуги категории '{self.category_name}'{category_clarification} в филиале '{self.filial_name}'{filial_clarification}{page_info}:")

        for service_name, info in paginated_services:
            if self.include_prices and info['price'] is not None:
                service_line = f"- {service_name} - {info['price']:.0f} руб."
            else:
                service_line = f"- {service_name}"
            
            response_parts.append(service_line)

        # Добавляем информацию о дополнительных страницах
        if end_idx < total_found:
            response_parts.append(f"\n... показано {len(paginated_services)} из {total_found} услуг. Используйте page_number={self.page_number + 1} для следующей страницы.")

        return "\n".join(response_parts)
