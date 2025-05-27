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
        
    # Нормализуем оба имени с сортировкой слов для сравнения
    normalized_query = normalize_text(query_name, keep_spaces=False, sort_words=True)
    normalized_db = normalize_text(db_name, keep_spaces=False, sort_words=True)
    
    # Точное совпадение после нормализации и сортировки
    if normalized_query == normalized_db:
        return True
    
    # Если точного совпадения нет, проверяем покрытие слов
    query_words = set(normalize_text(query_name, keep_spaces=True).lower().split())
    db_words = set(normalize_text(db_name, keep_spaces=True).lower().split())
    
    # Удаляем очень короткие слова (предлоги, сокращения)
    query_words = {word for word in query_words if len(word) > 1}
    db_words = {word for word in db_words if len(word) > 1}
    
    # Проверяем, содержатся ли все слова запроса в словах из БД
    if query_words and query_words.issubset(db_words):
        return True
    
    # Проверяем частичное совпадение (хотя бы 70% слов)
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
    page_size: int = Field(default=15, description="Количество результатов на странице")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[FindEmployees] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."
        
        logger.info(f"[FC Proc] Поиск сотрудников (Сотрудник: '{self.employee_name}', Услуга: '{self.service_name}', Филиал: '{self.filial_name}'), Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        # Нормализуем запросы для поиска
        norm_emp_name = normalize_text(self.employee_name, keep_spaces=True).lower() if self.employee_name else ""
        norm_service_name = normalize_text(self.service_name, keep_spaces=True).lower() if self.service_name else ""
        norm_filial_name = normalize_text(self.filial_name, keep_spaces=True).lower() if self.filial_name else ""

        # Группируем данные по сотрудникам
        employees_info: Dict[str, Dict[str, Any]] = {}
        
        for item in _clinic_data:
            item_emp_name_raw = item.get('employeeFullName')
            item_service_name_raw = item.get('serviceName')
            item_filial_name_raw = item.get('filialName')
            emp_id = item.get('employeeId')

            if not emp_id or not item_emp_name_raw:
                continue

            norm_item_emp = normalize_text(item_emp_name_raw, keep_spaces=True).lower()
            norm_item_service = normalize_text(item_service_name_raw, keep_spaces=True).lower() if item_service_name_raw else ""
            norm_item_filial = normalize_text(item_filial_name_raw, keep_spaces=True).lower() if item_filial_name_raw else ""

            # Проверяем совпадения с умным сравнением имен
            emp_match = (not self.employee_name or smart_name_match(self.employee_name, item_emp_name_raw))
            service_match = (not norm_service_name or (norm_item_service and norm_service_name in norm_item_service))
            filial_match = (not norm_filial_name or (norm_item_filial and norm_filial_name == norm_item_filial))

            # Если все условия выполнены, добавляем/обновляем информацию о сотруднике
            if emp_match and service_match and filial_match:
                if emp_id not in employees_info:
                    employees_info[emp_id] = {
                        'name': item_emp_name_raw,
                        'services': {},  # По филиалам: filial_name -> set(services)
                        'filials': set()
                    }

                # Добавляем филиал
                if item_filial_name_raw:
                    employees_info[emp_id]['filials'].add(item_filial_name_raw)
                    
                    # Добавляем услугу для этого филиала
                    if item_service_name_raw:
                        if item_filial_name_raw not in employees_info[emp_id]['services']:
                            employees_info[emp_id]['services'][item_filial_name_raw] = set()
                        employees_info[emp_id]['services'][item_filial_name_raw].add(item_service_name_raw)

        if not employees_info:
            search_criteria = []
            if self.employee_name: search_criteria.append(f"имя содержит '{self.employee_name}'")
            if self.service_name: search_criteria.append(f"услуга содержит '{self.service_name}'")
            if self.filial_name: search_criteria.append(f"филиал '{self.filial_name}'")
            criteria_str = ", ".join(search_criteria) if search_criteria else "указанным критериям"
            return f"Сотрудники, соответствующие {criteria_str}, не найдены."

        # Сортируем сотрудников по имени
        sorted_employees = sorted(employees_info.items(), 
                                key=lambda x: normalize_text(x[1].get('name', ''), keep_spaces=True))
        
        # Применяем пагинацию
        total_found = len(sorted_employees)
        start_idx = (self.page_number - 1) * self.page_size
        end_idx = start_idx + self.page_size
        paginated_employees = sorted_employees[start_idx:end_idx]

        if not paginated_employees and self.page_number > 1:
            max_pages = (total_found + self.page_size - 1) // self.page_size
            return f"Страница {self.page_number} не найдена. Доступно страниц: {max_pages}. Всего найдено сотрудников: {total_found}"

        response_parts = []
        
        # Определяем режим работы
        is_simple_filial_list = (self.filial_name and not self.employee_name and not self.service_name)
        is_employee_services_in_filial = (self.employee_name and self.filial_name and not self.service_name)
        
        if is_simple_filial_list:
            # Режим 1: Простой список сотрудников в филиале
            page_info = f" (страница {self.page_number} из {(total_found + self.page_size - 1) // self.page_size})" if total_found > self.page_size else ""
            response_parts.append(f"Сотрудники филиала '{self.filial_name}'{page_info}:")
            
            for emp_id, emp_data in paginated_employees:
                response_parts.append(f"- {emp_data['name']}")
                
        elif is_employee_services_in_filial:
            # Режим 2: Услуги конкретного сотрудника в конкретном филиале
            if len(paginated_employees) == 1:
                emp_id, emp_data = paginated_employees[0]
                emp_name = emp_data['name']
                
                # Получаем услуги ТОЛЬКО для указанного филиала
                target_filial_services = emp_data['services'].get(self.filial_name, set())
                
                if target_filial_services:
                    response_parts.append(f"Услуги сотрудника '{emp_name}' в филиале '{self.filial_name}':")
                    for service in sorted(target_filial_services, key=lambda s: normalize_text(s, keep_spaces=True)):
                        response_parts.append(f"- {service}")
                else:
                    response_parts.append(f"Сотрудник '{emp_name}' не оказывает услуг в филиале '{self.filial_name}'.")
            else:
                # Несколько сотрудников найдено - показываем список
                response_parts.append(f"Найдено несколько сотрудников в филиале '{self.filial_name}' с именем, содержащим '{self.employee_name}':")
                for emp_id, emp_data in paginated_employees:
                    target_filial_services = emp_data['services'].get(self.filial_name, set())
                    services_count = len(target_filial_services)
                    response_parts.append(f"- {emp_data['name']} ({services_count} услуг)")
                
        else:
            # Режим 3 и 4: Общий поиск
            page_info = f" (страница {self.page_number} из {(total_found + self.page_size - 1) // self.page_size})" if total_found > self.page_size else ""
            response_parts.append(f"Найдены следующие сотрудники{page_info}:")
            
            for emp_id, emp_data in paginated_employees:
                emp_name = emp_data['name']
                emp_info = f"- {emp_name}"
                
                # Показываем филиалы
                filials = sorted(list(emp_data['filials']), key=normalize_text)
                if filials:
                    emp_info += f"\n   Филиалы: {', '.join(filials)}"
                
                # Показываем услуги (ограниченно)
                all_services = set()
                for filial_services in emp_data['services'].values():
                    all_services.update(filial_services)
                
                if all_services:
                    sorted_services = sorted(list(all_services), key=lambda s: normalize_text(s, keep_spaces=True))
                    if len(sorted_services) <= 5:
                        emp_info += f"\n   Услуги: {', '.join(sorted_services)}"
                    else:
                        emp_info += f"\n   Услуги: {', '.join(sorted_services[:5])} (и еще {len(sorted_services) - 5})"
                
                response_parts.append(emp_info)

        # Добавляем информацию о дополнительных страницах
        if end_idx < total_found:
            response_parts.append(f"\n... показано {len(paginated_employees)} из {total_found} сотрудников. Используйте page_number={self.page_number + 1} для следующей страницы.")

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

        # Нормализованные запросы для поиска
        normalized_service_query = normalize_text(self.service_name, keep_spaces=True).lower()
        normalized_filial_query = normalize_text(self.filial_name, keep_spaces=True).lower() if self.filial_name else None
        
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
            logger.warning(f"[GetServicePrice] Услуга '{self.service_name}' не найдена")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        
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

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[CheckServiceInFilial] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Проверка услуги '{self.service_name}' в филиале '{self.filial_name}', В процессе записи: {self.in_booking_process}, Tenant: {_tenant_id_for_clinic_data}")

        # Нормализованные запросы для поиска
        normalized_service_query = normalize_text(self.service_name, keep_spaces=True).lower()
        normalized_filial_query = normalize_text(self.filial_name, keep_spaces=True).lower()
        
        # Найденные услуги и филиалы для отображения
        found_service_name = None
        found_filial_name = None
        service_exists_in_filial = False
        
        # Ищем прямые совпадения в данных
        for item in _clinic_data:
            service_name = item.get("serviceName")
            filial_name = item.get("filialName")
            
            if not service_name or not filial_name:
                continue
                
            normalized_item_service = normalize_text(service_name, keep_spaces=True).lower()
            normalized_item_filial = normalize_text(filial_name, keep_spaces=True).lower()
            
            # Проверяем совпадение услуги (точное или по подстроке)
            service_matches = (
                normalized_item_service == normalized_service_query or 
                normalized_service_query in normalized_item_service
            )
            
            # Проверяем совпадение филиала (точное или по подстроке)
            filial_matches = (
                normalized_item_filial == normalized_filial_query or 
                normalized_filial_query in normalized_item_filial
            )
            
            # Если нашли услугу, запоминаем её
            if service_matches and not found_service_name:
                found_service_name = service_name
                
            # Если нашли филиал, запоминаем его
            if filial_matches and not found_filial_name:
                found_filial_name = filial_name
                
            # Если нашли и услугу, и филиал в одной записи - это искомое совпадение
            if service_matches and filial_matches:
                service_exists_in_filial = True
                found_service_name = service_name
                found_filial_name = filial_name
                break
        
        # Формируем результат
        if not found_service_name:
            logger.warning(f"[CheckServiceInFilial] Услуга '{self.service_name}' не найдена")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
            
        if not found_filial_name:
            logger.warning(f"[CheckServiceInFilial] Филиал '{self.filial_name}' не найден")
            return f"Филиал с названием, похожим на '{self.filial_name}', не найден."
        
        # Создаем уточнения, если найденные названия отличаются от запрошенных
        service_clarification = ""
        if normalize_text(found_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True):
            if self.in_booking_process:
                service_clarification = f" (автоматически подобрана похожая услуга: '{found_service_name}')"
            else:
                service_clarification = f" (уточнено до '{found_service_name}')"
                
        filial_clarification = ""
        if normalize_text(found_filial_name, keep_spaces=True) != normalize_text(self.filial_name, keep_spaces=True):
            filial_clarification = f" (уточнено до '{found_filial_name}')"

        if service_exists_in_filial:
            return f"Да, услуга '{self.service_name}'{service_clarification} доступна в филиале '{self.filial_name}'{filial_clarification}."
        else:
            return f"Нет, услуга '{self.service_name}'{service_clarification} не найдена в филиале '{self.filial_name}'{filial_clarification}."


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

        # Нормализованный запрос для поиска
        normalized_query = normalize_text(self.service_name, keep_spaces=True).lower()
        
        # Найденные услуги по приоритету совпадения
        exact_matches = []
        substring_matches = []
        
        # Ищем совпадения в данных
        for item in _clinic_data:
            service_name = item.get("serviceName")
            if not service_name:
                continue
                
            normalized_service_name = normalize_text(service_name, keep_spaces=True).lower()
            
            # Точное совпадение
            if normalized_service_name == normalized_query:
                exact_matches.append(item)
            # Совпадение по подстроке
            elif normalized_query in normalized_service_name:
                substring_matches.append(item)
        
        # Выбираем лучшие совпадения
        matching_services = exact_matches if exact_matches else substring_matches
        
        if not matching_services:
            logger.warning(f"[FindServiceLocations] Услуга '{self.service_name}' не найдена")
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        
        # Применяем фильтрацию по ключевым словам
        keyword_filters = {
            "колен": "колени",
            "голен": "голени", 
            "бедр": "бедра",
            "рук": "руки"
        }
        
        filtered_services = matching_services
        for keyword, display_word in keyword_filters.items():
            if keyword in self.service_name.lower():
                keyword_services = [s for s in filtered_services if keyword in s.get('serviceName', '').lower()]
                if keyword_services:
                    filtered_services = keyword_services
                    logger.info(f"[FindServiceLocations] Фильтрация по '{keyword}': выбрано {len(filtered_services)} вариантов")
        
        # Обработка множественных результатов
        if len(filtered_services) > 1:
            if self.in_booking_process:
                # В процессе записи выбираем наиболее подходящую услугу
                best_service = self._select_best_service(filtered_services)
                filtered_services = [best_service]
                logger.info(f"[FindServiceLocations] Автоматически выбрана услуга '{best_service.get('serviceName')}'")
            else:
                # При обычном запросе предлагаем выбор
                return self._format_service_selection(filtered_services)
        
        # Собираем филиалы для найденных услуг
        locations = {}  # normalized_name -> original_name
        selected_service_names = {s.get('serviceName') for s in filtered_services}
        
        for item in _clinic_data:
            item_service_name = item.get('serviceName')
            filial_name = item.get('filialName')
            
            if (item_service_name in selected_service_names and 
                item_service_name and filial_name):
                norm_filial = normalize_text(filial_name)
                if norm_filial not in locations:
                    locations[norm_filial] = filial_name
        
        if not locations:
            display_service_name = filtered_services[0].get('serviceName') or self.service_name
            return f"Услуга '{display_service_name}' найдена, но информация о филиалах отсутствует."
        
        # Формируем результат
        display_service_name = filtered_services[0].get('serviceName') or self.service_name
        service_clarification = ""
        
        if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True):
            if self.in_booking_process:
                service_clarification = f" (автоматически подобрана похожая услуга: '{display_service_name}')"
            else:
                service_clarification = f" (найдена похожая услуга: '{display_service_name}')"
        
        sorted_filials = sorted(locations.values(), key=normalize_text)
        result_intro = f"Услуга '{self.service_name}'{service_clarification} доступна в филиалах:"
        return f"{result_intro}\n*   " + "\n*   ".join(sorted_filials)
    
    def _select_best_service(self, services):
        """Выбирает наиболее подходящую услугу из списка."""
        query_words = self.service_name.lower().split()
        max_matches = 0
        best_service = services[0]
        
        for service in services:
            service_name = service.get('serviceName', '').lower()
            matches = sum(1 for word in query_words if word in service_name)
            
            if matches > max_matches:
                max_matches = matches
                best_service = service
                
        return best_service
    
    def _format_service_selection(self, services):
        """Форматирует список услуг для выбора пользователем."""
        response_parts = [f"Для запроса '{self.service_name}' найдено {len(services)} похожих услуг:"]
        
        for i, service in enumerate(services, 1):
            response_parts.append(f"{i}. {service.get('serviceName')}")
        
        response_parts.append("\nПожалуйста, уточните, расположение какой именно услуги вы хотите найти, указав её название или номер из списка.")
        return "\n".join(response_parts)


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


# Оставляем без изменений функции, которые используют ID (GetSlots, BookAppointment)
# а также все остальные функции...

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
                self.from_date, self.to_date
            ))
            
            if not slots_data or not slots_data.get('result'):
                return f"Нет доступных слотов для записи к '{self.employee_name}' на услугу '{self.service_name}'."
            
            result = slots_data['result']
            response_parts = [f"Доступные слоты для записи к '{self.employee_name}' на услугу '{self.service_name}' в филиале '{self.filial_name}':"]
            
            for date_slot in result:
                date = date_slot.get('date', 'Неизвестная дата')
                slots = date_slot.get('slots', [])
                
                if slots:
                    slot_times = [slot.get('time', 'Неизвестное время') for slot in slots]
                    response_parts.append(f"\n{date}: {', '.join(slot_times)}")
            
            return '\n'.join(response_parts)
            
        except Exception as e:
            logger.error(f"[GetSlots] Ошибка при получении слотов: {e}")
            return f"Ошибка при получении доступных слотов: {str(e)}"


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
                self.appointment_date, self.appointment_time
            ))
            
            if booking_result and booking_result.get('success'):
                return f"Запись успешно создана: {self.client_name} записан к {self.employee_name} на {self.appointment_date} в {self.appointment_time} на услугу '{self.service_name}' в филиале '{self.filial_name}'."
            else:
                error_msg = booking_result.get('error', 'Неизвестная ошибка') if booking_result else 'Неизвестная ошибка'
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

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[ListServicesInFilial] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос всех услуг в филиале: {self.filial_name}, Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        # Ищем филиал напрямую в данных
        normalized_filial_query = normalize_text(self.filial_name, keep_spaces=True).lower()
        found_filial_name = None
        
        # Найдем правильное название филиала
        for item in _clinic_data:
            filial_name = item.get('filialName')
            if filial_name:
                normalized_filial = normalize_text(filial_name, keep_spaces=True).lower()
                if (normalized_filial == normalized_filial_query or 
                    normalized_filial_query in normalized_filial):
                    found_filial_name = filial_name
                    break
        
        if not found_filial_name:
            all_filials_db_orig: Set[str] = set()
            if _clinic_data:
                for item_data in _clinic_data:
                    f_name_val = item_data.get('filialName')
                    if f_name_val: all_filials_db_orig.add(f_name_val)
            suggestion = f"Доступные филиалы: {', '.join(sorted(list(all_filials_db_orig), key=normalize_text))}." if all_filials_db_orig else "Список филиалов в базе пуст."
            return f"Филиал '{self.filial_name}' не найден. {suggestion}".strip()

        # Собираем услуги в найденном филиале
        services_in_filial: Set[str] = set()
        normalized_found_filial = normalize_text(found_filial_name, keep_spaces=True).lower()
        
        for item in _clinic_data:
            item_filial_name = item.get('filialName')
            if item_filial_name:
                normalized_item_filial = normalize_text(item_filial_name, keep_spaces=True).lower()
                if normalized_item_filial == normalized_found_filial:
                    srv_name_raw = item.get('serviceName')
                    if srv_name_raw: 
                        services_in_filial.add(srv_name_raw)

        if not services_in_filial:
            return f"В филиале '{found_filial_name}' не найдено информации об услугах."
        else:
            sorted_services = sorted(list(services_in_filial), key=lambda s: normalize_text(s, keep_spaces=True))
            
            start_index = (self.page_number - 1) * self.page_size
            end_index = start_index + self.page_size
            output_services = sorted_services[start_index:end_index]

            if not output_services and self.page_number > 1:
                return f"В филиале '{found_filial_name}' больше нет услуг для отображения (страница {self.page_number})."

            total_services = len(sorted_services)
            more_services_info = ""
            if end_index < total_services:
                remaining_services = total_services - end_index
                more_services_info = f"... и еще {remaining_services} услуг."
            
            return (f"В филиале '{found_filial_name}' доступны услуги (страница {self.page_number}):\\n* "
                   + "\\n* ".join(output_services) + f"\\n{more_services_info}".strip())


# Добавим все остальные функции (ListCategories, ListServicesInCategory, FindServicesInPriceRange и т.д.)
# Они остаются без изменений, так как уже работают с прямым поиском или используют ID там, где это необходимо

class ListCategories(BaseModel):
    """Модель для получения списка всех категорий услуг."""
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество категорий на странице")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[ListCategories] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос списка категорий, Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        all_categories_orig: Set[str] = set()
        for item_data in _clinic_data:
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
    """Модель для получения списка услуг конкретного сотрудника, опционально в конкретном филиале."""
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала (опционально, для фильтрации услуг по филиалу)")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=20, description="Количество услуг на странице")

    def process(self) -> str:
        if not _clinic_data:
            return "Ошибка: База данных клиники пуста."
        
        filial_info = f" в филиале: {self.filial_name}" if self.filial_name else ""
        logger.info(f"[FC Proc] Запрос услуг сотрудника: {self.employee_name}{filial_info}, Page: {self.page_number}, Size: {self.page_size}")

        if not _tenant_id_for_clinic_data:
            logger.error(f"[GetEmployeeServices] _tenant_id_for_clinic_data не установлен. Невозможно найти сотрудника '{self.employee_name}'.")
            return "Системная ошибка: не удалось определить идентификатор клиники для поиска сотрудника."

        # Получаем ID сотрудника
        employee_id_found = get_id_by_name(_tenant_id_for_clinic_data, 'employee', self.employee_name)
        if not employee_id_found:
            logger.warning(f"[GetEmployeeServices] Сотрудник с именем '{self.employee_name}' не найден для тенанта {_tenant_id_for_clinic_data}.")
            return f"Сотрудник с именем, похожим на '{self.employee_name}', не найден."

        # Получаем ID филиала (если указан)
        filial_id_found = None
        actual_filial_name_from_db = None
        if self.filial_name:
            filial_id_found = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
            if not filial_id_found:
                logger.warning(f"[GetEmployeeServices] Филиал с именем '{self.filial_name}' не найден для тенанта {_tenant_id_for_clinic_data}.")
                return f"Филиал с названием, похожим на '{self.filial_name}', не найден."
            actual_filial_name_from_db = get_name_by_id(_tenant_id_for_clinic_data, 'filial', filial_id_found) or self.filial_name
        
        # Получаем настоящее имя сотрудника из базы данных
        actual_employee_name_from_db = get_name_by_id(_tenant_id_for_clinic_data, 'employee', employee_id_found) or self.employee_name
        
        if self.filial_name:
            logger.info(f"[GetEmployeeServices] Найден сотрудник: '{actual_employee_name_from_db}' (ID: '{employee_id_found}') в филиале: '{actual_filial_name_from_db}' (ID: '{filial_id_found}') для тенанта {_tenant_id_for_clinic_data}.")
        else:
            logger.info(f"[GetEmployeeServices] Найден сотрудник: '{actual_employee_name_from_db}' (ID: '{employee_id_found}') для тенанта {_tenant_id_for_clinic_data}.")

        # Собираем только уникальные названия услуг для данного сотрудника (опционально в указанном филиале)
        services_for_employee: Set[str] = set()
        
        for item in _clinic_data:
            if item.get('employeeId') == employee_id_found:
                # Если указан филиал, фильтруем по нему
                if self.filial_name and item.get('filialId') != filial_id_found:
                    continue
                    
                service_name = item.get('serviceName')
                if service_name:
                    services_for_employee.add(service_name)

        if not services_for_employee:
            if self.filial_name:
                return f"Не найдено услуг для сотрудника '{actual_employee_name_from_db}' в филиале '{actual_filial_name_from_db}'. Возможно, данный сотрудник не работает в этом филиале или не оказывает услуг."
            else:
                return f"Не найдено услуг для сотрудника '{actual_employee_name_from_db}'. Возможно, данный сотрудник не оказывает услуг."

        # Сортируем услуги и применяем пагинацию
        sorted_services = sorted(list(services_for_employee))
        total_services = len(sorted_services)
        
        # Вычисляем индексы для пагинации
        start_index = (self.page_number - 1) * self.page_size
        end_index = start_index + self.page_size
        
        if start_index >= total_services:
            return f"Страница {self.page_number} не существует. Всего услуг: {total_services}, услуг на странице: {self.page_size}."
        
        paginated_services = sorted_services[start_index:end_index]
        
        # Формируем ответ
        if self.filial_name:
            response_parts = [f"Услуги сотрудника '{actual_employee_name_from_db}' в филиале '{actual_filial_name_from_db}':"]
        else:
            response_parts = [f"Услуги сотрудника '{actual_employee_name_from_db}':"]
        
        for i, service_name in enumerate(paginated_services, 1):
            global_index = start_index + i
            response_parts.append(f"  {global_index}. {service_name}")
        
        # Добавляем информацию о пагинации
        total_pages = (total_services + self.page_size - 1) // self.page_size
        if total_pages > 1:
            response_parts.append(f"\nСтраница {self.page_number} из {total_pages} (всего услуг: {total_services})")
            if self.page_number < total_pages:
                response_parts.append("Для просмотра следующих услуг укажите page_number={}.".format(self.page_number + 1))
        
        return "\n".join(response_parts)


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
