# clinic_functions.py

import logging
import re
from typing import Optional, List, Dict, Any, Set, Tuple
from pydantic import BaseModel, Field
from client_data_service import get_free_times_of_employee_by_services, add_record
from clinic_index import get_name_by_id, normalize_text, get_id_by_name
import asyncio
from datetime import datetime

# --- Глобальное хранилище данных ---
_clinic_data: List[Dict[str, Any]] = []
_tenant_id_for_clinic_data: Optional[str] = None
logger = logging.getLogger(__name__)

# --- ДОБАВЛЕНО: Вспомогательная функция для склонения числительных ---
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

# --- Функция нормализации ---
def normalize_text(text: Optional[str], keep_spaces: bool = False) -> str:
    """
    Приводит строку к нижнему регистру, удаляет дефисы и опционально пробелы.
    Безопасно обрабатывает None, возвращая пустую строку.

    Args:
        text: Входная строка или None.
        keep_spaces: Если True, пробелы внутри строки сохраняются (но удаляются по краям).
                     Если False (по умолчанию), все пробелы удаляются.

    Returns:
        Нормализованная строка.
    """
    if not text: # Обрабатывает None и пустые строки
        return ""
    normalized = text.lower().replace("-", "")
    if keep_spaces:
        normalized = re.sub(r'\s+', ' ', normalized).strip()
    else:
        normalized = normalized.replace(" ", "")
    return normalized

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
    """Модель для поиска сотрудников по различным критериям."""
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")
    page_number: int = Field(default=1, description="Номер страницы (начиная с 1)")
    page_size: int = Field(default=15, description="Количество сотрудников на странице")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[FindEmployees] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."
        
        logger.info(f"[FC Proc] Поиск сотрудников (Сотрудник: '{self.employee_name}', Услуга: '{self.service_name}', Филиал: '{self.filial_name}'), Tenant: {_tenant_id_for_clinic_data}, Page: {self.page_number}, Size: {self.page_size}")

        target_filial_id: Optional[str] = None
        display_filial_name_query = self.filial_name
        if self.filial_name:
            target_filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
            if not target_filial_id:
                return f"Филиал с названием, похожим на '{self.filial_name}', не найден."
            display_filial_name_query = get_name_by_id(_tenant_id_for_clinic_data, 'filial', target_filial_id) or self.filial_name

        target_service_id: Optional[str] = None
        display_service_name_query = self.service_name
        if self.service_name:
            target_service_id = get_id_by_name(_tenant_id_for_clinic_data, 'service', self.service_name)
            if not target_service_id:
                return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
            display_service_name_query = get_name_by_id(_tenant_id_for_clinic_data, 'service', target_service_id) or self.service_name

        employees_info: Dict[str, Dict[str, Any]] = {}
        norm_employee_name_query = normalize_text(self.employee_name, keep_spaces=True, sort_words=True) if self.employee_name else None

        for item in _clinic_data:
            e_id = item.get('employeeId')
            e_name_raw = item.get('employeeFullName')
            if not e_id or not e_name_raw: continue

            if norm_employee_name_query and norm_employee_name_query not in normalize_text(e_name_raw, keep_spaces=True, sort_words=True):
                continue
            if target_filial_id and item.get('filialId') != target_filial_id:
                continue
            if target_service_id and item.get('serviceId') != target_service_id:
                continue
            
            if e_id not in employees_info:
                employees_info[e_id] = {
                    'name': e_name_raw,
                    'services': set(),
                    'filials': set(),
                    'categories': set()
                }
            
            s_name_raw = item.get('serviceName')
            f_name_raw = item.get('filialName')
            cat_name_raw = item.get('categoryName')

            if s_name_raw: employees_info[e_id]['services'].add(s_name_raw)
            if f_name_raw: employees_info[e_id]['filials'].add(f_name_raw)
            if cat_name_raw: employees_info[e_id]['categories'].add(cat_name_raw)

        if not employees_info:
            search_terms_list = []
            if self.employee_name: search_terms_list.append(f"имя содержит '{self.employee_name}'")
            if display_service_name_query: search_terms_list.append(f"услуга '{display_service_name_query}'")
            if display_filial_name_query: search_terms_list.append(f"филиал '{display_filial_name_query}'")
            search_terms_str = ", ".join(search_terms_list) if search_terms_list else "указанным критериям"
            return f"Сотрудники, соответствующие {search_terms_str}, не найдены."

        if self.filial_name and not self.employee_name and not self.service_name:
            employee_names_in_filial: Set[str] = set()
            # В этом режиме мы уже отфильтровали по filialId при сборе employees_info, 
            # но для чистоты можно еще раз пройтись по _clinic_data или использовать employees_info
            # Проще использовать employees_info, так как там уже есть ID сотрудников, работающих в этом филиале (если он был задан)
            if target_filial_id: # Убедимся, что target_filial_id был задан
                for emp_id_in_filial in employees_info: # employees_info уже содержит только тех, кто в нужном филиале
                    employee_names_in_filial.add(employees_info[emp_id_in_filial]['name'])
            else: # Этого случая быть не должно, т.к. (self.filial_name) истинно
                 return f"Логическая ошибка: filial_name указан, но target_filial_id не определен."

            if not employee_names_in_filial:
                return f"В филиале '{display_filial_name_query}' не найдено сотрудников (после всех фильтров)."

            sorted_employee_names = sorted(list(employee_names_in_filial), key=lambda n: normalize_text(n, keep_spaces=True, sort_words=True))
            total_employees_in_filial = len(sorted_employee_names)
            
            start_index = (self.page_number - 1) * self.page_size
            end_index = start_index + self.page_size
            paginated_employee_names = sorted_employee_names[start_index:end_index]

            if not paginated_employee_names and self.page_number > 1:
                return f"В филиале '{display_filial_name_query}' больше нет сотрудников для отображения (страница {self.page_number})."

            count_form = get_readable_count_form(total_employees_in_filial, ("сотрудник", "сотрудника", "сотрудников"))
            page_info = f"(страница {self.page_number} из {(total_employees_in_filial + self.page_size - 1) // self.page_size if self.page_size > 0 else 1})" if total_employees_in_filial > self.page_size else ""
            response_start = f"В филиале '{display_filial_name_query}' работает {total_employees_in_filial} {count_form} {page_info}:\\n- "
            response_list = "\\n- ".join(paginated_employee_names)
            
            more_info = ""
            if end_index < total_employees_in_filial:
                remaining = total_employees_in_filial - end_index
                more_info = f"\\n... и еще {remaining} {get_readable_count_form(remaining, ('сотрудник', 'сотрудника', 'сотрудников'))}."
            return response_start + response_list + more_info

        response_parts = []
        sorted_employees_ids = sorted(employees_info.keys(), key=lambda eid: normalize_text(employees_info[eid].get('name'), keep_spaces=True, sort_words=True))
        
        total_found_employees = len(sorted_employees_ids)
        start_index = (self.page_number - 1) * self.page_size
        end_index = start_index + self.page_size
        paginated_employee_ids = sorted_employees_ids[start_index:end_index]

        if not paginated_employee_ids and self.page_number > 1:
            return f"Больше нет сотрудников, соответствующих вашему запросу, для отображения (страница {self.page_number})."

        page_clarification = f" (страница {self.page_number} из { (total_found_employees + self.page_size - 1) // self.page_size if self.page_size > 0 else 1})" if total_found_employees > self.page_size else ""

        response_parts.append(f"Найденные сотрудники{page_clarification}:")

        for emp_id in paginated_employee_ids:
            emp_data = employees_info[emp_id]
            name = emp_data.get('name')
            if not name: continue

            services = sorted(list(emp_data.get('services', set())), key=lambda s: normalize_text(s, keep_spaces=True))
            filials = sorted(list(emp_data.get('filials', set())), key=normalize_text)
            
            emp_details = [f"**{name}**"]
            # В обычном режиме показываем услуги и филиалы только если они были частью запроса или не было фильтров
            if self.service_name or not target_service_id: # Показать услуги если искали по услуге или не фильтровали по услуге
                if services: emp_details.append(f"  Услуги: { '; '.join(services)}")
            if self.filial_name or not target_filial_id: # Показать филиалы если искали по филиалу или не фильтровали по филиалу
                 if filials: emp_details.append(f"  Филиалы: {', '.join(filials)}")
            
            # Если фильтровали и по услуге и по филиалу, то услуги и филиалы уже будут специфичны для этого контекста
            # Если не было фильтров, то показываем все услуги/филиалы сотрудника
            # Если был только один фильтр (услуга или филиал), то показываем соответствующую информацию
            # Логика выше уже должна это покрывать, но можно сделать ее еще более явной.
            # Пример: если искали по филиалу, но не по услуге, то servies будет всеми услугами сотрудника в этом филиале
            # А filials будет только этим филиалом.

            response_parts.append("\n".join(emp_details))

        if end_index < total_found_employees:
            remaining = total_found_employees - end_index
            response_parts.append(f"... еще {remaining} {get_readable_count_form(remaining, ('сотрудник', 'сотрудника', 'сотрудников'))} можно отобразить на следующих страницах.")
        
        return "\n\n".join(response_parts)


class GetServicePrice(BaseModel):
    """Модель для получения цены на конкретную услугу."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[GetServicePrice] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос цены (Услуга: {self.service_name}, Филиал: {self.filial_name}), Tenant: {_tenant_id_for_clinic_data}")

        service_id = get_id_by_name(_tenant_id_for_clinic_data, 'service', self.service_name)
        if not service_id:
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        display_service_name = get_name_by_id(_tenant_id_for_clinic_data, 'service', service_id) or self.service_name

        target_filial_id: Optional[str] = None
        display_filial_name = self.filial_name
        if self.filial_name:
            target_filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
            if not target_filial_id:
                return f"Филиал с названием, похожим на '{self.filial_name}', не найден при поиске цены на '{display_service_name}'."
            display_filial_name = get_name_by_id(_tenant_id_for_clinic_data, 'filial', target_filial_id) or self.filial_name
        
        candidate_prices: List[Dict[str, Any]] = []
        for item in _clinic_data:
            if item.get('serviceId') == service_id:
                price_raw = item.get('price')
                if price_raw is None or price_raw == '': continue
                try:
                    price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                except (ValueError, TypeError): continue

                item_filial_id = item.get('filialId')
                item_filial_name_from_db = get_name_by_id(_tenant_id_for_clinic_data, 'filial', item_filial_id) if item_filial_id else "(не указан)"

                # Если запрошен конкретный филиал, и он не совпадает с текущим, пропускаем
                if target_filial_id and item_filial_id != target_filial_id:
                    continue
                
                candidate_prices.append({
                    'price': price,
                    'filial_id': item_filial_id,
                    'filial_name': item_filial_name_from_db
                })

        if not candidate_prices:
            filial_context = f" в филиаle '{display_filial_name}'" if target_filial_id else ""
            return f"Цена на услугу '{display_service_name}'{filial_context} не найдена."

        # Если был запрос для конкретного филиала и нашлись цены (должна быть одна или ни одной после фильтра выше)
        if target_filial_id:
            if len(candidate_prices) == 1:
                price_info = candidate_prices[0]
                service_clarification = f" (уточнено до '{display_service_name}')" if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True) else ""
                filial_clarification = f" (филиал '{display_filial_name}')" 
                return f"Цена на '{self.service_name}'{service_clarification}{filial_clarification} составляет {price_info['price']:.0f} руб."
            else: # Не должно произойти, если target_filial_id был и фильтрация прошла
                logger.warning(f"Найдено {len(candidate_prices)} цен для {display_service_name} в {display_filial_name} ПОСЛЕ фильтрации. Это неожиданно.")
                return f"Найдено несколько цен для '{display_service_name}' в филиале '{display_filial_name}'. Пожалуйста, обратитесь к администратору."

        # Если филиал не был указан, а цены есть (могут быть разные для разных филиалов)
        # Группируем по филиалам, чтобы избежать дублирования, если в одном филиале одна и та же цена (маловероятно, но возможно)
        prices_by_filial: Dict[str, float] = {}
        for cp in candidate_prices:
            fid = cp['filial_id'] or "_any_filial_"
            # Берем первую попавшуюся цену для филиала, если вдруг их несколько (не должно быть)
            if fid not in prices_by_filial:
                prices_by_filial[fid] = cp['price']
        
        unique_prices_with_filials = [
            {'price': price, 'filial_name': (get_name_by_id(_tenant_id_for_clinic_data, 'filial', fid) if fid != "_any_filial_" else "(не указан)")}
            for fid, price in prices_by_filial.items()
        ]
        unique_prices_with_filials.sort(key=lambda x: (x['price'], normalize_text(x['filial_name'])))

        service_clarification = f" (уточнено до '{display_service_name}')" if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True) else ""

        if len(unique_prices_with_filials) == 1:
            price_info = unique_prices_with_filials[0]
            filial_text = f" в филиале {price_info['filial_name']}" if price_info['filial_name'] != "(не указан)" else ""
            return f"Цена на '{self.service_name}'{service_clarification}{filial_text} составляет {price_info['price']:.0f} руб."
        else:
            response_parts = [f"Цена на '{self.service_name}'{service_clarification} различается в зависимости от филиала:"]
            limit = 5
            for i, price_info in enumerate(unique_prices_with_filials):
                if i >= limit : break
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
        for item in _clinic_data:
            if item.get('employeeId') == employee_id_found:
                service_name = item.get('serviceName')
                if service_name: # Собираем только названия услуг
                    all_services_for_employee.add(service_name)
        
        if not all_services_for_employee:
            return f"Не найдено уникальных услуг для сотрудника '{actual_employee_name_from_db}'."

        sorted_unique_services = sorted(list(all_services_for_employee), key=lambda s: normalize_text(s, keep_spaces=True))
        total_unique_services = len(sorted_unique_services)

        start_index = (self.page_number - 1) * self.page_size
        end_index = start_index + self.page_size
        paginated_services = sorted_unique_services[start_index:end_index]

        if not paginated_services and self.page_number > 1:
            return f"Для сотрудника '{actual_employee_name_from_db}' больше нет услуг для отображения (страница {self.page_number})."

        # Теперь для каждой услуги из пагинированного списка найдем ее категории и филиалы
        # Это может быть не очень эффективно, если у услуги много записей, но для отображения подойдет.
        # Альтернатива - строить более сложную структуру данных при первоначальном сборе.
        output_details = []
        for service_name_paginated in paginated_services:
            categories_for_service: Set[str] = set()
            filials_for_service: Set[str] = set()
            for item in _clinic_data:
                if item.get('employeeId') == employee_id_found and item.get('serviceName') == service_name_paginated:
                    cat_name = item.get('categoryName')
                    fil_name = item.get('filialName')
                    if cat_name: categories_for_service.add(cat_name)
                    if fil_name: filials_for_service.add(fil_name)
            
            detail_str = f"**{service_name_paginated}**"
            if categories_for_service:
                detail_str += f" (Категории: { ', '.join(sorted(list(categories_for_service), key=normalize_text)) })"
            if filials_for_service:
                 detail_str += f" - Доступна в филиалах: { ', '.join(sorted(list(filials_for_service), key=normalize_text)) }"
            output_details.append(detail_str)

        name_clarification = f" (уточнено до '{actual_employee_name_from_db}')" if normalize_text(actual_employee_name_from_db, keep_spaces=True, sort_words=True) != normalize_text(self.employee_name, keep_spaces=True, sort_words=True) else ""
        page_info = f" (страница {self.page_number} из {(total_unique_services + self.page_size - 1) // self.page_size if self.page_size > 0 else 1})" if total_unique_services > self.page_size else ""

        response_intro = f"Сотрудник '{self.employee_name}'{name_clarification} выполняет следующие услуги{page_info}:"
        
        response_body = "\n- " + "\n- ".join(output_details)
        
        more_info = ""
        if end_index < total_unique_services:
            remaining = total_unique_services - end_index
            more_info = f"\n... и еще {remaining} {get_readable_count_form(remaining, ('услуга', 'услуги', 'услуг'))}."
            
        return f"{response_intro}{response_body}{more_info}"


class CheckServiceInFilial(BaseModel):
    """Модель для проверки наличия услуги в филиале."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[CheckServiceInFilial] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Проверка услуги '{self.service_name}' в филиале '{self.filial_name}', Tenant: {_tenant_id_for_clinic_data}")

        service_id = get_id_by_name(_tenant_id_for_clinic_data, 'service', self.service_name)
        if not service_id:
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена в базе."
        display_service_name = get_name_by_id(_tenant_id_for_clinic_data, 'service', service_id) or self.service_name

        filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
        if not filial_id:
            return f"Филиал с названием, похожим на '{self.filial_name}', не найден в базе."
        display_filial_name = get_name_by_id(_tenant_id_for_clinic_data, 'filial', filial_id) or self.filial_name

        service_found_in_filial = False
        for item in _clinic_data:
            if item.get('serviceId') == service_id and item.get('filialId') == filial_id:
                service_found_in_filial = True
                break
        
        service_clarification = f" (уточнено до '{display_service_name}')" if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True) else ""
        filial_clarification = f" (уточнено до '{display_filial_name}')" if normalize_text(display_filial_name, keep_spaces=True) != normalize_text(self.filial_name, keep_spaces=True) else ""

        if service_found_in_filial:
            return f"Да, услуга '{self.service_name}'{service_clarification} доступна в филиале '{self.filial_name}'{filial_clarification}."
        else:
            return f"Нет, услуга '{self.service_name}'{service_clarification} не найдена в филиале '{self.filial_name}'{filial_clarification}."


class CompareServicePriceInFilials(BaseModel):
    """Модель для сравнения цен на услугу в нескольких филиалах."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(min_length=2, description="Список из ДВУХ или БОЛЕЕ филиалов")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[CompareServicePriceInFilials] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."
        if not self.filial_names or len(self.filial_names) < 2:
            return "Ошибка: Нужно указать как минимум два названия филиала для сравнения."

        logger.info(f"[FC Proc] Сравнение цен на '{self.service_name}' в филиалах: {self.filial_names}, Tenant: {_tenant_id_for_clinic_data}")

        service_id = get_id_by_name(_tenant_id_for_clinic_data, 'service', self.service_name)
        if not service_id:
            return f"Услуга с названием, похожим на '{self.service_name}', не найдена."
        display_service_name = get_name_by_id(_tenant_id_for_clinic_data, 'service', service_id) or self.service_name

        # Словарь для хранения {input_filial_name: (found_filial_id, display_filial_name_from_db)}
        # или {input_filial_name: (None, None)} если не найден
        filial_resolution_map: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        unique_input_filial_names = sorted(list(set(f_name.strip() for f_name in self.filial_names if f_name.strip()) ), key=normalize_text)
        
        if len(unique_input_filial_names) < 2:
            return "Ошибка: Нужно указать как минимум два УНИКАЛЬНЫХ и непустых названия филиала для сравнения."

        invalid_filial_names_input: List[str] = []
        valid_filial_ids_to_query: Dict[str, str] = {} # {filial_id: display_name_from_db}

        for filial_name_input in unique_input_filial_names:
            filial_id_found = get_id_by_name(_tenant_id_for_clinic_data, 'filial', filial_name_input)
            if filial_id_found:
                db_name = get_name_by_id(_tenant_id_for_clinic_data, 'filial', filial_id_found) or filial_name_input
                filial_resolution_map[filial_name_input] = (filial_id_found, db_name)
                if filial_id_found not in valid_filial_ids_to_query:
                    valid_filial_ids_to_query[filial_id_found] = db_name
            else:
                filial_resolution_map[filial_name_input] = (None, None)
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
        
        # Эта проверка может быть излишней, если unique_input_filial_names уже проверен на >=2
        # Однако, если все уникальные имена оказались одним и тем же ID (маловероятно, но возможно), то len(valid_filial_ids_to_query) может быть < 2
        if len(valid_filial_ids_to_query) < 2:
            return "Нужно как минимум два РАЗЛИЧНЫХ корректных филиала для сравнения (после разрешения имен)."

        # {filial_id: price}
        prices_for_service_in_valid_filials: Dict[str, float] = {}
        for item_data in _clinic_data:
            if item_data.get('serviceId') == service_id:
                item_filial_id = item_data.get('filialId')
                if item_filial_id in valid_filial_ids_to_query: # Только для тех филиалов, что были запрошены и найдены
                    price_raw = item_data.get('price')
                    if price_raw is not None and price_raw != '':
                        try:
                            price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                            # Если для этого filial_id цена еще не записана, записываем
                            # (В идеале, одна услуга должна иметь одну цену в одном филиале, но для надежности)
                            if item_filial_id not in prices_for_service_in_valid_filials:
                                prices_for_service_in_valid_filials[item_filial_id] = price
                        except (ValueError, TypeError):
                            logger.warning(f"Не удалось сконвертировать цену '{price_raw}' для serviceId {service_id} в filialId {item_filial_id}")
                            continue
        
        service_clarification = f" (уточнено до '{display_service_name}')" if normalize_text(display_service_name, keep_spaces=True) != normalize_text(self.service_name, keep_spaces=True) else ""
        response_parts = [f"Сравнение цен на услугу '{self.service_name}'{service_clarification}:"]
        
        found_prices_count = 0
        # Итерируемся по УНИКАЛЬНЫМ запрошенным пользователем филиалам (которые были успешно разрешены в ID)
        # valid_filial_ids_to_query содержит {found_filial_id: display_name_from_db}
        # Нам нужно показать результат для каждого display_name_from_db, который соответствует запрошенным и валидным filial_id.
        # Используем display_names из valid_filial_ids_to_query.values() и сортируем их для консистентного вывода
        
        # Создаем список кортежей (display_name, price/None) для сортировки и вывода
        results_for_display: List[Tuple[str, Optional[float]]] = [] 
        for filial_id, display_name_db in valid_filial_ids_to_query.items():
            price_val = prices_for_service_in_valid_filials.get(filial_id)
            results_for_display.append((display_name_db, price_val))
        
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
            return f"Цена на услугу '{display_service_name}' не найдена ни в одном из запрошенных (и существующих) филиалов: {", ".join(sorted(valid_filial_ids_to_query.values(), key=normalize_text))}."
        
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

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Поиск филиалов для услуги: {self.service_name}")

        locations: Dict[str, str] = {} 
        service_found = False
        found_service_names_raw: Set[str] = set()
        norm_search_service = normalize_text(self.service_name, keep_spaces=True).strip()

        best_match_name_raw = None
        min_len_diff = float('inf')
        found_exact_match = False

        for item in _clinic_data:
            s_name_raw = item.get('serviceName')
            f_name_raw = item.get('filialName')
            if not s_name_raw or not f_name_raw: continue

            norm_item_s_name = normalize_text(s_name_raw, keep_spaces=True)
            if norm_search_service in norm_item_s_name:
                service_found = True
                found_service_names_raw.add(s_name_raw)
                norm_item_f_name = normalize_text(f_name_raw)
                if norm_item_f_name not in locations:
                     locations[norm_item_f_name] = f_name_raw

                is_exact_match = (norm_search_service == norm_item_s_name)
                if is_exact_match and not found_exact_match: 
                    found_exact_match = True
                    best_match_name_raw = s_name_raw
                    min_len_diff = 0 
                elif not found_exact_match: 
                    len_diff = abs(len(norm_item_s_name) - len(norm_search_service))
                    if len_diff < min_len_diff: 
                        min_len_diff = len_diff
                        best_match_name_raw = s_name_raw


        if not service_found:
            return f"Услуга, похожая на '{self.service_name}', не найдена ни в одном филиале."

        display_service_name_raw = best_match_name_raw if best_match_name_raw else self.service_name
        clarification = ""
        if normalize_text(display_service_name_raw, keep_spaces=True) != norm_search_service:
            clarification = f" (найдено для '{display_service_name_raw}')"

        if not locations:
            return f"Услуга '{display_service_name_raw}'{clarification} найдена, но информация о филиалах отсутствует."
        else:
            sorted_original_filials = sorted(locations.values(), key=normalize_text)
            return (f"Услуга '{display_service_name_raw}'{clarification} доступна в филиалах:\n*   "
                   + "\n*   ".join(sorted_original_filials))


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
        logger.info(f"Запрос свободных слотов: TenantID={self.tenant_id}, EmployeeID={self.employee_id}, ServiceIDs={self.service_ids}, Date={self.date_time}, FilialID={self.filial_id}")
        try:
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

