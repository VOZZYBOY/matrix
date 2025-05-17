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

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        logger.info(f"[FC Proc] Поиск сотрудников (Имя: {self.employee_name}, Услуга: {self.service_name}, Филиал: {self.filial_name}), Tenant: {_tenant_id_for_clinic_data}")

        norm_emp_name = normalize_text(self.employee_name, keep_spaces=True)
        norm_service_name = normalize_text(self.service_name, keep_spaces=True)
        # norm_filial_name = normalize_text(self.filial_name) # Больше не нормализуем здесь, get_id_by_name сделает это

        target_filial_id: Optional[str] = None
        if self.filial_name:
            if not _tenant_id_for_clinic_data:
                logger.error("[FindEmployees] _tenant_id_for_clinic_data не установлен, не могу искать ID филиала.")
                return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден для поиска филиала)."
            
            target_filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
            if not target_filial_id:
                return f"Филиал с названием, похожим на '{self.filial_name}', не найден."
            logger.info(f"Найден ID филиала '{target_filial_id}' для имени '{self.filial_name}'.")

        filtered_data = []
        for item in _clinic_data:
            item_emp_name_raw = item.get('employeeFullName')
            item_service_name_raw = item.get('serviceName')
            # item_filial_name_raw = item.get('filialName') # Не нужен для прямого сравнения
            item_filial_id_raw = item.get('filialId')

            norm_item_emp = normalize_text(item_emp_name_raw, keep_spaces=True)
            norm_item_service = normalize_text(item_service_name_raw, keep_spaces=True)
            # norm_item_filial = normalize_text(item_filial_name_raw) # Не нужен

            emp_match = (not norm_emp_name or (norm_item_emp and norm_emp_name in norm_item_emp))
            
            # filial_match = (not norm_filial_name or (norm_item_filial and norm_filial_name == norm_item_filial)) # Старая логика
            # Новая логика для filial_match:
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
        sorted_employees = sorted(employees_info.values(), key=lambda x: normalize_text(x.get('name'), keep_spaces=True))

        for emp in sorted_employees:
            name = emp.get('name')
            if not name: continue

            # Получаем и сортируем оригинальные имена услуг и филиалов
            services = sorted(list(emp.get('services', set())), key=lambda s: normalize_text(s, keep_spaces=True))
            filials = sorted(list(emp.get('filials', set())), key=normalize_text)

            # Пропускаем, если обязательные фильтры не дали результатов для этого сотрудника
            if norm_service_name and not services: continue
            if target_filial_id and not filials: # Если искали филиал и его нет у сотрудника после всех фильтров
                 # Это условие может быть избыточным, так как filtered_data уже должно содержать только нужный филиал
                 # Однако, если сотрудник мог быть в filtered_data по другим причинам, но его филиал не тот, что искали - пропускаем.
                 # Но лучше убедиться, что filials содержит ТОЛЬКО target_filial_id (или пусто, если не совпало)
                 # На самом деле, если target_filial_id есть, то в filials должен быть только ОДИН филиал (или ни одного).
                 # employees_info[e_id]['filials'].add(f_name) уже учитывает это.
                 # Поэтому, если target_filial_id задан, а filials пуст, значит, сотрудник не работает в этом филиале по данным.
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

    def process(self) -> str:
        if not _clinic_data:
            return "Ошибка: База данных клиники пуста."
        logger.info(f"[FC Proc] Запрос услуг сотрудника: {self.employee_name}")

        if not _tenant_id_for_clinic_data:
            logger.error(f"[GetEmployeeServices] _tenant_id_for_clinic_data не установлен. Невозможно найти сотрудника '{self.employee_name}'.")
            return "Системная ошибка: не удалось определить идентификатор клиники для поиска сотрудника."

        employee_id_found = get_id_by_name(_tenant_id_for_clinic_data, 'employee', self.employee_name)

        if not employee_id_found:
            logger.warning(f"[GetEmployeeServices] Сотрудник с именем '{self.employee_name}' (ID не найден) не найден для тенанта {_tenant_id_for_clinic_data}.")
            return f"Сотрудник с именем, похожим на '{self.employee_name}', не найден."
        
        logger.info(f"[GetEmployeeServices] Найден сотрудник: '{self.employee_name}' с ID: '{employee_id_found}' для тенанта {_tenant_id_for_clinic_data}.")

        employee_services: Dict[str, List[str]] = {}
        filials_of_employee: Set[str] = set()
        actual_employee_name_from_db = None

        for item in _clinic_data:
            if item.get('employeeId') == employee_id_found:
                if not actual_employee_name_from_db:
                    actual_employee_name_from_db = item.get('employeeFullName', self.employee_name) # Сохраняем имя из базы
                
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
            return f"Следующие запрошенные филиалы не найдены: {", ".join(invalid_filial_names_input)}. Доступные филиалы в базе: {existing_filials_str}."
        
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

    def process(self) -> str:
        if not self.query_term or not self.filial_name: return "Ошибка: Укажите услугу/категорию и филиал."
        if not _clinic_data: return "Ошибка: Данные клиники не загружены."
        if not _tenant_id_for_clinic_data:
            logger.error("[FindSpecialistsByServiceOrCategoryAndFilial] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Поиск специалистов (Запрос: '{self.query_term}', Филиал: '{self.filial_name}'), Tenant: {_tenant_id_for_clinic_data}")

        filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
        if not filial_id:
            return f"Филиал с названием, похожим на '{self.filial_name}', не найден."
        
        # Пытаемся получить оригинальное имя филиала для вывода, если оно было найдено
        original_filial_display_name = get_name_by_id(_tenant_id_for_clinic_data, 'filial', filial_id) or self.filial_name

        service_id_match = get_id_by_name(_tenant_id_for_clinic_data, 'service', self.query_term)
        category_id_match = None
        query_type = ""
        resolved_query_term_name = self.query_term # Для отображения в ответе

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
                    matching_employees[emp_id] = emp_name_raw

        if not matching_employees:
            return f"В филиале '{original_filial_display_name}' не найдено специалистов для '{resolved_query_term_name}'."
        else:
            employee_list = ", ".join(sorted(matching_employees.values(), key=lambda n: normalize_text(n, keep_spaces=True)))
            return f"В филиале '{original_filial_display_name}' для '{resolved_query_term_name}' найдены: {employee_list}."


class ListServicesInCategory(BaseModel):
    """Модель для получения списка услуг в конкретной категории."""
    category_name: str = Field(description="Точное название категории")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[ListServicesInCategory] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос услуг в категории: {self.category_name}, Tenant: {_tenant_id_for_clinic_data}")

        category_id = get_id_by_name(_tenant_id_for_clinic_data, 'category', self.category_name)
        if not category_id:
            return f"Категория с названием, похожим на '{self.category_name}', не найдена."

        # Получаем оригинальное имя категории для отображения
        display_category_name = get_name_by_id(_tenant_id_for_clinic_data, 'category', category_id) or self.category_name

        services_in_category: Set[str] = set()
        for item in _clinic_data:
            if item.get('categoryId') == category_id:
                srv_name_raw = item.get('serviceName')
                if srv_name_raw: services_in_category.add(srv_name_raw)

        if not services_in_category:
            return f"В категории '{display_category_name}' не найдено конкретных услуг."
        else:
            # Выясняем, было ли имя категории уточнено (если display_category_name отличается от self.category_name)
            clarification = ""
            if normalize_text(display_category_name, keep_spaces=True) != normalize_text(self.category_name, keep_spaces=True):
                clarification = f" (найдено по запросу '{self.category_name}')"

            limit = 20
            sorted_services = sorted(list(services_in_category), key=lambda s: normalize_text(s, keep_spaces=True))
            output_services = sorted_services[:limit]
            more_services_info = f"... и еще {len(sorted_services) - limit} услуг." if len(sorted_services) > limit else ""
            return (f"В категорию '{display_category_name}'{clarification} входят услуги:\n* "
                   + "\n* ".join(output_services) + f"\n{more_services_info}".strip())


class ListServicesInFilial(BaseModel):
    """Модель для получения списка всех услуг в конкретном филиале."""
    filial_name: str = Field(description="Точное название филиала")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[ListServicesInFilial] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Запрос всех услуг в филиале: {self.filial_name}, Tenant: {_tenant_id_for_clinic_data}")

        filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
        if not filial_id:
            # Если филиал не найден, попытаемся собрать список ВСЕХ филиалов из базы для подсказки
            all_filials_db_orig: Set[str] = set()
            if _clinic_data: # Проверяем, что _clinic_data вообще есть
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
            limit = 25
            sorted_services = sorted(list(services_in_filial), key=lambda s: normalize_text(s, keep_spaces=True))
            output_services = sorted_services[:limit]
            more_services_info = f"... и еще {len(sorted_services) - limit} услуг." if len(sorted_services) > limit else ""
            return (f"В филиале '{display_filial_name}' доступны услуги:\n* "
                   + "\n* ".join(output_services) + f"\n{more_services_info}".strip())


class FindServicesInPriceRange(BaseModel):
    """Модель для поиска услуг в заданном ценовом диапазоне."""
    min_price: float = Field(description="Минимальная цена")
    max_price: float = Field(description="Максимальная цена")
    category_name: Optional[str] = Field(default=None, description="Опционально: категория")
    filial_name: Optional[str] = Field(default=None, description="Опционально: филиал")

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        if not _tenant_id_for_clinic_data:
            logger.error("[FindServicesInPriceRange] _tenant_id_for_clinic_data не установлен.")
            return "Ошибка: Внутренняя ошибка конфигурации (tenant_id не найден)."

        logger.info(f"[FC Proc] Поиск услуг по цене ({self.min_price}-{self.max_price}), Кат: {self.category_name}, Фил: {self.filial_name}, Tenant: {_tenant_id_for_clinic_data}")

        if self.min_price > self.max_price: return "Ошибка: Минимальная цена больше максимальной."

        target_category_id: Optional[str] = None
        display_category_name = self.category_name
        if self.category_name:
            target_category_id = get_id_by_name(_tenant_id_for_clinic_data, 'category', self.category_name)
            if not target_category_id:
                return f"Категория с названием, похожим на '{self.category_name}', не найдена."
            display_category_name = get_name_by_id(_tenant_id_for_clinic_data, 'category', target_category_id) or self.category_name

        target_filial_id: Optional[str] = None
        display_filial_name = self.filial_name
        if self.filial_name:
            target_filial_id = get_id_by_name(_tenant_id_for_clinic_data, 'filial', self.filial_name)
            if not target_filial_id:
                return f"Филиал с названием, похожим на '{self.filial_name}', не найден."
            display_filial_name = get_name_by_id(_tenant_id_for_clinic_data, 'filial', target_filial_id) or self.filial_name

        matching_services: Dict[str, Dict] = {}
        for item in _clinic_data:
            price_raw = item.get('price')
            srv_id = item.get('serviceId')
            srv_name_raw = item.get('serviceName')
            # cat_name_raw = item.get('categoryName') # Не используется напрямую для фильтрации
            # f_name_raw = item.get('filialName')   # Не используется напрямую для фильтрации
            item_category_id = item.get('categoryId')
            item_filial_id = item.get('filialId')

            if not srv_id or not srv_name_raw or price_raw is None or price_raw == '': continue
            try: price = float(str(price_raw).replace(' ', '').replace(',', '.'))
            except (ValueError, TypeError): continue
            if not (self.min_price <= price <= self.max_price): continue

            # Фильтрация по ID категории, если указана
            if target_category_id and item_category_id != target_category_id:
                continue
            # Фильтрация по ID филиала, если указан
            if target_filial_id and item_filial_id != target_filial_id:
                continue

            # Собираем данные для вывода, включая оригинальные имена филиала и категории из текущего item
            # Это важно, т.к. услуга может быть в нескольких филиалах/категориях, но здесь мы уже отфильтровали
            # Если фильтра не было, берем из item. Если был - display_category_name/display_filial_name уже корректны.
            actual_filial_name_for_output = get_name_by_id(_tenant_id_for_clinic_data, 'filial', item_filial_id) if item_filial_id else "Не указан"
            actual_category_name_for_output = get_name_by_id(_tenant_id_for_clinic_data, 'category', item_category_id) if item_category_id else "Без категории"

            if srv_id not in matching_services or price < matching_services[srv_id]['price']:
                 matching_services[srv_id] = {
                     'name_raw': srv_name_raw,
                     'price': price,
                     'filial_raw': actual_filial_name_for_output,
                     'category_raw': actual_category_name_for_output
                 }

        if not matching_services:
            filters_str = []
            if self.category_name: filters_str.append(f"в категории '{display_category_name}'") # Используем display_name
            if self.filial_name: filters_str.append(f"в филиале '{display_filial_name}'")   # Используем display_name
            filter_desc = " ".join(filters_str)
            return f"Услуги от {self.min_price:.0f} до {self.max_price:.0f} руб. {filter_desc} не найдены.".strip()
        else:
            response_parts = [f"Услуги по цене от {self.min_price:.0f} до {self.max_price:.0f} руб.:"]
            limit = 15
            count = 0
            sorted_services = sorted(matching_services.values(), key=lambda x: x['price'])

            for service in sorted_services:
                if count < limit:
                    filial_info = f" (Филиал: {service['filial_raw']})" if not self.filial_name and service['filial_raw'] != "Не указан" else ""
                    cat_info = f" (Категория: {service['category_raw']})" if not self.category_name and service['category_raw'] != "Без категории" else ""
                    response_parts.append(f"- {service['name_raw']}: {service['price']:.0f} руб.{filial_info}{cat_info}")
                    count += 1
                else:
                     response_parts.append(f"\n... (и еще {len(sorted_services) - limit} услуг(и))")
                     break
            return "\n".join(response_parts)


class ListAllCategories(BaseModel):
    """Модель для получения списка всех категорий услуг."""

    def process(self) -> str:
        if not _clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info("[FC Proc] Запрос списка всех категорий")

        categories: Set[str] = set() 
        for item in _clinic_data:
            cat_name_raw = item.get('categoryName')
            if cat_name_raw: categories.add(cat_name_raw)

        if not categories:
            return "Информация о категориях услуг не найдена."

        limit = 30
        sorted_categories = sorted(list(categories), key=lambda c: normalize_text(c, keep_spaces=True))
        output_categories = sorted_categories[:limit]
        more_categories_info = f"... и еще {len(sorted_categories) - limit} категорий." if len(sorted_categories) > limit else ""

        return "Доступные категории услуг:\n*   " + "\n*   ".join(output_categories) + f"\n{more_categories_info}".strip()
    
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

