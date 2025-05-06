# clinic_functions.py

import logging
import re
import json # Добавляем для разбора JSON в документах
from typing import Optional, List, Dict, Any, Set
# Используем pydantic_v1, как и в matrixai для инструментов
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document # Импортируем Document

logger = logging.getLogger(__name__) # Добавляем инициализацию логгера

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
# Теперь принимает данные как аргумент
def get_original_filial_name(normalized_name: str, tenant_data: List[Dict[str, Any]]) -> Optional[str]:
    """Находит оригинальное название филиала по его нормализованному имени в данных тенанта."""
    if not normalized_name or not tenant_data: return None
    # Оптимизация: создаем карту нормализованных имен один раз, если она нужна часто
    # Но для редких вызовов можно итерировать
    for item in tenant_data:
        original_name = item.get("filialName")
        if original_name and normalize_text(original_name) == normalized_name:
            return original_name
    # Возвращаем исходное нормализованное имя, если точного не найдено, чтобы избежать None в ответах
    # Можно добавить .capitalize() для лучшего вида
    return normalized_name

# --- Определения Классов Функций (теперь методы process принимают raw_data) ---

class FindEmployees(BaseModel):
    """Модель для поиска сотрудников по различным критериям."""
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска сотрудников."
        logging.info(f"[FC Proc] Поиск сотрудников (Имя: {self.employee_name}, Услуга: {self.service_name}, Филиал: {self.filial_name}) по {len(tenant_data)} записям.")

        norm_emp_name = normalize_text(self.employee_name, keep_spaces=True)
        norm_service_name = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_name = normalize_text(self.filial_name)

        filtered_data = []
        for item in tenant_data: # Работаем напрямую с raw_data
            item_emp_name_raw = item.get('employeeFullName')
            item_service_name_raw = item.get('serviceName')
            item_filial_name_raw = item.get('filialName')

            norm_item_emp = normalize_text(item_emp_name_raw, keep_spaces=True)
            norm_item_service = normalize_text(item_service_name_raw, keep_spaces=True)
            norm_item_filial = normalize_text(item_filial_name_raw)

            emp_match = (not norm_emp_name or (norm_item_emp and norm_emp_name in norm_item_emp))
            filial_match = (not norm_filial_name or (norm_item_filial and norm_filial_name == norm_item_filial))
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
            if f_name and (not norm_filial_name or norm_filial_name == normalize_text(f_name)):
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
            if norm_filial_name and not filials: continue

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
             # Это условие может быть достигнуто, если фильтры сработали уже после группировки
             return "Сотрудники найдены по части критериев, но ни один не соответствует всем условиям одновременно."

        final_response = ["Найдены следующие сотрудники:"] + response_parts
        if found_count > limit:
             final_response.append(f"\n... (и еще {found_count - limit} сотрудник(ов). Уточните запрос.)")

        return "\n".join(final_response)


class GetServicePrice(BaseModel):
    """Модель для получения цены на конкретную услугу."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска цен."
        logging.info(f"[FC Proc] Запрос цены (Услуга: {self.service_name}, Филиал: {self.filial_name}) по {len(tenant_data)} записям.")

        matches = []
        norm_search_term = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_name = normalize_text(self.filial_name)

        for item in tenant_data: # Работаем напрямую с raw_data
            s_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')
            f_name_raw = item.get('filialName')
            price_raw = item.get('price')

            if price_raw is None or price_raw == '': continue

            try: price = float(str(price_raw).replace(' ', '').replace(',', '.'))
            except (ValueError, TypeError): continue

            norm_item_f_name = normalize_text(f_name_raw)
            if norm_filial_name and norm_item_f_name != norm_filial_name:
                continue

            norm_item_s_name = normalize_text(s_name_raw, keep_spaces=True)
            norm_item_cat_name = normalize_text(cat_name_raw, keep_spaces=True)

            service_name_match = False
            category_name_match = False
            exact_match_flag = False

            if norm_item_s_name and norm_search_term in norm_item_s_name:
                 service_name_match = True
                 exact_match_flag = (norm_search_term == norm_item_s_name)

            if not service_name_match and norm_item_cat_name and norm_search_term in norm_item_cat_name:
                 category_name_match = True
                 exact_match_flag = (norm_search_term == norm_item_cat_name)

            if service_name_match or category_name_match:
                display_name = s_name_raw if s_name_raw else cat_name_raw
                if category_name_match and s_name_raw and cat_name_raw:
                     display_name = f"{s_name_raw} (категория: {cat_name_raw})"

                matches.append({
                    'display_name': display_name, # Оригинальное имя
                    'price': price,
                    'filial_name': f_name_raw if f_name_raw else "Любой", # Оригинальное имя
                    'exact_match': exact_match_flag,
                    'match_type': 'service' if service_name_match else 'category'
                })

        if not matches:
            filial_str = f" в филиале '{self.filial_name}'" if self.filial_name else ""
            return f"Услуга или категория, содержащая '{self.service_name}'{filial_str}, не найдена или для нее не указана цена."

        # Сортировка: сначала точные совпадения, потом по цене
        matches.sort(key=lambda x: (not x['exact_match'], x['price']))

        # Выводим результаты
        limit = 5
        response_parts = []
        found_count = len(matches)

        if found_count == 1:
             match = matches[0]
             filial_context = f" в филиале {match['filial_name']}" if match['filial_name'] != "Любой" else ""
             return f"Цена на услугу '{match['display_name']}'{filial_context} составляет {match['price']:.0f} руб."
        else:
             response_parts.append(f"Найдено несколько услуг/категорий, содержащих '{self.service_name}':")
             for i, match in enumerate(matches):
                  if i >= limit: break
                  filial_context = f" ({match['filial_name']})" if match['filial_name'] != "Любой" else ""
                  response_parts.append(f"- {match['display_name']}{filial_context}: {match['price']:.0f} руб.")

             if found_count > limit:
                 response_parts.append(f"\n... (и еще {found_count - limit} услуг/категорий. Пожалуйста, уточните название услуги.)")

             return "\n".join(response_parts)


class ListFilials(BaseModel):
    """Модель для получения списка всех филиалов."""
    # Нет аргументов

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Извлекает уникальные филиалы из сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data:
            return "Ошибка: Не удалось получить структурированные данные тенанта для получения списка филиалов."
        logging.info(f"[FC Proc] Запрос списка филиалов по {len(tenant_data)} записям.")

        filials: Set[str] = set()
        for item in tenant_data: # Работаем напрямую с raw_data
            f_name = item.get("filialName")
            if f_name and isinstance(f_name, str) and f_name.strip():
                filials.add(f_name.strip())

        if not filials:
            return "Список филиалов пуст или не найден в данных."

        # Сортируем для консистентного вывода
        sorted_filials = sorted(list(filials), key=normalize_text)

        return "Доступны следующие филиалы:\n- " + "\n- ".join(sorted_filials)


class GetEmployeeServices(BaseModel):
    """Модель для получения списка услуг конкретного сотрудника."""
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг сотрудника."
        logging.info(f"[FC Proc] Запрос услуг сотрудника '{self.employee_name}' по {len(tenant_data)} записям.")

        norm_emp_name_search = normalize_text(self.employee_name, keep_spaces=True)
        services: Set[str] = set()
        found_employee_name: Optional[str] = None

        for item in tenant_data: # Работаем напрямую с raw_data
            e_name_raw = item.get('employeeFullName')
            s_name_raw = item.get('serviceName')

            if not e_name_raw or not s_name_raw: continue

            norm_item_e_name = normalize_text(e_name_raw, keep_spaces=True)

            # Ищем совпадение по имени сотрудника (частичное или полное)
            if norm_emp_name_search in norm_item_e_name:
                # Если нашли первое совпадение, запоминаем оригинальное имя
                if found_employee_name is None:
                     found_employee_name = e_name_raw
                # Если нашли другое имя, которое тоже подходит - неоднозначность
                elif normalize_text(found_employee_name, keep_spaces=True) != norm_item_e_name:
                     logger.warning(f"Найдено несколько сотрудников, подходящих под '{self.employee_name}': '{found_employee_name}' и '{e_name_raw}'. Запрос неоднозначен.")
                     return f"Найдено несколько сотрудников, подходящих под имя '{self.employee_name}'. Пожалуйста, уточните ФИО."

                # Добавляем услугу, если имя сотрудника совпало
                services.add(s_name_raw)

        if not found_employee_name:
            return f"Сотрудник с именем, содержащим '{self.employee_name}', не найден."
        if not services:
            return f"Для сотрудника '{found_employee_name}' не найдено услуг в базе данных."

        # Сортируем услуги по нормализованному имени
        sorted_services = sorted(list(services), key=lambda s: normalize_text(s, keep_spaces=True))
        limit = 15
        response_parts = [f"Сотрудник '{found_employee_name}' выполняет следующие услуги:"]
        response_parts.extend([f"- {s}" for s in sorted_services[:limit]])

        if len(services) > limit:
            response_parts.append(f"\n... (и еще {len(services) - limit} услуг)")

        return "\n".join(response_parts)


class CheckServiceInFilial(BaseModel):
    """Модель для проверки наличия услуги в филиале."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для проверки услуги."
        logging.info(f"[FC Proc] Проверка услуги '{self.service_name}' в филиале '{self.filial_name}' по {len(tenant_data)} записям.")

        norm_service_search = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_search = normalize_text(self.filial_name)

        service_found = False
        service_found_in_filial = False
        found_service_name: Optional[str] = None
        found_filial_name: Optional[str] = None # Для корректного отображения имени филиала
        found_in_other_filials: Set[str] = set()

        # Проверка существования филиала
        filial_exists = False
        for item in tenant_data:
            f_name_raw = item.get('filialName')
            if f_name_raw and normalize_text(f_name_raw) == norm_filial_search:
                 filial_exists = True
                 found_filial_name = f_name_raw # Запоминаем оригинальное имя
                 break
        if not filial_exists:
             return f"Филиал '{self.filial_name}' не найден."

        # Поиск услуги
        for item in tenant_data: # Работаем напрямую с raw_data
            s_name_raw = item.get('serviceName')
            f_name_raw = item.get('filialName')

            if not s_name_raw or not f_name_raw: continue

            norm_item_s_name = normalize_text(s_name_raw, keep_spaces=True)
            norm_item_f_name = normalize_text(f_name_raw)

            # Ищем совпадение услуги (частичное)
            if norm_service_search in norm_item_s_name:
                 service_found = True
                 # Запоминаем каноничное имя
                 if found_service_name is None:
                      found_service_name = s_name_raw
                 # Проверка на неоднозначность
                 elif normalize_text(found_service_name, keep_spaces=True) != norm_item_s_name:
                      # Игнорируем, если уже нашли точное
                       if normalize_text(self.service_name, keep_spaces=True) == normalize_text(found_service_name, keep_spaces=True):
                            continue
                       else:
                           logger.warning(f"Найдено несколько услуг, подходящих под '{self.service_name}'. Запрос неоднозначен.")
                           return f"Найдено несколько услуг, содержащих '{self.service_name}'. Пожалуйста, уточните название услуги."

                 # Добавляем оригинальное имя филиала в сет
                 filials.add(f_name_raw)

            # Проверяем совпадение филиала
            if norm_item_f_name == norm_filial_search:
                service_found_in_filial = True
                found_filial_name = f_name_raw # Запоминаем оригинальное имя найденного филиала
                # Если нашли в нужном филиале, дальше можно не искать для этой записи
                # Но нужно продолжить итерацию по другим записям, чтобы найти другие филиалы
            else:
                # Если услуга найдена, но в другом филиале, запоминаем его
                found_in_other_filials.add(f_name_raw) # Добавляем оригинальное имя

        if not service_found:
             return f"Услуга, содержащая '{self.service_name}', не найдена ни в одном филиале."

        if not found_service_name: found_service_name = self.service_name # Фоллбэк

        if not found_filial_name: # Если искали в конкретном филиале, но не нашли
            found_filial_name = self.filial_name # Используем имя из запроса

        if service_found_in_filial:
             return f"Да, услуга '{found_service_name}' доступна в филиале '{found_filial_name}'."
        else:
             response = f"Услуга '{found_service_name}' не найдена в филиале '{found_filial_name}'."
             if found_in_other_filials:
                  sorted_others = sorted(list(found_in_other_filials), key=normalize_text)
                  limit = 3
                  response += "\nНо она доступна в других филиалах: " + ", ".join(sorted_others[:limit])
                  if len(sorted_others) > limit:
                       response += f" и еще в {len(sorted_others) - limit}."
             return response


class CompareServicePriceInFilials(BaseModel):
    """Модель для сравнения цен на услугу в нескольких филиалах."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(min_length=2, description="Список из ДВУХ или БОЛЕЕ названий филиалов")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для сравнения цен."
        logging.info(f"[FC Proc] Сравнение цен на '{self.service_name}' в филиалах: {self.filial_names} по {len(tenant_data)} записям.")

        norm_service_search = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_names_search = {normalize_text(f) for f in self.filial_names}

        results: Dict[str, Dict[str, Any]] = {} # {norm_filial_name: {'original_filial_name': ..., 'price': ..., 'found_service_name': ...}}
        found_service_name_canonical: Optional[str] = None # Для консистентности названия услуги в ответе

        # Проверка существования филиалов и сбор оригинальных имен
        original_filial_names_map: Dict[str, str] = {} # norm -> original
        all_norm_filials_in_db: Set[str] = set()
        for item in tenant_data:
            f_name_raw = item.get('filialName')
            if f_name_raw:
                 norm_f = normalize_text(f_name_raw)
                 all_norm_filials_in_db.add(norm_f)
                 if norm_f not in original_filial_names_map:
                     original_filial_names_map[norm_f] = f_name_raw

        not_found_filials = []
        valid_norm_filial_names_search = set()
        for norm_f_search in norm_filial_names_search:
            if norm_f_search in all_norm_filials_in_db:
                valid_norm_filial_names_search.add(norm_f_search)
            else:
                # Ищем оригинальное имя для сообщения об ошибке
                original_missing_name = next((name for name in self.filial_names if normalize_text(name) == norm_f_search), norm_f_search)
                not_found_filials.append(original_missing_name)

        if not_found_filials:
            return f"Следующие филиалы не найдены: {', '.join(not_found_filials)}. Пожалуйста, проверьте названия."
        if len(valid_norm_filial_names_search) < 2:
             return "Нужно указать как минимум два существующих филиала для сравнения."

        # Поиск цен в валидных филиалах
        service_found_at_least_once = False
        for item in tenant_data: # Работаем напрямую с raw_data
            s_name_raw = item.get('serviceName')
            f_name_raw = item.get('filialName')
            price_raw = item.get('price')

            if not s_name_raw or not f_name_raw or price_raw is None or price_raw == '': continue

            norm_item_s_name = normalize_text(s_name_raw, keep_spaces=True)
            norm_item_f_name = normalize_text(f_name_raw)

            # Ищем услугу (частичное совпадение) и филиал (точное совпадение из списка)
            if norm_service_search in norm_item_s_name and norm_item_f_name in valid_norm_filial_names_search:
                service_found_at_least_once = True

                # Если нашли первое совпадение услуги, запоминаем каноничное имя
                if found_service_name_canonical is None:
                    found_service_name_canonical = s_name_raw
                # Проверка на неоднозначность названия услуги
                elif normalize_text(found_service_name_canonical, keep_spaces=True) != norm_item_s_name:
                    # Если ранее нашли точное совпадение, игнорируем это
                     if normalize_text(self.service_name, keep_spaces=True) == normalize_text(found_service_name_canonical, keep_spaces=True):
                          continue
                     else:
                         logger.warning(f"Найдено несколько услуг, подходящих под '{self.service_name}'. Запрос неоднозначен.")
                         return f"Найдено несколько услуг, содержащих '{self.service_name}'. Пожалуйста, уточните название услуги."

                try: price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                except (ValueError, TypeError): continue

                # Записываем или обновляем цену, если найдена более точная услуга
                # Или если это первая цена для данного филиала
                current_result = results.get(norm_item_f_name)
                is_exact_match = (normalize_text(self.service_name, keep_spaces=True) == norm_item_s_name)
                should_update = False
                if not current_result:
                    should_update = True
                elif is_exact_match and not current_result.get('exact_match'): # Новое - точное, старое - нет
                     should_update = True
                elif is_exact_match == current_result.get('exact_match') and price < current_result.get('price', float('inf')): # Такой же тип совпадения, но цена ниже? (маловероятно для одной услуги)
                    # Можно добавить логирование, если цена разная для той же услуги в том же филиале
                    should_update = True # Берем минимальную, если вдруг дубликаты

                if should_update:
                    results[norm_item_f_name] = {
                        'original_filial_name': original_filial_names_map.get(norm_item_f_name, f_name_raw),
                        'price': price,
                        'found_service_name': s_name_raw, # Запоминаем имя найденной услуги
                        'exact_match': is_exact_match
                    }

        if not service_found_at_least_once:
            filial_list_str = ", ".join(original_filial_names_map.get(norm_f, norm_f) for norm_f in valid_norm_filial_names_search)
            return f"Услуга, содержащая '{self.service_name}', не найдена ни в одном из указанных филиалов: {filial_list_str}."

        if not found_service_name_canonical: # На всякий случай
             found_service_name_canonical = self.service_name

        # Формируем ответ
        response_parts = [f"Сравнение цен на услугу '{found_service_name_canonical}':"]
        found_prices_count = 0
        for norm_f_search in valid_norm_filial_names_search:
            result = results.get(norm_f_search)
            original_filial_name = original_filial_names_map.get(norm_f_search, norm_f_search)
            if result:
                price_str = f"{result['price']:.0f} руб."
                # Если найденное имя услуги отличается от канонического, указываем это
                service_name_note = ""
                if normalize_text(result['found_service_name'], keep_spaces=True) != normalize_text(found_service_name_canonical, keep_spaces=True):
                    service_name_note = f" (для услуги '{result['found_service_name']}')"

                response_parts.append(f"- {original_filial_name}: {price_str}{service_name_note}")
                found_prices_count += 1
            else:
                response_parts.append(f"- {original_filial_name}: Цена не найдена или услуга недоступна.")

        if found_prices_count == 0:
             # Это может произойти, если услуга найдена, но цена не указана нигде
             return f"Услуга '{found_service_name_canonical}' найдена, но цена для нее не указана ни в одном из запрошенных филиалов."

        return "\n".join(response_parts)


class FindServiceLocations(BaseModel):
    """Модель для поиска филиалов, где доступна услуга."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска филиалов."
        logging.info(f"[FC Proc] Поиск филиалов для услуги '{self.service_name}' по {len(tenant_data)} записям.")

        norm_service_search = normalize_text(self.service_name, keep_spaces=True)
        filials: Set[str] = set()
        found_service_name: Optional[str] = None
        service_found_at_least_once = False

        for item in tenant_data: # Работаем напрямую с raw_data
            s_name_raw = item.get('serviceName')
            f_name_raw = item.get('filialName')

            if not s_name_raw or not f_name_raw: continue

            norm_item_s_name = normalize_text(s_name_raw, keep_spaces=True)

            # Ищем совпадение услуги (частичное)
            if norm_service_search in norm_item_s_name:
                 service_found_at_least_once = True
                 # Запоминаем каноничное имя
                 if found_service_name is None:
                      found_service_name = s_name_raw
                 # Проверка на неоднозначность
                 elif normalize_text(found_service_name, keep_spaces=True) != norm_item_s_name:
                      # Игнорируем, если уже нашли точное
                       if normalize_text(self.service_name, keep_spaces=True) == normalize_text(found_service_name, keep_spaces=True):
                            continue
                       else:
                           logger.warning(f"Найдено несколько услуг, подходящих под '{self.service_name}'. Запрос неоднозначен.")
                           return f"Найдено несколько услуг, содержащих '{self.service_name}'. Пожалуйста, уточните название услуги."

                 # Добавляем оригинальное имя филиала в сет
                 filials.add(f_name_raw)

        if not service_found_at_least_once:
            return f"Услуга, содержащая '{self.service_name}', не найдена ни в одном филиале."

        if not found_service_name: found_service_name = self.service_name # Фоллбэк

        if not filials:
            return f"Услуга '{found_service_name}' найдена, но не указано, в каких филиалах она доступна."

        sorted_filials = sorted(list(filials), key=normalize_text)
        return f"Услуга '{found_service_name}' доступна в следующих филиалах:\n- " + "\n- ".join(sorted_filials)


class FindSpecialistsByServiceOrCategoryAndFilial(BaseModel):
    """Модель для поиска специалистов по услуге/категории и филиалу."""
    query_term: str = Field(description="Название услуги ИЛИ категории")
    filial_name: str = Field(description="Точное название филиала")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска специалистов."
        logging.info(f"[FC Proc] Поиск специалистов по '{self.query_term}' в филиале '{self.filial_name}' по {len(tenant_data)} записям.")

        norm_query = normalize_text(self.query_term, keep_spaces=True)
        norm_filial = normalize_text(self.filial_name)

        specialists: Set[str] = set()
        found_filial_name: Optional[str] = None
        service_or_category_found = False

        # Проверка существования филиала
        filial_exists = False
        for item in tenant_data:
            f_name_raw = item.get('filialName')
            if f_name_raw and normalize_text(f_name_raw) == norm_filial:
                 filial_exists = True
                 found_filial_name = f_name_raw # Запоминаем оригинальное имя
                 break
        if not filial_exists:
             return f"Филиал '{self.filial_name}' не найден."

        # Поиск специалистов
        for item in tenant_data: # Работаем напрямую с raw_data
            e_name_raw = item.get('employeeFullName')
            s_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')
            f_name_raw = item.get('filialName')

            if not e_name_raw or not f_name_raw: continue

            norm_item_f = normalize_text(f_name_raw)
            # Фильтруем по филиалу
            if norm_item_f != norm_filial: continue

            norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
            norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True)

            # Ищем совпадение по услуге или категории
            if (norm_item_s and norm_query in norm_item_s) or \
               (norm_item_cat and norm_query in norm_item_cat):
                service_or_category_found = True
                specialists.add(e_name_raw)

        if not service_or_category_found:
            return f"Услуга или категория, содержащая '{self.query_term}', не найдена в филиале '{found_filial_name}'."
        if not specialists:
             # Это может случиться, если услуга есть, но сотрудник не привязан
             return f"Услуга/категория '{self.query_term}' найдена в филиале '{found_filial_name}', но специалисты для нее не указаны."

        sorted_specialists = sorted(list(specialists), key=lambda s: normalize_text(s, keep_spaces=True))
        limit = 10
        response_parts = [f"В филиале '{found_filial_name}' по запросу '{self.query_term}' найдены специалисты:"]
        response_parts.extend([f"- {s}" for s in sorted_specialists[:limit]])

        if len(specialists) > limit:
             response_parts.append(f"\n... (и еще {len(specialists) - limit})")

        return "\n".join(response_parts)


class ListServicesInCategory(BaseModel):
    """Модель для получения списка услуг в конкретной категории."""
    category_name: str = Field(description="Точное название категории")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг."
        logging.info(f"[FC Proc] Запрос услуг в категории '{self.category_name}' по {len(tenant_data)} записям.")

        norm_cat_search = normalize_text(self.category_name, keep_spaces=True)
        services: Dict[str, str] = {} # norm_service_name -> original_service_name
        found_category_name: Optional[str] = None
        category_found = False

        for item in tenant_data: # Работаем напрямую с raw_data
            s_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')

            if not s_name_raw or not cat_name_raw: continue

            norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True)

            # Ищем совпадение категории (частичное)
            if norm_cat_search in norm_item_cat:
                category_found = True
                 # Запоминаем каноничное имя
                if found_category_name is None:
                     found_category_name = cat_name_raw
                 # Проверка на неоднозначность
                elif normalize_text(found_category_name, keep_spaces=True) != norm_item_cat:
                     # Игнорируем, если уже нашли точное
                      if normalize_text(self.category_name, keep_spaces=True) == normalize_text(found_category_name, keep_spaces=True):
                           continue
                      else:
                          logger.warning(f"Найдено несколько категорий, подходящих под '{self.category_name}'. Запрос неоднозначен.")
                          return f"Найдено несколько категорий, содержащих '{self.category_name}'. Пожалуйста, уточните название категории."

                # Добавляем услугу в словарь (ключ - нормализованное имя, значение - оригинальное)
                norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
                if norm_item_s not in services:
                     services[norm_item_s] = s_name_raw

        if not category_found:
            return f"Категория, содержащая '{self.category_name}', не найдена."

        if not found_category_name: found_category_name = self.category_name # Фоллбэк

        if not services:
            return f"В категории '{found_category_name}' не найдено услуг."

        # Сортируем по оригинальному имени услуги
        sorted_service_names = sorted(services.values(), key=lambda s: normalize_text(s, keep_spaces=True))
        limit = 15
        response_parts = [f"В категории '{found_category_name}' доступны следующие услуги:"]
        response_parts.extend([f"- {s}" for s in sorted_service_names[:limit]])

        if len(services) > limit:
            response_parts.append(f"\n... (и еще {len(services) - limit} услуг)")

        return "\n".join(response_parts)


class ListServicesInFilial(BaseModel):
    """Модель для получения списка всех услуг в конкретном филиале."""
    filial_name: str = Field(description="Точное название филиала")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг."
        logging.info(f"[FC Proc] Запрос услуг в филиале '{self.filial_name}' по {len(tenant_data)} записям.")

        norm_filial_search = normalize_text(self.filial_name)
        services: Dict[str, str] = {} # norm_service_name -> original_service_name
        categories: Dict[str, Set[str]] = {} # norm_category_name -> set of norm_service_names
        found_filial_name: Optional[str] = None
        filial_found = False

        for item in tenant_data: # Работаем напрямую с raw_data
            s_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')
            f_name_raw = item.get('filialName')

            if not s_name_raw or not f_name_raw: continue

            norm_item_f = normalize_text(f_name_raw)

            # Ищем совпадение филиала (точное)
            if norm_item_f == norm_filial_search:
                filial_found = True
                 # Запоминаем каноничное имя
                if found_filial_name is None:
                     found_filial_name = f_name_raw
                 # Проверка на неоднозначность (маловероятно, но все же)
                elif normalize_text(found_filial_name) != norm_item_f:
                     logger.warning(f"Найдено несколько филиалов, подходящих под '{self.filial_name}'. Используется первое: '{found_filial_name}'")
                     # Продолжаем использовать первое найденное имя

                # Добавляем услугу и категорию
                norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
                if norm_item_s not in services:
                    services[norm_item_s] = s_name_raw

                if cat_name_raw:
                    norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True)
                    if norm_item_cat not in categories:
                         categories[norm_item_cat] = set()
                    categories[norm_item_cat].add(norm_item_s)

        if not filial_found:
            return f"Филиал '{self.filial_name}' не найден."

        if not found_filial_name: found_filial_name = self.filial_name # Фоллбэк

        if not services:
            return f"В филиале '{found_filial_name}' не найдено услуг."

        # Сортируем категории, а потом услуги внутри них
        sorted_category_names = sorted(categories.keys(), key=lambda c: normalize_text(c, keep_spaces=True)) # Сортируем по нормализованному имени категории

        response_parts = [f"В филиале '{found_filial_name}' доступны следующие услуги:"]
        limit = 20 # Общий лимит на вывод услуг
        count = 0

        # Сначала выводим услуги по категориям
        for norm_cat in sorted_category_names:
             original_cat_name = next((c for c_raw in tenant_data if normalize_text(c_raw.get('categoryName'), keep_spaces=True) == norm_cat for c in [c_raw.get('categoryName')]), norm_cat) # Находим оригинальное имя категории
             response_parts.append(f"\n**{original_cat_name}:**")
             service_names_in_cat = sorted([services[norm_s] for norm_s in categories[norm_cat]], key=lambda s: normalize_text(s, keep_spaces=True)) # Оригинальные имена, сортированные
             for service_name in service_names_in_cat:
                  if count < limit:
                      response_parts.append(f"- {service_name}")
                      count += 1
                  else:
                      break
             if count >= limit: break

        # Если лимит не достигнут, выводим услуги без категорий (если такие есть)
        if count < limit:
             services_without_category = []
             for norm_s, original_s in services.items():
                 in_category = False
                 for cat_services in categories.values():
                      if norm_s in cat_services:
                           in_category = True
                           break
                 if not in_category:
                      services_without_category.append(original_s)

             if services_without_category:
                  response_parts.append(f"\n**Другие услуги:**")
                  sorted_others = sorted(services_without_category, key=lambda s: normalize_text(s, keep_spaces=True))
                  for service_name in sorted_others:
                      if count < limit:
                          response_parts.append(f"- {service_name}")
                          count += 1
                      else:
                          break

        if len(services) > count:
             response_parts.append(f"\n... (и еще {len(services) - count} услуг)")

        return "\n".join(response_parts)


class FindServicesInPriceRange(BaseModel):
    """Модель для поиска услуг в заданном ценовом диапазоне."""
    min_price: float = Field(description="Минимальная цена")
    max_price: float = Field(description="Максимальная цена")
    category_name: Optional[str] = Field(default=None, description="Опционально: категория для фильтрации")
    filial_name: Optional[str] = Field(default=None, description="Опционально: филиал для фильтрации")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг."
        logging.info(f"[FC Proc] Поиск услуг в диапазоне {self.min_price}-{self.max_price} (Кат: {self.category_name}, Фил: {self.filial_name}) по {len(tenant_data)} записям.")

        if self.min_price > self.max_price:
             return "Ошибка: Минимальная цена не может быть больше максимальной."

        norm_cat_filter = normalize_text(self.category_name, keep_spaces=True) if self.category_name else None
        norm_filial_filter = normalize_text(self.filial_name) if self.filial_name else None

        matches: Dict[str, Dict[str, Any]] = {} # norm_service_name -> {'original': ..., 'price': ..., 'category': ..., 'filials': set()}

        # Предварительно проверим существование филиала, если он указан
        original_filial_filter_name = None
        if norm_filial_filter:
             filial_exists = False
             for item in tenant_data:
                  f_name_raw = item.get("filialName")
                  if f_name_raw and normalize_text(f_name_raw) == norm_filial_filter:
                      filial_exists = True
                      original_filial_filter_name = f_name_raw
                      break
             if not filial_exists:
                  return f"Филиал '{self.filial_name}' не найден."
        else:
            original_filial_filter_name = "Любой"


        # Предварительно проверим существование категории, если она указана
        original_category_filter_name = None
        if norm_cat_filter:
             category_exists = False
             for item in tenant_data:
                  cat_name_raw = item.get("categoryName")
                  if cat_name_raw and norm_cat_filter in normalize_text(cat_name_raw, keep_spaces=True):
                       category_exists = True
                       original_category_filter_name = cat_name_raw # Берем первое совпавшее имя
                       break
             if not category_exists:
                  return f"Категория, содержащая '{self.category_name}', не найдена."
        else:
             original_category_filter_name = "Любая"

        # Основной цикл фильтрации
        for item in tenant_data: # Работаем напрямую с raw_data
            s_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')
            f_name_raw = item.get('filialName')
            price_raw = item.get('price')

            if not s_name_raw or price_raw is None or price_raw == '': continue

            # Фильтр по категории
            if norm_cat_filter:
                norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True) if cat_name_raw else ""
                if norm_cat_filter not in norm_item_cat:
                     continue

            # Фильтр по филиалу
            if norm_filial_filter:
                 norm_item_f = normalize_text(f_name_raw) if f_name_raw else ""
                 if norm_item_f != norm_filial_filter:
                      continue

            # Фильтр по цене
            try: price = float(str(price_raw).replace(' ', '').replace(',', '.'))
            except (ValueError, TypeError): continue
            if not (self.min_price <= price <= self.max_price):
                 continue

            # Если все фильтры пройдены, добавляем или обновляем услугу
            norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
            if norm_item_s not in matches:
                matches[norm_item_s] = {
                    'original': s_name_raw,
                    'price': price, # Записываем цену, если она одна на услугу
                    'category': cat_name_raw if cat_name_raw else "Без категории", # Оригинальное имя категории
                    'filials': set()
                }
            # Добавляем филиал, если он есть (используем оригинальное имя)
            if f_name_raw:
                 matches[norm_item_s]['filials'].add(f_name_raw)
            # Если цена для услуги уже есть и отличается - это странно, логируем
            elif 'price' in matches[norm_item_s] and matches[norm_item_s]['price'] != price:
                 logger.warning(f"Обнаружена разная цена ({matches[norm_item_s]['price']} vs {price}) для одной услуги '{s_name_raw}' без указания филиала.")
                 # Можно выбрать минимальную или оставить первую
                 matches[norm_item_s]['price'] = min(matches[norm_item_s]['price'], price)


        if not matches:
             cat_str = f" в категории '{original_category_filter_name}'" if self.category_name else ""
             filial_str = f" в филиале '{original_filial_filter_name}'" if self.filial_name else ""
             return f"Услуги в ценовом диапазоне от {self.min_price:.0f} до {self.max_price:.0f} руб.{cat_str}{filial_str} не найдены."

        # Сортируем результаты по цене, затем по имени
        sorted_matches = sorted(matches.values(), key=lambda x: (x['price'], normalize_text(x['original'], keep_spaces=True)))

        limit = 15
        response_parts = [f"Найдены услуги от {self.min_price:.0f} до {self.max_price:.0f} руб.:"]
        if self.category_name: response_parts[0] += f" (Категория: '{original_category_filter_name}')"
        if self.filial_name: response_parts[0] += f" (Филиал: '{original_filial_filter_name}')"

        for i, match in enumerate(sorted_matches):
             if i >= limit: break
             filials_str = f" ({', '.join(sorted(list(match['filials']), key=normalize_text))})" if match['filials'] else ""
             # Если филиал был задан в фильтре, не выводим его снова
             if norm_filial_filter: filials_str = ""
             response_parts.append(f"- {match['original']}: {match['price']:.0f} руб.{filials_str}")

        if len(matches) > limit:
             response_parts.append(f"\n... (и еще {len(matches) - limit} услуг)")

        return "\n".join(response_parts)


class ListAllCategories(BaseModel):
    """Модель для получения списка всех категорий услуг."""
    # Нет аргументов

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Извлекает уникальные категории из сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска категорий."
        logging.info(f"[FC Proc] Запрос списка всех категорий по {len(tenant_data)} записям.")

        categories: Set[str] = set()
        for item in tenant_data: # Работаем напрямую с raw_data
            cat_name = item.get("categoryName")
            if cat_name and isinstance(cat_name, str) and cat_name.strip():
                categories.add(cat_name.strip())

        if not categories:
            return "Список категорий услуг пуст или не найден в данных."

        # Сортируем для консистентного вывода
        sorted_categories = sorted(list(categories), key=lambda c: normalize_text(c, keep_spaces=True))
        limit = 30
        response_parts = ["Доступны следующие категории услуг:"]
        response_parts.extend([f"- {c}" for c in sorted_categories[:limit]])

        if len(categories) > limit:
             response_parts.append(f"\n... (и еще {len(categories) - limit})")

        return "\n".join(response_parts)


class ListEmployeeFilials(BaseModel):
    """Модель для получения списка филиалов конкретного сотрудника."""
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        """Выполняет поиск по списку сырых данных тенанта."""
        tenant_data = raw_data # Используем сырые данные
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска филиалов сотрудника."
        logging.info(f"[FC Proc] Запрос филиалов сотрудника '{self.employee_name}' по {len(tenant_data)} записям.")

        norm_emp_name_search = normalize_text(self.employee_name, keep_spaces=True)
        filials: Set[str] = set()
        found_employee_name: Optional[str] = None

        for item in tenant_data: # Работаем напрямую с raw_data
            e_name_raw = item.get('employeeFullName')
            f_name_raw = item.get('filialName')

            if not e_name_raw or not f_name_raw: continue

            norm_item_e_name = normalize_text(e_name_raw, keep_spaces=True)

            # Ищем совпадение по имени сотрудника (частичное или полное)
            if norm_emp_name_search in norm_item_e_name:
                # Если нашли первое совпадение, запоминаем оригинальное имя
                if found_employee_name is None:
                     found_employee_name = e_name_raw
                # Если нашли другое имя, которое тоже подходит - неоднозначность
                elif normalize_text(found_employee_name, keep_spaces=True) != norm_item_e_name:
                     logger.warning(f"Найдено несколько сотрудников, подходящих под '{self.employee_name}': '{found_employee_name}' и '{e_name_raw}'. Запрос неоднозначен.")
                     return f"Найдено несколько сотрудников, подходящих под имя '{self.employee_name}'. Пожалуйста, уточните ФИО."

                # Добавляем филиал, если имя сотрудника совпало
                filials.add(f_name_raw)

        if not found_employee_name:
            return f"Сотрудник с именем, содержащим '{self.employee_name}', не найден."
        if not filials:
            return f"Для сотрудника '{found_employee_name}' не найдено филиалов в базе данных."

        # Сортируем филиалы по нормализованному имени
        sorted_filials = sorted(list(filials), key=normalize_text)

        return f"Сотрудник '{found_employee_name}' работает в следующих филиалах:\n- " + "\n- ".join(sorted_filials)
