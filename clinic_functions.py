import logging
import re
import json
from typing import Optional, List, Dict, Any, Set, Tuple

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Начало: Вспомогательная функция пагинации ---
def apply_pagination(
    full_list: List[Any], 
    page_number: int = 1, 
    page_size: int = 15
) -> Tuple[List[Any], Dict[str, Any]]:
    """
    Применяет пагинацию к списку.

    Args:
        full_list: Полный список элементов.
        page_number: Номер запрашиваемой страницы (начиная с 1).
        page_size: Количество элементов на странице.

    Returns:
        Кортеж: (срез_списка_для_страницы, информация_о_пагинации)
        Информация о пагинации - это словарь с ключами:
        'total_items', 'total_pages', 'current_page', 'page_size', 
        'start_index', 'end_index'.
    """
    if not isinstance(full_list, list):
        logger.warning("apply_pagination ожидал список, получен другой тип.")
        full_list = [] # Обработка как пустого списка
        
    total_items = len(full_list)
    
    # Обработка некорректных page_size
    if not isinstance(page_size, int) or page_size <= 0:
        logger.warning(f"Некорректный page_size '{page_size}', используется значение по умолчанию 15.")
        page_size = 15
        
    total_pages = (total_items + page_size - 1) // page_size if total_items > 0 else 1
    
    # Обработка некорректных page_number
    if not isinstance(page_number, int) or page_number <= 0:
        logger.warning(f"Некорректный page_number '{page_number}', используется 1.")
        page_number = 1
    elif page_number > total_pages:
        logger.debug(f"Запрошен номер страницы '{page_number}', превышающий общее количество страниц '{total_pages}'. Возвращается последняя страница.")
        page_number = total_pages

    start_index = (page_number - 1) * page_size
    end_index = start_index + page_size
    paginated_slice = full_list[start_index:end_index]

    pagination_info = {
        "total_items": total_items,
        "total_pages": total_pages,
        "current_page": page_number,
        "page_size": page_size,
        "start_index": start_index + 1 if total_items > 0 else 0, # 1-based for display
        "end_index": min(end_index, total_items) if total_items > 0 else 0 # 1-based inclusive
    }
    
    return paginated_slice, pagination_info
# --- Конец: Вспомогательная функция пагинации ---

# --- Функция нормализации ---
def normalize_text(text: Optional[str], keep_spaces: bool = False) -> str:
    """
    Приводит строку к нижнему регистру, удаляет дефисы и опционально пробелы.
    Безопасно обрабатывает None, возвращая пустую строку.
    """
    if not text:
        return ""
    normalized = text.lower().replace("-", "")
    if keep_spaces:
        normalized = re.sub(r'\s+', ' ', normalized).strip()
    else:
        normalized = normalized.replace(" ", "")
    return normalized

def get_original_filial_name(normalized_name: str, tenant_data: List[Dict[str, Any]]) -> Optional[str]:
    """Находит оригинальное название филиала по его нормализованному имени в данных тенанта."""
    if not normalized_name or not tenant_data: return None
    for item in tenant_data:
        original_name = item.get("filialName")
        if original_name and normalize_text(original_name) == normalized_name:
            return original_name
    return normalized_name


class FindEmployees(BaseModel):
    """
    Находит сотрудников.
    - Если указан ТОЛЬКО 'filial_name', возвращает ВСЕХ сотрудников этого филиала.
    - Можно также дополнительно фильтровать по 'employee_name' (ФИО сотрудника).
    - Можно также дополнительно фильтровать по 'service_name' (если нужно найти, кто оказывает КОНКРЕТНУЮ УСЛУГУ).
    - НЕ используйте эту функцию для поиска по КАТЕГОРИИ услуг. Для категорий есть FindSpecialistsByServiceOrCategoryAndFilial.
    """
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника для фильтрации (опционально)")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название КОНКРЕТНОЙ услуги для фильтрации (опционально)")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала. Если указано только это поле, вернет ВСЕХ сотрудников филиала.")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=15, description="Количество сотрудников на странице для пагинации")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска сотрудников."
        logging.info(f"[FC Proc] Поиск сотрудников (Имя: {self.employee_name}, Услуга: {self.service_name}, Филиал: {self.filial_name}, Стр: {self.page_number}, Разм: {self.page_size}) по {len(tenant_data)} записям.")

        norm_emp_name = normalize_text(self.employee_name, keep_spaces=True)
        norm_service_name = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_name = normalize_text(self.filial_name)

        filtered_data = []
        for item in tenant_data:
            item_emp_name_raw = item.get('employeeFullName')
            item_service_name_raw = item.get('serviceName')
            item_filial_name_raw = item.get('filialName')

            norm_item_emp = normalize_text(item_emp_name_raw, keep_spaces=True)
            norm_item_service = normalize_text(item_service_name_raw, keep_spaces=True)
            norm_item_filial = normalize_text(item_filial_name_raw)

            emp_match = (not norm_emp_name or (norm_item_emp and norm_emp_name in norm_item_emp))
            filial_match = (not norm_filial_name or (norm_item_filial and norm_filial_name == norm_item_filial))
            service_match = True 
            if norm_service_name and norm_item_service: 
                service_match = (norm_service_name in norm_item_service) or (norm_item_service in norm_service_name)
            elif norm_service_name and not norm_item_service: 
                service_match = False

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
            if s_name and (not norm_service_name or norm_service_name in normalize_text(s_name, keep_spaces=True)):
                 employees_info[e_id]['services'].add(s_name)
            if f_name and (not norm_filial_name or norm_filial_name == normalize_text(f_name)):
                 employees_info[e_id]['filials'].add(f_name)

        response_parts = []
        total_found_employees = 0

        # Сортируем ВСЕХ найденных сотрудников
        all_found_employees_list = sorted(
            [emp for emp in employees_info.values() if emp.get('name') 
             and (not norm_service_name or emp.get('services')) 
             and (not norm_filial_name or emp.get('filials'))], 
            key=lambda x: normalize_text(x.get('name'), keep_spaces=True)
        )
        
        # Применяем пагинацию к отсортированному списку сотрудников
        paginated_employees, pagination_info = apply_pagination(
            all_found_employees_list, 
            self.page_number, 
            self.page_size
        )
        
        total_found_employees = pagination_info['total_items']
        current_page_num = pagination_info['current_page']
        total_pages = pagination_info['total_pages']
        start_idx = pagination_info['start_index']
        end_idx = pagination_info['end_index']

        if total_found_employees == 0:
             return "Сотрудники найдены по части критериев, но ни один не соответствует всем условиям одновременно."

        header_message = f"Найдено {total_found_employees} сотрудник(ов)."
        if total_pages > 1:
            header_message += f" Показаны {start_idx}-{end_idx} (Страница {current_page_num} из {total_pages})."
        response_parts.append(header_message)

        # Выводим только сотрудников для текущей страницы
        for emp in paginated_employees:
            name = emp.get('name')
            emp_info = f"- {name}"
            # Опционально можно добавить детали (филиалы/услуги) если нужно
            # filials = sorted(list(emp.get('filials', set())), key=normalize_text)
            # if filials: emp_info += f" (Филиалы: {', '.join(filials)})"
            response_parts.append(emp_info)

        if total_pages > 1 and current_page_num < total_pages:
            response_parts.append("\n(Для просмотра следующей страницы укажите page_number)")
            
        final_response = response_parts

        return "\n".join(final_response)


class GetServicePrice(BaseModel):
    """Модель для получения цены на конкретную услугу."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска цен."
        logging.info(f"[FC Proc] Запрос цены (Услуга: {self.service_name}, Филиал: {self.filial_name}) по {len(tenant_data)} записям.")

        matches = []
        norm_search_term = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_name = normalize_text(self.filial_name)

        for item in tenant_data:
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

            if norm_item_s_name and norm_search_term: # Проверяем, что оба существуют
                if (norm_search_term in norm_item_s_name) or (norm_item_s_name in norm_search_term):
                    service_name_match = True
                    exact_match_flag = (norm_search_term == norm_item_s_name)

            # Если по названию услуги не нашли, или нашли, но хотим проверить и категорию (если вдруг запрос соответствует и тому и другому)
            # Лучше не делать elif, а проверять категорию независимо, если service_name_match не True или если хотим дать приоритет точному совпадению по услуге.
            # Для упрощения текущей логики, оставим последовательную проверку, но с симметричным поиском.
            if not service_name_match and norm_item_cat_name and norm_search_term: # Проверяем, что оба существуют
                if (norm_search_term in norm_item_cat_name) or (norm_item_cat_name in norm_search_term):
                    category_name_match = True
                    # exact_match_flag для категории также проверяем
                    exact_match_flag = (norm_search_term == norm_item_cat_name) 

            if service_name_match or category_name_match:
                display_name = s_name_raw if s_name_raw else cat_name_raw
                if category_name_match and s_name_raw and cat_name_raw:
                     display_name = f"{s_name_raw} (категория: {cat_name_raw})"

                matches.append({
                    'display_name': display_name,
                    'price': price,
                    'filial_name': f_name_raw if f_name_raw else "Любой",
                    'exact_match': exact_match_flag,
                    'match_type': 'service' if service_name_match else 'category'
                })

        if not matches:
            filial_str = f" в филиале '{self.filial_name}'" if self.filial_name else ""
            return f"Услуга или категория, содержащая '{self.service_name}'{filial_str}, не найдена или для нее не указана цена."

        matches.sort(key=lambda x: (not x['exact_match'], x['price']))

        output_limit = 5 # Лимит на количество вариантов цен, если найдено много
        response_parts = []
        total_found_matches = len(matches)

        if total_found_matches == 1:
             match = matches[0]
             filial_context = f" в филиале {match['filial_name']}" if match['filial_name'] != "Любой" else ""
             return f"Цена на услугу '{match['display_name']}'{filial_context} составляет {match['price']:.0f} руб."
        else:
             response_parts.append(f"Найдено {total_found_matches} вариантов цен/услуг для '{self.service_name}'. Показаны первые {min(total_found_matches, output_limit)}:")
             for i, match in enumerate(matches):
                  if i >= output_limit: break
                  filial_context = f" ({match['filial_name']})" if match['filial_name'] != "Любой" else ""
                  response_parts.append(f"- {match['display_name']}{filial_context}: {match['price']:.0f} руб.")

             if total_found_matches > output_limit:
                 response_parts.append(f"\n... (и еще {total_found_matches - output_limit} вариантов. Пожалуйста, уточните название услуги или филиал.)")

             return "\n".join(response_parts)


class ListFilials(BaseModel):
    """Модель для получения списка всех филиалов."""

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data:
            return "Ошибка: Не удалось получить структурированные данные тенанта для получения списка филиалов."
        logging.info(f"[FC Proc] Запрос списка филиалов по {len(tenant_data)} записям.")

        filials: Set[str] = set()
        for item in tenant_data:
            f_name = item.get("filialName")
            if f_name and isinstance(f_name, str) and f_name.strip():
                filials.add(f_name.strip())

        if not filials:
            return "Список филиалов пуст или не найден в данных."

        sorted_filials = sorted(list(filials), key=normalize_text)
        # Лимит для филиалов обычно не нужен, их не так много, но на всякий случай
        output_limit = 50 
        total_filials = len(sorted_filials)
        
        response_parts = [f"Доступно {total_filials} филиал(ов):"]
        response_parts.extend([f"- {f}" for f in sorted_filials[:output_limit]])

        if total_filials > output_limit:
            response_parts.append(f"\n... (и еще {total_filials - output_limit} филиал(ов))")
            
        return "\n".join(response_parts)


class GetEmployeeServices(BaseModel):
    """Модель для получения списка услуг конкретного сотрудника."""
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=20, description="Количество услуг на странице для пагинации")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг сотрудника."
        logging.info(f"[FC Proc] Запрос услуг сотрудника '{self.employee_name}' (Стр: {self.page_number}, Разм: {self.page_size}) по {len(tenant_data)} записям.")

        norm_emp_name_search = normalize_text(self.employee_name, keep_spaces=True)
        services: Set[str] = set()
        found_employee_name: Optional[str] = None

        for item in tenant_data:
            e_name_raw = item.get('employeeFullName')
            s_name_raw = item.get('serviceName')
            if not e_name_raw or not s_name_raw: continue
            norm_item_e_name = normalize_text(e_name_raw, keep_spaces=True)

            if norm_emp_name_search in norm_item_e_name:
                if found_employee_name is None:
                     found_employee_name = e_name_raw
                elif normalize_text(found_employee_name, keep_spaces=True) != norm_item_e_name:
                     logger.warning(f"Найдено несколько сотрудников, подходящих под '{self.employee_name}': '{found_employee_name}' и '{e_name_raw}'. Запрос неоднозначен.")
                     return f"Найдено несколько сотрудников, подходящих под имя '{self.employee_name}'. Пожалуйста, уточните ФИО."
                services.add(s_name_raw)

        if not found_employee_name:
            return f"Сотрудник с именем, содержащим '{self.employee_name}', не найден."
        if not services:
            return f"Для сотрудника '{found_employee_name}' не найдено услуг в базе данных."

        all_services_list = sorted(list(services), key=lambda s: normalize_text(s, keep_spaces=True))
        
        paginated_services, pagination_info = apply_pagination(
            all_services_list,
            self.page_number,
            self.page_size
        )
        
        total_services = pagination_info['total_items']
        current_page_num = pagination_info['current_page']
        total_pages = pagination_info['total_pages']
        start_idx = pagination_info['start_index']
        end_idx = pagination_info['end_index']
        
        response_parts = [f"Сотрудник '{found_employee_name}' выполняет {total_services} услуг(и)."]
        if total_pages > 1:
            response_parts[0] += f" Показаны {start_idx}-{end_idx} (Страница {current_page_num} из {total_pages})."
        
        response_parts.extend([f"- {s}" for s in paginated_services])

        if total_pages > 1 and current_page_num < total_pages:
            response_parts.append("\n(Для просмотра следующей страницы укажите page_number)")

        return "\n".join(response_parts)


class CheckServiceInFilial(BaseModel):
    """Модель для проверки наличия услуги в филиале."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для проверки услуги."
        logging.info(f"[FC Proc] Проверка услуги '{self.service_name}' в филиале '{self.filial_name}' по {len(tenant_data)} записям.")

        norm_service_search = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_search = normalize_text(self.filial_name)

        service_found_globally = False # Найдена ли услуга вообще где-либо
        service_found_in_target_filial = False
        found_service_canonical_name: Optional[str] = None # Каноническое имя найденной услуги
        is_canonical_exact_match_to_query: bool = False # Флаг для канонического имени
        original_target_filial_name: Optional[str] = None # Оригинальное имя целевого филиала
        
        filial_exists = False
        for item in tenant_data:
            f_name_raw_check = item.get('filialName')
            if f_name_raw_check and normalize_text(f_name_raw_check) == norm_filial_search:
                 filial_exists = True
                 original_target_filial_name = f_name_raw_check
                 break
        if not filial_exists:
             return f"Филиал '{self.filial_name}' не найден."
        if not original_target_filial_name: original_target_filial_name = self.filial_name


        found_in_other_filials_set: Set[str] = set()

        for item in tenant_data:
            s_name_raw = item.get('serviceName')
            f_name_raw_item = item.get('filialName')

            if not s_name_raw or not f_name_raw_item: continue

            norm_item_s_name = normalize_text(s_name_raw, keep_spaces=True)
            norm_item_f_name = normalize_text(f_name_raw_item)

            # Симметричный поиск услуги
            service_match_current_item = False
            if norm_item_s_name and norm_service_search: # Убедимся, что оба существуют
                service_match_current_item = (norm_service_search in norm_item_s_name) or \
                                             (norm_item_s_name in norm_service_search)

            if service_match_current_item and norm_item_f_name in valid_norm_filial_names_for_search:
                service_found_globally = True
                
                current_is_exact_match_to_query = (norm_item_s_name == norm_service_search)
                if found_service_canonical_name is None:
                    found_service_canonical_name = s_name_raw
                    is_canonical_exact_match_to_query = current_is_exact_match_to_query
                else:
                    if current_is_exact_match_to_query and not is_canonical_exact_match_to_query:
                        found_service_canonical_name = s_name_raw
                        is_canonical_exact_match_to_query = True
                    elif not current_is_exact_match_to_query and \
                         not is_canonical_exact_match_to_query and \
                         len(norm_item_s_name) > len(normalize_text(found_service_canonical_name, keep_spaces=True)):
                        found_service_canonical_name = s_name_raw

                if norm_item_f_name == norm_filial_search:
                    service_found_in_target_filial = True
                else:
                    found_in_other_filials_set.add(f_name_raw_item) # Собираем другие филиалы, где услуга есть

        if not service_found_globally: # Если услуга вообще не найдена
             return f"Услуга, содержащая '{self.service_name}', не найдена ни в одном филиале."
        
        # Если каноническое имя не установилось (маловероятно, но для полноты), используем исходное
        if not found_service_canonical_name: found_service_canonical_name = self.service_name

        if service_found_in_target_filial:
             return f"Да, услуга '{found_service_canonical_name}' доступна в филиале '{original_target_filial_name}'."
        else:
             response = f"Услуга '{found_service_canonical_name}' не найдена в филиале '{original_target_filial_name}'."
             if found_in_other_filials_set:
                  sorted_others = sorted(list(found_in_other_filials_set), key=normalize_text)
                  output_limit_others = 3
                  response += "\nНо она доступна в других филиалах: " + ", ".join(sorted_others[:output_limit_others])
                  if len(sorted_others) > output_limit_others:
                       response += f" и еще в {len(sorted_others) - output_limit_others}."
             return response


class CompareServicePriceInFilials(BaseModel):
    """Модель для сравнения цен на услугу в нескольких филиалах."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(min_length=2, description="Список из ДВУХ или БОЛЕЕ названий филиалов")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для сравнения цен."
        logging.info(f"[FC Proc] Сравнение цен на '{self.service_name}' в филиалах: {self.filial_names} по {len(tenant_data)} записям.")

        norm_service_search = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_names_search_input = {normalize_text(f):f for f in self.filial_names} # norm -> original from input

        results: Dict[str, Dict[str, Any]] = {} 
        found_service_name_canonical: Optional[str] = None 
        is_canonical_exact_match_to_query: bool = False # Флаг для канонического имени

        original_filial_names_map_db: Dict[str, str] = {} 
        all_norm_filials_in_db: Set[str] = set()
        for item in tenant_data:
            f_name_raw_db = item.get('filialName')
            if f_name_raw_db:
                 norm_f_db = normalize_text(f_name_raw_db)
                 all_norm_filials_in_db.add(norm_f_db)
                 if norm_f_db not in original_filial_names_map_db:
                     original_filial_names_map_db[norm_f_db] = f_name_raw_db

        not_found_filial_originals = []
        valid_norm_filial_names_for_search = set()
        
        for norm_f_input, original_f_input in norm_filial_names_search_input.items():
            if norm_f_input in all_norm_filials_in_db:
                valid_norm_filial_names_for_search.add(norm_f_input)
            else:
                not_found_filial_originals.append(original_f_input)

        if not_found_filial_originals:
            return f"Следующие филиалы не найдены: {', '.join(not_found_filial_originals)}. Пожалуйста, проверьте названия."
        if len(valid_norm_filial_names_for_search) < 2:
             return "Нужно указать как минимум два существующих филиала для сравнения."

        service_found_at_least_once = False
        for item in tenant_data:
            s_name_raw = item.get('serviceName')
            f_name_raw = item.get('filialName')
            price_raw = item.get('price')

            if not s_name_raw or not f_name_raw or price_raw is None or price_raw == '': continue

            norm_item_s_name = normalize_text(s_name_raw, keep_spaces=True)
            norm_item_f_name = normalize_text(f_name_raw)

            if norm_service_search in norm_item_s_name and norm_item_f_name in valid_norm_filial_names_for_search:
                service_found_at_least_once = True

                current_is_exact_match_to_query = (norm_item_s_name == norm_service_search)
                if found_service_name_canonical is None:
                    found_service_name_canonical = s_name_raw
                    is_canonical_exact_match_to_query = current_is_exact_match_to_query
                else:
                    if current_is_exact_match_to_query and not is_canonical_exact_match_to_query:
                        found_service_name_canonical = s_name_raw
                        is_canonical_exact_match_to_query = True
                    elif not current_is_exact_match_to_query and \
                         not is_canonical_exact_match_to_query and \
                         len(norm_item_s_name) > len(normalize_text(found_service_name_canonical, keep_spaces=True)):
                        found_service_name_canonical = s_name_raw

                try: price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                except (ValueError, TypeError): continue

                current_result = results.get(norm_item_f_name)
                should_update = False
                if not current_result:
                    should_update = True
                elif is_canonical_exact_match_to_query and not current_result.get('exact_match'):
                     should_update = True
                elif is_canonical_exact_match_to_query == current_result.get('exact_match') and price < current_result.get('price', float('inf')):
                    should_update = True

                if should_update:
                    results[norm_item_f_name] = {
                        'original_filial_name_from_db': original_filial_names_map_db.get(norm_item_f_name, f_name_raw),
                        'price': price,
                        'found_service_name_for_price': s_name_raw, 
                        'exact_match': is_canonical_exact_match_to_query
                    }

        if not service_found_at_least_once:
            filial_list_str = ", ".join([norm_filial_names_search_input.get(norm_f, norm_f) for norm_f in valid_norm_filial_names_for_search])
            return f"Услуга, содержащая '{self.service_name}', не найдена ни в одном из указанных филиалов: {filial_list_str}."

        if not found_service_name_canonical: found_service_name_canonical = self.service_name

        response_parts = [f"Сравнение цен на услугу '{found_service_name_canonical}':"]
        found_prices_count = 0
        for norm_f_search in valid_norm_filial_names_for_search: # Итерируемся по тем, что юзер просил и которые существуют
            result_for_filial = results.get(norm_f_search)
            # Используем оригинальное имя филиала из пользовательского ввода для вывода, если оно есть
            original_filial_name_for_display = norm_filial_names_search_input.get(norm_f_search, original_filial_names_map_db.get(norm_f_search, norm_f_search))

            if result_for_filial:
                price_str = f"{result_for_filial['price']:.0f} руб."
                service_name_note = ""
                # Если имя услуги, для которой найдена цена, отличается от канонического, указываем это
                if normalize_text(result_for_filial['found_service_name_for_price'], keep_spaces=True) != normalize_text(found_service_name_canonical, keep_spaces=True):
                    service_name_note = f" (для '{result_for_filial['found_service_name_for_price']}')"
                response_parts.append(f"- {original_filial_name_for_display}: {price_str}{service_name_note}")
                found_prices_count += 1
            else:
                response_parts.append(f"- {original_filial_name_for_display}: Цена не найдена или услуга '{found_service_name_canonical}' недоступна.")
        
        if found_prices_count == 0: # Если услуга нашлась, но ни для одного филиала нет цены
             return f"Услуга '{found_service_name_canonical}' найдена, но цена для нее не указана ни в одном из запрошенных филиалов."

        return "\n".join(response_parts)


class FindServiceLocations(BaseModel):
    """Модель для поиска филиалов, где доступна услуга."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска филиалов."
        logging.info(f"[FC Proc] Поиск филиалов для услуги '{self.service_name}' по {len(tenant_data)} записям.")

        norm_service_search = normalize_text(self.service_name, keep_spaces=True)
        filials_with_service: Set[str] = set()
        found_service_canonical_name: Optional[str] = None
        is_canonical_exact_match_to_query: bool = False # Флаг для канонического имени
        service_found_at_least_once = False

        for item in tenant_data:
            s_name_raw = item.get('serviceName')
            f_name_raw = item.get('filialName')
            if not s_name_raw or not f_name_raw: continue
            norm_item_s_name = normalize_text(s_name_raw, keep_spaces=True)

            service_match_current_item = False
            if norm_item_s_name and norm_service_search: # Убедимся, что оба существуют
                service_match_current_item = (norm_service_search in norm_item_s_name) or \
                                             (norm_item_s_name in norm_service_search)

            if service_match_current_item:
                service_found_at_least_once = True
                current_is_exact_match_to_query = (norm_item_s_name == norm_service_search)

                if found_service_canonical_name is None:
                    found_service_canonical_name = s_name_raw
                    is_canonical_exact_match_to_query = current_is_exact_match_to_query
                else:
                    if current_is_exact_match_to_query and not is_canonical_exact_match_to_query:
                        found_service_canonical_name = s_name_raw
                        is_canonical_exact_match_to_query = True
                    elif not current_is_exact_match_to_query and \
                         not is_canonical_exact_match_to_query and \
                         len(norm_item_s_name) > len(normalize_text(found_service_canonical_name, keep_spaces=True)):
                        found_service_canonical_name = s_name_raw
                
                filials_with_service.add(f_name_raw)

        if not service_found_at_least_once:
            return f"Услуга, содержащая '{self.service_name}', не найдена ни в одном филиале."

        if not found_service_canonical_name: found_service_canonical_name = self.service_name

        if not filials_with_service:
            return f"Услуга '{found_service_canonical_name}' найдена, но не указано, в каких филиалах она доступна."

        sorted_filials = sorted(list(filials_with_service), key=normalize_text)
        # Лимит для филиалов, где есть услуга, обычно не очень большой список.
        output_limit = 20 
        total_filials_found = len(sorted_filials)

        response_message = f"Услуга '{found_service_canonical_name}' доступна в {total_filials_found} филиал(ах)."
        if total_filials_found > output_limit:
            response_message += f" Показаны первые {output_limit}:"
        
        response_parts = [response_message]
        response_parts.extend([f"- {f}" for f in sorted_filials[:output_limit]])

        if total_filials_found > output_limit:
            response_parts.append(f"\n... (и еще {total_filials_found - output_limit} филиал(ов))")
            
        return "\n".join(response_parts)


class FindSpecialistsByServiceOrCategoryAndFilial(BaseModel):
    """
    Находит специалистов в УКАЗАННОМ 'filial_name', которые предоставляют УСЛУГУ или относятся к КАТЕГОРИИ УСЛУГ, указанной в 'query_term'.
    - 'query_term' ДОЛЖЕН БЫТЬ названием КОНКРЕТНОЙ услуги или КОНКРЕТНОЙ категории.
    - НЕ используйте эту функцию, если нужно получить ВСЕХ сотрудников филиала (для этого используйте FindEmployees только с параметром filial_name).
    - НЕ передавайте в 'query_term' общие слова вроде "специалисты", "врачи" и т.п. – только названия услуг/категорий.
    """
    query_term: str = Field(description="Название КОНКРЕТНОЙ услуги ИЛИ КОНКРЕТНОЙ категории")
    filial_name: str = Field(description="Точное название филиала")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=15, description="Количество специалистов на странице для пагинации")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска специалистов."
        logging.info(f"[FC Proc] Поиск специалистов по '{self.query_term}' в филиале '{self.filial_name}' (Стр: {self.page_number}, Разм: {self.page_size}) по {len(tenant_data)} записям.")

        norm_search_term = normalize_text(self.query_term, keep_spaces=True)
        norm_filial_search = normalize_text(self.filial_name)

        specialists: Set[str] = set()
        original_filial_name_from_db: Optional[str] = None
        service_or_category_found_in_filial = False
        found_query_term_canonical_name: Optional[str] = None # Для хранения найденного канонического имени услуги/категории

        filial_exists_in_db = False
        for item in tenant_data:
            f_name_raw_check = item.get('filialName')
            if f_name_raw_check and normalize_text(f_name_raw_check) == norm_filial_search:
                 filial_exists_in_db = True
                 original_filial_name_from_db = f_name_raw_check
                 break
        if not filial_exists_in_db:
             return f"Филиал '{self.filial_name}' не найден."
        if not original_filial_name_from_db : original_filial_name_from_db = self.filial_name


        for item in tenant_data:
            e_name_raw = item.get('employeeFullName')
            s_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')
            f_name_raw = item.get('filialName')

            if not e_name_raw or not f_name_raw: continue
            if normalize_text(f_name_raw) != norm_filial_search: continue # Только целевой филиал

            norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
            norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True)

            service_match = False
            category_match = False
            matched_name = None
            
            if norm_item_s and norm_search_term: 
                if (norm_search_term in norm_item_s) or (norm_item_s in norm_search_term):
                    service_match = True
                    matched_name = s_name_raw # Сохраняем имя услуги
            
            if not service_match and norm_item_cat and norm_search_term: 
                if (norm_search_term in norm_item_cat) or (norm_item_cat in norm_search_term):
                    category_match = True
                    matched_name = cat_name_raw # Сохраняем имя категории

            if service_match or category_match:
                service_or_category_found_in_filial = True
                specialists.add(e_name_raw)
                # Обновляем каноническое имя, если оно лучше текущего
                if matched_name:
                    if found_query_term_canonical_name is None:
                         found_query_term_canonical_name = matched_name
                    # Можно добавить логику выбора более точного имени, если нужно

        if not service_or_category_found_in_filial:
            return f"Услуга или категория, содержащая '{self.query_term}', не найдена в филиале '{original_filial_name_from_db}'."
        if not specialists:
             # Используем найденное каноническое имя или исходный запрос
             query_display_name = found_query_term_canonical_name or self.query_term
             return f"Услуга/категория '{query_display_name}' найдена в филиале '{original_filial_name_from_db}', но специалисты для нее не указаны."

        all_specialists_list = sorted(list(specialists), key=lambda s: normalize_text(s, keep_spaces=True))
        
        paginated_specialists, pagination_info = apply_pagination(
            all_specialists_list,
            self.page_number,
            self.page_size
        )
        
        total_specialists = pagination_info['total_items']
        current_page_num = pagination_info['current_page']
        total_pages = pagination_info['total_pages']
        start_idx = pagination_info['start_index']
        end_idx = pagination_info['end_index']
        
        query_display_name = found_query_term_canonical_name or self.query_term
        response_message = f"В филиале '{original_filial_name_from_db}' по запросу '{query_display_name}' найдено {total_specialists} специалист(ов)."
        if total_pages > 1:
             response_message += f" Показаны {start_idx}-{end_idx} (Страница {current_page_num} из {total_pages})."
        else:
             response_message += " Все найденные специалисты:"
        
        response_parts = [response_message]
        response_parts.extend([f"- {s}" for s in paginated_specialists])

        if total_pages > 1 and current_page_num < total_pages:
            response_parts.append("\n(Для просмотра следующей страницы укажите page_number)")

        return "\n".join(response_parts)


class ListServicesInCategory(BaseModel):
    """Модель для получения списка услуг в конкретной категории."""
    category_name: str = Field(description="Точное название категории")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=20, description="Количество услуг на странице для пагинации")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг."
        logging.info(f"[FC Proc] Запрос услуг в категории '{self.category_name}' (Стр: {self.page_number}, Разм: {self.page_size}) по {len(tenant_data)} записям.")

        norm_cat_search = normalize_text(self.category_name, keep_spaces=True)
        services_in_category: Dict[str, str] = {} 
        found_category_canonical_name: Optional[str] = None
        is_canonical_exact_match_to_query: bool = False 
        category_found_globally = False

        for item in tenant_data:
            s_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')
            if not s_name_raw or not cat_name_raw: continue
            norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True)

            category_match_current_item = False
            if norm_item_cat and norm_cat_search: 
                category_match_current_item = (norm_cat_search in norm_item_cat) or \
                                              (norm_item_cat in norm_cat_search)

            if category_match_current_item:
                category_found_globally = True
                current_is_exact_match_to_query = (norm_item_cat == norm_cat_search)

                if found_category_canonical_name is None:
                    found_category_canonical_name = cat_name_raw
                    is_canonical_exact_match_to_query = current_is_exact_match_to_query
                else:
                    if current_is_exact_match_to_query and not is_canonical_exact_match_to_query:
                        found_category_canonical_name = cat_name_raw
                        is_canonical_exact_match_to_query = True
                    elif not current_is_exact_match_to_query and \
                         not is_canonical_exact_match_to_query and \
                         len(norm_item_cat) > len(normalize_text(found_category_canonical_name, keep_spaces=True)):
                        found_category_canonical_name = cat_name_raw
                
                norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
                if norm_item_s not in services_in_category:
                     services_in_category[norm_item_s] = s_name_raw

        if not category_found_globally:
            return f"Категория, содержащая '{self.category_name}', не найдена."

        if not found_category_canonical_name: found_category_canonical_name = self.category_name

        if not services_in_category:
            return f"В категории '{found_category_canonical_name}' не найдено услуг."

        all_services_list = sorted(services_in_category.values(), key=lambda s: normalize_text(s, keep_spaces=True))
        
        paginated_services, pagination_info = apply_pagination(
            all_services_list,
            self.page_number,
            self.page_size
        )

        total_services = pagination_info['total_items']
        current_page_num = pagination_info['current_page']
        total_pages = pagination_info['total_pages']
        start_idx = pagination_info['start_index']
        end_idx = pagination_info['end_index']
        
        response_message = f"В категории '{found_category_canonical_name}' найдено {total_services} услуг(и)."
        if total_pages > 1:
            response_message += f" Показаны {start_idx}-{end_idx} (Страница {current_page_num} из {total_pages})."
        
        response_parts = [response_message]
        response_parts.extend([f"- {s}" for s in paginated_services])

        if total_pages > 1 and current_page_num < total_pages:
            response_parts.append("\n(Для просмотра следующей страницы укажите page_number)")

        return "\n".join(response_parts)


class ListServicesInFilial(BaseModel):
    """Модель для получения списка всех услуг в конкретном филиале."""
    filial_name: str = Field(description="Точное название филиала")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=30, description="Количество услуг на странице для пагинации (учитывайте, что вывод также содержит заголовки категорий)")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг."
        logging.info(f"[FC Proc] Запрос услуг в филиале '{self.filial_name}' (Стр: {self.page_number}, Разм: {self.page_size}) по {len(tenant_data)} записям.")

        norm_filial_search = normalize_text(self.filial_name)
        all_services_in_filial_map: Dict[str, Dict[str, Any]] = {} # norm_service_name -> {original_name, category_name}
        original_filial_name_from_db: Optional[str] = None
        filial_found_in_db = False

        for item in tenant_data:
            s_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')
            f_name_raw_item = item.get('filialName')

            if not s_name_raw or not f_name_raw_item: continue
            norm_item_f = normalize_text(f_name_raw_item)

            if norm_item_f == norm_filial_search:
                filial_found_in_db = True
                if original_filial_name_from_db is None:
                     original_filial_name_from_db = f_name_raw_item
                
                norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
                if norm_item_s not in all_services_in_filial_map:
                    all_services_in_filial_map[norm_item_s] = {
                        "original_name": s_name_raw,
                        "category_name": cat_name_raw if cat_name_raw and cat_name_raw.strip() else "Другие услуги"
                    }
                # Если услуга уже есть, но новая запись имеет более конкретную категорию, обновляем
                elif all_services_in_filial_map[norm_item_s]["category_name"] == "Другие услуги" and cat_name_raw and cat_name_raw.strip():
                    all_services_in_filial_map[norm_item_s]["category_name"] = cat_name_raw

        if not filial_found_in_db:
            return f"Филиал '{self.filial_name}' не найден."
        if not original_filial_name_from_db: original_filial_name_from_db = self.filial_name

        if not all_services_in_filial_map:
            return f"В филиаle '{original_filial_name_from_db}' не найдено услуг."

        # Сортируем все уникальные услуги по имени
        sorted_all_unique_services = sorted(all_services_in_filial_map.values(), 
                                          key=lambda x: (normalize_text(x["category_name"], keep_spaces=True),
                                                         normalize_text(x["original_name"], keep_spaces=True)))

        paginated_services_data, pagination_info = apply_pagination(
            sorted_all_unique_services,
            self.page_number,
            self.page_size
        )

        total_services = pagination_info['total_items']
        current_page_num = pagination_info['current_page']
        total_pages = pagination_info['total_pages']
        start_idx = pagination_info['start_index']
        end_idx = pagination_info['end_index']

        response_parts = []
        header_message = f"В филиале '{original_filial_name_from_db}' найдено {total_services} уникальных услуг."
        if total_pages > 1:
            header_message += f" Показаны {start_idx}-{end_idx} (Страница {current_page_num} из {total_pages})."
        response_parts.append(header_message)

        current_category_header = None
        for service_info in paginated_services_data:
            service_name_display = service_info["original_name"]
            category_name_display = service_info["category_name"]

            if category_name_display != current_category_header:
                response_parts.append(f"\n**{category_name_display}:**")
                current_category_header = category_name_display
            response_parts.append(f"- {service_name_display}")

        if total_pages > 1 and current_page_num < total_pages:
            response_parts.append("\n(Для просмотра следующей страницы укажите page_number)")

        return "\n".join(response_parts)


class FindServicesInPriceRange(BaseModel):
    """Модель для поиска услуг в заданном ценовом диапазоне, опционально в конкретной категории."""
    min_price: Optional[int] = Field(default=0, description="Минимальная цена (включительно). 0 или null, если не ограничено.")
    max_price: Optional[int] = Field(default=None, description="Максимальная цена (включительно). Null, если не ограничено.")
    category_name: Optional[str] = Field(default=None, description="Опциональное название категории для фильтрации услуг")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=20, description="Количество услуг на странице для пагинации")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска."
        
        min_p = self.min_price if self.min_price is not None and self.min_price >= 0 else 0
        max_p = self.max_price if self.max_price is not None and self.max_price >= 0 else float('inf')
        if min_p > max_p:
            return "Ошибка: Минимальная цена не может быть больше максимальной."
        
        norm_cat_filter = normalize_text(self.category_name, keep_spaces=True) if self.category_name else None
        
        logging.info(f"[FC Proc] Поиск услуг в диапазоне цен [{min_p}-{max_p}] (Категория: {self.category_name}, Стр: {self.page_number}, Разм: {self.page_size}) по {len(tenant_data)} записям.")
        
        original_category_filter_name_from_db = self.category_name
        is_category_filter_exact_match: bool = False 
        category_filter_validated = not norm_cat_filter 

        if norm_cat_filter:
            category_exists_in_db = False
            temp_canonical_cat_name: Optional[str] = None
            
            for item_check in tenant_data:
                cat_name_raw_check = item_check.get("categoryName")
                if not cat_name_raw_check: continue
                norm_cat_check = normalize_text(cat_name_raw_check, keep_spaces=True)

                current_item_cat_match_filter = False
                if norm_cat_check and norm_cat_filter: 
                     current_item_cat_match_filter = (norm_cat_filter in norm_cat_check) or \
                                                     (norm_cat_check in norm_cat_filter)

                if current_item_cat_match_filter:
                    category_exists_in_db = True
                    current_is_exact_match_to_filter = (norm_cat_check == norm_cat_filter)
                    
                    if temp_canonical_cat_name is None:
                        temp_canonical_cat_name = cat_name_raw_check
                        is_category_filter_exact_match = current_is_exact_match_to_filter
                    else:
                        if current_is_exact_match_to_filter and not is_category_filter_exact_match:
                             temp_canonical_cat_name = cat_name_raw_check
                             is_category_filter_exact_match = True
                        elif not current_is_exact_match_to_filter and \
                             not is_category_filter_exact_match and \
                             len(norm_cat_check) > len(normalize_text(temp_canonical_cat_name, keep_spaces=True)):
                              temp_canonical_cat_name = cat_name_raw_check
            
            if category_exists_in_db:
                category_filter_validated = True
                original_category_filter_name_from_db = temp_canonical_cat_name or self.category_name
            else:
                return f"Категория '{self.category_name}' не найдена для фильтрации."

        found_services: Dict[str, Dict[str, Any]] = {}

        for item in tenant_data:
            s_name_raw = item.get('serviceName')
            price_raw = item.get('servicePrice')
            cat_name_raw = item.get('categoryName')

            if not s_name_raw or price_raw is None: continue

            try:
                price = float(price_raw)
            except (ValueError, TypeError):
                continue
            
            if not (min_p <= price <= max_p): continue

            norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True) if cat_name_raw else None
            category_match_filter = True 
            if norm_cat_filter:
                if not norm_item_cat:
                     category_match_filter = False
                else:
                     category_match_filter = (norm_cat_filter in norm_item_cat) or \
                                             (norm_item_cat in norm_cat_filter)
            
            if category_match_filter:
                norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
                if norm_item_s not in found_services:
                    found_services[norm_item_s] = {
                        'name': s_name_raw,
                        'price': price,
                        'category': cat_name_raw if cat_name_raw else "Без категории"
                    }
                else:
                    # Можно добавить логику выбора цены, если она разная (например, минимальная)
                    found_services[norm_item_s]['price'] = min(found_services[norm_item_s]['price'], price)

        if not found_services:
            range_str = f"от {min_p} до {max_p}"
            if min_p == 0 and max_p == float('inf'): range_str = "любой ценовой категории"
            elif max_p == float('inf'): range_str = f"от {min_p}"
            elif min_p == 0: range_str = f"до {max_p}"
            
            category_str = f" в категории '{original_category_filter_name_from_db}'" if self.category_name else ""
            return f"Услуги {range_str}{category_str} не найдены."

        all_services_list = sorted(found_services.values(), key=lambda x: x['price'])
        
        paginated_services, pagination_info = apply_pagination(
            all_services_list,
            self.page_number,
            self.page_size
        )
        
        total_services = pagination_info['total_items']
        current_page_num = pagination_info['current_page']
        total_pages = pagination_info['total_pages']
        start_idx = pagination_info['start_index']
        end_idx = pagination_info['end_index']

        range_str = f"от {min_p} до {max_p}" # Переопределяем для заголовка
        if min_p == 0 and max_p == float('inf'): range_str = "в любой ценовой категории"
        elif max_p == float('inf'): range_str = f"дороже {min_p}"
        elif min_p == 0: range_str = f"дешевле {max_p}"

        category_str = f" в категории '{original_category_filter_name_from_db}'" if self.category_name else ""
        response_message = f"Найдено {total_services} услуг(и) {range_str}{category_str}."
        if total_pages > 1:
             response_message += f" Показаны {start_idx}-{end_idx}, отсортированные по цене (Страница {current_page_num} из {total_pages})."
        else:
             response_message += " Отсортированы по цене:"
        
        response_parts = [response_message]
        for srv in paginated_services:
            response_parts.append(f"- {srv['name']} (Цена: {srv['price']})")

        if total_pages > 1 and current_page_num < total_pages:
            response_parts.append("\n(Для просмотра следующей страницы укажите page_number)")

        return "\n".join(response_parts)


class ListAllCategories(BaseModel):
    """Модель для получения списка всех уникальных категорий услуг."""
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=30, description="Количество категорий на странице для пагинации")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для получения категорий."
        logging.info(f"[FC Proc] Запрос всех категорий (Стр: {self.page_number}, Разм: {self.page_size}) по {len(tenant_data)} записям.")

        categories_map: Dict[str, str] = {} # norm_name -> original_name
        for item in tenant_data:
            cat_name_raw = item.get('categoryName')
            if cat_name_raw and cat_name_raw.strip():
                norm_cat = normalize_text(cat_name_raw, keep_spaces=True)
                if norm_cat not in categories_map:
                    categories_map[norm_cat] = cat_name_raw
        
        if not categories_map:
            return "Категории услуг не найдены в базе данных."

        all_categories_list = sorted(categories_map.values(), key=lambda c: normalize_text(c, keep_spaces=True))
        
        paginated_categories, pagination_info = apply_pagination(
            all_categories_list,
            self.page_number,
            self.page_size
        )
        
        total_categories = pagination_info['total_items']
        current_page_num = pagination_info['current_page']
        total_pages = pagination_info['total_pages']
        start_idx = pagination_info['start_index']
        end_idx = pagination_info['end_index']

        response_message = f"Найдено {total_categories} уникальных категорий услуг."
        if total_pages > 1:
            response_message += f" Показаны {start_idx}-{end_idx} (Страница {current_page_num} из {total_pages})."
        
        response_parts = [response_message]
        response_parts.extend([f"- {cat}" for cat in paginated_categories])

        if total_pages > 1 and current_page_num < total_pages:
            response_parts.append("\n(Для просмотра следующей страницы укажите page_number)")

        return "\n".join(response_parts)


class ListEmployeeFilials(BaseModel):
    """Модель для получения списка филиалов конкретного сотрудника."""
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска филиалов сотрудника."
        logging.info(f"[FC Proc] Запрос филиалов сотрудника '{self.employee_name}' по {len(tenant_data)} записям.")

        norm_emp_name_search = normalize_text(self.employee_name, keep_spaces=True)
        filials_set: Set[str] = set()
        found_employee_canonical_name: Optional[str] = None

        for item in tenant_data:
            e_name_raw = item.get('employeeFullName')
            f_name_raw = item.get('filialName')
            if not e_name_raw or not f_name_raw: continue
            norm_item_e_name = normalize_text(e_name_raw, keep_spaces=True)

            if norm_emp_name_search in norm_item_e_name:
                if found_employee_canonical_name is None:
                     found_employee_canonical_name = e_name_raw
                elif normalize_text(found_employee_canonical_name, keep_spaces=True) != norm_item_e_name:
                     logger.warning(f"Найдено несколько сотрудников, подходящих под '{self.employee_name}': '{found_employee_canonical_name}' и '{e_name_raw}'. Запрос неоднозначен.")
                     return f"Найдено несколько сотрудников, подходящих под имя '{self.employee_name}'. Пожалуйста, уточните ФИО."
                filials_set.add(f_name_raw)

        if not found_employee_canonical_name:
            return f"Сотрудник с именем, содержащим '{self.employee_name}', не найден."
        if not filials_set:
            return f"Для сотрудника '{found_employee_canonical_name}' не найдено филиалов в базе данных."

        sorted_filials = sorted(list(filials_set), key=normalize_text)
        # Обычно у сотрудника не так много филиалов, лимит не сильно критичен
        # Но если их может быть >10-15, можно добавить логику с лимитом. Пока без него.
        return f"Сотрудник '{found_employee_canonical_name}' работает в следующих филиалах ({len(sorted_filials)}):\n- " + "\n- ".join(sorted_filials)