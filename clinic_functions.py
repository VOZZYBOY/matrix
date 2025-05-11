import logging
import re
import json
from typing import Optional, List, Dict, Any, Set

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

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

# --- Вспомогательная функция для получения оригинального названия филиала ---
def get_original_filial_name(normalized_name: str, tenant_data: List[Dict[str, Any]]) -> Optional[str]:
    """Находит оригинальное название филиала по его нормализованному имени в данных тенанта."""
    if not normalized_name or not tenant_data: return None
    for item in tenant_data:
        original_name = item.get("filialName")
        if original_name and normalize_text(original_name) == normalized_name:
            return original_name
    return normalized_name # Возвращаем исходное, если точного не найдено

# --- Определения Классов Функций ---

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

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска сотрудников."
        logging.info(f"[FC Proc] Поиск сотрудников (Имя: {self.employee_name}, Услуга: {self.service_name}, Филиал: {self.filial_name}) по {len(tenant_data)} записям.")

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
            service_match = True # По умолчанию true, если norm_service_name не указан
            if norm_service_name and norm_item_service: # Если и запрос, и услуга в базе существуют
                service_match = (norm_service_name in norm_item_service) or (norm_item_service in norm_service_name)
            elif norm_service_name and not norm_item_service: # Если запрос есть, а услуги в базе нет для этой строки
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

        sorted_employees = sorted(employees_info.values(), key=lambda x: normalize_text(x.get('name'), keep_spaces=True))

        for emp in sorted_employees:
            name = emp.get('name')
            if not name: continue

            services = sorted(list(emp.get('services', set())), key=lambda s: normalize_text(s, keep_spaces=True))
            filials = sorted(list(emp.get('filials', set())), key=normalize_text)

            if norm_service_name and not services: continue
            if norm_filial_name and not filials: continue

            total_found_employees += 1
            emp_info = f"- {name}"
            response_parts.append(emp_info)

        if total_found_employees == 0:
             return "Сотрудники найдены по части критериев, но ни один не соответствует всем условиям одновременно."

        header_message = f"Найдено {total_found_employees} сотрудник(ов). Все найденные сотрудники:"

        final_response = [header_message] + response_parts

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

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг сотрудника."
        logging.info(f"[FC Proc] Запрос услуг сотрудника '{self.employee_name}' по {len(tenant_data)} записям.")

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

        sorted_services = sorted(list(services), key=lambda s: normalize_text(s, keep_spaces=True))
        output_limit = 50 # Увеличим лимит для услуг сотрудника
        total_services = len(sorted_services)
        
        response_parts = [f"Сотрудник '{found_employee_name}' выполняет {total_services} услуг(и)."]
        if total_services > output_limit:
            response_parts[0] += f" Показаны первые {output_limit}:"
        
        response_parts.extend([f"- {s}" for s in sorted_services[:output_limit]])

        if total_services > output_limit:
            response_parts.append(f"\n... (и еще {total_services - output_limit} услуг(и))")

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
        
        # Сначала найдем оригинальное имя целевого филиала и проверим его существование
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

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска специалистов."
        logging.info(f"[FC Proc] Поиск специалистов по '{self.query_term}' в филиале '{self.filial_name}' по {len(tenant_data)} записям.")

        norm_query = normalize_text(self.query_term, keep_spaces=True)
        norm_filial_search = normalize_text(self.filial_name)

        specialists: Set[str] = set()
        original_filial_name_from_db: Optional[str] = None
        service_or_category_found_in_filial = False

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
            if norm_item_s and norm_query: # Услуга в базе не пустая И запрос не пустой
                service_match = (norm_query in norm_item_s) or (norm_item_s in norm_query)
            
            category_match = False
            if norm_item_cat and norm_query: # Категория в базе не пустая И запрос не пустой
                category_match = (norm_query in norm_item_cat) or (norm_item_cat in norm_query)

            if service_match or category_match:
                service_or_category_found_in_filial = True
                specialists.add(e_name_raw)

        if not service_or_category_found_in_filial:
            return f"Услуга или категория, содержащая '{self.query_term}', не найдена в филиале '{original_filial_name_from_db}'."
        if not specialists:
             return f"Услуга/категория '{self.query_term}' найдена в филиале '{original_filial_name_from_db}', но специалисты для нее не указаны."

        sorted_specialists = sorted(list(specialists), key=lambda s: normalize_text(s, keep_spaces=True))
        total_specialists = len(sorted_specialists)

        response_message = f"В филиале '{original_filial_name_from_db}' по запросу '{self.query_term}' найдено {total_specialists} специалист(ов). Все найденные специалисты:"
        
        response_parts = [response_message]
        response_parts.extend([f"- {s}" for s in sorted_specialists])

        return "\n".join(response_parts)


class ListServicesInCategory(BaseModel):
    """Модель для получения списка услуг в конкретной категории."""
    category_name: str = Field(description="Точное название категории")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг."
        logging.info(f"[FC Proc] Запрос услуг в категории '{self.category_name}' по {len(tenant_data)} записям.")

        norm_cat_search = normalize_text(self.category_name, keep_spaces=True)
        services_in_category: Dict[str, str] = {} 
        found_category_canonical_name: Optional[str] = None
        is_canonical_exact_match_to_query: bool = False # Флаг для канонического имени категории
        category_found_globally = False

        for item in tenant_data:
            s_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')
            if not s_name_raw or not cat_name_raw: continue
            norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True)

            category_match_current_item = False
            if norm_item_cat and norm_cat_search: # Убедимся, что оба существуют
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

        sorted_service_names = sorted(services_in_category.values(), key=lambda s: normalize_text(s, keep_spaces=True))
        output_limit = 75 # Можно больше, т.к. это специфичный запрос на категорию
        total_services = len(sorted_service_names)
        
        response_message = f"В категории '{found_category_canonical_name}' найдено {total_services} услуг(и)."
        if total_services > output_limit:
            response_message += f" Показаны первые {output_limit}:"
        
        response_parts = [response_message]
        response_parts.extend([f"- {s}" for s in sorted_service_names[:output_limit]])

        if total_services > output_limit:
            response_parts.append(f"\n... (и еще {total_services - output_limit} услуг(и). Для полного списка попросите продолжить или уточните запрос.)")

        return "\n".join(response_parts)


class ListServicesInFilial(BaseModel):
    """Модель для получения списка всех услуг в конкретном филиале."""
    filial_name: str = Field(description="Точное название филиала")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг."
        logging.info(f"[FC Proc] Запрос услуг в филиале '{self.filial_name}' по {len(tenant_data)} записям.")

        norm_filial_search = normalize_text(self.filial_name)
        services_map: Dict[str, str] = {} 
        categories_map: Dict[str, Set[str]] = {} 
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
                elif normalize_text(original_filial_name_from_db) != norm_item_f: # Маловероятно, но для полноты
                     logger.warning(f"Обнаружено несколько совпадающих имен для филиала '{self.filial_name}'. Используется: '{original_filial_name_from_db}'")

                norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
                if norm_item_s not in services_map:
                    services_map[norm_item_s] = s_name_raw

                if cat_name_raw and cat_name_raw.strip(): # Убедимся, что имя категории не пустое
                    norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True)
                    if norm_item_cat not in categories_map:
                         categories_map[norm_item_cat] = set()
                    categories_map[norm_item_cat].add(norm_item_s)

        if not filial_found_in_db:
            return f"Филиал '{self.filial_name}' не найден."
        if not original_filial_name_from_db: original_filial_name_from_db = self.filial_name

        total_unique_services_in_filial = len(services_map)
        if total_unique_services_in_filial == 0:
            return f"В филиале '{original_filial_name_from_db}' не найдено услуг."

        # Карта для получения оригинальных имен категорий из нормализованных
        original_category_names_map = {}
        if categories_map: # Только если есть категории
            temp_cat_map = {} # norm -> original
            for item in tenant_data: # Еще один проход для сбора оригинальных имен категорий
                 # Ограничимся только данными нужного филиала для эффективности
                 if normalize_text(item.get('filialName')) != norm_filial_search: continue
                 cat_name_raw_map = item.get('categoryName')
                 if cat_name_raw_map:
                     norm_cat_map = normalize_text(cat_name_raw_map, keep_spaces=True)
                     if norm_cat_map not in temp_cat_map:
                         temp_cat_map[norm_cat_map] = cat_name_raw_map
            original_category_names_map = temp_cat_map


        output_lines_limit = 100 # Общий лимит на количество строк в выводе (включая заголовки категорий)
        lines_count = 0
        services_listed_count = 0
        
        response_parts = []
        
        sorted_category_keys = sorted(categories_map.keys(), key=lambda c_norm: normalize_text(original_category_names_map.get(c_norm, c_norm), keep_spaces=True))

        for norm_cat_key in sorted_category_keys:
             if lines_count >= output_lines_limit: break
             
             original_cat_name_display = original_category_names_map.get(norm_cat_key, norm_cat_key)
             response_parts.append(f"\n**{original_cat_name_display}:**")
             lines_count += 1
             
             service_norm_names_in_cat = categories_map[norm_cat_key]
             # Получаем оригинальные имена и сортируем их
             service_original_names_in_cat = sorted(
                 [services_map[norm_s] for norm_s in service_norm_names_in_cat if norm_s in services_map],
                 key=lambda s_orig: normalize_text(s_orig, keep_spaces=True)
             )

             for service_name_display in service_original_names_in_cat:
                  if lines_count >= output_lines_limit: break
                  response_parts.append(f"- {service_name_display}")
                  lines_count += 1
                  services_listed_count += 1
        
        # Услуги без категорий
        if lines_count < output_lines_limit:
             services_without_category_originals = []
             processed_in_categories_norm = set()
             for cat_services_norm_set in categories_map.values():
                 processed_in_categories_norm.update(cat_services_norm_set)
             
             for norm_s, original_s in services_map.items():
                 if norm_s not in processed_in_categories_norm:
                      services_without_category_originals.append(original_s)
             
             if services_without_category_originals:
                  if lines_count < output_lines_limit:
                      response_parts.append(f"\n**Другие услуги:**")
                      lines_count +=1
                  
                  sorted_others_originals = sorted(services_without_category_originals, key=lambda s: normalize_text(s, keep_spaces=True))
                  for service_name_display in sorted_others_originals:
                      if lines_count >= output_lines_limit: break
                      response_parts.append(f"- {service_name_display}")
                      lines_count += 1
                      services_listed_count +=1

        header_message = f"В филиале '{original_filial_name_from_db}' найдено {total_unique_services_in_filial} уникальных услуг."
        if services_listed_count < total_unique_services_in_filial and services_listed_count > 0:
            header_message += f" Показаны первые {services_listed_count} из них (сгруппированные по категориям, если есть):"
        elif services_listed_count == total_unique_services_in_filial and total_unique_services_in_filial > 0:
            header_message += " Представлен полный список:"
        
        final_response_parts = [header_message] + response_parts

        if services_listed_count < total_unique_services_in_filial:
             final_response_parts.append(f"\n(Примечание для LLM: Показано {services_listed_count} из {total_unique_services_in_filial} услуг из-за внутреннего лимита строк вывода. Полный список может быть больше.)")

        return "\n".join(final_response_parts)


class FindServicesInPriceRange(BaseModel):
    """Модель для поиска услуг в заданном ценовом диапазоне."""
    min_price: float = Field(description="Минимальная цена")
    max_price: float = Field(description="Максимальная цена")
    category_name: Optional[str] = Field(default=None, description="Опционально: категория для фильтрации")
    filial_name: Optional[str] = Field(default=None, description="Опционально: филиал для фильтрации")

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска услуг."
        logging.info(f"[FC Proc] Поиск услуг в диапазоне {self.min_price}-{self.max_price} (Кат: {self.category_name}, Фил: {self.filial_name}) по {len(tenant_data)} записям.")

        if self.min_price > self.max_price:
             return "Ошибка: Минимальная цена не может быть больше максимальной."

        norm_cat_filter = normalize_text(self.category_name, keep_spaces=True) if self.category_name else None
        norm_filial_filter = normalize_text(self.filial_name) if self.filial_name else None

        matched_services_info: Dict[str, Dict[str, Any]] = {} 

        original_filial_filter_name_from_db = self.filial_name
        if norm_filial_filter:
             filial_exists = False
             for item_check in tenant_data:
                  f_name_raw_check = item_check.get("filialName")
                  if f_name_raw_check and normalize_text(f_name_raw_check) == norm_filial_filter:
                      filial_exists = True
                      original_filial_filter_name_from_db = f_name_raw_check
                      break
             if not filial_exists: return f"Филиал '{self.filial_name}' не найден."
        
        original_category_filter_name_from_db = self.category_name
        is_category_filter_exact_match: bool = False # Флаг для точного совпадения имени категории фильтра

        if norm_cat_filter:
            category_exists_in_db = False
            temp_canonical_cat_name: Optional[str] = None
            
            for item_check in tenant_data:
                cat_name_raw_check = item_check.get("categoryName")
                if not cat_name_raw_check: continue
                norm_cat_check = normalize_text(cat_name_raw_check, keep_spaces=True)

                current_item_cat_match_filter = False
                if norm_cat_check and norm_cat_filter: # Оба существуют
                    current_item_cat_match_filter = (norm_cat_filter in norm_cat_check) or \
                                                    (norm_cat_check in norm_cat_filter)

                if current_item_cat_match_filter:
                    category_exists_in_db = True
                    current_is_exact = (norm_cat_check == norm_cat_filter)
                    if temp_canonical_cat_name is None:
                        temp_canonical_cat_name = cat_name_raw_check
                        is_category_filter_exact_match = current_is_exact
                    else:
                        if current_is_exact and not is_category_filter_exact_match:
                            temp_canonical_cat_name = cat_name_raw_check
                            is_category_filter_exact_match = True
                        elif not current_is_exact and \
                             not is_category_filter_exact_match and \
                             len(norm_cat_check) > len(normalize_text(temp_canonical_cat_name, keep_spaces=True)):
                            temp_canonical_cat_name = cat_name_raw_check
            
            if not category_exists_in_db: return f"Категория, для которой применен фильтр ('{self.category_name}'), не найдена в базе."
            if temp_canonical_cat_name: original_category_filter_name_from_db = temp_canonical_cat_name


        for item in tenant_data:
            s_name_raw = item.get('serviceName')
            cat_name_raw_item = item.get('categoryName')
            f_name_raw_item = item.get('filialName')
            price_raw = item.get('price')

            if not s_name_raw or price_raw is None or price_raw == '': continue

            # Фильтрация по категории (теперь симметричная)
            if norm_cat_filter:
                norm_item_cat = normalize_text(cat_name_raw_item, keep_spaces=True) if cat_name_raw_item else ""
                category_passes_filter = False
                if norm_item_cat and norm_cat_filter: # Оба существуют
                     category_passes_filter = (norm_cat_filter in norm_item_cat) or \
                                              (norm_item_cat in norm_cat_filter)
                if not category_passes_filter: continue
            
            # Фильтрация по филиалу (остается как есть, т.к. там обычно точное совпадение)
            if norm_filial_filter:
                norm_item_f = normalize_text(f_name_raw_item) if f_name_raw_item else ""
                if norm_item_f != norm_filial_filter: continue

            try: price = float(str(price_raw).replace(' ', '').replace(',', '.'))
            except (ValueError, TypeError): continue
            if not (self.min_price <= price <= self.max_price): continue

            norm_item_s = normalize_text(s_name_raw, keep_spaces=True)
            if norm_item_s not in matched_services_info:
                matched_services_info[norm_item_s] = {
                    'original_service_name': s_name_raw,
                    'price': price, 
                    'original_category_name': cat_name_raw_item if cat_name_raw_item else "Без категории",
                    'filials_providing_at_this_price': set() # Филиалы, где эта услуга с этой ценой
                }
            # Если услуга в этом ценовом диапазоне предлагается в филиале (или без указания филиала, если фильтра нет)
            # и цена совпадает с уже записанной (или это первая запись для услуги)
            if matched_services_info[norm_item_s]['price'] == price:
                if f_name_raw_item: # Если филиал указан в данных
                    matched_services_info[norm_item_s]['filials_providing_at_this_price'].add(f_name_raw_item)
                elif not norm_filial_filter: # Если филиал не указан в данных И нет фильтра по филиалу
                    matched_services_info[norm_item_s]['filials_providing_at_this_price'].add("Любой (не указан)")

            # Логика для случая разных цен на одну и ту же услугу (без учета филиала)
            # Если такая же услуга найдена с другой ценой в этом же диапазоне,
            # это может потребовать более сложной обработки или выбора (например, минимальной цены)
            # Текущая логика просто возьмет первую встреченную цену.


        if not matched_services_info:
             cat_str = f" в категории '{original_category_filter_name_from_db}'" if self.category_name else ""
             filial_str = f" в филиаle '{original_filial_filter_name_from_db}'" if self.filial_name else ""
             return f"Услуги в ценовом диапазоне от {self.min_price:.0f} до {self.max_price:.0f} руб.{cat_str}{filial_str} не найдены."

        sorted_matches = sorted(matched_services_info.values(), key=lambda x: (x['price'], normalize_text(x['original_service_name'], keep_spaces=True)))

        output_limit = 30 # Лимит на количество услуг в диапазоне цен
        total_services_in_range = len(sorted_matches)
        
        header_parts = [f"Найдено {total_services_in_range} услуг(и) от {self.min_price:.0f} до {self.max_price:.0f} руб."]
        if self.category_name: header_parts.append(f"(Категория: '{original_category_filter_name_from_db}')")
        if self.filial_name: header_parts.append(f"(Филиал: '{original_filial_filter_name_from_db}')")
        if total_services_in_range > output_limit: header_parts.append(f". Показаны первые {output_limit}:")
        
        response_parts = [" ".join(header_parts)]

        for i, match_info in enumerate(sorted_matches):
             if i >= output_limit: break
             filials_str_suffix = ""
             # Если фильтра по филиалу не было, показываем, где услуга доступна по этой цене
             if not norm_filial_filter and match_info['filials_providing_at_this_price']:
                 sorted_filials_for_service = sorted(list(match_info['filials_providing_at_this_price']), key=normalize_text)
                 filials_str_suffix = f" ({', '.join(sorted_filials_for_service[:3])}" # Показать до 3 филиалов
                 if len(sorted_filials_for_service) > 3:
                     filials_str_suffix += f" и еще в {len(sorted_filials_for_service) - 3}"
                 filials_str_suffix += ")"

             response_parts.append(f"- {match_info['original_service_name']}: {match_info['price']:.0f} руб.{filials_str_suffix}")

        if total_services_in_range > output_limit:
             response_parts.append(f"\n... (и еще {total_services_in_range - output_limit} услуг(и))")

        return "\n".join(response_parts)


class ListAllCategories(BaseModel):
    """Модель для получения списка всех категорий услуг."""

    def process(self, tenant_data_docs: Optional[List[Document]] = None, raw_data: Optional[List[Dict]] = None) -> str:
        tenant_data = raw_data
        if not tenant_data: return "Ошибка: Не удалось получить структурированные данные тенанта для поиска категорий."
        logging.info(f"[FC Proc] Запрос списка всех категорий по {len(tenant_data)} записям.")

        categories_set: Set[str] = set()
        for item in tenant_data:
            cat_name = item.get("categoryName")
            if cat_name and isinstance(cat_name, str) and cat_name.strip():
                categories_set.add(cat_name.strip())

        if not categories_set:
            return "Список категорий услуг пуст или не найден в данных."

        sorted_categories = sorted(list(categories_set), key=lambda c: normalize_text(c, keep_spaces=True))
        output_limit = 75 # Категорий может быть много
        total_categories = len(sorted_categories)
        
        response_parts = [f"Доступно {total_categories} категорий услуг."]
        if total_categories > output_limit:
            response_parts[0] += f" Показаны первые {output_limit}:"
            
        response_parts.extend([f"- {c}" for c in sorted_categories[:output_limit]])

        if total_categories > output_limit:
             response_parts.append(f"\n... (и еще {total_categories - output_limit} категорий. Для полного списка попросите продолжить или используйте более точный запрос.)")

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