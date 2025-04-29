# clinic_functions.py

import logging
import re
from typing import Optional, List, Dict, Any, Set
from pydantic import BaseModel, Field

# --- Глобальное хранилище данных ---
_internal_clinic_data: List[Dict[str, Any]] = []

def set_clinic_data(data: List[Dict[str, Any]]):
    """
    Устанавливает данные клиники для использования функциями в этом модуле.
    Вызывается один раз при инициализации основного скрипта.
    """
    global _internal_clinic_data
    _internal_clinic_data = data
    logging.info(f"[Funcs] Данные клиники ({len(_internal_clinic_data)} записей) установлены для функций.")

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
    if not normalized_name or not _internal_clinic_data: return None
    # Оптимизация: создаем карту нормализованных имен один раз, если она нужна часто
    # Но для редких вызовов можно итерировать
    for item in _internal_clinic_data:
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
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Поиск сотрудников (Имя: {self.employee_name}, Услуга: {self.service_name}, Филиал: {self.filial_name})")

        norm_emp_name = normalize_text(self.employee_name, keep_spaces=True)
        norm_service_name = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_name = normalize_text(self.filial_name)

        filtered_data = []
        for item in _internal_clinic_data:
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
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Запрос цены (Услуга: {self.service_name}, Филиал: {self.filial_name})")

        matches = []
        norm_search_term = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_name = normalize_text(self.filial_name)

        for item in _internal_clinic_data:
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
             original_filial_req_name = get_original_filial_name(norm_filial_name) or self.filial_name
             filial_search_text = f" в филиале '{original_filial_req_name}'" if self.filial_name else ""
             return f"Цена на услугу, похожую на '{self.service_name}'{filial_search_text}, не найдена."

        matches.sort(key=lambda x: (not x['exact_match'], x['match_type'] == 'category', x['price']))

        # Проверка на неоднозначность
        unique_display_names = {m['display_name'] for m in matches if not m['exact_match']}
        if len(unique_display_names) > 3 and not matches[0]['exact_match']:
             response_parts = [f"По запросу '{self.service_name}' найдено несколько похожих услуг/категорий. Уточните, пожалуйста:"]
             limit = 5
             shown_names = set()
             for match in matches:
                 if match['display_name'] not in shown_names and len(shown_names) < limit:
                     price_str = f"{match['price']:.0f} руб."
                     filial_info = f" (в '{match['filial_name']}')" if match['filial_name'] != "Любой" else ""
                     response_parts.append(f"- {match['display_name']}{filial_info}: {price_str}")
                     shown_names.add(match['display_name'])
                 if len(shown_names) >= limit: break
             response_parts.append("...")
             return "\n".join(response_parts)

        best_match = matches[0]
        original_filial_req_name = get_original_filial_name(norm_filial_name) or self.filial_name
        filial_display = original_filial_req_name if self.filial_name else best_match['filial_name']

        filial_text = f" в филиале {filial_display}" if filial_display != "Любой" else ""
        price_text = f"{best_match['price']:.0f} руб."
        clarification = ""
        if not best_match['exact_match'] and best_match['display_name']:
             clarification = f" (найдено для '{best_match['display_name']}')"

        return f"Цена на '{self.service_name}'{clarification}{filial_text} составляет {price_text}".strip()


class ListFilials(BaseModel):
    """Модель для получения списка филиалов."""
    # Нет аргументов

    def process(self) -> str:
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info("[FC Proc] Запрос списка филиалов")

        # Собираем оригинальные имена филиалов
        filials: Set[str] = set(filter(None, (item.get('filialName') for item in _internal_clinic_data)))

        if not filials:
            return "Информация о филиалах не найдена."
        # Сортируем оригинальные имена по их нормализованным версиям
        return "Доступные филиалы клиники:\n*   " + "\n*   ".join(sorted(list(filials), key=normalize_text))


class GetEmployeeServices(BaseModel):
    """Модель для получения списка услуг конкретного сотрудника."""
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")

    def process(self) -> str:
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Запрос услуг сотрудника: {self.employee_name}")

        services: Set[str] = set() # Оригинальные имена услуг
        emp_found = False
        found_names: Set[str] = set() # Оригинальные имена сотрудников
        norm_search_name = normalize_text(self.employee_name, keep_spaces=True)

        for item in _internal_clinic_data:
            emp_name_raw = item.get('employeeFullName')
            norm_item_emp = normalize_text(emp_name_raw, keep_spaces=True)

            if norm_item_emp and norm_search_name in norm_item_emp:
                 emp_found = True
                 found_names.add(emp_name_raw)
                 service_raw = item.get('serviceName')
                 if service_raw: services.add(service_raw)

        if not emp_found:
            return f"Сотрудник с именем, похожим на '{self.employee_name}', не найден."

        # Определяем наиболее точное оригинальное имя для ответа
        best_match_name = self.employee_name
        if found_names:
            # Ищем точное совпадение после нормализации
            exact_match = next((name for name in found_names if normalize_text(name, keep_spaces=True) == norm_search_name), None)
            if exact_match:
                best_match_name = exact_match
            else:
                # Если точного нет, берем первое из отсортированного списка найденных
                best_match_name = sorted(list(found_names), key=lambda n: normalize_text(n, keep_spaces=True))[0]


        if not services:
            return f"Для сотрудника '{best_match_name}' не найдено информации об услугах."
        else:
            name_clarification = ""
            if normalize_text(best_match_name, keep_spaces=True) != norm_search_name:
                name_clarification = f" (найдено по запросу '{self.employee_name}')"

            limit = 15
            # Сортируем оригинальные имена услуг
            sorted_services = sorted(list(services), key=lambda s: normalize_text(s, keep_spaces=True))
            output_services = sorted_services[:limit]
            more_services_info = f"... и еще {len(sorted_services) - limit} услуг." if len(sorted_services) > limit else ""

            return (f"Сотрудник {best_match_name}{name_clarification} выполняет следующие услуги:\n* "
                   + "\n* ".join(output_services) + f"\n{more_services_info}".strip())


class CheckServiceInFilial(BaseModel):
    """Модель для проверки наличия услуги в филиале."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")

    def process(self) -> str:
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Проверка услуги '{self.service_name}' в филиале '{self.filial_name}'")

        norm_service_name = normalize_text(self.service_name, keep_spaces=True)
        norm_filial_name = normalize_text(self.filial_name)

        filial_exists = False
        service_found_in_filial = False
        found_service_name_raw = None
        original_filial_name = None
        all_filials_db_orig = set()

        for item in _internal_clinic_data:
             item_f_name_raw = item.get('filialName')
             if not item_f_name_raw: continue
             all_filials_db_orig.add(item_f_name_raw)
             norm_item_f = normalize_text(item_f_name_raw)
             if norm_item_f == norm_filial_name:
                  filial_exists = True
                  if original_filial_name is None: original_filial_name = item_f_name_raw

                  item_s_name_raw = item.get('serviceName')
                  norm_item_s = normalize_text(item_s_name_raw, keep_spaces=True)

                  if norm_item_s and norm_service_name in norm_item_s:
                       service_found_in_filial = True
                       found_service_name_raw = item_s_name_raw
                       break # Нашли услугу в нужном филиале, выходим

        if not filial_exists:
             suggestion = f"Доступные филиалы: {', '.join(sorted(list(all_filials_db_orig), key=normalize_text))}." if all_filials_db_orig else "Список филиалов пуст."
             return f"Филиал '{self.filial_name}' не найден. {suggestion}".strip()

        # Если филиал существует, но услуга в нем не найдена
        if not service_found_in_filial:
             # Проверяем, есть ли услуга вообще где-либо
             service_name_matches_anywhere = False
             any_service_name_raw = None
             for item in _internal_clinic_data:
                  item_s_name_raw = item.get('serviceName')
                  norm_item_s = normalize_text(item_s_name_raw, keep_spaces=True)
                  if norm_item_s and norm_service_name in norm_item_s:
                      service_name_matches_anywhere = True
                      any_service_name_raw = item_s_name_raw
                      break

             if service_name_matches_anywhere:
                  clarification = ""
                  if any_service_name_raw and normalize_text(any_service_name_raw, keep_spaces=True) != norm_service_name:
                     clarification = f" (например: '{any_service_name_raw}')"
                  return f"Услуга, похожая на '{self.service_name}'{clarification}, не найдена в филиале '{original_filial_name}', но может быть доступна в других."
             else:
                  return f"Услуга, похожая на '{self.service_name}', не найдена ни в одном из филиалов."

        # Если услуга найдена в филиале
        clarification = ""
        if found_service_name_raw and normalize_text(found_service_name_raw, keep_spaces=True) != norm_service_name:
             clarification = f" (найдено: '{found_service_name_raw}')"
        return f"Да, услуга '{self.service_name}'{clarification} доступна в филиале '{original_filial_name}'."


class CompareServicePriceInFilials(BaseModel):
    """Модель для сравнения цен на услугу в нескольких филиалах."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(min_length=2, description="Список из ДВУХ или БОЛЕЕ филиалов")

    def process(self) -> str:
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Сравнение цены услуги '{self.service_name}' в филиалах: {self.filial_names}")

        norm_service_name = normalize_text(self.service_name, keep_spaces=True)

        # Собираем карту нормализованных имен филиалов из базы к оригинальным
        all_filials_db_norm_map: Dict[str, str] = {}
        for item in _internal_clinic_data:
            f_name_raw = item.get('filialName')
            if f_name_raw:
                norm_f = normalize_text(f_name_raw)
                if norm_f not in all_filials_db_norm_map:
                     all_filials_db_norm_map[norm_f] = f_name_raw

        # Валидируем запрошенные филиалы
        valid_filials_to_compare: Dict[str, str] = {} # norm_name -> original_name из запроса
        invalid_filials_req = []
        unique_requested_norm = set()

        for filial_req_raw in self.filial_names:
            norm_filial_req = normalize_text(filial_req_raw)
            if not norm_filial_req or norm_filial_req in unique_requested_norm: continue
            unique_requested_norm.add(norm_filial_req)

            if norm_filial_req in all_filials_db_norm_map:
                valid_filials_to_compare[norm_filial_req] = filial_req_raw
            else:
                invalid_filials_req.append(filial_req_raw)

        if invalid_filials_req:
             existing_filials_str = ', '.join(sorted(all_filials_db_norm_map.values(), key=normalize_text))
             return f"Филиалы не найдены: {', '.join(invalid_filials_req)}. Доступные: {existing_filials_str}."
        if len(valid_filials_to_compare) < 2:
            return "Нужно минимум два корректных и уникальных филиала для сравнения."

        # Ищем цены
        prices: Dict[str, Dict] = {
            norm_f: {'req_name': req_name, 'db_name': all_filials_db_norm_map.get(norm_f), 'price': None, 'service_name': None, 'exact_match': False}
            for norm_f, req_name in valid_filials_to_compare.items()
        }
        service_name_matches: Set[str] = set() # Оригинальные имена найденных услуг
        exact_service_name_found_raw = None

        for item in _internal_clinic_data:
            s_name_raw = item.get('serviceName')
            f_name_raw = item.get('filialName')
            price_raw = item.get('price')

            if not s_name_raw or not f_name_raw or price_raw is None or price_raw == '': continue

            norm_item_f_name = normalize_text(f_name_raw)
            if norm_item_f_name in prices:
                norm_item_s_name = normalize_text(s_name_raw, keep_spaces=True)
                is_exact_match = (norm_service_name == norm_item_s_name)
                is_partial_match = (norm_service_name in norm_item_s_name)

                if is_exact_match or is_partial_match:
                    try: price = float(str(price_raw).replace(' ', '').replace(',', '.'))
                    except (ValueError, TypeError): continue

                    current_data = prices[norm_item_f_name]
                    current_best_price = current_data['price']
                    current_is_exact = current_data['exact_match']
                    should_update = False
                    if current_best_price is None: should_update = True
                    elif is_exact_match and not current_is_exact: should_update = True
                    elif is_exact_match and current_is_exact and price < current_best_price: should_update = True
                    elif not is_exact_match and not current_is_exact and price < current_best_price: should_update = True

                    if should_update:
                        current_data['price'] = price
                        current_data['service_name'] = s_name_raw # Оригинальное имя услуги
                        current_data['exact_match'] = is_exact_match
                        service_name_matches.add(s_name_raw)
                        if is_exact_match and not exact_service_name_found_raw:
                             exact_service_name_found_raw = s_name_raw

        # Формируем ответ
        service_display_name_raw = self.service_name # По умолчанию используем имя из запроса
        if service_name_matches:
             if exact_service_name_found_raw:
                  service_display_name_raw = exact_service_name_found_raw
             else:
                  # Берем самое похожее из найденных (сортируем по нормализованному)
                  service_display_name_raw = sorted(list(service_name_matches), key=lambda s: normalize_text(s, keep_spaces=True))[0]

        clarification = ""
        if service_name_matches and normalize_text(service_display_name_raw, keep_spaces=True) != norm_service_name:
             clarification = f" (по запросу '{self.service_name}')"

        response_parts = [f"Сравнение цен на услугу '{service_display_name_raw}'{clarification}:"]
        valid_prices_found: Dict[str, float] = {} # original_req_name -> price

        # Сортируем филиалы по нормализованному имени для вывода
        sorted_filial_norms = sorted(prices.keys())

        for norm_f in sorted_filial_norms:
            data = prices[norm_f]
            original_req_name = data['req_name']
            price_value = data['price']
            found_service_name_raw = data['service_name']

            if price_value is not None:
                name_clarification = ""
                if found_service_name_raw and normalize_text(found_service_name_raw, keep_spaces=True) != normalize_text(service_display_name_raw, keep_spaces=True):
                     name_clarification = f" (для: '{found_service_name_raw}')"
                response_parts.append(f"- {original_req_name}: {price_value:.0f} руб.{name_clarification}")
                valid_prices_found[original_req_name] = price_value
            else:
                response_parts.append(f"- {original_req_name}: Цена не найдена.")

        if not service_name_matches:
             response_parts = [f"Услуга, похожая на '{self.service_name}', не найдена ни в одном из указанных филиалов."]
        elif len(valid_prices_found) >= 2:
             min_price = min(valid_prices_found.values())
             cheapest_filials = sorted([f for f, p in valid_prices_found.items() if p == min_price], key=normalize_text)
             response_parts.append(f"\nСамая низкая цена ({min_price:.0f} руб.) в: {', '.join(cheapest_filials)}.")
        elif len(valid_prices_found) == 1:
             response_parts.append("\nНедостаточно данных для сравнения (цена найдена только в одном филиале).")
        else: # Услуга найдена, но цен нет
             response_parts.append("\nНе удалось найти цены для сравнения в указанных филиалах.")


        return "\n".join(response_parts)


class FindServiceLocations(BaseModel):
    """Модель для поиска филиалов, где доступна услуга."""
    service_name: str = Field(description="Точное или максимально близкое название услуги")

    def process(self) -> str:
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Поиск филиалов для услуги: {self.service_name}")

        locations: Dict[str, str] = {} # norm_filial -> original_filial_name
        service_found = False
        found_service_names_raw: Set[str] = set()
        norm_search_service = normalize_text(self.service_name, keep_spaces=True).strip()

        best_match_name_raw = None
        min_len_diff = float('inf')
        found_exact_match = False

        for item in _internal_clinic_data:
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

                # Определение лучшего имени услуги для вывода
                is_exact_match = (norm_search_service == norm_item_s_name)
                if is_exact_match and not found_exact_match: # Первое точное совпадение
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
        if not _internal_clinic_data: return "Ошибка: Данные клиники не загружены."
        logging.info(f"[FC Proc] Поиск специалистов (Запрос: '{self.query_term}', Филиал: '{self.filial_name}')")

        matching_employees: Dict[str, str] = {} # emp_id -> original_emp_name
        norm_query = normalize_text(self.query_term, keep_spaces=True)
        norm_filial = normalize_text(self.filial_name)
        original_filial_display_name = self.filial_name

        for item in _internal_clinic_data:
            item_filial_raw = item.get("filialName")
            emp_id = item.get("employeeId")
            emp_name_raw = item.get("employeeFullName")
            if not emp_id or not emp_name_raw: continue

            norm_item_filial = normalize_text(item_filial_raw)
            if norm_item_filial == norm_filial:
                if item_filial_raw: original_filial_display_name = item_filial_raw 

                item_service_raw = item.get("serviceName")
                item_category_raw = item.get("categoryName")
                norm_item_service = normalize_text(item_service_raw, keep_spaces=True)
                norm_item_category = normalize_text(item_category_raw, keep_spaces=True)

                if (norm_item_service and norm_query in norm_item_service) or \
                   (norm_item_category and norm_query in norm_item_category):
                    if emp_id not in matching_employees:
                         matching_employees[emp_id] = emp_name_raw

        if not matching_employees:
            return f"В филиале '{original_filial_display_name}' не найдено специалистов для '{self.query_term}'."
        else:
            employee_list = ", ".join(sorted(matching_employees.values(), key=lambda n: normalize_text(n, keep_spaces=True)))
            return f"В филиале '{original_filial_display_name}' для '{self.query_term}' найдены: {employee_list}."


class ListServicesInCategory(BaseModel):
    """Модель для получения списка услуг в конкретной категории."""
    category_name: str = Field(description="Точное название категории")

    def process(self) -> str:
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Запрос услуг в категории: {self.category_name}")

        services_in_category: Set[str] = set() 
        category_found = False
        exact_category_name_raw = None
        norm_search_category = normalize_text(self.category_name, keep_spaces=True)
        found_category_names_raw = set()

        for item in _internal_clinic_data:
            cat_name_raw = item.get('categoryName')
            srv_name_raw = item.get('serviceName')

            if cat_name_raw:
                norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True)
                if norm_search_category in norm_item_cat:
                    category_found = True
                    found_category_names_raw.add(cat_name_raw)
                    # Ищем наиболее точное совпадение для имени категории
                    if exact_category_name_raw is None or norm_search_category == norm_item_cat:
                        exact_category_name_raw = cat_name_raw

                    if srv_name_raw: services_in_category.add(srv_name_raw)

        if not category_found:
            return f"Категория, похожая на '{self.category_name}', не найдена."

        # Используем найденное точное имя или самое похожее, или из запроса
        display_category_name = exact_category_name_raw or \
                                (sorted(list(found_category_names_raw), key=lambda c: normalize_text(c, keep_spaces=True))[0] if found_category_names_raw else self.category_name)

        if not services_in_category:
            return f"В категории '{display_category_name}' не найдено конкретных услуг."
        else:
            clarification = ""
            if normalize_text(display_category_name, keep_spaces=True) != norm_search_category:
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
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Запрос всех услуг в филиале: {self.filial_name}")

        services_in_filial: Set[str] = set() # original service names
        norm_search_filial = normalize_text(self.filial_name)
        original_filial_name = None
        all_filials_db_orig = set()
        filial_exists = False

        for item in _internal_clinic_data:
             item_f_name_raw = item.get('filialName')
             if not item_f_name_raw: continue
             all_filials_db_orig.add(item_f_name_raw)
             norm_item_f = normalize_text(item_f_name_raw)
             if norm_item_f == norm_search_filial:
                  filial_exists = True
                  if original_filial_name is None: original_filial_name = item_f_name_raw

                  srv_name_raw = item.get('serviceName')
                  if srv_name_raw: services_in_filial.add(srv_name_raw)

        if not filial_exists:
            suggestion = f"Доступные филиалы: {', '.join(sorted(list(all_filials_db_orig), key=normalize_text))}." if all_filials_db_orig else "Список филиалов пуст."
            return f"Филиал '{self.filial_name}' не найден. {suggestion}".strip()

        # Используем найденное оригинальное имя филиала (или из запроса, если не нашли)
        display_filial_name = original_filial_name or self.filial_name

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
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Поиск услуг по цене ({self.min_price}-{self.max_price}), Кат: {self.category_name}, Фил: {self.filial_name}")

        if self.min_price > self.max_price: return "Ошибка: Минимальная цена больше максимальной."

        matching_services: Dict[str, Dict] = {} # service_id -> {name_raw, price, filial_raw, category_raw}
        norm_category_name = normalize_text(self.category_name, keep_spaces=True)
        norm_filial_name = normalize_text(self.filial_name)

        # Получаем оригинальные имена для сообщения "не найдено"
        original_category_name_display = self.category_name
        if norm_category_name:
             found_cat_name = next((item.get("categoryName") for item in _internal_clinic_data if normalize_text(item.get("categoryName"), keep_spaces=True) == norm_category_name), None)
             if found_cat_name: original_category_name_display = found_cat_name
        original_filial_name_display = self.filial_name
        if norm_filial_name:
             found_filial_name = get_original_filial_name(norm_filial_name)
             if found_filial_name: original_filial_name_display = found_filial_name

        for item in _internal_clinic_data:
            price_raw = item.get('price')
            srv_id = item.get('serviceId')
            srv_name_raw = item.get('serviceName')
            cat_name_raw = item.get('categoryName')
            f_name_raw = item.get('filialName')

            if not srv_id or not srv_name_raw or price_raw is None or price_raw == '': continue
            try: price = float(str(price_raw).replace(' ', '').replace(',', '.'))
            except (ValueError, TypeError): continue
            if not (self.min_price <= price <= self.max_price): continue

            norm_item_cat = normalize_text(cat_name_raw, keep_spaces=True)
            norm_item_filial = normalize_text(f_name_raw)

            if norm_category_name and (not norm_item_cat or norm_category_name not in norm_item_cat): continue
            if norm_filial_name and norm_item_filial != norm_filial_name: continue

            if srv_id not in matching_services or price < matching_services[srv_id]['price']:
                 matching_services[srv_id] = {
                     'name_raw': srv_name_raw,
                     'price': price,
                     'filial_raw': f_name_raw if f_name_raw else "Не указан",
                     'category_raw': cat_name_raw if cat_name_raw else "Без категории"
                 }

        if not matching_services:
            filters_str = []
            if self.category_name: filters_str.append(f"в категории '{original_category_name_display}'")
            if self.filial_name: filters_str.append(f"в филиале '{original_filial_name_display}'")
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
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info("[FC Proc] Запрос списка всех категорий")

        categories: Set[str] = set() # Original category names
        for item in _internal_clinic_data:
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
        if not _internal_clinic_data: return "Ошибка: База данных клиники пуста."
        logging.info(f"[FC Proc] Запрос филиалов сотрудника: {self.employee_name}")

        found_filials: Set[str] = set()
        employee_found = False
        found_employee_names: Set[str] = set()
        norm_search_name = normalize_text(self.employee_name, keep_spaces=True)

        for item in _internal_clinic_data:
            emp_name_raw = item.get('employeeFullName')
            filial_name_raw = item.get('filialName')
            norm_item_emp = normalize_text(emp_name_raw, keep_spaces=True)

            if norm_item_emp and norm_search_name in norm_item_emp:
                employee_found = True
                if emp_name_raw: found_employee_names.add(emp_name_raw)
                if filial_name_raw:
                    found_filials.add(filial_name_raw)

        if not employee_found:
            return f"Сотрудник с именем, похожим на '{self.employee_name}', не найден."

        best_match_name = self.employee_name
        if found_employee_names:
            exact_match = next((name for name in found_employee_names if normalize_text(name, keep_spaces=True) == norm_search_name), None)
            if exact_match:
                best_match_name = exact_match
            else:
                best_match_name = sorted(list(found_employee_names), key=lambda n: normalize_text(n, keep_spaces=True))[0]

        if not found_filials:
            return f"Для сотрудника '{best_match_name}' не найдено информации о филиалах."
        else:
            name_clarification = ""
            if normalize_text(best_match_name, keep_spaces=True) != norm_search_name:
                name_clarification = f" (найдено по запросу '{self.employee_name}')"

            sorted_filials = sorted(list(found_filials), key=normalize_text)

            if len(sorted_filials) == 1:
                 return f"Сотрудник {best_match_name}{name_clarification} работает в филиале: {sorted_filials[0]}."
            else:
                 return f"Сотрудник {best_match_name}{name_clarification} работает в следующих филиалах:\n*   " + "\n*   ".join(sorted_filials)
