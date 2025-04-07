import json
import time
import os
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
try:
    from IPython.display import display, Markdown
    is_ipython = True
except ImportError:
    is_ipython = False
    def display(x): print(x)
    class Markdown:
        def __init__(self, data):
            self.data = data
        def __str__(self):
            return str(self.data)
        def __repr__(self):
            return f"Markdown({repr(self.data)})"

try:
    from yandex_cloud_ml_sdk import YCloudML
    from yandex_cloud_ml_sdk.search_indexes import (
        StaticIndexChunkingStrategy,
        HybridSearchIndexType,
        ReciprocalRankFusionIndexCombinationStrategy,
    )
except ImportError:
    logging.error("Ошибка: Библиотека yandex_cloud_ml_sdk не найдена.")
    logging.error("Пожалуйста, установите её командой:")
    logging.error("pip install --quiet flit")
    logging.error("pip install --quiet -I git+https://github.com/yandex-cloud/yandex-cloud-ml-sdk.git@assistants_fc#egg=yandex-cloud-ml-sdk")
    logging.error("pip install --upgrade --quiet pydantic")
    raise ImportError("Требуемые библиотеки не установлены")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("google.auth.transport.requests").setLevel(logging.WARNING)
logging.getLogger("grpc").setLevel(logging.WARNING)
logging.getLogger("hpack").setLevel(logging.WARNING)

def printx(string):
    if is_ipython:
        try:
            if string is not None:
                 display(Markdown(str(string)))
            else:
                 logging.warning("Попытка отобразить None через printx.")
        except Exception as e:
            logging.warning(f"Ошибка отображения Markdown: {e}. Выводим как обычный текст.")
            print(str(string))
    else:
        print(str(string) if string is not None else "")


FOLDER_ID = "b1gnq2v60fut60hs9vfb"
API_KEY = "AQVNw5Kg0jXoaateYQWdSr2k8cbst_y4_WcbvZrW"
JSON_DATA_PATH = "base/cleaned_data.json"
MODEL_URI_SHORT = "yandexgpt/rc"
MODEL_URI_FULL = f"gpt://{FOLDER_ID}/{MODEL_URI_SHORT}"
RAG_INDEX_NAME = "clinic_rag_index_v2"
ASSISTANT_NAME = "ClinicAssistant_V2"


try:
    sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)
    logging.info(f"SDK инициализирован для папки {FOLDER_ID}.")
except Exception as e:
    logging.critical(f"Критическая ошибка инициализации SDK: {e}", exc_info=True)
    raise

global_clinic_data = []
try:
    base_dir = os.path.dirname(JSON_DATA_PATH)
    if base_dir and not os.path.exists(base_dir):
         try:
             os.makedirs(base_dir, exist_ok=True)
             logging.warning(f"Директория '{base_dir}' не найдена, создана.")
         except OSError as mkdir_err:
             logging.critical(f"Критическая ошибка: Не удалось создать директорию '{base_dir}': {mkdir_err}")
             raise

    if not os.path.exists(JSON_DATA_PATH):
        logging.critical(f"Критическая ошибка: Файл данных '{JSON_DATA_PATH}' не найден.")
        logging.info(f"Пожалуйста, убедитесь, что файл '{JSON_DATA_PATH}' существует и содержит JSON массив объектов.")
        raise FileNotFoundError(f"Файл данных '{JSON_DATA_PATH}' не найден")

    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        global_clinic_data = json.load(f)
    if not isinstance(global_clinic_data, list):
         raise ValueError("Ожидался список объектов в JSON")
    logging.info(f"Успешно загружено {len(global_clinic_data)} записей из {JSON_DATA_PATH}")
except Exception as e:
    logging.critical(f"Критическая ошибка при загрузке или парсинге JSON данных из {JSON_DATA_PATH}: {e}", exc_info=True)
    raise

def preprocess_json_for_rag(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rag_chunks_data = []
    employee_descriptions = {}
    service_descriptions = {}

    logging.info("Начало препроцессинга данных для RAG...")
    for item in data:
        emp_id = item.get("employeeId")
        emp_desc_raw = item.get("employeeDescription")
        emp_desc = (emp_desc_raw or "").strip()
        emp_name = item.get("employeeFullName", "Имя неизвестно")

        srv_id = item.get("serviceId")
        srv_desc_raw = item.get("serviceDescription")
        srv_desc = (srv_desc_raw or "").strip()
        srv_name = item.get("serviceName", "Название неизвестно")

        if emp_id and emp_desc and emp_desc.lower() not in ('', 'опытный специалист', 'нет описания', 'dnasdasj'):
            if emp_id not in employee_descriptions or len(emp_desc) > len(employee_descriptions[emp_id]['desc']):
                 employee_descriptions[emp_id] = {'name': emp_name, 'desc': emp_desc}

        if srv_id and srv_desc:
             if srv_id not in service_descriptions or len(srv_desc) > len(service_descriptions[srv_id]['desc']):
                 service_descriptions[srv_id] = {'name': srv_name, 'desc': srv_desc}

    for emp_id, info in employee_descriptions.items():
        text = f"Информация о сотруднике {info['name']}: {info['desc']}"
        rag_chunks_data.append({"id": f"emp_{emp_id}", "text": text})

    for srv_id, info in service_descriptions.items():
        # Добавляем название услуги в текст для лучшего поиска RAG
        text = f"Услуга: {info.get('name', 'Без названия')}\nОписание: {info['desc']}"
        rag_chunks_data.append({"id": f"srv_{srv_id}", "text": text})

    logging.info(f"Препроцессинг завершен. Создано {len(rag_chunks_data)} уникальных описаний для RAG.")
    return rag_chunks_data

def upload_rag_chunks(chunks_data: List[Dict[str, str]]) -> List[Any]:
    uploaded_files = []
    logging.info(f"Начало загрузки {len(chunks_data)} описаний в Yandex Cloud...")
    if not chunks_data:
        logging.warning("Нет данных для загрузки в RAG.")
        return []
    for chunk in tqdm(chunks_data, desc="Загрузка описаний RAG"):
        try:
            if not chunk.get('text'):
                 logging.warning(f"Пропуск пустого описания для ID: {chunk.get('id', 'N/A')}")
                 continue
            file_obj = sdk.files.upload_bytes(
                chunk['text'].encode('utf-8'),
                name=chunk.get('id', f'chunk_{int(time.time()*1000)}'),
                ttl_days=1,
                expiration_policy="static",
                mime_type="text/plain"
            )
            uploaded_files.append(file_obj)
        except Exception as e:
            logging.error(f"Ошибка загрузки описания {chunk.get('id', 'N/A')}: {e}")
    logging.info(f"Загрузка описаний завершена. Загружено {len(uploaded_files)} файлов.")
    return uploaded_files

def create_rag_index(files: List[Any], index_name: str) -> Optional[Any]:
    if not files:
        logging.warning("Нет файлов для создания RAG индекса.")
        return None
    logging.info(f"Начало создания RAG индекса '{index_name}' для {len(files)} файлов...")
    try:
        op = sdk.search_indexes.create_deferred(
            files,
            index_type=HybridSearchIndexType(
                chunking_strategy=StaticIndexChunkingStrategy(
                    max_chunk_size_tokens=1000,
                    chunk_overlap_tokens=100
                ),
                combination_strategy=ReciprocalRankFusionIndexCombinationStrategy(),
            ),
            name=index_name,
            description="Индекс по описаниям врачей и услуг клиники Med YU Med",
            ttl_days=1,
            expiration_policy="since_last_active"
        )
        logging.info(f"Запущена операция создания индекса: {op.id}. Ожидание завершения...")
        index = op.wait(timeout=300)
        logging.info(f"RAG индекс '{index_name}' успешно создан: id={index.id}")
        return index
    except TimeoutError:
         logging.error(f"Ошибка создания RAG индекса '{index_name}': Превышен таймаут ожидания операции.")
         return None
    except Exception as e:
        logging.error(f"Ошибка создания RAG индекса '{index_name}': {e}", exc_info=True)
        return None

class FindEmployees(BaseModel):
    employee_name: Optional[str] = Field(description="Часть или полное ФИО сотрудника", default=None)
    service_name: Optional[str] = Field(description="Точное или частичное название услуги, которую должен выполнять сотрудник", default=None)
    filial_name: Optional[str] = Field(description="Название филиала, где должен работать сотрудник", default=None)

    def process(self, thread) -> str:
        logging.info(f"[FC Proc] Поиск сотрудников (Имя: {self.employee_name}, Услуга: {self.service_name}, Филиал: {self.filial_name})")
        if not global_clinic_data: return "База данных клиники пуста."

        filtered_data = []
        for item in global_clinic_data:
            emp_name_item = item.get('employeeFullName', '')
            service_name_item = item.get('serviceName', '')
            filial_name_item = item.get('filialName', '')

            emp_match = (not self.employee_name or self.employee_name.lower() in (emp_name_item.lower() if emp_name_item else ''))

            service_match = False
            if not self.service_name:
                 service_match = True
            elif service_name_item:
                 search_lower = self.service_name.lower()
                 item_lower = service_name_item.lower()
                 service_match = (search_lower in item_lower) or (item_lower in search_lower)

            filial_match = (not self.filial_name or (filial_name_item and self.filial_name.lower() == filial_name_item.lower()))

            if emp_match and service_match and filial_match:
                filtered_data.append(item)

        if not filtered_data:
            return "Сотрудники по вашему запросу не найдены."

        employees_info = {}
        for item in filtered_data:
             e_id = item.get('employeeId')
             if e_id:
                  if e_id not in employees_info:
                       emp_desc = item.get('employeeDescription', '')
                       if emp_desc is None: emp_desc = ''
                       employees_info[e_id] = {
                            'name': item.get('employeeFullName'),
                            'description': emp_desc.strip(),
                            'services': set(),
                            'filials': set()
                       }
                  else:
                       current_desc = item.get('employeeDescription', '')
                       if current_desc and len(current_desc) > len(employees_info[e_id]['description']):
                            employees_info[e_id]['description'] = current_desc.strip()

                  s_name = item.get('serviceName')
                  f_name = item.get('filialName')
                  if s_name: employees_info[e_id]['services'].add(s_name)
                  if f_name: employees_info[e_id]['filials'].add(f_name)

        response_parts = ["Найдены следующие сотрудники:"]
        if not employees_info:
             return "Не удалось извлечь информацию о сотрудниках из отфильтрованных данных."

        limit = 7
        count = 0
        sorted_employees = sorted(employees_info.values(), key=lambda x: x.get('name') or '')
        for emp in sorted_employees:
            if count >= limit:
                 response_parts.append(f"\n... (найдено еще {len(employees_info) - limit} сотрудников, пожалуйста, уточните запрос для более точного поиска)")
                 break
            name = emp.get('name') or 'Имя не указано'
            services = ', '.join(sorted(list(emp.get('services', set()))))
            filials = ', '.join(sorted(list(emp.get('filials', set()))))
            service_str = f"   Услуги: {services}" if services else ""
            filial_str = f"   Филиалы: {filials}" if filials else ""

            emp_info = f"- {name}"
            if filial_str: emp_info += f"\n{filial_str}"
            if service_str: emp_info += f"\n{service_str}"

            response_parts.append(emp_info)
            count += 1

        return "\n".join(response_parts)

class GetServicePrice(BaseModel):
    service_name: Optional[str] = Field(description="Точное или максимально близкое название услуги (например, 'Soprano Пальцы для женщин')", default=None)
    filial_name: Optional[str] = Field(description="Точное название филиала (например, 'Москва-сити'), если нужно уточнить цену в конкретном месте", default=None)

    def process(self, thread) -> str:
        if self.service_name is None:
            return "Пожалуйста, уточните название услуги, цену на которую вы хотите узнать."

        logging.info(f"[FC Proc] Запрос цены (Услуга: {self.service_name}, Филиал: {self.filial_name})")
        if not global_clinic_data: return "База данных клиники пуста."

        matches = []
        search_term_lower = self.service_name.lower()

        for item in global_clinic_data:
            s_name = item.get('serviceName')
            cat_name = item.get('categoryName')
            f_name = item.get('filialName')
            price = item.get('price')

            if price is None: continue

            filial_match = True
            if self.filial_name:
                filial_match = f_name and self.filial_name.lower() == f_name.lower()
            if not filial_match: continue

            service_name_match = False
            category_name_match = False
            item_service_name = None
            exact_match_flag = False

            if s_name and search_term_lower in s_name.lower():
                service_name_match = True
                item_service_name = s_name
                exact_match_flag = (search_term_lower == s_name.lower())
            elif cat_name and search_term_lower in cat_name.lower():
                category_name_match = True
                item_service_name = cat_name
                exact_match_flag = (search_term_lower == cat_name.lower())

            if service_name_match or category_name_match:
                matches.append({
                    'service_name': item_service_name,
                    'price': price,
                    'filial_name': f_name if f_name else "Любой",
                    'exact_match': exact_match_flag,
                    'match_type': 'service' if service_name_match else 'category'
                })

        if not matches:
             filial_search_text = f" в филиале '{self.filial_name}'" if self.filial_name else ""
             return f"Услуга, похожая на '{self.service_name}'{filial_search_text}, не найдена или цена для нее не указана."

        # Если найдено много разных услуг по общему термину
        unique_service_names = {m['service_name'] for m in matches if m.get('service_name')}
        # Условие стало строже: если запрос короткий ИЛИ не было точного совпадения И найдено много вариантов
        is_generic_query = len(self.service_name) < 15 or not any(m['exact_match'] for m in matches)
        if len(unique_service_names) > 3 and is_generic_query:
             response_parts = [f"По вашему запросу '{self.service_name}' найдено несколько видов услуг. Уточните, пожалуйста, какая именно вас интересует:"]
             limit = 5
             for i, name in enumerate(sorted(list(unique_service_names))):
                  if i >= limit:
                       response_parts.append("...")
                       break
                  response_parts.append(f"- {name}")
             return "\n".join(response_parts)


        matches.sort(key=lambda x: (not x['exact_match'], x['match_type'] == 'category', x['price']))
        best_match = matches[0]
        filial_display_name = self.filial_name if self.filial_name else best_match['filial_name']
        filial_text = f" в филиале {filial_display_name}" if filial_display_name != "Любой" else ""
        match_type_info = f"(Найдено по {'точному' if best_match['exact_match'] else 'частичному'} совпадению в {'названии услуги' if best_match['match_type'] == 'service' else 'названии категории'})"

        return f"Цена на '{best_match['service_name']}'{filial_text} составляет {best_match['price']} руб. {match_type_info}".strip()

class ListFilials(BaseModel):
     def process(self, thread) -> str:
         logging.info("[FC Proc] Запрос списка филиалов")
         if not global_clinic_data: return "База данных клиники пуста."
         filials = set(filter(None, (item.get('filialName') for item in global_clinic_data)))
         if not filials:
             return "В базе данных нет информации о филиалах."
         return "Доступные филиалы клиники:\n*   " + "\n*   ".join(sorted(list(filials)))

class GetEmployeeServices(BaseModel):
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")

    def process(self, thread) -> str:
        logging.info(f"[FC Proc] Запрос услуг сотрудника: {self.employee_name}")
        if not global_clinic_data: return "База данных клиники пуста."

        services = set()
        emp_found = False
        found_names = set()

        for item in global_clinic_data:
            emp_name = item.get('employeeFullName')
            if emp_name and self.employee_name.lower() in emp_name.lower():
                 emp_found = True
                 found_names.add(emp_name)
                 service = item.get('serviceName')
                 if service: services.add(service)

        if not emp_found:
            return f"Сотрудник, похожий на '{self.employee_name}', не найден."

        exact_match = next((name for name in found_names if name.lower() == self.employee_name.lower()), None)
        emp_display_name = exact_match if exact_match else (sorted(list(found_names))[0] if found_names else self.employee_name)

        if not services:
            return f"Для сотрудника '{emp_display_name}' не найдено информации об услугах."
        else:
            name_clarification = f"(найдено по запросу '{self.employee_name}')" if not exact_match and len(found_names) > 1 else ""
            if not exact_match and len(found_names) == 1: name_clarification = ""

            limit = 15
            sorted_services = sorted(list(services))
            output_services = sorted_services[:limit]
            more_services_info = f"... и еще {len(sorted_services) - limit} услуг." if len(sorted_services) > limit else ""

            return f"Сотрудник {emp_display_name} {name_clarification} выполняет следующие услуги:\n* " + "\n* ".join(output_services) + f"\n{more_services_info}".strip()

class CheckServiceInFilial(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: Optional[str] = Field(description="Точное название филиала", default=None)

    def process(self, thread) -> str:
        logging.info(f"[FC Proc] Проверка наличия услуги '{self.service_name}' в филиале '{self.filial_name}'")

        if self.filial_name is None:
            return "Пожалуйста, уточните название филиала, в котором вы хотите проверить наличие услуги."

        if not global_clinic_data: return "База данных клиники пуста."

        service_found_in_filial = False
        exact_service_name = None
        exact_filial_name = None
        filial_exists = False
        service_name_matches_in_filial = set()

        all_filials_db_orig = set(item.get('filialName') for item in global_clinic_data if item.get('filialName'))
        all_filials_db_lower = {f.lower() for f in all_filials_db_orig}

        if self.filial_name.lower() in all_filials_db_lower:
            filial_exists = True
            exact_filial_name = next((f for f in all_filials_db_orig if f.lower() == self.filial_name.lower()), self.filial_name)
        else:
             suggestion = f"Возможно, вы имели в виду один из этих: {', '.join(sorted(list(all_filials_db_orig)))}?" if all_filials_db_orig else ""
             return f"Филиал '{self.filial_name}' не найден. {suggestion}".strip()

        for item in global_clinic_data:
             s_name = item.get('serviceName')
             f_name = item.get('filialName')

             if f_name and f_name.lower() == exact_filial_name.lower():
                  if s_name and self.service_name.lower() in s_name.lower():
                      service_found_in_filial = True
                      service_name_matches_in_filial.add(s_name)
                      if self.service_name.lower() == s_name.lower():
                          exact_service_name = s_name
                          break

        if service_found_in_filial:
             display_service_name = exact_service_name if exact_service_name else sorted(list(service_name_matches_in_filial))[0]
             clarification = f" (найдено по запросу '{self.service_name}')" if not exact_service_name else ""
             return f"Да, услуга '{display_service_name}'{clarification} доступна в филиале '{exact_filial_name}'."
        else:
             service_name_matches_anywhere = set()
             for item in global_clinic_data:
                  s_name = item.get('serviceName')
                  if s_name and self.service_name.lower() in s_name.lower():
                      service_name_matches_anywhere.add(s_name)

             if service_name_matches_anywhere:
                  any_service_name = sorted(list(service_name_matches_anywhere))[0]
                  return f"Услуга, похожая на '{any_service_name}', не найдена в филиале '{exact_filial_name}', но может быть доступна в других филиалах."
             else:
                  return f"Услуга, похожая на '{self.service_name}', не найдена ни в одном филиале."

class CompareServicePriceInFilials(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(description="Список из ДВУХ или БОЛЕЕ ТОЧНЫХ названий филиалов для сравнения")

    def process(self, thread) -> str:
        logging.info(f"[FC Proc] Сравнение цены услуги '{self.service_name}' в филиалах: {self.filial_names}")
        if not global_clinic_data: return "База данных клиники пуста."
        unique_filial_names_lower = set(fn.lower() for fn in self.filial_names)
        if not self.filial_names or len(unique_filial_names_lower) < 2:
             return "Для сравнения нужно указать хотя бы два РАЗНЫХ филиала."

        prices = {}
        exact_service_name_found = None
        service_name_matches = set()
        all_filials_in_db_orig_case = {item.get('filialName','').lower(): item.get('filialName')
                                      for item in global_clinic_data if item.get('filialName')}
        all_filials_in_db_lower = set(all_filials_in_db_orig_case.keys())

        invalid_filials = []
        filials_to_compare_orig_case = {}

        for filial_req in self.filial_names:
            filial_req_lower = filial_req.lower()
            if filial_req_lower in all_filials_in_db_lower:
                original_filial_name = all_filials_in_db_orig_case[filial_req_lower]
                if filial_req_lower not in filials_to_compare_orig_case:
                    filials_to_compare_orig_case[filial_req_lower] = original_filial_name
                    prices[filial_req_lower] = {'name': original_filial_name, 'price': "Не найдена"}
            else:
                invalid_filials.append(filial_req)

        if invalid_filials:
             existing_filials_str = ', '.join(sorted(list(all_filials_in_db_orig_case.values())))
             return f"Следующие филиалы не найдены: {', '.join(invalid_filials)}. Доступные филиалы: {existing_filials_str}."
        if len(filials_to_compare_orig_case) < 2:
            return "Недостаточно корректных филиалов для сравнения (нужно минимум два)."

        for item in global_clinic_data:
            s_name = item.get('serviceName')
            f_name = item.get('filialName')
            price = item.get('price')

            if not s_name or not f_name or price is None: continue

            f_name_lower = f_name.lower()
            if f_name_lower in prices:
                if self.service_name.lower() in s_name.lower():
                     current_status = prices[f_name_lower]['price']
                     is_new_exact_match = self.service_name.lower() == s_name.lower()
                     is_current_exact_match = isinstance(current_status, (int, float)) and \
                                              prices[f_name_lower].get('exact_service_match', False)

                     if current_status == "Не найдена" or (is_new_exact_match and not is_current_exact_match):
                           prices[f_name_lower]['price'] = price
                           prices[f_name_lower]['service_name'] = s_name
                           prices[f_name_lower]['exact_service_match'] = is_new_exact_match
                           service_name_matches.add(s_name)
                           if is_new_exact_match and not exact_service_name_found:
                               exact_service_name_found = s_name

        service_display_name = exact_service_name_found if exact_service_name_found else (sorted(list(service_name_matches))[0] if service_name_matches else self.service_name)

        response_parts = [f"Сравнение цен на услугу '{service_display_name}':"]
        valid_prices = {}
        for f_lower, f_orig in filials_to_compare_orig_case.items():
            data = prices.get(f_lower)
            if not data: continue

            filial = data['name']
            price_info = data['price']
            service_name_in_filial = data.get('service_name', service_display_name)

            if isinstance(price_info, (int, float)):
                name_clarification = ""
                if service_name_in_filial.lower() != service_display_name.lower():
                     name_clarification = f" (для услуги: '{service_name_in_filial}')"
                elif not exact_service_name_found:
                     name_clarification = f" (для услуги: '{service_name_in_filial}')"

                response_parts.append(f"- {filial}: {price_info} руб.{name_clarification}")
                valid_prices[filial] = price_info
            elif price_info == "Не найдена":
                response_parts.append(f"- {filial}: Услуга '{service_display_name}' не найдена.")

        if len(valid_prices) >= 2:
             min_price = min(valid_prices.values())
             cheapest_filials = [f for f, p in valid_prices.items() if p == min_price]
             response_parts.append(f"\nСамая низкая цена ({min_price} руб.) в филиале(ах): {', '.join(cheapest_filials)}.")
        elif len(valid_prices) == 1:
             response_parts.append("\nНедостаточно данных для сравнения (цена найдена только в одном филиале).")
        elif not service_name_matches:
             response_parts.append(f"\nУслуга, похожая на '{self.service_name}', не найдена ни в одном из указанных филиалов.")
        else:
             response_parts.append("\nНе удалось найти цены для сравнения в указанных филиалах (возможно, цена не указана).")

        return "\n".join(response_parts)

class Agent:
    def __init__(self, model_uri: str, assistant_name: str, instruction=None, search_index=None, tools_models=None):
        self.thread = None
        self.assistant = None
        self.model_uri = model_uri
        self.assistant_name = assistant_name
        self.search_index = search_index
        self.tools_models = tools_models if tools_models else []
        self.function_handlers = {f.__name__: f for f in self.tools_models}
        self._uploaded_files_for_cleanup = []
        self._index_for_cleanup = None
        self._assistant_for_cleanup = None

        try:
            self._create_assistant(instruction)
        except Exception as e:
             logging.critical(f"Критическая ошибка при создании объекта Agent: {e}", exc_info=True)
             raise

    def _create_assistant(self, instruction):
        tools_for_sdk = []
        if self.search_index:
            if hasattr(self.search_index, 'id'):
                tools_for_sdk.append(sdk.tools.search_index(self.search_index))
                self._index_for_cleanup = self.search_index
                logging.info(f"Добавлен инструмент RAG Search Index: {self.search_index.id}")
            else:
                logging.error("Ошибка: search_index не является валидным объектом индекса.")

        if self.tools_models:
            for tool_model in self.tools_models:
                tools_for_sdk.append(sdk.tools.function(tool_model))
                logging.info(f"Добавлен инструмент Function Calling: {tool_model.__name__}")

        try:
            logging.info(f"Поиск существующих ассистентов с именем '{self.assistant_name}'...")
            all_assistants = sdk.assistants.list()
            deleted_count = 0
            for assist in all_assistants:
                if hasattr(assist, 'name') and assist.name == self.assistant_name:
                    logging.warning(f"Найден существующий ассистент с именем '{self.assistant_name}' (id={assist.id}). Удаляем его...")
                    try:
                        assist.delete()
                        deleted_count += 1
                    except Exception as del_e:
                         logging.error(f"Не удалось удалить существующего ассистента {assist.id}: {del_e}")
            if deleted_count > 0:
                 logging.info(f"Удалено {deleted_count} существующих ассистентов с именем '{self.assistant_name}'.")

            self.assistant = sdk.assistants.create(
                model=self.model_uri,
                name=self.assistant_name,
                instruction=instruction if instruction else "Ты - полезный ассистент.",
                tools=tools_for_sdk if tools_for_sdk else None,
                ttl_days=1,
                expiration_policy="since_last_active"
            )
            self._assistant_for_cleanup = self.assistant
            logging.info(f"Ассистент '{self.assistant_name}' создан: id={self.assistant.id}. Использует модель: {self.model_uri}")
        except Exception as e:
            logging.error(f"Не удалось создать ассистента Yandex Cloud '{self.assistant_name}': {e}", exc_info=True)
            raise

    def get_thread(self, thread=None):
        if thread is not None:
            return thread
        if self.thread is None:
            try:
                thread_name = f"ClinicChat_{int(time.time())}"
                self.thread = sdk.threads.create(name=thread_name, ttl_days=1, expiration_policy="static")
                logging.info(f"Создан новый поток: id={self.thread.id}, name={thread_name}")
            except Exception as e:
                 logging.error(f"Не удалось создать поток Yandex Cloud: {e}", exc_info=True)
                 return None
        return self.thread

    def __call__(self, message, thread=None):
        if self.assistant is None:
             logging.error("Вызов агента невозможен: ассистент не был инициализирован.")
             return "Критическая ошибка: Ассистент не инициализирован."

        current_thread = self.get_thread(thread)
        if current_thread is None:
             logging.error("Вызов агента невозможен: не удалось получить или создать поток.")
             return "Критическая ошибка: Не удалось создать поток для диалога."

        try:
            current_thread.write(message)
            run = self.assistant.run(current_thread)
            logging.info(f"[{current_thread.id}] Запущен run: {run.id}")
            res = run.wait(timeout=120)

            while hasattr(res,'tool_calls') and res.tool_calls:
                logging.info(f"[{current_thread.id}/{run.id}] Требуется вызов функций: {len(res.tool_calls)} шт.")
                tool_results = []
                for tool_call in res.tool_calls:
                    func_name = tool_call.function.name
                    func_args = tool_call.function.arguments if hasattr(tool_call.function, 'arguments') else {}
                    logging.info(f"[{current_thread.id}/{run.id}] Вызов: {func_name}({func_args})")

                    if func_name in self.function_handlers:
                        handler_class = self.function_handlers[func_name]
                        try:
                            try:
                                handler_instance = handler_class(**func_args) if func_args else handler_class()
                            except Exception as pydantic_error:
                                logging.error(f"[{current_thread.id}/{run.id}] Ошибка валидации аргументов для функции {func_name}: {pydantic_error}", exc_info=False)
                                tool_results.append({"name": func_name, "content": f"Ошибка: Некорректные параметры для функции {func_name}. {pydantic_error}"})
                                continue

                            output = handler_instance.process(current_thread)
                            log_output = (str(output)[:200] + '...' if len(str(output)) > 200 else str(output)).replace('\n', ' ')
                            logging.debug(f"[{current_thread.id}/{run.id}] Результат {func_name}: {log_output}")
                            tool_results.append({"name": func_name, "content": str(output)})
                        except Exception as e:
                            logging.error(f"[{current_thread.id}/{run.id}] Ошибка выполнения логики функции {func_name}: {e}", exc_info=True)
                            tool_results.append({"name": func_name, "content": f"Внутренняя ошибка при обработке запроса функцией {func_name}."})
                    else:
                        logging.warning(f"[{current_thread.id}/{run.id}] Не найден обработчик для функции: {func_name}")
                        tool_results.append({"name": func_name, "content": "Ошибка: неизвестная функция."})

                if not tool_results:
                     logging.warning(f"[{current_thread.id}/{run.id}] Нет результатов для отправки после обработки tool_calls.")
                     return "Произошла ошибка при обработке запрошенных действий."

                logging.info(f"[{current_thread.id}/{run.id}] Отправка результатов функций...")
                run.submit_tool_results(tool_results)
                res = run.wait(timeout=120)

            has_error = hasattr(res, 'error') and res.error is not None
            tool_calls_exist = hasattr(res, 'tool_calls') and res.tool_calls

            if not tool_calls_exist and not has_error:
                final_text = getattr(getattr(res, 'message', None), 'text', None)
                if final_text is None:
                     final_text = getattr(res, 'text', "Не удалось получить ответ от ассистента.")

                logging.info(f"[{current_thread.id}/{run.id}] Ассистент ответил (статус COMPLETED).")
                logging.info(f"[{current_thread.id}/{run.id}] Извлеченный текст ответа (len={len(final_text)}): '{final_text}'")
                log_final_text = (final_text[:200] + '...' if len(final_text) > 200 else final_text).replace('\n', ' ')
                logging.debug(f"[{current_thread.id}/{run.id}] Ассистент: {log_final_text}")
                return final_text
            elif has_error:
                 error_msg = f"Ошибка выполнения run: {res.error}"
                 logging.error(f"[{current_thread.id}/{run.id}] {error_msg}")
                 details = getattr(res.error, 'details', str(res.error))
                 return f"Произошла ошибка во время обработки вашего запроса: {details}. Попробуйте еще раз."
            else:
                 status_val = getattr(res, 'status', 'Неизвестный статус')
                 error_msg = f"Неожиданное состояние: status={status_val}, tool_calls={res.tool_calls if tool_calls_exist else 'None'}"
                 logging.error(f"[{current_thread.id}/{run.id}] {error_msg}")
                 return f"Произошла ошибка во время обработки вашего запроса. Попробуйте еще раз. (Статус: {status_val})"

        except TimeoutError:
             logging.error(f"[{current_thread.id}] Превышен таймаут ожидания ответа от ассистента.", exc_info=True)
             return "Ассистент долго отвечает. Пожалуйста, попробуйте повторить ваш запрос позже."
        except Exception as e:
            run_id_str = f"/{run.id}" if 'run' in locals() and hasattr(run, 'id') else ""
            logging.error(f"[{current_thread.id}{run_id_str}] Неожиданная ошибка во время обработки запроса: {e}", exc_info=True)
            return "Произошла внутренняя ошибка сервера или сети. Пожалуйста, попробуйте повторить ваш запрос позже."

    def add_uploaded_files(self, files: List[Any]):
        if files:
            self._uploaded_files_for_cleanup.extend(files)

    def cleanup(self, delete_assistant=True):
        logging.info("Начало очистки ресурсов...")
        if self.thread:
            try:
                self.thread.delete()
                logging.info(f"Поток {self.thread.id} удален.")
            except Exception as e: logging.warning(f"Не удалось удалить поток {self.thread.id}: {e}")
            self.thread = None

        if self._index_for_cleanup:
            try:
                self._index_for_cleanup.delete()
                logging.info(f"RAG индекс {self._index_for_cleanup.id} удален.")
            except Exception as e: logging.warning(f"Не удалось удалить RAG индекс {self._index_for_cleanup.id}: {e}")
            self._index_for_cleanup = None

        if self._uploaded_files_for_cleanup:
            logging.info(f"Удаление {len(self._uploaded_files_for_cleanup)} RAG файлов...")
            unique_file_ids = {f.id for f in self._uploaded_files_for_cleanup}
            for file_id in tqdm(unique_file_ids, desc="Удаление RAG файлов"):
                 try:
                     file_obj = sdk.files.get(file_id)
                     file_obj.delete()
                 except Exception as e:
                     logging.warning(f"Не удалось удалить файл {file_id}: {e}")
            self._uploaded_files_for_cleanup = []
            logging.info("RAG файлы удалены.")

        assistant_to_delete = self._assistant_for_cleanup if hasattr(self, '_assistant_for_cleanup') else self.assistant
        if delete_assistant and assistant_to_delete:
            try:
                assistant_to_delete.delete()
                logging.info(f"Ассистент {assistant_to_delete.id} удален.")
            except Exception as e: logging.warning(f"Не удалось удалить ассистента {assistant_to_delete.id}: {e}")
            self.assistant = None
            self._assistant_for_cleanup = None
        logging.info("Очистка завершена.")


def initialize_clinic_assistant():
    """
    Инициализирует и возвращает экземпляр агента-ассистента клиники
    """
    rag_index = None
    uploaded_rag_files = []
    existing_index_found = False

    try:
        logging.info(f"Поиск существующего RAG индекса с именем '{RAG_INDEX_NAME}'...")
        all_indexes = sdk.search_indexes.list()
        found_index = None
        for index in all_indexes:
            if hasattr(index, 'name') and index.name == RAG_INDEX_NAME:
                found_index = index
                break

        if found_index:
            rag_index = found_index
            existing_index_found = True
            logging.warning(f"Найден существующий RAG индекс '{RAG_INDEX_NAME}' (id={rag_index.id}). Переиспользуем его.")
        else:
            logging.info(f"Существующий индекс с именем '{RAG_INDEX_NAME}' не найден. Создаем новый.")
            rag_chunks = preprocess_json_for_rag(global_clinic_data)
            if rag_chunks:
                uploaded_rag_files = upload_rag_chunks(rag_chunks)
                if uploaded_rag_files:
                    rag_index = create_rag_index(uploaded_rag_files, RAG_INDEX_NAME)
                    if rag_index is None:
                         logging.error("Не удалось создать RAG индекс. RAG функциональность будет недоступна.")
                else:
                    logging.warning("Не удалось загрузить файлы для RAG, индекс не будет создан.")
            else:
                 logging.warning("Нет данных для создания RAG чанков, индекс не будет создан.")

        function_models = [
            FindEmployees,
            GetServicePrice,
            ListFilials,
            GetEmployeeServices,
            CheckServiceInFilial,
            CompareServicePriceInFilials
        ]

        system_prompt = f"""Ты - вежливый, **ОЧЕНЬ ВНИМАТЕЛЬНЫЙ** и информативный ассистент медицинской клиники "Med YU Med".
Твоя главная задача - помогать пользователям, отвечая на их вопросы об услугах, ценах, специалистах и филиалах клиники, используя предоставленные инструменты и историю диалога.

**КЛЮЧЕВЫЕ ПРАВИЛА ВЫБОРА ИНСТРУМЕНТА И РАБОТЫ С КОНТЕКСТОМ:**

1.  **СНАЧАЛА ОПРЕДЕЛИ ТИП ЗАПРОСА:**
    *   **ОБЩИЙ ВОПРОС / ЗАПРОС ОПИСАНИЯ (Что? Как? Зачем? Посоветуй... Расскажи о...):** Если пользователь спрашивает **общее описание** услуги (что это, как работает, какой эффект), просит **подробности** об опыте/специализации врача, или задает **открытый вопрос** ("что для омоложения?", "какие пилинги бывают?", "посоветуй от морщин"), **ПОИСК ДОП. ИНФОРМАЦИИ:** Ты **должен** использовать предоставленные тебе текстовые материалы (описания услуг и врачей) для поиска релевантной информации. **СИНТЕЗИРУЙ** свой ответ на основе найденных описаний. НЕ пытайся сразу вызывать функции для таких общих вопросов.
    *   **КОНКРЕТНЫЙ ЗАПРОС (Сколько? Где? Кто? Сравни...):** Если пользователь спрашивает **конкретную цену**, **наличие в филиале**, **список врачей/услуг по критерию**, **сравнение цен**, **ИСПОЛЬЗУЙ Function Calling**.

2.  **ИСПОЛЬЗОВАНИЕ ИСТОРИИ:**
    *   **ПЕРЕД КАЖДЫМ ОТВЕТОМ/ВЫЗОВОМ ФУНКЦИИ:** **ОБЯЗАТЕЛЬНО** проанализируй **ПОСЛЕДНИЕ 3-4 сообщения**. Ищи имена, услуги, филиалы. **ИСПОЛЬЗУЙ ЭТУ ИНФОРМАЦИЮ!**
    *   **ЗАПОМИНАЙ:** Имя пользователя, обсуждаемые темы (услуги, филиалы). Обращайся по имени. Не задавай вопросы по информации, которая уже есть в недавней истории.

3.  **ИНСТРУКЦИИ ПО FUNCTION CALLING (Только для КОНКРЕТНЫХ запросов!):**
    *   **Уточняй ТОЛЬКО если ОБЯЗАТЕЛЬНОГО параметра ДЕЙСТВИТЕЛЬНО нет в истории!**
    *   `FindEmployees`: Поиск СПИСКА сотрудников по критериям. **Используй эту функцию также, если пользователь спрашивает "какие услуги есть в филиале X?" (передай `filial_name=X` и не передавай `service_name`).**
    *   `GetServicePrice`: Цена ОДНОЙ КОНКРЕТНОЙ услуги. **`service_name` ОБЯЗАТЕЛЕН**. Ищи его в истории перед уточнением. **НЕ ВЫЗЫВАЙ для общих категорий типа "процедуры для лица".**
    *   `ListFilials`: ТОЛЬКО по явному запросу списка филиалов/адресов.
    *   `GetEmployeeServices`: Список услуг ОДНОГО КОНКРЕТНОГО врача. **`employee_name` ОБЯЗАТЕЛЕН**.
    *   `CheckServiceInFilial`: Наличие ОДНОЙ КОНКРЕТНОЙ услуги в ОДНОМ КОНКРЕТНОМ филиале. **`service_name` и `filial_name` ОБЯЗАТЕЛЬНЫ**. НЕ ВЫЗЫВАЙ без `filial_name`.
    *   `CompareServicePriceInFilials`: Сравнение цены ОДНОЙ КОНКРЕТНОЙ услуги в НЕСКОЛЬКИХ филиалах. **`service_name` и `filial_names` (>=2) ОБЯЗАТЕЛЬНЫ**.

**Общие правила поведения:**
-   **Точность:** НЕ ПРИДУМЫВАЙ. Используй результаты поиска информации или функции. Сообщай, если не найдено.
-   **Краткость:** Отвечай по существу. Длинные списки из функций сокращай и предлагай уточнить.
-   **Вежливость:** Будь корректным и дружелюбным.
-   **Приветствия/Прощания:** Поздоровайся ОДИН РАЗ. Не прощайся без запроса.
-   **Медицинские советы:** Отклоняй и предлагай консультацию.
-   **Последовательность:** Завершай текущую задачу пользователя.
"""

        clinic_agent = Agent(
            model_uri=MODEL_URI_FULL,
            assistant_name=ASSISTANT_NAME,
            instruction=system_prompt,
            search_index=rag_index,
            tools_models=function_models
        )
        
        if uploaded_rag_files:
            clinic_agent.add_uploaded_files(uploaded_rag_files)
        
        return clinic_agent, existing_index_found
        
    except Exception as e:
        logging.critical(f"Не удалось инициализировать Агента: {e}", exc_info=True)
        raise


__all__ = ['initialize_clinic_assistant', 'Agent']
