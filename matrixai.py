#matrixai.py

import os # Оставляем импорт как был запрошен

import json
import logging
from typing import List, Dict, Any, Optional, Sequence
from operator import itemgetter
import shutil
import importlib # Для проверки импорта clinic_functions

try:
    # --- LangChain и GigaChat импорты ---

    # Базовые компоненты LangChain
    from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
    from langchain_core.documents import Document
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
    from langchain_core.tools import tool
    from langchain_core.runnables.history import RunnableWithMessageHistory

    # Модели GigaChat (оставляем только для Embeddings)
    # from langchain_gigachat.chat_models import GigaChat # <-- ЗАКОММЕНТИРОВАНО
    from langchain_gigachat.embeddings import GigaChatEmbeddings

    # --- НОВЫЙ ИМПОРТ ---
    from langchain_deepseek import ChatDeepSeek
    # --- КОНЕЦ НОВОГО ИМПОРТА ---

    # Векторное хранилище ChromaDB (используем новый рекомендованный импорт)
    from langchain_chroma import Chroma

    # Хранилище истории чата (используем новый рекомендованный импорт)
    from langchain_community.chat_message_histories import ChatMessageHistory

    # Текстовый сплиттер
    from langchain.text_splitter import RecursiveCharacterTextSplitter # Старый импорт, но пока широко используется

    # Pydantic для схем инструментов
    from pydantic import BaseModel, Field

    # --- Импорт нашего модуля с функциями ---
    try:
        # Проверяем, существует ли файл перед импортом
        if not os.path.exists("clinic_functions.py"):
            logging.critical("Критическая ошибка: Файл 'clinic_functions.py' не найден.")
            exit()
        import clinic_functions
        # Проверяем наличие ключевой функции после импорта
        if not hasattr(clinic_functions, 'set_clinic_data') or not callable(clinic_functions.set_clinic_data):
             logging.critical("Критическая ошибка: Функция 'set_clinic_data' не найдена или не является функцией в 'clinic_functions.py'.")
             exit()
    except ImportError as e:
        logging.critical(f"Критическая ошибка: Не удалось импортировать 'clinic_functions'. Убедитесь, что файл существует и не содержит синтаксических ошибок. Ошибка: {e}", exc_info=True)
        exit()
    # Use code with caution. - УДАЛЕНО, так как это артефакт, а не код
except ImportError as e:
    # --- Блок except для основного try ---
    # ОБНОВЛЕНО СООБЩЕНИЕ ОБ ОШИБКЕ И КОМАНДА PIP
    logging.critical(f"Критическая ошибка: Необходимые библиотеки LangChain/GigaChat/DeepSeek не установлены. Ошибка: {e}", exc_info=True)
    logging.error("Пожалуйста, установите их:")
    logging.error("pip install langchain-gigachat langchain langchain-core langchain-community langchain-chroma pydantic langchain-deepseek -U -q")
    exit()

# --- Настройка логирования ---
# Устанавливаем уровень INFO по умолчанию
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(lineno)d - %(message)s')

# Можно установить DEBUG для подробной отладки LangChain:
# logging.getLogger("langchain_core").setLevel(logging.DEBUG)
# logging.getLogger("langchain_community").setLevel(logging.DEBUG)
# logging.getLogger("langchain_gigachat").setLevel(logging.DEBUG)
# logging.getLogger(__name__).setLevel(logging.DEBUG) # Логирование этого модуля (исправлено name на __name__)

# --- Загрузка учетных данных GigaChat (БЕЗОПАСНО!) ---
# ВАЖНО: В реальном приложении используйте переменные окружения или другие безопасные методы
GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS", "OTkyYTgyNGYtMjRlNC00MWYyLTg3M2UtYWRkYWVhM2QxNTM1OjA5YWRkODc0LWRjYWItNDI2OC04ZjdmLWE4ZmEwMDIxMThlYw==") # ОСТАВЛЕНО ДЛЯ ЭМБЕДДИНГОВ
if not GIGACHAT_CREDENTIALS:
    logging.critical("Критическая ошибка: Учетные данные GigaChat не найдены (нужны для эмбеддингов).")
    logging.error("Установите переменную окружения GIGACHAT_CREDENTIALS.")
    exit()

# --- НОВЫЙ БЛОК: Загрузка учетных данных DeepSeek ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-1aae129014ac42e3804329d6d44497ce") # ИСПОЛЬЗУЕМ ПРЕДОСТАВЛЕННЫЙ КЛЮЧ КАК DEFAULT
if not DEEPSEEK_API_KEY:
    logging.critical("Критическая ошибка: Ключ DeepSeek API не найден.")
    logging.error("Установите переменную окружения DEEPSEEK_API_KEY или проверьте значение по умолчанию в коде.")
    exit()
# --- КОНЕЦ НОВОГО БЛОКА ---

# --- Константы ---
JSON_DATA_PATH = "base/cleaned_data.json"
CHROMA_PERSIST_DIR = "chroma_db_clinic_giga"
# GIGA_CHAT_MODEL = "GigaChat-2-Max" # ЗАКОММЕНТИРОВАНО
GIGA_EMBEDDING_MODEL = "EmbeddingsGigaR" # Оставляем для эмбеддингов
GIGA_SCOPE = "GIGACHAT_API_PERS" # ИЛИ GIGACHAT_API_CORP - Оставляем для эмбеддингов

# --- НОВАЯ КОНСТАНТА (название модели DeepSeek) ---
DEEPSEEK_CHAT_MODEL = "deepseek-chat"

# --- Инициализация DeepSeek Chat и GigaChat Эмбеддингов ---
try:
    # verify_ssl_certs=False - НЕ ИСПОЛЬЗУЙТЕ В ПРОДАКШЕНЕ без понимания рисков

    # --- ЗАМЕНА GigaChat НА DeepSeek ---
    # chat_model = GigaChat(
    #     credentials=GIGACHAT_CREDENTIALS, model=GIGA_CHAT_MODEL,
    #     verify_ssl_certs=False, scope=GIGA_SCOPE, timeout=60 # Увеличим таймаут
    # ) # <-- ЗАКОММЕНТИРОВАНО

    chat_model = ChatDeepSeek(
        model=DEEPSEEK_CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY, # Используем переменную
        temperature=0, # Настроим температуру
        # Можно добавить другие параметры ChatDeepSeek при необходимости
        # max_tokens=4096,
        # timeout=60,
        # max_retries=2,
    )
    # --- КОНЕЦ ЗАМЕНЫ ---

    # --- Эмбеддинги GigaChat ОСТАЮТСЯ БЕЗ ИЗМЕНЕНИЙ ---
    embeddings = GigaChatEmbeddings(
        credentials=GIGACHAT_CREDENTIALS, model=GIGA_EMBEDDING_MODEL,
        verify_ssl_certs=False, scope=GIGA_SCOPE, timeout=60 # Увеличим таймаут
    )
    # --- КОНЕЦ БЛОКА ЭМБЕДДИНГОВ ---

    # ОБНОВЛЕН ЛОГ
    logging.info(f"Chat модель: {DEEPSEEK_CHAT_MODEL}, Эмбеддинги: {GIGA_EMBEDDING_MODEL}")
except Exception as e:
    # ОБНОВЛЕН ТЕКСТ ОШИБКИ
    logging.critical(f"Ошибка инициализации моделей DeepSeek Chat / GigaChat Embeddings: {e}", exc_info=True)
    exit()

# --- Загрузка и передача данных в модуль функций ---
global_clinic_data = []
try:
    base_dir = os.path.dirname(JSON_DATA_PATH)
    if base_dir and not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)
        logging.info(f"Создана директория '{base_dir}'.")
    if not os.path.exists(JSON_DATA_PATH):
        with open(JSON_DATA_PATH, 'w', encoding='utf-8') as f: json.dump([], f)
        logging.warning(f"Файл '{JSON_DATA_PATH}' не найден, создан пустой файл.")

    with open(JSON_DATA_PATH, 'r', encoding='utf-8') as f:
        global_clinic_data = json.load(f)
    if not isinstance(global_clinic_data, list): raise ValueError("JSON должен быть списком объектов")
    if not global_clinic_data: logging.warning(f"Файл данных '{JSON_DATA_PATH}' пуст.")
    else: logging.info(f"Загружено {len(global_clinic_data)} записей из {JSON_DATA_PATH}.")

    # Передаем данные в модуль clinic_functions
    clinic_functions.set_clinic_data(global_clinic_data)
    # Use code with caution. - УДАЛЕНО
except Exception as e:
    logging.error(f"Ошибка загрузки/передачи данных клиники: {e}", exc_info=True)
    # Продолжаем работу, но функции не будут иметь данных

# --- Подготовка RAG (Векторная База и Ретривер) ---
# Копируем функцию preprocess_for_rag_v2 сюда для самодостаточности файла
def preprocess_for_rag_v2(data: List[Dict[str, Any]]) -> List[Document]:
    """Готовит документы для RAG на основе уникальных услуг и сотрудников."""
    services_data = {}
    employees_data = {}
    for item in data:
        srv_id = item.get("serviceId")
        if srv_id:
            if srv_id not in services_data:
                services_data[srv_id] = {
                    "name": item.get("serviceName", "Без названия"),
                    "category": item.get("categoryName", "Без категории"),
                    "description": (item.get("serviceDescription") or "").strip()
                }
            # Обновляем описание, если новое длиннее и осмысленнее
            new_desc = (item.get("serviceDescription") or "").strip()
            if new_desc and new_desc.lower() not in ('', 'null', 'нет описания') and len(new_desc) > len(services_data[srv_id]["description"]):
                services_data[srv_id]["description"] = new_desc

        emp_id = item.get("employeeId") # Исправлена ИНДЕНТАЦИЯ этой строки
        if emp_id:                     # Исправлена ИНДЕНТАЦИЯ этой строки
            if emp_id not in employees_data:
                employees_data[emp_id] = {
                    "name": item.get("employeeFullName", "Имя неизвестно"),
                    "description": (item.get("employeeDescription") or "").strip()
                }
            # Обновляем описание, если новое длиннее и осмысленнее
            new_desc = (item.get("employeeDescription") or "").strip()
            meaningful_desc = new_desc and new_desc.lower() not in ('', 'null', 'нет описания', 'опытный специалист')
            if meaningful_desc and len(new_desc) > len(employees_data[emp_id]["description"]):
                employees_data[emp_id]["description"] = new_desc

    documents = []
    # Документы по услугам
    for srv_id, info in services_data.items():
        # Добавляем только если есть осмысленное описание
        if not info["name"] or not info["description"] or info["description"].lower() in ('', 'null', 'нет описания'): continue
        text_content = f"Услуга: {info['name']}\nКатегория: {info['category']}\nОписание: {info['description']}"
        metadata = {"id": f"srv_{srv_id}", "type": "service", "name": info['name'], "category": info['category']}
        documents.append(Document(page_content=text_content, metadata=metadata))

    # Документы по сотрудникам
    for emp_id, info in employees_data.items():
         # Добавляем только если есть осмысленное описание
         if not info["name"] or not info["description"] or info["description"].lower() in ('', 'null', 'нет описания', 'опытный специалист'): continue
         text_content = f"Сотрудник: {info['name']}\nОписание: {info['description']}" # Исправлена ИНДЕНТАЦИЯ
         metadata = {"id": f"emp_{emp_id}", "type": "employee", "name": info['name']} # Исправлена ИНДЕНТАЦИЯ
         documents.append(Document(page_content=text_content, metadata=metadata)) # Исправлена ИНДЕНТАЦИЯ

    logging.info(f"Подготовлено {len(documents)} документов для RAG (услуги: {len(services_data)}, сотрудники: {len(employees_data)}).")
    return documents
# Use code with caution. - УДАЛЕНО

prepared_documents = []
if global_clinic_data:
    prepared_documents = preprocess_for_rag_v2(global_clinic_data)

vectorstore = None

# Проверяем, нужно ли пересоздавать базу (например, если данные изменились или база повреждена)
# Для простоты, будем пересоздавать только если папки нет
if os.path.exists(CHROMA_PERSIST_DIR):
    try:
        logging.info(f"Загрузка существующей векторной базы из '{CHROMA_PERSIST_DIR}'...")
        # Используем новый импорт Chroma
        vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
        logging.info("Векторная база успешно загружена.")
    except Exception as e:
        logging.warning(f"Не удалось загрузить существующую базу Chroma: {e}. База будет создана заново.", exc_info=True)
        try:
            shutil.rmtree(CHROMA_PERSIST_DIR) # Попытка очистить папку перед пересозданием
        except OSError as rm_err:
            logging.error(f"Не удалось удалить старую папку Chroma '{CHROMA_PERSIST_DIR}': {rm_err}")
        vectorstore = None # Сбрасываем, чтобы создать заново

# Создаем базу, если она не была загружена
if vectorstore is None:
    if prepared_documents:
        logging.info(f"Создание новой векторной базы в '{CHROMA_PERSIST_DIR}'...")
        try:
            # Используем text_splitter для очень больших описаний
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, # Можно увеличить до ~4000 для EmbeddingsGigaR, но 1000 надежнее
                chunk_overlap=100,
                length_function=len,
            )
            docs_for_chroma = text_splitter.split_documents(prepared_documents)
            logging.info(f"Документы разделены на {len(docs_for_chroma)} чанков для Chroma.")

            vectorstore = Chroma.from_documents(
                documents=docs_for_chroma,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR # Сохраняем на диск
            )
            logging.info("Новая векторная база данных Chroma создана и сохранена.")
        except Exception as e:
            logging.critical(f"Критическая ошибка создания векторной базы данных Chroma: {e}", exc_info=True)
            vectorstore = None # Не удалось создать
    else:
        logging.warning("Нет данных для создания RAG базы.")
        vectorstore = None
# Use code with caution. - УДАЛЕНО

# Создаем ретривер (или пустышку, если база не создана/не загружена)
if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Искать топ-3 релевантных документа
    logging.info("Ретривер настроен.")
else:
    # Создаем ретривер-пустышку
    def empty_retriever_func(query: str, *, config: Optional[RunnableConfig] = None, **kwargs) -> List[Document]:
        logging.warning("Ретривер-пустышка вызван, так как векторная база недоступна.")
        return []
    retriever = RunnableLambda(empty_retriever_func, name="EmptyRetriever")
    logging.warning("Создан пустой ретривер, RAG не будет использовать внешнюю базу.")

# --- Адаптация Функций под LangChain Tools (@tool) ---
# Определяем Pydantic схемы для входов инструментов
# (Можно вынести их в clinic_functions.py рядом с классами для лучшей организации)
class FindEmployeesArgs(BaseModel):
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название услуги, которую должен выполнять сотрудник")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала, где должен работать сотрудник")

@tool("find_employees", args_schema=FindEmployeesArgs)
def find_employees_tool(employee_name: Optional[str] = None, service_name: Optional[str] = None, filial_name: Optional[str] = None) -> str:
    """
    Ищет сотрудников клиники по ФИО, выполняемой услуге или филиалу.
    Вызывай эту функцию, чтобы найти список врачей или других сотрудников по критериям.
    """
    handler = clinic_functions.FindEmployees(
        employee_name=employee_name,
        service_name=service_name,
        filial_name=filial_name
    )
    return handler.process()

class GetServicePriceArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги (например, 'Soprano Пальцы для женщин')")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала (например, 'Москва-сити'), если нужно уточнить цену в конкретном месте")

@tool("get_service_price", args_schema=GetServicePriceArgs)
def get_service_price_tool(service_name: str, filial_name: Optional[str] = None) -> str:
    """
    Возвращает цену на КОНКРЕТНУЮ услугу клиники в указанном филиале (или любом, если филиал не указан).
    Вызывай эту функцию ТОЛЬКО когда пользователь спрашивает цену КОНКРЕТНОЙ названной услуги.
    Параметр service_name обязателен. Попробуй извлечь его из истории диалога, прежде чем спрашивать у пользователя.
    Не вызывай для общих категорий услуг (например, "массаж").
    """
    handler = clinic_functions.GetServicePrice(service_name=service_name, filial_name=filial_name)
    return handler.process()

@tool("list_filials")
def list_filials_tool() -> str:
    """Возвращает список всех доступных филиалов клиники."""
    handler = clinic_functions.ListFilials()
    return handler.process()

class GetEmployeeServicesArgs(BaseModel):
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")

@tool("get_employee_services", args_schema=GetEmployeeServicesArgs)
def get_employee_services_tool(employee_name: str) -> str:
    """
    Возвращает список услуг, которые выполняет КОНКРЕТНЫЙ сотрудник.
    Параметр employee_name обязателен. Ищи имя в истории диалога.
    """
    handler = clinic_functions.GetEmployeeServices(employee_name=employee_name)
    return handler.process()

class CheckServiceInFilialArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")

@tool("check_service_in_filial", args_schema=CheckServiceInFilialArgs)
def check_service_in_filial_tool(service_name: str, filial_name: str) -> str:
    """
    Проверяет, доступна ли КОНКРЕТНАЯ услуга в КОНКРЕТНОМ филиале.
    Вызывай эту функцию, только если пользователь указал и услугу, и филиал.
    """
    handler = clinic_functions.CheckServiceInFilial(service_name=service_name, filial_name=filial_name)
    return handler.process()

class CompareServicePriceInFilialsArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(min_length=2, description="Список из ДВУХ или БОЛЕЕ ТОЧНЫХ названий филиалов для сравнения")

@tool("compare_service_price_in_filials", args_schema=CompareServicePriceInFilialsArgs)
def compare_service_price_in_filials_tool(service_name: str, filial_names: List[str]) -> str:
    """
    Сравнивает цену на КОНКРЕТНУЮ услугу в НЕСКОЛЬКИХ (минимум два) указанных филиалах.
    """
    handler = clinic_functions.CompareServicePriceInFilials(service_name=service_name, filial_names=filial_names)
    return handler.process()

class FindServiceLocationsArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")

@tool("find_service_locations", args_schema=FindServiceLocationsArgs)
def find_service_locations_tool(service_name: str) -> str:
    """
    Ищет все филиалы, в которых доступна указанная КОНКРЕТНАЯ услуга.
    Используй эту функцию, когда пользователь спрашивает 'где делают', 'в каких филиалах есть',
    или просто 'где?' сразу после обсуждения КОНКРЕТНОЙ УСЛУГИ.
    Параметр service_name обязателен, извлеки его из истории.
    Не используй check_service_in_filial, если филиал не указан явно.
    """
    handler = clinic_functions.FindServiceLocations(service_name=service_name)
    return handler.process()

# === НОВЫЕ ОБЕРТКИ ДЛЯ НЕДОСТАЮЩИХ ИНСТРУМЕНТОВ ===

class FindSpecialistsByServiceOrCategoryAndFilialArgs(BaseModel):
    query_term: str = Field(description="Точное название услуги ИЛИ категории для поиска специалистов")
    filial_name: str = Field(description="Точное название филиала, в котором искать специалистов")

@tool("find_specialists_by_service_or_category_and_filial", args_schema=FindSpecialistsByServiceOrCategoryAndFilialArgs)
def find_specialists_by_service_or_category_and_filial_tool(query_term: str, filial_name: str) -> str:
    """
    Ищет СПЕЦИАЛИСТОВ (врачей, сотрудников), выполняющих конкретную УСЛУГУ или относящихся к конкретной КАТЕГОРИИ в указанном ФИЛИАЛЕ.
    Используй, когда спрашивают 'кто делает [услугу/категорию] в [филиале]?'.
    Оба параметра (query_term и filial_name) обязательны. Ищи их в запросе и истории.
    """
    handler = clinic_functions.FindSpecialistsByServiceOrCategoryAndFilial(query_term=query_term.lower(), filial_name=filial_name.lower()) # Приводим к нижнему регистру для консистентности поиска
    return handler.process()

class ListServicesInCategoryArgs(BaseModel):
    category_name: str = Field(description="Точное название категории для поиска услуг")

@tool("list_services_in_category", args_schema=ListServicesInCategoryArgs)
def list_services_in_category_tool(category_name: str) -> str:
    """
    Возвращает список КОНКРЕТНЫХ услуг, входящих в указанную КАТЕГОРИЮ.
    Используй, когда пользователь спрашивает 'какие услуги входят в категорию [название]?' или 'что есть в [названии категории]?'.
    Параметр category_name обязателен.
    """
    handler = clinic_functions.ListServicesInCategory(category_name=category_name)
    return handler.process()

class ListServicesInFilialArgs(BaseModel):
    filial_name: str = Field(description="Точное название филиала для поиска услуг")

@tool("list_services_in_filial", args_schema=ListServicesInFilialArgs)
def list_services_in_filial_tool(filial_name: str) -> str:
    """
    Возвращает ПОЛНЫЙ список УНИКАЛЬНЫХ услуг, доступных в КОНКРЕТНОМ филиале.
    Используй, когда пользователь спрашивает 'что делают в [филиале]?' или 'список услуг в [филиале]?'.
    Параметр filial_name обязателен.
    """
    handler = clinic_functions.ListServicesInFilial(filial_name=filial_name)
    return handler.process()

class FindServicesInPriceRangeArgs(BaseModel):
    min_price: float = Field(description="Минимальная цена услуги (число)")
    max_price: float = Field(description="Максимальная цена услуги (число)")
    category_name: Optional[str] = Field(default=None, description="Опционально: точное название категории для фильтрации")
    filial_name: Optional[str] = Field(default=None, description="Опционально: точное название филиала для фильтрации")

@tool("find_services_in_price_range", args_schema=FindServicesInPriceRangeArgs)
def find_services_in_price_range_tool(min_price: float, max_price: float, category_name: Optional[str] = None, filial_name: Optional[str] = None) -> str:
    """
    Ищет услуги клиники в ЗАДАННОМ ЦЕНОВОМ ДИАПАЗОНЕ (от min_price до max_price).
    Можно дополнительно отфильтровать по категории и/или филиалу.
    Используй, когда пользователь спрашивает об услугах 'в районе X рублей', 'от X до Y', 'дешевле X' и т.д.
    Извлеки min_price и max_price из запроса. Если указана только одна граница (например, 'дешевле 1000'), установи другую границу разумно (0 для min_price, большое число для max_price).
    """
    handler = clinic_functions.FindServicesInPriceRange(
        min_price=min_price,
        max_price=max_price,
        category_name=category_name,
        filial_name=filial_name
    )
    return handler.process()

@tool("list_all_categories")
def list_all_categories_tool() -> str:
    """
    Возвращает список ВСЕХ доступных категорий услуг в клинике.
    Используй, когда пользователь спрашивает 'какие есть категории услуг?'.
    Не принимает аргументов.
    """
    handler = clinic_functions.ListAllCategories()
    return handler.process()

# === КОНЕЦ НОВЫХ ОБЕРТОК ===

# Собираем все инструменты в список
tools = [
    find_employees_tool,
    get_service_price_tool,
    list_filials_tool,
    get_employee_services_tool,
    check_service_in_filial_tool,
    compare_service_price_in_filials_tool,
    find_service_locations_tool,
    # --- Добавляем новые инструменты ---
    find_specialists_by_service_or_category_and_filial_tool,
    list_services_in_category_tool,
    list_services_in_filial_tool,
    find_services_in_price_range_tool,
    list_all_categories_tool,
    # --- Конец добавленных инструментов ---
]
logging.info(f"Загружено {len(tools)} инструментов (функций).")

# --- Привязываем инструменты к модели ---
llm_with_tools = chat_model.bind_tools(tools)
logging.info("Инструменты привязаны к модели GigaChat.")

# --- Обновление Системного Промпта ---
SYSTEM_PROMPT = """Ты - вежливый, ОЧЕНЬ ВНИМАТЕЛЬНЫЙ и информативный ИИ-ассистент медицинской клиники "Med YU Med".
Твоя главная задача - помогать пользователям, отвечая на их вопросы об услугах, ценах, специалистах и филиалах клиники, И ПОДДЕРЖИВАТЬ ЕСТЕСТВЕННЫЙ ДИАЛОГ.

КЛЮЧЕВЫЕ ПРАВИЛА РАБОТЫ:

АНАЛИЗ ИСТОРИИ И ВЫБОР ДЕЙСТВИЯ:

ПЕРЕД ЛЮБЫМ ОТВЕТОМ внимательно проанализируй ПОЛНУЮ ИСТОРИЮ ДИАЛОГА (chat_history). Ищи имена (включая имя пользователя, если он представился), названия услуг, врачей, филиалы, цены.

ИСПОЛЬЗУЙ КОНТЕКСТ ИСТОРИИ! Не переспрашивай то, что уже обсуждалось.

ЗАПОМИНАЙ ИМЯ ПОЛЬЗОВАТЕЛЯ: Если пользователь говорит "меня зовут [Имя]", запомни это имя. Если он позже спросит "как меня зовут?", ответь, используя запомненное имя из истории. НЕ ИСПОЛЬЗУЙ find_employees для поиска имени пользователя.

ВЫБОР МЕЖДУ RAG, FUNCTION CALLING, ПАМЯТЬЮ ДИАЛОГА ИЛИ ПРЯМЫМ ОТВЕТОМ:

ПАМЯТЬ ДИАЛОГА: Используй для ответов на вопросы, связанные с предыдущим контекстом (местоимения "он/она/это", короткие вопросы "где?", "цена?", "кто?"), и для вопросов о самом пользователе (например, "как меня зовут?", если он представлялся).

RAG (Поиск по базе знаний): Используй ТОЛЬКО для ОБЩИХ вопросов и ЗАПРОСОВ ОПИСАНИЯ услуг или врачей ("Что такое X?", "Расскажи про Y"). Я предоставлю контекст. Синтезируй ответ на его основе. НЕ ВЫЗЫВАЙ функции для описаний.

FUNCTION CALLING (Вызов Инструментов): Используй ТОЛЬКО для запросов КОНКРЕТНЫХ ДАННЫХ КЛИНИКИ: цены, списки врачей/услуг/филиалов, проверка наличия, сравнение цен. Используй правильный инструмент для каждой задачи (см. описания инструментов).

ПРЯМОЙ ОТВЕТ: Для приветствий, прощаний, простых уточнений или вопросов не по теме клиники.

ПРАВИЛА FUNCTION CALLING:

Точность Параметров: Извлекай параметры МАКСИМАЛЬНО ТОЧНО из запроса и ИСТОРИИ ДИАЛОГА.

Не Выдумывай Параметры: Если обязательного параметра нет ни в запросе, ни в недавней истории, НЕ ВЫЗЫВАЙ функцию, а вежливо попроси пользователя уточнить.

ОБРАБОТКА НЕУДАЧНЫХ ВЫЗОВОВ: Если инструмент (функция) вернул ошибку или сообщил, что данные 'не найдены', НЕ ПЫТАЙСЯ вызвать тот же инструмент с теми же аргументами снова. Проанализируй ответ инструмента. Сообщи пользователю, что данные не найдены, или предложи альтернативу (уточнить запрос, поискать описание через RAG).

Интерпретация Результатов: Представляй результаты функций в понятной, человеческой форме.

ОБЩИЕ ПРАВИЛА:

Точность: НЕ ПРИДУМЫВАЙ.

Краткость и Ясность.

Вежливость.

Медицинские Советы: НЕ ДАВАЙ.

ВАЖНО: Всегда сначала анализируй историю и цель пользователя. Реши, нужен ли ответ из памяти, RAG, вызов функции или простой ответ. Действуй соответственно."""

# --- Основная цепочка с RAG и Function Calling ---
# Вспомогательные функции RAG
def format_docs(docs: List[Document]) -> str:
    """Форматирует найденные RAG документы для добавления в промпт."""
    if not docs:
        return "Дополнительная информация из базы знаний не найдена."
    formatted_docs = []
    for i, doc in enumerate(docs):
        source_info = doc.metadata.get('id', 'Неизвестный источник')
        doc_type = doc.metadata.get('type', 'данные')
        doc_name = doc.metadata.get('name', '')
        display_source = f"{doc_type} ('{doc_name}')" if doc_name else source_info
        formatted_docs.append(f"Источник {i+1} ({display_source}):\n{doc.page_content}")
    return "\n\n".join(formatted_docs)

def add_instruction_to_query(query: str) -> str:
    """Добавляет инструкцию к запросу для поиска в RAG с EmbeddingsGigaR."""
    # Инструкция подбирается экспериментально, эта - один из вариантов
    instruction = "Найди информацию об услугах, ценах, врачах или филиалах клиники по следующему запросу: \nвопрос: "
    # Используем repr() для корректной обработки переносов строк внутри f-строки
    # ИЛИ просто f-string, если уверены, что \n не сломает downstream JSON (обычно GigaChat нормально обработает)
    return f"{instruction}{query}" # Убрал repr, если он не нужен

# Цепочка для получения RAG-контекста
# Важно: она будет вызвана даже если ответ можно дать из истории или нужен только инструмент
# Оптимизация: добавить роутер, который решает, нужен ли RAG
context_chain = (
    RunnablePassthrough.assign(
        # Берем оригинальный input для запроса в ретривер
        query_for_retriever=itemgetter("input") | RunnableLambda(add_instruction_to_query, name="AddInstruction")
    ).assign(
        # Выполняем поиск
        docs=itemgetter("query_for_retriever") | retriever
    ).assign(
        # Форматируем результат
        context=itemgetter("docs") | RunnableLambda(format_docs, name="FormatDocs")
    )
    # Возвращаем только отформатированный контекст
    | itemgetter("context")
)

# --- Управление Историей Чата ---
chat_memory = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Возвращает историю для сессии или создает новую."""
    if session_id not in chat_memory:
        chat_memory[session_id] = ChatMessageHistory()
        logging.info(f"Создана новая история чата для сессии: {session_id}")
    return chat_memory[session_id]

# --- Функция для эмуляции цикла Агента (с исправлением истории) ---
def run_agent_like_chain(input_dict: Dict, config: RunnableConfig) -> str:
    """Эмулирует цикл вызова LLM с инструментами и RAG."""
    session_id = config["configurable"]["session_id"]
    user_input = input_dict["input"]
    logging.debug(f"[{session_id}] Вход в run_agent_like_chain: {user_input}")

 
    try:
        # Передаем вход пользователя для поиска
        rag_context = context_chain.invoke({"input": user_input}, config=config)
        logging.debug(f"[{session_id}] Получен RAG контекст: {rag_context[:200]}...")
    except Exception as e:
        logging.error(f"[{session_id}] Ошибка получения RAG контекста: {e}")
        rag_context = "Ошибка при получении информации из базы знаний."

    # 2. Формируем *начальные* сообщения для LLM в этом вызове
    # RAG контекст добавляем к сообщению пользователя, чтобы LLM его видел сразу
    context_block = f"\n\n[Информация из базы знаний для справки]:\n{rag_context}\n[/Информация из базы знаний]"
    # Получаем историю ДО текущего запроса
    current_history = get_session_history(session_id).messages
    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        *current_history,
        HumanMessage(content=user_input + context_block) # RAG добавляется к запросу
    ]

    MAX_TURNS = 5
    for turn in range(MAX_TURNS):
        logging.info(f"[{session_id}] Вызов LLM (Turn {turn + 1}/{MAX_TURNS}). Сообщений для LLM: {len(messages)}")

        try:
            ai_response: AIMessage = llm_with_tools.invoke(messages, config=config)
        except Exception as llm_error:
            logging.error(f"[{session_id}] Ошибка вызова LLM: {llm_error}", exc_info=True)
            return f"Произошла ошибка при обращении к языковой модели: {llm_error}"

        messages.append(ai_response) # До

        # Проверяем, хочет ли AI вызвать инструмент
        # У GigaChat tool_calls может быть None или пустой список
        if not ai_response.tool_calls:
            logging.info(f"[{session_id}] LLM вернул финальный ответ.")
            # История будет обновлена автоматически RunnableWithMessageHistory
            return ai_response.content # Возвращаем текстовый ответ

        # Если есть вызовы инструментов
        logging.info(f"[{session_id}] LLM запросил вызов инструментов: {len(ai_response.tool_calls)}")
        tool_messages: List[ToolMessage] = []

        for tool_call in ai_response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id") # ID для связи ответа
            logging.info(f"[{session_id}] Вызов инструмента '{tool_name}' с аргументами: {tool_args}")

            selected_tool = next((t for t in tools if t.name == tool_name), None)

            if not selected_tool:
                error_msg = f"Ошибка: LLM запросил неизвестный инструмент '{tool_name}'."
                logging.error(f"[{session_id}] {error_msg}")
                tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                continue

            # Выполняем инструмент
            try:
                # invoke инструмента автоматически обработает парсинг и вызов функции @tool
                tool_output = selected_tool.invoke(tool_args, config=config)
                # Ограничиваем длину вывода инструмента для логов и передачи в LLM
                tool_output_str = str(tool_output)
                max_tool_output_len = 2000 # Ограничение длины результата инструмента
                if len(tool_output_str) > max_tool_output_len:
                    tool_output_truncated = tool_output_str[:max_tool_output_len] + "... (результат обрезан)"
                    logging.warning(f"[{session_id}] Результат инструмента '{tool_name}' обрезан до {max_tool_output_len} символов.")
                else:
                    tool_output_truncated = tool_output_str

                logging.info(f"[{session_id}] Результат '{tool_name}': {tool_output_truncated[:200]}...")
                tool_messages.append(ToolMessage(content=tool_output_truncated, tool_call_id=tool_id))

            except Exception as e:
                # Ловим ошибки Pydantic при парсинге аргументов или ошибки выполнения process()
                error_msg = f"Ошибка выполнения инструмента '{tool_name}': {type(e).__name__}: {e}"
                logging.error(f"[{session_id}] {error_msg}", exc_info=True) # Логируем traceback
                tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))

        # Добавляем результаты инструментов в *локальный* список для следующего вызова LLM
        messages.extend(tool_messages)

    # Если вышли из цикла по MAX_TURNS
    logging.warning(f"[{session_id}] Достигнут лимит ({MAX_TURNS}) вызовов инструментов.")
    final_fallback_message = "Кажется, обработка вашего запроса заняла слишком много шагов или произошла ошибка при вызове функции. Попробуйте переформулировать."
    # RunnableWithMessageHistory добавит user_input и этот fallback_message в историю.
    return final_fallback_message
# Use code with caution. - УДАЛЕНО

agent_chain = RunnableLambda(run_agent_like_chain, name="AgentLikeChain")

chain_with_history = RunnableWithMessageHistory(
    agent_chain,
    get_session_history, # Функция для получения истории сессии
    input_messages_key="input",
    history_messages_key="chat_history",
    # output_messages_key="output" # Можно указать ключ для ответа, если агент возвращает BaseMessage
)

# --- Экспортируемые функции для использования в API ---
def initialize_assistant(
    # Параметры GigaChat (оставляем для эмбеддингов)
    gigachat_credentials: Optional[str] = None,
    giga_embedding_model: Optional[str] = None,
    giga_scope: Optional[str] = None,
    # --- НОВЫЕ ПАРАМЕТРЫ DeepSeek ---
    deepseek_api_key: Optional[str] = None,
    deepseek_chat_model: Optional[str] = None,
    # --- КОНЕЦ НОВЫХ ПАРАМЕТРОВ ---
    json_data_path: Optional[str] = None,
    chroma_persist_dir: Optional[str] = None
    # giga_chat_model: Optional[str] = None, # <-- УДАЛЕНО
) -> RunnableWithMessageHistory:
    """
    Инициализирует ассистента с заданными параметрами.
    Возвращает настроенную цепочку с историей.
    ПРИМЕЧАНИЕ: Для динамического изменения моделей во время работы API
    потребуется более сложная логика перезапуска/реконфигурации цепочки.
    Текущая реализация предполагает инициализацию при старте.
    """
    global GIGACHAT_CREDENTIALS, JSON_DATA_PATH, CHROMA_PERSIST_DIR
    global GIGA_EMBEDDING_MODEL, GIGA_SCOPE
    # --- ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ DEEPSEEK ---
    global DEEPSEEK_API_KEY, DEEPSEEK_CHAT_MODEL
    # --- КОНЕЦ ГЛОБАЛЬНЫХ DEEPSEEK ---

    # Обновляем глобальные переменные GigaChat (для эмбеддингов)
    if gigachat_credentials: GIGACHAT_CREDENTIALS = gigachat_credentials
    if giga_embedding_model: GIGA_EMBEDDING_MODEL = giga_embedding_model
    if giga_scope: GIGA_SCOPE = giga_scope

    # --- ОБНОВЛЕНИЕ ГЛОБАЛЬНЫХ DeepSeek ---
    if deepseek_api_key: DEEPSEEK_API_KEY = deepseek_api_key
    if deepseek_chat_model: DEEPSEEK_CHAT_MODEL = deepseek_chat_model
    # --- КОНЕЦ ОБНОВЛЕНИЯ DeepSeek ---

    # Обновляем остальные
    if json_data_path: JSON_DATA_PATH = json_data_path
    if chroma_persist_dir: CHROMA_PERSIST_DIR = chroma_persist_dir

    # ВАЖНО: Если параметры изменились, возможно, потребуется перезапустить
    # инициализацию моделей и цепочки. Текущая функция просто обновляет
    # глобальные переменные и возвращает уже созданную цепочку.
    logging.info("Параметры инициализации ассистента обновлены (если были переданы). Возвращаем существующую цепочку.")

    # Возвращаем готовую цепочку (которая была создана при запуске модуля
    # с использованием глобальных переменных, которые мы только что могли обновить)
    return chain_with_history

def clear_session_history(session_id: str) -> bool:
    """
    Очищает историю сессии.
    Возвращает True если сессия была найдена и очищена.
    """
    if session_id in chat_memory:
        chat_memory[session_id] = ChatMessageHistory()
        logging.info(f"История сессии {session_id} очищена")
        return True
    return False

def get_active_session_count() -> int:
    """
    Возвращает количество активных сессий в памяти.
    """
    return len(chat_memory)
