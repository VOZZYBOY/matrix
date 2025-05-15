# matrixai.py

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.runnables import  RunnableLambda, RunnableConfig
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from langchain_gigachat.embeddings import GigaChatEmbeddings
try:
    import tenant_config_manager
except ImportError:
    logging.error("Не удалось импортировать tenant_config_manager. Дополнения к промпту не будут загружаться.")
    tenant_config_manager = None
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
try:
    import rag_setup
except ImportError as e:
     logging.critical(f"Критическая ошибка: Не удалось импортировать 'rag_setup'. Убедитесь, что файл rag_setup.py существует. Ошибка: {e}", exc_info=True)
     exit()
try:
    import clinic_functions
except ImportError as e:
    logging.critical(f"Критическая ошибка: Не удалось импортировать 'clinic_functions'. Ошибка: {e}", exc_info=True)
    exit()
try:
    import redis_history
except ImportError as e:
    logging.critical(f"Критическая ошибка: Не удалось импортировать 'redis_history'. Ошибка: {e}", exc_info=True)
    exit()
from clinic_index import build_indexes_for_tenant, get_id_by_name # <--- ДОБАВЛЕНО get_id_by_name
import pytz
from datetime import datetime
import inspect # <--- ДОБАВЛЕНО
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
GIGACHAT_CREDENTIALS = "OTkyYTgyNGYtMjRlNC00MWYyLTg3M2UtYWRkYWVhM2QxNTM1OjA5YWRkODc0LWRjYWItNDI2OC04ZjdmLWE4ZmEwMDIxMThlYw=="
JSON_DATA_PATH = os.environ.get("JSON_DATA_PATH", "base/cleaned_data.json")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db_clinic_giga")
GIGA_EMBEDDING_MODEL = os.environ.get("GIGA_EMBEDDING_MODEL", "EmbeddingsGigaR")
GIGA_SCOPE = os.environ.get("GIGA_SCOPE", "GIGACHAT_API_PERS")
GIGA_VERIFY_SSL = os.getenv("GIGA_VERIFY_SSL", "False").lower() == "true"
TENANT_COLLECTION_PREFIX = "tenant_" 
try:
    chat_model = ChatOpenAI(
        model="gpt-4.1",
        max_tokens=4096,
        api_key="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"
    )
    logger.info("Чат модель OpenAI o3-mini инициализирована.")
except Exception as e:
    logger.critical(f"Ошибка инициализации модели OpenAI Chat: {e}", exc_info=True)
    exit()
# Глобальные переменные для RAG компонентов (будут инициализированы при старте)
CHROMA_CLIENT: Optional[chromadb.ClientAPI] = None
EMBEDDINGS_GIGA: Optional[GigaChatEmbeddings] = None
BM25_RETRIEVERS_MAP: Dict[str, BM25Retriever] = {}
TENANT_DOCUMENTS_MAP: Dict[str, List[Document]] = {}
TENANT_RAW_DATA_MAP: Dict[str, List[Dict[str, Any]]] = {}
SERVICE_DETAILS_MAP_GLOBAL: Dict[Tuple[str, str], Dict[str, Any]] = {} # <--- ДОБАВЛЕНО: Глобальная карта деталей услуг
search_k_global = 5
def initialize_rag_components():
    """Инициализирует RAG компоненты и сохраняет их в глобальные переменные."""
    global CHROMA_CLIENT, EMBEDDINGS_GIGA, BM25_RETRIEVERS_MAP, TENANT_DOCUMENTS_MAP, TENANT_RAW_DATA_MAP, SERVICE_DETAILS_MAP_GLOBAL
    global search_k_global
    chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 200))
    search_k_global = int(os.getenv("SEARCH_K", 5))
    data_dir_base = os.getenv("BASE_DATA_DIR", "base")
    chroma_persist_dir_global = os.getenv("CHROMA_PERSIST_DIR", "chroma_db_clinic_giga")
    if not GIGACHAT_CREDENTIALS:
        logger.critical("Внутренняя ошибка: GIGACHAT_CREDENTIALS пуста перед вызовом initialize_rag.")
        exit()
    logger.info("Запуск инициализации RAG компонентов...")
    try:
        chroma_client_init, embeddings_object_init, bm25_retrievers_init, \
            tenant_docs_init, raw_data_init, service_details_map_init = rag_setup.initialize_rag(
            data_dir=data_dir_base,
            chroma_persist_dir=chroma_persist_dir_global,
            embedding_credentials=GIGACHAT_CREDENTIALS,
            embedding_model=GIGA_EMBEDDING_MODEL,
            embedding_scope=GIGA_SCOPE,
            verify_ssl_certs=GIGA_VERIFY_SSL,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            search_k=search_k_global
        )
        if chroma_client_init and embeddings_object_init:
            CHROMA_CLIENT = chroma_client_init
            EMBEDDINGS_GIGA = embeddings_object_init
            BM25_RETRIEVERS_MAP = bm25_retrievers_init
            TENANT_DOCUMENTS_MAP = tenant_docs_init
            TENANT_RAW_DATA_MAP = raw_data_init
            SERVICE_DETAILS_MAP_GLOBAL = service_details_map_init
            # --- Построение индексов для каждого тенанта ---
            for tenant_id, raw_data in TENANT_RAW_DATA_MAP.items():
                build_indexes_for_tenant(tenant_id, raw_data)
            logger.info(f"Инициализация RAG завершена. Загружено:")
            logger.info(f"  - Chroma клиент: {'Да' if CHROMA_CLIENT else 'Нет'}")
            logger.info(f"  - Embeddings: {'Да' if EMBEDDINGS_GIGA else 'Нет'}")
            logger.info(f"  - BM25 ретриверы: {len(BM25_RETRIEVERS_MAP)} шт.")
            logger.info(f"  - Карта документов тенантов: {len(TENANT_DOCUMENTS_MAP)} тенантов.")
            logger.info(f"  - Карта сырых данных тенантов: {len(TENANT_RAW_DATA_MAP)} тенантов.")
        else:
            logger.error("Не удалось полностью инициализировать RAG компоненты.")
            exit()
    except Exception as e:
        logger.critical(f"Критическая ошибка во время инициализации RAG: {e}", exc_info=True)
        exit()
initialize_rag_components()
def format_docs(docs: List[Document]) -> str:
    """Форматирует найденные RAG документы для добавления в промпт."""
    if not docs:
        return "Дополнительная информация из базы знаний не найдена."
    formatted_docs = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata or {}
        source_info = metadata.get('id', 'Неизвестный источник')
        doc_type = metadata.get('type', 'данные')
        doc_name = metadata.get('name', '')
        doc_topic = metadata.get('topic', '')
        display_source = doc_type
        if doc_topic: display_source += f" ({doc_topic})"
        if doc_name: display_source += f" - '{doc_name}'"
        if not doc_topic and not doc_name: display_source = source_info
        score = metadata.get('relevance_score', None)
        score_str = f" (Score: {score:.4f})" if score is not None else ""
        formatted_docs.append(f"Источник {i+1}{score_str} ({display_source}):\n{doc.page_content}")
    return "\n\n".join(formatted_docs)
class RagQueryThought(BaseModel):
    analysis: str = Field(description="Анализ последнего запроса пользователя в контексте истории диалога. Если пользователь ссылается на что-то из истории (например, номер пункта, местоимение 'он'/'она'/'это'), явно укажи, к какой сущности (услуге, врачу, филиалу) это относится, используя ее полное название из истории.")
    best_rag_query: str = Field(description="Сформулируй оптимальный, самодостаточный запрос для поиска описания или детальной информации об основной сущности запроса в векторной базе знаний. Используй полное имя/название сущности. Если запрос общий (приветствие, не по теме) или явно запрашивает вызов функции (цена, список филиалов/услуг/врачей и т.д.), оставь поле пустым или верни исходный запрос.")
class FindEmployeesArgs(BaseModel):
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника для фильтрации (опционально)")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название КОНКРЕТНОЙ услуги для фильтрации (опционально)")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала. Если указано только это поле, вернет ВСЕХ сотрудников филиала.")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=15, description="Количество сотрудников на странице для пагинации")
def find_employees_tool(employee_name: Optional[str] = None, service_name: Optional[str] = None, filial_name: Optional[str] = None) -> str:
    """Ищет сотрудников клиники по ФИО, выполняемой услуге или филиалу."""
    handler = clinic_functions.FindEmployees(employee_name=employee_name, service_name=service_name, filial_name=filial_name)
    return handler.process()
class GetServicePriceArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги (например, 'Soprano Пальцы для женщин')")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала (например, 'Москва-сити'), если нужно уточнить цену в конкретном месте")
def get_service_price_tool(service_name: str, filial_name: Optional[str] = None) -> str:
    """Возвращает цену на КОНКРЕТНУЮ услугу клиники."""
    handler = clinic_functions.GetServicePrice(service_name=service_name, filial_name=filial_name)
    return handler.process()
def list_filials_tool() -> str:
    """Возвращает список всех доступных филиалов клиники."""
    handler = clinic_functions.ListFilials()
    return handler.process()
class GetEmployeeServicesArgs(BaseModel):
    employee_name: str = Field(description="Точное или максимально близкое ФИО сотрудника")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=20, description="Количество услуг на странице для пагинации")
def get_employee_services_tool(employee_name: str) -> str:
    """Возвращает список услуг КОНКРЕТНОГО сотрудника."""
    handler = clinic_functions.GetEmployeeServices(employee_name=employee_name)
    return handler.process()
class CheckServiceInFilialArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")
def check_service_in_filial_tool(service_name: str, filial_name: str) -> str:
    """Проверяет, доступна ли КОНКРЕТНАЯ услуга в КОНКРЕТНОМ филиале."""
    handler = clinic_functions.CheckServiceInFilial(service_name=service_name, filial_name=filial_name)
    return handler.process()
class CompareServicePriceInFilialsArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(min_length=2, description="Список из ДВУХ или БОЛЕЕ названий филиалов")
def compare_service_price_in_filials_tool(service_name: str, filial_names: List[str]) -> str:
    """Сравнивает цену КОНКРЕТНОЙ услуги в НЕСКОЛЬКИХ филиалах."""
    handler = clinic_functions.CompareServicePriceInFilials(service_name=service_name, filial_names=filial_names)
    return handler.process()
class FindServiceLocationsArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
def find_service_locations_tool(service_name: str) -> str:
    """Ищет все филиалы, где доступна КОНКРЕТНАЯ услуга."""
    handler = clinic_functions.FindServiceLocations(service_name=service_name)
    return handler.process()
class FindSpecialistsByServiceOrCategoryAndFilialArgs(BaseModel):
    query_term: str = Field(description="Название услуги ИЛИ категории")
    filial_name: str = Field(description="Точное название филиала")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=15, description="Количество специалистов на странице для пагинации")
def find_specialists_by_service_or_category_and_filial_tool(query_term: str, filial_name: str) -> str:
    """Ищет СПЕЦИАЛИСТОВ по УСЛУГЕ/КАТЕГОРИИ в КОНКРЕТНОМ филиале."""
    handler = clinic_functions.FindSpecialistsByServiceOrCategoryAndFilial(query_term=query_term.lower(), filial_name=filial_name.lower())
    return handler.process()
class ListServicesInCategoryArgs(BaseModel):
    category_name: str = Field(description="Точное название категории")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=20, description="Количество услуг на странице для пагинации")
def list_services_in_category_tool(category_name: str) -> str:
    """Возвращает список КОНКРЕТНЫХ услуг в указанной КАТЕГОРИИ."""
    handler = clinic_functions.ListServicesInCategory(category_name=category_name)
    return handler.process()
class ListServicesInFilialArgs(BaseModel):
    filial_name: str = Field(description="Точное название филиала")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=30, description="Количество услуг на странице для пагинации (учитывайте, что вывод также содержит заголовки категорий)")
def list_services_in_filial_tool(filial_name: str) -> str:
    """Возвращает ПОЛНЫЙ список УНИКАЛЬНЫХ услуг в КОНКРЕТНОМ филиале."""
    handler = clinic_functions.ListServicesInFilial(filial_name=filial_name)
    return handler.process()
class FindServicesInPriceRangeArgs(BaseModel):
    min_price: float = Field(description="Минимальная цена")
    max_price: float = Field(description="Максимальная цена")
    category_name: Optional[str] = Field(default=None, description="Опционально: категория")
    filial_name: Optional[str] = Field(default=None, description="Опционально: филиал")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=20, description="Количество услуг на странице для пагинации")
def find_services_in_price_range_tool(min_price: float, max_price: float, category_name: Optional[str] = None, filial_name: Optional[str] = None) -> str:
    """Ищет услуги в ЗАДАННОМ ЦЕНОВОМ ДИАПАЗОНЕ."""
    handler = clinic_functions.FindServicesInPriceRange(min_price=min_price, max_price=max_price, category_name=category_name, filial_name=filial_name)
    return handler.process()
class ListAllCategoriesArgs(BaseModel):
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=30, description="Количество категорий на странице для пагинации")
def list_all_categories_tool() -> str:
    """Возвращает список ВСЕХ категорий услуг."""
    handler = clinic_functions.ListAllCategories()
    return handler.process()
class ListEmployeeFilialsArgs(BaseModel):
    employee_name: str = Field(description="Точное или близкое ФИО сотрудника")
def list_employee_filials_tool(employee_name: str) -> str:
    """ОБЯЗАТЕЛЬНО ВЫЗЫВАЙ для получения ПОЛНОГО списка ВСЕХ филиалов КОНКРЕТНОГО сотрудника."""
    logger.info(f"Вызов list_employee_filials_tool для: {employee_name}")
    handler = clinic_functions.ListEmployeeFilials(employee_name=employee_name)
    return handler.process()
from clinic_functions import GetFreeSlots, BookAppointment
class GetFreeSlotsArgs(BaseModel):
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    employee_name: str = Field(description="ФИО сотрудника (точно или частично)")
    service_names: List[str] = Field(description="Список названий услуг (точно)")
    date_time: str = Field(description="Дата для поиска слотов (формат YYYY-MM-DD)")
    filial_name: str = Field(description="Название филиала (точно)")
    lang_id: str = Field(default="ru", description="Язык ответа")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")
class BookAppointmentArgs(BaseModel):
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    phone_number: str = Field(description="Телефон клиента")
    service_name: str = Field(description="Название услуги (точно)")
    employee_name: str = Field(description="ФИО сотрудника (точно)")
    filial_name: str = Field(description="Название филиала (точно)")
    category_name: str = Field(description="Название категории (точно)")
    date_of_record: str = Field(description="Дата записи (YYYY-MM-DD)")
    start_time: str = Field(description="Время начала (HH:MM)")
    end_time: str = Field(description="Время окончания (HH:MM)")
    duration_of_time: int = Field(description="Длительность услуги (минуты)")
    lang_id: str = Field(default="ru", description="Язык ответа")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")
    price: float = Field(default=0, description="Цена услуги")
    sale_price: float = Field(default=0, description="Цена со скидкой")
    complex_service_id: str = Field(default="", description="ID комплексной услуги (если есть)")
    color_code_record: str = Field(default="", description="Цвет записи (опционально)")
    total_price: float = Field(default=0, description="Общая цена")
    traffic_channel: int = Field(default=0, description="Канал трафика (опционально)")
    traffic_channel_id: str = Field(default="", description="ID канала трафика (опционально)")

class GetEmployeeScheduleArgs(BaseModel):
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    employee_name: str = Field(description="ФИО сотрудника (точно или частично)")
    filial_name: str = Field(description="Название филиала (точно)")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")

TOOL_CLASSES = [
    FindEmployeesArgs,
    GetServicePriceArgs,
    GetEmployeeServicesArgs,
    CheckServiceInFilialArgs,
    CompareServicePriceInFilialsArgs,
    FindServiceLocationsArgs,
    FindSpecialistsByServiceOrCategoryAndFilialArgs,
    ListServicesInCategoryArgs,
    ListServicesInFilialArgs,
    FindServicesInPriceRangeArgs,
    ListAllCategoriesArgs,
    ListEmployeeFilialsArgs,
    GetFreeSlotsArgs,  # Новый класс-аргументы
    BookAppointmentArgs,  # Новый класс-аргументы
    GetEmployeeScheduleArgs, # <--- НОВЫЙ ИНСТРУМЕНТ
]
logger.info(f"Определено {len(TOOL_CLASSES)} Pydantic классов для аргументов инструментов.")

async def get_free_slots_tool(*, config=None, **kwargs_from_llm) -> str:
    tenant_id_from_config = kwargs_from_llm.get('tenant_id')
    api_token_from_config = kwargs_from_llm.get('api_token')

    if not tenant_id_from_config:
        logger.error(f"tenant_id отсутствует в аргументах ({kwargs_from_llm}) для GetFreeSlots.")
        return "Критическая ошибка: ID тенанта не был предоставлен для вызова GetFreeSlots."

    employee_name_from_llm = kwargs_from_llm.get('employee_name')
    service_names_from_llm = kwargs_from_llm.get('service_names')
    filial_name_from_llm = kwargs_from_llm.get('filial_name')
    date_time_from_llm = kwargs_from_llm.get('date_time')

    logger.info(f"[get_free_slots_tool] Имена от LLM: employee='{employee_name_from_llm}', services='{service_names_from_llm}', filial='{filial_name_from_llm}'")

    if not all([employee_name_from_llm, service_names_from_llm, filial_name_from_llm, date_time_from_llm]):
        missing_fields = []
        if not employee_name_from_llm: missing_fields.append('employee_name')
        if not service_names_from_llm: missing_fields.append('service_names')
        if not filial_name_from_llm: missing_fields.append('filial_name')
        if not date_time_from_llm: missing_fields.append('date_time')
        return f"Ошибка: отсутствуют обязательные поля от LLM: {', '.join(missing_fields)}"

    # Преобразование имен в ID
    logger.info(f"[get_free_slots_tool] Попытка получить ID для employee: '{employee_name_from_llm}'")
    employee_id = get_id_by_name(tenant_id_from_config, 'employee', employee_name_from_llm)
    logger.info(f"[get_free_slots_tool] Попытка получить ID для filial: '{filial_name_from_llm}'")
    filial_id = get_id_by_name(tenant_id_from_config, 'filial', filial_name_from_llm)
    
    service_ids = []
    if service_names_from_llm:
        for s_name in service_names_from_llm:
            logger.info(f"[get_free_slots_tool] Попытка получить ID для service: '{s_name}'")
            s_id = get_id_by_name(tenant_id_from_config, 'service', s_name)
            service_ids.append(s_id)

    if not employee_id:
        return f"Не удалось найти ID для сотрудника: '{employee_name_from_llm}'"
    if not filial_id:
        return f"Не удалось найти ID для филиала: '{filial_name_from_llm}'"
    if not all(service_ids):
        problematic_services = [s_name for s_name, s_id in zip(service_names_from_llm, service_ids) if not s_id]
        return f"Не удалось найти ID для следующих услуг: {', '.join(problematic_services)}"

    handler_args = {
        "tenant_id": tenant_id_from_config,
        "employee_id": employee_id,
        "service_ids": service_ids,
        "date_time": date_time_from_llm,
        "filial_id": filial_id,
        "api_token": api_token_from_config
    }

    try:
        logger.debug(f"Создание экземпляра GetFreeSlots с аргументами (ID): {handler_args}")
        handler = clinic_functions.GetFreeSlots(**handler_args)
        logger.debug(f"Вызов handler.process() для GetFreeSlots")
        return await handler.process()
    except Exception as e:
        logger.error(f"Ошибка при создании или обработке GetFreeSlots: {e}", exc_info=True)
        error_type = getattr(e, '__class__', Exception).__name__
        return f"Ошибка при обработке запроса на свободные слоты ({error_type}): {str(e)}"

async def book_appointment_tool(*, config=None, **kwargs_from_llm) -> str:
    tenant_id_from_config = kwargs_from_llm.get('tenant_id')
    api_token_from_config = kwargs_from_llm.get('api_token')

    if not tenant_id_from_config:
        logger.error(f"tenant_id отсутствует в аргументах ({kwargs_from_llm}) для BookAppointment.")
        return "Критическая ошибка: ID тенанта не был предоставлен для вызова BookAppointment."

    # Извлекаем все необходимые поля от LLM
    phone_number_from_llm = kwargs_from_llm.get('phone_number')
    service_name_from_llm = kwargs_from_llm.get('service_name')
    employee_name_from_llm = kwargs_from_llm.get('employee_name')
    filial_name_from_llm = kwargs_from_llm.get('filial_name')
    category_name_from_llm = kwargs_from_llm.get('category_name')
    date_of_record_from_llm = kwargs_from_llm.get('date_of_record')
    start_time_from_llm = kwargs_from_llm.get('start_time')
    end_time_from_llm = kwargs_from_llm.get('end_time')
    duration_of_time_from_llm = kwargs_from_llm.get('duration_of_time')
    price_from_llm = kwargs_from_llm.get('price', 0)

    logger.info(f"[book_appointment_tool] Имена от LLM: service='{service_name_from_llm}', employee='{employee_name_from_llm}', filial='{filial_name_from_llm}', category='{category_name_from_llm}'")

    required_names = {
        'phone_number': phone_number_from_llm,
        'service_name': service_name_from_llm,
        'employee_name': employee_name_from_llm,
        'filial_name': filial_name_from_llm,
        'category_name': category_name_from_llm,
        'date_of_record': date_of_record_from_llm,
        'start_time': start_time_from_llm,
        'end_time': end_time_from_llm,
        'duration_of_time': duration_of_time_from_llm
    }
    missing_fields = [name for name, val in required_names.items() if val is None]
    if missing_fields:
        logger.error(f"[book_appointment_tool] Отсутствуют обязательные поля от LLM: {', '.join(missing_fields)}. Аргументы от LLM: {kwargs_from_llm}")
        return f"Ошибка: отсутствуют обязательные поля от LLM для записи: {', '.join(missing_fields)}"
    
    # Преобразование имен в ID
    logger.info(f"[book_appointment_tool] Попытка получить ID для service: '{service_name_from_llm}'")
    service_id = get_id_by_name(tenant_id_from_config, 'service', service_name_from_llm)
    logger.info(f"[book_appointment_tool] Попытка получить ID для employee: '{employee_name_from_llm}'")
    employee_id = get_id_by_name(tenant_id_from_config, 'employee', employee_name_from_llm)
    logger.info(f"[book_appointment_tool] Попытка получить ID для filial: '{filial_name_from_llm}'")
    filial_id = get_id_by_name(tenant_id_from_config, 'filial', filial_name_from_llm)
    logger.info(f"[book_appointment_tool] Попытка получить ID для category: '{category_name_from_llm}'")
    category_id = get_id_by_name(tenant_id_from_config, 'category', category_name_from_llm)

    if not service_id: return f"Не удалось найти ID для услуги: '{service_name_from_llm}'"
    if not employee_id: return f"Не удалось найти ID для сотрудника: '{employee_name_from_llm}'"
    if not filial_id: return f"Не удалось найти ID для филиала: '{filial_name_from_llm}'"
    if not category_id: return f"Не удалось найти ID для категории: '{category_name_from_llm}'"

    handler_args = {
        "tenant_id": tenant_id_from_config,
        "phone_number": phone_number_from_llm,
        "service_id": service_id,
        "employee_id": employee_id,
        "filial_id": filial_id,
        "category_id": category_id,
        "date_of_record": date_of_record_from_llm,
        "start_time": start_time_from_llm,
        "end_time": end_time_from_llm,
        "duration_of_time": duration_of_time_from_llm,
        "api_token": api_token_from_config,
        "price": price_from_llm,
        "sale_price": kwargs_from_llm.get('sale_price', 0),
        "complex_service_id": kwargs_from_llm.get('complex_service_id', ""),
        "color_code_record": kwargs_from_llm.get('color_code_record', ""),
        "total_price": kwargs_from_llm.get('total_price', 0),
        "traffic_channel": kwargs_from_llm.get('traffic_channel', 0),
        "traffic_channel_id": kwargs_from_llm.get('traffic_channel_id', "")
    }
        
    try:
        logger.debug(f"Создание экземпляра BookAppointment с аргументами (ID): {handler_args}")
        handler = clinic_functions.BookAppointment(**handler_args)
        logger.debug(f"Вызов handler.process() для BookAppointment")
        return await handler.process()
    except Exception as e:
        logger.error(f"Ошибка при создании или обработке BookAppointment: {e}", exc_info=True)
        error_type = getattr(e, '__class__', Exception).__name__
        return f"Ошибка при обработке запроса на запись ({error_type}): {str(e)}"

async def get_employee_schedule_tool(*, config=None, **kwargs_from_llm) -> str:
    tenant_id_from_config = kwargs_from_llm.get('tenant_id') # Этот ID нужен для get_id_by_name
    api_token_from_config = kwargs_from_llm.get('api_token')

    if not tenant_id_from_config: # Проверка остается, т.к. нужен для get_id_by_name
        logger.error(f"tenant_id отсутствует в аргументах ({kwargs_from_llm}) для GetEmployeeSchedule.")
        return "Критическая ошибка: ID тенанта не был предоставлен для разрешения имен."
    
    if not api_token_from_config: # Теперь токен критичен для API вызова
        logger.error(f"api_token отсутствует в аргументах ({kwargs_from_llm}) для GetEmployeeSchedule.")
        return "Критическая ошибка: API токен не был предоставлен для вызова GetEmployeeSchedule."

    employee_name_from_llm = kwargs_from_llm.get('employee_name')
    filial_name_from_llm = kwargs_from_llm.get('filial_name')

    logger.info(f"[get_employee_schedule_tool] Имена от LLM: employee='{employee_name_from_llm}', filial='{filial_name_from_llm}'")

    if not employee_name_from_llm or not filial_name_from_llm:
        missing_fields = []
        if not employee_name_from_llm: missing_fields.append('employee_name')
        if not filial_name_from_llm: missing_fields.append('filial_name')
        return f"Ошибка: отсутствуют обязательные поля от LLM: {', '.join(missing_fields)} для получения расписания."

    logger.info(f"[get_employee_schedule_tool] Попытка получить ID для employee: '{employee_name_from_llm}'")
    employee_id = get_id_by_name(tenant_id_from_config, 'employee', employee_name_from_llm)
    logger.info(f"[get_employee_schedule_tool] Попытка получить ID для filial: '{filial_name_from_llm}'")
    filial_id = get_id_by_name(tenant_id_from_config, 'filial', filial_name_from_llm)

    if not employee_id:
        return f"Не удалось найти ID для сотрудника: '{employee_name_from_llm}'"
    if not filial_id:
        return f"Не удалось найти ID для филиала: '{filial_name_from_llm}'"

    handler_args = {
        "tenant_id": tenant_id_from_config, # Передаем tenant_id для get_original_name в clinic_functions
        "employee_id": employee_id,
        "filial_id": filial_id,
        "api_token": api_token_from_config # Это основной аутентификатор для API
    }

    try:
        logger.debug(f"Создание экземпляра GetEmployeeSchedule с аргументами (ID): {handler_args}")
        handler = clinic_functions.GetEmployeeSchedule(**handler_args)
        logger.debug(f"Вызов handler.process() для GetEmployeeSchedule")
        return await handler.process()
    except Exception as e:
        logger.error(f"Ошибка при создании или обработке GetEmployeeSchedule: {e}", exc_info=True)
        error_type = getattr(e, '__class__', Exception).__name__
        return f"Ошибка при обработке запроса на расписание сотрудника ({error_type}): {str(e)}"

TOOL_FUNCTIONS = [
    find_employees_tool,
    get_service_price_tool,
    list_filials_tool,
    get_employee_services_tool,
    check_service_in_filial_tool,
    compare_service_price_in_filials_tool,
    find_service_locations_tool,
    find_specialists_by_service_or_category_and_filial_tool,
    list_services_in_category_tool,
    list_services_in_filial_tool,
    find_services_in_price_range_tool,
    list_all_categories_tool,
    list_employee_filials_tool,
    get_free_slots_tool, 
    book_appointment_tool,  
    get_employee_schedule_tool, # <--- НОВЫЙ ИНСТРУМЕНТ
]
logger.info(f"Определено {len(TOOL_FUNCTIONS)} функций-инструментов для динамической привязки.")
SYSTEM_PROMPT = """Ты - вежливый, ОЧЕНЬ ВНИМАТЕЛЬНЫЙ и информативный ИИ-ассистент.
Твоя главная задача - помогать пользователям, отвечая на их вопросы об услугах, ценах, специалистах и филиалах , И ПОДДЕРЖИВАТЬ ЕСТЕСТВЕННЫЙ ДИАЛОГ.
ИСПОЛЬЗУЙ RAG ПОИСК ТОЛЬКО ДЛЯ ОПИСАНИЙ УСЛУГ И СПЕЦИАЛИСТОВ, А НЕ ДЛЯ КОНКРЕТНЫХ ДАННЫХ (ЦЕНЫ, СПИСКИ СОТРУДНИКОВ, ФИЛИАЛОВ И Т.Д.). СТАРАЙСЯ ВСЕ РЕШАТЬ ЧЕРЕЗ ВЫЗОВ ФУНКЦИЙ (ИНСТРУМЕНТОВ).

КЛЮЧЕВЫЕ ПРАВИЛА РАБОТЫ:

АНАЛИЗ ИСТОРИИ И ВЫБОР ДЕЙСТВИЯ:
- Внимательно проанализируй ПОЛНУЮ ИСТОРИЮ ДИАЛОГА (chat_history).
- ИСПОЛЬЗУЙ КОНТЕКСТ ИСТОРИИ! Не переспрашивай.
- ЗАПОМИНАЙ ИМЯ ПОЛЬЗОВАТЕЛЯ, если он представился.
Всегда используй функции для уточнения информации кроме случаяв когда клиент просит что-то посветовать или рассказать о чем-то

ПЕРСОНАЛИЗАЦИЯ НА ОСНОВЕ РЕЗЮМЕ ПРЕДПОЧТЕНИЙ КЛИЕНТА:
- Если в истории диалога или в информации, добавленной к текущему запросу пользователя, присутствует блок "Резюме предпочтений клиента:", ОБЯЗАТЕЛЬНО используй эту информацию для персонализации общения.
- Это резюме может содержать:
    - Имя клиента (если он был идентифицирован).
    - Краткую историю последних записей.
    - Информацию о часто используемых услугах.
    - Информацию о часто посещаемых специалистах (с пометкой "[часто]", если визитов много).
    - Информацию о всех посещенных филиалах с указанием количества визитов (отсортированы по частоте).
- Как использовать эту информацию:
    - Если есть имя клиента, обращайся к нему по имени (например, "Здравствуйте, Анна!").
    - Если есть информация о частых услугах или специалистах, ты можешь тактично упомянуть это и предложить, например, повторную запись:
        - "Анна, я вижу, вы часто пользуетесь услугой 'Название Услуги' (X раз(а)). Не хотели бы записаться на нее снова?"
        - "Заметил(а), что вы часто посещаете специалиста 'Имя Специалиста' (Y раз(а)). Могу предложить вам доступное время для записи к нему."
    - Если есть информация о предпочтительном филиале, учитывай это при предложениях или ответах на вопросы о доступности услуг/специалистов.
    - Если в резюме указано несколько посещенных филиалов, ты можешь уточнить у клиента, какой из них для него наиболее удобен или актуален для текущего запроса. Например: "Вижу, вы посещали филиалы X (3 раза) и Y (2 раза). Какой из них вас интересует сейчас?"
    - Не повторяй дословно всю информацию из резюме. Используй ее для того, чтобы сделать диалог более естественным и релевантным для клиента.
    - Если клиент задает вопрос, который можно связать с его предпочтениями (например, "посоветуйте процедуру"), его прошлые услуги могут быть хорошей отправной точкой для рекомендации.

ПРИВЕТСТВИЕ И ИНФОРМАЦИЯ:
- Если ты получаешь информацию о предыдущих записях клиента (блок "Предыдущие записи клиента...") ИЛИ "Резюме предпочтений клиента", И это начало диалога (например, первое сообщение от пользователя или история чата пуста/короткая), начни свой ПЕРВЫЙ ответ с краткого и дружелюбного упоминания этой информации.
- Например: "Здравствуйте, [Имя клиента, если известно]! Вижу у вас есть недавние записи: [кратко 1-2 последние записи]. Чем могу помочь сегодня?"
- Или, если есть только резюме: "Приветствую! Заметил, вы часто пользуетесь [услуга] и посещаете [специалист/филиал]. Что вас интересует сегодня?"
- СДЕЛАЙ ЭТО ТОЛЬКО ОДИН РАЗ В НАЧАЛЕ ДИАЛОГА. В последующих сообщениях отвечай на вопросы пользователя как обычно.
- Если информации о клиенте или его записях нет, просто начни диалог стандартным приветствием.

ВЫБОР МЕЖДУ RAG, FUNCTION CALLING, ПАМЯТЬЮ ДИАЛОГА ИЛИ ПРЯМЫМ ОТВЕТОМ:
- ПАМЯТЬ ДИАЛОГА: Для ответов на вопросы, связанные с предыдущим контекстом (местоимения "он/она/это", короткие вопросы "где?", "цена?", "кто?") , и для вопросов о самом пользователе.
- RAG (Поиск по базе знаний): Используй ТОЛЬКО для ЗАПРОСОВ **ОПИСАНИЯ** услуг, врачей ИЛИ **ОБЩЕЙ ИНФОРМАЦИИ О КЛИНИКЕ** (например, "расскажи о компании", "какой у вас подход?", "сколько филиалов?"). Я предоставлю контекст. Синтезируй ответ на его основе.
    - **НОВИНКА: Информация о показаниях и противопоказаниях**: Для некоторых услуг в RAG-контексте теперь может содержаться информация о ПОКАЗАНИЯХ и ПРОТИВОПОКАЗАНИЯХ.
        - Если пользователь спрашивает общее описание услуги, ты можешь кратко упомянуть о наличии таких деталей (например, "...также у процедуры есть свои показания и противопоказания.") и предложить рассказать подробнее, если ему интересно.
        - Если пользователь ЦЕЛЕНАПРАВЛЕННО спрашивает о показаниях или противопоказаниях к услуге, используй информацию из RAG-контекста для ответа.
        - **ВАЖНО: НЕ ДАВАЙ МЕДИЦИНСКИХ СОВЕТОВ!** Не говори "вам это подойдет" или "вам это нельзя". Формулируй нейтрально: "Среди показаний к [Услуга] указываются...", "Противопоказаниями являются...". Всегда подчеркивай, что окончательное решение принимает врач.
- FUNCTION CALLING (Вызов Инструментов): Используй ТОЛЬКО для запросов КОНКРЕТНЫХ ДАННЫХ: цены, списки врачей/услуг/филиалов, проверка наличия, сравнение цен, ПОЛНЫЙ список филиалов сотрудника. Используй правильный инструмент.
- ПРЯМОЙ ОТВЕТ: Для приветствий, прощаний, простых уточнений или вопросов не по теме.

ПРАВИЛА FUNCTION CALLING:
- **Приоритет над RAG для списков и конкретных данных:** Если запрос касается получения СПИСКА (услуг сотрудника, филиалов сотрудника, цен, списка сотрудников по критерию и т.д.) или КОНКРЕТНЫХ ДАННЫХ, ВЫЗОВ СООТВЕТСТВУЮЩЕГО ИНСТРУМЕНТА ЯВЛЯЕТСЯ ОБЯЗАТЕЛЬНЫМ, ДАЖЕ ЕСЛИ RAG-КОНТЕКСТ СОДЕРЖИТ ПОХОЖУЮ ИНФОРМАЦИЮ. Это гарантирует использование актуальных данных и форматирования из инструмента.
- **Обязательность вызова для списков:** Если пользователь спрашивает список (услуг, врачей, филиалов, категорий) или просит найти что-то по критериям, ТЫ ОБЯЗАН ВЫЗВАТЬ СООТВЕТСТВУЮЩИЙ ИНСТРУМЕНТ. Особенно:
    - Для вопроса о ВСЕХ филиалах сотрудника ('где еще работает?', 'в каких филиалах?') -> ОБЯЗАТЕЛЬНО вызывай `list_employee_filials_tool`.
    - Для вопроса о ВСЕХ услугах сотрудника (например, "какие услуги выполняет ХХХ?", "чем занимается YYY?") -> ОБЯЗАТЕЛЬНО вызывай `get_employee_services_tool`. Даже если RAG-контекст содержит упоминания или частичный список услуг этого сотрудника (например, в его общем описании), для получения ПОЛНОГО и ТОЧНОГО списка услуг используй ИСКЛЮЧИТЕЛЬНО этот инструмент.
    - Для вопроса о ЦЕНЕ КОНКРЕТНОЙ услуги -> ОБЯЗАТЕЛЬНО вызывай `get_service_price_tool`.
    - **Для вопроса о СПЕЦИАЛИСТАХ в КОНКРЕТНОМ ФИЛИАЛЕ (например, "кто работает в X?", "какие врачи есть в Y?") -> ОБЯЗАТЕЛЬНО вызывай `find_employees_tool` ТОЛЬКО с параметром `filial_name`.**
    - **Для запроса РАБОЧИХ ДНЕЙ КОНКРЕТНОГО СОТРУДНИКА В КОНКРЕТНОМ ФИЛИАЛЕ (например, "когда работает ХХХ в филиале YYY?", "по каким дням принимает ZZZ в AAA?") -> ОБЯЗАТЕЛЬНО вызывай `get_employee_schedule_tool`.**
- Точность Параметров: Извлекай параметры ТОЧНО из запроса и ИСТОРИИ.
- Не Выдумывай Параметры: Если обязательного параметра нет, НЕ ВЫЗЫВАЙ функцию, а вежливо попроси уточнить.
    - **ОСОБЕННО ВАЖНО для `find_specialists_by_service_or_category_and_filial_tool`**: Этот инструмент ТРЕБУЕТ указания `filial_name`. Если пользователь спрашивает, например, "Кто делает [Услуга/Категория]?" и НЕ УКАЗЫВАЕТ ФИЛИАЛ (и его нет в недавней истории диалога), ТЫ ОБЯЗАН СНАЧАЛА УТОЧНИТЬ у пользователя: "В каком филиале вас интересуют специалисты по [Услуга/Категория]?" и только ПОСЛЕ ПОЛУЧЕНИЯ ответа на этот вопрос вызывать инструмент с указанным филиалом.
- **Если результат вызова функции содержит информацию о страницах (например, "Показаны 1-15 из 100. Страница 1 из 7."), ОБЯЗАТЕЛЬНО СООБЩИ пользователю об общем количестве и спроси, хочет ли он увидеть следующую страницу.**
    - **Если пользователь ПОСЛЕ получения пагинированного ответа (например, ответа, содержащего "Страница X из Y") пишет что-то вроде "дальше", "следующая", "еще", "давай", "продолжай", "покажи остальных" и т.п., это означает запрос на СЛЕДУЮЩУЮ СТРАНИЦУ того же списка.**
        - **Твои действия в этом случае:**
            1. **Идентифицируй ТОЧНО ТУ ЖЕ ФУНКЦИЮ (инструмент), которая вернула этот пагинированный список.**
            2. **Вспомни (или найди в истории диалога в твоем предыдущем сообщении AIMessage с tool_calls) ВСЕ АРГУМЕНТЫ, которые ты использовал для вызова этой функции в прошлый раз (кроме `page_number`).**
            3. **Определи ТЕКУЩИЙ НОМЕР СТРАНИЦЫ (X) из ответа функции (например, из "Страница X из Y").**
            4. **Вызови эту ЖЕ ФУНКЦИЮ СНОВА, используя ВСЕ ТЕ ЖЕ САМЫЕ АРГУМЕНТЫ, что и в прошлый раз, НО установи `page_number` равным X + 1.**
            5. **НЕ ПЫТАЙСЯ выполнить RAG-поиск, НЕ вызывай другую функцию, и НЕ меняй другие аргументы функции.** Твоя единственная задача - показать следующую страницу того же списка.
    - Если ты ожидаешь, что список будет длинным, можешь сразу вызвать функцию с `page_number=1` и стандартным `page_size`, а затем сообщить пользователю об общем количестве найденных элементов и страниц.
- ОБРАБОТКА НЕУДАЧНЫХ ВЫЗОВОВ: Если инструмент вернул ошибку или 'не найдено', НЕ ПЫТАЙСЯ вызвать его с теми же аргументами. Сообщи пользователю или предложи альтернативу.
- Интерпретация Результатов: Представляй результаты функций в понятной, человеческой форме. **ВАЖНО: Если функция вернула список (например, услуг, врачей, филиалов с пагинацией), ты ДОЛЖЕН показать этот список ПОЛНОСТЬЮ, как он был возвращен функцией, включая всю информацию о страницах. НЕ СОКРАЩАЙ и НЕ СУММИРУЙ такие списки.**

ОБЩИЕ ПРАВИЛА:
- Точность: НЕ ПРИДУМЫВАЙ.
- Краткость и Ясность.
- Вежливость.
- Медицинские Советы: НЕ ДАВАЙ. Напоминай, что консультация со специалистом необходима.

ВАЖНО: Всегда сначала анализируй историю и цель пользователя. Реши, нужен ли ответ из памяти, RAG, вызов функции или простой ответ. Действуй соответственно.
"""
class TenantAwareRedisChatMessageHistory(BaseChatMessageHistory):
    """Хранилище истории сообщений в Redis с учетом tenant_id."""
    def __init__(self, tenant_id: str, session_id: str):
        if not tenant_id or not session_id:
            raise ValueError("Tenant ID и Session ID не могут быть пустыми.")
        self.tenant_id = tenant_id
        self.session_id = session_id
        logger.debug(f"Инициализирован TenantAwareRedisChatMessageHistory для tenant='{self.tenant_id}', session='{self.session_id}'")
    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Получение сообщений из Redis."""
        logger.debug(f"Запрос истории для tenant='{self.tenant_id}', session='{self.session_id}'")
        items = redis_history.get_history(self.tenant_id, self.session_id)
        try:
            messages = messages_from_dict(items)
            logger.debug(f"Загружено {len(messages)} сообщений из Redis.")
            return messages
        except Exception as e:
             logger.error(f"Ошибка при преобразовании истории из dict в BaseMessage для tenant='{self.tenant_id}', session='{self.session_id}': {e}", exc_info=True)
             return []
    def add_message(self, message: BaseMessage) -> None:
        """Добавление сообщения в Redis."""
        logger.debug(f"Добавление сообщения для tenant='{self.tenant_id}', session='{self.session_id}': {message.type}")
        try:
             message_dict_list = messages_to_dict([message])
             if not message_dict_list:
                  raise ValueError("messages_to_dict вернул пустой список")
             message_dict = message_dict_list[0]
             success = redis_history.add_message(self.tenant_id, self.session_id, message_dict)
             if not success:
                  logger.error(f"Не удалось добавить сообщение в Redis для tenant='{self.tenant_id}', session='{self.session_id}'")
        except Exception as e:
             logger.error(f"Ошибка при преобразовании BaseMessage в dict или добавлении в Redis для tenant='{self.tenant_id}', session='{self.session_id}': {e}", exc_info=True)
    def clear(self) -> None:
        """Очистка истории в Redis."""
        logger.info(f"Очистка истории для tenant='{self.tenant_id}', session='{self.session_id}'")
        redis_history.clear_history(self.tenant_id, self.session_id)
def get_tenant_aware_history(composite_session_id: str) -> BaseChatMessageHistory:
    """
    Фабричная функция для создания экземпляра истории чата.
    Принимает составной ID вида 'tenant_id:user_id'.
    """
    logger.debug(f"get_tenant_aware_history вызван с composite_session_id: {composite_session_id}")
    try:
        parts = composite_session_id.split(":", 1)
        if len(parts) != 2:
            raise ValueError(f"Некорректный формат composite_session_id: {composite_session_id}. Ожидался 'tenant_id:user_id'")
        tenant_id, user_id = parts
        if not tenant_id or not user_id:
             raise ValueError("Извлеченные tenant_id или user_id пусты.")
        logger.info(f"Создание/получение истории для tenant_id='{tenant_id}', user_id='{user_id}'")
        return redis_history.TenantAwareRedisChatMessageHistory(tenant_id=tenant_id, session_id=user_id)
    except Exception as e:
        logger.error(f"Ошибка в get_tenant_aware_history для '{composite_session_id}': {e}", exc_info=True)
        raise ValueError(f"Не удалось создать историю чата: {e}")
from langchain_core.tools import Tool, StructuredTool

async def run_agent_like_chain(input_dict: Dict, config: RunnableConfig) -> str: # <--- ИЗМЕНЕНО на async def
    """
    Функция, имитирующая выполнение основного агента или цепочки.
    Принимает словарь с 'input' (вопрос пользователя) и RunnableConfig.
    Извлекает tenant_id и session_id (user_id) из config.
    Выполняет RAG-поиск, формирует промпт и вызывает LLM.
    Динамически создает инструменты для LLM.
    """
    question = input_dict.get("input")
    history_messages: List[BaseMessage] = input_dict.get("history", [])
    configurable = config.get("configurable", {})
    composite_session_id = configurable.get("session_id")
    if not composite_session_id:
        raise ValueError("Отсутствует 'session_id' (composite_session_id) в конфигурации Runnable.")
    try:
        tenant_id, user_id = composite_session_id.split(":", 1)
        if not tenant_id or not user_id:
            raise ValueError("tenant_id или user_id пусты после разделения composite_session_id.")
    except ValueError as e:
        raise ValueError(f"Ошибка разбора composite_session_id '{composite_session_id}': {e}")
    logger.info(f"run_agent_like_chain для Tenant: {tenant_id}, User: {user_id}, Вопрос: {question[:50]}...")
    if not CHROMA_CLIENT or not EMBEDDINGS_GIGA:
        logger.error(f"RAG компоненты (Chroma/Embeddings) не инициализированы.")
        return "Ошибка: Базовые компоненты RAG не готовы."
    if not tenant_id:
        logger.error("Tenant ID не найден в config.")
        return "Ошибка: Не удалось определить тенанта."
    bm25_retriever = BM25_RETRIEVERS_MAP.get(tenant_id)
    collection_name = f"{TENANT_COLLECTION_PREFIX}{tenant_id}"
    if not CHROMA_CLIENT or not EMBEDDINGS_GIGA:
        logger.error("Глобальный Chroma клиент или эмбеддинги не инициализированы.")
        return "Ошибка: Глобальные компоненты RAG не готовы."
    try:
        embeddings_wrapper = ChromaGigaEmbeddingsWrapper(EMBEDDINGS_GIGA)
    except ValueError as e:
         logger.error(f"Ошибка создания обертки эмбеддингов: {e}")
         return "Ошибка: Некорректный объект эмбеддингов."
    try:
        chroma_collection = CHROMA_CLIENT.get_collection(
            name=collection_name,
            embedding_function=embeddings_wrapper
        )
        chroma_vectorstore = Chroma(
            client=CHROMA_CLIENT,
            collection_name=collection_name,
            embedding_function=embeddings_wrapper,
        )
        chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": search_k_global})
        logger.info(f"Chroma ретривер для коллекции '{collection_name}' (tenant: {tenant_id}) получен.")
    except ValueError as e: 
        logger.error(f"Ошибка ChromaDB при получении коллекции или создании ретривера '{collection_name}' для тенанта {tenant_id}: {e}", exc_info=True)
        return f"Ошибка: Проблема совместимости с базой знаний ChromaDB для филиала '{tenant_id}'. {e}"
    except Exception as e:
        logger.error(f"Не удалось получить Chroma коллекцию или ретривер '{collection_name}' для тенанта {tenant_id}: {e}", exc_info=True)
        return f"Ошибка: Не удалось получить доступ к базе знаний для филиала '{tenant_id}'."
    if not bm25_retriever:
        logger.warning(f"BM25 ретривер для тенанта {tenant_id} не найден. Поиск будет только векторным.")
        final_retriever = chroma_retriever
    else:
        final_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )
        logger.info(f"Ensemble ретривер (BM25 + Chroma) создан для тенанта {tenant_id}.")
    rag_context = ""
    effective_rag_query = question
    try:
        rag_query_llm = chat_model.with_structured_output(RagQueryThought)
        rag_prompt_messages: List[BaseMessage] = [
            SystemMessage(content=(
                "Твоя задача - проанализировать последний запрос пользователя ('input') в контексте предыдущего диалога ('history'). "
                "Определи главную сущность (услуга, врач, филиал, категория, общая информация о клинике), о которой спрашивает пользователь, "
                "особенно если он использует ссылки на историю (номер пункта, местоимения 'он', 'она', 'это', 'они'). "
                "Затем сформулируй оптимальный, самодостаточный поисковый запрос ('best_rag_query') для векторной базы знаний, "
                "чтобы найти ОПИСАНИЕ или детали этой сущности. Используй полные названия. "
                "Пример: Если история содержит '1. Услуга А\n2. Услуга Б', а пользователь спрашивает 'расскажи о 2', "
                "то best_rag_query должен быть что-то вроде 'Описание услуги Б'. "
                "ВАЖНО: Если запрос пользователя является простым приветствием (например, 'Привет', 'Добрый день', 'Как дела?'), коротким общим вопросом, не требующим поиска информации, "
                "или очевидным согласием/просьбой продолжить предыдущий ответ ассистента (например, 'да', 'дальше', 'продолжай', 'еще', 'следующая страница'), "
                "то best_rag_query ДОЛЖЕН БЫТЬ ПУСТОЙ СТРОКОЙ. "
                "Также, если запрос очевидно должен быть обработан вызовом функции (например, запросы на списки чего-либо, цены, проверки наличия, сравнения, поиск по точным критериям), "
                "то best_rag_query ДОЛЖЕН БЫТЬ ПУСТОЙ СТРОКОЙ. "
                "Во всех остальных случаях (запросы на описание, общую информацию о клинике) сформулируй поисковый запрос. "
                "Поле 'analysis' должно содержать краткий анализ твоего решения, особенно пояснение, почему RAG-запрос нужен или не нужен."
            ))
        ]
        if history_messages:
             rag_prompt_messages.extend(history_messages)
        rag_prompt_messages.append(HumanMessage(content=question))
        logger.debug(f"[{tenant_id}:{user_id}] Вызов LLM для генерации RAG-запроса...")
        rag_thought_result = await rag_query_llm.ainvoke(rag_prompt_messages, config=config) # <--- ИЗМЕНЕНО на await .ainvoke
        if isinstance(rag_thought_result, RagQueryThought):
            rag_thought = rag_thought_result
            if rag_thought.best_rag_query and rag_thought.best_rag_query.strip():
                 effective_rag_query = rag_thought.best_rag_query
                 logger.info(f"[{tenant_id}:{user_id}] Сгенерирован RAG-запрос от LLM: '{effective_rag_query}' (Анализ: {rag_thought.analysis})")
            elif hasattr(rag_thought, 'best_rag_query') and not rag_thought.best_rag_query.strip():
                effective_rag_query = ""
                logger.info(f"[{tenant_id}:{user_id}] LLM указал, что RAG-запрос должен быть пустым (вероятно, это вызов функции). Анализ: {rag_thought.analysis}")
            else:
                effective_rag_query = ""
                logger.info(f"[{tenant_id}:{user_id}] LLM вернул RagQueryThought, но best_rag_query невалиден/пуст. RAG-запрос будет пустым. Анализ: {rag_thought.analysis if hasattr(rag_thought, 'analysis') else 'нет анализа'}")
        else:
            effective_rag_query = ""
            logger.warning(f"[{tenant_id}:{user_id}] LLM для генерации RAG-запроса вернул неожиданный тип: {type(rag_thought_result)}. RAG-запрос будет пустым.")
            if hasattr(rag_thought_result, 'analysis'):
                 logger.warning(f"Анализ от LLM (при неожиданном типе returnValue): {rag_thought_result.analysis}")
    except Exception as e:
        effective_rag_query = ""
        logger.warning(f"[{tenant_id}:{user_id}] Исключение при улучшении RAG-запроса: {e}. RAG-запрос будет пустым.", exc_info=True)
    try:
        if effective_rag_query and effective_rag_query.strip():
            logger.info(f"[{tenant_id}:{user_id}] Выполнение RAG-поиска с запросом: '{effective_rag_query[:100]}...'")
            relevant_docs = final_retriever.invoke(effective_rag_query, config=config)
            rag_context = format_docs(relevant_docs)
            logger.info(f"RAG: Найдено {len(relevant_docs)} док-в для запроса: '{effective_rag_query[:50]}...'. Контекст: {len(rag_context)} симв.")
        else:
            logger.info(f"[{tenant_id}:{user_id}] RAG-запрос пуст, RAG поиск пропущен.")
            rag_context = "Поиск по базе знаний не выполнялся, так как запрос не предполагает этого или является вызовом функции."
    except Exception as e:
        logger.error(f"Ошибка выполнения RAG поиска для тенанта {tenant_id} с запросом '{effective_rag_query}': {e}", exc_info=True)
        rag_context = "[Ошибка получения информации из базы знаний]"
    system_prompt = SYSTEM_PROMPT
    # Добавляем инфо о дате/времени в начало system_prompt
    tz = pytz.timezone('Europe/Moscow')
    now = datetime.now(tz)
    date_str = now.strftime('%d %B %Y')
    weekday_str = now.strftime('%A')
    time_str = now.strftime('%H:%M')
    datetime_info = f"Сегодня: {date_str}, {weekday_str}, текущее время: {time_str} (Europe/Moscow)"
    system_prompt = f"{datetime_info}\n\n" + system_prompt
    prompt_addition = None
    if tenant_config_manager:
        settings = tenant_config_manager.load_tenant_settings(tenant_id)
        prompt_addition = settings.get('prompt_addition')
        if prompt_addition:
            system_prompt += f"\n\n[Дополнительные инструкции от администратора филиала {tenant_id}]:\n{prompt_addition}"
            logger.info(f"Добавлено дополнение к промпту для тенанта {tenant_id}.")
    tenant_tools = []
    tenant_specific_docs: Optional[List[Document]] = TENANT_DOCUMENTS_MAP.get(tenant_id)
    if not tenant_specific_docs:
         logger.warning(f"Не найдены загруженные документы для тенанта {tenant_id} в tenant_documents_map_global. Инструменты, требующие данные, не будут созданы или могут работать некорректно.")
    logger.info(f"Создание динамических инструментов для тенанта {tenant_id}...")
    tool_func_to_schema_map = {
        find_employees_tool: FindEmployeesArgs,
        get_service_price_tool: GetServicePriceArgs,
        list_filials_tool: None,
        get_employee_services_tool: GetEmployeeServicesArgs,
        check_service_in_filial_tool: CheckServiceInFilialArgs,
        compare_service_price_in_filials_tool: CompareServicePriceInFilialsArgs,
        find_service_locations_tool: FindServiceLocationsArgs,
        find_specialists_by_service_or_category_and_filial_tool: FindSpecialistsByServiceOrCategoryAndFilialArgs,
        list_services_in_category_tool: ListServicesInCategoryArgs,
        list_services_in_filial_tool: ListServicesInFilialArgs,
        find_services_in_price_range_tool: FindServicesInPriceRangeArgs,
        list_all_categories_tool: ListAllCategoriesArgs,
        list_employee_filials_tool: ListEmployeeFilialsArgs,
    }
    def create_tool_wrapper(original_tool_func: callable, data_docs: Optional[List[Document]], raw_data: Optional[List[Dict]]):
        def actual_wrapper(*args, **kwargs) -> str:
            tool_name = original_tool_func.__name__
            logger.info(f"Вызов обертки для инструмента {tool_name} с args: {args}, kwargs: {kwargs}")
            
            # Устанавливаем данные клиники для текущего тенанта
            clinic_functions.set_clinic_data(raw_data if raw_data is not None else [])
            
            try:
                handler_class_name = ''.join(word.capitalize() for word in tool_name.replace('_tool', '').split('_'))
                HandlerClass = getattr(clinic_functions, handler_class_name, None)
                if not HandlerClass:
                    logger.error(f"Не найден класс-обработчик '{handler_class_name}' в clinic_functions для функции {tool_name}")
                    return f"Ошибка: Некорректная конфигурация инструмента {tool_name}."
                handler_instance = HandlerClass(**kwargs)
                # Универсальный вызов process без лишних аргументов
                if hasattr(handler_instance, 'process') and callable(getattr(handler_instance, 'process')):
                    return handler_instance.process()
                else:
                    logger.error(f"У обработчика {handler_class_name} отсутствует метод process.")
                    return f"Ошибка: У обработчика {handler_class_name} отсутствует метод process."
            except Exception as e:
                logger.error(f"{tool_name}: {e}", exc_info=True)
                return f"При выполнении инструмента {tool_name} произошла ошибка."
        return actual_wrapper
    for tool_function in TOOL_FUNCTIONS:
        tool_name = tool_function.__name__
        tool_description = tool_function.__doc__ or f"Инструмент {tool_name}"
        
        current_args_schema = None
        func_for_tool = None

        # Обработка новых инструментов (get_free_slots_tool, book_appointment_tool)
        if tool_name == "get_free_slots_tool":
            current_args_schema = GetFreeSlotsArgs
            func_for_tool = tool_function # Используется прямая функция из matrixai.py
        elif tool_name == "book_appointment_tool":
            current_args_schema = BookAppointmentArgs
            func_for_tool = tool_function # Используется прямая функция из matrixai.py
        elif tool_name == "get_employee_schedule_tool": # <--- НОВЫЙ ИНСТРУМЕНТ
            current_args_schema = GetEmployeeScheduleArgs
            func_for_tool = tool_function # Используется прямая функция из matrixai.py
        # Обработка старых инструментов, которые используют wrapper и schema_map
        elif tool_function in tool_func_to_schema_map:
            current_args_schema = tool_func_to_schema_map.get(tool_function)
            func_for_tool = create_tool_wrapper(tool_function, tenant_specific_docs, TENANT_RAW_DATA_MAP.get(tenant_id, []))
        # Обработка инструментов без аргументов (например, list_filials_tool уже покрыт выше)
        # или других старых инструментов, которые могли быть пропущены в schema_map, но требуют wrapper.
        else: 
            # Этот блок для инструментов, которые не являются get_free_slots/book_appointment
            # и не находятся в tool_func_to_schema_map (например, если у них нет аргументов
            # и они не были явно добавлены в map с None).
            # Все инструменты из TOOL_FUNCTIONS должны быть либо новыми, либо в schema_map.
            # Если инструмент не попал в предыдущие условия, это может быть ошибкой конфигурации.
            # Однако, list_filials_tool обрабатывается `elif tool_function in tool_func_to_schema_map`
            # т.к. он там есть с args_schema=None.
            # Для безопасности, если инструмент не опознан, логируем и пропускаем,
            # но лучшe чтобы все инструменты были явно обработаны.
            # На данный момент, предполагается, что все "старые" инструменты, требующие обертку,
            # имеют запись в tool_func_to_schema_map (даже если схема None).
            logger.warning(f"Инструмент '{tool_name}' не был классифицирован для создания (не новый и не в schema_map). Использование стандартной обертки и schema=None.")
            current_args_schema = None 
            func_for_tool = create_tool_wrapper(tool_function, tenant_specific_docs, TENANT_RAW_DATA_MAP.get(tenant_id, []))

        if func_for_tool is None: # Дополнительная проверка на случай, если func_for_tool не был назначен
            logger.error(f"Не удалось определить функцию для инструмента '{tool_name}'. Пропуск создания инструмента.")
            continue
            
        langchain_tool = StructuredTool.from_function(
            func=func_for_tool,
            name=tool_name,
            description=tool_description,
            args_schema=current_args_schema
        )
        tenant_tools.append(langchain_tool)
        logger.debug(f"Инструмент '{tool_name}' создан для тенанта {tenant_id}. Schema: {current_args_schema.__name__ if current_args_schema else 'None'}")
    messages_for_llm = []
    messages_for_llm.append(SystemMessage(content=system_prompt))
    messages_for_llm.extend(history_messages)
    rag_query_display = effective_rag_query if effective_rag_query and effective_rag_query.strip() else "не выполнялся"
    rag_context_block = f"\n\n[Информация из базы знаний (поисковый запрос: '{rag_query_display}')]:\n{rag_context}\n[/Информация из базы знаний]"
    messages_for_llm.append(HumanMessage(content=question + rag_context_block))
    llm_with_tools = chat_model.bind_tools(tenant_tools)
    logger.info(f"{chat_model.model_name} {len(tenant_tools)} {tenant_id}...")
    try:
        ai_response_message: AIMessage = await llm_with_tools.ainvoke(messages_for_llm, config=config) # <--- ИЗМЕНЕНО на await .ainvoke
        tool_calls = ai_response_message.tool_calls
        if tool_calls:
             logger.info(f"{[tc['name'] for tc in tool_calls]}")
             tool_outputs = []
             for tool_call in tool_calls:
                  tool_name = tool_call["name"]
                  tool_args = tool_call["args"]
                  logger.info(f"{tool_name} {tool_args}")
                  found_tool = next((t for t in tenant_tools if t.name == tool_name), None)
                  if found_tool:
                      try:
                          current_tool_args_for_invoke = tool_args

                          expects_args = True
                          if not found_tool.args_schema or not found_tool.args_schema.model_fields:
                              expects_args = False

                          if not expects_args:
                              if tool_args is not None and tool_args != {}: # Логируем, если LLM прислал что-то для инструмента без аргументов
                                  logger.info(f"Инструмент {tool_name} не ожидает аргументов, но LLM передал: {tool_args}. Используем {{}}.")
                              current_tool_args_for_invoke = {}
                          elif isinstance(tool_args, dict) and tool_args.get('args') is None and len(tool_args) == 1 and expects_args:
                              # Этот случай (LLM вернул {'args': None} для инструмента, ожидающего аргументы) приведет к ошибке Pydantic.
                              # Оставляем current_tool_args_for_invoke как есть ({'args': None}), чтобы Pydantic обработал.
                              logger.warning(f"LLM передал {{'args': None}} для инструмента {tool_name}, который ожидает аргументы. Pydantic, вероятно, выдаст ошибку.")
                          # В остальных случаях (expects_args is True и tool_args это корректный словарь или None, который Pydantic обработает)
                          # current_tool_args_for_invoke остается равным tool_args.

                          # --- Начало модификации для get_free_slots_tool, book_appointment_tool, get_employee_schedule_tool ---
                          if tool_name in ["get_free_slots_tool", "book_appointment_tool", "get_employee_schedule_tool"]:
                              if not isinstance(current_tool_args_for_invoke, dict):
                                  logger.error(f"Ожидался dict для tool_args у {tool_name}, получено: {type(current_tool_args_for_invoke)}. Формирование вызова невозможно.")
                                  raise ValueError(f"Аргументы для {tool_name} должны быть словарем, получено {type(current_tool_args_for_invoke)}")

                              current_tenant_id = configurable.get("tenant_id")
                              current_client_api_token = configurable.get("client_api_token")

                              if not current_tenant_id:
                                  logger.error(f"Критическая ошибка: tenant_id не найден в 'configurable' ({configurable}) при попытке подготовить вызов {tool_name}.")
                                  raise ValueError(f"tenant_id отсутствует в конфигурации для обязательного использования в инструменте {tool_name}")
                              
                              current_tool_args_for_invoke['tenant_id'] = current_tenant_id
                              if current_client_api_token:
                                  current_tool_args_for_invoke['api_token'] = current_client_api_token
                              elif 'api_token' not in current_tool_args_for_invoke:
                                   current_tool_args_for_invoke['api_token'] = None
                              
                              logger.info(f"Обновленные аргументы для '{tool_name}' после добавления tenant_id/api_token: {current_tool_args_for_invoke}")
                          # --- Конец модификации ---

                          raw_output = await found_tool.ainvoke(current_tool_args_for_invoke, config=config) # Используем current_tool_args_for_invoke
                          
                          if inspect.iscoroutine(raw_output):
                              logger.warning(f"Инструмент '{tool_name}' .ainvoke вернул короутину. Ожидаем ее явно.")
                              output = await raw_output
                          else:
                              output = raw_output
                              
                          output_str = str(output)
                          tool_outputs.append(ToolMessage(content=output_str, tool_call_id=tool_call["id"]))
                          logger.info(f"Результат вызова инструмента '{tool_name}': {output_str[:100]}...")
                      except Exception as e:
                           error_message = f"Ошибка при вызове инструмента '{tool_name}': {type(e).__name__} - {str(e)}"
                           logger.error(f"Ошибка вызова инструмента '{tool_name}' с аргументами {current_tool_args_for_invoke}: {e}", exc_info=True)
                           tool_outputs.append(ToolMessage(content=error_message, tool_call_id=tool_call["id"]))
                  else:
                      logger.error(f"Инструмент '{tool_name}' не найден среди доступных для тенанта {tenant_id}. Доступные инструменты: {[t.name for t in tenant_tools]}.")
                      tool_outputs.append(ToolMessage(content=f"Ошибка: Инструмент {tool_name} не найден.", tool_call_id=tool_call["id"]))
             messages_for_llm.append(ai_response_message)
             messages_for_llm.extend(tool_outputs)
             logger.info("")
             final_response_message = await llm_with_tools.ainvoke(messages_for_llm, config=config) # <--- ИЗМЕНЕНО на await .ainvoke
             final_answer = final_response_message.content
             logger.info(f"[{tenant_id}:{user_id}] Итоговый ответ (первые 100 симв): {final_answer[:100]}...")
             return final_answer
        else:
             logger.info(f"[{tenant_id}:{user_id}] Итоговый ответ LLM без вызова инструментов (первые 100 симв): {ai_response_message.content[:100]}...")
             return ai_response_message.content
    except Exception as e:
        logger.error(f"Критическая ошибка в run_agent_like_chain для тенанта {tenant_id}, пользователя {user_id}: {e}", exc_info=True)
        return "Произошла ошибка при обработке вашего запроса."

# --- Начало: Асинхронный триггер для переиндексации данных одного тенанта ---
async def trigger_reindex_tenant_async(tenant_id: str) -> bool:
    """
    Асинхронно запускает переиндексацию данных для указанного тенанта.
    Использует глобальные RAG компоненты и конфигурации.
    """
    logger.info(f"[Async Trigger] Запрос на переиндексацию для тенанта: {tenant_id}")
    if not CHROMA_CLIENT or not EMBEDDINGS_GIGA:
        logger.error(f"[Async Trigger] Глобальные RAG компоненты (Chroma/Embeddings) не инициализированы. Переиндексация для {tenant_id} отменена.")
        return False

    global BM25_RETRIEVERS_MAP, TENANT_DOCUMENTS_MAP, TENANT_RAW_DATA_MAP, SERVICE_DETAILS_MAP_GLOBAL
    global search_k_global

    data_dir_base = os.getenv("BASE_DATA_DIR", "base")
    chunk_size_cfg = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap_cfg = int(os.getenv("CHUNK_OVERLAP", 200))

    try:
        import asyncio 

        success = await asyncio.to_thread(
            rag_setup.reindex_tenant_specific_data,
            tenant_id=tenant_id,
            chroma_client=CHROMA_CLIENT,
            embeddings_object=EMBEDDINGS_GIGA,
            bm25_retrievers_map=BM25_RETRIEVERS_MAP,
            tenant_documents_map=TENANT_DOCUMENTS_MAP,
            tenant_raw_data_map=TENANT_RAW_DATA_MAP,
            service_details_map=SERVICE_DETAILS_MAP_GLOBAL,
            base_data_dir=data_dir_base,
            chunk_size=chunk_size_cfg,
            chunk_overlap=chunk_overlap_cfg,
            search_k=search_k_global
        )
        if success:
            logger.info(f"[Async Trigger] Переиндексация для тенанта {tenant_id} успешно запущена и завершена.")
        else:
            logger.error(f"[Async Trigger] Переиндексация для тенанта {tenant_id} не удалась.")
        return success
    except Exception as e:
        logger.error(f"[Async Trigger] Исключение во время запуска переиндексации для тенанта {tenant_id}: {e}", exc_info=True)
        return False

agent_runnable = RunnableLambda(run_agent_like_chain)
agent_with_history = RunnableWithMessageHistory(
    agent_runnable,
    get_tenant_aware_history,
    input_messages_key="input",
    history_messages_key="history",
)
logger.info("Основной Runnable агент с историей и RAG создан.")
class ChromaGigaEmbeddingsWrapper(EmbeddingFunction):
    def __init__(self, gigachat_embeddings: GigaChatEmbeddings):
        self._gigachat_embeddings = gigachat_embeddings
        if not hasattr(self._gigachat_embeddings, 'embed_documents'):
             raise ValueError("Предоставленный объект эмбеддингов не имеет метода 'embed_documents'")
        if not hasattr(self._gigachat_embeddings, 'embed_query'):
             raise ValueError("Предоставленный объект эмбеддингов не имеет метода 'embed_query'")
    def __call__(self, input: Documents) -> Embeddings:         
        embeddings = self._gigachat_embeddings.embed_documents(input)
        return embeddings
    def embed_query(self, text: str) -> List[float]:
        return self._gigachat_embeddings.embed_query(text)

