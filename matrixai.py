# matrixai.py

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
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
from clinic_index import build_indexes_for_tenant, get_id_by_name, get_category_id_by_service_id # <--- ДОБАВЛЕНО get_category_id_by_service_id
import pytz
from datetime import datetime
import inspect
from langchain_core.runnables import RunnableLambda, RunnableConfig # <--- Убрали get_config_value

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Импорт анализатора завершенности сообщений
try:
    from message_completeness_analyzer import (
        analyze_message,
        analyze_message_with_accumulation,
        should_wait_for_completion,
        initialize_analyzer,
        MessageAnalysis,
        CompletenessStatus,
        clear_accumulator
    )
    MESSAGE_ANALYZER_AVAILABLE = True
    logger.info("Анализатор завершенности сообщений успешно импортирован")
except ImportError as e:
    logger.warning(f"Анализатор завершенности сообщений недоступен: {e}")
    MESSAGE_ANALYZER_AVAILABLE = False
    analyze_message = None
    analyze_message_with_accumulation = None
    should_wait_for_completion = None
    initialize_analyzer = None
    clear_accumulator = None

GIGACHAT_CREDENTIALS = ""
JSON_DATA_PATH = os.environ.get("JSON_DATA_PATH", "base/cleaned_data.json")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db_clinic_giga")
GIGA_EMBEDDING_MODEL = os.environ.get("GIGA_EMBEDDING_MODEL", "EmbeddingsGigaR")
GIGA_SCOPE = os.environ.get("GIGA_SCOPE", "GIGACHAT_API_PERS")
GIGA_VERIFY_SSL = os.getenv("GIGA_VERIFY_SSL", "False").lower() == "true"
TENANT_COLLECTION_PREFIX = "tenant_" 
try:
    chat_model = ChatOpenAI(
        model="gpt-4.1",
        max_tokens=16384, 
        api_key=""
    )
    logger.info("Чат модель OpenAI (gpt-4.1) инициализирована.")
except Exception as e:
    logger.critical(f"Ошибка инициализации модели OpenAI Chat: {e}", exc_info=True)
    exit()
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

# Инициализация анализатора завершенности сообщений
def initialize_message_analyzer():
    """Инициализирует анализатор завершенности сообщений"""
    if MESSAGE_ANALYZER_AVAILABLE and initialize_analyzer:
        try:
            # Используем тот же ключ, что и для основной модели OpenAI
            openai_api_key = 
            # Инициализируем анализатор с o3-mini моделью
            initialize_analyzer(openai_api_key, model_name="o3-mini")
            logger.info("Анализатор завершенности сообщений успешно инициализирован")
            return True
        except Exception as e:
            logger.error(f"Ошибка инициализации анализатора завершенности: {e}")
            return False
    else:
        logger.warning("Анализатор завершенности сообщений недоступен")
        return False

# Инициализируем анализатор
ANALYZER_INITIALIZED = initialize_message_analyzer()

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
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника для фильтрации. В комбинации с filial_name покажет услуги конкретного сотрудника в филиале. С service_name покажет филиалы, где сотрудник оказывает услугу.")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название КОНКРЕТНОЙ услуги для фильтрации. В комбинации с filial_name покажет сотрудников, оказывающих услугу в филиале. С employee_name покажет филиалы, где сотрудник оказывает услугу.")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала. Самостоятельно - покажет ВСЕХ сотрудников филиала. С employee_name - услуги сотрудника в филиале. С service_name - сотрудников, оказывающих услугу в филиале.")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=15, description="Количество сотрудников на странице для пагинации")
def find_employees_tool(employee_name: Optional[str] = None, service_name: Optional[str] = None, filial_name: Optional[str] = None) -> str:
    """Унифицированный поиск сотрудников клиники с 6 режимами работы:
    1. Только filial_name → Все сотрудники в филиале
    2. Только service_name → Все сотрудники, оказывающие услугу
    3. Только employee_name → Все услуги сотрудника
    4. filial_name + employee_name → Услуги сотрудника в конкретном филиале
    5. filial_name + service_name → Сотрудники, оказывающие услугу в филиале
    6. service_name + employee_name → Филиалы, где сотрудник оказывает услугу
    
    Поддерживает пагинацию: 'page_number' (по умолчанию 1) и 'page_size' (по умолчанию 15)."""
    handler = clinic_functions.FindEmployees(employee_name=employee_name, service_name=service_name, filial_name=filial_name)
    return handler.process()
class GetServicePriceArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги (например, 'Soprano Пальцы для женщин')")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала (например, 'Москва-сити'), если нужно уточнить цену в конкретном месте")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос цены происходит в процессе записи на прием")
def get_service_price_tool(service_name: str, filial_name: Optional[str] = None, in_booking_process: bool = False) -> str:
    """Возвращает цену на КОНКРЕТНУЮ услугу клиники."""
    handler = clinic_functions.GetServicePrice(service_name=service_name, filial_name=filial_name, in_booking_process=in_booking_process)
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
    """Возвращает список услуг КОНКРЕТНОГО сотрудника. Поддерживает пагинацию: используйте 'page_number' (по умолчанию 1) и 'page_size' (по умолчанию 20), если ожидается много результатов или пользователь просит показать все/дальше."""
    handler = clinic_functions.GetEmployeeServices(employee_name=employee_name)
    return handler.process()
class CheckServiceInFilialArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос происходит в процессе записи на прием")
def check_service_in_filial_tool(service_name: str, filial_name: str, in_booking_process: bool = False) -> str:
    """Проверяет, доступна ли КОНКРЕТНАЯ услуга в КОНКРЕТНОМ филиале."""
    handler = clinic_functions.CheckServiceInFilial(service_name=service_name, filial_name=filial_name, in_booking_process=in_booking_process)
    return handler.process()
class CompareServicePriceInFilialsArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(min_length=2, description="Список из ДВУХ или БОЛЕЕ названий филиалов")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос происходит в процессе записи на прием")
def compare_service_price_in_filials_tool(service_name: str, filial_names: List[str], in_booking_process: bool = False) -> str:
    """Сравнивает цену КОНКРЕТНОЙ услуги в НЕСКОЛЬКИХ филиалах."""
    handler = clinic_functions.CompareServicePriceInFilials(service_name=service_name, filial_names=filial_names, in_booking_process=in_booking_process)
    return handler.process()
class FindServiceLocationsArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    in_booking_process: bool = Field(default=False, description="Флаг, указывающий, что запрос происходит в процессе записи на прием")
def find_service_locations_tool(service_name: str, in_booking_process: bool = False) -> str:
    """Ищет все филиалы, где доступна КОНКРЕТНАЯ услуга."""
    handler = clinic_functions.FindServiceLocations(service_name=service_name, in_booking_process=in_booking_process)
    return handler.process()
class FindSpecialistsByServiceOrCategoryAndFilialArgs(BaseModel):
    query_term: str = Field(description="Название услуги ИЛИ категории")
    filial_name: str = Field(description="Точное название филиала")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=15, description="Количество специалистов на странице для пагинации")
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")
async def find_specialists_by_service_or_category_and_filial_tool(**kwargs_from_llm) -> str:
    """Ищет СПЕЦИАЛИСТОВ по УСЛУГЕ/КАТЕГОРИИ в КОНКРЕТНОМ филиале. Поддерживает пагинацию: используйте 'page_number' (по умолчанию 1) и 'page_size' (по умолчанию 15), если ожидается много результатов или пользователь просит показать все/дальше."""
    logger.info(f"ENTERING find_specialists_by_service_or_category_and_filial_tool. Raw kwargs_from_llm: {kwargs_from_llm}")

    try:
        validated_args = FindSpecialistsByServiceOrCategoryAndFilialArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для find_specialists_by_service_or_category_and_filial_tool: {e}. Аргументы: {kwargs_from_llm}", exc_info=True)
        return f"Ошибка: неверные или отсутствующие аргументы для поиска специалистов: {e}"
    
    tenant_id_from_kwargs = validated_args.tenant_id
    api_token_from_kwargs = validated_args.api_token

    logger.info(f"find_specialists_by_service_or_category_and_filial_tool - Retrieved from validated_args - tenant_id: {tenant_id_from_kwargs}")
    logger.info(f"find_specialists_by_service_or_category_and_filial_tool - Retrieved from validated_args - api_token: {'******' if api_token_from_kwargs else None}")

    if not tenant_id_from_kwargs:
        logger.error(f"CRITICAL (find_specialists_by_service_or_category_and_filial_tool): tenant_id (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: ID тенанта не был предоставлен системе для вызова find_specialists_by_service_or_category_and_filial_tool."

    if not api_token_from_kwargs:
        logger.error(f"CRITICAL (find_specialists_by_service_or_category_and_filial_tool): api_token (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: API токен не был предоставлен системе для вызова find_specialists_by_service_or_category_and_filial_tool."

    handler = clinic_functions.FindSpecialistsByServiceOrCategoryAndFilial(
        query_term=validated_args.query_term.lower(), 
        filial_name=validated_args.filial_name.lower(),
        page_number=validated_args.page_number,
        page_size=validated_args.page_size
    )
    return await handler.process(tenant_id_from_kwargs, api_token_from_kwargs)
class ListServicesInCategoryArgs(BaseModel):
    category_name: str = Field(description="Точное название категории")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=20, description="Количество услуг на странице для пагинации")
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")
async def list_services_in_category_tool(**kwargs_from_llm) -> str:
    """Возвращает список КОНКРЕТНЫХ услуг в указанной КАТЕГОРИИ. Поддерживает пагинацию: используйте 'page_number' (по умолчанию 1) и 'page_size' (по умолчанию 20), если ожидается много результатов или пользователь просит показать все/дальше."""
    logger.info(f"ENTERING list_services_in_category_tool. Raw kwargs_from_llm: {kwargs_from_llm}")

    try:
        validated_args = ListServicesInCategoryArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для list_services_in_category_tool: {e}. Аргументы: {kwargs_from_llm}", exc_info=True)
        return f"Ошибка: неверные или отсутствующие аргументы для получения списка услуг: {e}"
    
    tenant_id_from_kwargs = validated_args.tenant_id
    api_token_from_kwargs = validated_args.api_token

    logger.info(f"list_services_in_category_tool - Retrieved from validated_args - tenant_id: {tenant_id_from_kwargs}")
    logger.info(f"list_services_in_category_tool - Retrieved from validated_args - api_token: {'******' if api_token_from_kwargs else None}")

    if not tenant_id_from_kwargs:
        logger.error(f"CRITICAL (list_services_in_category_tool): tenant_id (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: ID тенанта не был предоставлен системе для вызова list_services_in_category_tool."

    if not api_token_from_kwargs:
        logger.error(f"CRITICAL (list_services_in_category_tool): api_token (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: API токен не был предоставлен системе для вызова list_services_in_category_tool."

    handler = clinic_functions.ListServicesInCategory(
        category_name=validated_args.category_name,
        page_number=validated_args.page_number,
        page_size=validated_args.page_size
    )
    return await handler.process(tenant_id_from_kwargs, api_token_from_kwargs)
class ListServicesInFilialArgs(BaseModel):
    filial_name: str = Field(description="Точное название филиала")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=30, description="Количество услуг на странице для пагинации (учитывайте, что вывод также содержит заголовки категорий)")
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")
async def list_services_in_filial_tool(**kwargs_from_llm) -> str:
    """Возвращает ПОЛНЫЙ список УНИКАЛЬНЫХ услуг в КОНКРЕТНОМ филиале. Поддерживает пагинацию: используйте 'page_number' (по умолчанию 1) и 'page_size' (по умолчанию 30), если ожидается много результатов или пользователь просит показать все/дальше. Учитывайте, что вывод page_size также включает заголовки категорий."""
    logger.info(f"ENTERING list_services_in_filial_tool. Raw kwargs_from_llm: {kwargs_from_llm}")

    try:
        validated_args = ListServicesInFilialArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для list_services_in_filial_tool: {e}. Аргументы: {kwargs_from_llm}", exc_info=True)
        return f"Ошибка: неверные или отсутствующие аргументы для получения списка услуг в филиале: {e}"
    
    tenant_id_from_kwargs = validated_args.tenant_id
    api_token_from_kwargs = validated_args.api_token

    logger.info(f"list_services_in_filial_tool - Retrieved from validated_args - tenant_id: {tenant_id_from_kwargs}")
    logger.info(f"list_services_in_filial_tool - Retrieved from validated_args - api_token: {'******' if api_token_from_kwargs else None}")

    if not tenant_id_from_kwargs:
        logger.error(f"CRITICAL (list_services_in_filial_tool): tenant_id (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: ID тенанта не был предоставлен системе для вызова list_services_in_filial_tool."

    if not api_token_from_kwargs:
        logger.error(f"CRITICAL (list_services_in_filial_tool): api_token (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: API токен не был предоставлен системе для вызова list_services_in_filial_tool."

    handler = clinic_functions.ListServicesInFilial(
        filial_name=validated_args.filial_name,
        page_number=validated_args.page_number,
        page_size=validated_args.page_size
    )
    return await handler.process(tenant_id_from_kwargs, api_token_from_kwargs)
class FindServicesInPriceRangeArgs(BaseModel):
    min_price: float = Field(description="Минимальная цена")
    max_price: float = Field(description="Максимальная цена")
    category_name: Optional[str] = Field(default=None, description="Опционально: категория")
    filial_name: Optional[str] = Field(default=None, description="Опционально: филиал")
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=20, description="Количество услуг на странице для пагинации")
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")
async def find_services_in_price_range_tool(**kwargs_from_llm) -> str:
    """Ищет услуги в ЗАДАННОМ ЦЕНОВОМ ДИАПАЗОНЕ. Поддерживает пагинацию: используйте 'page_number' (по умолчанию 1) и 'page_size' (по умолчанию 20), если ожидается много результатов или пользователь просит показать все/дальше."""
    try:
        validated_args = FindServicesInPriceRangeArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для find_services_in_price_range_tool: {e}")
        return f"Ошибка: неверные или отсутствующие аргументы для поиска услуг в диапазоне цен: {e}"
    
    tenant_id_from_kwargs = validated_args.tenant_id
    api_token_from_kwargs = validated_args.api_token

    logger.info(f"find_services_in_price_range_tool - Retrieved from validated_args - tenant_id: {tenant_id_from_kwargs}")
    logger.info(f"find_services_in_price_range_tool - Retrieved from validated_args - api_token: {'******' if api_token_from_kwargs else None}")

    if not tenant_id_from_kwargs:
        logger.error(f"CRITICAL (find_services_in_price_range_tool): tenant_id (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: ID тенанта не был предоставлен системе для вызова find_services_in_price_range_tool."

    if not api_token_from_kwargs:
        logger.error(f"CRITICAL (find_services_in_price_range_tool): api_token (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: API токен не был предоставлен системе для вызова find_services_in_price_range_tool."

    handler = clinic_functions.FindServicesInPriceRange(
        min_price=validated_args.min_price, 
        max_price=validated_args.max_price, 
        category_name=validated_args.category_name, 
        filial_name=validated_args.filial_name,
        page_number=validated_args.page_number,
        page_size=validated_args.page_size
    )
    return await handler.process(tenant_id_from_kwargs, api_token_from_kwargs)

class ListCategoriesArgs(BaseModel):
    page_number: Optional[int] = Field(default=1, description="Номер страницы для пагинации (начиная с 1)")
    page_size: Optional[int] = Field(default=30, description="Количество категорий на странице для пагинации")
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")

async def list_all_categories_tool(**kwargs_from_llm) -> str:
    """Возвращает список ВСЕХ категорий услуг. Поддерживает пагинацию: используйте 'page_number' (по умолчанию 1) и 'page_size' (по умолчанию 30), если ожидается много результатов или пользователь просит показать все/дальше."""
    try:
        validated_args = ListCategoriesArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для list_all_categories_tool: {e}")
        return f"Ошибка: неверные или отсутствующие аргументы для получения списка категорий: {e}"
    
    tenant_id_from_kwargs = validated_args.tenant_id
    api_token_from_kwargs = validated_args.api_token

    logger.info(f"list_all_categories_tool - Retrieved from validated_args - tenant_id: {tenant_id_from_kwargs}")
    logger.info(f"list_all_categories_tool - Retrieved from validated_args - api_token: {'******' if api_token_from_kwargs else None}")

    if not tenant_id_from_kwargs:
        logger.error(f"CRITICAL (list_all_categories_tool): tenant_id (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: ID тенанта не был предоставлен системе для вызова list_all_categories_tool."

    if not api_token_from_kwargs:
        logger.error(f"CRITICAL (list_all_categories_tool): api_token (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: API токен не был предоставлен системе для вызова list_all_categories_tool."

    handler = clinic_functions.ListCategories(
        page_number=validated_args.page_number,
        page_size=validated_args.page_size
    )
    return await handler.process(tenant_id_from_kwargs, api_token_from_kwargs)
class ListEmployeeFilialsArgs(BaseModel):
    employee_name: str = Field(description="Точное или близкое ФИО сотрудника")
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")

async def list_employee_filials_tool(**kwargs_from_llm) -> str:
    """ОБЯЗАТЕЛЬНО ВЫЗЫВАЙ для получения ПОЛНОГО списка ВСЕХ филиалов КОНКРЕТНОГО сотрудника."""
    try:
        validated_args = ListEmployeeFilialsArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для list_employee_filials_tool: {e}")
        return f"Ошибка: неверные или отсутствующие аргументы для получения филиалов сотрудника: {e}"
    
    tenant_id_from_kwargs = validated_args.tenant_id
    api_token_from_kwargs = validated_args.api_token

    logger.info(f"list_employee_filials_tool - Вызов для: {validated_args.employee_name}")
    logger.info(f"list_employee_filials_tool - Retrieved from validated_args - tenant_id: {tenant_id_from_kwargs}")
    logger.info(f"list_employee_filials_tool - Retrieved from validated_args - api_token: {'******' if api_token_from_kwargs else None}")

    if not tenant_id_from_kwargs:
        logger.error(f"CRITICAL (list_employee_filials_tool): tenant_id (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: ID тенанта не был предоставлен системе для вызова list_employee_filials_tool."

    if not api_token_from_kwargs:
        logger.error(f"CRITICAL (list_employee_filials_tool): api_token (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: API токен не был предоставлен системе для вызова list_employee_filials_tool."

    handler = clinic_functions.ListEmployeeFilials(employee_name=validated_args.employee_name)
    return await handler.process(tenant_id_from_kwargs, api_token_from_kwargs)

class GetFreeSlotsArgs(BaseModel):
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    employee_name: str = Field(description="ФИО сотрудника (точно или частично)")
    service_names: List[str] = Field(description="Список названий услуг (точно)")
    date_time: str = Field(description="Дата для начала поиска слотов (формат YYYY-MM-DD). ВАЖНО: Система автоматически найдет ближайшие доступные слоты, начиная с этой даты. Если на указанную дату нет слотов, поиск продолжится на +7, +14, +21 дней и т.д. (максимум 7 итераций). Не нужно указывать несколько дат - система сама найдет оптимальный период.")
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

# Pydantic модель для нового инструмента записи (AI Payload)
class ServiceDetailItemFromLLM(BaseModel): # <--- НОВАЯ МОДЕЛЬ
    rowNumber: int = Field(description="Порядковый номер услуги в записи, начиная с 1.")
    serviceName: str = Field(description="ТОЧНОЕ название услуги для поиска ее serviceId.")
    categoryName: Optional[str] = Field(default=None, description="НАЗВАНИЕ КАТЕГОРИИ. По возможности передайте, если известно или было явно уточнено. Система в первую очередь попытается определить категорию автоматически по ID услуги, но это поле может служить уточнением или запасным вариантом.")
    countService: int = Field(default=1, description="Количество данной услуги.")
    complexServiceId: Optional[str] = Field(default=None, description="ID комплексной услуги, если эта услуга является частью комплекса.")
    price: float = Field(description="Цена ИМЕННО ЭТОЙ УСЛУГИ (за единицу, если countService > 1). ВАЖНО: Ты ОБЯЗАН получить эту цену, вызвав инструмент get_service_price_tool с параметром in_booking_process=True ПЕРЕД формированием этого объекта. Не придумывай цену.")
    durationService: int = Field(description="Длительность ИМЕННО ЭТОЙ УСЛУГИ в минутах. LLM должен определить это значение (например, из описания услуги или общих знаний).")

class BookAppointmentAIPayloadArgs(BaseModel):
    # client_phone_number будет извлечен из конфигурации автоматически (Устаревшее утверждение, теперь он часть модели)
    filial_name: str = Field(description="ТОЧНОЕ название филиала для записи.")
    date_of_record: str = Field(description="Дата записи в формате YYYY-MM-DD.")
    start_time: str = Field(description="Время начала записи в формате HH:MM.")
    end_time: str = Field(description="Время окончания записи в формате HH:MM.")
    duration_of_time: int = Field(description="ОБЩАЯ длительность всей записи в минутах.")
    employee_name: str = Field(description="ТОЧНОЕ ФИО сотрудника для записи.")
    services_details: List[ServiceDetailItemFromLLM] = Field( # <--- ИЗМЕНЕНО List[Dict[str, Any]] на List[ServiceDetailItemFromLLM]
        min_length=1, 
        description=(
            "Список услуг для записи. ОБЯЗАТЕЛЬНО должен содержать хотя бы одну услугу. Каждая услуга - это ОБЪЕКТ со следующими полями: "
            "\'rowNumber\' (int, порядковый номер), "
            "\'serviceName\' (str, ТОЧНОЕ название услуги для поиска ее serviceId, цены и длительности), "
            "\'categoryName\' (Optional[str], НАЗВАНИЕ КАТЕГОРИИ. По возможности передайте его, если оно известно или было явно уточнено. Система в первую очередь попытается определить категорию автоматически по ID услуги, но это поле может служить уточнением или запасным вариантом), "
            "\'countService\' (int, количество), "
            "\'complexServiceId\' (Optional[str], ID комплексной услуги, если эта услуга является частью комплекса), "
            "\'price\' (float, цена ИМЕННО ЭТОЙ услуги, ОБЯЗАТЕЛЬНО полученная через get_service_price_tool с параметром in_booking_process=True), " # <--- ИЗМЕНЕНО В ОПИСАНИЕ
            "\'durationService\' (int, длительность ИМЕННО ЭТОЙ услуги в минутах)." # <--- ДОБАВЛЕНО В ОПИСАНИЕ
        )
    )
    total_price: float = Field(description="ОБЩАЯ цена всей записи.")
    lang_id: str = Field(default="ru", description="Язык ответа (например, 'ru').")
    color_code_record: Optional[str] = Field(default=None, description="Код цвета для записи (опционально).")
    traffic_channel: Optional[int] = Field(default=None, description="ID канала трафика (опционально).")
    traffic_channel_id: Optional[str] = Field(default=None, description="Строковый ID канала трафика (опционально).")

    # Системные поля, которые будут добавлены в run_agent_like_chain
    tenant_id: Optional[str] = Field(default=None, description="[СИСТЕМНОЕ ПОЛЕ] ID тенанта. Не заполняется LLM.")
    api_token: Optional[str] = Field(default=None, description="[СИСТЕМНОЕ ПОЛЕ] API токен. Не заполняется LLM.")
    client_phone_number: str = Field(description="[СИСТЕМНОЕ ПОЛЕ] Телефон клиента. Не заполняется LLM, должен быть добавлен системой.")

from clinic_functions import BookAppointmentAIPayload

# --- ДОБАВЛЕНО: Схема для инструментов без аргументов ---

class RescheduleAppointmentArgs(BaseModel):
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    record_id: str = Field(description="ID переносимой записи")
    date_of_record: str = Field(description="Новая дата (YYYY-MM-DD)")
    start_time: str = Field(description="Новое время начала (HH:MM)")
    end_time: str = Field(description="Новое время окончания (HH:MM)")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")

class UpdateRecordTimeArgs(BaseModel):
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники) - будет установлен автоматически")
    record_id: str = Field(description="ID переносимой записи")
    date_of_record: str = Field(description="Новая дата (YYYY-MM-DD)")
    start_time: str = Field(description="Новое время начала (HH:MM)")
    end_time: str = Field(description="Новое время окончания (HH:MM)")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")

class CancelAppointmentArgs(BaseModel):
    tenant_id: Optional[str] = Field(default=None, description="ID тенанта (клиники)")
    record_id: str = Field(description="ID отменяемой записи")
    chain_id: str = Field(description="Chain ID, которому принадлежит запись")
    canceling_reason: Optional[str] = Field(default=None, description="ID причины отмены (если None, ассистент выберет автоматически)")
    api_token: Optional[str] = Field(default=None, description="Bearer-токен для авторизации (client_api_token)")

class NoArgsSchema(BaseModel):
    pass
# --- КОНЕЦ ДОБАВЛЕНИЯ ---

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
    ListCategoriesArgs,
    ListEmployeeFilialsArgs,
    GetFreeSlotsArgs,
    BookAppointmentArgs, 
    BookAppointmentAIPayloadArgs,
    BookAppointmentAIPayload,
]
logger.info(f"Определено {len(TOOL_CLASSES)} Pydantic классов для аргументов инструментов.")

async def get_free_slots_tool(**kwargs_from_llm) -> str:
    logger.info(f"ENTERING get_free_slots_tool. Raw kwargs_from_llm: {kwargs_from_llm}")

    try:
        validated_args = GetFreeSlotsArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для get_free_slots_tool: {e}. Аргументы: {kwargs_from_llm}", exc_info=True)
        return f"Ошибка: неверные или отсутствующие аргументы для получения слотов: {e}"
    
    # Базовый tenant_id для API запросов
    base_tenant_id = validated_args.tenant_id
    api_token_from_kwargs = validated_args.api_token
    
    # Получаем chain_id из kwargs (не из схемы Pydantic)
    chain_id = kwargs_from_llm.get('chain_id')
    
    # Комбинированный tenant_id для поиска в локальных данных (с chain_id)
    combined_tenant_id = f"{base_tenant_id}_{chain_id}" if chain_id else base_tenant_id

    logger.info(f"get_free_slots - Base tenant_id для API: {base_tenant_id}")
    logger.info(f"get_free_slots - Chain ID: {chain_id}")
    logger.info(f"get_free_slots - Combined tenant_id для локального поиска: {combined_tenant_id}")
    logger.info(f"get_free_slots - Retrieved from validated_args - api_token: {'******' if api_token_from_kwargs else None}")

    if not base_tenant_id:
        logger.error(f"CRITICAL (get_free_slots): base_tenant_id (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: ID тенанта не был предоставлен системе для вызова get_free_slots_tool."

    if not api_token_from_kwargs: # Токен тоже важен
        logger.error(f"CRITICAL (get_free_slots): api_token (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: API токен не был предоставлен системе для вызова get_free_slots_tool."

    # Используем остальные поля из validated_args
    employee_name_from_llm = validated_args.employee_name
    service_names_from_llm = validated_args.service_names
    filial_name_from_llm = validated_args.filial_name
    date_time_from_llm = validated_args.date_time

    logger.info(f"[get_free_slots_tool] Имена от LLM (из validated_args): employee='{employee_name_from_llm}', services='{service_names_from_llm}', filial='{filial_name_from_llm}'")

    # Проверка обязательных полей, генерируемых LLM, остается
    if not all([employee_name_from_llm, service_names_from_llm, filial_name_from_llm, date_time_from_llm]):
        missing_fields = []
        if not employee_name_from_llm: missing_fields.append('employee_name')
        if not service_names_from_llm: missing_fields.append('service_names')
        if not filial_name_from_llm: missing_fields.append('filial_name')
        if not date_time_from_llm: missing_fields.append('date_time')
        return f"Ошибка: отсутствуют обязательные поля от LLM (после валидации): {', '.join(missing_fields)}"

    # Преобразование имен в ID
    logger.info(f"[get_free_slots_tool] Попытка получить ID для employee: '{employee_name_from_llm}'")
    employee_id = get_id_by_name(combined_tenant_id, 'employee', employee_name_from_llm)
    logger.info(f"[get_free_slots_tool] Попытка получить ID для filial: '{filial_name_from_llm}'")
    filial_id = get_id_by_name(combined_tenant_id, 'filial', filial_name_from_llm)
    
    # Создаем переменные для отслеживания нечетких совпадений
    ambiguous_services = []
    service_ids = []
    
    # Используем модуль service_disambiguation для поиска услуг
    import service_disambiguation
    
    if service_names_from_llm:
        for s_name in service_names_from_llm:
            logger.info(f"[get_free_slots_tool] Попытка получить ID для service: '{s_name}'")
            s_id = get_id_by_name(combined_tenant_id, 'service', s_name)
            
            # Если ID не найден, ищем похожие услуги
            if not s_id:
                # Сначала проверим, есть ли похожие услуги в указанном филиале
                if filial_id:
                    message, similar_services = service_disambiguation.suggest_services_in_filial(
                        combined_tenant_id, s_name, filial_id, TENANT_RAW_DATA_MAP.get(combined_tenant_id, [])
                    )
                    # Если в филиале нет похожих услуг, ищем везде
                    if not similar_services:
                        message, similar_services = service_disambiguation.suggest_services(
                            combined_tenant_id, s_name, TENANT_RAW_DATA_MAP.get(combined_tenant_id, [])
                        )
                else:
                    # Если филиал не определен, ищем везде
                    message, similar_services = service_disambiguation.suggest_services(
                        combined_tenant_id, s_name, TENANT_RAW_DATA_MAP.get(combined_tenant_id, [])
                    )
                
                if similar_services and len(similar_services) <= 5:  # Ограничиваем количество предложений
                    ambiguous_services.append({
                        "query": s_name,
                        "suggestions": similar_services,
                        "message": message
                    })
                elif not similar_services:
                    ambiguous_services.append({
                        "query": s_name,
                        "suggestions": [],
                        "message": f"Услуга '{s_name}' не найдена."
                    })
                else:
                    ambiguous_services.append({
                        "query": s_name,
                        "suggestions": similar_services[:5],  # Ограничиваем до 5 предложений
                        "message": f"Найдено слишком много ({len(similar_services)}) похожих услуг для '{s_name}'. Уточните запрос."
                    })
            service_ids.append(s_id)

    if not employee_id:
        return f"Не удалось найти ID для сотрудника: '{employee_name_from_llm}'"
    
    if not filial_id:
        return f"Не удалось найти ID для филиала: '{filial_name_from_llm}'"
    
    # Если для некоторых услуг не найдены ID и есть предложения
    if ambiguous_services:
        response_parts = ["Необходимо уточнить некоторые услуги:"]
        for amb_service in ambiguous_services:
            response_parts.append(amb_service["message"])
        return "\n\n".join(response_parts)
    
    # Если ID для каких-то услуг не найдены, но нет предложений
    if not all(service_ids):
        problematic_services = [s_name for s_name, s_id in zip(service_names_from_llm, service_ids) if not s_id]
        return f"Не удалось найти ID для следующих услуг: {', '.join(problematic_services)}"

    # API сам проверит совместимость услуг, сотрудника и филиала
    # Убираем избыточные локальные проверки - доверяем API
    valid_service_ids = [s_id for s_id in service_ids if s_id]

    # Важно: tenant_id_from_kwargs уже содержит ТОЛЬКО базовый tenant_id из configurable
    # Он НЕ содержит chain_id, так как в app.py передается отдельно tenant_id и chain_id
    
    handler_args = {
        "tenant_id": base_tenant_id, # Используем только базовый tenant_id для API
        "employee_id": employee_id,
        "service_ids": valid_service_ids,  # Используем только валидные ID услуг
        "date_time": date_time_from_llm,
        "filial_id": filial_id,
        "api_token": api_token_from_kwargs # Используем извлеченное значение
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
    base_tenant_id = kwargs_from_llm.get('tenant_id')
    api_token_from_config = kwargs_from_llm.get('api_token')
    chain_id = kwargs_from_llm.get('chain_id')  # Получаем chain_id из kwargs

    if not base_tenant_id:
        logger.error(f"tenant_id отсутствует в аргументах ({kwargs_from_llm}) для BookAppointment.")
        return "Критическая ошибка: ID тенанта не был предоставлен для вызова BookAppointment."

    # Формируем комбинированный tenant_id для поиска в локальных данных
    combined_tenant_id = f"{base_tenant_id}_{chain_id}" if chain_id else base_tenant_id
    
    logger.info(f"book_appointment_tool - Base tenant_id для API: {base_tenant_id}")
    logger.info(f"book_appointment_tool - Chain ID: {chain_id}")
    logger.info(f"book_appointment_tool - Combined tenant_id для локального поиска: {combined_tenant_id}")

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
    
    # Преобразование имен в ID - используем комбинированный tenant_id для локального поиска
    logger.info(f"[book_appointment_tool] Попытка получить ID для service: '{service_name_from_llm}'")
    service_id = get_id_by_name(combined_tenant_id, 'service', service_name_from_llm)
    logger.info(f"[book_appointment_tool] Попытка получить ID для employee: '{employee_name_from_llm}'")
    employee_id = get_id_by_name(combined_tenant_id, 'employee', employee_name_from_llm)
    logger.info(f"[book_appointment_tool] Попытка получить ID для filial: '{filial_name_from_llm}'")
    filial_id = get_id_by_name(combined_tenant_id, 'filial', filial_name_from_llm)
    logger.info(f"[book_appointment_tool] Попытка получить ID для category: '{category_name_from_llm}'")
    category_id = get_id_by_name(combined_tenant_id, 'category', category_name_from_llm)

    if not service_id: return f"Не удалось найти ID для услуги: '{service_name_from_llm}'"
    if not employee_id: return f"Не удалось найти ID для сотрудника: '{employee_name_from_llm}'"
    if not filial_id: return f"Не удалось найти ID для филиала: '{filial_name_from_llm}'"
    if not category_id: return f"Не удалось найти ID для категории: '{category_name_from_llm}'"

    # Используем базовый tenant_id для API запросов (без chain_id)
    handler_args = {
        "tenant_id": base_tenant_id, # Используем только базовый tenant_id для API
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

async def book_appointment_ai_payload_tool(**kwargs_from_llm) -> str:
    logger.info(f"ENTERING book_appointment_ai_payload_tool. Raw kwargs_from_llm: {kwargs_from_llm}")

    try:
        # Валидируем все аргументы, включая системные, которые теперь часть модели
        validated_args = BookAppointmentAIPayloadArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для book_appointment_ai_payload_tool: {e}. Аргументы: {kwargs_from_llm}", exc_info=True)
        return f"Ошибка: неверные или отсутствующие аргументы для создания записи: {e}"

    # Извлекаем системные поля из validated_args
    base_tenant_id = validated_args.tenant_id
    api_token_from_kwargs = validated_args.api_token
    client_phone_number_from_kwargs = validated_args.client_phone_number
    
    # Получаем chain_id из kwargs (не из схемы Pydantic)
    chain_id = kwargs_from_llm.get('chain_id')

    # Формируем комбинированный tenant_id для поиска в локальных данных
    combined_tenant_id = f"{base_tenant_id}_{chain_id}" if chain_id else base_tenant_id

    logger.info(f"book_appointment_ai_payload_tool - Base tenant_id для API: {base_tenant_id}")
    logger.info(f"book_appointment_ai_payload_tool - Chain ID: {chain_id}")
    logger.info(f"book_appointment_ai_payload_tool - Combined tenant_id для локального поиска: {combined_tenant_id}")
    logger.info(f"book_appointment_ai_payload_tool - Retrieved from validated_args - api_token: {'******' if api_token_from_kwargs else None}")
    logger.info(f"book_appointment_ai_payload_tool - Retrieved from validated_args - client_phone_number: {client_phone_number_from_kwargs}")

    if not base_tenant_id:
        logger.error(f"CRITICAL (book_appointment_ai_payload_tool): tenant_id (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: ID тенанта не был предоставлен системе для вызова book_appointment_ai_payload_tool."
    
    if not api_token_from_kwargs:
        logger.error(f"CRITICAL (book_appointment_ai_payload_tool): api_token (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: API токен не был предоставлен системе для вызова book_appointment_ai_payload_tool."

    if not client_phone_number_from_kwargs:
        logger.error(f"CRITICAL (book_appointment_ai_payload_tool): client_phone_number (из validated_args) is missing or None. Validated Args: {validated_args.model_dump_json(indent=2)}")
        return "Критическая ошибка: Номер телефона клиента не был предоставлен системе для вызова book_appointment_ai_payload_tool."

    # Используем комбинированный tenant_id для поиска в локальных данных
    filial_id = get_id_by_name(combined_tenant_id, 'filial', validated_args.filial_name)
    to_employee_id = get_id_by_name(combined_tenant_id, 'employee', validated_args.employee_name)

    if not filial_id:
        return f"Не удалось найти ID для филиала: '{validated_args.filial_name}'"
    if not to_employee_id:
        return f"Не удалось найти ID для сотрудника: '{validated_args.employee_name}'"

    processed_services_payload = []
    if not validated_args.services_details:
        # Это должно было быть поймано Pydantic, если min_length=1, но на всякий случай
        logger.error(f"book_appointment_ai_payload_tool: validated_args.services_details пусто или отсутствует, хотя Pydantic модель требует его. Validated_args: {validated_args.model_dump_json(indent=2)}")
        return "Ошибка: Список услуг (services_details) не был предоставлен или пуст."

    for service_item_from_llm in validated_args.services_details:
        # Доступ к полям через точечную нотацию, так как service_item_from_llm это Pydantic модель
        service_name = service_item_from_llm.serviceName
        category_name = service_item_from_llm.categoryName
        count_service = service_item_from_llm.countService
        row_number = service_item_from_llm.rowNumber
        complex_service_id = service_item_from_llm.complexServiceId
        price_from_llm = service_item_from_llm.price
        duration_from_llm = service_item_from_llm.durationService

        # Проверка на None остается актуальной, так как Pydantic может присвоить None опциональным полям, 
        # если они не переданы, даже если в модели они не Optional (например, если LLM не передал)
        # Однако, price и durationService у нас обязательны в ServiceDetailItemFromLLM, 
        # поэтому Pydantic должен был бы выдать ошибку валидации раньше, если они None.
        # Но для serviceName и rowNumber проверка все еще имеет смысл, т.к. они тоже обязательны.
        if not service_name or row_number is None or price_from_llm is None or duration_from_llm is None:
            missing_fields_in_item = []
            if not service_name: missing_fields_in_item.append("'serviceName'")
            if row_number is None: missing_fields_in_item.append("'rowNumber'")
            if price_from_llm is None: missing_fields_in_item.append("'price'")
            if duration_from_llm is None: missing_fields_in_item.append("'durationService'")
            return f"Ошибка: неполные данные в одном из элементов services_details: {', '.join(missing_fields_in_item)} обязательны. Получено: {service_item_from_llm.model_dump_json(indent=2)}"

        service_id = get_id_by_name(combined_tenant_id, 'service', service_name)
        if not service_id:
            return f"Не удалось найти ID для услуги: '{service_name}'"

        # --- Определение parent_id (categoryId) ---
        parent_id = get_category_id_by_service_id(combined_tenant_id, service_id)
        
        if not parent_id and category_name: # Если по service_id не нашли, и LLM передал category_name
            logger.warning(f"Не удалось найти categoryId по serviceId='{service_id}'. Пытаемся найти по categoryName='{category_name}' от LLM.")
            parent_id = get_id_by_name(combined_tenant_id, 'category', category_name)
        
        if not parent_id: # Если parent_id все еще не найден
            # Попытка извлечь categoryId или categoryName из SERVICE_DETAILS_MAP_GLOBAL как последний шанс перед ошибкой
            service_info_for_category_fallback = SERVICE_DETAILS_MAP_GLOBAL.get((combined_tenant_id, service_id))
            if service_info_for_category_fallback:
                map_category_id = service_info_for_category_fallback.get('categoryId')
                if map_category_id:
                    parent_id = map_category_id
                    logger.info(f"Извлечен categoryId '{parent_id}' из SERVICE_DETAILS_MAP_GLOBAL для serviceId '{service_id}'.")
                else:
                    map_category_name = service_info_for_category_fallback.get('categoryName')
                    if map_category_name:
                        logger.warning(f"В SERVICE_DETAILS_MAP_GLOBAL для serviceId '{service_id}' есть categoryName '{map_category_name}', но нет categoryId. Пытаюсь найти ID для этого имени.")
                        parent_id = get_id_by_name(combined_tenant_id, 'category', map_category_name)
                        if parent_id:
                            logger.info(f"Найден categoryId '{parent_id}' по categoryName из SERVICE_DETAILS_MAP_GLOBAL.")
            
            if not parent_id: # Если все еще не нашли
                 return f"Не удалось определить ID категории для услуги '{service_name}' (serviceId: {service_id}). Пробовали по serviceId, по categoryName от LLM (если было), и по данным из SERVICE_DETAILS_MAP_GLOBAL."
        
        # service_info_from_map больше не используется для цены и длительности
        # но может быть использован для получения канонического имени услуги, если оно отличается
        service_info_from_map = SERVICE_DETAILS_MAP_GLOBAL.get((combined_tenant_id, service_id))
        
        api_service_name_for_payload = service_name # По умолчанию используем имя от LLM
        if service_info_from_map and service_info_from_map.get('serviceName'):
            api_service_name_for_payload = service_info_from_map.get('serviceName')
            logger.info(f"Имя услуги для API взято из SERVICE_DETAILS_MAP_GLOBAL: '{api_service_name_for_payload}' (LLM передал: '{service_name}')")
        else:
            logger.info(f"Имя услуги для API будет использовано как передал LLM: '{service_name}' (SERVICE_DETAILS_MAP_GLOBAL не содержит альтернативного имени). ")

        # Используем цену и длительность от LLM
        api_price = float(price_from_llm)
        api_duration_service = int(duration_from_llm)

        item_for_api_payload = {
            "rowNumber": row_number,
            "serviceId": service_id,
            "CategortyId": parent_id,  # <--- ИЗМЕНЕНО с parentId на CategortyId (с опечаткой как в API)
            "countService": count_service,
            "price": float(api_price),
            "salePrice": float(api_price), # По умолчанию salePrice = price. Можно доработать, если есть отдельное поле.
            "complexServiceId": complex_service_id,
            "durationService": int(api_duration_service),
            "serviceName": api_service_name_for_payload # Имя услуги, которое ожидает API
        }
        processed_services_payload.append(item_for_api_payload)
    
    if not processed_services_payload: # Дополнительная проверка, хотя validated_args.services_details должен был это покрыть
        return "Критическая ошибка: не удалось обработать ни одну услугу из services_details."

    # Аргументы для вызова BookAppointmentAIPayload
    handler_args = {
        "lang_id": validated_args.lang_id,
        "client_phone_number": client_phone_number_from_kwargs, 
        "services_payload": processed_services_payload, # <--- Используем обработанный список
        "filial_id": filial_id, 
        "date_of_record": validated_args.date_of_record,
        "start_time": validated_args.start_time,
        "end_time": validated_args.end_time,
        "duration_of_time": validated_args.duration_of_time, # Это общая длительность от LLM
        "to_employee_id": to_employee_id, 
        "total_price": validated_args.total_price, # Это общая цена от LLM
        "api_token": api_token_from_kwargs, 
        "color_code_record": validated_args.color_code_record or "",
        "traffic_channel": validated_args.traffic_channel or 0,
        "traffic_channel_id": validated_args.traffic_channel_id or "9", # API может ожидать строку '9' по умолчанию
        "tenant_id": base_tenant_id # Используем базовый tenant_id для API
    }

    try:
        logger.debug(f"Создание экземпляра BookAppointmentAIPayload с аргументами: {handler_args}")
        handler = clinic_functions.BookAppointmentAIPayload(**handler_args)
        logger.debug(f"Вызов handler.process() для BookAppointmentAIPayload")
        return await handler.process()
    except Exception as e:
        logger.error(f"Ошибка при создании или обработке BookAppointmentAIPayload: {e}", exc_info=True)
        error_type = getattr(e, '__class__', Exception).__name__
        return f"Ошибка при обработке запроса на запись (AI Payload) ({error_type}): {str(e)}"

async def reschedule_appointment_tool(**kwargs_from_llm) -> str:
    logger.info(f"ENTERING reschedule_appointment_tool. Raw kwargs_from_llm: {kwargs_from_llm}")
    try:
        validated_args = RescheduleAppointmentArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для reschedule_appointment_tool: {e}. Аргументы: {kwargs_from_llm}", exc_info=True)
        return f"Ошибка: неверные или отсутствующие аргументы для переноса записи: {e}"

    tenant_id = validated_args.tenant_id or kwargs_from_llm.get('tenant_id')
    api_token = validated_args.api_token or kwargs_from_llm.get('api_token')

    if not tenant_id:
        logger.warning("tenant_id не передан LLM. Продолжим без него (не обязателен для API).")
    else:
        validated_args.tenant_id = tenant_id

    if not api_token:
        logger.error("CRITICAL (reschedule_appointment_tool): api_token отсутствует.")
        return "Критическая ошибка: API токен не был предоставлен для переноса записи."

    validated_args.api_token = api_token  # гарантируем передачу токена в доменный слой

    handler_args = validated_args.model_dump()
    try:
        handler = clinic_functions.RescheduleAppointment(**handler_args)
        return await handler.process()
    except Exception as e:
        logger.error(f"Ошибка при создании или обработке RescheduleAppointment: {e}", exc_info=True)
        error_type = getattr(e, '__class__', Exception).__name__
        return f"Ошибка при обработке запроса на перенос записи ({error_type}): {str(e)}"


async def update_record_time_tool(**kwargs_from_llm) -> str:
    """Перенос записи (обёртка LangChain, аналогична reschedule_appointment_tool, но relies on ReAct injection tenant_id/api_token)."""
    logger.info(f"ENTERING update_record_time_tool. Raw kwargs_from_llm: {kwargs_from_llm}")
    try:
        validated_args = UpdateRecordTimeArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для update_record_time_tool: {e}. Аргументы: {kwargs_from_llm}", exc_info=True)
        return f"Ошибка: неверные или отсутствующие аргументы для переноса записи: {e}"

    handler_args = validated_args.model_dump()
    try:
        handler = clinic_functions.RescheduleAppointment(**handler_args)
        return await handler.process()
    except Exception as e:
        logger.error(f"Ошибка при создании или обработке RescheduleAppointment (update_record_time_tool): {e}", exc_info=True)
        error_type = getattr(e, '__class__', Exception).__name__
        return f"Ошибка при переносе записи ({error_type}): {e}"


async def cancel_appointment_tool(**kwargs_from_llm) -> str:
    """Отмена записи клиента. Если canceling_reason отсутствует, ассистент должен заранее определить её."""
    logger.info(f"ENTERING cancel_appointment_tool. Raw kwargs_from_llm: {kwargs_from_llm}")
    try:
        validated_args = CancelAppointmentArgs(**kwargs_from_llm)
    except Exception as e:
        logger.error(f"Ошибка валидации аргументов для cancel_appointment_tool: {e}. Arg: {kwargs_from_llm}", exc_info=True)
        return f"Ошибка: неверные или отсутствующие аргументы для отмены записи: {e}"

    # Если canceling_reason не указан, оставить None; доменный слой может выбрать самостоятельно (или LLM заранее подставит).
    handler_args = validated_args.model_dump()
    try:
        handler = clinic_functions.CancelAppointment(**handler_args)
        return await handler.process()
    except Exception as e:
        logger.error(f"Ошибка при создании/обработке CancelAppointment: {e}", exc_info=True)
        error_type = getattr(e, '__class__', Exception).__name__
        return f"Ошибка при отмене записи ({error_type}): {e}"

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
    book_appointment_ai_payload_tool,  
    reschedule_appointment_tool,
    update_record_time_tool,  
    cancel_appointment_tool,
]
logger.info(f"Определено {len(TOOL_FUNCTIONS)} функций-инструментов для динамической привязки.")
SYSTEM_PROMPT = """
**САМОЕ ГЛАВНОЕ ПРАВИЛО / THE MOST IMPORTANT RULE**
-**RU:** Всегда отвечай на том же языке, на котором говорит пользователь.
- **EN:** Always respond in the same language the user is speaking.
- **RU:** Если вопрос НЕ связан с медициной, услугами клиники, расписанием врачей или записью, вежливо откажись отвечать. Кратко сообщи, что ассистент специализируется только на медицинских вопросах и записи в клинику. **ЗАПРЕЩЕНО давать фактический ответ на немедицинскую тему.**
- **EN:** If the question is NOT related to medicine, clinic services, doctor schedules, or appointments, politely refuse to answer. Briefly state that the assistant specializes only in medical questions and clinic appointments. **DO NOT give a factual answer on a non-medical topic.**

Ты — вежливый и информативный ассистент, который помогает администраторам клиники (людям-операторам) в интерфейсе бэкэнда.
- Ты подготавливаешь черновики ответов, которые администратор может отредактировать перед отправкой пациенту.
- **КРИТИЧЕСКИ ВАЖНО О ПОДТВЕРЖДЕНИИ ДЕЙСТВИЙ:** Прежде чем вызывать какие-либо функции для записи, переноса или отмены (`book_appointment_tool`, `reschedule_appointment_tool`, `cancel_appointment_tool`), ты должен СНАЧАЛА предложить администратору конкретное действие и попросить подтвердить его нажатием кнопки "Принять" в интерфейсе.
- **ПОДТВЕРЖДЕНИЕ ОТ АДМИНИСТРАТОРА:** Когда администратор нажимает кнопку "Принять" в интерфейсе, тебе приходит сообщение **строго** вида `Давай(техническое сообщение,от бэкэнда)` — это техническое сообщение от бэкэнда, означающее одобрение администратора.
- **ТОЛЬКО ПОСЛЕ** получения такого технического сообщения выполняй соответствующую функцию и формируй финальное подтверждение для пациента.
- **НЕ ВЫПОЛНЯЙ** функции записи/переноса/отмены без получения подтверждения `Давай(техническое сообщение,от бэкэнда)`!

**ПРИМЕР РАБОЧЕГО ПРОЦЕССА:**
1. Пациент: "Запишите меня к врачу Иванову на завтра"
2. Ты: "Готов записать вас к доктору Иванову на [дата] в [время]. Для подтверждения записи администратор должен нажать кнопку 'Принять'."
3. Администратор нажимает "Принять" → приходит: `Давай(техническое сообщение,от бэкэнда)`
4. Ты вызываешь функцию записи и отвечаешь: "Запись подтверждена! Вы записаны к доктору Иванову на [дата] в [время]."

**КАК ФОРМУЛИРОВАТЬ ПРЕДЛОЖЕНИЯ О ДЕЙСТВИЯХ:**
- Всегда ясно опиши, какое именно действие будет выполнено
- Укажи все детали: врач, услуга, дата, время, филиал
- Обязательно добавь фразу о необходимости подтверждения администратором
- Примеры формулировок:
  * "Готов записать вас к [врач] на услугу '[услуга]' в филиале '[филиал]' на [дата] в [время]. Для подтверждения записи администратор должен нажать кнопку 'Принять'."
  * "Могу перенести вашу запись на [новая дата] в [новое время]. Для подтверждения переноса администратор должен нажать кнопку 'Принять'."
  * "Готов отменить вашу запись на [дата] в [время] к [врач]. Для подтверждения отмены администратор должен нажать кнопку 'Принять'."


Главные правила:
- Для **ОБЩИХ ОПИСАНИЙ** услуг (что такое МРТ, виды массажа), специалистов (их квалификации, опыта, но НЕ того, какие конкретно услуги они где оказывают), общей справочной информации о клинике (адреса, общие правила) — используй RAG-поиск.
- **ЗАПРЕЩЕНО ИСПОЛЬЗОВАТЬ RAG** для получения информации о том:
    - какие конкретные услуги выполняет тот или иной специалист.
    - в каких филиалах работает специалист.
    - в каких филиалах доступна конкретная услуга.
    - какие специалисты выполняют конкретную услугу в конкретном филиале.
  Для ВСЕХ этих данных — **ВСЕГДА ВЫЗЫВАЙ СООТВЕТСТВУЮЩУЮ ФУНКЦИЮ (ИНСТРУМЕНТ)**. Не пытайся отвечать на такие вопросы из RAG или памяти.
- Для всех остальных конкретных данных (цены, списки (кроме списков услуг/специалистов по филиалам), расписания, наличие, сравнения, фильтрация) — также всегда вызывай соответствующую функцию (инструмент).

**ВАЖНО: Классификация вопросов RAG vs Functions**
- **RAG используй для:** общих описаний услуг (что такое МРТ), квалификации врачей, общей информации о клинике, правил, адресов
- **Functions используй для:** конкретных данных - кто где работает, какие услуги оказывает конкретный врач, цены, расписания, записи

**ВАЖНО: Работа с find_employees_tool (унифицированный поиск)**
Функция find_employees_tool имеет 6 режимов работы в зависимости от переданных параметров:
1. Только filial_name → Возвращает всех сотрудников в филиале
2. Только service_name → Возвращает всех сотрудников, оказывающих услугу (независимо от филиала)
3. Только employee_name → Возвращает все услуги конкретного сотрудника (независимо от филиала)
4. filial_name + employee_name → Возвращает услуги сотрудника в конкретном филиале
5. filial_name + service_name → Возвращает сотрудников, оказывающих услугу в конкретном филиале
6. service_name + employee_name → Возвращает филиалы, где сотрудник оказывает конкретную услугу

Используй эти режимы осознанно в зависимости от того, что именно спрашивает пользователь.

- Если не хватает параметров для функции — вежливо уточни у пользователя.
- Не выдумывай данные и не сокращай списки, возвращай их полностью как выдала функция.
- Не давай медицинских советов, только информируй.

**ВАЖНО: Процесс записи на услугу**
1. Сначала всегда вызывай функцию получения свободных слотов (окон) сотрудника по услуге и филиалу (get_free_slots_tool). Убедись, что ты знаешь точное название услуги, для которой ищешь слоты, ФИО сотрудника и филиал. Используй инструменты для уточнения этой информации, если необходимо, **не полагайся на RAG**.

   **ВАЖНО О ПОИСКЕ СЛОТОВ**: Функция get_free_slots_tool использует умную логику поиска дат:
   - Начинает поиск с указанной даты (или сегодняшней, если дата не указана)
   - Если на указанную дату нет свободных слотов, автоматически ищет на 7 дней позже
   - Максимум проверяет 7 периодов (примерно месяц вперед) с интервалом в 7 дней
   - В ответе ясно указывает, в каком диапазоне дат проводился поиск
   - Не нужно вызывать функцию несколько раз с разными датами - она сама найдет ближайшие доступные слоты

2. Покажи пользователю доступные времена для записи (start_time), полученные от get_free_slots_tool. Если система нашла слоты не на запрашиваемую дату, а позже, обязательно объясни это пользователю.
3. После того как пользователь выберет конкретное время (start_time) из предложенных — вызывай функцию записи (book_appointment_ai_payload_tool),
   где:
   - duration_of_time всегда равен 60 (минут, если иное не было уточнено ранее для конкретной услуги).
   - end_time = start_time + duration_of_time (вычисляй автоматически, start_time — это выбранное пользователем время из слотов). Убедись, что end_time корректно рассчитан.
   - services_details: ОБЯЗАТЕЛЬНЫЙ список объектов. ВСЕГДА включай это поле, даже если запись на одну услугу.
     Каждый объект описывает одну услугу для записи и ДОЛЖЕН содержать:
       - 'rowNumber' (int, порядковый номер, начиная с 1 для первой услуги).
       - 'serviceName' (str, ТОЧНОЕ название услуги, которое было подтверждено с пользователем и для которого искались/были найдены слоты).
       - 'categoryName' (Optional[str], НАЗВАНИЕ КАТЕГОРИИ. По возможности передайте его, если оно известно или было явно уточнено. Система в первую очередь попытается определить категорию автоматически по ID услуги, но это поле может служить уточнением или запасным вариантом). 
       - 'countService' (int, количество данной услуги, обычно 1).
       - 'complexServiceId' (Optional[str], ID комплексной услуги, если эта услуга является частью комплекса).
       - 'price' (float, цена ИМЕННО ЭТОЙ услуги. ВАЖНО: Эту цену ты ОБЯЗАН получить, предварительно вызвав инструмент get_service_price_tool с параметром in_booking_process=True. Не выдумывай цену!).
       - 'durationService' (int, длительность ИМЕННО ЭТОЙ услуги в минутах. LLM должен определить это значение, например, из описания услуги или из ее стандартной длительности, если она не была явно запрошена у get_service_price_tool. Если сомневаешься, уточни у пользователя или предположи стандартную длительность 60 минут).
       # Убрал price и durationService из полей, которые LLM должен найти через get_service_price_tool, так как get_service_price_tool возвращает только цену.
       # Для durationService - LLM определяет сам.
   - total_price: ОБЩАЯ цена всей записи (сумма цен всех услуг из 'services_details'). Ты должен рассчитать это значение.
   - client_phone_number: Телефон клиента будет добавлен автоматически системой. НЕ ПЫТАЙСЯ ЕГО УКАЗАТЬ ИЛИ ЗАПРОСИТЬ.
   - filial_name, date_of_record, start_time, end_time, employee_name - должны быть точно определены перед вызовом.

**ВАЖНО: Перенос существующей записи**
- Вызывай `reschedule_appointment_tool` ТОЛЬКО если пользователь ЯВНО просит изменить дату или время уже существующей записи. Ключевые формулировки: «перенести запись», «сдвинуть запись», «изменить дату/время записи», «move appointment», «reschedule appointment».
- Если пользователь явно НЕ упоминает перенос (например, спрашивает «запиши меня…»), считай это созданием НОВОЙ записи и следуй стандартному процессу (get_free_slots_tool → book_appointment_*).
- Если запрос неоднозначен (например, «хочу пораньше»), уточни у пользователя, имеется ли в виду перенос существующей записи или новая запись.

Дополнительные общие инструкции:
- Приветствуй пользователя и предлагай помощь.
- **Если вопрос НЕ связан с медициной, услугами клиники, расписанием врачей или записью**, вежливо откажись отвечать: кратко сообщи, что ассистент специализируется только на медицинских вопросах и записи в клинику. НЕ давай фактический ответ на немедицинскую тему. Эта логика действует на любом языке (русском, английском и др.).
- Обращайся к пользователю по имени, если оно известно из истории диалога.
- Если пользователь спрашивает о доступных врачах или услугах без указания филиала, уточни филиал. **Для получения этой информации используй инструменты.**
- Прежде чем предлагать запись, убедись, что известны: филиал, сотрудник, услуга, дата и время. **Для получения этой информации используй инструменты.**
- Для дат используй формат YYYY-MM-DD, для времени HH:MM.
- Если поиск по базе знаний (RAG) не дал результатов (для разрешенных RAG-запросов), сообщи об этом и предложи пользователю переформулировать запрос или запросить конкретные данные через функции.
- Избегай неоднозначных ответов. Если информация не найдена или действие не может быть выполнено, четко сообщи об этом.
- Старайся минимизировать количество последовательных вызовов функций. Если возможно, получай всю необходимую информацию за один вызов или сначала собери все данные от пользователя.
- Не повторяй ту информацию, которую уже предоставил в предыдущих сообщениях, если пользователь не просит об этом явно.
- Если пользователь просит "показать еще" или "дальше" для списка, который был ограничен (например, "показано 5 из 10"), используй параметр page_number соответствующего инструмента для отображения следующей страницы.
- Если ты вызвал инструмент и он вернул ошибку, сообщи пользователю об ошибке корректно и не пытайся вызвать тот же инструмент с теми же аргументами снова, если только ошибка не была связана с временной недоступностью. Вместо этого, попроси пользователя проверить данные или переформулировать запрос.
- Если пользователь здоровается или задает общий вопрос, ответь вежливо и спроси, чем можешь помочь. Не нужно сразу предлагать услуги или филиалы, если пользователь об этом не просил.
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
    def messages(self) -> List[BaseMessage]:
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
    Функция, имитирующая выполнение основного агента или цепочки с ReAct-подобным циклом.
    Принимает словарь с 'input' (вопрос пользователя или мультимодальные данные) и RunnableConfig.
    Извлекает tenant_id и session_id (user_id) из config.
    Выполняет RAG-поиск (один раз в начале), формирует промпт и вызывает LLM в цикле.
    Динамически создает инструменты для LLM.
    Поддерживает мультимодальный ввод (текст + изображения).
    """
    raw_input = input_dict.get("input")
    
    # Обработка мультимодального ввода
    is_multimodal = isinstance(raw_input, list)
    if is_multimodal:
        # Извлекаем текст из мультимодального ввода для RAG-поиска
        text_blocks = [block for block in raw_input if block.get("type") == "text"]
        question = text_blocks[0]["text"] if text_blocks else ""
        logger.info(f"Получен мультимодальный ввод с {len(raw_input)} блоками. Извлечен текст для RAG: {question[:100]}...")
    else:
        # Обычный текстовый ввод
        question = raw_input or ""
        logger.info(f"Получен текстовый ввод: {question[:100]}...")
    
    # Получаем историю сообщений из input_dict
    history_messages: List[BaseMessage] = input_dict.get("history", [])
    
    # Проверка на техническое сообщение от бэкэнда (подтверждение администратора)
    if question.strip() == "Давай(техническое сообщение,от бэкэнда)":
        logger.info("Получено техническое подтверждение от администратора.")
        
        # Анализируем историю сообщений для поиска последнего предложения о действии
        last_assistant_message = None
        for msg in reversed(history_messages):
            if isinstance(msg, AIMessage):
                last_assistant_message = msg
                break
        
        if last_assistant_message and last_assistant_message.content:
            content = str(last_assistant_message.content)
            logger.info(f"Найдено последнее сообщение ассистента: {content[:100]}...")
            
            # Ищем ключевые фразы, указывающие на предложение действия
            action_keywords = [
                "для подтверждения", "нажать кнопку", "принять", 
                "подтвердить", "готов записать", "запишу", "перенесу", "отменю"
            ]
            
            if any(keyword in content.lower() for keyword in action_keywords):
                logger.info("Обнаружено предложение действия в последнем сообщении. Система готова выполнить его.")
                # Вместо простого ответа, продолжаем обработку с модифицированным промптом
                # Добавляем специальную инструкцию в системный промпт
                special_instruction = "\n\n**ТЕХНИЧЕСКОЕ ПОДТВЕРЖДЕНИЕ ПОЛУЧЕНО:** Администратор подтвердил выполнение действия. Выполни соответствующую функцию СЕЙЧАС, основываясь на контексте предыдущего сообщения."
                config["system_message"] = special_instruction
                # Заменяем вопрос на запрос о выполнении действия
                question = "Выполни подтвержденное администратором действие"
                logger.info("Модифицирован промпт для выполнения подтвержденного действия.")
            else:
                return "Техническое подтверждение получено, но не найдено соответствующее предложение действия в истории."
        else:
            return "Техническое подтверждение получено, но история сообщений пуста или не содержит сообщений ассистента."
    
    configurable = config.get("configurable", {})
    composite_session_id = configurable.get("session_id")

    MAX_ITERATIONS = 7 # Максимальное количество циклов LLM-Tools
    current_iteration = 0

    if not composite_session_id:
        raise ValueError("Отсутствует 'session_id' (composite_session_id) в конфигурации Runnable.")
    try:
        tenant_id, user_id = composite_session_id.split(":", 1)
        if not tenant_id or not user_id:
            raise ValueError("tenant_id или user_id пусты после разделения composite_session_id.")
    except ValueError as e:
        raise ValueError(f"Ошибка разбора composite_session_id '{composite_session_id}': {e}")

    logger.info(f"run_agent_like_chain (ReAct) для Tenant: {tenant_id}, User: {user_id}, Вопрос: {question[:50]}...")

    if not CHROMA_CLIENT or not EMBEDDINGS_GIGA:
        logger.error(f"RAG компоненты (Chroma/Embeddings) не инициализированы.")
        return "Ошибка: Базовые компоненты RAG не готовы."
    if not tenant_id:
        logger.error("Tenant ID не найден в config.")
        return "Ошибка: Не удалось определить тенанта."

    # --- Инициализация RAG компонентов для тенанта (как и раньше) ---
    from rag_setup import build_collection_name  # импорт внутри функции, чтобы избежать циклов
    from chromadb.errors import InvalidCollectionException
    bm25_retriever = BM25_RETRIEVERS_MAP.get(tenant_id)
    collection_name = build_collection_name(tenant_id)
    embeddings_wrapper = ChromaGigaEmbeddingsWrapper(EMBEDDINGS_GIGA)
    try:
        chroma_collection = CHROMA_CLIENT.get_collection(
            name=collection_name,
            embedding_function=embeddings_wrapper
        )
    except InvalidCollectionException:
        logger.warning(f"Коллекция '{collection_name}' для тенанта {tenant_id} отсутствует. Запускаем переиндексацию...")
        success = rag_setup.reindex_tenant_specific_data(
            tenant_id=tenant_id,
            chroma_client=CHROMA_CLIENT,
            embeddings_object=EMBEDDINGS_GIGA,
            bm25_retrievers_map=BM25_RETRIEVERS_MAP,
            tenant_documents_map=TENANT_DOCUMENTS_MAP,
            tenant_raw_data_map=TENANT_RAW_DATA_MAP,
            service_details_map=SERVICE_DETAILS_MAP_GLOBAL,
            base_data_dir=os.getenv("BASE_DATA_DIR", "base"),
            chunk_size=int(os.getenv("CHUNK_SIZE", 1000)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 200)),
            search_k=search_k_global
        )
        if not success:
            logger.error(f"Не удалось создать коллекцию '{collection_name}' для тенанта {tenant_id}.")
            return f"Ошибка: Не удалось подготовить базу знаний для филиала '{tenant_id}'."
        logger.info(f"Коллекция '{collection_name}' успешно создана. Повторяем получение...")
        chroma_collection = CHROMA_CLIENT.get_collection(
            name=collection_name,
            embedding_function=embeddings_wrapper
        )
        bm25_retriever = BM25_RETRIEVERS_MAP.get(tenant_id)  # Обновляем BM25 после переиндексации
    except ValueError as e:
        logger.error(f"Ошибка ChromaDB при получении коллекции '{collection_name}' для тенанта {tenant_id}: {e}", exc_info=True)
        return f"Ошибка: Проблема совместимости с базой знаний ChromaDB для филиала '{tenant_id}'. {e}"
    except Exception as e:
        logger.error(f"Не удалось получить или создать Chroma коллекцию '{collection_name}' для тенанта {tenant_id}: {e}", exc_info=True)
        return f"Ошибка: Не удалось получить доступ к базе знаний для филиала '{tenant_id}'."
    
    chroma_vectorstore = Chroma(
        client=CHROMA_CLIENT,
        collection_name=collection_name,
        embedding_function=embeddings_wrapper,
    )
    chroma_retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": search_k_global})
    logger.info(f"Chroma ретривер для коллекции '{collection_name}' (tenant: {tenant_id}) готов.")

    if not bm25_retriever:
        logger.warning(f"BM25 ретривер для тенанта {tenant_id} не найден. Будет создан EnsembleRetriever только с векторным поиском.")
        # Создаем пустой BM25 ретривер для EnsembleRetriever
        from langchain_core.documents import Document
        dummy_docs = [Document(page_content="dummy", metadata={})]
        dummy_bm25 = BM25Retriever.from_documents(dummy_docs, k=search_k_global)
        final_retriever = EnsembleRetriever(
            retrievers=[dummy_bm25, chroma_retriever],
            weights=[0.0, 1.0]  # Вес 0.0 для BM25, 1.0 для векторного поиска
        )
        logger.info(f"Ensemble ретривер создан для тенанта {tenant_id} с весами [BM25: 0.0, Chroma: 1.0] (только векторный поиск).")
    else:
        final_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, chroma_retriever],
            weights=[0.5, 0.5]
        )
        logger.info(f"Ensemble ретривер (BM25 + Chroma) создан для тенанта {tenant_id} с весами [0.5, 0.5].")

    # --- Шаг 1: Определение необходимости RAG и формирование RAG-запроса (один раз) ---
    rag_context = ""
    effective_rag_query = question # По умолчанию, если rag_query_llm не отработает или вернет пустой запрос
    try:
        rag_query_llm = chat_model.with_structured_output(RagQueryThought)
        rag_prompt_messages_for_rag_llm: List[BaseMessage] = [
            SystemMessage(content=(
                "Твоя задача - проанализировать последний запрос пользователя ('input') в контексте предыдущего диалога ('history'). "
                "Определи главную сущность (услуга, врач, филиал, категория, общая информация о клинике), о которой спрашивает пользователь, "
                "особенно если он использует ссылки на историю (номер пункта, местоимения 'он', 'она', 'это', 'они'). "
                "Затем сформулируй оптимальный, самодостаточный поисковый запрос ('best_rag_query') для векторной базы знаний, "
                "чтобы найти ОПИСАНИЕ или детали этой сущности. Используй полные названия. "
                "Пример: Если история содержит '1. Услуга А\\n2. Услуга Б', а пользователь спрашивает 'расскажи о 2', "
                "то best_rag_query должен быть что-то вроде 'Описание услуги Б'. "
                "ВАЖНО: Если запрос пользователя является простым приветствием (например, 'Привет', 'Добрый день', 'Как дела?'), "
                "или очевидным согласием/просьбой продолжить предыдущий ответ ассистента (например, 'да', 'дальше', 'продолжай', 'еще', 'следующая страница'), "
                "то best_rag_query ДОЛЖЕН БЫТЬ ПУСТОЙ СТРОКОЙ. "
                "Также, если запрос очевидно требует точных данных через вызов функции (например, запросы на цены, поиск сотрудника по имени, списки филиалов, проверки наличия, сравнения, поиск по точным критериям), "
                "то best_rag_query ДОЛЖЕН БЫТЬ ПУСТОЙ СТРОКОЙ. "
                "Во всех остальных случаях (запросы на описание услуг/процедур, информацию о показаниях/противопоказаниях, общую информацию о клинике, что помогает от морщин и т.п.) сформулируй поисковый запрос для поиска описательной информации. "
                "Поле 'analysis' должно содержать краткий анализ твоего решения, особенно пояснение, почему RAG-запрос нужен или не нужен."
            ))
        ]
        if history_messages: # Используем общую историю для rag_query_llm
             rag_prompt_messages_for_rag_llm.extend(history_messages)
        rag_prompt_messages_for_rag_llm.append(HumanMessage(content=question))
        
        logger.debug(f"[{tenant_id}:{user_id}] Вызов LLM для генерации RAG-запроса...")
        rag_thought_result = await rag_query_llm.ainvoke(rag_prompt_messages_for_rag_llm, config=config)
        
        if isinstance(rag_thought_result, RagQueryThought):
            rag_thought = rag_thought_result
            if rag_thought.best_rag_query and rag_thought.best_rag_query.strip():
                 effective_rag_query = rag_thought.best_rag_query
                 logger.info(f"[{tenant_id}:{user_id}] Сгенерирован RAG-запрос от LLM: '{effective_rag_query}' (Анализ: {rag_thought.analysis})")
            elif hasattr(rag_thought, 'best_rag_query') and not rag_thought.best_rag_query.strip(): # Явный пустой запрос
                effective_rag_query = ""
                logger.info(f"[{tenant_id}:{user_id}] LLM указал, что RAG-запрос должен быть пустым. Анализ: {rag_thought.analysis}")
            else: # Невалидный или отсутствующий best_rag_query
                effective_rag_query = ""
                logger.info(f"[{tenant_id}:{user_id}] LLM вернул RagQueryThought, но best_rag_query невалиден/пуст. RAG-запрос будет пустым. Анализ: {rag_thought.analysis if hasattr(rag_thought, 'analysis') else 'нет анализа'}")
        else:
            effective_rag_query = "" # Если вернулся не тот тип, RAG-запрос пустой
            logger.warning(f"[{tenant_id}:{user_id}] LLM для генерации RAG-запроса вернул неожиданный тип: {type(rag_thought_result)}. RAG-запрос будет пустым.")
            if hasattr(rag_thought_result, 'analysis'): logger.warning(f"Анализ от LLM (при неожиданном типе): {rag_thought_result.analysis}")

    except Exception as e:
        effective_rag_query = "" # При любой ошибке RAG-запрос пустой
        logger.warning(f"[{tenant_id}:{user_id}] Исключение при улучшении RAG-запроса: {e}. RAG-запрос будет пустым.", exc_info=True)

    # --- Шаг 2: Выполнение RAG-поиска, если необходимо (один раз) ---
    rag_context = ""
    max_retries = 3
    retry_delay = 2  # секунды
    
    try:
        if effective_rag_query and effective_rag_query.strip():
            logger.info(f"[{tenant_id}:{user_id}] Выполнение RAG-поиска с запросом: '{effective_rag_query[:100]}...'")
            
            # Retry-механизм для RAG поиска с улучшенной обработкой таймаутов
            for attempt in range(max_retries):
                try:
                    logger.debug(f"[{tenant_id}:{user_id}] RAG поиск попытка {attempt + 1}/{max_retries}")
                    relevant_docs = await final_retriever.ainvoke(effective_rag_query, config=config)
                    rag_context = format_docs(relevant_docs)
                    logger.info(f"RAG: Найдено {len(relevant_docs)} док-в для запроса: '{effective_rag_query[:50]}...'. Контекст: {len(rag_context)} симв.")
                    break  # Успешно выполнено, выходим из retry-цикла
                    
                except Exception as retry_error:
                    error_message = str(retry_error)
                    is_timeout_error = any(timeout_keyword in error_message.lower() for timeout_keyword in 
                                         ['timeout', 'timed out', 'time out', 'connection timeout', 'read timeout'])
                    
                    if is_timeout_error:
                        logger.warning(f"[{tenant_id}:{user_id}] RAG поиск неудачен из-за таймаута (попытка {attempt + 1}/{max_retries}): {retry_error}")
                    else:
                        logger.warning(f"[{tenant_id}:{user_id}] RAG поиск неудачен (попытка {attempt + 1}/{max_retries}): {retry_error}")
                    
                    if attempt < max_retries - 1:  # Если это не последняя попытка
                        if is_timeout_error:
                            # Для таймаутов используем больший интервал повтора
                            timeout_delay = retry_delay * 2
                            logger.info(f"[{tenant_id}:{user_id}] Ожидание {timeout_delay} сек перед повтором RAG поиска (таймаут)")
                            await asyncio.sleep(timeout_delay)
                        else:
                            logger.info(f"[{tenant_id}:{user_id}] Ожидание {retry_delay} сек перед повтором RAG поиска")
                            await asyncio.sleep(retry_delay)
                        retry_delay *= 1.5  # Постепенное увеличение задержки
                    else:
                        # Последняя попытка неудачна, пробрасываем исключение
                        logger.error(f"[{tenant_id}:{user_id}] Все {max_retries} попытки RAG поиска неудачны. Последняя ошибка: {retry_error}")
                        raise retry_error
        else:
            logger.info(f"[{tenant_id}:{user_id}] RAG-запрос пуст, RAG поиск пропущен.")
            rag_context = "Поиск по базе знаний не выполнялся, так как запрос не предполагает этого или является вызовом функции."
            
    except Exception as e:
        error_message = str(e)
        is_timeout_error = any(timeout_keyword in error_message.lower() for timeout_keyword in 
                             ['timeout', 'timed out', 'time out', 'connection timeout', 'read timeout'])
        
        if is_timeout_error:
            logger.error(f"Таймаут выполнения RAG поиска для тенанта {tenant_id} с запросом '{effective_rag_query}' после {max_retries} попыток: {e}", exc_info=True)
            rag_context = "[Превышено время ожидания при поиске в базе знаний. Возможно, сервер перегружен. Попробуйте переформулировать запрос или повторить позже.]"
        else:
            logger.error(f"Ошибка выполнения RAG поиска для тенанта {tenant_id} с запросом '{effective_rag_query}' после {max_retries} попыток: {e}", exc_info=True)
            rag_context = "[Ошибка получения информации из базы знаний. Попробуйте переформулировать запрос или повторить позже.]"

    # --- Подготовка к циклу ReAct ---
    system_prompt_base = SYSTEM_PROMPT # Базовый системный промпт
    # Добавляем инфо о дате/времени
    tz = pytz.timezone('Europe/Moscow')
    now = datetime.now(tz)
    datetime_info = f"Сегодня: {now.strftime('%d %B %Y, %A, %H:%M')} (Europe/Moscow)"
    final_system_prompt = f"{datetime_info}\\n\\n" + system_prompt_base
    # --- Apply external system message override, if provided (e.g., from FastAPI layer) ---
    override_system_msg = config.get("system_message") if isinstance(config, dict) else None
    if override_system_msg:
        # Append so that override is the last instruction (higher priority for the LLM)
        final_system_prompt = final_system_prompt + "\n\n" + override_system_msg
        logger.info("[SystemPrompt Override] External system_message appended for language/tool-call priority.")
    
    if tenant_config_manager:
        settings = tenant_config_manager.load_tenant_settings(tenant_id)
        prompt_addition = settings.get('prompt_addition')
        if prompt_addition:
            final_system_prompt += f"\\n\\n[Дополнительные инструкции от администратора филиала {tenant_id}]:\\n{prompt_addition}"
            logger.info(f"Добавлено дополнение к промпту для тенанта {tenant_id}.")

    # --- Создание инструментов (один раз перед циклом) ---
    tenant_tools = []
    tenant_specific_docs: Optional[List[Document]] = TENANT_DOCUMENTS_MAP.get(tenant_id)
    if not tenant_specific_docs:
         logger.warning(f"Не найдены загруженные документы для тенанта {tenant_id} в tenant_documents_map_global. Некоторые инструменты могут работать некорректно.")
    
    tool_func_to_schema_map = {
        find_employees_tool: FindEmployeesArgs, get_service_price_tool: GetServicePriceArgs,
        list_filials_tool: NoArgsSchema, # <--- ИЗМЕНЕНО с None
        get_employee_services_tool: GetEmployeeServicesArgs,
        check_service_in_filial_tool: CheckServiceInFilialArgs, compare_service_price_in_filials_tool: CompareServicePriceInFilialsArgs,
        find_service_locations_tool: FindServiceLocationsArgs, find_specialists_by_service_or_category_and_filial_tool: FindSpecialistsByServiceOrCategoryAndFilialArgs,
        list_services_in_category_tool: ListServicesInCategoryArgs, list_services_in_filial_tool: ListServicesInFilialArgs,
        find_services_in_price_range_tool: FindServicesInPriceRangeArgs, 
        list_all_categories_tool: ListCategoriesArgs,
        list_employee_filials_tool: ListEmployeeFilialsArgs, 
        reschedule_appointment_tool: RescheduleAppointmentArgs, 
        update_record_time_tool: UpdateRecordTimeArgs, 
        cancel_appointment_tool: CancelAppointmentArgs
    }

    def create_tool_wrapper_react(original_tool_func: callable, raw_data_for_tenant: Optional[List[Dict]], tenant_id_for_tool: str, configurable_dict: Dict = None): # <--- ДОБАВЛЕН configurable_dict
        import inspect
        async def actual_wrapper(*args, **kwargs):
            tool_name = original_tool_func.__name__
            logger.info(f"[ReAct Tool] Вызов обертки для {tool_name} с args: {args}, kwargs: {kwargs}")
            clinic_functions.set_clinic_data(raw_data_for_tenant if raw_data_for_tenant is not None else [], tenant_id=tenant_id_for_tool)
            try:
                handler_class_name = ''.join(word.capitalize() for word in tool_name.replace('_tool', '').split('_'))
                HandlerClass = getattr(clinic_functions, handler_class_name, None)
                if not HandlerClass:
                    logger.error(f"Не найден класс-обработчик '{handler_class_name}' для {tool_name}")
                    return f"Ошибка: Некорректная конфигурация инструмента {tool_name}."
                handler_instance = HandlerClass(**kwargs)
                process_method = getattr(handler_instance, 'process', None)
                if process_method:
                    params = inspect.signature(process_method).parameters
                    needs_tenant = 'tenant_id' in params
                    needs_token = 'api_token' in params
                    # Получаем tenant_id и api_token
                    tenant_id = kwargs.get('tenant_id') or tenant_id_for_tool
                    api_token = kwargs.get('api_token')
                    
                    # Если api_token не найден в kwargs, пытаемся получить из configurable_dict
                    if not api_token and configurable_dict:
                        api_token = configurable_dict.get("client_api_token")
                        logger.info(f"[ReAct Tool] api_token получен из configurable_dict: {api_token[:50] if api_token else None}...")
                    
                    logger.info(f"[ReAct Tool] Передача в {tool_name}: tenant_id='{tenant_id}', api_token='{api_token[:50] if api_token else None}...'")
                    # --- Проверяем, асинхронная ли функция ---
                    if inspect.iscoroutinefunction(process_method):
                        if needs_tenant and needs_token:
                            return await process_method(tenant_id, api_token)
                        elif needs_tenant:
                            return await process_method(tenant_id)
                        elif needs_token:
                            return await process_method(api_token)
                        else:
                            return await process_method()
                    else:
                        if needs_tenant and needs_token:
                            return process_method(tenant_id, api_token)
                        elif needs_tenant:
                            return process_method(tenant_id)
                        elif needs_token:
                            return process_method(api_token)
                        else:
                            return process_method()
                else:
                    logger.error(f"У обработчика {handler_class_name} отсутствует метод process.")
                    return f"Ошибка: У обработчика {handler_class_name} отсутствует метод process."
            except Exception as e:
                logger.error(f"{tool_name}: {e}", exc_info=True)
                return f"При выполнении инструмента {tool_name} произошла ошибка."
        return actual_wrapper

    for tool_function in TOOL_FUNCTIONS: # TOOL_FUNCTIONS из глобальной области видимости
        tool_name = tool_function.__name__
        tool_description = tool_function.__doc__ or f"Инструмент {tool_name}"
        current_args_schema = None
        func_for_tool = None

        if tool_name == "get_free_slots_tool":
            current_args_schema = GetFreeSlotsArgs
            # Создаём обёртку для передачи chain_id из configurable
            async def get_free_slots_wrapper(tool_func=tool_function, **kwargs):
                # Добавляем chain_id из configurable в kwargs
                if configurable and 'chain_id' in configurable:
                    kwargs['chain_id'] = configurable['chain_id']
                return await tool_func(**kwargs)
            func_for_tool = get_free_slots_wrapper
        elif tool_name == "book_appointment_tool":
            current_args_schema = BookAppointmentArgs
            # Создаём обёртку для передачи chain_id из configurable
            async def book_appointment_wrapper(tool_func=tool_function, **kwargs):
                # Добавляем chain_id из configurable в kwargs
                if configurable and 'chain_id' in configurable:
                    kwargs['chain_id'] = configurable['chain_id']
                return await tool_func(**kwargs)
            func_for_tool = book_appointment_wrapper
        elif tool_name == "book_appointment_ai_payload_tool":
            current_args_schema = BookAppointmentAIPayloadArgs
            # Создаём обёртку для передачи chain_id из configurable  
            async def book_appointment_ai_payload_wrapper(tool_func=tool_function, **kwargs):
                # Добавляем chain_id из configurable в kwargs
                if configurable and 'chain_id' in configurable:
                    kwargs['chain_id'] = configurable['chain_id']
                return await tool_func(**kwargs)
            func_for_tool = book_appointment_ai_payload_wrapper
        elif tool_function in tool_func_to_schema_map:
            current_args_schema = tool_func_to_schema_map.get(tool_function)
            func_for_tool = create_tool_wrapper_react(tool_function, TENANT_RAW_DATA_MAP.get(tenant_id, []), tenant_id, configurable) # <--- ПЕРЕДАЕМ configurable
        else:
            logger.warning(f"Инструмент '{tool_name}' не классифицирован. Использование стандартной обертки и schema=None.")
            current_args_schema = None # Явно None, если не классифицирован
            func_for_tool = create_tool_wrapper_react(tool_function, TENANT_RAW_DATA_MAP.get(tenant_id, []), tenant_id, configurable) # <--- ПЕРЕДАЕМ configurable

        if func_for_tool is None:
            logger.error(f"Не удалось определить функцию для инструмента '{tool_name}'. Пропуск.")
            continue
            
        # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
        is_async_func = inspect.iscoroutinefunction(func_for_tool)
        langchain_tool = StructuredTool.from_function(
            func=func_for_tool, 
            name=tool_name, 
            description=tool_description, 
            args_schema=current_args_schema,
            coroutine=func_for_tool if is_async_func else None # Явно указываем корутину, если функция асинхронная
        )
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        tenant_tools.append(langchain_tool)
        logger.debug(f"Инструмент '{tool_name}' создан. Schema: {current_args_schema.__name__ if current_args_schema else 'None'}. Async: {is_async_func}")

    # --- Инициализация сообщений для LLM перед циклом ---
    messages_for_llm_cycle: List[BaseMessage] = []
    messages_for_llm_cycle.append(SystemMessage(content=final_system_prompt))
    messages_for_llm_cycle.extend(history_messages) # Общая история диалога
    
    rag_query_display_initial = effective_rag_query if effective_rag_query and effective_rag_query.strip() else "не выполнялся"
    rag_context_block = f"\\n\\n[Информация из базы знаний (поисковый запрос: '{rag_query_display_initial}')]:\\n{rag_context}\\n[/Информация из базы знаний]"
    
    # Создаем итоговое сообщение пользователя с учетом мультимодальности
    if is_multimodal:
        # Для мультимодального ввода обновляем текстовый блок, добавляя RAG-контекст
        final_multimodal_content = []
        for block in raw_input:
            if block.get("type") == "text":
                # Добавляем RAG-контекст к первому текстовому блоку
                enhanced_text = block["text"] + rag_context_block
                final_multimodal_content.append({
                    "type": "text",
                    "text": enhanced_text
                })
            else:
                # Изображения и другие блоки копируем как есть
                final_multimodal_content.append(block)
        
        messages_for_llm_cycle.append(HumanMessage(content=final_multimodal_content))
        logger.info(f"Создано мультимодальное сообщение с {len(final_multimodal_content)} блоками контента")
    else:
        # Обычное текстовое сообщение
        messages_for_llm_cycle.append(HumanMessage(content=question + rag_context_block))
        logger.info("Создано текстовое сообщение для LLM")

    llm_with_tools = chat_model.bind_tools(tenant_tools)
    final_answer_content = "Не удалось получить ответ от ассистента." # Ответ по умолчанию

    # --- ReAct цикл ---
    while current_iteration < MAX_ITERATIONS:
        current_iteration += 1
        logger.info(f"[{tenant_id}:{user_id}] ReAct Итерация: {current_iteration}/{MAX_ITERATIONS}")

        try:
            # Логгирование текущего состояния messages_for_llm_cycle перед вызовом
            if logger.isEnabledFor(logging.DEBUG):
                debug_messages = []
                for msg_idx, msg in enumerate(messages_for_llm_cycle):
                    content_preview = str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        tool_call_summary = f" (Tool Calls: {[tc['name'] for tc in msg.tool_calls]})"
                        content_preview += tool_call_summary
                    debug_messages.append(f"  Msg {msg_idx} ({msg.type}): {content_preview}")
                logger.debug(f"Сообщения для LLM на итерации {current_iteration}:\\n" + "\\n".join(debug_messages))

            ai_response_message: AIMessage = await llm_with_tools.ainvoke(messages_for_llm_cycle, config=config)
            messages_for_llm_cycle.append(ai_response_message) # Добавляем ответ AI в историю цикла

            tool_calls = ai_response_message.tool_calls
            if not tool_calls:
                logger.info(f"[{tenant_id}:{user_id}] LLM ответил без tool_calls на итерации {current_iteration}. Завершение цикла.")
                final_answer_content = str(ai_response_message.content)
                break 
            
            logger.info(f"[{tenant_id}:{user_id}] LLM запросил инструменты на итерации {current_iteration}: {[tc['name'] for tc in tool_calls]}")
            tool_outputs = []
            for tool_call in tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                logger.info(f"Вызов инструмента '{tool_name}' с аргументами: {tool_args}")
                
                found_tool = next((t for t in tenant_tools if t.name == tool_name), None)
                if found_tool:
                    try:
                        current_tool_args_for_invoke = tool_args
                        # Проверка, ожидает ли инструмент аргументы
                        expects_args = bool(found_tool.args_schema and found_tool.args_schema.model_fields)
                        if not expects_args and tool_args and tool_args != {}:
                            logger.info(f"Инструмент {tool_name} не ожидает аргументов, но LLM передал: {tool_args}. Используем {{}}.")
                            current_tool_args_for_invoke = {}
                        
                        # --- Системные параметры из configurable (как и раньше) ---
                        cfg_tenant_id = configurable.get("tenant_id")
                        cfg_client_api_token = configurable.get("client_api_token")
                        cfg_phone_number = configurable.get("phone_number")

                        # --- ДОБАВЛЕНО: Обработка плейсхолдера для номера телефона ---
                        PLACEHOLDER_PHONE = "<to be added by system>"
                        if cfg_phone_number == PLACEHOLDER_PHONE:
                            logger.warning(f"Получен плейсхолдер '{PLACEHOLDER_PHONE}' в качестве номера телефона из конфигурации. Обрабатывается как отсутствующий номер (None).")
                            cfg_phone_number = None
                        # --- КОНЕЦ ДОБАВЛЕНИЯ ---

                        if tool_name in ["get_free_slots_tool", "book_appointment_tool", "book_appointment_ai_payload_tool"]:
                            if not cfg_tenant_id:
                                raise ValueError(f"tenant_id обязателен для {tool_name}, но не найден в configurable.")
                            current_tool_args_for_invoke['tenant_id'] = cfg_tenant_id
                            current_tool_args_for_invoke.setdefault('api_token', cfg_client_api_token) # Устанавливаем, если нет
                            
                            # Добавляем chain_id для формирования комбинированного tenant_id в локальном поиске
                            cfg_chain_id = configurable.get("chain_id")
                            if cfg_chain_id:
                                current_tool_args_for_invoke['chain_id'] = cfg_chain_id
                            
                            if tool_name == "book_appointment_ai_payload_tool":
                                logger.info(f"[ReAct System Override] Перед установкой client_phone_number для book_appointment_ai_payload_tool. cfg_phone_number = '{cfg_phone_number}'") # <--- ДОБАВЛЕН ЛОГ
                                current_tool_args_for_invoke['client_phone_number'] = cfg_phone_number # <-- НОВЫЙ КОД: Принудительная установка
                                if not current_tool_args_for_invoke.get('client_phone_number'): # Проверка после setdefault
                                    # Pydantic возбудит ошибку, если client_phone_number обязателен и None
                                    logger.warning(f"client_phone_number не был предоставлен для book_appointment_ai_payload_tool, Pydantic может вызвать ошибку, если поле обязательное.")
                        
                        # Добавляем tenant_id и api_token для всех инструментов, которые используют новый API
                        elif tool_name == "cancel_appointment_tool":
                            cfg_chain_id = configurable.get("chain_id")
                            if not cfg_chain_id:
                                raise ValueError("chain_id обязателен для cancel_appointment_tool, но не найден в configurable.")
                            current_tool_args_for_invoke['chain_id'] = cfg_chain_id
                            current_tool_args_for_invoke['tenant_id'] = cfg_tenant_id
                            current_tool_args_for_invoke['api_token'] = cfg_client_api_token
                            logger.info(f"[ReAct System Override] Для cancel_appointment_tool: chain_id='{cfg_chain_id}', tenant_id='{cfg_tenant_id}'")

                        elif tool_name in ["find_employees_tool", "get_employee_services_tool", "list_employee_filials_tool"]:
                            if not cfg_tenant_id:
                                raise ValueError(f"tenant_id обязателен для {tool_name}, но не найден в configurable.")
                            current_tool_args_for_invoke['tenant_id'] = cfg_tenant_id
                            current_tool_args_for_invoke['api_token'] = cfg_client_api_token
                            logger.info(f"[ReAct System Override] Для {tool_name}: tenant_id='{cfg_tenant_id}', api_token='{cfg_client_api_token[:50] if cfg_client_api_token else None}...'")


                        # --- Подготовка КОНФИГУРАЦИИ (tool_config) для вызова инструмента ---
                        tool_config_payload = {}
                        if cfg_tenant_id: tool_config_payload["tenant_id"] = cfg_tenant_id
                        if cfg_client_api_token: tool_config_payload["client_api_token"] = cfg_client_api_token
                        if cfg_phone_number: tool_config_payload["phone_number"] = cfg_phone_number
                        
                        final_tool_config_to_pass = {"configurable": tool_config_payload} if tool_config_payload else config
                        
                        # Вызов асинхронного инструмента
                        if inspect.iscoroutinefunction(found_tool.func):
                            output = await found_tool.ainvoke(current_tool_args_for_invoke, config=final_tool_config_to_pass)
                        else: # 
                            output = await found_tool.ainvoke(current_tool_args_for_invoke, config=final_tool_config_to_pass)


                        output_str = str(output)
                        # MAX_TOOL_OUTPUT_LENGTH = 3000
                        # if len(output_str) > MAX_TOOL_OUTPUT_LENGTH:
                        #     output_str = output_str[:MAX_TOOL_OUTPUT_LENGTH] + "\\n[...Вывод инструмента был усечен...]"
                        #     logger.warning(f"Вывод инструмента '{tool_name}' был усечен.")
                        
                        tool_outputs.append(ToolMessage(content=output_str, tool_call_id=tool_call["id"]))
                        logger.info(f"Результат вызова '{tool_name}': {output_str[:100]}...")

                    except Exception as e:
                        error_message = f"Ошибка при вызове инструмента '{tool_name}': {type(e).__name__} - {str(e)}"
                        logger.error(f"Ошибка вызова '{tool_name}' с аргументами {current_tool_args_for_invoke}: {e}", exc_info=True)
                        tool_outputs.append(ToolMessage(content=error_message, tool_call_id=tool_call["id"]))
                else:
                    logger.error(f"Инструмент '{tool_name}' не найден. Доступные: {[t.name for t in tenant_tools]}.")
                    tool_outputs.append(ToolMessage(content=f"Ошибка: Инструмент {tool_name} не найден.", tool_call_id=tool_call["id"]))
            
            messages_for_llm_cycle.extend(tool_outputs) # Добавляем результаты инструментов в историю цикла

        except Exception as e:
            logger.error(f"[{tenant_id}:{user_id}] Ошибка на итерации {current_iteration} ReAct цикла: {e}", exc_info=True)
            final_answer_content = "Произошла ошибка при обработке вашего запроса во время выполнения."
            break # Прерываем цикл при ошибке

        if current_iteration == MAX_ITERATIONS:
            logger.warning(f"[{tenant_id}:{user_id}] Достигнут лимит итераций ({MAX_ITERATIONS}). Завершение цикла.")
            # Пытаемся взять последний ответ LLM, если он есть, или сообщение об ошибке/лимите.
            if ai_response_message and ai_response_message.content and not tool_calls : # если последний ответ был финальным
                 final_answer_content = str(ai_response_message.content)
            elif ai_response_message and ai_response_message.content: # если были tool_calls, но есть какой-то content
                 final_answer_content = str(ai_response_message.content) + "\\n[Достигнут лимит итераций обработки]"
            else:
                 final_answer_content = "Ассистент достиг лимита шагов обработки. Пожалуйста, попробуйте переформулировать запрос или разбить его на части."
            break

    logger.info(f"[{tenant_id}:{user_id}] Итоговый ответ ReAct (первые 100 симв): {final_answer_content[:100]}...")
    return final_answer_content

# === ФУНКЦИИ АНАЛИЗА ЗАВЕРШЕННОСТИ СООБЩЕНИЙ ===

async def analyze_user_message_completeness(
    message: str, 
    user_id: str, 
    tenant_id: str,
    previous_messages: Optional[List[str]] = None,
    time_since_last: Optional[float] = None
) -> Tuple[Optional['MessageAnalysis'], str]:
    """
    Анализирует завершенность сообщения пользователя с накоплением
    
    Args:
        message: Текст сообщения для анализа
        user_id: ID пользователя  
        tenant_id: ID тенанта
        previous_messages: Предыдущие сообщения (опционально)
        time_since_last: Время с последнего сообщения в секундах
        
    Returns:
        Tuple[MessageAnalysis, str]: Результат анализа и итоговое сообщение для ассистента
    """
    logger.info(f"[ДЕБАГ АНАЛИЗА] Входящие параметры:")
    logger.info(f"[ДЕБАГ АНАЛИЗА] - message: '{message}'")
    logger.info(f"[ДЕБАГ АНАЛИЗА] - user_id: '{user_id}'")
    logger.info(f"[ДЕБАГ АНАЛИЗА] - tenant_id: '{tenant_id}'")
    logger.info(f"[ДЕБАГ АНАЛИЗА] - ANALYZER_INITIALIZED: {ANALYZER_INITIALIZED}")
    logger.info(f"[ДЕБАГ АНАЛИЗА] - analyze_message_with_accumulation функция доступна: {'Да' if analyze_message_with_accumulation else 'Нет'}")
    
    if not ANALYZER_INITIALIZED or not analyze_message_with_accumulation:
        logger.warning(f"[{tenant_id}:{user_id}] Анализатор завершенности недоступен")
        return None, message
        
    try:
        logger.info(f"[ДЕБАГ АНАЛИЗА] Вызываем analyze_message_with_accumulation для: '{message[:50]}...'")
        
        analysis_result, final_message = await analyze_message_with_accumulation(
            message=message,
            user_id=f"{tenant_id}:{user_id}",
            previous_messages=previous_messages,
            time_since_last=time_since_last
        )
        
        logger.info(f"[{tenant_id}:{user_id}] Анализ завершенности: {analysis_result.status} "
                   f"(уверенность: {analysis_result.confidence:.2f})")
        logger.info(f"[{tenant_id}:{user_id}] Итоговое сообщение: '{final_message[:100]}...'")
        
        return analysis_result, final_message
        
    except Exception as e:
        logger.error(f"[{tenant_id}:{user_id}] Ошибка анализа завершенности: {e}", exc_info=True)
        return None, message

def should_wait_for_message_completion(analysis: Optional['MessageAnalysis']) -> bool:
    """
    Определяет, нужно ли ждать завершения сообщения
    
    Args:
        analysis: Результат анализа сообщения или None
        
    Returns:
        bool: True если нужно ждать, False если можно отвечать
    """
    if not analysis or not should_wait_for_completion:
        return False
        
    return should_wait_for_completion(analysis)

def get_message_analysis_response(analysis: 'MessageAnalysis', user_id: str) -> str:
    """
    Формирует ответ для неполного сообщения
    
    Args:
        analysis: Результат анализа сообщения
        user_id: ID пользователя
        
    Returns:
        str: Текст ответа для пользователя
    """
    if analysis.status == CompletenessStatus.INCOMPLETE:
        return "Кажется, ваше сообщение не завершено. Продолжайте, пожалуйста..."
    elif analysis.status == CompletenessStatus.WAITING_FOR_MORE:
        return "Я жду продолжения вашего сообщения..."
    elif analysis.status == CompletenessStatus.AMBIGUOUS:
        return "Не совсем понял. Можете продолжить или уточнить?"
    else:
        return "Продолжайте, пожалуйста..."

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

# === ЭКСПОРТ ФУНКЦИЙ ===
# Основные функции для использования в app.py
__all__ = [
    'agent_with_history',
    'GIGACHAT_CREDENTIALS', 
    'trigger_reindex_tenant_async',
    'analyze_user_message_completeness',
    'should_wait_for_message_completion', 
    'get_message_analysis_response',
    'MESSAGE_ANALYZER_AVAILABLE',
    'ANALYZER_INITIALIZED',
    'clear_accumulator'
]

