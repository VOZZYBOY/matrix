# matrixai.py

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from operator import itemgetter
from functools import partial
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage, messages_from_dict, messages_to_dict
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_deepseek import ChatDeepSeek
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)
GIGACHAT_CREDENTIALS = "OTkyYTgyNGYtMjRlNC00MWYyLTg3M2UtYWRkYWVhM2QxNTM1OjA5YWRkODc0LWRjYWItNDI2OC04ZjdmLWE4ZmEwMDIxMThlYw=="
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-1aae129014ac42e3804329d6d44497ce")
if not DEEPSEEK_API_KEY:
    logger.critical("Критическая ошибка: Ключ DeepSeek API не найден (DEEPSEEK_API_KEY).")
    exit()
JSON_DATA_PATH = os.environ.get("JSON_DATA_PATH", "base/cleaned_data.json")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db_clinic_giga")
GIGA_EMBEDDING_MODEL = os.environ.get("GIGA_EMBEDDING_MODEL", "EmbeddingsGigaR")
GIGA_SCOPE = os.environ.get("GIGA_SCOPE", "GIGACHAT_API_PERS")
GIGA_VERIFY_SSL = os.getenv("GIGA_VERIFY_SSL", "False").lower() == "true"
DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
TENANT_COLLECTION_PREFIX = "tenant_" 
try:
    chat_model = ChatDeepSeek(
        model=DEEPSEEK_CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY,
        temperature=0.1,
        max_tokens=4096,
    )
    logger.info(f"Чат модель DeepSeek '{DEEPSEEK_CHAT_MODEL}' инициализирована.")
except Exception as e:
    logger.critical(f"Ошибка инициализации модели DeepSeek Chat: {e}", exc_info=True)
    exit()
chroma_client_global: Optional[chromadb.ClientAPI] = None
embeddings_object_global: Optional[GigaChatEmbeddings] = None
bm25_retrievers_map_global: Dict[str, BM25Retriever] = {}
tenant_documents_map_global: Dict[str, List[Document]] = {}
tenant_raw_data_map_global: Dict[str, List[Dict]] = {}
search_k_global = 5
def initialize_rag_components():
    """Инициализирует RAG компоненты и сохраняет их в глобальные переменные."""
    global chroma_client_global, embeddings_object_global
    global bm25_retrievers_map_global, tenant_documents_map_global
    global tenant_raw_data_map_global
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
        chroma_client, embeddings, bm25_map, tenant_docs_map, tenant_raw_map = rag_setup.initialize_rag(
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
        if chroma_client and embeddings:
            chroma_client_global = chroma_client
            embeddings_object_global = embeddings
            bm25_retrievers_map_global = bm25_map
            tenant_documents_map_global = tenant_docs_map
            tenant_raw_data_map_global = tenant_raw_map
            logger.info(f"Инициализация RAG завершена. Загружено:")
            logger.info(f"  - Chroma клиент: {'Да' if chroma_client_global else 'Нет'}")
            logger.info(f"  - Embeddings: {'Да' if embeddings_object_global else 'Нет'}")
            logger.info(f"  - BM25 ретриверы: {len(bm25_retrievers_map_global)} шт.")
            logger.info(f"  - Карта документов тенантов: {len(tenant_documents_map_global)} тенантов.")
            logger.info(f"  - Карта сырых данных тенантов: {len(tenant_raw_data_map_global)} тенантов.")
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
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")
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
def find_specialists_by_service_or_category_and_filial_tool(query_term: str, filial_name: str) -> str:
    """Ищет СПЕЦИАЛИСТОВ по УСЛУГЕ/КАТЕГОРИИ в КОНКРЕТНОМ филиале."""
    handler = clinic_functions.FindSpecialistsByServiceOrCategoryAndFilial(query_term=query_term.lower(), filial_name=filial_name.lower())
    return handler.process()
class ListServicesInCategoryArgs(BaseModel):
    category_name: str = Field(description="Точное название категории")
def list_services_in_category_tool(category_name: str) -> str:
    """Возвращает список КОНКРЕТНЫХ услуг в указанной КАТЕГОРИИ."""
    handler = clinic_functions.ListServicesInCategory(category_name=category_name)
    return handler.process()
class ListServicesInFilialArgs(BaseModel):
    filial_name: str = Field(description="Точное название филиала")
def list_services_in_filial_tool(filial_name: str) -> str:
    """Возвращает ПОЛНЫЙ список УНИКАЛЬНЫХ услуг в КОНКРЕТНОМ филиале."""
    handler = clinic_functions.ListServicesInFilial(filial_name=filial_name)
    return handler.process()
class FindServicesInPriceRangeArgs(BaseModel):
    min_price: float = Field(description="Минимальная цена")
    max_price: float = Field(description="Максимальная цена")
    category_name: Optional[str] = Field(default=None, description="Опционально: категория")
    filial_name: Optional[str] = Field(default=None, description="Опционально: филиал")
def find_services_in_price_range_tool(min_price: float, max_price: float, category_name: Optional[str] = None, filial_name: Optional[str] = None) -> str:
    """Ищет услуги в ЗАДАННОМ ЦЕНОВОМ ДИАПАЗОНЕ."""
    handler = clinic_functions.FindServicesInPriceRange(min_price=min_price, max_price=max_price, category_name=category_name, filial_name=filial_name)
    return handler.process()
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
    ListEmployeeFilialsArgs,
]
logger.info(f"Определено {len(TOOL_CLASSES)} Pydantic классов для аргументов инструментов.")
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
ВЫБОР МЕЖДУ RAG, FUNCTION CALLING, ПАМЯТЬЮ ДИАЛОГА ИЛИ ПРЯМЫМ ОТВЕТОМ:
- ПАМЯТЬ ДИАЛОГА: Для ответов на вопросы, связанные с предыдущим контекстом (местоимения "он/она/это", короткие вопросы "где?", "цена?", "кто?"), и для вопросов о самом пользователе.
- RAG (Поиск по базе знаний): Используй ТОЛЬКО для ЗАПРОСОВ **ОПИСАНИЯ** услуг, врачей ИЛИ **ОБЩЕЙ ИНФОРМАЦИИ О КЛИНИКЕ** (например, "расскажи о компании", "какой у вас подход?", "сколько филиалов?"). Я предоставлю контекст. Синтезируй ответ на его основе.
- FUNCTION CALLING (Вызов Инструментов): Используй ТОЛЬКО для запросов КОНКРЕТНЫХ ДАННЫХ: цены, списки врачей/услуг/филиалов, проверка наличия, сравнение цен, ПОЛНЫЙ список филиалов сотрудника. Используй правильный инструмент.
- ПРЯМОЙ ОТВЕТ: Для приветствий, прощаний, простых уточнений или вопросов не по теме.

ПРАВИЛА FUNCTION CALLING:
- **Приоритет над RAG для списков и конкретных данных:** Если запрос касается получения СПИСКА (услуг сотрудника, филиалов сотрудника, цен, списка сотрудников по критерию и т.д.) или КОНКРЕТНЫХ ДАННЫХ, ВЫЗОВ СООТВЕТСТВУЮЩЕГО ИНСТРУМЕНТА ЯВЛЯЕТСЯ ОБЯЗАТЕЛЬНЫМ, ДАЖЕ ЕСЛИ RAG-КОНТЕКСТ СОДЕРЖИТ ПОХОЖУЮ ИНФОРМАЦИЮ. Это гарантирует использование актуальных данных и форматирования из инструмента.
- **Обязательность вызова для списков:** Если пользователь спрашивает список (услуг, врачей, филиалов, категорий) или просит найти что-то по критериям, ТЫ ОБЯЗАН ВЫЗВАТЬ СООТВЕТСТВУЮЩИЙ ИНСТРУМЕНТ. Особенно:
    - Для вопроса о ВСЕХ филиалах сотрудника ('где еще работает?', 'в каких филиалах?') -> ОБЯЗАТЕЛЬНО вызывай `list_employee_filials_tool`.
    - Для вопроса о ВСЕХ услугах сотрудника (например, "какие услуги выполняет ХХХ?", "чем занимается YYY?") -> ОБЯЗАТЕЛЬНО вызывай `get_employee_services_tool`. Даже если RAG-контекст содержит упоминания или частичный список услуг этого сотрудника (например, в его общем описании), для получения ПОЛНОГО и ТОЧНОГО списка услуг используй ИСКЛЮЧИТЕЛЬНО этот инструмент.
    - Для вопроса о ЦЕНЕ КОНКРЕТНОЙ услуги -> ОБЯЗАТЕЛЬНО вызывай `get_service_price_tool`.
- Точность Параметров: Извлекай параметры ТОЧНО из запроса и ИСТОРИИ.
- Не Выдумывай Параметры: Если обязательного параметра нет, НЕ ВЫЗЫВАЙ функцию, а вежливо попроси уточнить.
- ОБРАБОТКА НЕУДАЧНЫХ ВЫЗОВОВ: Если инструмент вернул ошибку или 'не найдено', НЕ ПЫТАЙСЯ вызвать его с теми же аргументами. Сообщи пользователю или предложи альтернативу.
- Интерпретация Результатов: Представляй результаты функций в понятной, человеческой форме.

ОБЩИЕ ПРАВИЛА:
- Точность: НЕ ПРИДУМЫВАЙ.
- Краткость и Ясность.
- Вежливость.
- Медицинские Советы: НЕ ДАВАЙ.

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
def run_agent_like_chain(input_dict: Dict, config: RunnableConfig) -> str:
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
    if not chroma_client_global or not embeddings_object_global:
        logger.error(f"RAG компоненты (Chroma/Embeddings) не инициализированы.")
        return "Ошибка: Базовые компоненты RAG не готовы."
    if not tenant_id:
        logger.error("Tenant ID не найден в config.")
        return "Ошибка: Не удалось определить тенанта."
    bm25_retriever = bm25_retrievers_map_global.get(tenant_id)
    collection_name = f"{TENANT_COLLECTION_PREFIX}{tenant_id}"
    if not chroma_client_global or not embeddings_object_global:
        logger.error("Глобальный Chroma клиент или эмбеддинги не инициализированы.")
        return "Ошибка: Глобальные компоненты RAG не готовы."
    try:
        embeddings_wrapper = ChromaGigaEmbeddingsWrapper(embeddings_object_global)
    except ValueError as e:
         logger.error(f"Ошибка создания обертки эмбеддингов: {e}")
         return "Ошибка: Некорректный объект эмбеддингов."
    try:
        chroma_collection = chroma_client_global.get_collection(
            name=collection_name,
            embedding_function=embeddings_wrapper
        )
        chroma_vectorstore = Chroma(
            client=chroma_client_global,
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
                "Если запрос не требует поиска описания в базе знаний (например, 'привет', 'какая цена на Х?', 'найди врачей Y в филиале Z', 'какие есть филиалы?', 'где еще работает доктор А?', 'сравни цены на Y в филиалах M и N'), "
                "то best_rag_query должен быть пустой строкой или просто исходным запросом."
            ))
        ]
        if history_messages:
             rag_prompt_messages.extend(history_messages)
        rag_prompt_messages.append(HumanMessage(content=question))
        logger.debug(f"[{tenant_id}:{user_id}] Вызов LLM для генерации RAG-запроса...")
        rag_thought_result = rag_query_llm.invoke(rag_prompt_messages, config=config)
        if isinstance(rag_thought_result, RagQueryThought):
            rag_thought = rag_thought_result
            if rag_thought.best_rag_query and rag_thought.best_rag_query.strip().lower() != question.strip().lower():
                 effective_rag_query = rag_thought.best_rag_query
                 logger.info(f"[{tenant_id}:{user_id}] Сгенерирован улучшенный RAG-запрос: '{effective_rag_query}'")
                 logger.debug(f"[{tenant_id}:{user_id}] Анализ LLM для RAG: {rag_thought.analysis}")
            else:
                 logger.info(f"[{tenant_id}:{user_id}] LLM не сгенерировал специфичный RAG-запрос, используем исходный для RAG: '{question[:50]}...'")
        else:
            logger.warning(f"[{tenant_id}:{user_id}] LLM для генерации RAG-запроса вернул неожиданный тип: {type(rag_thought_result)}. Используем исходный запрос.")
    except Exception as e:
        logger.warning(f"[{tenant_id}:{user_id}] Исключение при улучшении RAG-запроса: {e}. Используем исходный: '{question[:50]}...'", exc_info=True)
    try:
        relevant_docs = final_retriever.invoke(effective_rag_query, config=config)
        rag_context = format_docs(relevant_docs)
        logger.info(f"RAG: Найдено {len(relevant_docs)} док-в для запроса: '{effective_rag_query[:50]}...'. Контекст: {len(rag_context)} симв.")
    except Exception as e:
        logger.error(f"Ошибка выполнения RAG поиска для тенанта {tenant_id}: {e}", exc_info=True)
        rag_context = "[Ошибка получения информации из базы знаний]"
    system_prompt = SYSTEM_PROMPT
    prompt_addition = None
    if tenant_config_manager:
        settings = tenant_config_manager.load_tenant_settings(tenant_id)
        prompt_addition = settings.get('prompt_addition')
        if prompt_addition:
            system_prompt += f"\n\n[Дополнительные инструкции от администратора филиала {tenant_id}]:\n{prompt_addition}"
            logger.info(f"Добавлено дополнение к промпту для тенанта {tenant_id}.")
    tenant_tools = []
    tenant_specific_docs: Optional[List[Document]] = tenant_documents_map_global.get(tenant_id)
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
        list_all_categories_tool: None, 
        list_employee_filials_tool: ListEmployeeFilialsArgs,
    }
    def create_tool_wrapper(original_tool_func: callable, data_docs: Optional[List[Document]], raw_data: Optional[List[Dict]]):
        def actual_wrapper(*args, **kwargs) -> str:
            tool_name = original_tool_func.__name__
            logger.info(f"Вызов обертки для инструмента {tool_name} с args: {args}, kwargs: {kwargs}")
            try:
                logger.info(f"Вызов обертки для инструмента {tool_name} с args: {args}, kwargs: {kwargs}")
                handler_class_name = ''.join(word.capitalize() for word in tool_name.replace('_tool', '').split('_'))
                HandlerClass = getattr(clinic_functions, handler_class_name, None)
                if not HandlerClass:
                    logger.error(f"Не найден класс-обработчик '{handler_class_name}' в clinic_functions для функции {tool_name}")
                    return f"Ошибка: Некорректная конфигурация инструмента {tool_name}."
                handler_instance = HandlerClass(**kwargs)
                if hasattr(handler_instance, 'process') and callable(getattr(handler_instance, 'process')):
                    if data_docs is not None:
                         logger.debug(f"Передача {len(data_docs)} документов в {handler_class_name}.process")
                         return handler_instance.process(
                             tenant_data_docs=data_docs,
                             raw_data=raw_data
                         )
                    else:
                         logger.warning(f"{handler_class_name} {tool_name}")
                         return f"Ошибка: Отсутствуют данные тенанта для инструмента {tool_name}."
                else:
                     logger.error(f"{handler_class_name}")
                     return f"Ошибка: Некорректная конфигурация инструмента {handler_class_name}."
            except Exception as e:
                logger.error(f"{tool_name}: {e}", exc_info=True)
                return f"При выполнении инструмента {tool_name} произошла ошибка."
        return actual_wrapper
    for tool_function in TOOL_FUNCTIONS:
        tool_name = tool_function.__name__
        tool_description = tool_function.__doc__ or f"Инструмент {tool_name}"
        args_schema = tool_func_to_schema_map.get(tool_function)
        wrapped_func = create_tool_wrapper(tool_function, tenant_specific_docs, tenant_raw_data_map_global.get(tenant_id, []))
        langchain_tool = StructuredTool.from_function(
            func=wrapped_func,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema
        )
        tenant_tools.append(langchain_tool)
        logger.debug(f"{tool_name} {tenant_id}. Schema: {args_schema.__name__ if args_schema else 'None'}")
    messages_for_llm = []
    messages_for_llm.append(SystemMessage(content=system_prompt))
    messages_for_llm.extend(history_messages)
    rag_context_block = f"\n\n[Информация из базы знаний (по запросу '{effective_rag_query}')]:\n{rag_context}\n[/Информация из базы знаний]"
    messages_for_llm.append(HumanMessage(content=question + rag_context_block))
    llm_with_tools = chat_model.bind_tools(tenant_tools)
    logger.info(f"{chat_model.model_name} {len(tenant_tools)} {tenant_id}...")
    try:
        ai_response_message: AIMessage = llm_with_tools.invoke(messages_for_llm, config=config)
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
                          output = found_tool.invoke(tool_args, config=config)
                          output_str = str(output)
                          tool_outputs.append(ToolMessage(content=output_str, tool_call_id=tool_call["id"]))
                          logger.info(f"{tool_name}: {output_str[:100]}...")
                      except Exception as e:
                           error_message = f"Ошибка: Инструмент '{tool_name}' не смог успешно выполнить действие."
                           logger.error(f"{tool_name} {e}", exc_info=True)
                           tool_outputs.append(ToolMessage(content=error_message, tool_call_id=tool_call["id"]))
                  else:
                      logger.error(f"{tool_name} {tenant_tools}.")
                      tool_outputs.append(ToolMessage(content=f"Ошибка: Инструмент {tool_name} не найден.", tool_call_id=tool_call["id"]))
             messages_for_llm.append(ai_response_message)
             messages_for_llm.extend(tool_outputs)
             logger.info("")
             final_response_message = llm_with_tools.invoke(messages_for_llm, config=config)
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
    if not chroma_client_global or not embeddings_object_global:
        logger.error(f"[Async Trigger] Глобальные RAG компоненты (Chroma/Embeddings) не инициализированы. Переиндексация для {tenant_id} отменена.")
        return False

    # Получаем необходимые глобальные переменные и конфигурации
    global bm25_retrievers_map_global, tenant_documents_map_global, tenant_raw_data_map_global
    global search_k_global

    # Эти значения обычно устанавливаются при инициализации, но для полноты можно их передать явно
    # или убедиться, что они доступны в этой области видимости.
    # В данном случае они глобальные, но для reindex_tenant_specific_data нужны их значения.
    data_dir_base = os.getenv("BASE_DATA_DIR", "base")
    chunk_size_cfg = int(os.getenv("CHUNK_SIZE", 1000))
    chunk_overlap_cfg = int(os.getenv("CHUNK_OVERLAP", 200))
    # search_k_global уже есть

    try:
        # Запускаем синхронную функцию reindex_tenant_specific_data в отдельном потоке,
        # чтобы не блокировать основной цикл asyncio.
        # asyncio.to_thread доступен в Python 3.9+
        # Для более старых версий Python можно использовать loop.run_in_executor(None, ...)
        import asyncio # Импортируем здесь, чтобы быть уверенным в доступности в этой функции

        success = await asyncio.to_thread(
            rag_setup.reindex_tenant_specific_data,
            tenant_id=tenant_id,
            chroma_client=chroma_client_global,
            embeddings_object=embeddings_object_global,
            bm25_retrievers_map=bm25_retrievers_map_global,
            tenant_documents_map=tenant_documents_map_global,
            tenant_raw_data_map=tenant_raw_data_map_global,
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

# --- Конец: Асинхронный триггер для переиндексации данных одного тенанта ---

# Инициализация RAG должна быть вызвана до создания agent_runnable
# initialize_rag_components() # Убедимся, что она вызывается один раз, если уже не сделано ранее

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

