# matrixai.py

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from operator import itemgetter

from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableConfig
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field

try:
    import rag_setup
except ImportError as e:
     logging.critical(f"Критическая ошибка: Не удалось импортировать 'rag_setup'. Убедитесь, что файл rag_setup.py существует. Ошибка: {e}", exc_info=True)
     exit()
try:
    import clinic_functions
    if not hasattr(clinic_functions, 'set_clinic_data') or not callable(clinic_functions.set_clinic_data):
         logging.critical("Критическая ошибка: Функция 'set_clinic_data' не найдена в 'clinic_functions.py'.")
         exit()
except ImportError as e:
    logging.critical(f"Критическая ошибка: Не удалось импортировать 'clinic_functions'. Ошибка: {e}", exc_info=True)
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s:%(name)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Credentials ---
GIGACHAT_CREDENTIALS = os.environ.get("GIGACHAT_CREDENTIALS", "OTkyYTgyNGYtMjRlNC00MWYyLTg3M2UtYWRkYWVhM2QxNTM1OjA5YWRkODc0LWRjYWItNDI2OC04ZjdmLWE4ZmEwMDIxMThlYw==")
if not GIGACHAT_CREDENTIALS:
    logger.critical("Критическая ошибка: Учетные данные GigaChat не найдены (GIGACHAT_CREDENTIALS).")
    exit()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-1aae129014ac42e3804329d6d44497ce")
if not DEEPSEEK_API_KEY:
    logger.critical("Критическая ошибка: Ключ DeepSeek API не найден (DEEPSEEK_API_KEY).")
    exit()

# --- Constants ---
JSON_DATA_PATH = os.environ.get("JSON_DATA_PATH", "base/cleaned_data.json")
CHROMA_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db_clinic_giga")
GIGA_EMBEDDING_MODEL = os.environ.get("GIGA_EMBEDDING_MODEL", "EmbeddingsGigaR")
GIGA_SCOPE = os.environ.get("GIGA_SCOPE", "GIGACHAT_API_PERS")
DEEPSEEK_CHAT_MODEL = os.environ.get("DEEPSEEK_CHAT_MODEL", "deepseek-chat")

# --- Initialize LLM ---
try:
    chat_model = ChatDeepSeek(
        model=DEEPSEEK_CHAT_MODEL,
        api_key=DEEPSEEK_API_KEY,
        temperature=0,
        max_tokens=4096,
    )
    logger.info(f"Чат модель DeepSeek '{DEEPSEEK_CHAT_MODEL}' инициализирована.")
except Exception as e:
    logger.critical(f"Ошибка инициализации модели DeepSeek Chat: {e}", exc_info=True)
    exit()

# --- Initialize RAG & Load Data ---
retriever = None
embeddings_model_obj = None
prepared_docs_for_bm25 = None
clinic_data_for_functions = None
try:
    logger.info("Инициализация RAG системы через rag_setup...")
    rag_components = rag_setup.initialize_rag(
        json_data_path=JSON_DATA_PATH,
        chroma_persist_dir=CHROMA_PERSIST_DIR,
        embedding_credentials=GIGACHAT_CREDENTIALS,
        embedding_model=GIGA_EMBEDDING_MODEL,
        embedding_scope=GIGA_SCOPE,
        verify_ssl_certs=False,
        search_k= 5 
    )
    if isinstance(rag_components, tuple) and len(rag_components) >= 4:
         retriever, embeddings_model_obj, prepared_docs_for_bm25, clinic_data_for_functions = rag_components[:4]
    else:
         raise RuntimeError(f"Функция rag_setup.initialize_rag вернула неожиданный результат: {rag_components}")

    if retriever is None or clinic_data_for_functions is None:
         raise RuntimeError("Не удалось инициализировать RAG ретривер или загрузить данные клиники из rag_setup.")
    logger.info("RAG система успешно инициализирована.")

    clinic_functions.set_clinic_data(clinic_data_for_functions)
    logger.info("Данные клиники переданы в модуль функций.")
except Exception as e:
    logger.critical(f"Критическая ошибка во время инициализации RAG или передачи данных: {e}", exc_info=True)
    exit()

# --- Pydantic model for RAG Query Improvement ---
class RagQueryThought(BaseModel):
    analysis: str = Field(description="Анализ последнего запроса пользователя в контексте истории диалога. Если пользователь ссылается на что-то из истории (например, номер пункта, местоимение 'он'/'она'/'это'), явно укажи, к какой сущности (услуге, врачу, филиалу) это относится, используя ее полное название из истории.")
    best_rag_query: str = Field(description="Сформулируй оптимальный, самодостаточный запрос для поиска описания или детальной информации об основной сущности запроса в векторной базе знаний. Используй полное имя/название сущности. Если запрос общий (приветствие, не по теме) или явно запрашивает вызов функции (цена, список филиалов/услуг/врачей и т.д.), оставь поле пустым или верни исходный запрос.")

# --- Tool Definitions ---
# ----- НАЧАЛО БЛОКА ИНСТРУМЕНТОВ -----
class FindEmployeesArgs(BaseModel):
    employee_name: Optional[str] = Field(default=None, description="Часть или полное ФИО сотрудника")
    service_name: Optional[str] = Field(default=None, description="Точное или частичное название услуги")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала")

@tool("find_employees", args_schema=FindEmployeesArgs)
def find_employees_tool(employee_name: Optional[str] = None, service_name: Optional[str] = None, filial_name: Optional[str] = None) -> str:
    """Ищет сотрудников клиники по ФИО, выполняемой услуге или филиалу."""
    handler = clinic_functions.FindEmployees(employee_name=employee_name, service_name=service_name, filial_name=filial_name)
    return handler.process()

class GetServicePriceArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги (например, 'Soprano Пальцы для женщин')")
    filial_name: Optional[str] = Field(default=None, description="Точное название филиала (например, 'Москва-сити'), если нужно уточнить цену в конкретном месте")

@tool("get_service_price", args_schema=GetServicePriceArgs)
def get_service_price_tool(service_name: str, filial_name: Optional[str] = None) -> str:
    """Возвращает цену на КОНКРЕТНУЮ услугу клиники."""
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
    """Возвращает список услуг КОНКРЕТНОГО сотрудника."""
    handler = clinic_functions.GetEmployeeServices(employee_name=employee_name)
    return handler.process()

class CheckServiceInFilialArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_name: str = Field(description="Точное название филиала")

@tool("check_service_in_filial", args_schema=CheckServiceInFilialArgs)
def check_service_in_filial_tool(service_name: str, filial_name: str) -> str:
    """Проверяет, доступна ли КОНКРЕТНАЯ услуга в КОНКРЕТНОМ филиале."""
    handler = clinic_functions.CheckServiceInFilial(service_name=service_name, filial_name=filial_name)
    return handler.process()

class CompareServicePriceInFilialsArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")
    filial_names: List[str] = Field(min_length=2, description="Список из ДВУХ или БОЛЕЕ названий филиалов")

@tool("compare_service_price_in_filials", args_schema=CompareServicePriceInFilialsArgs)
def compare_service_price_in_filials_tool(service_name: str, filial_names: List[str]) -> str:
    """Сравнивает цену КОНКРЕТНОЙ услуги в НЕСКОЛЬКИХ филиалах."""
    handler = clinic_functions.CompareServicePriceInFilials(service_name=service_name, filial_names=filial_names)
    return handler.process()

class FindServiceLocationsArgs(BaseModel):
    service_name: str = Field(description="Точное или максимально близкое название услуги")

@tool("find_service_locations", args_schema=FindServiceLocationsArgs)
def find_service_locations_tool(service_name: str) -> str:
    """Ищет все филиалы, где доступна КОНКРЕТНАЯ услуга."""
    handler = clinic_functions.FindServiceLocations(service_name=service_name)
    return handler.process()

class FindSpecialistsByServiceOrCategoryAndFilialArgs(BaseModel):
    query_term: str = Field(description="Название услуги ИЛИ категории")
    filial_name: str = Field(description="Точное название филиала")

@tool("find_specialists_by_service_or_category_and_filial", args_schema=FindSpecialistsByServiceOrCategoryAndFilialArgs)
def find_specialists_by_service_or_category_and_filial_tool(query_term: str, filial_name: str) -> str:
    """Ищет СПЕЦИАЛИСТОВ по УСЛУГЕ/КАТЕГОРИИ в КОНКРЕТНОМ филиале."""
    handler = clinic_functions.FindSpecialistsByServiceOrCategoryAndFilial(query_term=query_term.lower(), filial_name=filial_name.lower())
    return handler.process()

class ListServicesInCategoryArgs(BaseModel):
    category_name: str = Field(description="Точное название категории")

@tool("list_services_in_category", args_schema=ListServicesInCategoryArgs)
def list_services_in_category_tool(category_name: str) -> str:
    """Возвращает список КОНКРЕТНЫХ услуг в указанной КАТЕГОРИИ."""
    handler = clinic_functions.ListServicesInCategory(category_name=category_name)
    return handler.process()

class ListServicesInFilialArgs(BaseModel):
    filial_name: str = Field(description="Точное название филиала")

@tool("list_services_in_filial", args_schema=ListServicesInFilialArgs)
def list_services_in_filial_tool(filial_name: str) -> str:
    """Возвращает ПОЛНЫЙ список УНИКАЛЬНЫХ услуг в КОНКРЕТНОМ филиале."""
    handler = clinic_functions.ListServicesInFilial(filial_name=filial_name)
    return handler.process()

class FindServicesInPriceRangeArgs(BaseModel):
    min_price: float = Field(description="Минимальная цена")
    max_price: float = Field(description="Максимальная цена")
    category_name: Optional[str] = Field(default=None, description="Опционально: категория")
    filial_name: Optional[str] = Field(default=None, description="Опционально: филиал")

@tool("find_services_in_price_range", args_schema=FindServicesInPriceRangeArgs)
def find_services_in_price_range_tool(min_price: float, max_price: float, category_name: Optional[str] = None, filial_name: Optional[str] = None) -> str:
    """Ищет услуги в ЗАДАННОМ ЦЕНОВОМ ДИАПАЗОНЕ."""
    handler = clinic_functions.FindServicesInPriceRange(min_price=min_price, max_price=max_price, category_name=category_name, filial_name=filial_name)
    return handler.process()

@tool("list_all_categories")
def list_all_categories_tool() -> str:
    """Возвращает список ВСЕХ категорий услуг."""
    handler = clinic_functions.ListAllCategories()
    return handler.process()

class ListEmployeeFilialsArgs(BaseModel):
    employee_name: str = Field(description="Точное или близкое ФИО сотрудника")

@tool("list_employee_filials", args_schema=ListEmployeeFilialsArgs)
def list_employee_filials_tool(employee_name: str) -> str:
    """ОБЯЗАТЕЛЬНО ВЫЗЫВАЙ для получения ПОЛНОГО списка ВСЕХ филиалов КОНКРЕТНОГО сотрудника."""
    logger.info(f"Вызов list_employee_filials_tool для: {employee_name}")
    handler = clinic_functions.ListEmployeeFilials(employee_name=employee_name)
    return handler.process()
# ----- КОНЕЦ БЛОКА ИНСТРУМЕНТОВ -----


# --- Tools List ---
tools = [
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
logger.info(f"Загружено {len(tools)} инструментов (функций).")

# --- Bind Tools to LLM ---
llm_with_tools = chat_model.bind_tools(tools)
logger.info(f"Инструменты привязаны к модели {DEEPSEEK_CHAT_MODEL}.")

# --- System Prompt ---
SYSTEM_PROMPT = """Ты - вежливый, ОЧЕНЬ ВНИМАТЕЛЬНЫЙ и информативный ИИ-ассистент медицинской клиники "Med YU Med".
Твоя главная задача - помогать пользователям, отвечая на их вопросы об услугах, ценах, специалистах и филиалах клиники, И ПОДДЕРЖИВАТЬ ЕСТЕСТВЕННЫЙ ДИАЛОГ.
ИСПОЛЬЗУЙ RAG ПОИСК ТОЛЬКО ДЛЯ ОПИСАНИЙ УСЛУГ И ВРАЧЕЙ, А НЕ ДЛЯ КОНКРЕТНЫХ ДАННЫХ (ЦЕНЫ, СПИСКИ ВРАЧЕЙ, ФИЛИАЛОВ И Т.Д.). СТАРАЙСЯ ВСЕ РЕШАТЬ ЧЕРЕЗ ВЫЗОВ ФУНКЦИЙ (ИНСТРУМЕНТОВ).

КЛЮЧЕВЫЕ ПРАВИЛА РАБОТЫ:

АНАЛИЗ ИСТОРИИ И ВЫБОР ДЕЙСТВИЯ:
- Внимательно проанализируй ПОЛНУЮ ИСТОРИЮ ДИАЛОГА (chat_history).
- ИСПОЛЬЗУЙ КОНТЕКСТ ИСТОРИИ! Не переспрашивай.
- ЗАПОМИНАЙ ИМЯ ПОЛЬЗОВАТЕЛЯ, если он представился.

ВЫБОР МЕЖДУ RAG, FUNCTION CALLING, ПАМЯТЬЮ ДИАЛОГА ИЛИ ПРЯМЫМ ОТВЕТОМ:
- ПАМЯТЬ ДИАЛОГА: Для ответов на вопросы, связанные с предыдущим контекстом (местоимения "он/она/это", короткие вопросы "где?", "цена?", "кто?"), и для вопросов о самом пользователе.
- RAG (Поиск по базе знаний): Используй ТОЛЬКО для ЗАПРОСОВ ОПИСАНИЯ услуг или врачей ("Что такое X?", "Расскажи про Y", "Подробнее о Z"). Я предоставлю контекст. Синтезируй ответ на его основе.
- FUNCTION CALLING (Вызов Инструментов): Используй ТОЛЬКО для запросов КОНКРЕТНЫХ ДАННЫХ: цены, списки врачей/услуг/филиалов, проверка наличия, сравнение цен, ПОЛНЫЙ список филиалов сотрудника. Используй правильный инструмент.
- ПРЯМОЙ ОТВЕТ: Для приветствий, прощаний, простых уточнений или вопросов не по теме.

ПРАВИЛА FUNCTION CALLING:
- **Обязательность вызова для списков:** Если пользователь спрашивает список (услуг, врачей, филиалов, категорий) или просит найти что-то по критериям, ТЫ ОБЯЗАН ВЫЗВАТЬ СООТВЕТСТВУЮЩИЙ ИНСТРУМЕНТ. Особенно:
    - Для вопроса о ВСЕХ филиалах сотрудника ('где еще работает?', 'в каких филиалах?') -> ОБЯЗАТЕЛЬНО вызывай `list_employee_filials`.
    - Для вопроса о ВСЕХ услугах сотрудника -> ОБЯЗАТЕЛЬНО вызывай `get_employee_services`.
    - Для вопроса о ЦЕНЕ КОНКРЕТНОЙ услуги -> ОБЯЗАТЕЛЬНО вызывай `get_service_price`.
- Точность Параметров: Извлекай параметры ТОЧНО из запроса и ИСТОРИИ.
- Не Выдумывай Параметры: Если обязательного параметра нет, НЕ ВЫЗЫВАЙ функцию, а вежливо попроси уточнить.
- ОБРАБОТКА НЕУДАЧНЫХ ВЫЗОВОВ: Если инструмент вернул ошибку или 'не найдено', НЕ ПЫТАЙСЯ вызвать его снова с теми же аргументами. Сообщи пользователю или предложи альтернативу.
- Интерпретация Результатов: Представляй результаты функций в понятной, человеческой форме.

ОБЩИЕ ПРАВИЛА:
- Точность: НЕ ПРИДУМЫВАЙ.
- Краткость и Ясность.
- Вежливость.
- Медицинские Советы: НЕ ДАВАЙ.

ВАЖНО: Всегда сначала анализируй историю и цель пользователя. Реши, нужен ли ответ из памяти, RAG, вызов функции или простой ответ. Действуй соответственно.
"""

context_chain = (
    RunnablePassthrough.assign(
        query_with_instruction=itemgetter("input") | RunnableLambda(rag_setup.add_instruction_to_query)
    ).assign(
        docs=itemgetter("query_with_instruction") | retriever
    ).assign(
        context=itemgetter("docs") | RunnableLambda(rag_setup.format_docs)
    )
    | itemgetter("context")
)
logger.info("Цепочка RAG-контекста настроена (с добавлением инструкции).")


# --- Chat History Management ---
chat_memory = {}
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in chat_memory:
        chat_memory[session_id] = ChatMessageHistory()
        logger.info(f"Создана новая история чата для сессии: {session_id}")
    return chat_memory[session_id]

# --- Agent Execution Function ---
def run_agent_like_chain(input_dict: Dict, config: RunnableConfig) -> str:
    session_id = config["configurable"]["session_id"]
    user_input = input_dict["input"]
    logger.debug(f"[{session_id}] Вход: {user_input}")

    # Step 1: Improve RAG query using CoT
    effective_rag_query = user_input
    try:
        rag_query_llm = chat_model.with_structured_output(RagQueryThought)
        current_history = get_session_history(session_id).messages
        rag_prompt_messages = [
            SystemMessage(content=(
                "Твоя задача - проанализировать последний запрос пользователя в контексте предыдущего диалога. "
                "Определи главную сущность (услуга, врач, филиал, категория), о которой спрашивает пользователь, "
                "особенно если он использует ссылки на историю (номер пункта, местоимения). "
                "Затем сформулируй оптимальный, самодостаточный поисковый запрос для векторной базы знаний, "
                "чтобы найти описание или детали этой сущности. Используй полные названия."
                "Пример: Если история содержит '1. Услуга А\n2. Услуга Б', а пользователь спрашивает 'расскажи о 2', "
                "то best_rag_query должен быть 'Описание Услуга Б'."
                "Если запрос не требует поиска описания (например, 'привет', 'цена Х', 'найди врачей Y', 'какие филиалы?', 'где еще работает Х?'), "
                "то best_rag_query должен быть пустой строкой или исходным запросом."
            )),
            *current_history,
            HumanMessage(content=user_input)
        ]
        rag_thought: RagQueryThought = rag_query_llm.invoke(rag_prompt_messages, config=config)
        if rag_thought.best_rag_query and rag_thought.best_rag_query.strip().lower() != user_input.lower():
             effective_rag_query = rag_thought.best_rag_query
             logger.info(f"[{session_id}] Сгенерирован улучшенный RAG-запрос: '{effective_rag_query}'")
             logger.debug(f"[{session_id}] Анализ LLM для RAG: {rag_thought.analysis}")
        else:
             logger.info(f"[{session_id}] LLM не сгенерировал специфичный RAG-запрос, используем исходный для RAG.")
             effective_rag_query = user_input

    except Exception as e:
        logger.warning(f"[{session_id}] Не удалось улучшить RAG-запрос: {e}. Используем исходный: '{user_input}'")
        effective_rag_query = user_input

    # Step 2: Perform RAG search (with potentially improved query)
    try:
        # Вызываем context_chain, который теперь внутри себя добавляет инструкцию
        rag_context = context_chain.invoke({"input": effective_rag_query}, config=config)
        logger.debug(f"[{session_id}] Получен RAG контекст для запроса '{effective_rag_query}': {rag_context[:200]}...")
    except Exception as e:
        logger.error(f"[{session_id}] Ошибка получения RAG контекста: {e}")
        rag_context = "Ошибка при получении информации из базы знаний."

    # Step 3: Main LLM loop with tools
    context_block = f"\n\n[Информация из базы знаний (по запросу '{effective_rag_query}')]:\n{rag_context}\n[/Информация из базы знаний]"
    current_history_for_main_llm = get_session_history(session_id).messages
    messages: List[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        *current_history_for_main_llm,
        HumanMessage(content=user_input + context_block)
    ]

    MAX_TURNS = 5
    for turn in range(MAX_TURNS):
        logger.info(f"[{session_id}] Вызов основного LLM (Turn {turn + 1}/{MAX_TURNS}). Сообщений: {len(messages)}")
        try:
            ai_response: AIMessage = llm_with_tools.invoke(messages, config=config)
        except Exception as llm_error:
            logger.error(f"[{session_id}] Ошибка вызова основного LLM: {llm_error}", exc_info=True)
            return f"Произошла ошибка при обращении к языковой модели: {llm_error}"

        messages.append(ai_response)

        if not ai_response.tool_calls:
            logger.info(f"[{session_id}] LLM вернул финальный ответ.")
            return ai_response.content

        logger.info(f"[{session_id}] LLM запросил вызов инструментов: {len(ai_response.tool_calls)}")
        tool_messages: List[ToolMessage] = []
        for tool_call in ai_response.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")
            logger.info(f"[{session_id}] Вызов инструмента '{tool_name}' с аргументами: {tool_args}")

            selected_tool = next((t for t in tools if t.name == tool_name), None)

            if not selected_tool:
                error_msg = f"Ошибка: LLM запросил неизвестный инструмент '{tool_name}'."
                logger.error(f"[{session_id}] {error_msg}")
                tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))
                continue

            try:
                tool_output = selected_tool.invoke(tool_args, config=config)
                tool_output_str = str(tool_output)
                max_tool_output_len = 2000
                if len(tool_output_str) > max_tool_output_len:
                    tool_output_truncated = tool_output_str[:max_tool_output_len] + "... (результат обрезан)"
                    logger.warning(f"[{session_id}] Результат инструмента '{tool_name}' обрезан.")
                else:
                    tool_output_truncated = tool_output_str
                logger.info(f"[{session_id}] Результат '{tool_name}': {tool_output_truncated[:200]}...")
                tool_messages.append(ToolMessage(content=tool_output_truncated, tool_call_id=tool_id))
            except Exception as e:
                error_msg = f"Ошибка выполнения инструмента '{tool_name}': {type(e).__name__}: {e}"
                logger.error(f"[{session_id}] {error_msg}", exc_info=True)
                tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))

        messages.extend(tool_messages)

    logger.warning(f"[{session_id}] Достигнут лимит ({MAX_TURNS}) вызовов инструментов.")
    return "Кажется, обработка вашего запроса заняла слишком много шагов. Попробуйте переформулировать."

agent_chain = RunnableLambda(run_agent_like_chain, name="AgentLikeChainWithRagPreprocessing")
chain_with_history = RunnableWithMessageHistory(
    agent_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
logger.info("Цепочка с историей и RAG препроцессингом настроена.")

# --- API Helper Functions ---
def initialize_assistant(
    gigachat_credentials: Optional[str] = None,
    giga_embedding_model: Optional[str] = None,
    giga_scope: Optional[str] = None,
    deepseek_api_key: Optional[str] = None,
    deepseek_chat_model: Optional[str] = None,
    json_data_path: Optional[str] = None,
    chroma_persist_dir: Optional[str] = None
) -> RunnableWithMessageHistory:
    global GIGACHAT_CREDENTIALS, JSON_DATA_PATH, CHROMA_PERSIST_DIR
    global GIGA_EMBEDDING_MODEL, GIGA_SCOPE
    global DEEPSEEK_API_KEY, DEEPSEEK_CHAT_MODEL

    if gigachat_credentials: GIGACHAT_CREDENTIALS = gigachat_credentials
    if giga_embedding_model: GIGA_EMBEDDING_MODEL = giga_embedding_model
    if giga_scope: GIGA_SCOPE = giga_scope
    if deepseek_api_key: DEEPSEEK_API_KEY = deepseek_api_key
    if deepseek_chat_model: DEEPSEEK_CHAT_MODEL = deepseek_chat_model
    if json_data_path: JSON_DATA_PATH = json_data_path
    if chroma_persist_dir: CHROMA_PERSIST_DIR = chroma_persist_dir

    logger.info("Параметры конфигурации обновлены (если переданы). Возвращаем существующую цепочку.")
    logger.warning("Примечание: Динамическое обновление моделей/RAG во время работы API не реализовано.")
    return chain_with_history

def clear_session_history(session_id: str) -> bool:
    if session_id in chat_memory:
        chat_memory[session_id] = ChatMessageHistory()
        logger.info(f"История сессии {session_id} очищена")
        return True
    logger.warning(f"Попытка очистить несуществующую сессию: {session_id}")
    return False

def get_active_session_count() -> int:
    count = len(chat_memory)
    logger.debug(f"Запрошено количество активных сессий: {count}")
    return count
