# router.py
import logging
from typing import List, Dict, Any, Callable
from operator import itemgetter

# --- LangChain Imports ---
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch, RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory # Используется для type hinting

# --- Роутер: Промпт для определения типа запроса ---
ROUTER_PROMPT_TEMPLATE = """Ты - ИИ-диспетчер, который анализирует ПОСЛЕДНИЙ запрос пользователя (`input`) и историю диалога (`chat_history`), чтобы определить НАИЛУЧШИЙ СПОСОБ ответа.

Доступные варианты:
1.  `RAG`: Если пользователю нужно ОБЩЕЕ ОПИСАНИЕ услуги, врача, или ответ на ОБЩИЙ вопрос о клинике, который может быть в базе знаний. Пример: "Расскажи про лазерную эпиляцию", "Что такое УЗИ?". НЕ используй для запросов КОНКРЕТНЫХ цен, наличия, списков.
2.  `TOOL`: Если пользователю нужны КОНКРЕТНЫЕ ДАННЫЕ: цена услуги, список врачей/филиалов, проверка доступности услуги в филиале, сравнение цен, поиск мест оказания услуги. Пример: "Сколько стоит массаж спины?", "Какие дерматологи есть в Москва-Сити?", "Где делают УЗИ?". Также используй, если пользователь явно просит выполнить действие (найти врача, узнать цену).
3.  `MEMORY`: Если ответ СКОРЕЕ ВСЕГО содержится в ПРЕДЫДУЩИХ сообщениях (истории чата). Пример: короткие вопросы "где?", "цена?", "а он?", вопросы про имя пользователя ("как меня зовут?"). Также если пользователь просто продолжает предыдущую мысль.
4.  `DIRECT`: Для приветствий, прощаний, простых подтверждений ("да", "нет"), благодарностей, общих разговоров не по теме клиники, или если ни один другой вариант не подходит. Пример: "Привет", "Спасибо", "Пока", "Хорошо".

Внимательно изучи ПОСЛЕДНЕЕ СООБЩЕНИЕ пользователя (`input`) и ИСТОРИЮ ДИАЛОГА (`chat_history`).

Примеры:
- input: "Расскажи про лазерную эпиляцию Soprano" -> RAG
- input: "Сколько стоит Soprano Пальцы для женщин?" -> TOOL
- input: "А где ее делают?" (после обсуждения услуги 'Soprano Пальцы') -> TOOL (Нужен инструмент find_service_locations)
- input: "Кто делает эту процедуру?" (после обсуждения услуги) -> TOOL (find_employees)
- input: "Привет" -> DIRECT
- input: "Спасибо!" -> DIRECT
- input: "Какие врачи у вас есть?" -> TOOL (find_employees без фильтров)
- input: "В каких филиалах есть УЗИ?" -> TOOL (find_service_locations)
- input: "Меня зовут Алексей" -> DIRECT
- input: "Как меня зовут?" (после "Меня зовут Алексей") -> MEMORY

Твой ответ должен быть ТОЛЬКО ОДНИМ СЛОВОМ из списка: RAG, TOOL, MEMORY, DIRECT. Без кавычек, без объяснений.

<chat_history>
{chat_history}
</chat_history>

<input>
{input}
</input>

Классификация:"""


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
    instruction = "Найди информацию об услугах, ценах, врачах или филиалах клиники по следующему запросу: \nвопрос: "
    return f"{instruction}{query}"

def create_router_chain(
    chat_model: Any, # Базовая модель без инструментов (для роутинга и простых ответов)
    llm_with_tools: Any, # Модель, к которой привязаны инструменты
    retriever: Any, # Ретривер для RAG
    system_prompt: str, # Основной системный промпт для финальных ответов
    get_session_history_func: Callable[[str], ChatMessageHistory] # Функция для получения истории
) -> RunnableLambda:
    """
    Создает и возвращает полную цепочку ассистента с логикой роутинга.

    Args:
        chat_model: Экземпляр базовой языковой модели (например, GigaChat).
        llm_with_tools: Экземпляр языковой модели с привязанными инструментами.
        retriever: Экземпляр ретривера LangChain.
        system_prompt: Текст системного промпта для основной модели.
        get_session_history_func: Функция, принимающая session_id и возвращающая ChatMessageHistory.

    Returns:
        Готовая к использованию цепочка LangChain (Runnable).
    """
    logging.info("Создание цепочки роутера...")

    # --- Цепочка для классификации запроса ---
    router_prompt = ChatPromptTemplate.from_messages([
        ("system", ROUTER_PROMPT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    # Используем базовую модель для роутинга
    routing_chain = router_prompt | chat_model | StrOutputParser()
    logging.info("Цепочка классификации роутера создана.")

    # --- Цепочка для получения RAG-контекста ---
    context_chain = (
        {"input": RunnablePassthrough()} # Принимает вход пользователя
        | RunnablePassthrough.assign(
            query_for_retriever=itemgetter("input") | RunnableLambda(add_instruction_to_query, name="AddInstruction")
        ).assign(
            docs=itemgetter("query_for_retriever") | retriever
        ).assign(
            context=itemgetter("docs") | RunnableLambda(format_docs, name="FormatDocs")
        )
        | itemgetter("context") # Возвращаем только отформатированный контекст
    ).with_config(run_name="FetchRAGContext")
    logging.info("Цепочка получения RAG-контекста создана.")

    # --- Определение Под-цепочек для RunnableBranch ---

    # 1. Цепочка для RAG
    rag_chain_logic = RunnablePassthrough.assign(
        rag_context=lambda x: context_chain.invoke(x["input"], config=x.get("config")) # Получаем RAG контекст
    ).assign(
        final_response=lambda x: llm_with_tools.invoke( # Используем LLM с инструментами
            [
                SystemMessage(content=system_prompt),
                *x["chat_history"],
                HumanMessage(content=f"{x['input']}\n\n[Информация из базы знаний]:\n{x['rag_context']}\n[/Информация из базы знаний]")
            ],
            config=x.get("config")
        )
    ) | itemgetter("final_response") | StrOutputParser()
    logging.info("Логика для ветки RAG определена.")

    # 2. Цепочка для TOOL
    tool_chain_logic = RunnableLambda(
        lambda x: llm_with_tools.invoke( # Используем LLM с инструментами
            [
                SystemMessage(content=system_prompt),
                *x["chat_history"],
                HumanMessage(content=x["input"]) # Без RAG контекста
            ],
            config=x.get("config")
        ),
        name="ToolChainLLMInvoke"
    ) | StrOutputParser()
    logging.info("Логика для ветки TOOL определена.")

    # 3. Цепочка для MEMORY/DIRECT
    memory_direct_chain_logic = RunnableLambda(
        lambda x: chat_model.invoke( # Используем базовую LLM без инструментов
            [
                SystemMessage(content=system_prompt), # Можно использовать укороченный промпт для скорости
                *x["chat_history"],
                HumanMessage(content=x["input"])
            ],
            config=x.get("config")
        ),
        name="MemoryDirectChainLLMInvoke"
    ) | StrOutputParser()
    logging.info("Логика для веток MEMORY/DIRECT определена.")

    # --- Сборка основной цепочки с роутингом ---
    def prepare_branch_input(input_dict: Dict, config: RunnableConfig) -> Dict:
        """Подготавливает входные данные для роутера и последующих веток."""
        session_id = config["configurable"]["session_id"]
        chat_history = get_session_history_func(session_id).messages
        # Этот словарь будет передан в routing_chain и затем в выбранную ветку Branch
        return {"input": input_dict["input"], "chat_history": chat_history, "config": config}

    # Цепочка, которая сначала подготавливает данные, затем запускает роутер
    structured_routing_chain = RunnableLambda(
        prepare_branch_input, name="PrepareBranchInput"
    ).assign(
        classification=routing_chain # Вызываем роутер на подготовленных данных
    ).with_config(run_name="RoutingDecision")
    logging.info("Цепочка подготовки данных и классификации создана.")

    # Основная цепочка с использованием RunnableBranch
    full_routed_chain = structured_routing_chain | RunnableBranch(
        (lambda x: x["classification"] == "RAG", RunnableLambda(lambda x: rag_chain_logic.invoke(x), name="RAG_Branch")),
        (lambda x: x["classification"] == "TOOL", RunnableLambda(lambda x: tool_chain_logic.invoke(x), name="TOOL_Branch")),
        (lambda x: x["classification"] == "MEMORY", RunnableLambda(lambda x: memory_direct_chain_logic.invoke(x), name="MEMORY_Branch")),
        # По умолчанию (включая 'DIRECT' и возможные ошибки классификации) используем memory_direct_chain_logic
        RunnableLambda(lambda x: memory_direct_chain_logic.invoke(x), name="DIRECT_Fallback_Branch")
    )
    logging.info("Полная цепочка роутинга с ветвлением создана.")

    # Возвращаем финальную цепочку (без обертки в RunnableWithMessageHistory, это делается в matrixai.py)
    return full_routed_chain 