# rag_setup.py

import os
import json
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple

# --- LangChain Imports ---
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langchain_core.retrievers import BaseRetriever 
from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever 
from langchain.retrievers import EnsembleRetriever 

logger = logging.getLogger(__name__)


def preprocess_for_rag_v2(data: List[Dict[str, Any]]) -> List[Document]:
    # ... (код функции preprocess_for_rag_v
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
            new_desc = (item.get("serviceDescription") or "").strip()
            if new_desc and new_desc.lower() not in ('', 'null', 'нет описания') and len(new_desc) > len(services_data[srv_id].get("description", "")):
                services_data[srv_id]["description"] = new_desc

        emp_id = item.get("employeeId")
        if emp_id:
            if emp_id not in employees_data:
                employees_data[emp_id] = {
                    "name": item.get("employeeFullName", "Имя неизвестно"),
                    "description": (item.get("employeeDescription") or "").strip()
                }
            new_desc = (item.get("employeeDescription") or "").strip()
            meaningful_desc = new_desc and new_desc.lower() not in ('', 'null', 'нет описания', 'опытный специалист')
            current_desc = employees_data[emp_id].get("description", "")
            if meaningful_desc and len(new_desc) > len(current_desc):
                employees_data[emp_id]["description"] = new_desc

    documents = []
    for srv_id, info in services_data.items():
        if not info.get("name") or not info.get("description") or info["description"].lower() in ('', 'null', 'нет описания'): continue
        text_content = f"Услуга: {info['name']}\nКатегория: {info['category']}\nОписание: {info['description']}"
        metadata = {"id": f"srv_{srv_id}", "type": "service", "name": info['name'], "category": info['category']}
        documents.append(Document(page_content=text_content, metadata=metadata))

    for emp_id, info in employees_data.items():
         if not info.get("name") or not info.get("description") or info["description"].lower() in ('', 'null', 'нет описания', 'опытный специалист'): continue
         text_content = f"Сотрудник: {info['name']}\nОписание: {info['description']}"
         metadata = {"id": f"emp_{emp_id}", "type": "employee", "name": info['name']}
         documents.append(Document(page_content=text_content, metadata=metadata))

    logger.info(f"Подготовлено {len(documents)} документов для RAG (услуги: {len(services_data)}, сотрудники: {len(employees_data)}).")
    return documents


def add_instruction_to_query(query: str) -> str:
    """Добавляет инструкцию к запросу для поиска в RAG с EmbeddingsGigaR."""
    instruction = "Дан запрос, необходимо найти релевантный документ с информацией об описании услуги или описании специалиста: "
    if query:
        logger.debug(f"Добавляем инструкцию к RAG-запросу: '{query}'")
        return f"{instruction}{query}"
    else:
        logger.warning("Попытка добавить инструкцию к пустому RAG-запросу.")
        return ""


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
        
        score = doc.metadata.get('relevance_score', None)
        score_str = f" (Score: {score:.4f})" if score is not None else ""
        formatted_docs.append(f"Источник {i+1}{score_str} ({display_source}):\n{doc.page_content}")
    return "\n\n".join(formatted_docs)


# --- Основная функция инициализации RAG ---
def initialize_rag(
    json_data_path: str,
    chroma_persist_dir: str,
    embedding_credentials: str,
    embedding_model: str,
    embedding_scope: str,
    verify_ssl_certs: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 100,
    search_k: int = 5 # Увеличил k по умолчанию для EnsembleRetriever
) -> Tuple[Optional[BaseRetriever], Optional[GigaChatEmbeddings], Optional[List[Document]], Optional[List[Dict]]]:
    """
    Инициализирует RAG-систему с гибридным поиском (Chroma + BM25).

    Возвращает:
        Tuple: (ensemble_retriever, embeddings_object, prepared_documents, clinic_data)
               ensemble_retriever: Настроенный гибридный ретривер LangChain.
               embeddings_object: Созданный объект GigaChatEmbeddings.
               prepared_documents: Список документов, подготовленных для RAG.
               clinic_data: Загруженные исходные данные клиники.
               Возвращает None для компонентов, если произошла критическая ошибка.
    """
    embeddings_object = None
    chroma_retriever = None
    bm25_retriever = None
    ensemble_retriever = None
    prepared_documents = []
    clinic_data = None

    # --- 1. Инициализация Эмбеддингов ---
    try:
        embeddings_object = GigaChatEmbeddings(
            credentials=embedding_credentials,
            model=embedding_model,
            verify_ssl_certs=verify_ssl_certs,
            scope=embedding_scope,
            timeout=60
        )
        logger.info(f"Эмбеддинги GigaChat ({embedding_model}) инициализированы.")
    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации эмбеддингов: {e}", exc_info=True)
        return None, None, None, None

    # --- 2. Загрузка Данных ---
    try:
        # ... (код загрузки clinic_data без изменений) ...
        base_dir = os.path.dirname(json_data_path)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"Создана директория '{base_dir}'.")
        if not os.path.exists(json_data_path):
            with open(json_data_path, 'w', encoding='utf-8') as f: json.dump([], f)
            logger.warning(f"Файл '{json_data_path}' не найден, создан пустой файл.")

        with open(json_data_path, 'r', encoding='utf-8') as f:
            clinic_data = json.load(f)
        if not isinstance(clinic_data, list):
             raise ValueError("JSON должен быть списком объектов")
        if not clinic_data:
             logger.warning(f"Файл данных '{json_data_path}' пуст.")
        else:
             logger.info(f"Загружено {len(clinic_data)} записей из {json_data_path}.")
    except Exception as e:
        logger.error(f"Ошибка загрузки данных клиники из {json_data_path}: {e}", exc_info=True)
        return None, embeddings_object, None, None

    if clinic_data:
        try:
            prepared_documents = preprocess_for_rag_v2(clinic_data)
        except Exception as e:
             logger.error(f"Ошибка предобработки данных для RAG: {e}", exc_info=True)
             prepared_documents = []
    else:
        logger.warning("Нет данных клиники для подготовки RAG документов.")
        prepared_documents = []

    
    vectorstore = None
    if prepared_documents:
        if os.path.exists(chroma_persist_dir):
            try:
                logger.info(f"Загрузка существующей векторной базы Chroma из '{chroma_persist_dir}'...")
                vectorstore = Chroma(
                    persist_directory=chroma_persist_dir,
                    embedding_function=embeddings_object
                )
                logger.info("Векторная база Chroma успешно загружена.")
            except Exception as e:
                logger.warning(f"Не удалось загрузить базу Chroma: {e}. База будет создана заново.", exc_info=True)
                try: shutil.rmtree(chroma_persist_dir)
                except OSError as rm_err: logger.error(f"Не удалось удалить '{chroma_persist_dir}': {rm_err}")
                vectorstore = None

        if vectorstore is None:
            logger.info(f"Создание новой векторной базы Chroma в '{chroma_persist_dir}'...")
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
                )
                docs_for_chroma = text_splitter.split_documents(prepared_documents)
                logger.info(f"Документы разделены на {len(docs_for_chroma)} чанков для Chroma.")
                vectorstore = Chroma.from_documents(
                    documents=docs_for_chroma,
                    embedding=embeddings_object,
                    persist_directory=chroma_persist_dir
                )
                logger.info("Новая векторная база Chroma создана и сохранена.")
            except Exception as e:
                logger.error(f"Ошибка создания векторной базы Chroma: {e}", exc_info=True)
                vectorstore = None
    else:
         logger.warning("Нет подготовленных документов, Chroma база не будет создана/загружена.")

    if vectorstore:
        try:
            chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
            logger.info(f"Chroma ретривер настроен (k={search_k}).")
        except Exception as e:
             logger.error(f"Ошибка создания Chroma ретривера: {e}", exc_info=True)
             chroma_retriever = None
    else:
         chroma_retriever = None 

    if prepared_documents:
        try:
            bm25_retriever = BM25Retriever.from_documents(
                prepared_documents,
                k=search_k 
                )
            logger.info(f"BM25 ретривер настроен (k={search_k}).")
        except Exception as e:
            logger.error(f"Ошибка создания BM25 ретривера: {e}", exc_info=True)
            bm25_retriever = None
    else:
        logger.warning("Нет подготовленных документов для создания BM25 ретривера.")
        bm25_retriever = None

    retrievers_list = []
    if chroma_retriever:
        retrievers_list.append(chroma_retriever)
    if bm25_retriever:
        retrievers_list.append(bm25_retriever)

    if len(retrievers_list) >= 1: 
        if len(retrievers_list) == 2:
             try:
                  ensemble_retriever = EnsembleRetriever(
                      retrievers=retrievers_list,
                      weights=[0.4, 0.6]
                  )
                  logger.info("EnsembleRetriever (гибридный поиск Chroma + BM25) настроен.")
             except Exception as e:
                  logger.error(f"Ошибка создания EnsembleRetriever: {e}. Используем только доступные.", exc_info=True)
                  ensemble_retriever = retrievers_list[0]
                  logger.warning(f"Используется только {type(ensemble_retriever).__name__}")
        else:
             ensemble_retriever = retrievers_list[0]
             logger.warning(f"Доступен только один ретривер: {type(ensemble_retriever).__name__}. Гибридный поиск не используется.")
    else:

         ensemble_retriever = None
         logger.error("Не удалось создать ни Chroma, ни BM25 ретривер.")


  
    if ensemble_retriever is None:
        def empty_retriever_func(query: str, *, config: Optional[RunnableConfig] = None, **kwargs) -> List[Document]:
            logger.warning("Ретривер-пустышка вызван, так как ни один реальный ретривер не был создан.")
            return []
        ensemble_retriever = RunnableLambda(empty_retriever_func, name="EmptyRetriever").with_types(output_type=List[Document])
        logger.warning("Создан ретривер-пустышка.")
    return ensemble_retriever, embeddings_object, prepared_documents, clinic_data