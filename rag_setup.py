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


# --- Функция создания документов с общей информацией о клинике ---
def create_clinic_info_docs() -> List[Document]:
    """Создает список документов LangChain с общей информацией о клинике."""
    docs = []
    try:
        # Блок 1: Ключевые метрики
        docs.append(Document(
            page_content=(
                "О клинике Med YU Med: Ключевые показатели.\n"
                "Высочайший уровень сервиса позволяет каждому нашему пациенту чувствовать себя комфортно во время визита.\n"
                "- Филиалов по всему миру: 3\n"
                "- Специалистов в команде: 50+\n"
                "- Пациентов обращаются повторно: 87%\n"
                "- Средний стаж сотрудника: 6,3 лет"
            ),
            metadata={"type": "clinic_info", "topic": "key_metrics", "id": "clinic_info_metrics"}
        ))

        # Блок 2: Подход клиники
        docs.append(Document(
            page_content=(
                "Подход клиники Med YU Med: Заслуженное доверие пациентов.\n"
                "1. Индивидуальный подход: Учитываем все пожелания пациента. Перед процедурой врач проведёт подробную консультацию, подберёт процедуру, идеально подходящую под вашу потребность. Результат будет соответствовать вашим ожиданиям.\n"
                "2. Качественное оборудование: В наших клиниках представлены эффективные и проверенные технологии. Оборудование и препараты, которые мы используем отвечают высоким мировым стандартам.\n"
                "3. Cпециалисты высокого уровня: Все действующие сотрудники постоянно повышают свою квалификацию у лучших экспертов в сфере косметологии."
            ),
            metadata={"type": "clinic_info", "topic": "company_approach", "id": "clinic_info_approach"}
        ))

        # Блок 3: Направления деятельности
        docs.append(Document(
            page_content=(
                "Направления деятельности Med YU Med:\n"
                "- Услуги косметологии\n"
                "- Интернет-магазин"
            ),
            metadata={"type": "clinic_info", "topic": "business_directions", "id": "clinic_info_directions"}
        ))

        # Блок 4: Филиалы и адреса
        docs.append(Document(
            page_content=(
                "Филиалы и адреса клиники Med YU Med:\n"
                "1. Москва, Пресненская наб., 8, стр. 1, 2 этаж (Башня Город Столиц)\n"
                "2. Москва, улица Авиаконструктора Сухого, 2, корп. 1 (м. ЦСКА, ЖК Лица)\n"
                "3. Bluewaters Island, Dubai - UAE"
            ),
            metadata={"type": "clinic_info", "topic": "locations", "id": "clinic_info_locations"}
        ))

        # Блок 5: Контакты и время работы
        docs.append(Document(
            page_content=(
                "Контакты и время работы Med YU Med:\n"
                "- Телефон для связи: 8 800 550-08-96\n"
                "- Время работы: пн-вс: 10.00—22.00"
            ),
            metadata={"type": "clinic_info", "topic": "contacts", "id": "clinic_info_contacts"}
        ))

        # Блок 6: Юридическая информация
        docs.append(Document(
            page_content=(
                "Юридическая информация Med YU Med:\n"
                "- Наименование: ООО «МЕД-Ю-МЕД»\n"
                "- Медицинская лицензия: Л041-01137-77/01322474 от 29.07.2024\n"
                "- © Косметологическая клиника MED YU MED. Все права защищены.\n"
                "- Доступны Политика конфиденциальности и Публичная информация."
            ),
            metadata={"type": "clinic_info", "topic": "legal", "id": "clinic_info_legal"}
        ))

        logger.info(f"Создано {len(docs)} документов с общей информацией о клинике.")
    except Exception as e:
        logger.error(f"Ошибка при создании документов clinic_info: {e}", exc_info=True)
        # Возвращаем пустой список в случае ошибки, чтобы не прерывать остальную инициализацию
        return []
    return docs


# --- Функция предобработки данных из JSON ---
def preprocess_for_rag_v2(data: List[Dict[str, Any]]) -> List[Document]:
    """Готовит документы по услугам и сотрудникам из JSON для RAG."""
    services_data = {}
    employees_data = {}
    if not isinstance(data, list):
        logger.error(f"Ожидался список для preprocess_for_rag_v2, получен {type(data)}.")
        return []

    for item in data:
        if not isinstance(item, dict):
            logger.warning(f"Пропуск элемента не-словаря в данных: {item}")
            continue

        # Обработка услуг
        srv_id = item.get("serviceId")
        if srv_id:
            if srv_id not in services_data:
                services_data[srv_id] = {
                    "name": item.get("serviceName", "Без названия"),
                    "category": item.get("categoryName", "Без категории"),
                    "description": (item.get("serviceDescription") or "").strip()
                }
            new_desc = (item.get("serviceDescription") or "").strip()
            current_desc = services_data[srv_id].get("description", "")
            if new_desc and new_desc.lower() not in ('', 'null', 'нет описания') and len(new_desc) > len(current_desc):
                services_data[srv_id]["description"] = new_desc

        # Обработка сотрудников
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
    # Документы по услугам
    for srv_id, info in services_data.items():
        # Добавляем только если есть осмысленное имя и описание
        if not info.get("name") or info.get("name") == "Без названия" or \
           not info.get("description") or info["description"].lower() in ('', 'null', 'нет описания'):
            continue
        text_content = f"Услуга: {info['name']}\nКатегория: {info['category']}\nОписание: {info['description']}"
        metadata = {"id": f"srv_{srv_id}", "type": "service", "name": info['name'], "category": info['category']}
        documents.append(Document(page_content=text_content, metadata=metadata))

    # Документы по сотрудникам
    for emp_id, info in employees_data.items():
         # Добавляем только если есть осмысленное имя и описание
         if not info.get("name") or info.get("name") == "Имя неизвестно" or \
            not info.get("description") or info["description"].lower() in ('', 'null', 'нет описания', 'опытный специалист'):
             continue
         text_content = f"Сотрудник: {info['name']}\nОписание: {info['description']}"
         metadata = {"id": f"emp_{emp_id}", "type": "employee", "name": info['name']}
         documents.append(Document(page_content=text_content, metadata=metadata))

    logger.info(f"Подготовлено {len(documents)} док-ов из JSON (услуги: {len(services_data)}, сотрудники: {len(employees_data)}).")
    return documents



def add_instruction_to_query(query: str) -> str:
    """Добавляет инструкцию к запросу для поиска в RAG с EmbeddingsGigaR."""
    # Пример инструкции, можно адаптировать
    instruction = "Найди наиболее релевантный документ с описанием услуги, врача или общей информацией о клинике по следующему запросу: "
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
        metadata = doc.metadata or {}
        source_info = metadata.get('id', 'Неизвестный источник')
        doc_type = metadata.get('type', 'данные')
        doc_name = metadata.get('name', '')
        doc_topic = metadata.get('topic', '') # Добавлено для clinic_info

        display_source = doc_type
        if doc_topic: display_source += f" ({doc_topic})"
        if doc_name: display_source += f" - '{doc_name}'"
        if not doc_topic and not doc_name: display_source = source_info # Фоллбэк

        score = metadata.get('relevance_score', None) # EnsembleRetriever добавляет это поле
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
    chunk_overlap: int = 200,
    search_k: int = 5,
    # Флаг для принудительного пересоздания базы Chroma
    force_recreate_chroma: bool = False
) -> Tuple[Optional[BaseRetriever], Optional[GigaChatEmbeddings], Optional[List[Document]], Optional[List[Dict]]]:
    """
    Инициализирует RAG-систему с гибридным поиском (Chroma + BM25),
    включая общую информацию о клинике.

    Args:
        json_data_path: Путь к JSON файлу с данными услуг/сотрудников.
        chroma_persist_dir: Директория для сохранения/загрузки Chroma базы.
        embedding_credentials: Учетные данные GigaChat для эмбеддингов.
        embedding_model: Название модели эмбеддингов GigaChat.
        embedding_scope: Scope для GigaChat API.
        verify_ssl_certs: Проверять ли SSL сертификаты.
        chunk_size: Размер чанка для сплиттера.
        chunk_overlap: Перекрытие чанков.
        search_k: Количество документов для возврата ретриверами.
        force_recreate_chroma: Если True, удалит существующую базу Chroma и создаст заново.

    Returns:
        Tuple: (ensemble_retriever, embeddings_object, all_prepared_documents, clinic_data)
               Или (None, None, None, None) в случае критической ошибки эмбеддингов.
    """
    embeddings_object = None
    ensemble_retriever = None
    all_prepared_documents = [] 
    clinic_data = None

    try:
        embeddings_object = GigaChatEmbeddings(
            credentials=embedding_credentials, model=embedding_model,
            verify_ssl_certs=verify_ssl_certs, scope=embedding_scope, timeout=60
        )
        logger.info(f"Эмбеддинги GigaChat ({embedding_model}) инициализированы.")
    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации эмбеддингов: {e}", exc_info=True)
        return None, None, None, None

    try:
        base_dir = os.path.dirname(json_data_path)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            logger.info(f"Создана директория '{base_dir}'.")
        if not os.path.exists(json_data_path):
            with open(json_data_path, 'w', encoding='utf-8') as f: json.dump([], f)
            logger.warning(f"Файл '{json_data_path}' не найден, создан пустой файл.")

        with open(json_data_path, 'r', encoding='utf-8') as f:
            clinic_data = json.load(f)
        if not isinstance(clinic_data, list): raise ValueError("JSON должен быть списком объектов")
        if not clinic_data: logger.warning(f"Файл данных '{json_data_path}' пуст.")
        else: logger.info(f"Загружено {len(clinic_data)} записей из {json_data_path}.")
    except Exception as e:
        logger.error(f"Ошибка загрузки данных клиники из {json_data_path}: {e}", exc_info=True)
        clinic_data = []

  
    if clinic_data:
        try: all_prepared_documents.extend(preprocess_for_rag_v2(clinic_data))
        except Exception as e: logger.error(f"Ошибка предобработки данных из JSON: {e}", exc_info=True)
    try: all_prepared_documents.extend(create_clinic_info_docs())
    except Exception as e: logger.error(f"Ошибка создания общих док-ов о клинике: {e}", exc_info=True)

    if not all_prepared_documents:
        logger.error("Нет документов для создания RAG системы. Проверьте JSON файл и функцию create_clinic_info_docs.")
        ensemble_retriever = RunnableLambda(lambda query, **kwargs: [], name="EmptyRetriever").with_types(output_type=List[Document])
        logger.warning("Создан ретривер-пустышка из-за отсутствия документов.")
        return ensemble_retriever, embeddings_object, all_prepared_documents, clinic_data

    logger.info(f"Всего подготовлено {len(all_prepared_documents)} документов для RAG.")

    vectorstore = None
    chroma_retriever = None
    if force_recreate_chroma and os.path.exists(chroma_persist_dir):
        logger.warning(f"Принудительное удаление существующей базы Chroma в '{chroma_persist_dir}'...")
        try: shutil.rmtree(chroma_persist_dir)
        except OSError as rm_err: logger.error(f"Не удалось удалить '{chroma_persist_dir}': {rm_err}")

    if os.path.exists(chroma_persist_dir):
        try:
            logger.info(f"Загрузка Chroma из '{chroma_persist_dir}'...")
            vectorstore = Chroma(persist_directory=chroma_persist_dir, embedding_function=embeddings_object)
            logger.info("База Chroma загружена.")
        except Exception as e:
            logger.warning(f"Не удалось загрузить Chroma: {e}. База будет создана заново.", exc_info=True)
            try: shutil.rmtree(chroma_persist_dir)
            except OSError as rm_err: logger.error(f"Не удалось удалить '{chroma_persist_dir}': {rm_err}")
            vectorstore = None 

    if vectorstore is None:
        logger.info(f"Создание новой базы Chroma в '{chroma_persist_dir}'...")
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
            )
            docs_for_chroma = text_splitter.split_documents(all_prepared_documents)
            logger.info(f"Документы разделены на {len(docs_for_chroma)} чанков для Chroma.")
            vectorstore = Chroma.from_documents(
                documents=docs_for_chroma,
                embedding=embeddings_object,
                persist_directory=chroma_persist_dir
            )
            logger.info("Новая база Chroma создана и сохранена.")
        except Exception as e:
            logger.error(f"Ошибка создания базы Chroma: {e}", exc_info=True)
            vectorstore = None

    if vectorstore:
        try:
            chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": search_k})
            logger.info(f"Chroma ретривер настроен (k={search_k}).")
        except Exception as e:
             logger.error(f"Ошибка создания Chroma ретривера: {e}", exc_info=True)

    bm25_retriever = None
    try:
        bm25_retriever = BM25Retriever.from_documents(
            all_prepared_documents,
            k=search_k
            )
        logger.info(f"BM25 ретривер настроен (k={search_k}).")
    except Exception as e:
        logger.error(f"Ошибка создания BM25 ретривера: {e}", exc_info=True)

    retrievers_list = []
    if chroma_retriever: retrievers_list.append(chroma_retriever)
    if bm25_retriever: retrievers_list.append(bm25_retriever)

    if not retrievers_list:
         logger.error("Не удалось создать ни Chroma, ни BM25 ретривер.")
         ensemble_retriever = RunnableLambda(lambda query, **kwargs: [], name="EmptyRetriever").with_types(output_type=List[Document])
         logger.warning("Создан ретривер-пустышка.")
    elif len(retrievers_list) == 2:
         try:
              ensemble_retriever = EnsembleRetriever(
                  retrievers=retrievers_list,
                  weights=[0.4, 0.6] 
              )
              logger.info("EnsembleRetriever (гибридный поиск Chroma + BM25) настроен.")
         except Exception as e:
              logger.error(f"Ошибка создания EnsembleRetriever: {e}. Используем только первый.", exc_info=True)
              ensemble_retriever = retrievers_list[0]
              logger.warning(f"Используется только {type(ensemble_retriever).__name__}")
    else: 
         ensemble_retriever = retrievers_list[0]
         logger.warning(f"Доступен только один ретривер: {type(ensemble_retriever).__name__}. Гибридный поиск не используется.")

    return ensemble_retriever, embeddings_object, all_prepared_documents, clinic_data