# rag_setup.py

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
import chromadb


from tenant_chain_data import discover_all_json_files, download_chain_data, build_file_path

try:
    import tenant_config_manager
except ImportError:
    logging.error("[RAG Setup] Не удалось импортировать tenant_config_manager. Информация о клинике из настроек не будет загружена.")
    tenant_config_manager = None

import tenant_chain_data 

from langchain_core.documents import Document

from langchain_gigachat.embeddings import GigaChatEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

TENANT_COLLECTION_PREFIX = "tenant_"

import re as _re

def _sanitize_fragment(fragment: str) -> str:
    """Sanitize a part of collection name (tenant or chain)."""
    return _re.sub(r"[^A-Za-z0-9_-]", "_", fragment) or "default"

def build_collection_name(tenant_chain: str) -> str:
    """Build a Chroma collection name from '<tenant>_<chain>' pattern."""
    if "_" in tenant_chain:
        tenant_part, chain_part = tenant_chain.split("_", 1)
    else:
        tenant_part, chain_part = tenant_chain, "default"
    name = f"{TENANT_COLLECTION_PREFIX}{_sanitize_fragment(tenant_part)}_{_sanitize_fragment(chain_part)}"
  
    name = name[:63].strip("-_")
    return name or "tenant_default"
SERVICE_DETAILS_FILE = "base/service_details.json" 


def load_service_details(file_path: str = SERVICE_DETAILS_FILE) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Загружает детали услуг (показания, противопоказания) из JSON-файла.
    Ключом в словаре будет кортеж (нормализованное_имя_услуги, нормализованное_имя_категории).
    """
    service_details_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Файл с деталями услуг '{file_path}' не найден. Показания/противопоказания не будут загружены.")
            return service_details_map

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            logger.error(f"Файл '{file_path}' должен содержать список объектов услуг. Получен: {type(data)}")
            return service_details_map

        for service_info in data:
            if not isinstance(service_info, dict):
                logger.warning(f"Пропуск элемента не-словаря в файле деталей услуг: {service_info}")
                continue

           
            service_name = (
                service_info.get("service_name")
                or service_info.get("serviceName")
                or service_info.get("service")
                or service_info.get("serviceTitle")
            )
            category_name = (
                service_info.get("category")
                or service_info.get("category_name")
                or service_info.get("categoryName")
            )
            indications = (
                service_info.get("indications")
                or service_info.get("indicationList")
                or service_info.get("indicationsList")
            )
            contraindications = (
                service_info.get("contraindications")
                or service_info.get("contraindicationList")
                or service_info.get("contraindicationsList")
            )

            if not service_name or not category_name:
                logger.warning(f"Пропуск услуги без имени или категории в файле деталей: {service_info}")
                continue

            
            norm_service_name = normalize_text(service_name, keep_spaces=True)
            norm_category_name = normalize_text(category_name, keep_spaces=True)

            key = (norm_service_name, norm_category_name)
            details = {}
            if isinstance(indications, list) and indications:
                details["indications"] = indications
            if isinstance(contraindications, list) and contraindications:
                details["contraindications"] = contraindications
            
            if details: 
                if key in service_details_map:
                    logger.warning(f"Дублирующаяся запись для услуги '{service_name}' в категории '{category_name}' в файле деталей. Используется первая.")
                else:
                    service_details_map[key] = details
        
        logger.info(f"Загружено {len(service_details_map)} записей с деталями услуг из '{file_path}'.")

    except Exception as e:
        logger.error(f"Ошибка загрузки или обработки файла деталей услуг '{file_path}': {e}", exc_info=True)
    return service_details_map





def preprocess_for_rag_v2(data: List[Dict[str, Any]], service_details_map: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Document]:
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
          
            services_data[srv_id]["original_service_name"] = item.get("serviceName", "Без названия")
            services_data[srv_id]["original_category_name"] = item.get("categoryName", "Без категории")
           

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
        
        if not info.get("name") or info.get("name") == "Без названия":
            continue
        desc_text = info.get("description") or "Описание отсутствует"
        text_content = f"Услуга: {info['name']}\nКатегория: {info['category']}\nОписание: {desc_text}"
        
    -
        norm_s_name_lookup = normalize_text(info.get("original_service_name"), keep_spaces=True)
        norm_cat_name_lookup = normalize_text(info.get("original_category_name"), keep_spaces=True)
        service_key = (norm_s_name_lookup, norm_cat_name_lookup)
        
        if service_details_map and service_key in service_details_map:
            details = service_details_map[service_key]
            indications_text = ""
            if details.get("indications"):
                indications_text = "\nПоказания: " + "; ".join(details["indications"])
            contraindications_text = ""
            if details.get("contraindications"):
                contraindications_text = "\nПротивопоказания: " + "; ".join(details["contraindications"])
            
            if indications_text or contraindications_text:
                text_content += indications_text + contraindications_text
                logger.debug(f"Добавлены показания/противопоказания для услуги '{info['name']}' категории '{info['category']}'.")
      

        metadata = {"id": f"srv_{srv_id}", "type": "service", "name": info['name'], "category": info['category'], "source": "base_json"}
        documents.append(Document(page_content=text_content, metadata=metadata))

    # Документы по сотрудникам
    for emp_id, info in employees_data.items():
         # Добавляем только если есть осмысленное имя и описание
         if not info.get("name") or info.get("name") == "Имя неизвестно" or \
            not info.get("description") or info["description"].lower() in ('', 'null', 'нет описания', 'опытный специалист'):
             continue
         text_content = f"Сотрудник: {info['name']}\nОписание: {info['description']}"
         metadata = {"id": f"emp_{emp_id}", "type": "employee", "name": info['name'], "source": "base_json"}
         documents.append(Document(page_content=text_content, metadata=metadata))

    logger.info(f"Подготовлено {len(documents)} док-ов из JSON (услуги: {len(services_data)}, сотрудники: {len(employees_data)}), помечены как 'base_json'.")
    return documents


# --- Функция загрузки документов одного тенанта из base ---
def load_tenant_base_data(tenant_id: str, data_path: str, service_details_map: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[List[Document], Optional[List[Dict[str, Any]]]]:
    """Загружает документы услуг/сотрудников одного тенанта из JSON файла в 'base'."""
    documents = []
    raw_data = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list): raise ValueError("JSON должен быть списком объектов")
        if not raw_data: logger.warning(f"Файл данных '{data_path}' пуст.")
        else: logger.info(f"Загружено {len(raw_data)} записей из {data_path} для тенанта {tenant_id}.")

        documents = preprocess_for_rag_v2(raw_data, service_details_map)
    except Exception as e:
        logger.error(f"Ошибка загрузки базовых данных тенанта из {data_path}: {e}", exc_info=True)
        raw_data = []

    return documents, raw_data


# --- Функция конвертации данных clinic_info в документы LangChain ---
def clinic_info_data_to_docs(clinic_info_data: List[Dict[str, Any]]) -> List[Document]:
    """Преобразует список словарей clinic_info в список объектов Document."""
    docs = []
    for doc_data in clinic_info_data:
        try:
            # Дополнительная проверка, хотя она есть в load_tenant_clinic_info
            if isinstance(doc_data, dict) and \
               isinstance(doc_data.get("page_content"), str) and \
               isinstance(doc_data.get("metadata"), dict):
                # Убедимся, что metadata не None перед передачей
                metadata = doc_data.get("metadata") or {}
                docs.append(Document(page_content=doc_data["page_content"], metadata=metadata))
            else:
                 logger.warning(f"Пропуск некорректного словаря при конвертации clinic_info: {doc_data}")
        except Exception as e:
            logger.error(f"Ошибка конвертации словаря clinic_info в Document: {e}. Данные: {doc_data}", exc_info=True)
    return docs



def index_documents_to_collection(
    chroma_client: chromadb.ClientAPI,
    embeddings_object: GigaChatEmbeddings,
    collection_name: str,
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
    force_recreate: bool = False
):
    """
    Индексирует документы в коллекцию Chroma.

    Args:
        chroma_client: Клиент ChromaDB.
        embeddings_object: Объект эмбеддингов GigaChat.
        collection_name: Название коллекции.
        documents: Список документов для индексации.
        chunk_size: Размер чанка для сплиттера.
        chunk_overlap: Перекрытие чанков.
        force_recreate: Если True, удалит существующую коллекцию и создаст заново.
    """
    logger.info(f"Проверка/Индексация коллекции '{collection_name}'...")

    try:
        
        existing_collection = chroma_client.get_collection(name=collection_name)
        logger.info(f"Коллекция '{collection_name}' уже существует.")

       
        if force_recreate:
            logger.warning(f"Флаг force_recreate установлен. Удаление существующей коллекции '{collection_name}'...")
            chroma_client.delete_collection(collection_name)
            logger.info(f"Коллекция '{collection_name}' удалена. Будет создана заново.")
            
        else:
            
            logger.info(f"Используем существующую коллекцию '{collection_name}'. Переиндексация пропускается.")
            return Chroma(client=chroma_client, collection_name=collection_name, embedding_function=embeddings_object)

    except Exception as e: 
        logger.info(f"Коллекция '{collection_name}' не найдена или произошла ошибка при проверке: {e}. Создаем новую.")
        
        pass 
   
    if not documents:
        logger.warning(f"Нет документов для создания новой коллекции '{collection_name}'.")
        return None 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len,
    )
    splits = text_splitter.split_documents(documents)
    logger.info(f"Документы разделены на {len(splits)} чанков для новой коллекции Chroma.")

    if not splits:
        logger.warning(f"Нет чанков для индексации в новую коллекцию '{collection_name}'.")
        return None

    
    logger.info(f"Создание новой коллекции '{collection_name}' и индексация документов...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings_object,
        collection_name=collection_name,
        client=chroma_client,
    )
    logger.info(f"Документы успешно проиндексированы в НОВУЮ коллекцию '{collection_name}'.")
    return vectorstore


def add_instruction_to_query(query: str) -> str:
    """Добавляет инструкцию к запросу для поиска в RAG с EmbeddingsGigaR."""
   
    instruction = "Найди наиболее релевантный документ с описанием услуги, врача или общей информацией о клинике по следующему запросу: "
    if query:
        logger.debug(f"Добавляем инструкцию к RAG-запросу: '{query}'")
        return f"{instruction}{query}"
    else:
        logger.warning("Попытка добавить инструкцию к пустому RAG-запросу.")
        return ""



def initialize_rag(
    data_dir: str,
    chroma_persist_dir: str,
    embedding_credentials: str,
    embedding_model: str,
    embedding_scope: str,
    verify_ssl_certs: bool = False,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    search_k: int = 5,
    force_recreate_chroma: bool = False
) -> Tuple[
    Optional[chromadb.ClientAPI],
    Optional[GigaChatEmbeddings],
    Dict[str, BM25Retriever], 
    Dict[str, List[Document]], 
    Dict[str, List[Dict]],      
    Dict[Tuple[str, str], Dict[str, Any]] 
]:
    """
    Инициализирует RAG-систему:
    - Создает Chroma коллекции и BM25 ретриверы для КАЖДОГО тенанта,
      объединяя данные из 'base/' и 'tenant_configs/'.
    - Больше не создает общих компонентов.

    Args:
        data_dir: Директория с base/*.json
        chroma_persist_dir: Директория для сохранения/загрузки Chroma базы.
        embedding_credentials: Учетные данные GigaChat для эмбеддингов.
        embedding_model: Название модели эмбеддингов GigaChat.
        embedding_scope: Scope для GigaChat API.
        verify_ssl_certs: Проверять ли SSL сертификаты.
        chunk_size: Размер чанка для сплиттера.
        chunk_overlap: Перекрытие чанков.
        search_k: Количество документов для возврата ретриверами (используется для BM25).
        force_recreate_chroma: Если True, удалит все существующие коллекции Chroma и создаст заново.

    Returns:
        Tuple: (chroma_client, embeddings_object, bm25_retrievers_map, tenant_documents_map, tenant_raw_data_map, service_details_map)
               Или (None, None, {}, {}, {}, {}) в случае критической ошибки.
    """
    embeddings_object = None
    chroma_client = None
    bm25_retrievers_map: Dict[str, BM25Retriever] = {}
    tenant_documents_map: Dict[str, List[Document]] = {} 
    tenant_raw_data_map: Dict[str, List[Dict]] = {} 
    service_details_map_loaded: Dict[Tuple[str, str], Dict[str, Any]] = {} 

    #
    try:
        embeddings_object = GigaChatEmbeddings(
            credentials=embedding_credentials, model=embedding_model,
            verify_ssl_certs=verify_ssl_certs, scope=embedding_scope, timeout=180  
        )
        logger.info(f"Эмбеддинги GigaChat ({embedding_model}) инициализированы.")
    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации эмбеддингов: {e}", exc_info=True)
        return None, None, {}, {}, {}, {} 

    try:
        chroma_client = chromadb.PersistentClient(path=chroma_persist_dir)
        logger.info(f"Клиент ChromaDB инициализирован. Данные в: {chroma_persist_dir}")

        
        if force_recreate_chroma:
            logger.warning("Флаг force_recreate_chroma установлен. Попытка удаления всех коллекций тенантов...")
           
            try:
                existing_collections = chroma_client.list_collections()
                for collection in existing_collections:
                    if collection.name.startswith(TENANT_COLLECTION_PREFIX):
                        try:
                            chroma_client.delete_collection(collection.name)
                            logger.info(f"Коллекция тенанта '{collection.name}' удалена.")
                        except Exception as del_e:
                            logger.error(f"Ошибка удаления коллекции '{collection.name}': {del_e}")
            except Exception as list_e:
                 logger.error(f"Не удалось получить список коллекций для удаления: {list_e}")

    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации ChromaDB клиента: {e}", exc_info=True)
        return None, embeddings_object, {}, {}, {}, {} 
  
    try:
        service_details_map_loaded = load_service_details()
    except Exception as e:
        logger.error(f"Не удалось загрузить service_details.json: {e}. Продолжение без дополнительных деталей услуг.", exc_info=True)
       


    tenant_files = discover_all_json_files(base_dir=data_dir)
    if not tenant_files:
        logger.warning(f"Не найдено JSON файлов тенантов в директории: {data_dir}")
      
        return chroma_client, embeddings_object, {}, {}, {}, service_details_map_loaded 

    for tenant_file_path in tenant_files:
        base_name = os.path.basename(tenant_file_path)
        tenant_id, _ = os.path.splitext(base_name)
        if not tenant_id:
            logger.warning(f"Не удалось извлечь tenant_id из имени файла: {base_name}. Пропуск файла.")
            continue

        collection_name = build_collection_name(tenant_id)
        logger.info(f"--- Обработка тенанта: {tenant_id} (Коллекция: {collection_name}) ---")

        
        base_docs, tenant_raw_data = load_tenant_base_data(tenant_id, tenant_file_path, service_details_map_loaded)
        if tenant_raw_data:
            
            tenant_raw_data_map[tenant_id] = tenant_raw_data # 

       
        clinic_info_docs = []
        if tenant_config_manager:
            clinic_info_data = tenant_config_manager.load_tenant_clinic_info(tenant_id)
            clinic_info_docs = clinic_info_data_to_docs(clinic_info_data)
        else:
            logger.warning(f"tenant_config_manager не импортирован, clinic_info для тенанта {tenant_id} не будет загружена.")

      
        all_tenant_docs = base_docs + clinic_info_docs

        if not all_tenant_docs:
            logger.warning(f"Нет документов для индексации для тенанта {tenant_id}. Пропуск индексации.")
            continue

        logger.info(f"Всего {len(all_tenant_docs)} документов для индексации для тенанта {tenant_id} (Base: {len(base_docs)}, ClinicInfo: {len(clinic_info_docs)}).")

        
        tenant_documents_map[tenant_id] = all_tenant_docs

        
        index_documents_to_collection(
            chroma_client=chroma_client,
            embeddings_object=embeddings_object,
            collection_name=collection_name,
            documents=all_tenant_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
          
            force_recreate=False
        )

      
        try:
            tenant_bm25 = BM25Retriever.from_documents(all_tenant_docs, k=search_k)
            bm25_retrievers_map[tenant_id] = tenant_bm25
            logger.info(f"Создан BM25 ретривер для тенанта '{tenant_id}' (k={search_k}) на {len(all_tenant_docs)} док-х.")
        except Exception as e:
            logger.error(f"Ошибка создания BM25 ретривера для тенанта '{tenant_id}': {e}", exc_info=True)

    logger.info(f"=== Инициализация RAG завершена. Обработано тенантов: {len(bm25_retrievers_map)} ===")
    
    return chroma_client, embeddings_object, bm25_retrievers_map, tenant_documents_map, tenant_raw_data_map, service_details_map_loaded



def reindex_tenant_specific_data(
    tenant_id: str,
    chroma_client: chromadb.ClientAPI,
    embeddings_object: GigaChatEmbeddings,
    bm25_retrievers_map: Dict[str, BM25Retriever],
    tenant_documents_map: Dict[str, List[Document]],
    tenant_raw_data_map: Dict[str, List[Dict[str, Any]]], 
    service_details_map: Dict[Tuple[str, str], Dict[str, Any]], 
    base_data_dir: str, 
    chunk_size: int,
    chunk_overlap: int,
    search_k: int
) -> bool:
    """
    Переиндексирует данные для одного конкретного тенанта.
    Обновляет его Chroma коллекцию, BM25 ретривер и записи в картах документов.

    Args:
        tenant_id: ID тенанта для переиндексации.
        chroma_client: Инициализированный клиент ChromaDB.
        embeddings_object: Инициализированный объект эмбеддингов.
        bm25_retrievers_map: Глобальная карта BM25 ретриверов (будет обновлена).
        tenant_documents_map: Глобальная карта документов тенантов (будет обновлена).
        tenant_raw_data_map: Глобальная карта сырых данных тенантов (будет обновлена).
        service_details_map: Глобальная карта деталей услуг (будет обновлена).
        base_data_dir: Путь к директории с базовыми JSON файлами (например, 'base').
        chunk_size: Размер чанка для сплиттера.
        chunk_overlap: Перекрытие чанков.
        search_k: Количество документов для возврата BM25 ретривером.

    Returns:
        True, если переиндексация прошла успешно, иначе False.
    """
    logger.info(f"--- Начало переиндексации для тенанта: {tenant_id} ---")

    if not tenant_id:
        logger.error("Tenant ID не может быть пустым для переиндексации.")
        return False
    if not chroma_client or not embeddings_object:
        logger.error(f"Chroma клиент или объект эмбеддингов не предоставлены для переиндексации тенанта {tenant_id}.")
        return False

    collection_name = build_collection_name(tenant_id)
    tenant_file_path = os.path.join(base_data_dir, f"{tenant_id}.json")

    if not os.path.exists(tenant_file_path):
        logger.warning(f"Базовый JSON файл для тенанта {tenant_id} не найден по пути: {tenant_file_path}. Будут использованы только clinic_info, если есть.")
        base_docs = []
        current_tenant_raw_base_data = []
    else:
 
        base_docs, current_tenant_raw_base_data = load_tenant_base_data(tenant_id, tenant_file_path, service_details_map)
        if current_tenant_raw_base_data is None: 
            current_tenant_raw_base_data = []


    clinic_info_docs = []
    if tenant_config_manager:
        try:
            clinic_info_data = tenant_config_manager.load_tenant_clinic_info(tenant_id)
            clinic_info_docs = clinic_info_data_to_docs(clinic_info_data)
            logger.info(f"Для тенанта {tenant_id} загружено {len(clinic_info_docs)} док-ов из clinic_info.")
        except Exception as e:
            logger.error(f"Ошибка загрузки clinic_info для тенанта {tenant_id} при переиндексации: {e}", exc_info=True)
            
    else:
        logger.warning(f"tenant_config_manager не импортирован, clinic_info для тенанта {tenant_id} не будет перезагружена.")

 
    all_tenant_docs = base_docs + clinic_info_docs

    if not all_tenant_docs:
        logger.warning(f"Нет документов (base + clinic_info) для переиндексации для тенанта {tenant_id}. Возможно, будут удалены старые данные.")
       
        collection_name = build_collection_name(tenant_id)

      
        try:
            chroma_client.delete_collection(collection_name)
        except Exception as e:
            logger.info(f"Попытка удаления коллекции '{collection_name}' для тенанта {tenant_id} (возможно, ее не было): {e}")

       
        try:
            chroma_client.create_collection(name=collection_name, embedding_function=embeddings_object)
            logger.info(f"Создана пустая коллекция-заглушка '{collection_name}' для тенанта {tenant_id}.")
        except Exception as e:
            logger.warning(f"Не удалось создать пустую коллекцию '{collection_name}' для тенанта {tenant_id}: {e}")

      
        bm25_retrievers_map.pop(tenant_id, None)
        tenant_documents_map.pop(tenant_id, None)
        tenant_raw_data_map.pop(tenant_id, None)

    
        return True
    
        tenant_documents_map.pop(tenant_id, None)
        tenant_raw_data_map.pop(tenant_id, None)
        bm25_retrievers_map.pop(tenant_id, None)
        try:
            chroma_client.delete_collection(collection_name)
            logger.info(f"Коллекция Chroma '{collection_name}' для тенанта {tenant_id} удалена, так как нет новых документов.")
        except Exception as e:
          
            logger.info(f"Попытка удаления коллекции '{collection_name}' для тенанта {tenant_id} (возможно, ее не было): {e}")
        logger.info(f"--- Переиндексация для тенанта: {tenant_id} завершена (данные удалены, так как нет новых документов) ---")
        return True 

    logger.info(f"Всего {len(all_tenant_docs)} документов для переиндексации для тенанта {tenant_id} (Base: {len(base_docs)}, ClinicInfo: {len(clinic_info_docs)}).")

    tenant_documents_map[tenant_id] = all_tenant_docs
    tenant_raw_data_map[tenant_id] = current_tenant_raw_base_data 

   
    logger.info(f"Принудительное пересоздание коллекции '{collection_name}' для тенанта {tenant_id}.")
    vectorstore = index_documents_to_collection(
        chroma_client=chroma_client,
        embeddings_object=embeddings_object,
        collection_name=collection_name,
        documents=all_tenant_docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_recreate=True 
    )

    if not vectorstore:
        logger.error(f"Не удалось пересоздать/проиндексировать коллекцию Chroma '{collection_name}' для тенанта {tenant_id}.")
        
        return False

  
    try:
        tenant_bm25 = BM25Retriever.from_documents(all_tenant_docs, k=search_k)
        bm25_retrievers_map[tenant_id] = tenant_bm25
        logger.info(f"BM25 ретривер для тенанта '{tenant_id}' (k={search_k}) обновлен/создан на {len(all_tenant_docs)} док-х.")
    except Exception as e:
        logger.error(f"Ошибка обновления/создания BM25 ретривера для тенанта '{tenant_id}': {e}", exc_info=True)
        return False 

    logger.info(f"--- Переиндексация для тенанта: {tenant_id} успешно завершена ---")
    return True



_text_normalize_pattern = re.compile(r'\s+')

def normalize_text(text: Optional[str], keep_spaces: bool = False) -> str:
    """
    Приводит строку к нижнему регистру, удаляет дефисы и опционально пробелы.
    Безопасно обрабатывает None, возвращая пустую строку.
    """
    if not text:
        return ""
    normalized = text.lower().replace("-", "")
    if keep_spaces:
        # Заменяем множественные пробелы на один и убираем пробелы по краям
        normalized = _text_normalize_pattern.sub(' ', normalized).strip()
    else:
        normalized = normalized.replace(" ", "")
    return normalized
# --- КОНЕЦ ДОБАВЛЕНИЯ ФУНКЦИИ НОРМАЛИЗАЦИИ ---
