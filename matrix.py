import aiohttp
import asyncio
import aiofiles
import json
import uvicorn
import logging
import time
import os
import numpy as np
import re
import uuid
import shutil # Добавлен импорт shutil
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModel # Убран AutoModelForSequenceClassification
import torch
from voicerecognise import recognize_audio_with_sdk # Убедись, что этот импорт работает
from typing import Dict, List, Optional, Sequence, Any, Union, Literal
from typing_extensions import Annotated, TypedDict
from contextlib import asynccontextmanager

# --- LangChain & LangGraph ---
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage, trim_messages, messages_to_dict # Добавлен messages_to_dict
from langchain_core.documents import Document # Добавлен Document
from langchain.tools import tool
from pydantic import BaseModel, Field as PydanticField
from langchain_core.output_parsers import JsonOutputParser

# --- LangChain Embeddings, VectorStores & Retrievers ---
from langchain_community.embeddings import HuggingFaceEmbeddings # Замена SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# --- GigaChat Specific ---
from langchain_gigachat.chat_models import GigaChat

# --- Настройка логирования ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)-8s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === Константы и Пути ===
BASE_DIR = "base"
EMBEDDINGS_DIR = "embeddings_data" # Папка для FAISS индексов и др. данных
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"
# <<<--- ВАШ КЛЮЧ GIGACHAT ---<<<
# Убедитесь, что переменная окружения GIGACHAT_API_KEY установлена, или вставьте ключ прямо сюда
# GIGACHAT_API_KEY = "ВАШ_КЛЮЧ_ЗДЕСЬ"
GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY", "OTkyYTgyNGYtMjRlNC00MWYyLTg3M2UtYWRkYWVhM2QxNTM1OmNmZjQ0YTkyLTU2YTktNGUwNi04NDY4LTU1NTU0MGNhZGE3MQ==")
if not GIGACHAT_API_KEY:
    logger.error("!!! GIGACHAT_API_KEY не найден! Установите переменную окружения или вставьте ключ в код.")
    # Можно либо выйти, либо продолжить без GigaChat, если логика это позволяет
    # raise SystemExit("GigaChat API Key not found.")

EMBEDDING_MODEL_NAME = "ai-forever/sbert_large_nlu_ru"
RETRIEVER_K = 10     # Кол-во финальных документов для LLM

# === Глобальные переменные ===
device = "mps"
embedding_model_instance = None # Теперь это объект HuggingFaceEmbeddings
giga_chat_model = None
conversation_history: Dict[str, Dict] = {}
tenant_data_cache: Dict[str, Dict] = {}
rag_agent_app = None

# === Модели Pydantic ===
class ExtractedEntities(BaseModel):
    service_name: Optional[str] = PydanticField(None, description="Название или ключевые слова для поиска услуги (из ПОСЛЕДНЕГО запроса)")
    filial: Optional[str] = PydanticField(None, description="Название филиала (например, Москва-сити, Ходынка) (приоритет у ПОСЛЕДНЕГО запроса)")
    employee_name: Optional[str] = PydanticField(None, description="Имя или фамилия специалиста (НЕ глагол, НЕ общее слово типа 'специалист') (приоритет у ПОСЛЕДНЕГО запроса)")
    category_name: Optional[str] = PydanticField(None, description="Название категории услуг (приоритет у ПОСЛЕДНЕГО запроса)")
    price_constraint: Optional[str] = PydanticField(None, description="Описание ценового ОГРАНИЧЕНИЯ из запроса (например, 'дешевле 5000', 'около 10000'), НЕ цена из истории")

class RagAgentState(TypedDict):
    user_query: str
    entities: dict
    search_results: List[dict] # Метаданные топ-N документов
    answer: str
    chat_history: List[dict] # Список словарей для совместимости
    tenant_id: str
    error_message: Optional[str]

# === Вспомогательные Функции ===
def get_tenant_path(tenant_id: str) -> Path:
    tenant_path = Path(EMBEDDINGS_DIR) / tenant_id
    tenant_path.mkdir(parents=True, exist_ok=True)
    return tenant_path

def normalize_text(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text) # Заменяем множественные пробелы на один
    # Оставляем буквы (включая русские), цифры, пробелы и некоторые знаки пунктуации
    text = re.sub(r"[^\w\s\d\n.,?!()+-]", "", text, flags=re.UNICODE)
    return text.lower()

# tokenize_text нужен для BM25Retriever
def tokenize_text(text: str) -> List[str]:
     stopwords = {"и", "в", "на", "с", "по", "для", "как", "что", "это", "но", "а", "или", "у", "о", "же", "за", "к", "из", "от", "так", "то", "все", "он", "она", "они", "мы", "вы", "ты", "я"}
     tokens = re.findall(r'\b\w{2,}\b', text.lower()) # Берем слова длиной 2+
     return [word for word in tokens if word not in stopwords]

def extract_text_fields(record: dict) -> str:
    filial = record.get("filialName", "")
    category = record.get("categoryName", "")
    service = record.get("serviceName", "")
    service_desc = record.get("serviceDescription", "")
    price = record.get("price", "")
    specialist = record.get("employeeFullName", "")
    spec_desc = record.get("employeeDescription", "")
    employeeExperience = record.get("employeeExperience", "")
    employeeTechnologies = record.get("employeeTechnologies", [])
    # Преобразуем список технологий в строку, если это список
    tech_str = ", ".join(map(str, employeeTechnologies)) if isinstance(employeeTechnologies, list) else str(employeeTechnologies)

    # Собираем текст, который будет использоваться и для эмбеддингов, и для BM25
    text = (
        f"Услуга: {service}. Категория: {category}. Цена: {price} руб. Филиал: {filial}. "
        f"Специалист: {specialist}. "
        f"Описание услуги: {service_desc}. Описание специалиста: {spec_desc}. "
        f"Опыт: {employeeExperience}. Технологии: {tech_str}."
    )
    return normalize_text(text)

# --- Функции Загрузки/Подготовки Данных (Обновленные под EnsembleRetriever) ---

async def load_json_data(tenant_id: str) -> List[dict]:
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    if not os.path.exists(file_path):
        logger.error(f"Файл {file_path} не найден")
        return []
    logger.info(f"Загрузка данных из {file_path}")
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
    except Exception as e:
        logger.error(f"Ошибка чтения/парсинга JSON {file_path}: {e}", exc_info=True)
        return []

    records = []
    branches = data.get("data", {}).get("branches", [])
    if not branches:
        logger.warning(f"В файле {file_path} не найдены 'branches'.")
        return []

    for branch in branches:
         filial_id = branch.get("id", "")
         filial_name = branch.get("name", "Филиал не указан")
         categories = branch.get("categories", [])
         for category in categories:
            category_id = category.get("id", "")
            category_name = category.get("name", "Категория не указана")
            services = category.get("services", [])
            for service in services:
                service_id = service.get("id", "")
                service_name = service.get("name", "Услуга не указана")
                price = service.get("price", "Цена не указана")
                service_description = service.get("description", "")
                duration = service.get("duration", 0)
                employees = service.get("employees", [])
                if employees:
                    for emp in employees:
                        employee_id = emp.get("id", "")
                        employee_full_name = emp.get("full_name", "Специалист не указан")
                        employee_description = emp.get("description", "Описание не указано")
                        employee_experience = emp.get("experience", "")
                        # Убедимся, что technologies - это список строк
                        techs_raw = emp.get("technologies", [])
                        employee_technologies = [str(t) for t in techs_raw if t] if isinstance(techs_raw, list) else []

                        record = {
                            "filialId": filial_id, "filialName": filial_name,
                            "categoryId": category_id, "categoryName": category_name,
                            "serviceId": service_id, "serviceName": service_name,
                            "serviceDescription": service_description, "price": price,
                            "duration": duration,
                            "employeeId": employee_id, "employeeFullName": employee_full_name,
                            "employeeDescription": employee_description,
                            "employeeExperience": employee_experience,
                            "employeeTechnologies": employee_technologies # Сохраняем как список
                        }
                        records.append(record)
                else: # Услуга без конкретного сотрудника
                    record = {
                        "filialId": filial_id, "filialName": filial_name,
                        "categoryId": category_id, "categoryName": category_name,
                        "serviceId": service_id, "serviceName": service_name,
                        "serviceDescription": service_description, "price": price,
                        "duration": duration,
                        "employeeId": None, "employeeFullName": "Любой специалист",
                        "employeeDescription": "",
                        "employeeExperience": "", "employeeTechnologies": []
                    }
                    records.append(record)
    logger.info(f"Загружено {len(records)} записей для тенанта {tenant_id}")
    return records

# Функция prepare_data теперь создает и кэширует EnsembleRetriever
async def prepare_data_for_tenant(tenant_id: str) -> Optional[Dict]:
    """Загружает или создает данные и EnsembleRetriever для тенанта."""
    global tenant_data_cache, embedding_model_instance
    if tenant_id in tenant_data_cache:
        logger.info(f"Используем кэшированный EnsembleRetriever для тенанта {tenant_id}")
        return tenant_data_cache[tenant_id]

    if not embedding_model_instance:
         logger.error("Модель эмбеддингов не инициализирована!")
         return None

    tenant_path = get_tenant_path(tenant_id)
    faiss_index_path = tenant_path / "faiss_index" # Папка для FAISS индекса
    records_cache_file = tenant_path / "records_cache.json" # Файл для кэша записей

    # Пытаемся загрузить из кэша
    if faiss_index_path.exists() and records_cache_file.exists():
        try:
            logger.info(f"Загрузка кэшированного FAISS и документов для {tenant_id}...")
            # Загружаем FAISS индекс
            faiss_store = FAISS.load_local(
                folder_path=str(faiss_index_path),
                embeddings=embedding_model_instance,
                index_name="index", # Имя файла индекса по умолчанию
                allow_dangerous_deserialization=True # Необходимо для pickle внутри FAISS load
            )
            # Увеличиваем k для ретривера, чтобы ансамбль имел больше выбора
            vector_retriever = faiss_store.as_retriever(search_kwargs={"k": max(RETRIEVER_K * 2, 15)})

            # Загружаем записи/документы для BM25
            async with aiofiles.open(records_cache_file, "r", encoding="utf-8") as f:
                cached_data = json.loads(await f.read())
            records = cached_data["records"]
            # Создаем документы LangChain (нужны для BM25)
            documents = []
            for r in records:
                try:
                    page_content = extract_text_fields(r)
                    # Очистка метаданных перед созданием Document
                    clean_metadata = {k: v for k, v in r.items() if v is not None and isinstance(v, (str, int, float, bool))}
                    # Списки преобразуем в строки или JSON-строки для хранения в метаданных
                    for k, v in r.items():
                        if isinstance(v, list):
                            # Можно просто объединить в строку
                            clean_metadata[k] = ", ".join(map(str, v))
                            # Или сохранить как JSON-строку, если структура важна
                            # try: clean_metadata[k] = json.dumps(v, ensure_ascii=False)
                            # except: clean_metadata[k] = "[неконвертируемый список]"
                    documents.append(Document(page_content=page_content, metadata=clean_metadata))
                except Exception as doc_err:
                    logger.warning(f"Ошибка создания документа из кэша для записи {r.get('serviceId', 'N/A')}: {doc_err}")
                    continue

            if not documents:
                 raise ValueError("Не удалось создать документы из кэшированных записей.")

            # Создаем BM25 ретривер (требует токены, если нет 'text')
            bm25_retriever = BM25Retriever.from_documents(
                 documents=documents,
                 # Можно явно указать функцию токенизации, если нужно
                 # preprocess_func=tokenize_text
            )
            bm25_retriever.k = max(RETRIEVER_K * 2, 15) # Берем больше кандидатов для ансамбля

            # Создаем Ensemble Retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6] # Веса можно настроить (чуть больший вес векторному)
            )
            logger.info(f"EnsembleRetriever для {tenant_id} успешно загружен из кэша.")
            data = {"ensemble_retriever": ensemble_retriever, "records": records} # Сохраняем записи
            tenant_data_cache[tenant_id] = data
            return data
        except Exception as e:
            logger.error(f"Ошибка загрузки кэшированных данных для {tenant_id}: {e}. Пересоздаем.", exc_info=True)
            # Очищаем поврежденные файлы кэша
            if faiss_index_path.exists(): shutil.rmtree(faiss_index_path, ignore_errors=True)
            if records_cache_file.exists():
                 try: os.remove(records_cache_file)
                 except OSError as os_err: logger.warning(f"Не удалось удалить {records_cache_file}: {os_err}")

    else:
         logger.info(f"Кэшированные файлы для {tenant_id} не найдены или повреждены. Создаем новые.")

    # --- Создаем заново ---
    logger.info(f"Подготовка новых данных и ретривера для тенанта {tenant_id}")
    records = await load_json_data(tenant_id)
    if not records:
        logger.error(f"Не удалось загрузить записи для тенанта {tenant_id}. Ретривер не будет создан.")
        return None

    # Создаем документы LangChain
    documents = []
    for r in records:
         try:
             page_content = extract_text_fields(r)
             # Очищаем метаданные от None и сложных типов перед созданием Document
             clean_metadata = {k: v for k, v in r.items() if v is not None and isinstance(v, (str, int, float, bool))}
             # Списки преобразуем в строки или JSON-строки
             for k, v in r.items():
                 if isinstance(v, list):
                     # Можно просто объединить в строку
                     clean_metadata[k] = ", ".join(map(str, v))
                     # Или сохранить как JSON-строку
                     # try: clean_metadata[k] = json.dumps(v, ensure_ascii=False)
                     # except: clean_metadata[k] = "[неконвертируемый список]"

             # Проверяем размер метаданных (некоторые хранилища имеют ограничения)
             metadata_size = len(json.dumps(clean_metadata))
             if metadata_size > 4000: # Примерное ограничение, настройте при необходимости
                 logger.warning(f"Метаданные для записи {r.get('serviceId', 'N/A')} слишком велики ({metadata_size} байт), могут быть урезаны.")
                 # Можно урезать или удалить большие поля метаданных здесь
             documents.append(Document(page_content=page_content, metadata=clean_metadata))
         except Exception as doc_err:
              logger.warning(f"Ошибка создания документа для записи {r.get('serviceId', 'N/A')}: {doc_err}", exc_info=True)
              continue # Пропускаем проблемную запись

    if not documents:
        logger.error(f"Не удалось создать документы LangChain для тенанта {tenant_id}.")
        return None
    logger.info(f"Создано {len(documents)} документов LangChain.")

    try:
        # Создаем FAISS индекс и сохраняем его
        logger.info("Создание и сохранение FAISS индекса...")
        faiss_store = await FAISS.afrom_documents(documents, embedding_model_instance)
        faiss_store.save_local(folder_path=str(faiss_index_path), index_name="index")
        vector_retriever = faiss_store.as_retriever(search_kwargs={"k": max(RETRIEVER_K * 2, 15)})
        logger.info("FAISS индекс создан и сохранен.")

        # Создаем BM25 ретривер
        logger.info("Создание BM25 ретривера...")
        bm25_retriever = BM25Retriever.from_documents(
            documents=documents,
            # preprocess_func=tokenize_text # Явно указываем, если нужно
        )
        bm25_retriever.k = max(RETRIEVER_K * 2, 15)
        logger.info("BM25 ретривер создан.")

        # Создаем Ensemble Retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.4, 0.6] # Настройте веса
        )
        logger.info(f"EnsembleRetriever для {tenant_id} успешно создан.")

        # Сохраняем записи в кэш
        logger.info("Сохранение записей в кэш...")
        async with aiofiles.open(records_cache_file, "w", encoding="utf-8") as f:
             await f.write(json.dumps({"records": records}, ensure_ascii=False, indent=2)) # Сохраняем только записи

        data = {"ensemble_retriever": ensemble_retriever, "records": records}
        tenant_data_cache[tenant_id] = data # Кэшируем
        return data

    except Exception as e:
        logger.error(f"Ошибка во время подготовки данных для {tenant_id}: {e}", exc_info=True)
        # Очищаем возможно поврежденные файлы
        if faiss_index_path.exists(): shutil.rmtree(faiss_index_path, ignore_errors=True)
        if records_cache_file.exists():
            try: os.remove(records_cache_file)
            except OSError as os_err: logger.warning(f"Не удалось удалить {records_cache_file}: {os_err}")
        return None

async def update_json_file(mydtoken: str, tenant_id: str):
    """Обновляет JSON файл с данными тенанта из API, если файл устарел."""
    global tenant_data_cache
    tenant_path = get_tenant_path(tenant_id)
    base_file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    faiss_index_path = tenant_path / "faiss_index"
    records_cache_file = tenant_path / "records_cache.json"

    needs_update = True
    update_interval = 86400 # 24 часа в секундах
    if os.path.exists(base_file_path):
        try:
            file_age = time.time() - os.path.getmtime(base_file_path)
            if file_age < update_interval:
                logger.info(f"Файл {base_file_path} актуален (возраст {file_age:.0f}с < {update_interval}с), пропускаем обновление API.")
                needs_update = False
        except OSError as e:
             logger.warning(f"Не удалось получить время модификации {base_file_path}: {e}. Обновляем.")

    if needs_update:
        logger.info(f"Обновление базового JSON для тенанта {tenant_id} из API {API_URL}...")
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {mydtoken}"}
                params = {"tenantId": tenant_id}
                async with session.get(API_URL, headers=headers, params=params, timeout=60) as response: # Добавлен таймаут
                    logger.info(f"API запрос для {tenant_id}: Статус {response.status}")
                    response.raise_for_status() # Вызовет исключение для 4xx/5xx
                    data = await response.json()

                    if data.get("code") == 200 and "data" in data and "branches" in data["data"]:
                        async with aiofiles.open(base_file_path, "w", encoding="utf-8") as json_file:
                            await json_file.write(json.dumps(data, ensure_ascii=False, indent=4))
                        logger.info(f"Базовый JSON для {tenant_id} успешно обновлен.")
                        # --- Очистка кэша и старых индексов ПЕРЕД новой подготовкой ---
                        if tenant_id in tenant_data_cache:
                            del tenant_data_cache[tenant_id]
                            logger.info(f"Кэш ретривера для {tenant_id} очищен.")
                        if faiss_index_path.exists():
                            shutil.rmtree(faiss_index_path, ignore_errors=True)
                            logger.info(f"Удален старый FAISS индекс для {tenant_id}.")
                        if records_cache_file.exists():
                             try: os.remove(records_cache_file)
                             except OSError as e: logger.warning(f"Не удалось удалить кэш записей {records_cache_file}: {e}")
                             logger.info(f"Удален кэш записей для {tenant_id}.")
                        # --------------------------------------
                        # Не нужно вызывать prepare_data_for_tenant здесь, он вызовется при первом запросе
                    else:
                        logger.error(f"Неожиданный ответ API при обновлении {tenant_id}: code={data.get('code')}, message={data.get('message', 'N/A')}")
        except aiohttp.ClientError as e:
            logger.error(f"Ошибка сети при обновлении файла {tenant_id} из API: {e}")
        except asyncio.TimeoutError:
            logger.error(f"Таймаут при запросе к API для обновления файла {tenant_id}")
        except Exception as e:
            logger.error(f"Неожиданная ошибка при обновлении файла {tenant_id} из API: {e}", exc_info=True)

# === Узлы LangGraph для RAG (Адаптированные) ===

# 1. Узел извлечения сущностей
async def service_entity_extraction_node(state: RagAgentState) -> RagAgentState:
    global giga_chat_model
    node_name = "entity_extraction"
    logger.info(f"--- Entering Node: {node_name} ---")
    user_query = state["user_query"]
    chat_history_dicts = state["chat_history"] # Ожидаем List[Dict]

    if not giga_chat_model:
        error_msg = "GigaChat model not initialized for entity extraction."
        logger.error(f"[{node_name}] {error_msg}")
        return {**state, "entities": {"error": error_msg}, "error_message": error_msg}

    parser = JsonOutputParser(pydantic_object=ExtractedEntities)
    formatted_history = ""
    # Берем только последние сообщения для краткого контекста
    history_to_format = chat_history_dicts[-4:] # Например, последние 2 обмена

    if history_to_format:
        history_lines = []
        for msg_dict in history_to_format:
             role = "Пользователь" if msg_dict.get("type") == "human" else "Ассистент"
             content = msg_dict.get("data", {}).get("content", "")
             if content: history_lines.append(f"{role}: {content}")
        if history_lines: formatted_history = "Контекст последнего обмена:\n" + "\n".join(history_lines) + "\n\n---\n"

    # Промпт для извлечения сущностей
    prompt_text = f"""Твоя задача - извлечь из ПОСЛЕДНЕГО запроса пользователя информацию об услуге, филиале, специалисте, категории или цене.
ПРИОРИТЕТ У ИНФОРМАЦИИ ИЗ ПОСЛЕДНЕГО ЗАПРОСА! Краткий контекст диалога используй ТОЛЬКО если в последнем запросе нет явных деталей (например, "а он доступен?" или "запишите меня к нему").

{formatted_history}
ПОСЛЕДНИЙ Запрос пользователя: "{user_query}"

Проанализируй ПОСЛЕДНИЙ запрос (учитывая контекст, если нужно) и верни ТОЛЬКО валидный JSON объект в ОДНУ СТРОКУ со следующими ключами:
"service_name": string | null, "filial": string | null, "employee_name": string | null, "category_name": string | null, "price_constraint": string | null

КРАЙНЕ ВАЖНО:
- `employee_name`: ИЗВЛЕКАЙ ТОЛЬКО ИМЯ И/ИЛИ ФАМИЛИЮ специалиста. НЕ извлекай глаголы ("выполняет", "делает"), общие слова ("специалист"), местоимения, или странные фразы типа "Ожидание Лист" (если это не реальное имя). Если имя не указано явно в ПОСЛЕДНЕМ запросе, ставь null.
- `price_constraint`: Извлекай ТОЛЬКО ОГРАНИЧЕНИЕ цены из ПОСЛЕДНЕГО запроса (например, "дешевле 5000", "около 10000"). НЕ извлекай конкретную цену из КОНТЕКСТА или предыдущих ответов. Если ограничения нет, ставь null.
- `category_name`: Извлекай название категории, если оно явно упомянуто в ПОСЛЕДНЕМ запросе.
- `service_name`: Извлекай название услуги или ключевые слова для ее поиска.
- `filial`: Извлекай название филиала.
- Если какая-то информация отсутствует в ПОСЛЕДНЕМ запросе (и не подразумевается контекстом короткого запроса), ставь null для соответствующего ключа.
- Не придумывай информацию. Строго по тексту запроса и минимальному контексту.

---
Теперь обработай ПОСЛЕДНИЙ запрос пользователя. Помни: ТОЛЬКО JSON В ОДНУ СТРОКУ.

ПОСЛЕДНИЙ Запрос пользователя: "{user_query}"
Ответ (ТОЛЬКО JSON):"""

    # Используем ChatPromptTemplate для более структурированного подхода, хотя PromptTemplate тоже подойдет
    prompt_template = ChatPromptTemplate.from_messages([("system", prompt_text)])
    chain = prompt_template | giga_chat_model | parser

    try:
        logger.info(f"[{node_name}] Extracting entities from query: '{user_query}'")
        # GigaChat ожидает список сообщений или строку, передаем промпт как system message
        # Если используем ChatPromptTemplate, инвок может быть проще: chain.ainvoke({})
        # Но т.к. промпт уже сформирован выше, можно передать напрямую:
        response = await giga_chat_model.ainvoke(prompt_text)
        # Парсим ответ JSON
        entities_result = parser.parse(response.content)
        entities_dict = entities_result.dict() if isinstance(entities_result, BaseModel) else entities_result
        logger.info(f"[{node_name}] Extracted Entities: {entities_dict}")
        return {**state, "entities": entities_dict, "error_message": None}
    except Exception as e:
        error_msg = f"Entity extraction failed: {str(e)}"
        logger.error(f"[{node_name}] !!! {error_msg}", exc_info=True)
        # Попытка извлечь JSON даже при ошибке парсера LangChain, если LLM вернул что-то похожее
        fallback_entities = {}
        try:
            if 'response' in locals() and isinstance(response.content, str):
                 match = re.search(r'\{.*\}', response.content)
                 if match: fallback_entities = json.loads(match.group(0))
        except Exception: pass # Игнорируем ошибки парсинга фоллбэка
        logger.warning(f"[{node_name}] Fallback extracted entities: {fallback_entities}")
        return {**state, "entities": fallback_entities or {"error": error_msg}, "error_message": error_msg}

# 2. Узел поиска (адаптированный под EnsembleRetriever)
async def retrieval_node(state: RagAgentState) -> RagAgentState:
    node_name = "retrieval"
    logger.info(f"--- Entering Node: {node_name} ---")
    user_query = state["user_query"]
    tenant_id = state["tenant_id"]
    entities = state.get("entities", {})

    # Проверяем ошибки предыдущего шага
    if state.get("error_message") and "Entity extraction failed" in state["error_message"]:
         logger.warning(f"[{node_name}] Skipping retrieval due to entity extraction error.")
         # Можно вернуть пустой результат или ошибку
         return {**state, "search_results": [], "error_message": state["error_message"]} # Передаем ошибку дальше

    # Получаем данные и ретривер для тенанта
    tenant_data = await prepare_data_for_tenant(tenant_id)
    if not tenant_data or "ensemble_retriever" not in tenant_data:
        error_msg = f"Retriever not available for tenant {tenant_id}."
        logger.error(f"[{node_name}] {error_msg}")
        return {**state, "search_results": [], "error_message": error_msg}

    ensemble_retriever: EnsembleRetriever = tenant_data["ensemble_retriever"]

    # Используем сущности для улучшения запроса (если они есть)
    # Это простой пример, можно усложнить логику
    search_query = user_query
    query_parts = [user_query]
    if entities and not entities.get('error'):
        # Добавляем ключевые слова из сущностей
        if entities.get("service_name"): query_parts.append(f"Услуга {entities['service_name']}")
        if entities.get("category_name"): query_parts.append(f"Категория {entities['category_name']}")
        if entities.get("employee_name"): query_parts.append(f"Специалист {entities['employee_name']}")
        if entities.get("filial"): query_parts.append(f"Филиал {entities['filial']}")
        # Цена сложнее, т.к. это ограничение, а не ключ. слова. Можно использовать для фильтрации *после* поиска.
        search_query = " ".join(list(dict.fromkeys(query_parts))) # Удаляем дубликаты, сохраняя порядок
    logger.info(f"[{node_name}] Using enhanced search query: '{search_query}'")


    try:
        logger.info(f"[{node_name}] Searching using Ensemble Retriever...")
        # Используем асинхронный вызов ретривера
        search_results_docs: List[Document] = await ensemble_retriever.ainvoke(search_query)
        logger.info(f"[{node_name}] Ensemble Retriever returned {len(search_results_docs)} documents.")

        # Извлекаем метаданные из найденных документов
        unique_results_metadata: List[dict] = []
        processed_keys = set()
        for doc in search_results_docs:
             meta = doc.metadata
             if not isinstance(meta, dict):
                 logger.warning(f"[{node_name}] Document has non-dict metadata: {type(meta)}")
                 continue # Пропускаем, если метаданные не словарь

             # Ключ для дедупликации (услуга+филиал+сотрудник)
             key = (meta.get("serviceId"), meta.get("filialId"), meta.get("employeeId"))

             # Простая дедупликация по ключу
             # Используем None как часть ключа, если ID отсутствует
             if key not in processed_keys:
                 # Пытаемся восстановить списки из строк/JSON-строк, если нужно для LLM
                 restored_meta = meta.copy()
                 for k, v in restored_meta.items():
                      if k == "employeeTechnologies" and isinstance(v, str):
                           # Пытаемся распарсить строку обратно в список
                           if v.startswith('[') and v.endswith(']'): # Похоже на JSON
                                try: restored_meta[k] = json.loads(v)
                                except: restored_meta[k] = [s.strip() for s in v.split(',') if s.strip()] # Простой сплит
                           else: # Просто сплит по запятой
                                restored_meta[k] = [s.strip() for s in v.split(',') if s.strip()]

                 unique_results_metadata.append(restored_meta)
                 processed_keys.add(key)

        # Оставляем только топ-K уникальных результатов
        final_results = unique_results_metadata[:RETRIEVER_K]

        logger.info(f"[{node_name}] Returning {len(final_results)} final unique results (metadata) for LLM processing.")
        if final_results:
             logger.debug(f"[{node_name}] Results sample: {json.dumps(final_results[0], ensure_ascii=False, indent=2)}")

        return {**state, "search_results": final_results, "error_message": None}

    except Exception as e:
        error_msg = f"Retrieval failed: {str(e)}"
        logger.error(f"[{node_name}] !!! {error_msg}", exc_info=True)
        return {**state, "search_results": [], "error_message": error_msg}

# 3. Узел генерации ответа
async def answer_generation_node(state: RagAgentState) -> RagAgentState:
    global giga_chat_model
    node_name = "answer_generation"
    logger.info(f"--- Entering Node: {node_name} ---")

    # Проверяем ошибки предыдущих шагов
    if state.get("error_message"):
        logger.warning(f"[{node_name}] Skipping generation due to previous error: {state['error_message']}")
        # Формируем ответ об ошибке для пользователя
        error_source = "при поиске информации"
        if "Entity extraction failed" in state['error_message']: error_source = "при анализе вашего запроса"
        if "Retriever not available" in state['error_message']: error_source = "при доступе к базе данных"

        answer = f"Извините, произошла внутренняя ошибка {error_source}. Пожалуйста, попробуйте переформулировать запрос или повторите попытку позже."
        return {**state, "answer": answer} # Возвращаем state с сообщением об ошибке

    user_query = state["user_query"]
    entities = state.get("entities", {})
    search_results = state["search_results"] # Это список словарей (метаданные)
    chat_history_dicts = state["chat_history"] # Список словарей

    if not giga_chat_model:
        error_msg = "GigaChat model not initialized for answer generation."
        logger.error(f"[{node_name}] {error_msg}")
        # Отдаем пользователю сообщение об ошибке
        return {**state, "answer": "Извините, сервис временно недоступен. Попробуйте позже.", "error_message": error_msg}

    # Формирование истории для LLM (как список BaseMessage)
    messages_for_gigachat: List[BaseMessage] = []
    # Добавляем системный промпт в начало
    system_prompt = """ТЫ — ЭКСПЕРТНЫЙ ассистент клиники MED YU MED по имени Аида. Твоя главная задача — отвечать на вопросы пользователя об услугах, филиалах, специалистах и ценах, основываясь СТРОГО на предоставленной информации ('Найденная информация'). Ты работаешь как RAG-модель.

ИНСТРУКЦИИ:
1.  **Анализ:** Внимательно прочитай "Вопрос пользователя" и "Историю диалога". Изучи "Найденную информацию" (это топ результатов поиска) и "Извлеченные детали из запроса" (что пользователь ищет).
2.  **Выбор:** Из списка "Найденная информация" выбери ТОЛЬКО те записи, которые **наиболее точно соответствуют** "Вопросу пользователя" И "Извлеченным деталям" (услуга, филиал, специалист, категория, цена, если указаны). Обрати внимание на `price_constraint`, если он есть.
3.  **Ответ:**
    *   **Если найдены релевантные записи:** Сформируй четкий, вежливый и полезный ответ на "Вопрос пользователя", используя информацию **ИСКЛЮЧИТЕЛЬНО из выбранных релевантных записей**. Укажи название услуги, филиал, цену, специалиста (если он не "Любой специалист"). Если записей несколько, перечисли их кратко и понятно, возможно, сгруппировав. Подчеркни ключевую информацию.
    *   **Если релевантных записей НЕ найдено (или список пуст):** Вежливо сообщи пользователю, что по его критериям информация в базе не найдена. Можно предложить изменить запрос (например, "Попробуйте указать другую категорию или филиал").
    *   **Уточнение:** Если запрос неясный или найденных вариантов много, задай уточняющий вопрос.
4.  **ЗАПРЕЩЕНО:**
    *   **Придумывать информацию:** Не добавляй детали, которых нет в найденных релевантных записях.
    *   **Использовать старую информацию:** НЕ используй информацию из "Истории диалога", если она НЕ подтверждается актуальной "Найденной информацией".
    *   **Смешивать данные:** Не бери имя специалиста из одной найденной записи, а цену из другой для формирования описания ОДНОЙ услуги/специалиста.
    *   **Давать общие советы или извинения не по теме**, если информация найдена. Будь конкретным.
    *   **Предлагать запись на прием** - твоя задача только информировать.
5.  **Стиль:** Говори вежливо, профессионально, но дружелюбно. Обращайся на "Вы".

---"""
    messages_for_gigachat.append(SystemMessage(content=system_prompt))

    # Добавляем историю диалога
    history_to_process = chat_history_dicts[-6:] # Берем недавнюю историю
    for msg_dict in history_to_process:
         msg_type = msg_dict.get("type")
         content = msg_dict.get("data", {}).get("content")
         if content:
             if msg_type == "human": messages_for_gigachat.append(HumanMessage(content=content))
             elif msg_type == "ai": messages_for_gigachat.append(AIMessage(content=content))

    # Форматирование результатов поиска для контекста LLM
    context_text = ""
    valid_results_count = 0
    if search_results:
        context_parts = [f"Найденная информация (результаты поиска, до {len(search_results)} записей):"]
        for i, r in enumerate(search_results):
            if isinstance(r, dict):
                service_name = r.get('serviceName', 'Н/Д')
                filial_name = r.get('filialName', 'Н/Д')
                price = r.get('price', 'Н/Д')
                employee_name = r.get('employeeFullName', 'Любой специалист')
                category_name = r.get('categoryName', '')
                service_desc = r.get('serviceDescription', '')
                employee_desc = r.get('employeeDescription', '')

                # Формируем более подробное описание для LLM
                entry = f"{i+1}. Услуга: {service_name}"
                if category_name: entry += f" (Категория: {category_name})"
                entry += f"\n   Филиал: {filial_name}"
                entry += f"\n   Цена: {price} руб."
                entry += f"\n   Специалист: {employee_name}"
                if service_desc: entry += f"\n   Описание услуги: {service_desc[:150]}..." # Ограничиваем длину
                if employee_name != "Любой специалист" and employee_desc: entry += f"\n   Описание специалиста: {employee_desc[:150]}..."
                context_parts.append(entry)
                valid_results_count += 1
            else:
                 logger.warning(f"[{node_name}] search_result item is not a dict: {r}")

        if valid_results_count > 0:
            context_text = "\n---\n".join(context_parts)
        else:
            context_text = "Поиск не дал релевантных результатов в базе."
    else:
        context_text = "Поиск не дал результатов."

    # Формирование строки с извлеченными сущностями
    entities_prompt_part = ""
    if entities and not entities.get('error'):
        valid_entities = {k: v for k, v in entities.items() if v is not None and k != 'error'}
        if valid_entities:
            entities_prompt_part = f"Извлеченные детали из запроса (для фокусировки поиска и ответа): {json.dumps(valid_entities, ensure_ascii=False)}"
        else: entities_prompt_part = "Детали из запроса не извлечены (общий вопрос)."
    elif entities and entities.get('error'): entities_prompt_part = f"Ошибка извлечения деталей: {entities['error']}"
    else: entities_prompt_part = "Детали из запроса не извлечены."

    # Формируем финальный HumanMessage с контекстом и вопросом
    final_prompt_content = f"""{context_text}

{entities_prompt_part}
--- КОНЕЦ ДАННЫХ ---

Вопрос пользователя: {user_query}

Пожалуйста, сформируй ответ, следуя ИНСТРУКЦИЯМ из системного сообщения. Базируйся СТРОГО на 'Найденной информации', релевантной запросу и деталям. Если ничего не найдено, сообщи об этом.
Ассистент Аида:"""

    messages_for_gigachat.append(HumanMessage(content=final_prompt_content))
    logger.info(f"[{node_name}] Total messages for LLM: {len(messages_for_gigachat)}. Final prompt length approx: {len(final_prompt_content)} chars.")
    # logger.debug(f"[{node_name}] Final prompt content sample:\n{final_prompt_content[:1000]}...")

    try:
        # Асинхронный вызов GigaChat
        response = await giga_chat_model.ainvoke(input=messages_for_gigachat)
        answer = response.content
        logger.info(f"[{node_name}] Answer from GigaChat received ({len(answer)} chars).")
        # logger.debug(f"[{node_name}] Answer sample: {answer[:500]}...")
        return {**state, "answer": answer, "error_message": None} # Успешная генерация
    except Exception as e:
        error_msg = f"GigaChat answer generation failed: {str(e)}"
        logger.error(f"[{node_name}] !!! {error_msg}", exc_info=True)
        # Возвращаем состояние с сообщением об ошибке
        return {**state, "answer": "Извините, произошла ошибка при подготовке ответа. Попробуйте позже.", "error_message": error_msg}

# === Определение LangGraph графа ===
def build_rag_graph():
    workflow = StateGraph(RagAgentState)
    # Добавляем узлы
    workflow.add_node("entity_extraction", service_entity_extraction_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("answer_generation", answer_generation_node)

    # Определяем ребра (порядок выполнения)
    workflow.set_entry_point("entity_extraction")
    workflow.add_edge("entity_extraction", "retrieval")
    workflow.add_edge("retrieval", "answer_generation")
    workflow.add_edge("answer_generation", END) # Конец графа после генерации ответа

    # Компилируем граф
    app = workflow.compile()
    logger.info("RAG LangGraph agent compiled successfully.")
    return app

# === FastAPI Приложение ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка моделей и компиляция графа при старте FastAPI."""

    global device, embedding_model_instance, giga_chat_model, rag_agent_app

    logger.info("FastAPI app starting up...")
    start_time = time.time()
    try:
        # Определяем устройство
        if torch.backends.mps.is_available(): device = "mps"
        elif torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"
        logger.info(f"Using device: {device}")

        # Загрузка Embedding модели через LangChain
        logger.info(f"Loading embedding model via LangChain: {EMBEDDING_MODEL_NAME}...")
        embedding_model_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 128} # Нормализация важна, batch_size для ускорения
        )
        # Можно добавить "прогрев" модели, если первая загрузка медленная
        # await asyncio.to_thread(embedding_model_instance.embed_query, "тест")
        logger.info("Embedding model loaded successfully.")

        # Инициализация GigaChat
        logger.info("Initializing GigaChat model...")
        if not GIGACHAT_API_KEY:
            raise ValueError("GigaChat API Key not found in environment or code.")
        giga_chat_model = GigaChat(
            credentials=GIGACHAT_API_KEY,
            model="GigaChat", # Или GigaChat-Pro, GigaChat-2-Max если доступен
            verify_ssl_certs=False, # Оставить False если есть проблемы с сертификатом
            scope="GIGACHAT_API_PERS" # Или другой нужный scope
        )
        # Тестовый вызов для проверки работоспособности
        await giga_chat_model.ainvoke("Привет!")
        logger.info("GigaChat model initialized successfully.")

        # Компиляция RAG LangGraph агента
        logger.info("Compiling RAG LangGraph agent...")
        rag_agent_app = build_rag_graph()
        logger.info("RAG LangGraph agent compiled.")

        end_time = time.time()
        logger.info(f"--- Application Ready (startup time: {end_time - start_time:.2f}s) ---")
        yield # Приложение работает

    except Exception as e:
        logger.error(f"!!! FATAL ERROR DURING APPLICATION STARTUP: {e}", exc_info=True)
        # Завершаем приложение, если стартап не удался
        # Можно добавить более чистый механизм остановки uvicorn, если нужно
        raise SystemExit(f"Startup failed: {e}")
    finally:
        # --- Очистка ресурсов при завершении работы ---
        logger.info("--- FastAPI app shutting down... ---")
        # Используем переменные, объявленные global в начале функции
        # Безопасное удаление с проверкой существования
        if 'embedding_model_instance' in globals() and embedding_model_instance is not None:
             logger.info("Deleting embedding model instance...")
             del embedding_model_instance
             embedding_model_instance = None # Обнуляем для ясности
             logger.info("Embedding model instance deleted.")
        if 'giga_chat_model' in globals() and giga_chat_model is not None:
             logger.info("Deleting GigaChat model instance...")
             del giga_chat_model
             giga_chat_model = None # Обнуляем
             logger.info("GigaChat model instance deleted.")
        if 'rag_agent_app' in globals() and rag_agent_app is not None:
             logger.info("Deleting RAG agent app...")
             del rag_agent_app
             rag_agent_app = None
             logger.info("RAG agent app deleted.")

        # Очистка CUDA кэша, если использовалась GPU
        if device == 'cuda':
             logger.info("Clearing CUDA cache...")
             torch.cuda.empty_cache()
             logger.info("CUDA cache cleared.")
        logger.info("--- Resources released. ---")


app = FastAPI(title="Matrix AI Assistant API (RAG Only)", lifespan=lifespan)

@app.post("/ask")
async def ask_assistant(
    user_id: str = Form(...),
    question: Optional[str] = Form(None),
    mydtoken: str = Form(...), # Токен для API MatrixCRM
    tenant_id: str = Form(...),
    file: UploadFile = File(None) # Для аудио-ввода
):
    """
    Основной эндпоинт для RAG запросов к ассистенту.
    Принимает текстовый вопрос или аудиофайл, ID пользователя и тенанта, токен MatrixCRM.
    Возвращает ответ ассистента.
    """
    global conversation_history, rag_agent_app # Доступ к глобальным переменным

    # Проверка инициализации ключевых компонентов
    if not rag_agent_app or not giga_chat_model or not embedding_model_instance:
         logger.error("!!! Critical components not initialized. Service unavailable.")
         raise HTTPException(status_code=503, detail="Сервис временно недоступен, идет инициализация или произошла ошибка при старте.")

    request_id = str(uuid.uuid4())[:8] # Короткий ID для логов
    logger.info(f"[Req ID: {request_id}] RAG request received. user_id: {user_id}, tenant_id: {tenant_id}")

    try:
        # --- Очистка старых сессий (простая реализация) ---
        current_time = time.time()
        session_timeout = 1800 # 30 минут
        expired_users = [uid for uid, data in conversation_history.items() if current_time - data.get("last_active", 0) > session_timeout]
        for uid in expired_users:
            if uid in conversation_history: del conversation_history[uid]
            logger.info(f"Removed inactive RAG session for user_id: {uid} (timeout: {session_timeout}s)")

        # --- Обработка аудио (если есть) ---
        recognized_text = None
        if file and file.filename:
             audio_start_time = time.time()
             logger.info(f"[Req ID: {request_id}] Processing audio file: {file.filename} ({file.content_type})")
             # Сохраняем файл временно
             temp_dir = Path("/tmp/audio_files") # Используйте временную директорию
             temp_dir.mkdir(exist_ok=True)
             temp_path = temp_dir / f"{request_id}_{file.filename}"
             try:
                 async with aiofiles.open(temp_path, "wb") as temp_file:
                    # Читаем файл чанками для больших файлов
                    while content := await file.read(1024 * 1024): # По 1MB
                        await temp_file.write(content)
                 logger.info(f"[Req ID: {request_id}] Audio file saved to {temp_path}")

                 # Запускаем распознавание в отдельном потоке (если SDK блокирующий)
                 loop = asyncio.get_event_loop()
                 recognized_text = await loop.run_in_executor(None, lambda: recognize_audio_with_sdk(str(temp_path)))
                 audio_end_time = time.time()
                 if recognized_text:
                      logger.info(f"[Req ID: {request_id}] Recognized text ({audio_end_time - audio_start_time:.2f}s): '{recognized_text[:100]}...'")
                 else:
                      logger.warning(f"[Req ID: {request_id}] Audio recognition returned empty result.")

             except Exception as audio_err:
                 logger.error(f"[Req ID: {request_id}] Error processing audio file {file.filename}: {audio_err}", exc_info=True)
                 # Не прерываем запрос, просто используем текстовый вопрос, если он есть
             finally:
                 # Удаляем временный файл
                 if temp_path.exists():
                     try: os.remove(temp_path)
                     except OSError as e: logger.warning(f"Could not remove temp audio file {temp_path}: {e}")
                 await file.close() # Закрываем файл

        input_text = recognized_text or question
        if not input_text or not input_text.strip():
            logger.error(f"[Req ID: {request_id}] Empty query after processing input.")
            raise HTTPException(status_code=400, detail="Запрос не может быть пустым.")
        logger.info(f"[Req ID: {request_id}] Effective input text: '{input_text[:200]}...'")

        # --- Обновление данных тенанта (если необходимо) ---
        # Эта функция сама проверит актуальность файла
        await update_json_file(mydtoken, tenant_id)

        # --- Подготовка данных для RAG агента ---
        # Получаем историю для LangGraph из нашего формата
        history_for_graph: List[Dict] = []
        if user_id in conversation_history:
             raw_history = conversation_history[user_id].get("history", [])
             # Преобразуем в формат {type: 'human'/'ai', data: {content: '...'}}
             for entry in raw_history[-10:]: # Берем последние N=10 сообщений
                  if "user_query" in entry and entry["user_query"]:
                      history_for_graph.append({"type": "human", "data": {"content": entry["user_query"]}})
                  if "assistant_response" in entry and entry["assistant_response"]:
                      history_for_graph.append({"type": "ai", "data": {"content": entry["assistant_response"]}})

        # Входные данные для графа LangGraph
        graph_input: RagAgentState = {
            "user_query": input_text,
            "chat_history": history_for_graph,
            "tenant_id": tenant_id,
            # Начальные пустые значения для других полей состояния
            "entities": {},
            "search_results": [],
            "answer": "",
            "error_message": None
        }

        # --- Выполнение RAG через LangGraph ---
        logger.info(f"[Req ID: {request_id}] Invoking RAG agent for tenant {tenant_id}...")
        start_rag_time = time.time()
        # Асинхронный вызов скомпилированного графа
        # Передаем graph_input, ожидаем конечное состояние RagAgentState
        result_state: RagAgentState = await rag_agent_app.ainvoke(graph_input)
        end_rag_time = time.time()
        logger.info(f"[Req ID: {request_id}] RAG agent invocation complete ({end_rag_time - start_rag_time:.2f}s).")

        # Получаем финальный ответ и результаты поиска из состояния
        final_answer = result_state.get("answer")
        final_search_results = result_state.get("search_results") # Список метаданных
        error_message = result_state.get("error_message")

        # Обработка возможных ошибок во время выполнения графа
        if error_message:
             logger.error(f"[Req ID: {request_id}] RAG agent finished with error: {error_message}")
             # Ответ пользователю уже должен быть сформирован в узле генерации или раньше
             response_text = final_answer or f"Извините, произошла внутренняя ошибка: {error_message}. Попробуйте позже."
        elif not final_answer:
             logger.error(f"[Req ID: {request_id}] RAG agent finished without error but answer is empty.")
             response_text = "Извините, не удалось сформировать ответ. Пожалуйста, попробуйте переформулировать запрос."
        else:
             response_text = final_answer
             logger.info(f"[Req ID: {request_id}] RAG agent successful. Answer length: {len(response_text)} chars.")

        # --- Сохранение истории диалога (в нашем формате) ---
        if user_id not in conversation_history:
            conversation_history[user_id] = {"history": [], "last_active": time.time()}

        history_entry = {
            "user_query": input_text,
            "assistant_response": response_text,
            "timestamp": time.time(),
            "request_id": request_id # Сохраняем ID запроса для отладки
        }
        # Добавляем краткую сводку RAG результатов для логгирования/отладки
        if final_search_results:
             history_entry["rag_results_summary"] = [
                 {k: r.get(k) for k in ["serviceName", "filialName", "employeeFullName", "price", "serviceId"]}
                 for r in final_search_results[:3] # Первые 3 результата
             ]
        elif not error_message: # Если не было ошибки, но и результатов нет
             history_entry["rag_results_summary"] = "No relevant results found."

        conversation_history[user_id]["history"].append(history_entry)
        conversation_history[user_id]["last_active"] = time.time()

        # Ограничение длины истории (например, последние 20 обменов)
        max_history_len = 20
        if len(conversation_history[user_id]["history"]) > max_history_len:
            conversation_history[user_id]["history"] = conversation_history[user_id]["history"][-max_history_len:]

        # --- Возвращаем ответ пользователю ---
        logger.info(f"[Req ID: {request_id}] Sending response: '{response_text[:200]}...'")
        return JSONResponse(content={"response": response_text})

    # Обработка ожидаемых ошибок FastAPI/HTTP
    except HTTPException as http_exc:
        logger.error(f"[Req ID: {request_id}] HTTP Exception: {http_exc.status_code} - {http_exc.detail}")
        # Перевыбрасываем исключение, FastAPI его обработает
        raise http_exc
    # Обработка всех остальных непредвиденных ошибок
    except Exception as e:
        logger.error(f"[Req ID: {request_id}] Unhandled Exception in /ask endpoint: {e}", exc_info=True)
        # Возвращаем общую ошибку сервера
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера. Пожалуйста, свяжитесь с администратором.")


@app.get("/", include_in_schema=False)
async def root():
    """Корневой эндпоинт для проверки работы API."""
    return {"message": "Matrix AI Assistant API (RAG Only) is running."}

# === Запуск Сервера ===
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001)) # Берем порт из окружения или по умолчанию
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info").lower() # Уровень логирования из окружения

    logger.info(f"Starting Uvicorn server on {host}:{port} with log level '{log_level}'")
    uvicorn.run(
        "matrixai:app", #
        host=host,
        port=port,
        log_level=log_level,
        reload=False 
        
    )
