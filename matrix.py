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
import pickle
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from voicerecognise import recognize_audio_with_sdk
from yandex_cloud_ml_sdk import YCloudML
from typing import Dict, List, Optional
import faiss  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

BASE_DIR = "base"
EMBEDDINGS_DIR = "embeddings_data"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"

YANDEX_FOLDER_ID = "b1gnq2v60fut60hs9vfb"
YANDEX_API_KEY = "AQVNw5Kg0jXoaateYQWdSr2k8cbst_y4_WcbvZrW"

logger.info("Загрузка моделей...")
search_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
logger.info("Модели успешно загружены.")

conversation_history: Dict[str, Dict] = {}

app = FastAPI()


def get_tenant_path(tenant_id: str) -> Path:
    """Создает папку для конкретного тенанта"""
    tenant_path = Path(EMBEDDINGS_DIR) / tenant_id
    tenant_path.mkdir(parents=True, exist_ok=True)
    return tenant_path


def normalize_text(text: str) -> str:
    # Сохраним переводы строк, удаляя лишние символы, кроме \n
    text = text.strip()
    text = re.sub(r"[^\w\s\d\n]", "", text)
    return text.lower()


def tokenize_text(text: str) -> List[str]:
    stopwords = {
        "и", "в", "на", "с", "по", "для", "как", "что", "это", "но",
        "а", "или", "у", "о", "же", "за", "к", "из", "от", "так", "то", "все"
    }
    tokens = text.split()
    return [word for word in tokens if word not in stopwords]


def extract_text_fields(record: dict) -> str:
    """
    Формирует многострочное текстовое представление записи, сохраняющее иерархию:
      Филиал: <filialName>
      Категория: <categoryName>
      Услуга: <serviceName>
      Описание услуги: <serviceDescription>
      Цена: <price>
      Специалист: <employeeFullName>
      Описание специалиста: <employeeDescription>
    """
    filial = record.get("filialName", "Филиал не указан")
    category = record.get("categoryName", "Категория не указана")
    service = record.get("serviceName", "Услуга не указана")
    service_desc = record.get("serviceDescription", "Описание услуги не указано")
    price = record.get("price", "Цена не указана")
    specialist = record.get("employeeFullName", "Специалист не указан")
    spec_desc = record.get("employeeDescription", "Описание не указано")
    text = (
        f"Филиал: {filial}\n"
        f"Категория: {category}\n"
        f"Услуга: {service}\n"
        f"Описание услуги: {service_desc}\n"
        f"Цена: {price}\n"
        f"Специалист: {specialist}\n"
        f"Описание специалиста: {spec_desc}"
    )
    return normalize_text(text)


async def load_json_data(tenant_id: str) -> List[dict]:
    """
    Загружает данные из JSON-файла и преобразует их в список записей.
    """
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл для tenant_id={tenant_id} не найден.")
    
    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
        data = json.loads(content)
    
    records = []
    branches = data.get("data", {}).get("branches", [])
    for branch in branches:
        filial_name = branch.get("name", "Филиал не указан")
        categories = branch.get("categories", [])
        for category in categories:
            category_name = category.get("name", "Категория не указана")
            services = category.get("services", [])
            for service in services:
                service_name = service.get("name", "Услуга не указана")
                price = service.get("price", "Цена не указана")
                service_description = service.get("description", "")
                employees = service.get("employees", [])
                if employees:
                    for emp in employees:
                        employee_full_name = emp.get("full_name", "Специалист не указан")
                        employee_description = emp.get("description", "Описание не указано")
                        record = {
                            "filialName": filial_name,
                            "categoryName": category_name,
                            "serviceName": service_name,
                            "serviceDescription": service_description,
                            "price": price,
                            "employeeFullName": employee_full_name,
                            "employeeDescription": employee_description
                        }
                        records.append(record)
                else:
                    record = {
                        "filialName": filial_name,
                        "categoryName": category_name,
                        "serviceName": service_name,
                        "serviceDescription": service_description,
                        "price": price,
                        "employeeFullName": "Специалист не указан",
                        "employeeDescription": "Описание не указано"
                    }
                    records.append(record)
    return records


async def prepare_data(tenant_id: str):
    """
    Подготавливает данные для тенанта: загружает JSON, строит эмбеддинги, BM25 и FAISS-индекс.
    """
    tenant_path = get_tenant_path(tenant_id)
    data_file = tenant_path / "data.json"
    embeddings_file = tenant_path / "embeddings.npy"
    bm25_file = tenant_path / "bm25.pkl"
    faiss_index_file = tenant_path / "faiss_index.index"
    
    if all([f.exists() for f in [data_file, embeddings_file, bm25_file, faiss_index_file]]):
        file_age = time.time() - os.path.getmtime(data_file)
        if file_age < 2_592_000:
            async with aiofiles.open(data_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
            embeddings = np.load(embeddings_file)
            with open(bm25_file, "rb") as f:
                bm25 = pickle.load(f)
            index = faiss.read_index(str(faiss_index_file))
            return data, embeddings, bm25, index

    records = await load_json_data(tenant_id)
    documents = [extract_text_fields(record) for record in records]

    loop = asyncio.get_event_loop()
    embeddings, bm25 = await asyncio.gather(
        loop.run_in_executor(None, lambda: search_model.encode(documents, convert_to_tensor=True).cpu().numpy()),
        loop.run_in_executor(None, lambda: BM25Okapi([tokenize_text(doc) for doc in documents]))
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(faiss_index_file))

    async with aiofiles.open(data_file, "w", encoding="utf-8") as f:
        await f.write(json.dumps({
            "records": records,
            "raw_texts": documents,
            "timestamp": time.time()
        }, ensure_ascii=False, indent=4))

    np.save(embeddings_file, embeddings)
    with open(bm25_file, "wb") as f:
        pickle.dump(bm25, f)

    return {"records": records, "raw_texts": documents}, embeddings, bm25, index


async def update_json_file(mydtoken: str, tenant_id: str):
    """
    Обновляет данные, запрашивая страницы последовательно, но ограничиваясь первыми 50 страницами.
    """
    tenant_path = get_tenant_path(tenant_id)
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    
    if os.path.exists(file_path):
        file_age = time.time() - os.path.getmtime(file_path)
        if file_age < 2_592_000:
            logger.info(f"Файл {file_path} актуален, пропускаем обновление.")
            return
    
    for f in tenant_path.glob("*"):
        try:
            os.remove(f)
        except Exception as e:
            logger.error(f"Ошибка удаления файла {f}: {e}")
    
    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {mydtoken}"}
            params = {"tenantId": tenant_id, "page": 1}
            all_data = []
            max_pages = 500
            while True:
                if params["page"] > max_pages:
                    logger.info(f"Достигнут лимит {max_pages} страниц, завершаем загрузку.")
                    break
                async with session.get(API_URL, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    branches = data.get("data", {}).get("branches", [])
                    if not branches:
                        logger.info(f"Страница {params['page']} пустая, завершаем загрузку.")
                        break
                    all_data.extend(branches)
                    logger.info(f"Получено {len(branches)} записей с страницы {params['page']}.")
                    params["page"] += 1

            logger.info(f"Общее число полученных филиалов: {len(all_data)}")
            async with aiofiles.open(file_path, "w", encoding="utf-8") as json_file:
                await json_file.write(json.dumps(
                    {"code": data.get("code", 200), "data": {"branches": all_data}},
                    ensure_ascii=False,
                    indent=4
                ))
    except Exception as e:
        logger.error(f"Ошибка при обновлении файла: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка обновления данных.")


async def rerank_with_cross_encoder(query: str, candidates: List[int], raw_texts: List[str]) -> List[int]:
    """Реранкинг топ-10 кандидатов с использованием кросс-энкодера"""
    cross_inp = [(query, raw_texts[idx]) for idx in candidates]
    loop = asyncio.get_event_loop()
    cross_scores = await loop.run_in_executor(None, lambda: cross_encoder.predict(cross_inp))
    sorted_indices = np.argsort(cross_scores)[::-1].tolist()
    return [candidates[i] for i in sorted_indices]



free_times_function = {
    "name": "getFreeTimesOfEmployeeByChoosenServices",
    "description": "Получить свободное время сотрудника по выбранным услугам",
    "parameters": {
        "type": "object",
        "properties": {
            "employeeId": {"type": "string", "description": "ID сотрудника"},
            "serviceId": {"type": "array", "items": {"type": "string"}, "description": "Список ID услуг"},
            "dateTime": {"type": "string", "description": "Дата в формате YYYY-MM-DD"},
            "tenantId": {"type": "string", "description": "ID тенанта"},
            "filialId": {"type": "string", "description": "ID филиала"},
            "langId": {"type": "string", "description": "Язык (например, ru)"}
        },
        "required": ["employeeId", "serviceId", "dateTime", "tenantId", "filialId", "langId"]
    }
}


async def generate_yandexgpt_response(context: str, history: List[dict], question: str) -> str:
    """
    Ассистент косметологической клиники. Генерирует ответ с использованием YandexGPT.
    """
    system_prompt = """# 🔹 Системный промпт для модели 🔹

Ты – ассистент клиники MED YU MED по имени Аида. Твоя задача – помогать пользователям находить информацию о специалистах, услугах, филиалах и ценах. Ты работаешь как RAG-модель (Retrieval-Augmented Generation), что означает, что:

1. Весь контекст уже загружен – в нём есть подробные данные о специалистах, услугах, ценах и филиалах. Эта информация всегда доступна для поиска в контексте.

2. Тебе не нужно выдумывать информацию – если данных нет в контексте, прямо сообщай об этом. 

3. Если пользователь уточняет запрос, ты обязана искать ответ именно в загруженном контексте, а не только в истории диалога. 

## 📌 1. Анализ запроса

- Прочитай контекст и пойми, к чему относится запрос: услуги, цены, специалисты, филиалы и т. д.
- Если запрос неясен – задай уточняющий вопрос вместо того, чтобы догадываться.

## 🔍 2. Подбор услуг

Если пользователь спрашивает про услуги, ты обязана:
- Найти все подходящие услуги.
- Указать цену (например, "от 12000 рублей 💸").
- Назвать филиал, где доступна услуга (Москва – Ходынка, Москва – Сити, Дубай).
- Перечислить всех специалистов, которые выполняют эту услугу (без слов "и другие", только полные списки!).
- Объяснить пользу услуги в 1-2 предложениях (например: "Эта процедура поможет убрать морщины и сделать кожу более упругой ✨").

## 📍 3. Если ничего не найдено

- Честно скажи, что данных нет, и предложи уточнить запрос.
- Пример: "Пока не нашла информацию, но могу помочь, если уточнишь детали 😊".

## 🎯 4. Завершение диалога

- Если уместно, предложи записаться ("Хочешь записаться на удобное время? 🗓").
- Заканчивай дружелюбно, но не перегружай ответ эмодзи.

## 💬 5. Как вести диалог

- Пиши живо и дружелюбно, избегая канцеляризмов.
- Используй эмодзи умеренно, по смыслу (например, "💸" для цен, "🗓" для записи).
- Учитывай историю диалога – если пользователь уточняет детали, ты должна использовать свои предыдущие ответы.
- Проявляй эмпатию: если человек делится проблемой, покажи, что понимаешь его ситуацию.
- Пиши все цены на услуги с предлогом от

## 🚨 6. Особые указания

- В клинике три филиала: Москва (Ходынка, Сити), Дубай (Bluewaters).
- Держи весь контекст в памяти, так как пользователь может задавать много уточняющих вопросов.
- Если услуга/специалист относится к конкретному филиалу, обязательно указывай это.
- Не пиши "Добрый день" повторно в рамках одного диалога.
- Если пользователь спрашивает про специалиста, обязательно ищи информацию в контексте, даже если ранее давал краткий ответ.

## ⚠️ 7. Главное правило

Ты – RAG-модель, и всё, что тебе нужно, уже есть в загруженном контексте. Если информации нет, не выдумывай – лучше спроси уточняющий вопрос. Не предлагай услуги, если в вопросе это явно не указано.

## 💡 8. Обработка уточняющих вопросов

- Если пользователь уточняет информацию, постарайся использовать старые данные, которые уже были получены.
- Всегда проверяй контекст, прежде чем отвечать. Убедись, что ты не повторяешь информацию, а используешь данные, которые уже были предоставлены ранее.
- При поиске уточняющих запросов используй как старые, так и новые данные.

## 📝 Пример

Контекст:
{context}

Вопрос пользователя:
{question}

Ты – ассистент клиники. Твоя задача – предоставить точную информацию из контекста. Понимай, что могут быть уточняющие вопросы, и отвечай исходя из того, что уже было сказано. Пожалуйста, учитывай все данные из контекста.
"""
    messages = [
        {"role": "system", "text": system_prompt},
        {"role": "system", "text": f"Вот список 10 услуг:\n{context}\n"}
    ]
    
    for entry in history[-10:]:
        messages.append({"role": "user", "text": entry['user_query']})
        messages.append({"role": "assistant", "text": entry['assistant_response']})
    messages.append({"role": "user", "text": question})
    
    try:
        loop = asyncio.get_event_loop()
        sdk = YCloudML(folder_id=YANDEX_FOLDER_ID, auth=YANDEX_API_KEY)
        model_uri = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-32k/rc"
        result = await loop.run_in_executor(
            None,
            lambda: sdk.models.completions(model_uri)
                .configure(temperature=0.55, max_tokens=1000)
                .run(messages)
        )
        if result and result.alternatives:
            return result.alternatives[0].text.strip()
        return "Извините, мне не удалось сгенерировать ответ."
    except Exception as e:
        logger.error(f"Ошибка YandexGPT API: {str(e)}")
        return "Извините, произошла ошибка при генерации ответа."


@app.post("/ask")
async def ask_assistant(
    user_id: str = Form(...),
    question: Optional[str] = Form(None),
    mydtoken: str = Form(...),
    tenant_id: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        current_time = time.time()
        expired_users = [uid for uid, data in conversation_history.items() if current_time - data["last_active"] > 22296]
        for uid in expired_users:
            del conversation_history[uid]

        recognized_text = None
        if file and file.filename:
            temp_path = f"/tmp/{file.filename}"
            try:
                async with aiofiles.open(temp_path, "wb") as temp_file:
                    await temp_file.write(await file.read())
                loop = asyncio.get_event_loop()
                recognized_text = await loop.run_in_executor(None, lambda: recognize_audio_with_sdk(temp_path))
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            if not recognized_text:
                raise HTTPException(status_code=500, detail="Ошибка распознавания речи из файла.")

        input_text = recognized_text or question
        if not input_text:
            raise HTTPException(status_code=400, detail="Необходимо передать текст или файл.")

        force_update = False 
        if force_update or not (get_tenant_path(tenant_id) / "data.json").exists():
            await update_json_file(mydtoken, tenant_id)
        
        data_dict, embeddings, bm25, faiss_index = await prepare_data(tenant_id)
        normalized_question = normalize_text(input_text)
        tokenized_query = tokenize_text(normalized_question)
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:50].tolist()
        
        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: search_model.encode(normalized_question, convert_to_tensor=True).cpu().numpy()
        )
        D, I = faiss_index.search(query_embedding.reshape(1, -1), 50)
        DISTANCE_THRESHOLD = 1.0
        filtered_faiss = [idx for idx, dist in zip(I[0].tolist(), D[0].tolist()) if dist < DISTANCE_THRESHOLD]
        if not filtered_faiss:
            filtered_faiss = I[0].tolist()
        top_faiss_indices = filtered_faiss
        
        combined_indices = list(set(top_bm25_indices + top_faiss_indices))[:50]
        top_10_indices = await rerank_with_cross_encoder(
            query=normalized_question,
            candidates=combined_indices[:30],
            raw_texts=data_dict["raw_texts"]
        )
        
        context = "\n\n".join([
    f"**Документ {i+1}:**\n" 
    f"* Филиал: {data_dict['records'][idx].get('filialName', 'Не указан')}\n"
    f"* Категория: {data_dict['records'][idx].get('categoryName', 'Не указана')}\n"
    f"* Услуга: {data_dict['records'][idx].get('serviceName', 'Не указана')}\n"
    f"* Цена: {data_dict['records'][idx].get('price', 'Цена не указана')} руб.\n"
    f"* Специалист: {data_dict['records'][idx].get('employeeFullName', 'Не указан')}\n"
    f"* Описание: {data_dict['records'][idx].get('employeeDescription', 'Описание не указано')}"  
    for i, idx in enumerate(top_10_indices[:5])
])
        
        if user_id not in conversation_history:
            conversation_history[user_id] = {"history": [], "last_active": time.time(), "greeted": False}

        conversation_history[user_id]["last_active"] = time.time()
        response_text = await generate_yandexgpt_response(context, conversation_history[user_id]["history"], input_text)
        conversation_history[user_id]["history"].append({
            "user_query": input_text,
            "assistant_response": response_text,
            "search_results": [data_dict['records'][idx] for idx in top_10_indices]
        })

        return {"response": response_text}
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")


if __name__ == "__main__":
    logger.info("Запуск сервера на порту 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
