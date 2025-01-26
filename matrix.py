import asyncio
import aiohttp
import aiofiles
import json
import uvicorn
import logging
import time
import os
import numpy as np
import re
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from voicerecognise import recognize_audio_with_sdk
from yandex_cloud_ml_sdk import YCloudML
from cachetools import TTLCache
from typing import Dict, List, Optional


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


BASE_DIR = "base"
os.makedirs(BASE_DIR, exist_ok=True)
API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"
YANDEX_FOLDER_ID = "b1gnq2v60fut60hs9vfb"
YANDEX_API_KEY = "AQVNw5Kg0jXoaateYQWdSr2k8cbst_y4_WcbvZrW"


logger.info("Загрузка модели векторного поиска...")
search_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
logger.info("Модель успешно загружена.")


data_cache = TTLCache(maxsize=100, ttl=1800)
embeddings_cache = TTLCache(maxsize=100, ttl=1800)
bm25_cache = TTLCache(maxsize=100, ttl=1800)
conversation_history: Dict[str, Dict] = {}


cache_locks = {
    "data": asyncio.Lock(),
    "embeddings": asyncio.Lock(),
    "bm25": asyncio.Lock()
}

app = FastAPI()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\d]", "", text)
    return text

def tokenize_text(text: str) -> List[str]:
    stopwords = {"и", "в", "на", "с", "по", "для", "как", "что", "это", "но", "а", "или", "у", "о", "же", "за", "к", "из", "от", "так", "то", "все"}
    tokens = text.split()
    return [word for word in tokens if word not in stopwords]

def extract_text_fields(record: dict) -> str:
    excluded_keys = {"id", "categoryId", "currencyId", "langId", "employeeId", "employeeDescription"}
    raw_text = " ".join(
        str(value) for key, value in record.items()
        if key not in excluded_keys and value is not None and value != ""
    )
    return normalize_text(raw_text)


async def load_json_data(tenant_id: str) -> List[dict]:
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл с tenant_id={tenant_id} не найден.")

    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
        content = await f.read()
        return json.loads(content).get("data", {}).get("items", [])

async def prepare_data(tenant_id: str):
    async with cache_locks["data"]:
        if tenant_id in data_cache:
            return

        records = await load_json_data(tenant_id)
        documents = [extract_text_fields(record) for record in records]
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, 
            lambda: search_model.encode(documents, convert_to_tensor=True)
        )
        
        tokenized_corpus = [tokenize_text(doc) for doc in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        async with cache_locks["embeddings"], cache_locks["bm25"]:
            data_cache[tenant_id] = records
            embeddings_cache[tenant_id] = embeddings
            bm25_cache[tenant_id] = bm25

async def update_json_file(mydtoken: str, tenant_id: str):
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    
    if os.path.exists(file_path):
        logger.info(f"Файл {file_path} уже существует. Используем данные из файла.")
        await prepare_data(tenant_id)
        return

    try:
        async with aiohttp.ClientSession() as session:
            headers = {"Authorization": f"Bearer {mydtoken}"}
            params = {"tenantId": tenant_id, "page": 1}
            all_data = []

            while True:
                async with session.get(API_URL, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()
                    items = data.get("data", {}).get("items", [])

                    if not items:
                        break

                    all_data.extend(items)
                    logger.info(f"Получено {len(items)} записей с страницы {params['page']}.")
                    params["page"] += 1

            async with aiofiles.open(file_path, "w", encoding="utf-8") as json_file:
                await json_file.write(json.dumps(
                    {"data": {"items": all_data}}, 
                    ensure_ascii=False, 
                    indent=4
                ))

        await prepare_data(tenant_id)

    except Exception as e:
        logger.error(f"Ошибка при запросе данных из API: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка обновления JSON файла.")


async def generate_yandexgpt_response(context: str, history: List[dict], question: str) -> str:
    messages = [
        {
            "role": "system",
            "text": """Ты — виртуальный ассистент Аида. Используй контекст для ответа. Если информации нет — сообщи об этом."""
        },
        {"role": "system", "text": context}
    ]
    
    for entry in history[-5:]:
        messages.append({"role": "user", "text": entry['user_query']})
        messages.append({"role": "assistant", "text": entry['assistant_response']})
    
    messages.append({"role": "user", "text": question})

    try:
        loop = asyncio.get_event_loop()
        sdk = YCloudML(
            folder_id=YANDEX_FOLDER_ID,
            auth=YANDEX_API_KEY
        )
        model_uri = f"gpt://{YANDEX_FOLDER_ID}/yandexgpt-32k/rc"
        
        result = await loop.run_in_executor(
            None,
            lambda: sdk.models.completions(model_uri)
                .configure(temperature=0.7, max_tokens=2000)
                .run(messages)
        )

        return result.alternatives[0].text if result and result.alternatives else "Извините, не удалось сгенерировать ответ"

    except Exception as e:
        logger.error(f"Ошибка YandexGPT API: {str(e)}")
        return "Извините, произошла ошибка обработки запроса"


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
        expired_users = [uid for uid, data in conversation_history.items() 
                        if current_time - data["last_active"] > 1800]
        for uid in expired_users:
            del conversation_history[uid]

        recognized_text = None

        
        if file and file.filename:
            temp_path = f"/tmp/{file.filename}"
            try:
                async with aiofiles.open(temp_path, "wb") as temp_file:
                    await temp_file.write(await file.read())
                
                loop = asyncio.get_event_loop()
                recognized_text = await loop.run_in_executor(
                    None, 
                    lambda: recognize_audio_with_sdk(temp_path)
                )
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            if not recognized_text:
                raise HTTPException(status_code=500, detail="Ошибка распознавания речи из файла.")

        input_text = recognized_text or question
        if not input_text:
            raise HTTPException(status_code=400, detail="Необходимо передать текст или файл.")

 
        if tenant_id not in data_cache:
            await update_json_file(mydtoken, tenant_id)

        await prepare_data(tenant_id)

       
        normalized_question = normalize_text(input_text)
        tokenized_query = tokenize_text(normalized_question)

        bm25_scores = bm25_cache[tenant_id].get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:10].tolist()

        loop = asyncio.get_event_loop()
        query_embedding = await loop.run_in_executor(
            None,
            lambda: search_model.encode(normalized_question, convert_to_tensor=True)
        )
        similarities = util.pytorch_cos_sim(query_embedding, embeddings_cache[tenant_id])
        top_vector_indices = similarities[0].topk(10).indices.tolist()

        combined_indices = list(set(top_bm25_indices + top_vector_indices))[:10]

        search_results = [
            {
                "text": data_cache[tenant_id][idx].get("serviceName", "Не указано"),
                "price": data_cache[tenant_id][idx].get("price", "Цена не указана"),
                "filial": data_cache[tenant_id][idx].get("filialName", "Филиал не указан"),
                "specialist": data_cache[tenant_id][idx].get("employeeFullName", "Специалист не указан")
            }
            for idx in combined_indices
        ]

        context = "\n".join([
            f"Услуга: {res['text']}\nЦена: {res['price']} руб.\nФилиал: {res['filial']}\nСпециалист: {res['specialist']}"
            for res in search_results
        ])

      
        if user_id not in conversation_history:
            conversation_history[user_id] = {
                "history": [],
                "last_active": time.time(),
                "greeted": False
            }
    
        if not conversation_history[user_id]["greeted"]:
            context = "Здравствуйте! Чем могу помочь?\n" + context
            conversation_history[user_id]["greeted"] = True

        conversation_history[user_id]["last_active"] = time.time()

        
        response_text = await generate_yandexgpt_response(
            context=context,
            history=conversation_history[user_id]["history"],
            question=input_text
        )

      
        conversation_history[user_id]["history"].append({
            "user_query": input_text,
            "assistant_response": response_text,
            "search_results": search_results
        })

        return {"response": response_text}

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

if __name__ == "__main__":
    logger.info("Запуск сервера на порту 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
