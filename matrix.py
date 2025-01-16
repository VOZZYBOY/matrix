from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from voicerecognise import recognize_audio_with_sdk
from yandex_cloud_ml_sdk import YCloudML
import json
import uvicorn
import logging
import time
import os
import threading
import numpy as np
import re
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = "base"
os.makedirs(BASE_DIR, exist_ok=True)
API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"

logger.info("Загрузка модели векторного поиска...")
search_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
logger.info("Модель успешно загружена.")

FOLDER_ID = "b1gnq2v60fut60hs9vfb"
API_KEY = "AQVNw5Kg0jXoaateYQWdSr2k8cbst_y4_WcbvZrW"

sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)
instruction = """
### Основные принципы работы:
1. **Использование контекста**:
   - Когда передаётся информация о контексте, она включает:
     - Услуга
     - Цена
     - Филиал
     - Специалист
   - Если клиент уточняет цену, филиал или специалиста, используй предоставленные данные.
   - Если какой-то информации не хватает, вежливо сообщи об этом.
   Твой ответ должен включать всю релевантную информацию из контекста, чтобы помочь пользователю.
    Например:
    - Услуга: Удаление фибром
    - Цена: 3000 руб.
    - Филиал: Сити 38
    - Специалист: Иванова Мария Сергеевна

2. **Тон и стиль общения**:
   - Общайся с пользователем так, как общался бы человек: дружелюбно, профессионально и тепло.
   - Избегай повторного приветствия, если оно уже было отправлено в рамках текущего диалога.
   - Поддерживай разговор, чтобы клиент чувствовал внимание и заботу. Например:
     - "Я рада, что вы обратились! Давайте я помогу."
     - "Замечательный выбор услуги, могу подсказать дополнительную информацию!"
   - Учитывай эмоциональное состояние клиента и выражай эмпатию.

3. **Персонализация и адаптивность**:
   - Если клиент называет своё имя, обращайся к нему по имени.
   - При необходимости уточняй детали, чтобы предложить максимально подходящее решение.
   - Если клиент упоминает свои пожелания, используй их в ответах.

4. **Динамика общения**:
   - Структура ответа должна быть такой, чтобы чувствовалось живое общение.
     Например:
     - "Хороший выбор! Услуга 'Удаление фибром' доступна в филиале Сити 38 за 3000 руб. Хотите узнать, когда свободен специалист?"
     - "Увы, я сейчас не вижу данных по этой услуге. Возможно, вас интересует что-то другое? Давайте уточним!"
   - Если клиент задаёт запрос, который не имеет отношения к услугам, сообщай об этом прямо: 
     - "К сожалению, я могу помочь только с информацией о наших услугах. Если вам нужно что-то конкретное, уточните запрос."

5. **Этика и ограничения**:
   - Если клиент задаёт вопрос, который выходит за рамки предоставления услуг, корректно возвращай его к обсуждению услуг:
     - "Извините, я могу помочь только с вопросами о наших услугах, филиалах или специалистах."
   - Не скрывай, что ты виртуальный ассистент, если об этом спрашивают.
   - Не предлагай услуги, если клиент задал непонятный или общий вопрос. Вместо этого уточняй:
     - "Извините, я не совсем поняла ваш запрос. Могу ли я уточнить, что именно вы хотите узнать?"

6. **Цель**:
   - Сделать взаимодействие максимально комфортным, полезным и приятным.
   - Старайся, чтобы клиент ощущал, что он важен, и его запрос решается с полной отдачей.

7. **Примеры**:
   - Если запрос понятен: 
     - "Удаление фибром доступно в филиале Сити 38. Стоимость услуги — 3000 руб. Хотите записаться?"
   - Если запрос неясен:
     - "Извините, я не совсем поняла ваш вопрос. Могу ли я уточнить, что именно вы хотите узнать?"
   - Если запрос не связан с услугами:
     - "К сожалению, я могу помочь только с информацией о наших услугах. Что именно вас интересует?"
   - Если информации нет:
     - "Извините, у меня сейчас нет информации по этой услуге. Но могу помочь с другими запросами. Что именно вас интересует?"
8.Перестнаь здороваться с пользоваталем черещ каждое сообшение,здоровайся только тогда,когда thread создается или когда информация о внутренности thread устарела
ЗДОРОВАЙСЯ С КЛИЕНТОМ ТОЛЬКО ОДИН РАЗ ПРИ ПЕРВОМ СООБЩЕНИИ

"""

assistant = sdk.assistants.create(
    model=sdk.models.completions("yandexgpt", model_version="rc"),
    ttl_days=365,
    expiration_policy="since_last_active",
    max_tokens=1000,
    instruction=instruction
)

logger.info("Ассистент успешно инициализирован.")

app = FastAPI()
threads = {}
data_cache = {}
embeddings_cache = {}
bm25_cache = {}

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text

def load_json_data(tenant_id):
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл с tenant_id={tenant_id} не найден.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Данные для tenant_id={tenant_id} загружены.")
    return data.get("data", {}).get("items", [])

def extract_text_fields(record):
    excluded_keys = {"id", "categoryId", "currencyId", "langId", "employeeId", "employeeDescription"}
    raw_text = " ".join(
        str(value) for key, value in record.items()
        if key not in excluded_keys and value is not None
    )
    return normalize_text(raw_text)

def prepare_data(tenant_id):
    records = load_json_data(tenant_id)
    documents = [extract_text_fields(record) for record in records]

    tokenized_corpus = [doc.split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    embeddings = search_model.encode(documents, convert_to_tensor=True)

    data_cache[tenant_id] = records
    embeddings_cache[tenant_id] = embeddings
    bm25_cache[tenant_id] = bm25

def update_json_file(mydtoken, tenant_id):
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    headers = {"Authorization": f"Bearer {mydtoken}"}
    params = {"tenantId": tenant_id, "page": 1}
    all_data = []

    if os.path.exists(file_path):
        logger.info(f"Файл {file_path} уже существует. Используем данные из файла.")
        prepare_data(tenant_id)
        return

    try:
        logger.info(f"Запрос данных с tenant_id={tenant_id} с пагинацией.")
        while True:
            response = requests.get(API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            items = data.get("data", {}).get("items", [])

            if not items:
                break

            all_data.extend(items)
            logger.info(f"Получено {len(items)} записей с страницы {params['page']}.")

            params["page"] += 1

        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump({"data": {"items": all_data}}, json_file, ensure_ascii=False, indent=4)
        logger.info(f"JSON файл для tenant_id={tenant_id} успешно обновлен, всего записей: {len(all_data)}.")
        prepare_data(tenant_id)

    except requests.RequestException as e:
        logger.error(f"Ошибка при запросе данных из API: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка обновления JSON файла.")

@app.post("/ask")
async def ask_assistant(
    user_id: str = Form(...),
    question: str = Form(None),
    mydtoken: str = Form(...),
    tenant_id: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        recognized_text = None

        if file:
            temp_path = f"/tmp/{file.filename}"
            try:
                with open(temp_path, "wb") as temp_file:
                    temp_file.write(await file.read())
                recognized_text = recognize_audio_with_sdk(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            if not recognized_text:
                raise HTTPException(status_code=500, detail="Ошибка распознавания речи из файла.")

        input_text = recognized_text if recognized_text else question
        if not input_text:
            raise HTTPException(status_code=400, detail="Необходимо передать текст или файл.")

        if tenant_id not in data_cache:
            update_json_file(mydtoken, tenant_id)

        normalized_question = normalize_text(input_text)

        query_embedding = search_model.encode(normalized_question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, embeddings_cache[tenant_id])
        similarities = similarities[0]
        top_vector_indices = similarities.topk(10).indices.tolist()

        bm25 = bm25_cache[tenant_id]
        tokenized_query = normalized_question.split()
        bm25_scores = bm25.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:10]

        hybrid_scores = {}
        for idx in top_vector_indices:
            hybrid_scores[idx] = similarities[idx].item() * 1.3
        for idx in top_bm25_indices:
            hybrid_scores[idx] = hybrid_scores.get(idx, 0) + bm25_scores[idx]

        sorted_indices = sorted(hybrid_scores, key=hybrid_scores.get, reverse=True)[:10]

        search_results = [
            {
                "text": data_cache[tenant_id][idx].get("serviceName", "Не указано"),
                "price": data_cache[tenant_id][idx].get("price", "Цена не указана"),
                "filial": data_cache[tenant_id][idx].get("filialName", "Филиал не указан"),
                "specialist": data_cache[tenant_id][idx].get("employeeFullName", "Специалист не указан")
            }
            for idx in sorted_indices
        ]

        context = "\n".join([f"Услуга: {res['text']}\nЦена: {res['price']} руб.\nФилиал: {res['filial']}\nСпециалист: {res['specialist']}" for res in search_results])

        if user_id not in threads:
            threads[user_id] = {
                "thread": sdk.threads.create(
                    name=f"Thread-{user_id}",
                    ttl_days=5,
                    expiration_policy="since_last_active"
                ),
                "last_active": time.time(),
                "context": ""
            }

        threads[user_id]["last_active"] = time.time()
        thread = threads[user_id]["thread"]

        new_context = f"\nКонтекст:\n{context}\nПользователь спрашивает: {input_text}"
        if len(threads[user_id]["context"]) + len(new_context) > 29000:
            threads[user_id]["context"] = threads[user_id]["context"][-20000:]
        threads[user_id]["context"] += new_context

        thread.write(threads[user_id]["context"])

        run = assistant.run(thread)
        result = run.wait()

        threads[user_id]["context"] += f"\nОтвет ассистента: {result.text}"

        logger.info(f"Контекст: {threads[user_id]['context']}")
        logger.info(f"Ответ ассистента: {result.text}")
        return {
            "response": result.text
        }

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

if __name__ == "__main__":
    logger.info("Запуск сервера на порту 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
