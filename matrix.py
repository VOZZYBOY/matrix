from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
from voicerecognise import recognize_audio_with_sdk
from openai import OpenAI
import json
import uvicorn
import logging
import time
import os
import numpy as np
import re
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

BASE_DIR = "base"
os.makedirs(BASE_DIR, exist_ok=True)
API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"
OPENAI_API_KEY = "sk-proj-Kxlkan6vQKnVFr1oPlKZT3BlbkFJ0MYPTz6RHXl7NRnShBIB"

client = OpenAI(api_key=OPENAI_API_KEY)

logger.info("Загрузка модели векторного поиска...")
search_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
logger.info("Модель успешно загружена.")

app = FastAPI()
conversation_history = {}
data_cache = {}
embeddings_cache = {}
bm25_cache = {}

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\d]", "", text)
    return text

def tokenize_text(text: str) -> list[str]:
    stopwords = {"и", "в", "на", "с", "по", "для", "как", "что", "это", "но", "а", "или", "у", "о", "же", "за", "к", "из", "от", "так", "то", "все"}
    tokens = text.split()
    return [word for word in tokens if word not in stopwords]

def load_json_data(tenant_id: str) -> list:
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл с tenant_id={tenant_id} не найден.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Данные для tenant_id={tenant_id} загружены.")
    return data.get("data", {}).get("items", [])

def extract_text_fields(record: dict) -> str:
    excluded_keys = {"id", "categoryId", "currencyId", "langId", "employeeId", "employeeDescription"}
    raw_text = " ".join(
        str(value) for key, value in record.items()
        if key not in excluded_keys and value is not None and value != ""
    )
    return normalize_text(raw_text)

def prepare_data(tenant_id: str):
    records = load_json_data(tenant_id)
    documents = [extract_text_fields(record) for record in records]
    

    embeddings = search_model.encode(documents, convert_to_tensor=True)
    
    tokenized_corpus = [tokenize_text(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    data_cache[tenant_id] = records
    embeddings_cache[tenant_id] = embeddings
    bm25_cache[tenant_id] = bm25

def update_json_file(mydtoken: str, tenant_id: str):
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
        
        prepare_data(tenant_id)

    except requests.RequestException as e:
        logger.error(f"Ошибка при запросе данных из API: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка обновления JSON файла.")

def generate_openai_response(context: str, history: list, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": """""
    Ты — виртуальный ассистент Аида, который помогает клиентам с выбором услуг. Твоя задача — предоставлять информацию о услугах, филиалах, специалистах и ценах на основе данных из гибридного поиска. Ты должен общаться с клиентами дружелюбно, профессионально и тепло, соблюдая следующие правила:

    1. **Использование контекста из гибридного поиска**:
    - Используй данные из результатов гибридного поиска (услуги, филиалы, специалисты, цены) для формирования ответа.
    - Если клиент спрашивает об услуге, предоставь информацию из результатов поиска: название услуги, цена, филиал, специалист.
    - Если данных нет, вежливо сообщи об этом и предложи уточнить запрос.

    2. **Тон и стиль общения**:
    - Общайся с клиентом так, как общался бы человек: дружелюбно, профессионально и тепло.
    - Избегай повторного приветствия. Здоровайся только один раз при первом сообщении или если диалог был прерван и возобновлён через длительное время.
    - Поддерживай разговор, чтобы клиент чувствовал внимание и заботу.

    3. **Персонализация и адаптивность**:
    - Если клиент называет своё имя, обращайся к нему по имени.
    - При необходимости уточняй детали, чтобы предложить максимально подходящее решение.
    - Если клиент упоминает свои пожелания, используй их в ответах.

    4. **Динамика общения**:
    - Структура ответа должна быть такой, чтобы чувствовалось живое общение.
    - Если запрос неясен, уточняй: "Извините, я не совсем поняла ваш вопрос. Могу ли я уточнить, что именно вы хотите узнать?"
    - Если запрос не связан с услугами, сообщай об этом прямо: "К сожалению, я могу помочь только с информацией о наших услугах."

    5. **Этика и ограничения**:
    - Если клиент задаёт вопрос, который выходит за рамки предоставления услуг, корректно возвращай его к обсуждению услуг.
    - Не скрывай, что ты виртуальный ассистент, если об этом спрашивают.
    - Не придумывай информацию, которой нет в контексте.

    6. **Цель**:
    - Сделать взаимодействие максимально комфортным, полезным и приятным.
    - Старайся, чтобы клиент ощущал, что он важен, и его запрос решается с полной отдачей.

    7. **Примеры ответов**:
    - Если запрос понятен и есть результаты поиска:
        - "Услуга 'Удаление фибром' доступна в филиале Сити 38 за 3000 руб. Специалист — Иванова Мария Сергеевна. Хотите записаться?"
    - Если запрос неясен:
        - "Извините, я не совсем поняла ваш вопрос. Могу ли я уточнить, что именно вы хотите узнать?"
    - Если запрос не связан с услугами:
        - "К сожалению, я могу помочь только с информацией о наших услугах. Что именно вас интересует?"
    - Если информации нет:
        - "Извините, у меня сейчас нет информации по этой услуге. Но могу помочь с другими запросами. Что именно вас интересует?"

    8. **Дополнительные указания**:
    - Не здоровайся с клиентом в каждом сообщении. Здоровайся только один раз при первом сообщении или если диалог был прерван и возобновлён через длительное время.
    - Анализируй настроение клиента и адаптируй тон общения.
    - Избегай повторений и шаблонных фраз, чтобы общение не выглядело искусственным.
    - Отвечай только на основе данных, предоставленных в контексте.
    - Если клиент спрашивает о чём-то, чего нет в контексте, отвечай: "Извините, у меня нет информации об этой услуге."
    """
        },
        {"role": "system", "content": context}
    ]
    

    for entry in history[-3:]:
        messages.append({"role": "user", "content": entry['user_query']})
        messages.append({"role": "assistant", "content": entry['assistant_response']})
    
    messages.append({"role": "user", "content": question})
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Ошибка OpenAI API: {str(e)}")
        return "Извините, произошла ошибка обработки запроса"

@app.post("/ask")
async def ask_assistant(
    user_id: str = Form(...),
    question: str = Form(None),
    mydtoken: str = Form(...),
    tenant_id: str = Form(...),
    file: UploadFile = File(None)
):
    try:
        # Очистка старых сессий (30 минут неактивности)
        current_time = time.time()
        expired_users = [uid for uid, data in conversation_history.items() 
                        if current_time - data["last_active"] > 1800]
        for uid in expired_users:
            del conversation_history[uid]

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
        tokenized_query = tokenize_text(normalized_question)

        # Поиск по BM25
        bm25_scores = bm25_cache[tenant_id].get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:10].tolist()

        # Векторный поиск
        query_embedding = search_model.encode(normalized_question, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, embeddings_cache[tenant_id])
        top_vector_indices = similarities[0].topk(10).indices.tolist()

        # Объединение результатов
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

        # Инициализация или обновление истории
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

        
        response_text = generate_openai_response(
            context=context,
            history=conversation_history[user_id]["history"],
            question=input_text
        )

        conversation_history[user_id]["history"].append({
            "user_query": input_text,
            "assistant_response": response_text,
            "search_results": search_results
        })

        
        logger.info(f"Контекст: {context}")
        logger.info(f"Ответ ассистента: {response_text}")

        return {"response": response_text}

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")

if __name__ == "__main__":
    logger.info("Запуск сервера на порту 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
