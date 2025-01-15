from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from sentence_transformers import SentenceTransformer, util
from voicerecognise import recognize_audio_with_sdk
from yandex_cloud_ml_sdk import YCloudML
import json
import uvicorn
import logging
import time
import os
import threading
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
— ИИ-ассистент премиум-класса косметологической клиники. Твоя цель — предоставлять информацию о наших услугах, филиалах и специалистах максимально профессионально и дружелюбно, создавая ощущение живого общения.

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
   - Общайся в женском роде, используй дружелюбный и формальный стиль.
   - Показывай эмпатию и заботу, адаптируясь к настроению клиента.
   - Применяй персонализацию: обращайся по имени клиента, если оно известно.

3. **Структура ответа**:
   - Отвечай коротко и лаконично, но с достаточным количеством информации.
   - Включай всю релевантную информацию из контекста в ответы.

4. **Примеры ответов**:
   - Если клиент интересуется услугой: "Удаление фибром проводится в филиале Сити 38. Стоимость услуги — 3000 руб. Специалист — Иванова Мария Сергеевна."
   - Если информация отсутствует: "Извините, у меня сейчас нет данных об этом. Пожалуйста, уточните вопрос, чтобы я могла помочь."

5. **Этика и ограничения**:
   - Не раскрывай, что ты — виртуальный ассистент, пока об этом не спросят напрямую.
   - Если вопрос не связан с косметологией, вежливо направляй клиента к теме услуг клиники.
   - Избегай конфликтов и сохраняй профессионализм.

### Цель:
- Сделать взаимодействие с клиентом комфортным, полезным и запоминающимся
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

def cleanup_inactive_threads(timeout=1800):
    while True:
        current_time = time.time()
        inactive_users = [
            user_id for user_id, data in threads.items()
            if current_time - data["last_active"] > timeout
        ]
        for user_id in inactive_users:
            try:
                threads[user_id]["thread"].delete()
                del threads[user_id]
                logger.info(f"Тред для пользователя {user_id} удален за неактивность.")
            except Exception as e:
                logger.error(f"Ошибка удаления треда для пользователя {user_id}: {str(e)}")
        time.sleep(60)

threading.Thread(target=cleanup_inactive_threads, daemon=True).start()

def update_json_file(mydtoken, tenant_id):
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    headers = {"Authorization": f"Bearer {mydtoken}"}
    params = {"tenantId": tenant_id, "page": 1}
    all_data = []

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
        load_json_data(tenant_id)

    except requests.RequestException as e:
        logger.error(f"Ошибка при запросе данных из API: {str(e)}")
        raise HTTPException(status_code=500, detail="Ошибка обновления JSON файла.")

def load_json_data(tenant_id):
    file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл с tenant_id={tenant_id} не найден.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"Данные для tenant_id={tenant_id} загружены.")
    data_cache[tenant_id] = data

    documents = [extract_text_fields(doc) for doc in data.get("data", {}).get("items", [])]
    embeddings_cache[tenant_id] = search_model.encode(documents, convert_to_tensor=True)

def extract_text_fields(record):
    excluded_keys = {"id", "categoryId", "currencyId", "langId", "employeeId", "employeeDescription"}
    return " ".join(str(value) for key, value in record.items() if key not in excluded_keys and value is not None)

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

        file_path = os.path.join(BASE_DIR, f"{tenant_id}.json")
        if not os.path.exists(file_path):
            update_json_file(mydtoken, tenant_id)

        if tenant_id not in embeddings_cache:
            load_json_data(tenant_id)

        document_embeddings = embeddings_cache[tenant_id]

        query_embedding = search_model.encode(input_text, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)
        similarities = similarities[0]
        top_results = similarities.topk(10)

        search_results = [
            {
                "text": data_cache[tenant_id]["data"]["items"][idx].get("serviceName", "Не указано"),
                "price": data_cache[tenant_id]["data"]["items"][idx].get("price", "Цена не указана"),
                "filial": data_cache[tenant_id]["data"]["items"][idx].get("filialName", "Филиал не указан"),
                "specialist": data_cache[tenant_id]["data"]["items"][idx].get("employeeFullName", "Специалист не указан")
            }
            for idx in top_results[1].tolist()
        ]

        context = "\n".join([
            f"Услуга: {res['text']}\nЦена: {res['price']} руб.\nФилиал: {res['filial']}\nСпециалист: {res['specialist']}"
            for res in search_results
        ])

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
