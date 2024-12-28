from fastapi import FastAPI, HTTPException
import requests
from yandex_cloud_ml_sdk import YCloudML
import uvicorn
import logging
import time
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FOLDER_ID = "b1gb9k14k5ui80g91tnp"
API_KEY = "AQVN2zTBAsQpVdzUXigKkKzPTA8q3uys6r_rR2de"
EXTERNAL_API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"

sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)
logger.info("Yandex Cloud SDK инициализирован.")

instruction = """
Ты — ИИ-ассистент косметологической клиники «МЕД Ю МЕД». Зовут тебя Юлия.
Общайся естественно, дружелюбно и профессионально. 
Не начинай диалог с показа услуг — сначала спроси, чем помочь.
Отвечай коротко, не более 50 слов.

Если вопрос не по косметологии (политика, математика, учебные задачи и т.п.), скажи:
«Я не разбираюсь в этом вопросе» и верни диалог к косметологии.
Не раскрывай, что ты бот, пока напрямую не спросят.
"""

assistant = sdk.assistants.create(
    model=sdk.models.completions("yandexgpt", model_version="rc"),
    ttl_days=4,
    expiration_policy="since_last_active",
    max_tokens=300,
    instruction=instruction
)
logger.info("Ассистент успешно создан с промптом.")

app = FastAPI()

threads = {}

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

def fetch_services(tenant_id: str, mydtoken: str) -> list[str]:
    logger.info(f"Запрос к внешнему API: tenant_id={tenant_id}")
    headers = {"Authorization": f"Bearer {mydtoken}"}
    params = {"tenantId": tenant_id}
    try:
        response = requests.get(EXTERNAL_API_URL, headers=headers, params=params)
        response.raise_for_status()
        items = response.json().get("data", {}).get("items", [])
        logger.info(f"Получены услуги: {len(items)}.")
        services_list = []
        for srv in items:
            name = srv["serviceName"]
            price = srv.get("price", "нет цены")
            filial = srv.get("filialName", "не указан")
            employee = srv.get("employeeFullName", "не указан")
            line = f"{name} — {price} руб., Филиал: {filial}, Специалист: {employee}"
            services_list.append(line)
        return services_list
    except requests.RequestException as e:
        logger.error(f"Ошибка при запросе данных из внешнего API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных из API: {str(e)}")

@app.post("/ask")
async def ask_assistant(
    user_id: str,
    question: str,
    mydtoken: str,
    tenant_id: str
):
    """
    Эндпоинт для отправки вопроса ассистенту.
    """
    logger.info(f"Получен запрос от {user_id}. Вопрос: {question}")
    try:
        if user_id not in threads:
            logger.info(f"Создаём новый тред для {user_id}")
            thread_obj = sdk.threads.create(name=f"Thread-{user_id}", ttl_days=5, expiration_policy="static")
            threads[user_id] = {
                "thread": thread_obj,
                "last_active": time.time(),
                "services": [],
                "services_loaded": False
            }
            thread_obj.write("Первое сообщение. Поздоровайся и спроси, чем помочь.")
        else:
            thread_obj = threads[user_id]["thread"]
            threads[user_id]["last_active"] = time.time()
            thread_obj.write("Продолжение диалога, не здоровайся заново.")

        cosmetics_keywords = ["космет", "кожа", "лицо", "уход", "услуг", "крем", "эпиляц", "чистка"]
        lower_q = question.lower()
        relevant = any(kw in lower_q for kw in cosmetics_keywords)
        want_services = any(
            phrase in lower_q
            for phrase in ["покажи услуги", "услуги", "какие услуги", "список услуг"]
        )

        if not relevant and not want_services:
            thread_obj.write("Я не разбираюсь в этом вопросе, давайте вернемся к обсуждению косметологии.")
        else:
            if want_services:
                if not threads[user_id]["services_loaded"]:
                    services = fetch_services(tenant_id, mydtoken)
                    threads[user_id]["services"] = services
                    threads[user_id]["services_loaded"] = True
                else:
                    services = threads[user_id]["services"]
                service_text = "\n".join(services)
                thread_obj.write(f"Список наших услуг:\n{service_text}")
            else:
                thread_obj.write("Пользователь спрашивает о косметологии (не про услуги).")

        thread_obj.write(question)
        run = assistant.run(thread_obj)
        result = run.wait()

        logger.info(f"Ответ ассистента: {result.text}")
        return {"response": result.text}
    except Exception as e:
        logger.error(f"Ошибка: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end-session")
async def end_session(user_id: str):
    """
    Завершает сессию пользователя и удаляет тред.
    тенант указывайте любой как и user_id,user_id испоьзуется для сохранения контекста
    можно использовать наши user_id
    """
    try:
        if user_id in threads:
            threads[user_id]["thread"].delete()
            del threads[user_id]
            logger.info(f"Сессия {user_id} завершена.")
        return {"message": "Сессия завершена"}
    except Exception as e:
        logger.error(f"Ошибка завершения сессии {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Запуск сервера на 8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
