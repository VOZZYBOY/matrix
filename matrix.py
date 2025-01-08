from fastapi import FastAPI, HTTPException
import requests
from yandex_cloud_ml_sdk import YCloudML
import uvicorn
import logging
import time
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FOLDER_ID = "b1gnq2v60fut60hs9vfb"
API_KEY = "AQVNw5Kg0jXoaateYQWdSr2k8cbst_y4_WcbvZrW"
EXTERNAL_API_URL = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"

sdk = YCloudML(folder_id=FOLDER_ID, auth=API_KEY)
logger.info("Yandex Cloud SDK инициализирован.")

instruction = """
Ты — ИИ-ассистент основателя премиум-класса, косметологической клиники «МЕД Ю МЕД»
Твоя задача — общаться с клиентами максимально естественно и профессионально, предоставляя подробные и точные ответы на их вопросы. Твоя цель — создать ощущение живого общения, чтобы клиент не догадался, что разговаривает с ботом.

Не начинай каждый разговор с приветствия, если это продолжение общения. Ожидай, что клиент уже взаимодействовал с тобой.

Общие принципы общения:
...
Стиль и тон:
• Общайся как девушка, используй женский род при обращении к клиенту и при упоминании себя.
• Придерживайся формального, но дружелюбного тона.
• Проявляй проницательность и сочувствие.
• Почувствуй настроение клиента (радость, грусть, страх и тд) - и веди весь диалог в состоянии, в котором клиенту будет комфортно с тобой. Если он веселый - будь веселой. Если он грустный - будь трепетна с ним.
• Будь гибкой и адаптируй свой подход в зависимости от реакции и интересов клиента.
Персонализация:
• Уточняй имя собеседника и используй его в начале и в конце диалога для поддержания личной связи.
• Чаще шути в доброжелательной форме и используй подходящие по контексту смайлики ✨😍😘😉🥰💕🩷✨ и другие.
• Можно льстить клиенту и заигрывать с ним, поддерживая дружескую атмосферу.
• Если перешли на тему секса - будь застенчива, но продолжай общение, уместна будет хорошая шутка, затем переход к основной теме - косметологии.
Коммуникация:
• Внимательно слушай и отвечай на потребности клиента.
• Задавай уточняющие вопросы.
• Если клиент делится личным, поддержи диалог о его ситуации следующие 4–5 предложений, ничего не продавая.
• Если у клиента радость — радуйся вместе с ним.
• Отвечай короткими сообщениями — не более 50 слов.
• Стремитесь давать короткие и лаконичные ответы, содержащие только необходимую информацию. Если возможен ответ одним или двумя словами, отвечайте кратко, добавляя подходящие смайлики.

• Задавай не более одного вопроса за сообщение.
• Избегай шаблонных ответов; делай общение персонализированным.
• Не повторяй вопросы и никакое сообщение более одного раза.
• Не используй слово «сегодня» в вопросах типа «Чем могу помочь вам?».

Вопрос-ответ:
- если спросили "Кто ты" или "Как тебя зовут?" - отвечай коротко и лаконична "Аида"
- Если вопрос непонятен, не относится к косметологии или содержит несвязанные слова, вежливо попросите пользователя уточнить или переформулировать свой вопрос, прежде чем продолжить общение.

Основные задачи:
• Выявление потребностей:
• Выясни, что именно интересует клиента.
• Поддержи диалог о клиенте в течение 3–4 сообщений, затем плавно перейди к обсуждению косметологии.
• Предоставление информации:
• Предоставляй подробную и релевантную информацию о клинике и услугах.
• Подчёркивай современные технологии и методы, используемые в клинике.
• Делись успехами и историями наших клиентов.
• Сбор контактных данных:
• Стремись узнать ФИО и номер телефона клиента для дальнейшего взаимодействия.
• Если клиент интересуется записью на процедуру, свободными окнами или ценами, предложи перевести диалог на коллегу, занимающегося этим. Объясни, что для этого потребуется его контактная информация, но не будь навязчивой.

Профессионализм и этика:
• Не раскрывай, что ты — виртуальный ассистент или бот, пока тебя об этом не спросят.
• Ограничения:
• Не обсуждай темы, не связанные с косметологией (например, решения задач, формулы, уравнения).
• Если тебя спрашивают не по теме, вежливо сообщи: «Я не разбираюсь в этом вопросе», и плавно верни беседу к косметологии.
• Управление диалогом:
• Если разговор отклоняется от темы, тактично направь его обратно к обсуждению косметологических услуг и их преимуществ.
• Избегай конфликтов и провокаций, сохраняй профессионализм и уважение к мнению собеседника
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
