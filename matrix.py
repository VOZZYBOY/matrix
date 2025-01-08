from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from voicerecognise import recognize_audio_with_sdk
from yandex_cloud_ml_sdk import YCloudML
import aiohttp
import uvicorn
import asyncio
import logging
import os
import time


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", encoding="utf-8")
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
    ttl_days=365,
    expiration_policy="since_last_active",
    max_tokens=300,
    instruction=instruction
)
logger.info("Ассистент успешно создан с максимальным временем жизни (365 дней).")

app = FastAPI()
threads = {}

async def fetch_services(tenant_id: str, mydtoken: str) -> list[dict]:
    """
    Асинхронное получение списка услуг из внешнего API.
    """
    logger.info(f"Запрос к внешнему API: tenant_id={tenant_id}")
    headers = {"Authorization": f"Bearer {mydtoken}"}
    params = {"tenantId": tenant_id}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(EXTERNAL_API_URL, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                items = data.get("data", {}).get("items", [])
                logger.info(f"Получено {len(items)} услуг.")
                return items
    except Exception as e:
        logger.error(f"Ошибка при запросе данных из внешнего API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения данных из API: {str(e)}")

@app.post("/ask")
async def ask_assistant(
    user_id: str = Form(...),
    mydtoken: str = Form(...),
    tenant_id: str = Form(...),
    question: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Эндпоинт для обработки запросов ассистенту.
    """
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

        
        if user_id not in threads:
            logger.info(f"Создаём новый тред для {user_id}")
            thread_obj = sdk.threads.create(name=f"Thread-{user_id}", ttl_days=365, expiration_policy="since_last_active")
            threads[user_id] = {
                "thread": thread_obj,
                "last_active": time.time(),
                "services": None
            }
            thread_obj.write("Первое сообщение. Чем могу помочь?")
        else:
            thread_obj = threads[user_id]["thread"]
            threads[user_id]["last_active"] = time.time()

        
        if "услуги" in input_text.lower():
            if not threads[user_id]["services"]:
                services = await fetch_services(tenant_id, mydtoken)
                threads[user_id]["services"] = services
            else:
                services = threads[user_id]["services"]

            
            service_context = [{"name": srv["serviceName"], 
                                "price": srv.get("price", "нет цены"), 
                                "filial": srv.get("filialName", "не указан"), 
                                "employee": srv.get("employeeFullName", "не указан")} for srv in services]
            thread_obj.write(f"Контекст услуг:\n{service_context}")

        thread_obj.write(input_text)

        
        logger.info("Отправка треда ассистенту.")
        run = assistant.run(thread_obj)
        result = run.wait()

        logger.info(f"Ответ ассистента: {result.text}")
        return {"response": result.text}

    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/end-session")
async def end_session(user_id: str):
    """
    Завершение сессии пользователя.
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
