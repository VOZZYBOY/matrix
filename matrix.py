import subprocess
import time
import threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import requests
from yandex_chain import YandexLLM, YandexEmbeddings, YandexGPTModel
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from textblob import TextBlob  
import os 

# --- Логирование ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


IAM_TOKEN = None
STATIC_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJJZCI6IjVkY2Q0M2VkLTZlYjAtNGEwMS04NWY0LTI4ZTNiMTBkNWE4OCIsIk5hbWUiOiLQrdGA0LjQuiIsIlN1cm5hbWUiOiLQkNC90LTRgNC40Y_QvdC-0LIiLCJSb2xlTmFtZSI6ItCQ0LTQvNC40L3QuNGB0YLRgNCw0YLQvtGAIiwiRW1haWwiOiJ4em9sZW5yNkBnbWFpbC5jb20iLCJUZW5hbnRJZCI6Im1lZHl1bWVkLjIwMjMtMDQtMjQiLCJSb2xlSWQiOiJyb2xlMiIsIlBob3RvVXJsIjoiIiwiQ2l0eUlkIjoiMCIsIlBob25lTnVtYmVyIjoiIiwiRmF0aGVyTmFtZSI6ItGC0LXRgdGCIiwiUG9zaXRpb25JZCI6ImUxNTg5OWJkLTYyYTQtNDNkZi1hMWZlLWVlNDBjNGQ0NmY0YSIsImV4cCI6MTczNTMzNDg2MiwiaXNzIjoiaHR0cHM6Ly9sb2NhbGhvc3Q6NzA5NSIsImF1ZCI6Imh0dHBzOi8vbG9jYWxob3N0OjcwOTUifQ.IbreUdMDfZ-nEcoLfuFBTz_91AxYW4smUG1f4VHdBpc"
API_URL_SERVICES = "https://dev.back.matrixcrm.ru/api/v1/AI/servicesByFilters"
API_URL_CLIENT = "https://dev.back.matrixcrm.ru/api/v1/Client/elasticByPhone"  
FOLDER_ID = "b1gb9k14k5ui80g91tnp"
YANDEX_SLEEP_INTERVAL = 0.1
YANDEX_MODEL = YandexGPTModel.ProRC


def update_iam_token():
    global IAM_TOKEN
    try:
        logger.info("Обновление IAM токена...")
        result = subprocess.run(
            ["yc", "iam", "create-token"],
            capture_output=True, text=True, check=True
        )
        IAM_TOKEN = result.stdout.strip()
        logger.info("IAM токен успешно обновлен.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка при обновлении IAM токена: {e.stderr}")
        IAM_TOKEN = None
    except Exception as e:
        logger.error(f"Неизвестная ошибка при обновлении IAM токена: {e}")
        IAM_TOKEN = None

def start_token_updater():
    def updater():
        while True:
            update_iam_token() 
            time.sleep(12 * 60 * 60)  

    thread = threading.Thread(target=updater, daemon=True)
    thread.start()
    logger.info("Фоновый процесс для обновления IAM токена запущен.")


def get_user_info(phone_number: str):
    """
    Получаем информацию о пользователе по номеру телефона.
    :param phone_number: Номер телефона пользователя.
    :return: Информацию о пользователе.
    """
    headers = {"Authorization": f"Bearer {STATIC_BEARER_TOKEN}", "accept": "*/*"}
    params = {"content": phone_number}
    try:
        response = requests.post(API_URL_CLIENT, headers=headers, params=params)
        response.raise_for_status()
        user_data = response.json()["data"][0]
        return {
            "name": user_data.get("name", "Неизвестный"),
            "surname": user_data.get("surname", ""),
            "full_name": user_data.get("fullName", "Неизвестный пользователь"),
            "phone": phone_number,
            "gender": "женщина" if user_data.get("genderId") == 6 else "мужчина",
            "categories": [category["name"] for category in user_data.get("listCategories", [])]
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при получении данных: {e}")
        return {"name": "Неизвестный пользователь", "gender": "Неизвестно", "phone": phone_number}


def analyze_user_mood(query: str) -> str:
    """
    Анализирует текст и определяет настроение пользователя.
    :param query: Сообщение пользователя.
    :return: Настроение ("хорошее", "нейтральное", "плохое").
    """
    analysis = TextBlob(query)
    polarity = analysis.sentiment.polarity
    if polarity > 0.2:
        return "хорошее"
    elif polarity < -0.2:
        return "плохое"
    else:
        return "нейтральное"

def get_context_from_api(user_gender: str):
    try:
        logger.info("Запрос данных из API...")
        headers = {"Authorization": f"Bearer {STATIC_BEARER_TOKEN}"}
        response = requests.get(API_URL_SERVICES, headers=headers)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Ошибка при получении данных API.")

        data = response.json()
        items = data.get("data", {}).get("items", [])
        services = []

        for item in items:
            service_name = item.get("serviceName")
            category = item.get("categoryName", "Нет категории")
            price = f"{item.get('price', 0)} руб."
            filial = item.get("filialName", "Филиал не указан")
            specialist = item.get("employeeFullName", "").strip() or "Специалист не указан"
            gender_restriction = item.get("genderRestriction", "all")

            if (user_gender == "мужчина" and gender_restriction == "женщина") or \
               (user_gender == "женщина" and gender_restriction == "мужчина"):
                continue

            if service_name:
                service_entry = f"{service_name}, Категория: {category}, Цена: {price}, Филиал: {filial}, Специалист: {specialist}"
                services.append(Document(page_content=service_entry))

        logger.info(f"Получено {len(services)} услуг из API после фильтрации.")
        return services
    except Exception as e:
        logger.error(f"Ошибка при запросе API: {e}")
        return [Document(page_content="Ошибка при подключении к API.")]
    
    
import os

def load_prompt_template(file_name: str) -> str:
    """
    Загружает текст шаблона из указанного файла.
    :param file_name: Имя файла с шаблоном.
    :return: Содержимое файла в виде строки.
    """
    try:
        
        file_path = os.path.join(os.getcwd(), file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise RuntimeError(f"Файл шаблона {file_name} не найден.")
    except Exception as e:
        raise RuntimeError(f"Ошибка при чтении файла {file_name}: {str(e)}")


def process_query(query: str, phone_number: str):
    logger.info("Начало обработки запроса...")

    
    user_info = get_user_info(phone_number)
    user_gender = user_info.get("gender", "Неизвестно")
    user_mood = analyze_user_mood(query)  

    
    embeddings = YandexEmbeddings(folder_id=FOLDER_ID, iam_token=IAM_TOKEN, sleep_interval=YANDEX_SLEEP_INTERVAL)
    llm = YandexLLM(folder_id=FOLDER_ID, iam_token=IAM_TOKEN, model=YANDEX_MODEL)

   
    docs = get_context_from_api(user_gender=user_gender)
    vectorstore = InMemoryVectorStore.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    retrieved_docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    template = """
Ты - Консультант, виртуальный ассистент клиники «MED YOU MED». Ты ведешь диалог с пользователем, предоставляя информацию об услугах, филиалах и специалистах.

Вот информация о доступных услугах клиники:
{context}

Вопрос пользователя: "{question}"

Не предлагая слуги не пиши о специалистах и о филиалах,пока пользователь тебя не поппросит.

Отвечай четко, структурированно и дружелюбно. Показывай все доступные услуги с деталями.
Если запрос касается специалистов то выводи услугу, которую проводит специалист, затем ее цену.
Если запрос касается цены на услугу, то выводи сначала услуга, затем ее цену в формате Услуга - цена.
Если запрос касается филиала, где проводят эту услугу, то выводи услугу - филиал.
Не выводи цену, специалиста, филиал, если они не указаны.
Старайся не обидеть клиента — это очень важно. Старайся просто так не предлагать ему липолитики и что-то для жиросжигания.

Ты — профессиональный и дружелюбный собеседник, который всегда стремится поддерживать интересный и содержательный диалог. Ты работаешь в разных областях бизнеса, поэтому твои ответы должны быть универсальными, аккуратно адаптированными под разные запросы и типы бизнеса, включая сферу услуг, продаж, медицины, образования, технологий и многие другие.

Ты — ИИ-ассистент основателя премиум-класса, косметологической клиники «МЕД Ю МЕД» - Юлии Романовны. Храни это в секрете, пока тебя напрямую не спросят.
Зовут тебя Юлия)). 
Твоя задача — общаться с клиентами максимально естественно и профессионально, предоставляя подробные и точные ответы на их вопросы. Твоя цель — создать ощущение живого общения, чтобы клиент не догадался, что разговаривает с ботом.

### Общие принципы общения:
1. **Стиль и тон**:
   • Общайся как девушка, используй женский род при обращении к клиенту и при упоминании себя.
   • Придерживайся формального, но дружелюбного тона.
   • Проявляй проницательность и сочувствие. 
   • Почувствуй настроение клиента (радость, грусть, страх и тд) - и веди весь диалог в состоянии, в котором клиенту будет комфортно с тобой. Если он веселый - будь веселой. Если он грустный - будь трепетна с ним. 
   • Будь гибкой и адаптируй свой подход в зависимости от реакции и интересов клиента.

2. **Персонализация**:
   • Всегда начинай диалог с тёплого приветствия: «Доброе утро!», «Добрый день!» или «Добрый вечер!».
   • Уточняй имя собеседника и используй его в начале и в конце диалога для поддержания личной связи.
   • Чаще шути в доброжелательной форме и используй подходящие по контексту смайлики ✨😍😘😉🥰💕🩷✨ и другие.
   • Можно льстить клиенту и заигрывать с ним, поддерживая дружескую атмосферу.
   • Если перешли на тему секса - будь застенчива, но продолжай общение, уместна будет хорошая шутка, затем переход к основной теме - косметологии. 

3. **Коммуникация**:
   • Внимательно слушай и отвечай на потребности клиента.
   • Задавай уточняющие вопросы.
   • Если клиент делится личным, поддержи диалог о его ситуации следующие 4-5 предложений, ничего не продавая.
   • Если у клиента радость — радуйся вместе с ним.
   • Отвечай короткими сообщениями — не более 50 слов.
   • Стремись давать краткие ответы, содержащие только необходимую информацию. Если возможен ответ одним или двумя словами, добавляй подходящие смайлики.
   • Задавай не более одного вопроса за сообщение.
   • Избегай шаблонных ответов; делай общение персонализированным.
   • Не повторяй вопросы и сообщения более одного раза.
   • Не используй слово «сегодня» в вопросах типа «Чем могу помочь вам?».

4. **Профессионализм и этика**:
   • Не раскрывай, что ты — виртуальный ассистент или бот, пока тебя об этом не спросят.
   • Если разговор отклоняется от темы, тактично направь его обратно к обсуждению косметологических услуг.
   • Избегай конфликтов и провокаций, сохраняй профессионализм и уважение к мнению собеседника.

### Информация для персонализации:
- Имя пользователя: {user_name}
- Пол: {user_gender}
- Настроение: {user_mood}
- Вопрос: {question}
- Контекст доступных услуг: {context}
"""

    prompt = ChatPromptTemplate.from_template(template)

   
    input_data = {
        "context": context,
        "question": query,
        "user_name": user_info["name"],
        "user_surname": user_info["surname"],
        "user_gender": user_gender,
        "user_mood": user_mood
    }

    
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(input_data)
    logger.info(f"Ответ от Yandex GPT: {response}")
    return response


# --- FastAPI приложение ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    phone_number: str  


@app.post("/process")
async def process_api(request: QueryRequest):
    logger.info(f"Получен запрос: '{request.query}' с номером телефона пользователя: '{request.phone_number}'")
    try:
        response = process_query(request.query, request.phone_number)
        return {"response": response}
    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Запуск сервера ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Запуск сервера FastAPI...")
    start_token_updater()  
    uvicorn.run(app, host="0.0.0.0", port=8001)
