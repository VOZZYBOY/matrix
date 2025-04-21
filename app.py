# --- START OF FILE app.py (ТОЛЬКО REDIS + ВСЕ ИСПРАВЛЕНИЯ) ---

import logging
import time
import uvicorn
import asyncio
import os
import json
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import redis
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ИМПОРТЫ ДЛЯ НОВОГО СПОСОБА ОБРАБОТКИ ОШИБОК
from matrixai import initialize_clinic_assistant, sdk
try:
    from yandex_cloud_ml_sdk._exceptions import AioRpcError
    from grpc import StatusCode
    YANDEX_SDK_ERRORS_AVAILABLE = True
except ImportError:
    YANDEX_SDK_ERRORS_AVAILABLE = False
    AioRpcError = Exception # Заглушка
    StatusCode = None # Заглушка
    logging.warning("Не удалось импортировать AioRpcError или StatusCode. Обработка NotFound будет менее точной.")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("clinic_api")


# УБИРАЕМ user_sessions ПОЛНОСТЬЮ
# user_sessions: Dict[str, Any] = {}


class RedisManager:
    """
    Управляет сессиями пользователей через Redis.
    Подключение к Redis ОБЯЗАТЕЛЬНО для работы приложения.
    """
    def __init__(self, host=None, port=None, password=None, db=0, expiration_hours=24):
        self.host = host or os.environ.get("REDIS_HOST", "localhost")
        self.port = port or int(os.environ.get("REDIS_PORT", 6379))
        self.password = password or os.environ.get("REDIS_PASSWORD", None)
        self.db = db
        self.expiration_seconds = expiration_hours * 3600
        self.redis_client = None

        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info(f"Успешное подключение к Redis ({self.host}:{self.port})")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            # ПРЕРЫВАЕМ ЗАПУСК, ЕСЛИ REDIS НЕДОСТУПЕН
            logger.critical(f"Критическая ошибка: Не удалось подключиться к Redis ({self.host}:{self.port}): {e}")
            raise ConnectionError(f"Не удалось подключиться к Redis: {e}") from e
        except Exception as e:
            logger.critical(f"Критическая ошибка при инициализации Redis: {e}", exc_info=True)
            raise # Перевыброс для остановки

    def get_thread_id(self, user_id: str) -> Optional[str]:
        # Убрана проверка на self.redis_client, т.к. он гарантированно есть
        try:
            thread_id = self.redis_client.get(f"thread:{user_id}")
            if thread_id:
                self.redis_client.expire(f"thread:{user_id}", self.expiration_seconds)
                logger.debug(f"Получен thread_id {thread_id} из Redis для пользователя {user_id}")
            return thread_id
        except Exception as e:
            logger.error(f"Ошибка получения thread_id для {user_id} из Redis: {e}")
            # В теории, сюда не должны попадать при обязательном Redis, но оставим
            self.redis_client = None # Считаем соединение потерянным
            raise ConnectionError(f"Потеряно соединение с Redis при get_thread_id: {e}") from e

    def save_thread_id(self, user_id: str, thread_id: str) -> bool:
        # Убрана проверка на self.redis_client
        try:
            self.redis_client.setex(f"thread:{user_id}", self.expiration_seconds, thread_id)
            logger.info(f"Сохранен thread_id {thread_id} в Redis для пользователя {user_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка сохранения thread_id для {user_id} в Redis: {e}")
            self.redis_client = None
            raise ConnectionError(f"Потеряно соединение с Redis при save_thread_id: {e}") from e

    def delete_thread_id(self, user_id: str) -> bool:
        # Убрана проверка на self.redis_client
        try:
            result = self.redis_client.delete(f"thread:{user_id}")
            if result > 0:
                logger.info(f"Удален thread_id из Redis для пользователя {user_id}")
                return True
            else:
                logger.warning(f"Попытка удаления несуществующего thread_id из Redis для пользователя {user_id}")
                return False
        except Exception as e:
            logger.error(f"Ошибка удаления thread_id для {user_id} из Redis: {e}")
            self.redis_client = None
            raise ConnectionError(f"Потеряно соединение с Redis при delete_thread_id: {e}") from e

    def list_active_users(self) -> list:
        # Убрана проверка на self.redis_client
        try:
            keys = self.redis_client.keys("thread:*")
            user_ids = [key.split(":", 1)[1] for key in keys]
            return user_ids
        except Exception as e:
            logger.error(f"Ошибка получения списка активных пользователей из Redis: {e}")
            self.redis_client = None
            raise ConnectionError(f"Потеряно соединение с Redis при list_active_users: {e}") from e


class MessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    reset_session: bool = False


class MessageResponse(BaseModel):
    response: str
    user_id: str


clinic_agent = None
is_existing_index = False
redis_manager = None


# ЗАДАЧА ТОЛЬКО ДЛЯ МОНИТОРИНГА REDIS
async def monitor_active_sessions():
    """
    Асинхронная задача для периодического мониторинга активных сессий в Redis.
    """
    while True:
        await asyncio.sleep(3600)
        try:
            # redis_manager гарантированно существует, если приложение запущено
            active_users = redis_manager.list_active_users()
            logger.info(f"Мониторинг сессий: Активных сессий в Redis: {len(active_users)}")
        except ConnectionError as redis_conn_err:
            logger.error(f"Ошибка соединения с Redis в задаче мониторинга: {redis_conn_err}")
            # Можно добавить логику переподключения или ожидания
        except Exception as e:
            logger.error(f"Ошибка при мониторинге активных сессий: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global clinic_agent, is_existing_index, redis_manager
    monitoring_task = None

    try:
        # Инициализация Redis-менеджера (ПРЕРЫВАЕТ ЗАПУСК ПРИ ОШИБКЕ)
        redis_manager = RedisManager(
            host=os.environ.get("REDIS_HOST", "localhost"),
            port=int(os.environ.get("REDIS_PORT", 6379)),
            password=os.environ.get("REDIS_PASSWORD", None),
            expiration_hours=int(os.environ.get("SESSION_EXPIRY_HOURS", 24))
        )
        # Лог об успехе Redis теперь внутри RedisManager

        # Инициализация ассистента клиники
        logger.info("Инициализация ассистента клиники...")
        clinic_agent, is_existing_index = initialize_clinic_assistant()
        logger.info(f"Ассистент успешно инициализирован. Использует существующий индекс: {is_existing_index}")

        # Запуск задачи мониторинга активных сессий
        monitoring_task = asyncio.create_task(monitor_active_sessions())
        logger.info("Запущена задача мониторинга активных сессий")

    except ConnectionError as redis_err:
        # Логгер уже сработал в RedisManager
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось подключиться к Redis. Запуск приложения остановлен. Ошибка: {redis_err}")
        # Позволяем исключению прервать запуск
        raise
    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации приложения: {e}", exc_info=True)
        raise

    yield

    # Остановка задачи мониторинга сессий
    if monitoring_task:
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            logger.info("Задача мониторинга активных сессий остановлена")

    logger.info("Завершение работы API, очистка ресурсов...")
    if clinic_agent:
        try:
            clinic_agent.cleanup(delete_assistant=False) # Не удаляем ассистента
            logger.info("Ресурсы ассистента очищены")
        except Exception as e:
            logger.error(f"Ошибка при очистке ресурсов ассистента: {e}", exc_info=True)


app = FastAPI(
    title="Med YU Med Clinic Assistant API",
    description="API для взаимодействия с ассистентом (ТОЛЬКО Redis)", # Обновляем описание
    version="1.0.3", # Увеличим версию
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ЗАВИСИМОСТЬ НЕ ПРОВЕРЯЕТ REDIS, ТАК КАК ОН ОБЯЗАТЕЛЕН
def get_clinic_agent():
    if clinic_agent is None:
        logger.error("Запрос к API, но ассистент не инициализирован.")
        raise HTTPException(
            status_code=503, # Service Unavailable
            detail="Ассистент не инициализирован. Пожалуйста, попробуйте позже."
        )
    # redis_manager гарантированно существует и подключен, если приложение запустилось
    return clinic_agent


def generate_user_id():
    return f"user_{int(time.time() * 1000)}_{os.urandom(2).hex()}"


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ЛОГИКА /ask СНОВА ТОЛЬКО С REDIS
@app.post("/ask", response_model=MessageResponse, tags=["assistant"])
async def ask_assistant(
    request: MessageRequest,
    background_tasks: BackgroundTasks, # Порядок исправлен
    agent=Depends(get_clinic_agent)
):
    user_id = request.user_id if request.user_id else generate_user_id()
    reset = request.reset_session
    logger.info(f"Получен запрос от пользователя {user_id}: {request.message[:50]}... Reset: {reset}")

    thread_to_cleanup = None

    # Обработка сброса сессии - только через Redis
    if reset:
        logger.info(f"Запрос на сброс сессии для пользователя {user_id}")
        thread_id_to_delete = redis_manager.get_thread_id(user_id)
        if thread_id_to_delete:
            logger.info(f"Найден thread_id {thread_id_to_delete} в Redis для сброса.")
            redis_manager.delete_thread_id(user_id) # Удаляем из Redis
            try:
                thread_to_cleanup = sdk.threads.get(thread_id_to_delete)
                logger.info(f"Поток {thread_id_to_delete} будет удален в фоне.")
            except AioRpcError as e:
                 if YANDEX_SDK_ERRORS_AVAILABLE and e.code == StatusCode.NOT_FOUND:
                      logger.warning(f"Поток {thread_id_to_delete} для сброса уже не найден в YC.")
                 else:
                      logger.error(f"Ошибка gRPC при получении потока {thread_id_to_delete} для фонового удаления: {e.details} (Code: {getattr(e, 'code', 'N/A')})")
            except Exception as e:
                 logger.error(f"Неожиданная ошибка при получении потока {thread_id_to_delete} для фонового удаления: {e}")

            if thread_to_cleanup:
                background_tasks.add_task(cleanup_thread, thread_to_cleanup)
        else:
             logger.info(f"Сессия для пользователя {user_id} в Redis не найдена для сброса.")

    # Инициализация потока - только из Redis
    user_thread = None
    thread_id = redis_manager.get_thread_id(user_id) # Получаем ID из Redis
    if thread_id:
        logger.debug(f"Пытаемся восстановить поток {thread_id} из Redis для {user_id}")
        try:
            user_thread = sdk.threads.get(thread_id)
            logger.info(f"Восстановлена сессия из Redis для пользователя {user_id}, thread_id: {thread_id}")
        except AioRpcError as e:
            if YANDEX_SDK_ERRORS_AVAILABLE and e.code == StatusCode.NOT_FOUND:
                logger.warning(f"Поток {thread_id} (из Redis) для {user_id} не найден в YC. Удаляем запись из Redis.")
                redis_manager.delete_thread_id(user_id)
            else:
                logger.error(f"Ошибка gRPC при получении потока {thread_id} из YC: {e.details} (Code: {getattr(e, 'code', 'N/A')})")
                redis_manager.delete_thread_id(user_id)
            user_thread = None # Сбрасываем, чтобы не использовать
        except Exception as e:
            logger.error(f"Неожиданная ошибка при получении потока {thread_id} из YC: {e}")
            redis_manager.delete_thread_id(user_id)
            user_thread = None

    # Основной вызов агента
    try:
        response = agent(request.message, thread=user_thread) # Передаем поток (или None)

        # Сохранение информации о потоке - только в Redis
        if agent.thread and agent.thread.id:
            new_thread_id = agent.thread.id
            # Сохраняем/обновляем ID в Redis, если он изменился или его не было
            if new_thread_id != thread_id: # Сравниваем с ID, который был в Redis
                saved_redis = redis_manager.save_thread_id(user_id, new_thread_id)
                # Лог внутри save_thread_id
        else:
             logger.warning(f"Агент завершил обработку для {user_id}, но не вернул объект потока с ID.")

        logger.info(f"Ответ для пользователя {user_id}: {response[:50]}...")
        return MessageResponse(response=response, user_id=user_id)

    except Exception as e:
        # Ловим ошибки от agent() здесь
        logger.error(f"Критическая ошибка при обработке запроса агентом для пользователя {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Внутренняя ошибка сервера при обработке запроса."
        )

# Функция cleanup_thread остается без изменений
async def cleanup_thread(thread):
    if not thread:
        return
    try:
        thread_id = getattr(thread, 'id', 'unknown_id')
        if hasattr(thread, 'delete') and callable(thread.delete):
            thread.delete()
            logger.info(f"Фоновое удаление: Поток {thread_id} успешно удален из Yandex Cloud.")
        else:
             logger.warning(f"Фоновое удаление: Объект для потока {thread_id} не имеет метода delete.")
    except AioRpcError as e:
        if YANDEX_SDK_ERRORS_AVAILABLE and e.code == StatusCode.NOT_FOUND:
             logger.info(f"Фоновое удаление: Поток {thread_id} уже был удален.")
        else:
             logger.error(f"Фоновое удаление: Ошибка gRPC при удалении потока {thread_id}: {e.details} (Code: {getattr(e, 'code', 'N/A')})")
    except Exception as e:
        logger.error(f"Фоновое удаление: Неожиданная ошибка при удалении потока {thread_id}: {e}", exc_info=True)

# Эндпоинт reset_session СНОВА ТОЛЬКО С REDIS
@app.post("/reset_session", tags=["session"])
async def reset_session_endpoint(
    background_tasks: BackgroundTasks,    # Порядок исправлен
    user_id: str = Body(..., embed=True)
):
    logger.info(f"Запрос на сброс сессии для пользователя {user_id}")
    thread_to_cleanup = None
    thread_id_to_delete = redis_manager.get_thread_id(user_id) # Ищем ID в Redis

    if thread_id_to_delete:
        logger.info(f"Найден thread_id {thread_id_to_delete} в Redis для сброса сессии {user_id}.")
        if redis_manager.delete_thread_id(user_id): # Удаляем из Redis
            try:
                logger.info(f"Планируем удаление потока {thread_id_to_delete} в YC для {user_id}.")
                thread_to_cleanup = sdk.threads.get(thread_id_to_delete)
                background_tasks.add_task(cleanup_thread, thread_to_cleanup)
                return {"message": f"Сессия для пользователя {user_id} успешно сброшена (удаление потока запланировано)."}
            except AioRpcError as e:
                 if YANDEX_SDK_ERRORS_AVAILABLE and e.code == StatusCode.NOT_FOUND:
                      logger.warning(f"Поток {thread_id_to_delete} для сброса сессии {user_id} уже не найден в YC.")
                      return {"message": f"Сессия для пользователя {user_id} сброшена (поток уже был удален)."}
                 else:
                      logger.error(f"Ошибка gRPC при получении потока {thread_id_to_delete} для фонового удаления при сбросе сессии {user_id}: {e.details} (Code: {getattr(e, 'code', 'N/A')})")
                      return {"message": f"Сессия для пользователя {user_id} сброшена (ошибка при планировании очистки потока)."}
            except Exception as e:
                 logger.error(f"Неожиданная ошибка при получении потока {thread_id_to_delete} для фонового удаления при сбросе сессии {user_id}: {e}")
                 return {"message": f"Сессия для пользователя {user_id} сброшена (ошибка при планировании очистки потока)."}
        else:
             # Ошибка удаления из Redis (хотя get_thread_id сработал) - маловероятно
             logger.error(f"Не удалось удалить thread_id из Redis для {user_id}, хотя он был найден.")
             raise HTTPException(status_code=500, detail="Ошибка при сбросе сессии в хранилище.")
    else:
        # Сессия не найдена в Redis
        logger.info(f"Сессия для пользователя {user_id} не найдена в Redis.")
        raise HTTPException(status_code=404, detail=f"Сессия для пользователя {user_id} не найдена.")

# Health check СНОВА ТОЛЬКО С REDIS
@app.get("/health", tags=["health"])
async def health_check():
    redis_status = "disconnected"
    redis_sessions_count = -1

    # redis_manager гарантированно существует, если приложение запустилось
    try:
        # Проверим соединение еще раз
        redis_manager.redis_client.ping()
        redis_status = "connected"
        # Получаем количество активных сессий
        active_users = redis_manager.list_active_users()
        redis_sessions_count = len(active_users)
    except (redis.ConnectionError, redis.TimeoutError, AttributeError): # Добавили AttributeError на случай если redis_client стал None
         redis_status = "disconnected"
         logger.warning("Health check: Не удалось подключиться к Redis.")
    except Exception as e:
         redis_status = "error"
         logger.error(f"Health check: Ошибка при получении данных из Redis: {e}")

    active_sessions_count = redis_sessions_count if redis_sessions_count >= 0 else 0

    return {
        "status": "ok" if redis_status == "connected" and clinic_agent is not None else "partial_error",
        "agent_initialized": clinic_agent is not None,
        "redis_status": redis_status,
        "active_sessions": active_sessions_count, # Это и есть сессии Redis
    }

# Функция read_logs остается без изменений
def read_logs(max_lines=100):
    logs = []
    log_file_path = "api.log"
    try:
        with open(log_file_path, 'a+', encoding='utf-8') as log_file:
            log_file.seek(0)
            lines = log_file.readlines()[-max_lines:]
        if not lines and not os.path.exists(log_file_path):
            return [{"time": datetime.now().strftime("%H:%M:%S"), "level": "INFO", "message": f"Log file '{log_file_path}' not found, will be created."}]
        elif not lines:
            return [{"time": datetime.now().strftime("%H:%M:%S"), "level": "INFO", "message": f"Log file '{log_file_path}' is empty."}]
        for line in lines:
            try:
                parts = line.split(' - ', 3)
                if len(parts) == 4:
                    time_str_part = parts[0].split(' ')[1].split(',')[0]
                    level = parts[1]
                    message = parts[3].strip()
                    logs.append({"time": time_str_part, "level": level, "message": message})
                else:
                    logs.append({"time": "--:--:--", "level": "UNKNOWN", "message": line.strip()})
            except Exception as parse_err:
                logger.warning(f"Error parsing log line: {parse_err}. Line: {line.strip()}")
                logs.append({"time": "--:--:--", "level": "RAW", "message": line.strip()})
        return logs
    except FileNotFoundError:
        return [{"time": datetime.now().strftime("%H:%M:%S"), "level": "ERROR", "message": f"Log file '{log_file_path}' not found"}]
    except Exception as e:
        logger.error(f"Critical error reading logs: {e}", exc_info=True)
        return [{"time": datetime.now().strftime("%H:%M:%S"), "level": "ERROR", "message": f"Error reading logs: {str(e)}"}]


@app.get("/logs", tags=["logs"])
async def get_logs(request: Request, max_lines: int = 100):
    logs_data = read_logs(max_lines)
    return JSONResponse(content={"logs": logs_data})


if __name__ == "__main__":
    logger.info("Starting FastAPI server (use uvicorn to run)")
    uvicorn.run(app, host="0.0.0.0", port=8001)

