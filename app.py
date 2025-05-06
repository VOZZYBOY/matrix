# app.py (Пример использования matrixai_module.py)

import logging
import uvicorn
import os
import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import datetime
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable

# Импортируем менеджер конфигураций
try:
    import tenant_config_manager
except ImportError as e:
    logging.critical(f"Не удалось импортировать tenant_config_manager: {e}", exc_info=True)
    tenant_config_manager = None
try:
    from redis_history import clear_history as redis_clear_history
except ImportError as e:
    logging.critical(f"Не удалось импортировать redis_history: {e}", exc_info=True)
    redis_clear_history = None

try:
    from matrixai import agent_with_history
except ImportError as e:
     logging.critical(f"Не удалось импортировать agent_with_history из matrixai.py: {e}", exc_info=True)
     raise SystemExit(f"Критическая ошибка импорта: {e}")


log_file_path = "api.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file_path, encoding='utf-8')
    ]
)
logger = logging.getLogger("clinic_api_module_based")

agent: Optional[Runnable] = None

# Словарь для хранения активных RAG компонентов для разных тенантов
# (Сейчас не используется, т.к. компоненты глобальные в matrixai.py)
# rag_components_cache = {}

# --- Модели данных Pydantic ---
class AskRequest(BaseModel):
    user_id: str # Обязательный ID пользователя (для логирования и потенциально для истории)
    message: str
    session_id: Optional[str] = None # Сессия для истории диалога
    tenant_id: Optional[str] = None 
    reset_session: bool = False
    stream: bool = False 

class TenantSettings(BaseModel):
    prompt_addition: Optional[str] = None
    clinic_info_docs: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Список словарей, описывающих документы с общей информацией для тенанта. Каждый словарь должен содержать 'page_content' (str) и 'metadata' (dict)."
    )
    # Можно д

class SetTenantSettingsRequest(BaseModel):
    tenant_id: str
    settings: TenantSettings

class GetTenantSettingsResponse(BaseModel):
    prompt_addition: Optional[str] = None
    clinic_info_docs: Optional[List[Dict[str, Any]]] = None
    last_modified_general: Optional[str] = None # Добавим время последней модификации
    last_modified_clinic_info: Optional[str] = None

class MessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    reset_session: bool = False
    tenant_id: str 

class MessageResponse(BaseModel):
    response: str
    user_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    logger.info("Запуск FastAPI приложения...")
    try:
        logger.info("Получение инициализированного агента из matrixai...")
        agent = agent_with_history
        if agent:
             logger.info("Агент (agent_with_history) успешно получен из matrixai.")
        else:
             logger.critical("Критическая ошибка: agent_with_history из matrixai равен None.")
    except Exception as e:
        logger.critical(f"Критическая ошибка при получении агента из matrixai: {e}", exc_info=True)
        agent = None 
    yield
    logger.info("Завершение работы API.")


app = FastAPI(
    title="Clinic Assistant API (Module)",
    description="API для взаимодействия с ассистентом (использует matrixai)",
    version="3.2.0", # Обновим версию
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_agent() -> Runnable:
    if agent is None:
        logger.error("Запрос к API, но агент не инициализирован.")
        raise HTTPException(status_code=503, detail="Агент не инициализирован.")
    return agent

def generate_user_id() -> str:
    return f"user_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"



@app.get("/")
async def read_root(request: Request):
    logger.info("Запрос корневой страницы (Admin+Chat)")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_model=MessageResponse, tags=["assistant"])
async def ask_assistant(
    request: MessageRequest,
    agent_dependency: Runnable = Depends(get_agent)
):
    user_id = request.user_id if request.user_id else generate_user_id()
    reset = request.reset_session
    tenant_id = request.tenant_id 
    if not tenant_id:
        logger.error(f"Получен запрос без tenant_id от {user_id}")
        raise HTTPException(status_code=400, detail="Параметр 'tenant_id' обязателен.")

    logger.info(f"Получен запрос от tenant '{tenant_id}', user '{user_id}': {request.message[:50]}... Reset: {reset}")


    if reset:
        target_user_id = user_id
        if redis_clear_history:
            cleared = redis_clear_history(tenant_id=tenant_id, user_id=target_user_id)
            logger.info(f"Запрос на сброс сессии для tenant '{tenant_id}', user '{target_user_id}'. Сессия удалена: {cleared}")
        else:
             logger.error(f"Функция redis_clear_history не доступна для tenant '{tenant_id}', user '{target_user_id}'")

    try:
        start_time = time.time()
        target_user_id = user_id
        composite_session_id = f"{tenant_id}:{target_user_id}"
        logger.debug(f"Создан composite_session_id для истории: {composite_session_id}")

        response_data = await agent_dependency.ainvoke(
            {"input": request.message},
            config={
                "configurable": {
                    "session_id": composite_session_id,
                    "user_id": target_user_id,
                    "tenant_id": tenant_id
                 }
            }
        )
        end_time = time.time()
        logger.info(f"Агент ответил для tenant '{tenant_id}', user '{target_user_id}' за {end_time - start_time:.2f} сек.")

        if isinstance(response_data, str):
            response_text = response_data
        else:
            logger.warning(f"Агент вернул тип {type(response_data)}. Преобразуем в строку.")
            response_text = str(response_data)

        logger.info(f"Ответ для {target_user_id}: {response_text[:50]}...")
        return MessageResponse(response=response_text, user_id=target_user_id)

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке запроса для {target_user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера.")

@app.get("/health", tags=["health"])
async def health_check():
    agent_ok = agent is not None
    return { "status": "ok" if agent_ok else "error", "agent_initialized": agent_ok } 

@app.post("/tenant_settings", tags=["tenant_config"], status_code=200)
async def set_tenant_settings(request: SetTenantSettingsRequest):
    if not tenant_config_manager:
        raise HTTPException(status_code=503, detail="Менеджер конфигураций тенантов недоступен.")

    tenant_id = request.tenant_id
    settings_data = request.settings
    general_settings = settings_data.dict(exclude_unset=True, exclude={"clinic_info_docs"})
    clinic_info_docs_data = settings_data.clinic_info_docs

    success_general = True
    success_clinic_info = True

    if general_settings: 
        logger.info(f"Получен запрос на сохранение общих настроек для тенанта '{tenant_id}': {general_settings}")
        current_settings = tenant_config_manager.load_tenant_settings(tenant_id)
        current_settings.update(general_settings)
        success_general = tenant_config_manager.save_tenant_settings(tenant_id, current_settings)
        if not success_general:
             logger.error(f"Не удалось сохранить общие настройки для тенанта '{tenant_id}'.")

    if clinic_info_docs_data is not None: 
        logger.info(f"Получен запрос на сохранение clinic_info_docs для тенанта '{tenant_id}' ({len(clinic_info_docs_data)} док-ов)")
        success_clinic_info = tenant_config_manager.save_tenant_clinic_info(tenant_id, clinic_info_docs_data)
        if not success_clinic_info:
            logger.error(f"Не удалось сохранить clinic_info_docs для тенанта '{tenant_id}'.")

    if success_general and success_clinic_info:
        return {"message": f"Настройки для тенанта '{tenant_id}' успешно сохранены/обновлены."}
    else:
        error_details = []
        if not success_general: error_details.append("общие настройки")
        if not success_clinic_info: error_details.append("clinic_info_docs")
        raise HTTPException(status_code=500, detail=f"Не удалось сохранить части настроек ({', '.join(error_details)}) для тенанта '{tenant_id}'.")

@app.get("/tenant_settings/{tenant_id}", response_model=GetTenantSettingsResponse, tags=["tenant_config"])
async def get_tenant_settings(tenant_id: str):
    if not tenant_config_manager:
        raise HTTPException(status_code=503, detail="Менеджер конфигураций тенантов недоступен.")

    logger.info(f"Запрос настроек для тенанта '{tenant_id}'")
    try:
        # Загружаем основные настройки (включая prompt_addition)
        settings = tenant_config_manager.load_tenant_settings(tenant_id)
        # Загружаем clinic_info_docs
        clinic_info = tenant_config_manager.load_tenant_clinic_info(tenant_id)
        # Получаем время модификации файлов
        general_mod_time = tenant_config_manager.get_settings_file_mtime(tenant_id)
        clinic_info_mod_time = tenant_config_manager.get_clinic_info_file_mtime(tenant_id)

        # Если файлы не найдены, время будет None
        general_mod_time_str = datetime.datetime.fromtimestamp(general_mod_time).isoformat() if general_mod_time else None
        clinic_info_mod_time_str = datetime.datetime.fromtimestamp(clinic_info_mod_time).isoformat() if clinic_info_mod_time else None

        # Если файлов нет, load_tenant_settings вернет {}, а load_tenant_clinic_info вернет []
        # или они могут выбросить исключение, если мы так решим в менеджере
        # Пока предполагаем, что они возвращают пустые значения

        return GetTenantSettingsResponse(
            prompt_addition=settings.get('prompt_addition'),
            clinic_info_docs=clinic_info,
            last_modified_general=general_mod_time_str,
            last_modified_clinic_info=clinic_info_mod_time_str
        )

    except FileNotFoundError:
        logger.warning(f"Файлы настроек для тенанта '{tenant_id}' не найдены.")
        raise HTTPException(status_code=404, detail=f"Настройки для тенанта '{tenant_id}' не найдены.")
    except Exception as e:
        logger.error(f"Ошибка при загрузке настроек для тенанта '{tenant_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка при чтении настроек для '{tenant_id}'.")

@app.get("/tenants", response_model=List[str], tags=["tenant_config"])
async def get_tenant_list():
    if not tenant_config_manager:
        raise HTTPException(status_code=503, detail="Менеджер конфигураций тенантов недоступен.")
    try:
        tenant_ids = tenant_config_manager.list_tenants()
        logger.info(f"Запрос списка тенантов. Найдено: {len(tenant_ids)}")
        return tenant_ids
    except Exception as e:
        logger.error(f"Ошибка при получении списка тенантов: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Не удалось получить список тенантов.")

def read_logs(max_lines=100):
    logs = []
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
                if len(parts) >= 4:
                    time_str_part = parts[0].split(' ')[1].split(',')[0]
                    level = parts[1]
                    message = parts[3].strip()
                    logs.append({"time": time_str_part, "level": level, "message": message})
                elif len(parts) > 0:
                     logs.append({"time": "--:--:--", "level": "RAW", "message": line.strip()})
            except Exception as parse_err:
                logger.warning(f"Error parsing log line: {parse_err}. Line: {line.strip()}")
                logs.append({"time": "--:--:--", "level": "PARSE_ERR", "message": line.strip()})
        return logs
    except FileNotFoundError:
        logger.warning(f"Log file '{log_file_path}' not found during read.")
        return [{"time": datetime.now().strftime("%H:%M:%S"), "level": "ERROR", "message": f"Log file '{log_file_path}' not found"}]
    except Exception as e:
        logger.error(f"Critical error reading logs: {e}", exc_info=True)
        return [{"time": datetime.now().strftime("%H:%M:%S"), "level": "ERROR", "message": f"Error reading logs: {str(e)}"}]


@app.get("/logs", tags=["logs"])
async def get_logs(request: Request, max_lines: int = 100):
    logs_data = read_logs(max_lines)
    return JSONResponse(content={"logs": logs_data})

if __name__ == "__main__":
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = int(os.getenv("APP_PORT", 8001))
    log_level = os.getenv("APP_LOG_LEVEL", "info").lower()
    logger.info(f"Starting FastAPI server via uvicorn on {app_host}:{app_port}")
    uvicorn.run("app:app", host=app_host, port=app_port, log_level=log_level, reload=True)  