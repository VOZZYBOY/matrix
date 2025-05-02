# app.py (Пример использования matrixai_module.py)

import logging
import uvicorn
import os
import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict
import datetime
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel


try:
    from matrixai import (
        initialize_assistant,
        clear_session_history,
        get_active_session_count
    )
    from langchain_core.runnables import Runnable
except ImportError as e:
     logging.critical(f"Не удалось импортировать компоненты из matrixai_module.py: {e}", exc_info=True)
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

assistant_chain: Optional[Runnable] = None

class MessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    reset_session: bool = False

class MessageResponse(BaseModel):
    response: str
    user_id: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global assistant_chain
    logger.info("Запуск FastAPI приложения...")
    try:
        logger.info("Инициализация Ассистента...")
        assistant_chain = initialize_assistant(
        )
        logger.info("Ассистент успешно инициализирован.")
    except Exception as e:
        logger.critical(f"Критическая ошибка инициализации ассистента: {e}", exc_info=True)
        assistant_chain = None 
    yield
    logger.info("Завершение работы API.")


app = FastAPI(
    title="Clinic Assistant API (Module)",
    description="API для взаимодействия с ассистентом (использует matrixai_module)",
    version="3.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


def get_assistant() -> Runnable:
    if assistant_chain is None:
        logger.error("Запрос к API, но ассистент не инициализирован.")
        raise HTTPException(status_code=503, detail="Ассистент не инициализирован.")
    return assistant_chain

def generate_user_id() -> str:
    return f"user_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"



@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_model=MessageResponse, tags=["assistant"])
async def ask_assistant(
    request: MessageRequest,
    agent: Runnable = Depends(get_assistant)
):
    user_id = request.user_id if request.user_id else generate_user_id()
    reset = request.reset_session
    logger.info(f"Получен запрос от {user_id}: {request.message[:50]}... Reset: {reset}")

    if reset:
        cleared = clear_session_history(user_id) 
        logger.info(f"Запрос на сброс сессии для {user_id}. Сессия удалена: {cleared}")

    try:
        start_time = time.time()
        response_data = await agent.ainvoke(
            {"input": request.message},
            config={"configurable": {"session_id": user_id}}
        )
        end_time = time.time()
        logger.info(f"Агент ответил для {user_id} за {end_time - start_time:.2f} сек.")

        if isinstance(response_data, str):
            response_text = response_data
        else:
            logger.warning(f"Агент вернул тип {type(response_data)}. Преобразуем в строку.")
            response_text = str(response_data)

        logger.info(f"Ответ для {user_id}: {response_text[:50]}...")
        return MessageResponse(response=response_text, user_id=user_id)

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке запроса для {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера.")

@app.post("/reset_session", tags=["session"])
async def reset_session_endpoint(
    user_id_body: Dict[str, str] = Body(...)
):
    user_id = user_id_body.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="'user_id' обязателен.")

    logger.info(f"Запрос на сброс сессии для {user_id}")
    if clear_session_history(user_id): 
        return {"message": f"Сессия для {user_id} сброшена."}
    else:
        raise HTTPException(status_code=404, detail=f"Сессия для {user_id} не найдена.")

@app.get("/health", tags=["health"])
async def health_check():
    agent_ok = assistant_chain is not None
    active_sessions = -1
    try:
        active_sessions = get_active_session_count() 
    except Exception as e:
        logger.error(f"Ошибка получения кол-ва сессий: {e}")
    return { "status": "ok" if agent_ok else "error", "agent_initialized": agent_ok, "active_sessions_in_memory": active_sessions }

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