import logging
import time
import uvicorn
import atexit
from typing import Dict, Optional, List, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from matrixai import initialize_clinic_assistant
import os
import json
import uuid
from datetime import datetime, timedelta


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("clinic_api")


user_sessions = {}


class MessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    reset_session: bool = False
    prompt_type: Optional[str] = None
    prompt_text: Optional[str] = None


class MessageResponse(BaseModel):
    response: str
    user_id: str


clinic_agent = None
is_existing_index = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global clinic_agent, is_existing_index
    try:
        logger.info("Инициализация ассистента клиники...")
        clinic_agent, is_existing_index = initialize_clinic_assistant()
        logger.info(f"Ассистент успешно инициализирован. Использует существующий индекс: {is_existing_index}")
    except Exception as e:
        logger.critical(f"Ошибка инициализации ассистента: {e}", exc_info=True)
        raise
    
    yield
    
    
    logger.info("Завершение работы API, очистка ресурсов...")
    if clinic_agent:
        try:
            clinic_agent.cleanup(delete_assistant=True)
            logger.info("Ресурсы ассистента очищены")
        except Exception as e:
            logger.error(f"Ошибка при очистке ресурсов ассистента: {e}", exc_info=True)

app = FastAPI(
    title="Med YU Med Clinic Assistant API",
    description="API для взаимодействия с ассистентом медицинской клиники Med YU Med",
    version="1.0.0",
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


def get_clinic_agent():
    if clinic_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Ассистент не инициализирован. Пожалуйста, попробуйте позже."
        )
    return clinic_agent


def generate_user_id():
    return f"user_{int(time.time() * 1000)}"


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_model=MessageResponse, tags=["assistant"])
async def ask_assistant(
    request: MessageRequest,
    background_tasks: BackgroundTasks,
    agent=Depends(get_clinic_agent)
):
    user_id = request.user_id if request.user_id else generate_user_id()
    reset = request.reset_session
    
    logger.info(f"Получен запрос от пользователя {user_id}: {request.message[:50]}... Reset: {reset}")
    

    if reset and user_id in user_sessions:
        logger.info(f"Сброс сессии для пользователя {user_id}")
        try:
            old_thread = user_sessions.pop(user_id)
            background_tasks.add_task(cleanup_thread, old_thread)
        except Exception as e:
            logger.error(f"Ошибка при сбросе сессии пользователя {user_id}: {e}")
    
  
    user_thread = None
    if user_id in user_sessions:
        user_thread = user_sessions[user_id]
    
    try:
        
        response = agent(request.message, thread=user_thread)
        
        
        if user_id not in user_sessions and agent.thread:
            user_sessions[user_id] = agent.thread
        
        logger.info(f"Ответ для пользователя {user_id}: {response[:50]}...")
        return MessageResponse(response=response, user_id=user_id)
    
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса от пользователя {user_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Внутренняя ошибка сервера при обработке запроса: {str(e)}"
        )


async def cleanup_thread(thread):
    try:
        if thread and hasattr(thread, 'delete'):
            thread.delete()
            logger.info(f"Поток {thread.id} успешно удален")
    except Exception as e:
        logger.error(f"Ошибка при удалении потока {getattr(thread, 'id', 'unknown')}: {e}")


@app.post("/reset_session", tags=["session"])
async def reset_session(
    user_id: str,
    background_tasks: BackgroundTasks
):
    if user_id in user_sessions:
        old_thread = user_sessions.pop(user_id)
        background_tasks.add_task(cleanup_thread, old_thread)
        logger.info(f"Сессия сброшена для пользователя {user_id}")
        return {"message": f"Сессия сброшена для пользователя {user_id}"}
    return {"message": f"Сессия для пользователя {user_id} не найдена"}


@app.get("/health", tags=["health"])
async def health_check():

    return {
        "status": "ok",
        "agent_initialized": clinic_agent is not None,
        "active_sessions": len(user_sessions)
    }


async def cleanup_inactive_sessions():
    while True:
        await asyncio.sleep(3600)  
        try:
            
            logger.info(f"Активных сессий: {len(user_sessions)}")
        except Exception as e:
            logger.error(f"Ошибка при очистке неактивных сессий: {e}")


def read_logs(max_lines=100):
    logs = []
    try:
        with open('api.log', 'r', encoding='utf-8') as log_file:
            lines = log_file.readlines()[-max_lines:]
            
            for line in lines:
                try:
                    parts = line.split(' [', 1)
                    if len(parts) >= 2:
                        time_str = parts[0].split(',')[0]
                        level_and_message = parts[1].split('] ', 1)
                        
                        if len(level_and_message) >= 2:
                            level = level_and_message[0]
                            message = level_and_message[1].strip()
                            
                            logs.append({
                                "time": time_str.split(' ')[1],
                                "level": level,
                                "message": message
                            })
                except Exception as e:
                    logs.append({
                        "time": "--:--:--",
                        "level": "INFO",
                        "message": line.strip()
                    })
        
        return logs
    except FileNotFoundError:
        return [
            {"time": datetime.now().strftime("%H:%M:%S"), "level": "INFO", "message": "Файл логов не найден"},
            {"time": datetime.now().strftime("%H:%M:%S"), "level": "WARNING", "message": "Это тестовый лог для демонстрации функциональности"},
            {"time": datetime.now().strftime("%H:%M:%S"), "level": "ERROR", "message": "Тестовая ошибка"}
        ]
    except Exception as e:
        return [
            {"time": datetime.now().strftime("%H:%M:%S"), "level": "ERROR", "message": f"Ошибка чтения логов: {str(e)}"}
        ]


@app.get("/logs")
async def get_logs(max_lines: int = 100):
    logs = read_logs(max_lines)
    return {"logs": logs}


if __name__ == "__main__":
    logger.info("Запуск сервера FastAPI")
    uvicorn.run(app, host="0.0.0.0", port=8001)
