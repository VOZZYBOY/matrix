import logging
import time
import uvicorn
import atexit
from typing import Dict, Optional, List, Any
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
from contextlib import asynccontextmanager
from matrixai import initialize_clinic_assistant


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
    reset_session: Optional[bool] = False


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
async def root():
    return {"message": "API ассистента клиники Med YU Med работает"}


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
    if clinic_agent is None:
        raise HTTPException(
            status_code=503,
            detail="Ассистент не инициализирован"
        )
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

if __name__ == "__main__":
    logger.info("Запуск сервера FastAPI")
    uvicorn.run(app, host="0.0.0.0", port=8001)
