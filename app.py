import logging
import uvicorn
import os
import uuid
import time
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Union
import datetime
import base64
from fastapi import FastAPI, HTTPException, Depends, Request, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, validator
from langchain_core.runnables import Runnable

from client_data_service import get_client_context_for_agent

try:
    import tenant_config_manager
except ImportError as e:
    logging.critical(f"Не удалось импортировать tenant_config_manager: {e}", exc_info=True)
    tenant_config_manager = None
try:
    from redis_history import clear_history as redis_clear_history, get_history as redis_get_history
except ImportError as e:
    logging.critical(f"Не удалось импортировать redis_history: {e}", exc_info=True)
    redis_clear_history = None
    redis_get_history = None

try:
    from matrixai import agent_with_history, trigger_reindex_tenant_async
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
class AskRequest(BaseModel):
    user_id: str
    message: str
    session_id: Optional[str] = None 
    tenant_id: Optional[str] = None 
    reset_session: bool = False

class TenantSettings(BaseModel):
    prompt_addition: Optional[str] = None
    clinic_info_docs: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Список словарей, описывающих документы с общей информацией для тенанта. Каждый словарь должен содержать 'page_content' (str) и 'metadata' (dict)."
    )

class SetTenantSettingsRequest(BaseModel):
    tenant_id: str
    settings: TenantSettings

class GetTenantSettingsResponse(BaseModel):
    prompt_addition: Optional[str] = None
    clinic_info_docs: Optional[List[Dict[str, Any]]] = None
    last_modified_general: Optional[str] = None 

class ImageData(BaseModel):
    """Модель для передачи изображений в мультимодальном формате"""
    type: str = Field(default="image", description="Тип контента - всегда 'image'")
    source_type: str = Field(description="Тип источника: 'base64' или 'url'")
    data: Optional[str] = Field(default=None, description="Base64 данные изображения (для source_type='base64')")
    url: Optional[str] = Field(default=None, description="URL изображения (для source_type='url')")
    mime_type: Optional[str] = Field(default="image/jpeg", description="MIME тип изображения")
    
    @validator("data")
    def validate_base64_data(cls, v, values):
        if values.get("source_type") == "base64" and not v:
            raise ValueError("Для source_type='base64' поле 'data' обязательно")
        return v
    
    @validator("url")
    def validate_url_data(cls, v, values):
        if values.get("source_type") == "url" and not v:
            raise ValueError("Для source_type='url' поле 'url' обязательно")
        return v

class MessageRequest(BaseModel):
    message: str
    user_id: Optional[str] = None
    reset_session: bool = False
    tenant_id: str 
    phone_number: Optional[str] = None
    client_api_token: Optional[str] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJJZCI6ImUwOTA0YWU2LWQ4NzMtNDA5MS1iNjFjLTQyODlhYTI3ZjY2ZSIsIk5hbWUiOiJhZG1pbiIsIlN1cm5hbWUiOiJhZG1pbiIsIlJvbGVOYW1lIjoi0JDQtNC80LjQvdC40YHRgtGA0LDRgtC-0YAiLCJFbWFpbCI6ImhhbGFsb2xzdW5AbWFpbC5jb20iLCJUZW5hbnRJZCI6Im1lZHl1bWVkLjIwMjMtMDQtMjQiLCJSb2xlSWQiOiJyb2xlMiIsIlBob3RvVXJsIjoiaHR0cHM6Ly9jZG4ubWF0cml4Y3JtLnJ1L21lZHl1bWVkLjIwMjMtMDQtMjQvOTFhMGZhYzAtYmQ1Zi00M2RkLThmNTAtNTc5YmI0NjEwZGUyLmpwZWciLCJDaXR5SWQiOiIyIiwiZXhwIjoxNzg3MTI4MDY4LCJpc3MiOiJodHRwczovL2xvY2FsaG9zdDo3MDk1IiwiYXVkIjoiaHR0cHM6Ly9sb2NhbGhvc3Q6NzA5NSJ9.v9fQ_6Fepbov-BYZIg5RgcTluQVgWZSaDDK71OIsOjE"
    images: Optional[List[ImageData]] = Field(default=None, description="Список изображений для мультимодальной обработки")
class MessageResponse(BaseModel):
    response: str
    user_id: str

class BookAppointmentArgs(BaseModel):
    phone_number: str
    service_name: str
    employee_name: str
    filial_name: str
    date_of_record: str
    start_time: str
    end_time: str
    duration_of_time: int


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
    version="3.2.0", 
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

async def prepare_multimodal_input(request: MessageRequest, client_context_str: Optional[str]) -> Union[str, List[Dict[str, Any]]]:
    """
    Подготавливает входные данные для мультимодального агента.
    
    Возвращает:
    - Обычную строку, если изображений нет
    - Список блоков контента в мультимодальном формате LangChain, если есть изображения
    """
    
    # Формируем основное текстовое сообщение
    text_message = request.message
    if client_context_str:
        text_message = f"{client_context_str}\n\nИсходное сообщение: {request.message}"
        logger.info(f"Добавлен контекст о клиенте. Новое сообщение для агента (начало): {text_message[:150]}...")
    else:
        logger.info("Контекст о клиенте не был сформирован.")
    
    # Если нет изображений, возвращаем обычную строку
    if not request.images:
        return text_message
    
    # Если есть изображения, формируем мультимодальный формат
    logger.info(f"Обрабатываем мультимодальный запрос с {len(request.images)} изображениями")
    
    content_blocks = []
    
    # Добавляем текстовый блок
    content_blocks.append({
        "type": "text",
        "text": text_message
    })
    
    # Добавляем блоки изображений
    for i, image in enumerate(request.images):
        try:
            if image.source_type == "base64":
                # Валидируем base64 данные
                if not image.data:
                    logger.warning(f"Изображение {i+1}: отсутствуют base64 данные")
                    continue
                    
                # Проверяем, что это валидный base64
                try:
                    base64.b64decode(image.data, validate=True)
                except Exception as e:
                    logger.error(f"Изображение {i+1}: невалидные base64 данные: {e}")
                    continue
                
                image_block = {
                    "type": "image",
                    "source_type": "base64",
                    "data": image.data,
                    "mime_type": image.mime_type or "image/jpeg"
                }
                
            elif image.source_type == "url":
                if not image.url:
                    logger.warning(f"Изображение {i+1}: отсутствует URL")
                    continue
                    
                image_block = {
                    "type": "image", 
                    "source_type": "url",
                    "url": image.url
                }
                
            else:
                logger.error(f"Изображение {i+1}: неподдерживаемый source_type: {image.source_type}")
                continue
                
            content_blocks.append(image_block)
            logger.info(f"Добавлено изображение {i+1} ({image.source_type})")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке изображения {i+1}: {e}")
            continue
    
    # Если все изображения оказались невалидными, возвращаем только текст
    if len(content_blocks) == 1:
        logger.warning("Все изображения оказались невалидными, возвращаем только текст")
        return text_message
    
    logger.info(f"Подготовлен мультимодальный ввод с {len(content_blocks)} блоками контента")
    return content_blocks


@app.get("/")
async def read_root(request: Request):
    logger.info("Запрос корневой страницы (Admin+Chat)")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_model=MessageResponse, tags=["assistant"])
async def ask_assistant(
    request: MessageRequest,
    agent_dependency: Runnable = Depends(get_agent)
):
    user_id_for_crm_visit_history = request.user_id 
    reset = request.reset_session
    tenant_id = request.tenant_id 

    if not tenant_id:
        logger.error(f"Получен запрос без tenant_id.")
        raise HTTPException(status_code=400, detail="Параметр 'tenant_id' обязателен.")

    # Формируем базовый user_id
    base_user_id = f"{tenant_id}_{request.phone_number}" if request.phone_number else f"{tenant_id}_{generate_user_id()}"
    
    # Получаем текущий номер сессии (без инкремента)
    session_counter = 1  # По умолчанию первая сессия
    if redis_clear_history:
        try:
            from redis_history import get_current_session_number
            session_counter = get_current_session_number(tenant_id, base_user_id)
        except Exception as e:
            logger.warning(f"Не удалось получить счетчик сессий: {e}")
            session_counter = 1
    
    # Формируем финальный user_id с номером сессии (всегда с суффиксом)
    user_id_for_agent_chat_history = f"{base_user_id}_s{session_counter}"
    
    logger.info(f"Получен запрос от tenant '{tenant_id}', НАШ user_chat_history_id '{user_id_for_agent_chat_history}' (оригинальный request.user_id: '{request.user_id}', phone '{request.phone_number}', session: {session_counter}): {request.message[:50]}... Reset: {reset}")

    if reset:
        if redis_clear_history:
            # Очищаем старую историю
            cleared = redis_clear_history(tenant_id=tenant_id, user_id=user_id_for_agent_chat_history)
            
            # Инкрементируем счетчик сессий с помощью встроенной Redis функции
            try:
                from redis_history import get_next_session_number
                new_session_counter = get_next_session_number(tenant_id, base_user_id)
                # Обновляем user_id для ответа с новым номером сессии
                user_id_for_agent_chat_history = f"{base_user_id}_s{new_session_counter}"
                logger.info(f"Создана новая сессия {new_session_counter} для {base_user_id}")
            except Exception as e:
                logger.error(f"Ошибка при инкременте счетчика сессий: {e}")
            
            logger.info(f"Запрос на сброс сессии для tenant '{tenant_id}', user_id '{user_id_for_agent_chat_history}'. Сессия удалена: {cleared}")
            return MessageResponse(response="История чата была очищена. Начинаем новую сессию.", user_id=user_id_for_agent_chat_history)
        else:
             logger.error(f"Функция redis_clear_history не доступна для tenant '{tenant_id}', user_id '{user_id_for_agent_chat_history}'")
             return MessageResponse(response="Запрос на сброс сессии получен, но функция очистки истории в Redis недоступна. Сообщение не было обработано.", user_id=user_id_for_agent_chat_history)

    try:
        start_time = time.time()
        composite_session_id = f"{tenant_id}:{user_id_for_agent_chat_history}"
        logger.debug(f"Создан composite_session_id для истории чата ассистента: {composite_session_id}")

        client_context_str = await get_client_context_for_agent(
            phone_number=request.phone_number, 
            client_api_token=request.client_api_token,
            user_id_for_crm_history=user_id_for_crm_visit_history, 
            visit_history_display_limit=15,
            visit_history_analysis_limit=100,
            frequent_visit_threshold=3      
        )

        # Формируем мультимодальное сообщение для агента
        multimodal_input = await prepare_multimodal_input(request, client_context_str)

        response_data = await agent_dependency.ainvoke(
            {"input": multimodal_input}, 
            config={
                "configurable": {
                    "session_id": composite_session_id,
                    "user_id": user_id_for_agent_chat_history,
                    "tenant_id": tenant_id,
                    "client_api_token": request.client_api_token,
                    "phone_number": request.phone_number
                 }
            }
        )
        end_time = time.time()
        logger.info(f"Агент ответил для tenant '{tenant_id}', user '{user_id_for_agent_chat_history}' за {end_time - start_time:.2f} сек.")

        if isinstance(response_data, str):
            response_text = response_data
        else:
            logger.warning(f"Агент вернул тип {type(response_data)}. Преобразуем в строку.")
            response_text = str(response_data)

        logger.info(f"Ответ для {user_id_for_agent_chat_history}: {response_text[:50]}...")
        return MessageResponse(response=response_text, user_id=user_id_for_agent_chat_history)

    except Exception as e:
        logger.error(f"Критическая ошибка при обработке запроса для {user_id_for_agent_chat_history}: {e}", exc_info=True)
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
    needs_reindex = False

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
        else:
            needs_reindex = True

    response_message = f"Настройки для тенанта '{tenant_id}' успешно сохранены/обновлены."
    status_code = 200

    if not success_general or not success_clinic_info:
        error_details = []
        if not success_general: error_details.append("общие настройки")
        if not success_clinic_info: error_details.append("clinic_info_docs")
        if not success_clinic_info: needs_reindex = False 
        response_message = f"Не удалось сохранить части настроек ({', '.join(error_details)}) для тенанта '{tenant_id}'."
        status_code = 500
        if not success_general and not success_clinic_info:
             raise HTTPException(status_code=500, detail=response_message)
    
    if needs_reindex:
        logger.info(f"Изменения в clinic_info_docs для тенанта '{tenant_id}' требуют переиндексации. Запускаем фоновую задачу...")
        import asyncio
        asyncio.create_task(trigger_reindex_tenant_async(tenant_id))
        response_message += " Начата фоновая переиндексация данных клиники."

    if status_code == 500 and (success_general or success_clinic_info):
        return JSONResponse(content={"message": response_message}, status_code=200)

    if status_code == 200:
        return {"message": response_message}
    else:
        raise HTTPException(status_code=status_code, detail=response_message)

@app.get("/tenant_settings/{tenant_id}", response_model=GetTenantSettingsResponse, tags=["tenant_config"])
async def get_tenant_settings(tenant_id: str):
    if not tenant_config_manager:
        raise HTTPException(status_code=503, detail="Менеджер конфигураций тенантов недоступен.")

    logger.info(f"Запрос настроек для тенанта '{tenant_id}'")
    try:
        
        settings = tenant_config_manager.load_tenant_settings(tenant_id)
        
        clinic_info = tenant_config_manager.load_tenant_clinic_info(tenant_id)
        
        general_mod_time = tenant_config_manager.get_settings_file_mtime(tenant_id)
        clinic_info_mod_time = tenant_config_manager.get_clinic_info_file_mtime(tenant_id)

        
        general_mod_time_str = datetime.datetime.fromtimestamp(general_mod_time).isoformat() if general_mod_time else None
        clinic_info_mod_time_str = datetime.datetime.fromtimestamp(clinic_info_mod_time).isoformat() if clinic_info_mod_time else None

        

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

@app.get("/history/{tenant_id}/{user_id}", tags=["history"], response_model=List[Dict[str, Any]])
async def get_user_chat_history(tenant_id: str, user_id: str, limit: Optional[int] = 50):
    """
    Получает историю чата для указанного tenant_id и user_id.
    По умолчанию возвращает последние 50 сообщений.
    """
    logger.info(f"Запрос истории чата для tenant: '{tenant_id}', user: '{user_id}', limit: {limit}")
    if not redis_get_history:
        logger.error("Функция redis_get_history не доступна.")
        raise HTTPException(status_code=503, detail="Функциональность истории чата недоступна.")

    try:
        history_data = redis_get_history(tenant_id=tenant_id, user_id=user_id, limit=limit if limit is not None else 50) # Убедимся что limit не None
        return history_data 
    except ValueError as ve: 
        logger.error(f"Ошибка формирования ключа для истории: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при получении истории для tenant: '{tenant_id}', user: '{user_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при получении истории чата.")

class ClearHistoryRequest(BaseModel):
    tenant_id: str
    user_id: Optional[str] = None 
    phone_number: Optional[str] = None 

@app.post("/clear_history", tags=["history"], status_code=200)
async def clear_user_chat_history_endpoint(request: ClearHistoryRequest):
    tenant_id = request.tenant_id
    user_id_to_clear = None

    if not tenant_id:
        raise HTTPException(status_code=400, detail="Параметр 'tenant_id' обязателен.")

    if request.phone_number:
        user_id_to_clear = f"{tenant_id}_{request.phone_number}"
    
    if not user_id_to_clear:
        raise HTTPException(status_code=400, detail="Необходимо предоставить 'phone_number' для идентификации сессии для очистки.")

    logger.info(f"Запрос на очистку истории для tenant: '{tenant_id}', user_id_to_clear (сформирован по номеру): '{user_id_to_clear}'")
    
    if not redis_clear_history:
        logger.error("Функция redis_clear_history не доступна.")
        raise HTTPException(status_code=503, detail="Функциональность очистки истории чата недоступна.")

    try:
        cleared = redis_clear_history(tenant_id=tenant_id, user_id=user_id_to_clear)
        if cleared:
            logger.info(f"История для tenant '{tenant_id}', user '{user_id_to_clear}' успешно очищена.")
            return {"message": f"История для пользователя {user_id_to_clear} в тенанте {tenant_id} была очищена."}
        else:
            logger.warning(f"Очистка истории для tenant '{tenant_id}', user '{user_id_to_clear}' не удалась (возможно, истории и не было или функция вернула False).")
            return {"message": f"Запрос на очистку истории для {user_id_to_clear} в тенанте {tenant_id} обработан. Возможно, истории не существовало."}
    except ValueError as ve: 
        logger.error(f"Ошибка формирования ключа для очистки истории: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при очистке истории для tenant: '{tenant_id}', user '{user_id_to_clear}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера при очистке истории чата.")

if __name__ == "__main__":
    app_host = os.getenv("APP_HOST", "0.0.0.0")
    app_port = int(os.getenv("APP_PORT", 8001))
    log_level = os.getenv("APP_LOG_LEVEL", "info").lower()
    logger.info(f"Starting FastAPI server via uvicorn on {app_host}:{app_port}")
    uvicorn.run("app:app", host=app_host, port=app_port, log_level=log_level, reload=True)