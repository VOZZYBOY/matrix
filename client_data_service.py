import httpx
import logging
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
import datetime
from clinic_index import get_id_by_name
import json
import aiohttp # Ensure aiohttp is imported

logger = logging.getLogger(__name__)

CLIENT_API_BASE_URL = "https://back.matrixcrm.ru/api/v1"

# --- New API endpoint for getMultipleData ---
async def get_multiple_data_from_api(
    api_token: str,
    filial_id: Optional[str] = None,
    employee_id: Optional[str] = None,
    service_id: Optional[str] = None,
    tenant_id: Optional[str] = None # Optional, primarily for logging context
) -> Optional[List[Dict[str, Any]]]:
    """
    Вызывает эндпоинт /api/v1/AI/getMultipleData для получения связанных данных.
    
    ВАЖНО: Функция поддерживает 6 режимов работы в зависимости от переданных параметров:
    
    1. **Только filial_id** → Возвращает всех сотрудников, которые работают в этом филиале,
       с их услугами и категориями.
       
    2. **Только service_id** → Возвращает всех сотрудников, которые оказывают эту услугу,
       во всех филиалах где она доступна.
       
    3. **Только employee_id** → Возвращает все услуги конкретного сотрудника
       во всех филиалах где он работает.
       
    4. **filial_id + employee_id** → Возвращает услуги конкретного сотрудника
       только в указанном филиале.
       
    5. **filial_id + service_id** → Возвращает всех сотрудников, которые оказывают
       конкретную услугу в конкретном филиале.
       
    6. **service_id + employee_id** → Возвращает информацию о том, в каких филиалах
       конкретный сотрудник оказывает конкретную услугу.
    
    Args:
        api_token: Bearer-токен для авторизации (обязательно)
        filial_id: ID филиала (опционально)
        employee_id: ID сотрудника (опционально) 
        service_id: ID услуги (опционально)
        tenant_id: ID тенанта для логирования (опционально)
    
    Returns:
        Список словарей с данными или None при ошибке/отсутствии данных.
        Каждый элемент содержит: employeeId, employeeFullname, serviceId, serviceName,
        categoryId, categoryName, filialId, filialName
    """
    if not api_token:
        logger.error(f"[Tenant: {tenant_id}] api_token не предоставлен для вызова getMultipleData.")
        return None

    url = f"{CLIENT_API_BASE_URL}/AI/getMultipleData"
    
    # Подготавливаем query параметры (отправляем только непустые значения)
    params = {}
    if filial_id:
        params["filialId"] = filial_id
    if employee_id:
        params["employeeId"] = employee_id
    if service_id:
        params["serviceId"] = service_id

    # Проверяем, заданы ли какие-то фильтры
    has_filters = bool(filial_id or employee_id or service_id)
    
    if not has_filters:
         logger.warning(f"[Tenant: {tenant_id}] getMultipleData вызван без фильтров (filial_id, employee_id, service_id). Будет возвращен весь список. URL: {url}")
    else:
         logger.info(f"[Tenant: {tenant_id}] Вызов getMultipleData с query параметрами: {params}. URL: {url}")


    headers = {
        "Authorization": f"Bearer {api_token}",
        "accept": "*/*"
    }

    try:
        async with httpx.AsyncClient(verify=False) as client: # verify=False для отключения проверки SSL
            # Логируем детали запроса
            logger.info(f"[Tenant: {tenant_id}] Отправка GET запроса с query параметрами на {url}")
            logger.info(f"[Tenant: {tenant_id}] Headers: {headers}")
            logger.info(f"[Tenant: {tenant_id}] Query params: {params}")
            
            # Используем обычный GET запрос с query параметрами
            response = await client.get(url, params=params, headers=headers, timeout=20.0)
            
            # Логируем ответ
            logger.info(f"[Tenant: {tenant_id}] Response status: {response.status_code}")
            logger.info(f"[Tenant: {tenant_id}] Response headers: {dict(response.headers)}")
            logger.info(f"[Tenant: {tenant_id}] Response body: {response.text[:500]}...")
            
            response.raise_for_status() # Поднимет исключение для 4xx/5xx статусов

            response_data = response.json()
            
            # Ожидаемый формат ответа: {"code": 200, "data": [...], "message": "..."}
            if response_data.get("code") == 200 and isinstance(response_data.get("data"), list):
                logger.info(f"[Tenant: {tenant_id}] Успешно получен ответ от getMultipleData. Найдено записей: {len(response_data['data'])}")
                return response_data["data"]
            else:
                logger.warning(f"[Tenant: {tenant_id}] API getMultipleData вернуло код {response_data.get('code')} или данные не в формате списка. Ответ: {response_data.get('message', 'Без сообщения')}. Полный ответ: {str(response_data)[:500]}...")
                return None # Возвращаем None, если ответ не соответствует ожидаемому успешному формату

    except httpx.HTTPStatusError as e:
        error_content = e.response.text
        logger.error(f"[Tenant: {tenant_id}] Ошибка HTTP Status {e.response.status_code} при вызове getMultipleData: {error_content}", exc_info=True)
        return None
    except httpx.RequestError as e:
        logger.error(f"[Tenant: {tenant_id}] Ошибка запроса при вызове getMultipleData: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Неизвестная ошибка при вызове getMultipleData: {e}", exc_info=True)
        return None

# --- End new API endpoint function ---


class ClientApiError(BaseModel):
    code: int
    message: Optional[str] = None
    errors: Optional[List[str]] = None

class ClientDetailsData(BaseModel):
    id: str 
    fullName: Optional[str] = None


class ElasticByPhoneResponse(BaseModel):
    code: int
    data: Optional[List[ClientDetailsData]] = None
    message: Optional[str] = None 

class ServiceInRecord(BaseModel):
    serviceName: Optional[str] = None
    serviceId: Optional[str] = None

class VisitRecordData(BaseModel):
    id: str
    filialName: Optional[str] = None
    startTime: Optional[str] = None 
    toEmployeeId: Optional[str] = None
    employeeName: Optional[str] = None
    employeeSurname: Optional[str] = None
    employeeFatherName: Optional[str] = None
    servicesList: Optional[List[ServiceInRecord]] = []
    payStatus: Optional[bool] = None
    statusId: Optional[str] = None
    statusName: Optional[str] = None

class ClientRecordsResponse(BaseModel):
    code: int
    count: Optional[int] = None
    data: Optional[List[VisitRecordData]] = None
    message: Optional[str] = None # На случай ошибок от API


# --- Обновленные Pydantic модели для результатов анализа ---
class ServiceVisitSummary(BaseModel):
    service_name: str
    visit_count: int

class SpecialistVisitSummary(BaseModel):
    employee_name: str
    employee_id: Optional[str] = None
    visit_count: int

class FilialVisitSummary(BaseModel):
    filial_name: str
    visit_count: int

class VisitPatternAnalysisResult(BaseModel):
    top_services: List[ServiceVisitSummary] = []
    top_specialists: List[SpecialistVisitSummary] = []
    top_filials: List[FilialVisitSummary] = []
# --- Конец обновленных моделей ---


async def fetch_client_details_by_phone(phone_number: str, api_token: str) -> Optional[ClientDetailsData]:
    """
    Получает ФИО и ID клиента по номеру телефона.
    Возвращает объект ClientDetailsData или None в случае ошибки или если клиент не найден.
    """
    if not phone_number or not api_token:
        return None

    client_api_url = f"{CLIENT_API_BASE_URL}/Client/elastic-by-phone-v2?content={phone_number}"
    headers = {"Authorization": f"Bearer {api_token}"}

    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Запрос данных клиента по номеру {phone_number} к {client_api_url} (Метод: POST)")
            response = await client.get(client_api_url, headers=headers, timeout=10.0)

            if response.status_code == 200:
                response_data = response.json()
                parsed_response = ElasticByPhoneResponse(**response_data)
                
                if parsed_response.code == 200 and parsed_response.data:
                    if parsed_response.data: # Убедимся, что список не пустой
                        client_details = parsed_response.data[0]
                        logger.info(f"Получены данные клиента: ID {client_details.id}, ФИО {client_details.fullName} для номера {phone_number}")
                        return client_details
                    else:
                        logger.warning(f"Ответ API для elasticByPhone не содержит данных (пустой список) для номера {phone_number}.")
                        return None
                else:
                    logger.warning(f"API elasticByPhone вернуло code={parsed_response.code} или отсутствуют данные для номера {phone_number}. Сообщение API: {parsed_response.message}")
                    return None
            else:
                logger.error(f"Ошибка запроса к elasticByPhone для номера {phone_number}. Статус: {response.status_code}, Ответ: {response.text}")
                return None
    except httpx.RequestError as e:
        logger.error(f"Ошибка HTTP запроса при получении данных клиента (elasticByPhone) для {phone_number}: {e}", exc_info=True)
    except Exception as e: # Включая JSONDecodeError и pydantic.ValidationError
        logger.error(f"Непредвиденная ошибка при обработке ответа от elasticByPhone для {phone_number}: {e}", exc_info=True)
    return None

# --- Функция получения истории визитов ---
async def fetch_client_visit_history(client_id: str, api_token: str, internal_fetch_limit: int = 10) -> Optional[Tuple[List[VisitRecordData], List[Dict[str, str]]]]:
    """
    Получает историю записей клиента по его ID.
    internal_fetch_limit: Количество записей, запрашиваемых у API и используемых для анализа.
    """
    if not client_id or not api_token:
        return None

    history_api_url = f"{CLIENT_API_BASE_URL}/Record/getClientRecords?clientId={client_id}" # TODO: Добавить параметры сортировки/лимита, если API их поддерживает
    headers = {"Authorization": f"Bearer {api_token}"}

    try:
        async with httpx.AsyncClient() as client:
            logger.info(f"Запрос истории записей для clientId {client_id} к {history_api_url} (Метод: GET, лимит до {internal_fetch_limit} для анализа)")
            response = await client.get(history_api_url, headers=headers, timeout=15.0)

            if response.status_code == 200:
                response_data = response.json()
                parsed_response = ClientRecordsResponse(**response_data)

                if parsed_response.code == 200 and parsed_response.data is not None:
                    valid_records = [r for r in parsed_response.data if r.startTime]
                    sorted_records = sorted(
                        valid_records,
                        key=lambda r: datetime.datetime.strptime(r.startTime, "%Y-%m-%d %H:%M:%S") if r.startTime else datetime.datetime.min,
                        reverse=True
                    )
                    logger.info(f"Получено {len(sorted_records)} записей для clientId {client_id}. Будет возвращено до {internal_fetch_limit} для анализа.")

                    # Используем глобальную функцию extract_waiting_services, чтобы не дублировать логику.
                    limited_records = sorted_records[:internal_fetch_limit]
                    waiting_services = extract_waiting_services(limited_records)

                    # Возвращаем кортеж: (история, ожидающие услуги)
                    return limited_records, waiting_services
                else:
                    logger.warning(f"API getClientRecords вернуло code={parsed_response.code} или отсутствуют данные для clientId {client_id}. Сообщение API: {parsed_response.message}")
                    return None
            else:
                logger.error(f"Ошибка запроса к getClientRecords для clientId {client_id}. Статус: {response.status_code}, Ответ: {response.text}")
                return None
    except httpx.RequestError as e:
        logger.error(f"Ошибка HTTP запроса при получении истории записей (getClientRecords) для clientId {client_id}: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при обработке ответа от getClientRecords для clientId {client_id}: {e}", exc_info=True)
    return None

def format_visit_history_for_prompt(visit_history: List[VisitRecordData], display_limit: int = 3) -> str:
    """
    Форматирует историю записей в строку для промпта ассистента.
    display_limit: Количество последних записей для отображения в промпте.
    """
    if not visit_history:
        return "История предыдущих записей не найдена или пуста."

    prompt_lines = ["Предыдущие записи клиента (последние до {}):".format(display_limit)]
    
    last_paid_visit_info = None
    
    for record in visit_history:
        if record.payStatus is True:
            try:
                dt_object_paid = datetime.datetime.strptime(record.startTime, "%Y-%m-%d %H:%M:%S") if record.startTime else None
                formatted_dt_paid = dt_object_paid.strftime("%d.%m.%Y %H:%M") if dt_object_paid else record.startTime
                
                emp_parts_paid = []
                if record.employeeSurname: emp_parts_paid.append(record.employeeSurname.strip())
                if record.employeeName: emp_parts_paid.append(record.employeeName.strip())
                if record.employeeFatherName: emp_parts_paid.append(record.employeeFatherName.strip())
                employee_full_name_paid = " ".join(emp_parts_paid) if emp_parts_paid else "Специалист не указан"
                
                service_names_paid = []
                if record.servicesList:
                    for srv_paid in record.servicesList:
                        if srv_paid.serviceName:
                            service_names_paid.append(srv_paid.serviceName.strip())
                services_str_paid = ", ".join(service_names_paid) if service_names_paid else "Услуга не указана"
                filial_name_str_paid = record.filialName.strip() if record.filialName else "Филиал не указан"
                
                last_paid_visit_info = f"Последняя подтвержденная (оплаченная) запись: {formatted_dt_paid} - {services_str_paid} у {employee_full_name_paid} в '{filial_name_str_paid}'."
                break 
            except Exception as e_paid: 
                logger.error(f"Ошибка форматирования последней оплаченной записи {record.id}: {e_paid}", exc_info=True)
                # Не прерываем, просто не будет этой информации

    if last_paid_visit_info:
        prompt_lines.insert(0, last_paid_visit_info) # Вставляем в начало списка
    else:
        prompt_lines.insert(0, "Оплаченных записей в предоставленной истории не найдено.")

    for record in visit_history[:display_limit]:
        try:
            dt_object = None
            if record.startTime:
                try:
                    dt_object = datetime.datetime.strptime(record.startTime, "%Y-%m-%d %H:%M:%S")
                    formatted_dt = dt_object.strftime("%d.%m.%Y %H:%M")
                except ValueError:
                    formatted_dt = record.startTime
            else:
                formatted_dt = "Дата/время не указаны"

        
            emp_parts = []
            if record.employeeSurname: emp_parts.append(record.employeeSurname.strip())
            if record.employeeName: emp_parts.append(record.employeeName.strip())
            if record.employeeFatherName: emp_parts.append(record.employeeFatherName.strip())
            employee_full_name = " ".join(emp_parts) if emp_parts else "Специалист не указан"

            service_names = []
            if record.servicesList:
                for srv in record.servicesList:
                    if srv.serviceName:
                        service_names.append(srv.serviceName.strip())
            
            services_str = ", ".join(service_names) if service_names else "Услуга не указана"
            
            filial_name_str = record.filialName.strip() if record.filialName else "Филиал не указан"

            waiting_statuses = {"imwaiting", "inwaiting"}
            status_info = ""  # сброс по умолчанию
            if record.statusId and record.statusId.lower() in waiting_statuses:
                status_info = " (статус: ожидание клиента)"
            elif record.payStatus is False:
                status_info = " (статус: не оплачено/отменено)"
            elif record.payStatus is True:
                status_info = " (статус: оплачено)"

            prompt_lines.append(f"- {formatted_dt}: {services_str} у {employee_full_name} в филиале '{filial_name_str}'.{status_info}")
        except Exception as e:
            logger.error(f"Ошибка форматирования записи {record.id}: {e}", exc_info=True)
            prompt_lines.append(f"- Не удалось полностью отформатировать запись ID: {record.id}")
            
    return "\n".join(prompt_lines)

# --- Обновленная функция для анализа паттернов визитов ---
def analyze_visit_patterns(
    visit_history: List[VisitRecordData],
    top_n_services: int = 3,
    top_n_specialists: int = 2,
    top_n_filials: int = 0
) -> VisitPatternAnalysisResult:
    """
    Анализирует историю визитов для выявления часто используемых услуг, 
    посещаемых специалистов и филиалов.
    """
    service_counts: Dict[str, int] = {}
    specialist_counts: Dict[str, Dict[str, Any]] = {}
    filial_counts: Dict[str, int] = {}

    if not visit_history:
        return VisitPatternAnalysisResult()

    for record in visit_history:
        # Подсчет услуг
        if record.servicesList:
            for srv in record.servicesList:
                if srv.serviceName:
                    s_name = srv.serviceName.strip()
                    service_counts[s_name] = service_counts.get(s_name, 0) + 1

        # Подсчет специалистов
        emp_id = record.toEmployeeId
        if emp_id:
            emp_parts = []
            if record.employeeSurname: emp_parts.append(record.employeeSurname.strip())
            if record.employeeName: emp_parts.append(record.employeeName.strip())
            if record.employeeFatherName: emp_parts.append(record.employeeFatherName.strip())
            employee_full_name = " ".join(emp_parts) if emp_parts else None
            
            if employee_full_name: 
                if emp_id not in specialist_counts:
                    specialist_counts[emp_id] = {"name": employee_full_name, "count": 0, "id": emp_id}
                specialist_counts[emp_id]["count"] += 1
        
        # Подсчет филиалов
        if record.filialName:
            f_name = record.filialName.strip()
            filial_counts[f_name] = filial_counts.get(f_name, 0) + 1
    
    # Сортировка и выбор топ-N услуг
    sorted_services = sorted(service_counts.items(), key=lambda item: item[1], reverse=True)
    top_services_summary = [
        ServiceVisitSummary(service_name=name, visit_count=count)
        for name, count in sorted_services[:top_n_services] if count > 0
    ]

    # Сортировка и выбор топ-N специалистов
    valid_specialists = [data for data in specialist_counts.values() if data.get("count", 0) > 0]
    sorted_specialists_values = sorted(valid_specialists, key=lambda item: item["count"], reverse=True)
    top_specialists_summary = [
        SpecialistVisitSummary(employee_name=data["name"], employee_id=data["id"], visit_count=data["count"])
        for data in sorted_specialists_values[:top_n_specialists]
    ]

    # Обновленная логика для филиалов: возвращаем все, отсортированные по количеству
    sorted_filials_with_counts = sorted(filial_counts.items(), key=lambda item: item[1], reverse=True)
    all_filials_summary = [
        FilialVisitSummary(filial_name=name, visit_count=count)
        # Если top_n_filials > 0, то слайсим, иначе берем все
        for name, count in (sorted_filials_with_counts[:top_n_filials] if top_n_filials > 0 else sorted_filials_with_counts) if count > 0
    ]

    return VisitPatternAnalysisResult(
        top_services=top_services_summary,
        top_specialists=top_specialists_summary,
        top_filials=all_filials_summary
    )

async def get_client_context_for_agent(
    phone_number: Optional[str], 
    client_api_token: Optional[str],
    user_id_for_crm_history: Optional[str], 
    visit_history_display_limit: int = 5, 
    visit_history_analysis_limit: int = 100,
    frequent_visit_threshold: int = 3, 
    analyze_top_n_services: int = 3,
    analyze_top_n_specialists: int = 2,
    analyze_top_n_filials: int = 0
) -> str:
    """
    Основная функция для получения полного контекста о клиенте для ассистента.
    """
    if not client_api_token:
        logger.warning("client_api_token не предоставлен. Контекст о клиенте не будет сформирован.")
        return "" 

    context_parts = []
    client_identified_by_name = False
    full_visit_history_for_analysis: List[VisitRecordData] = []
    
    client_id_resolved_for_history: Optional[str] = None

    if phone_number:
        client_details = await fetch_client_details_by_phone(phone_number, client_api_token)
        if client_details:
            if client_details.id: 
                client_id_resolved_for_history = client_details.id
                logger.info(f"ID клиента '{client_details.id}' получен по номеру телефона {phone_number} и будет использован для истории посещений.")
            
            if client_details.fullName:
                context_parts.append(f"Клиент идентифицирован как: {client_details.fullName.strip()}.")
                client_identified_by_name = True
            elif client_details.id: 
                 logger.info(f"ФИО клиента по номеру {phone_number} не найдено, но ID '{client_details.id}' был получен.")
            else: 
                logger.info(f"Ни ID, ни ФИО клиента не были получены по номеру {phone_number} из fetch_client_details_by_phone.")
        else: 
            logger.info(f"Клиент по номеру {phone_number} не найден. Попытка использовать fallback ID для истории.")
    
    if not client_id_resolved_for_history and user_id_for_crm_history:
        client_id_resolved_for_history = user_id_for_crm_history
        logger.info(f"ID клиента по номеру телефона не определен/не предоставлен. Используется fallback ID '{user_id_for_crm_history}' для истории посещений.")
    elif not client_id_resolved_for_history and not user_id_for_crm_history:
        logger.info("Ни номер телефона для поиска ID, ни fallback ID для истории не предоставлены.")


    if client_id_resolved_for_history:
        logger.info(f"Запрос истории посещений для ID клиента: {client_id_resolved_for_history} (лимит для анализа: {visit_history_analysis_limit})")
        fetched = await fetch_client_visit_history(client_id_resolved_for_history, client_api_token, internal_fetch_limit=visit_history_analysis_limit)
        if fetched:
            # fetched теперь кортеж (history, waiting_services)
            if isinstance(fetched, tuple):
                fetched_history, waiting_services = fetched
            else:
                fetched_history = fetched
                waiting_services = []

            full_visit_history_for_analysis = fetched_history
            formatted_history_display = format_visit_history_for_prompt(full_visit_history_for_analysis, display_limit=visit_history_display_limit)
            context_parts.append(formatted_history_display)

            # Добавляем информацию о предстоящих визитах со статусом 'Imwaiting'
            if waiting_services:
                upcoming_lines = [f"Запланированы (ожидают подтверждения) услуги:"]
                for item in waiting_services:
                    upcoming_lines.append(f"- {item['serviceName']} (ID услуги: {item['serviceId']}, ID записи: {item['recordId']})")
                context_parts.append("\n".join(upcoming_lines))
        else:
            context_parts.append(f"История предыдущих записей для ID клиента '{client_id_resolved_for_history}' не найдена или пуста.")
            logger.info(f"История записей для ID клиента: {client_id_resolved_for_history} не найдена или пуста.")
    else: 
        if client_identified_by_name: 
            context_parts.append("ФИО клиента было определено, но ID для запроса истории посещений получить не удалось. История не запрашивалась.")
        elif phone_number : 
            context_parts.append(f"Клиент по номеру телефона {phone_number} не найден, ID для истории не определен. История не запрашивалась.")
        else: 
            context_parts.append("ID клиента для запроса истории посещений не был предоставлен (ни по телефону, ни напрямую), поэтому история не запрашивалась.")

    if full_visit_history_for_analysis:
        analysis_results = analyze_visit_patterns(
            full_visit_history_for_analysis,
            top_n_services=analyze_top_n_services,
            top_n_specialists=analyze_top_n_specialists,
            top_n_filials=analyze_top_n_filials
        )
        
        summary_for_prompt = []
        if analysis_results.top_services:
            service_descs = [f"{s.service_name} ({s.visit_count} раз(а))" for s in analysis_results.top_services]
            summary_for_prompt.append(f"Часто пользуется услугами: {', '.join(service_descs)}.")
        
        if analysis_results.top_specialists:
            specialist_descs = []
            for spec_summary in analysis_results.top_specialists:
                desc = f"{spec_summary.employee_name} ({spec_summary.visit_count} раз(а))"
                if spec_summary.visit_count >= frequent_visit_threshold:
                    desc += " [часто]" 
                specialist_descs.append(desc)
            if specialist_descs:
                 summary_for_prompt.append(f"Часто посещает специалистов: {', '.join(specialist_descs)}.")

        if analysis_results.top_filials:
            filial_descs = [f"{f.filial_name} ({f.visit_count} раз(а))" for f in analysis_results.top_filials]
            if filial_descs:
                prefix = "Посещал(а) филиал(ы): " if len(filial_descs) > 1 else "Основной филиал посещений: "
                if len(analysis_results.top_filials) == 1:
                     prefix = "Чаще всего посещал(а) филиал: " 
                summary_for_prompt.append(f"{prefix}{', '.join(filial_descs)}.")

        if summary_for_prompt:
            context_parts.append("\nРезюме предпочтений клиента: " + " ".join(summary_for_prompt))
            logger.info(f"Добавлено резюме предпочтений клиента: Услуги - {len(analysis_results.top_services)}, Специалисты - {len(analysis_results.top_specialists)}, Филиалы - {len(analysis_results.top_filials)}")
    

    if not context_parts:
        logger.info("Не удалось сформировать контекст о клиенте.")
        return ""

    return "\n".join(context_parts)

async def get_free_times_of_employee_by_services(
    tenant_id: str,
    employee_id: str,
    service_ids: list,
    date_time: str,
    filial_id: str,
    lang_id: str = "ru",
    api_url: str = "https://back.matrixcrm.ru/api/v1/AI/getFreeTimesOfEmployeeByChoosenServices",
    api_token: str = None
) -> dict:
    """
    Получить свободные слоты для сотрудника по выбранным услугам.
    Ожидает ID для сотрудника, услуг и филиала.
    Включает retry-логику для обработки нестабильных 500 ошибок.
    """
    import asyncio
    
    payload = {
        "employeeId": employee_id,
        "serviceId": service_ids,
        "dateTime": date_time,
        "tenantId": tenant_id,
        "filialId": filial_id,
        "langId": lang_id
    }
    headers = {"accept": "*/*", "Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
        
    # Retry настройки для борьбы с "магическими" 500 ошибками
    max_retries = 3
    base_delay = 1.0  # Базовая задержка в секундах
    backoff_multiplier = 2.0  # Множитель для экспоненциальной задержки
    
    logger.info(f"Отправка запроса к API getFreeTimesOfEmployeeByChoosenServices: URL={api_url}, Параметры={payload}")
    
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(api_url, json=payload, headers=headers, timeout=30.0)
                
                # Если получили успешный ответ
                if resp.status_code == 200:
                    response_data = resp.json()
                    response_snippet = str(response_data)[:200] + "..." if len(str(response_data)) > 200 else str(response_data)
                    
                    if attempt > 0:
                        logger.info(f"[Retry успех] Получен ответ от API getFreeTimesOfEmployeeByChoosenServices после {attempt} попыток: {response_snippet}")
                    else:
                        logger.info(f"Получен ответ от API getFreeTimesOfEmployeeByChoosenServices: {response_snippet}")
                    
                    return response_data
                
                # Если 500 ошибка и есть попытки для повтора
                elif resp.status_code == 500 and attempt < max_retries:
                    delay = base_delay * (backoff_multiplier ** attempt)
                    logger.warning(f"[Retry {attempt + 1}/{max_retries}] API getFreeTimesOfEmployeeByChoosenServices вернул 500 ошибку. Повтор через {delay}с. Response: {resp.text[:200]}")
                    await asyncio.sleep(delay)
                    continue
                
                # Для других ошибок или если закончились попытки
                else:
                    resp.raise_for_status()
                    
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 500 and attempt < max_retries:
                delay = base_delay * (backoff_multiplier ** attempt)
                logger.warning(f"[Retry {attempt + 1}/{max_retries}] HTTP 500 ошибка в getFreeTimesOfEmployeeByChoosenServices. Повтор через {delay}с. Error: {e.response.text[:200]}")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"HTTP ошибка при вызове API getFreeTimesOfEmployeeByChoosenServices (после {attempt} попыток): {e}", exc_info=True)
                raise
                
        except Exception as e:
            if attempt < max_retries:
                delay = base_delay * (backoff_multiplier ** attempt)
                logger.warning(f"[Retry {attempt + 1}/{max_retries}] Ошибка в getFreeTimesOfEmployeeByChoosenServices. Повтор через {delay}с. Error: {e}")
                await asyncio.sleep(delay)
                continue
            else:
                logger.error(f"Ошибка при вызове API getFreeTimesOfEmployeeByChoosenServices (после {max_retries} попыток): {e}", exc_info=True)
                raise

async def add_record(
    tenant_id: str,
    client_phone_number: str,
    services_payload: List[Dict[str, Any]],
    filial_id: str,
    date_of_record: str,
    start_time: str,
    end_time: str,
    duration_of_time: int,
    to_employee_id: str,
    total_price: float,
    lang_id: str = "ru",
    api_url: str = "https://back.matrixcrm.ru/api/v1/AI/addRecord",
    api_token: Optional[str] = None,
    color_code_record: Optional[str] = None,
    traffic_channel: Optional[int] = None,
    traffic_channel_id: Optional[str] = None
) -> dict:
    """
    Создать запись клиента на одну или несколько услуг.
    """
    api_request_payload = {
        "tenantId": tenant_id,
        "clientPhoneNumber": client_phone_number,
        "services": services_payload,
        "filialId": filial_id,
        "dateOfRecord": date_of_record,
        "startTime": start_time,
        "endTime": end_time,
        "durationOfTime": duration_of_time,
        "toEmployeeId": to_employee_id,
        "langId": lang_id,
        "totalPrice": total_price,
    }

   
    if color_code_record is not None:
        api_request_payload["colorCodeRecord"] = color_code_record
    if traffic_channel is not None:
        api_request_payload["trafficChannel"] = traffic_channel
    if traffic_channel_id is not None:
        api_request_payload["trafficChannelId"] = traffic_channel_id
    
    headers = {"Content-Type": "application/json", "accept": "*/*"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    try:
        async with httpx.AsyncClient(verify=False) as client: # verify=False для отключения проверки SSL
            logger.info(f"Отправка запроса на {api_url} с payload: {api_request_payload}")
            response = await client.post(api_url, json=api_request_payload, headers=headers, timeout=20.0)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"Ответ от API записи: Code {response_data.get('code')}, Data: {str(response_data.get('data', 'N/A'))[:200]}")
            return response_data
    except httpx.HTTPStatusError as e:
        error_content = e.response.text
        logger.error(f"Ошибка HTTP Status {e.response.status_code} при вызове API записи: {error_content}", exc_info=True)
        try:
            return {"code": e.response.status_code, "message": f"Ошибка API: {error_content}"}
        except json.JSONDecodeError:
            return {"code": e.response.status_code, "message": f"Ошибка API (не JSON): {error_content}"}
    except httpx.RequestError as e:
        logger.error(f"Ошибка запроса при вызове API записи: {e}", exc_info=True)
        return {"code": 500, "message": f"Ошибка сети или соединения: {e}"}
    except Exception as e:
        logger.error(f"Неизвестная ошибка при вызове API записи: {e}", exc_info=True)
        return {"code": 500, "message": f"Внутренняя ошибка: {e}"}

# ---------------------------------------------------------------------------
# ФУНКЦИЯ ОБНОВЛЕНИЯ ВРЕМЕНИ СУЩЕСТВУЮЩЕЙ ЗАПИСИ (ПЕРЕНОС)
# ---------------------------------------------------------------------------

async def update_record_time(
    record_id: str,
    date_of_record: str,
    start_time: str,
    end_time: str,
    tenant_id: Optional[str] = None,
    api_url: str = "https://back.matrixcrm.ru/api/v1/AI/updateRecordTime",
    api_token: Optional[str] = None
) -> dict:
    """Обновляет дату/время существующей записи (перенос).

    Args:
        record_id: ID записи, которую требуется перенести.
        date_of_record: Новая дата (YYYY-MM-DD).
        start_time: Новое время начала (HH:MM).
        end_time: Новое время окончания (HH:MM).
        api_url: Полный URL эндпоинта updateRecordTime.
        api_token: Bearer-токен для авторизации (опционально).

    Returns:
        Словарь ответа API.
    """
    payload = {
        "recordId": record_id,
        "dateOfRecord": date_of_record,
        "startTime": start_time,
        "endTime": end_time,
    }
    if tenant_id:
        payload["tenantId"] = tenant_id

    headers = {"Content-Type": "application/json", "accept": "*/*"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    try:
        async with httpx.AsyncClient(verify=False) as client:
            logger.info(f"[updateRecordTime] POST {api_url} payload={payload} headers={ {k: (v[:10] + '...' if k=='Authorization' else v) for k, v in headers.items()} }")
            response = await client.post(api_url, json=payload, headers=headers, timeout=20.0)
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"[updateRecordTime] Response code={response_data.get('code')} data={str(response_data.get('data', ''))[:200]}")
            return response_data
    except httpx.HTTPStatusError as e:
        error_content = e.response.text
        logger.error(f"HTTP {e.response.status_code} from updateRecordTime: {error_content}", exc_info=True)
        return {"code": e.response.status_code, "message": f"Ошибка API: {error_content}"}
    except httpx.RequestError as e:
        logger.error(f"Network error updateRecordTime: {e}", exc_info=True)
        return {"code": 500, "message": f"Ошибка сети или соединения: {e}"}
    except Exception as e:
        logger.error(f"Unknown error updateRecordTime: {e}", exc_info=True)
        return {"code": 500, "message": f"Внутренняя ошибка: {e}"}


async def get_cancel_reasons(
    chain_id: str,
    tenant_id: Optional[str] = None,
    api_url: str = f"{CLIENT_API_BASE_URL}/ReasonRecordCancellation",
    api_token: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Возвращает список причин отмены записи для chain_id.
    Args:
        chain_id: Chain ID (обязателен)
        api_token: Bearer-токен (обязателен)
    """
    if not api_token:
        logger.error(f"[Tenant: {tenant_id}] api_token отсутствует для get_cancel_reasons")
        return None
    if not chain_id:
        logger.error(f"[Tenant: {tenant_id}] chain_id отсутствует для get_cancel_reasons")
        return None
    params = {"chainId": chain_id}
    headers = {"Authorization": f"Bearer {api_token}", "accept": "*/*"}
    try:
        async with httpx.AsyncClient(verify=False) as client:
            logger.info(f"[get_cancel_reasons] GET {api_url} params={params}")
            response = await client.get(api_url, params=params, headers=headers, timeout=20.0)
            response.raise_for_status()
            data = response.json()
            # API может возвращать два формата:
            # 1) { "code": 200, "data": [...] }
            # 2) { "count": N, "data": [...] }
            if "code" in data:
                if data.get("code") == 200:
                    return data.get("data") or []
                logger.warning(f"[get_cancel_reasons] API вернул код {data.get('code')}. Сообщение: {data.get('message')}")
                return None
            else:
                # Нет поля code — считаем формат 2
                if isinstance(data.get("data"), list):
                    return data["data"]
                logger.warning(f"[get_cancel_reasons] Неожиданный формат ответа: {data}")
                return None
    except httpx.RequestError as e:
        logger.error(f"Network error get_cancel_reasons: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unknown error get_cancel_reasons: {e}", exc_info=True)
        return None

async def cancel_record(
    record_id: str,
    chain_id: str,
    canceling_reason: str,
    tenant_id: Optional[str] = None,
    api_url: str = f"{CLIENT_API_BASE_URL}/AI/cancelRecord",
    api_token: Optional[str] = None,
) -> dict:
    """Отменяет запись клиента.
    Args:
        record_id: ID записи
        chain_id: chainId, которому принадлежит запись
        canceling_reason: ID либо текст причины отмены (API принимает строку)
    Returns: ответ API JSON
    """
    payload = {
        "recordId": record_id,
        "chainId": chain_id,
        "cancelingReason": canceling_reason,
    }
    if tenant_id:
        payload["tenantId"] = tenant_id
    headers = {"Content-Type": "application/json", "accept": "*/*"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    try:
        async with httpx.AsyncClient(verify=False) as client:
            logger.info(f"[cancelRecord] POST {api_url} payload={payload} headers={{'Authorization': 'Bearer ***'}}")
            response = await client.post(api_url, json=payload, headers=headers, timeout=20.0)
            response.raise_for_status()
            data = response.json()
            logger.info(f"[cancelRecord] Response code={data.get('code')} data={str(data.get('data', ''))[:100]}")
            return data
    except httpx.HTTPStatusError as e:
        error_content = e.response.text
        logger.error(f"HTTP {e.response.status_code} from cancelRecord: {error_content}", exc_info=True)
        return {"code": e.response.status_code, "message": error_content}
    except httpx.RequestError as e:
        logger.error(f"Network error cancelRecord: {e}", exc_info=True)
        return {"code": 500, "message": str(e)}
    except Exception as e:
        logger.error(f"Unknown error cancelRecord: {e}", exc_info=True)
        return {"code": 500, "message": str(e)}

# ---------------------------------------------------------------------------
# ПРОЧИЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------------------------------------------------------

async def get_filial_id_by_name_api(
    filial_name: str,
    api_token: str,
    tenant_id: str
) -> Optional[str]:
    """Получает ID филиала по его имени через API."""
    if not filial_name or not api_token:
        logger.warning(f"[Tenant: {tenant_id}] Отсутствует filial_name или api_token для get_filial_id_by_name_api.")
        return None

    url = f"{CLIENT_API_BASE_URL}/AI/getFilialIdByName"
    params = {"filialName": filial_name}
    headers = {"Authorization": f"Bearer {api_token}", "accept": "*/*"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    try:
                        json_response = await response.json()
                        if json_response.get("code") == 200 and json_response.get("data") is not None:
                            logger.info(f"[Tenant: {tenant_id}] Успешно получен filial_id для '{filial_name}': {json_response['data']}")
                            return str(json_response["data"])
                        else:
                            logger.warning(f"[Tenant: {tenant_id}] API запрос на получение filial_id для '{filial_name}' вернул код {json_response.get('code')} или null данные. Ответ: {json_response}")
                            return None
                    except aiohttp.ContentTypeError:
                        logger.error(f"[Tenant: {tenant_id}] API запрос на получение filial_id для '{filial_name}' вернул не JSON ответ. Статус: {response.status}, Тело: {await response.text()}")
                        return None
                    except Exception as e:
                        logger.error(f"[Tenant: {tenant_id}] Ошибка парсинга JSON ответа для get_filial_id для '{filial_name}': {e}. Ответ: {await response.text()}", exc_info=True)
                        return None
                else:
                    logger.error(f"[Tenant: {tenant_id}] API запрос на получение filial_id для '{filial_name}' завершился ошибкой со статусом {response.status}. Ответ: {await response.text()}")
                    return None
    except aiohttp.ClientError as e:
        logger.error(f"[Tenant: {tenant_id}] Сетевая ошибка при вызове get_filial_id для '{filial_name}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Непредвиденная ошибка в get_filial_id_by_name_api для '{filial_name}': {e}", exc_info=True)
        return None

async def get_service_id_by_name_api(
    service_name: str,
    filial_name: str,
    api_token: str,
    tenant_id: str
) -> Optional[str]:
    """Получает ID услуги по ее имени и имени филиала через API."""
    if not service_name or not filial_name or not api_token:
        logger.warning(f"[Tenant: {tenant_id}] Отсутствует service_name, filial_name или api_token для get_service_id_by_name_api.")
        return None

    url = f"{CLIENT_API_BASE_URL}/AI/getServiceIdByName"
    params = {"serviceName": service_name, "filialName": filial_name}
    headers = {"Authorization": f"Bearer {api_token}", "accept": "*/*"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    try:
                        json_response = await response.json()
                        if json_response.get("code") == 200 and json_response.get("data") is not None:
                            logger.info(f"[Tenant: {tenant_id}] Успешно получен service_id для '{service_name}' в филиале '{filial_name}': {json_response['data']}")
                            return str(json_response["data"])
                        else:
                            logger.warning(f"[Tenant: {tenant_id}] API запрос на получение service_id для '{service_name}' в филиале '{filial_name}' вернул код {json_response.get('code')} или null данные. Ответ: {json_response}")
                            return None
                    except aiohttp.ContentTypeError:
                        logger.error(f"[Tenant: {tenant_id}] API запрос на получение service_id для '{service_name}' вернул не JSON ответ. Статус: {response.status}, Тело: {await response.text()}")
                        return None
                    except Exception as e:
                        logger.error(f"[Tenant: {tenant_id}] Ошибка парсинга JSON ответа для get_service_id для '{service_name}': {e}. Ответ: {await response.text()}", exc_info=True)
                        return None
                else:
                    logger.error(f"[Tenant: {tenant_id}] API запрос на получение service_id для '{service_name}' в филиале '{filial_name}' завершился ошибкой со статусом {response.status}. Ответ: {await response.text()}")
                    return None
    except aiohttp.ClientError as e:
        logger.error(f"[Tenant: {tenant_id}] Сетевая ошибка при вызове get_service_id для '{service_name}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Непредвиденная ошибка в get_service_id_by_name_api для '{service_name}': {e}", exc_info=True)
        return None


async def get_filial_services_by_categories(
    api_token: str,
    filial_id: str,
    tenant_id: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    Вызывает новый эндпоинт для получения услуг по категориям в конкретном филиале.
    
    Возвращает структуру вида:
    [
        {
            "categoryId": "uuid",
            "categoryName": "название категории",
            "services": [
                {
                    "serviceId": "uuid",
                    "serviceName": "название услуги",
                    "description": "описание",
                    "price": цена,
                    "duration": длительность
                }
            ]
        }
    ]
    
    Args:
        api_token: Bearer-токен для авторизации
        filial_id: ID филиала
        tenant_id: ID тенанта для логирования (опционально)
    
    Returns:
        Список категорий с услугами или None при ошибке
    """
    if not api_token:
        logger.error(f"[Tenant: {tenant_id}] api_token не предоставлен для вызова filial services by categories.")
        return None
    
    if not filial_id:
        logger.error(f"[Tenant: {tenant_id}] filial_id не предоставлен для вызова filial services by categories.")
        return None

    # Правильный URL для нового эндпоинта
    url = f"{CLIENT_API_BASE_URL}/AI/filial/{filial_id}/categories-services"

    headers = {
        "Authorization": f"Bearer {api_token}",
        "accept": "*/*"
    }

    try:
        async with httpx.AsyncClient(verify=False) as client:
            logger.info(f"[Tenant: {tenant_id}] Отправка GET запроса на получение услуг по категориям для филиала {filial_id}")
            logger.info(f"[Tenant: {tenant_id}] URL: {url}")
            
            
            response = await client.get(url, headers=headers, timeout=20.0)
            
            logger.info(f"[Tenant: {tenant_id}] Response status: {response.status_code}")
            logger.info(f"[Tenant: {tenant_id}] Response body: {response.text[:500]}...")
            
            response.raise_for_status()

            response_data = response.json()
            
        
            if response_data.get("code") == 200 and isinstance(response_data.get("data"), list):
                logger.info(f"[Tenant: {tenant_id}] Успешно получены услуги по категориям. Найдено категорий: {len(response_data['data'])}")
                return response_data["data"]
            else:
                logger.warning(f"[Tenant: {tenant_id}] API вернуло код {response_data.get('code')} или данные не в формате списка. Ответ: {response_data}")
                return None

    except httpx.HTTPStatusError as e:
        error_content = e.response.text
        logger.error(f"[Tenant: {tenant_id}] Ошибка HTTP Status {e.response.status_code} при получении услуг по категориям: {error_content}", exc_info=True)
        return None
    except httpx.RequestError as e:
        logger.error(f"[Tenant: {tenant_id}] Ошибка запроса при получении услуг по категориям: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"[Tenant: {tenant_id}] Неизвестная ошибка при получении услуг по категориям: {e}", exc_info=True)
        return None


# --- Вспомогательная публичная функция: можно использовать отдельно ---
def extract_waiting_services(visit_history: List[VisitRecordData]) -> List[Dict[str, str]]:
    """Возвращает список услуг (serviceId, serviceName, recordId) у записей, находящихся в статусе ожидания клиента."""
    waiting_statuses = {"imwaiting", "inwaiting"}
    waiting: List[Dict[str, str]] = []

    for rec in visit_history:
        if rec.statusId and rec.statusId.lower() in waiting_statuses:
            if rec.servicesList:
                for srv in rec.servicesList:
                    srv_id = getattr(srv, "serviceId", None)
                    srv_name = getattr(srv, "serviceName", None)
                    if srv_id and srv_name:
                        waiting.append({
                            "recordId": rec.id,
                            "serviceId": srv_id,
                            "serviceName": srv_name
                        })

    return waiting

