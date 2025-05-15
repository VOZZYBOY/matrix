import httpx
import logging
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import datetime
from clinic_index import get_id_by_name

logger = logging.getLogger(__name__)

CLIENT_API_BASE_URL = "https://back.matrixcrm.ru/api/v1"


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

async def fetch_client_visit_history(client_id: str, api_token: str, internal_fetch_limit: int = 10) -> Optional[List[VisitRecordData]]:
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
                    return sorted_records[:internal_fetch_limit] # Возвращаем N записей для анализа
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

            status_info = ""
            if record.payStatus is False:
                status_info = " (статус: не оплачено/отменено)" 
            elif record.payStatus is True:
                status_info = " (статус: оплачено)"
            # Если payStatus is None, ничего не добавляем про статус

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
        fetched_history = await fetch_client_visit_history(client_id_resolved_for_history, client_api_token, internal_fetch_limit=visit_history_analysis_limit)
        if fetched_history:
            full_visit_history_for_analysis = fetched_history
            formatted_history_display = format_visit_history_for_prompt(full_visit_history_for_analysis, display_limit=visit_history_display_limit)
            context_parts.append(formatted_history_display)
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
    """
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
    async with httpx.AsyncClient() as client:
        resp = await client.post(api_url, json=payload, headers=headers, timeout=15.0)
        resp.raise_for_status()
        return resp.json()

async def add_record(
    tenant_id: str,
    phone_number: str,
    service_id: str,
    employee_id: str,
    filial_id: str,
    category_id: str,
    service_original_name: str,
    date_of_record: str,
    start_time: str,
    end_time: str,
    duration_of_time: int,
    lang_id: str = "ru",
    api_url: str = "https://back.matrixcrm.ru/api/v1/AI/addRecord",
    api_token: str = None,
    price: float = 0,
    sale_price: float = 0,
    complex_service_id: str = "",
    color_code_record: str = "",
    total_price: float = 0,
    traffic_channel: int = 0,
    traffic_channel_id: str = ""
) -> dict:
    """
    Создать запись клиента на услугу.
    Ожидает ID для услуги, сотрудника, филиала, категории.
    """
    payload = {
        "langId": lang_id,
        "clientPhoneNumber": phone_number,
        "services": [
            {
                "rowNumber": 0,
                "categoryId": category_id,
                "serviceId": service_id,
                "serviceName": service_original_name,
                "countService": 1,
                "price": price,
                "salePrice": sale_price,
                "complexServiceId": complex_service_id,
                "durationService": duration_of_time
            }
        ],
        "filialId": filial_id,
        "dateOfRecord": date_of_record,
        "startTime": start_time,
        "endTime": end_time,
        "durationOfTime": duration_of_time,
        "colorCodeRecord": color_code_record,
        "toEmployeeId": employee_id,
        "totalPrice": total_price,
        "trafficChannel": traffic_channel,
        "trafficChannelId": traffic_channel_id
    }
    headers = {"accept": "*/*", "Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    async with httpx.AsyncClient() as client:
        resp = await client.post(api_url, json=payload, headers=headers, timeout=15.0)
        resp.raise_for_status()
        return resp.json()