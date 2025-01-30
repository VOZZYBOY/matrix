from fastapi import Form

@app.post("/ask")
async def ask_assistant(
    user_id: str = Form(...),
    question: Optional[str] = Form(None),
    mydtoken: str = Form(...),
    tenant_id: str = Form(...),
    file: UploadFile = File(None),
    # Добавляем параметры для записи
    service_id: Optional[str] = Form(None),  # ID выбранной услуги
    record_date: Optional[str] = Form(None),  # Дата записи
    record_time: Optional[str] = Form(None),  # Время записи
):
    try:

        
        if service_id and record_date and record_time:
            
            record_data = {
                "langId": "ru",  
                "products": [],  
                "client": {
                    "name": "Иван",  # 
                    "surname": "Иванов",
                    "phoneNumber": "1234567890",
                   
                },
                "services": [
                    {
                        "serviceId": service_id,
                        "serviceName": "Услуга",  
                        "countService": 1,
                        "price": 1000,  
                        "salePrice": 1000,
                        "durationService": 60, 
                    }
                ],
                "filialId": "filial_1",  
                "dateOfRecord": record_date,
                "startTime": record_time,
                "endTime": "12:00",  
                "durationOfTime": 60,
                "toEmployeeId": "employee_1",  
                "recordStatusId": "status_1", 
                "comment": "Запись через ассистента",
            }

            await create_record(record_data)

            
            return {"response": "Вы успешно записаны на услугу!"}

       
        response_text = await generate_yandexgpt_response(
            context=context,
            history=conversation_history[user_id]["history"],
            question=input_text
        )

        
        if "записаться" in response_text.lower():
           
            response_text += "\n\nЧтобы записаться, укажите ID услуги, дату и время."

        conversation_history[user_id]["history"].append({
            "user_query": input_text,
            "assistant_response": response_text,
            "search_results": [data_cache[tenant_id]['records'][idx] for idx in top_10_indices]
        })

        return {"response": response_text}

    except Exception as e:
        logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка обработки запроса: {str(e)}")
