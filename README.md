# Med YU Med - Документация виртуального ассистента

## Обзор
Виртуальный ассистент Med YU Med - это AI-помощник на базе Yandex GPT, использующий технологию RAG (Retrieval-Augmented Generation) для поиска релевантной информации о врачах, услугах и ценах клиники.

## Архитектура ассистента

Ассистент построен на базе следующих ключевых компонентов:

- **YandexGPT API** - для общения ассистента с пользователями
- **RAG-индекс** - векторная база знаний о врачах и услугах клиники
- **Function Calling** - набор функций для выполнения конкретных задач

### Управление промтами

Ассистент поддерживает два типа промтов:

- **Стандартный промт** - базовая инструкция для ассистента, включающая описание задач, ограничений и способов работы с инструментами
- **Индивидуальный промт** - настраиваемый пользователем промт для специфических сценариев использования

Обратите внимание: из-за ограничений Yandex Cloud API, промт не меняется в самом ассистенте, а добавляется к каждому сообщению пользователя перед отправкой в LLM:

```python
# Если передан полный текст промта, используем его
if prompt_type == "custom" and prompt_text:
    context = prompt_text
    modified_message = f"{context}\n\nВопрос пользователя: {message}"
elif prompt_type == "default":
    # Добавляем стандартный промт
    context = default_prompt
    modified_message = f"{context}\n\nВопрос пользователя: {message}"
```

### Function Calling API

Ассистент использует следующие функции для обработки конкретных запросов:

| Функция | Описание | Обязательные параметры |
|---------|----------|------------------------|
| `FindEmployees` | Поиск сотрудников по имени, услуге или филиалу | Нет |
| `GetServicePrice` | Получение цены услуги в конкретном филиале | `service_name` |
| `ListFilials` | Получение списка всех филиалов клиники | Нет |
| `GetEmployeeServices` | Получение списка услуг конкретного врача | `employee_name` |
| `CheckServiceInFilial` | Проверка наличия услуги в конкретном филиале | `service_name`, `filial_name` |
| `CompareServicePriceInFilials` | Сравнение цен на услугу в разных филиалах | `service_name`, `filial_names` (≥2) |

### RAG система

Система использует векторный поиск для извлечения релевантной информации из:
- Описаний врачей и их квалификации
- Описаний услуг и методик

**Ограничение:** Yandex Cloud API позволяет использовать максимум 100 файлов для RAG.

Процесс создания RAG индекса:

```python
def initialize_clinic_assistant():
    rag_index = None
    uploaded_rag_files = []
    
    # Поиск существующего индекса
    all_indexes = sdk.search_indexes.list()
    found_index = next((index for index in all_indexes 
                      if index.name == RAG_INDEX_NAME), None)
                      
    if found_index:
        rag_index = found_index
        logging.warning(f"Найден существующий RAG индекс...")
    else:
        # Создание нового индекса
        rag_chunks = preprocess_json_for_rag(global_clinic_data)
        uploaded_rag_files = upload_rag_chunks(rag_chunks)
        rag_index = create_rag_index(uploaded_rag_files, RAG_INDEX_NAME)
```

## Потоки работы

Для каждого типа промта создается отдельный поток (thread) в Yandex Cloud, что позволяет сохранять разные контексты диалогов. При переключении между типами промтов меняется и используемый поток.

```python
# Создаем структуру для хранения потоков по типам промтов
if user_id not in user_sessions:
    user_sessions[user_id] = {}

# Получаем поток для конкретного типа промта
user_thread = None
if user_id in user_sessions and prompt_type in user_sessions[user_id]:
    user_thread = user_sessions[user_id][prompt_type]
```

## Решение проблем

### Работа с потоками

Основная проблема при работе - возможная ситуация с удаленными потоками. Если поток был удален, но ссылка на него осталась в приложении, это приводит к ошибкам (`ValueError: you can't perform an action on Thread because it is deleted`).

Реализована проверка существования потока перед использованием:

```python
def get_thread(self, thread=None):
    if thread is not None:
        # Проверяем, не был ли поток удален
        try:
            thread.get()  # Проверка существования потока
            return thread
        except ValueError as e:
            if "deleted" in str(e):
                logging.warning(f"Обнаружен удаленный поток: {e}")
                return None
```

### Ограничения RAG

Yandex Cloud ограничивает количество файлов до 100, что может быть недостаточно для больших баз знаний. Решения:

1. Объединение меньших чанков в более крупные
2. Ограничение общего количества файлов до 90
3. Регулярная очистка старых файлов через `cleanup_cloud_files()`

## API Endpoints

Основной сервер предоставляет следующие API:

| Endpoint | Метод | Описание |
|----------|-------|----------|
| `/ask` | POST | Отправка запроса ассистенту |
| `/reset_session` | POST | Сброс сессии пользователя |
| `/health` | GET | Проверка состояния системы |

### Пример запроса к API:

```json
{
  "message": "Какие услуги доступны в филиале Москва-сити?",
  "user_id": "user_1234567890",
  "reset_session": false,
  "prompt_type": "default",
  "prompt_text": null
}
```

## Инициализация и конфигурация

Базовые параметры настраиваются в начале файла `matrixai.py`:

```python
FOLDER_ID = "b1gnq2v60fut60hs9vfb"
API_KEY = "AQVNw5Kg0jXoaateYQWdSr2k8cbst_y4_WcbvZrW"
JSON_DATA_PATH = "base/cleaned_data.json"
MODEL_URI_SHORT = "yandexgpt/rc"
RAG_INDEX_NAME = "clinic_rag_index_v2"
ASSISTANT_NAME = "ClinicAssistant_V2"
```

## Рекомендации по доработке

1. **Улучшение веб-интерфейса**:
   - Добавление аутентификации пользователей
   - Улучшение управления потоками промтов
   
2. **Оптимизация RAG**:
   - Уменьшение количества файлов через более эффективное чанкирование
   - Регулярная очистка устаревших файлов
   
3. **Расширение функциональности**:
   - Добавление функций для записи на прием
   - Интеграция с другими сервисами клиники

## Ограничения

1. Максимум 100 файлов для RAG индекса
2. Промт не сохраняется в ассистенте, а добавляется к каждому сообщению
3. Возможны проблемы с потоками при их удалении

## Лицензия

Проект распространяется под MIT License.
