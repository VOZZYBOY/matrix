# redis_history.py
import redis
import os
import json
import logging
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
# Добавляем импорт BaseChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
# Добавляем импорт для конвертации
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

# Загружаем переменные окружения (если есть .env файл)
load_dotenv()

# Настраиваем логирование
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Настройки Redis (можно вынести в переменные окружения)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
REDIS_KEY_PREFIX = "chat_history"
DEFAULT_TTL_SECONDS = 3600

# Инициализация клиента Redis
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        password=REDIS_PASSWORD,
        decode_responses=False # Будем декодировать сами после JSON
    )
    # Проверяем соединение
    redis_client.ping()
    logger.info(f"Успешное подключение к Redis: {REDIS_HOST}:{REDIS_PORT}, DB: {REDIS_DB}")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Не удалось подключиться к Redis: {e}", exc_info=True)
    redis_client = None # Устанавливаем в None, чтобы функции могли это проверить

def _get_redis_key(tenant_id: str, user_id: str) -> str:
    """Формирует ключ для Redis."""
    if not tenant_id or not user_id:
        raise ValueError("tenant_id и user_id не могут быть пустыми")
    return f"{REDIS_KEY_PREFIX}:{tenant_id}:{user_id}"

def add_message(tenant_id: str, user_id: str, message_data: Dict[str, Any]) -> bool:
    """
    Добавляет сообщение в историю чата для указанного tenant_id и user_id.
    Устанавливает/обновляет TTL для сессии.

    :param tenant_id: Идентификатор тенанта (филиала).
    :param user_id: Идентификатор пользователя (сессии).
    :param message_data: Словарь с данными сообщения (e.g., {'type': 'human', 'content': '...'}).
    :return: True если успешно, False при ошибке.
    """
    if not redis_client:
        logger.error("Клиент Redis не инициализирован. Невозможно добавить сообщение.")
        return False
    if not isinstance(message_data, dict):
        logger.error("message_data должен быть словарем.")
        return False

    try:
        key = _get_redis_key(tenant_id, user_id)
        # Сериализуем сообщение в JSON строку
        message_json = json.dumps(message_data, ensure_ascii=False)
        # Добавляем в конец списка
        redis_client.rpush(key, message_json)
        # Устанавливаем/обновляем время жизни ключа
        redis_client.expire(key, DEFAULT_TTL_SECONDS)
        logger.debug(f"Сообщение добавлено для {key}")
        return True
    except ValueError as ve:
         logger.error(f"Ошибка формирования ключа: {ve}")
         return False
    except redis.exceptions.RedisError as e:
        logger.error(f"Ошибка Redis при добавлении сообщения для {tenant_id}/{user_id}: {e}", exc_info=True)
        return False
    except json.JSONDecodeError as e:
         logger.error(f"Ошибка сериализации сообщения в JSON для {tenant_id}/{user_id}: {e}", exc_info=True)
         return False
    except Exception as e:
        logger.error(f"Неизвестная ошибка при добавлении сообщения для {tenant_id}/{user_id}: {e}", exc_info=True)
        return False

def get_history(tenant_id: str, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Получает последние 'limit' сообщений из истории чата.

    :param tenant_id: Идентификатор тенанта.
    :param user_id: Идентификатор пользователя.
    :param limit: Максимальное количество сообщений для возврата.
    :return: Список словарей сообщений или пустой список при ошибке/отсутствии истории.
    """
    if not redis_client:
        logger.error("Клиент Redis не инициализирован. Невозможно получить историю.")
        return []

    try:
        key = _get_redis_key(tenant_id, user_id)
        # Получаем последние limit элементов (от -limit до -1)
        # Redis возвращает байтовые строки, т.к. decode_responses=False
        history_bytes = redis_client.lrange(key, -limit, -1)

        history_dicts = []
        for msg_bytes in history_bytes:
            try:
                # Декодируем байты в строку и парсим JSON
                msg_str = msg_bytes.decode('utf-8')
                history_dicts.append(json.loads(msg_str))
            except json.JSONDecodeError as e:
                 logger.warning(f"Ошибка декодирования JSON из истории для {key}: {e}. Пропускаем сообщение: {msg_bytes[:100]}...")
            except UnicodeDecodeError as e:
                 logger.warning(f"Ошибка декодирования UTF-8 из истории для {key}: {e}. Пропускаем сообщение: {msg_bytes[:100]}...")

        logger.debug(f"Получено {len(history_dicts)} сообщений для {key}")
        return history_dicts
    except ValueError as ve:
         logger.error(f"Ошибка формирования ключа: {ve}")
         return []
    except redis.exceptions.RedisError as e:
        logger.error(f"Ошибка Redis при получении истории для {tenant_id}/{user_id}: {e}", exc_info=True)
        return []
    except Exception as e:
        logger.error(f"Неизвестная ошибка при получении истории для {tenant_id}/{user_id}: {e}", exc_info=True)
        return []


def clear_history(tenant_id: str, user_id: str) -> bool:
    """
    Удаляет всю историю чата для указанного tenant_id и user_id.

    :param tenant_id: Идентификатор тенанта.
    :param user_id: Идентификатор пользователя.
    :return: True если успешно или ключ не найден, False при ошибке Redis.
    """
    if not redis_client:
        logger.error("Клиент Redis не инициализирован. Невозможно очистить историю.")
        return False

    try:
        key = _get_redis_key(tenant_id, user_id)
        deleted_count = redis_client.delete(key)
        logger.info(f"История для {key} удалена. Количество удаленных ключей: {deleted_count}")
        return True # Возвращаем True даже если ключа не было (deleted_count=0)
    except ValueError as ve:
         logger.error(f"Ошибка формирования ключа: {ve}")
         return False
    except redis.exceptions.RedisError as e:
        logger.error(f"Ошибка Redis при удалении истории для {tenant_id}/{user_id}: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Неизвестная ошибка при удалении истории для {tenant_id}/{user_id}: {e}", exc_info=True)
        return False

# <<< НАЧАЛО НОВОГО КОДА >>>
class TenantAwareRedisChatMessageHistory(BaseChatMessageHistory):
    """
    Класс истории сообщений чата, совместимый с LangChain,
    который хранит сообщения в Redis с учетом tenant_id.

    Использует функции из этого модуля (add_message, get_history, clear_history).
    """
    def __init__(self, tenant_id: str, session_id: str, ttl: Optional[int] = DEFAULT_TTL_SECONDS):
        """
        Инициализирует историю.

        Args:
            tenant_id: Идентификатор тенанта.
            session_id: Идентификатор сессии (пользователя).
            ttl: Время жизни истории в секундах (используется при добавлении).
        """
        if not tenant_id or not session_id:
             raise ValueError("tenant_id и session_id не могут быть пустыми")
        self.tenant_id = tenant_id
        self.session_id = session_id # Используем session_id для user_id
        self.key = _get_redis_key(self.tenant_id, self.session_id)
        self.ttl = ttl

    @property
    def messages(self) -> List[BaseMessage]:  # type: ignore
        """Получает сообщения из Redis."""
        items_dict = get_history(self.tenant_id, self.session_id, limit=1000) # Получаем как словари
        # Конвертируем словари обратно в объекты BaseMessage
        messages = messages_from_dict(items_dict)
        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Добавляет сообщение в Redis."""
        # Конвертируем объект BaseMessage в словарь
        message_dict = messages_to_dict([message])[0]
        # Используем существующую функцию add_message
        success = add_message(self.tenant_id, self.session_id, message_dict)
        if not success:
            # Можно добавить более специфичную обработку ошибок, если нужно
            logger.error(f"Не удалось добавить сообщение в Redis для ключа {self.key}")
            # raise RuntimeError(f"Failed to add message to Redis for key {self.key}") # Опционально

    def clear(self) -> None:
        """Очищает историю в Redis."""
        # Используем существующую функцию clear_history
        clear_history(self.tenant_id, self.session_id)

