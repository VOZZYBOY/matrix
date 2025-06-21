"""
Модуль для анализа завершенности сообщений пользователей медицинской клиники.
Использует LLM для определения, является ли сообщение законченным или неполным.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, messages_from_dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class CompletenessStatus(str, Enum):
    """Статус завершенности сообщения"""
    COMPLETE = "complete"      
    INCOMPLETE = "incomplete"      

class MessageAnalysis(BaseModel):
    """Результат анализа завершенности сообщения"""
    status: CompletenessStatus = Field(description="Статус завершенности сообщения")
    confidence: float = Field(description="Уверенность в оценке (0.0-1.0)", ge=0.0, le=1.0)
    reasoning: str = Field(description="Обоснование решения")
    suggested_wait_time: float = Field(description="Рекомендуемое время ожидания в секундах", ge=0.0, le=30.0)
    indicators: List[str] = Field(description="Ключевые индикаторы, повлиявшие на решение")

@dataclass
class MessageContext:
    """Контекст сообщения для анализа"""
    current_message: str
    previous_messages: List[str]
    user_id: str
    time_since_last_message: Optional[float] = None
    conversation_topic: Optional[str] = None

class MessageCompletenessAnalyzer:
    """Анализатор завершенности сообщений с использованием OpenAI o3-mini"""
    
    def __init__(self, openai_api_key: str, model_name: str = "o3-mini"):
        """
        Инициализация анализатора
        
        Args:
            openai_api_key: API ключ OpenAI
            model_name: Название модели для использования (o3-mini или gpt-4o-mini)
        """
        base_llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            max_tokens=2000, 
            timeout=20      
          
        )
        
      
        self.llm = base_llm.with_structured_output(MessageAnalysis)
        
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "{analysis_request}")
        ])
        self._analysis_cache: Dict[str, Tuple[MessageAnalysis, float]] = {}
        self._cache_ttl = 60.0
        
    def _get_system_prompt(self) -> str:
        """Возвращает системный промпт с инструкциями и примерами"""
        return """Ты - эксперт по анализу завершенности сообщений в медицинской клинике.

ЗАДАЧА: Определи, завершено ли сообщение пользователя или нужно ждать продолжения.

ПРИНЦИПЫ АНАЛИЗА:
1. Естественная речь БЕЗ знаков препинания = НОРМАЛЬНО и ЗАВЕРШЕНО
2. Семантическая полнота важнее грамматики  
3. При сомнениях выбирай COMPLETE (лучше ответить, чем заставить ждать)
4. Медицинские вопросы обычно простые и прямые
5. УЧИТЫВАЙ КОНТЕКСТ предыдущих сообщений пользователя для понимания завершенности

ТОЛЬКО ДВА СТАТУСА:
- COMPLETE: мысль выражена полностью, можно отвечать
- INCOMPLETE: явно обрывается на полуслове, нужно ждать продолжения

ПРИМЕРЫ ЗАВЕРШЕННЫХ (без знаков препинания):
✅ "подскажи где работает амбулатория" 
✅ "у меня болит голова"
✅ "нужно записаться к врачу"  
✅ "когда работаете"
✅ "сколько стоит прием"
✅ "можно прийти завтра"
✅ "работает ли клиника"
✅ "где находится ваша больница"
✅ "хочу узнать расписание врачей"
✅ "принимаете ли детей"

ПРИМЕРЫ НЕЗАВЕРШЕННЫХ (редкие случаи):
❌ "подскажи пожалуйста где" - обрыв на вопросительном слове
❌ "хочу записаться к" - обрыв на предлоге  
❌ "у меня болит" - может продолжаться ("что именно")
❌ "нужно узнать про" - незавершенная мысль
❌ "я хочу сказать что" - явно будет продолжение

ПРИМЕРЫ С КОНТЕКСТОМ ДИАЛОГА:
🤖 Ассистент: Какие у вас симптомы?
👤 Пользователь: болит живот
Текущее сообщение: "еще и тошнота" → COMPLETE (дополнение к симптомам)

🤖 Ассистент: К какому врачу хотите записаться?
👤 Пользователь: хочу к врачу
Текущее сообщение: "к терапевту" → COMPLETE (ответ на вопрос)

👤 Пользователь: нужна справка
🤖 Ассистент: Для чего нужна справка?
Текущее сообщение: "для работы" → COMPLETE (ответ на уточнение)

КРИТЕРИИ ДЛЯ COMPLETE:
- Есть законченная мысль (понятно, что хочет пользователь)
- Вопрос содержит все ключевые элементы  
- Можно дать осмысленный ответ
- Грамматически завершенная конструкция

ВРЕМЯ ОЖИДАНИЯ:
- COMPLETE: 0 секунд (отвечаем сразу)
- INCOMPLETE: 5 секунд (ждем продолжения)

КРАТКИЕ ПОДТВЕРЖДЕНИЯ:
Сообщения, состоящие ТОЛЬКО из коротких слов подтверждения или одиночных эмодзи ("да", "ок", "супер", "👍" и др.), полученные менее чем через 15 секунд после ответа ассистента, считаются INCOMPLETE и не требуют ответа.

ПРИМЕРЫ ПОДТВЕРЖДЕНИЙ (INCOMPLETE):
❌ "да"
❌ "ок"
❌ "супер"
❌ "👍"

"""

    def _create_cache_key(self, context: MessageContext) -> str:
        """Создает ключ для кэширования анализа"""
        return f"{context.user_id}:{hash(context.current_message + str(context.previous_messages[-3:]))}"
    
    def _clean_cache(self):
        """Очищает устаревшие записи кэша"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._analysis_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_keys:
            del self._analysis_cache[key]
    
    async def analyze_message_completeness(self, context: MessageContext) -> MessageAnalysis:
        """
        Анализирует завершенность сообщения
        
        Args:
            context: Контекст сообщения для анализа
            
        Returns:
            MessageAnalysis: Результат анализа
        """

        cache_key = self._create_cache_key(context)
        self._clean_cache()
        
        if cache_key in self._analysis_cache:
            cached_analysis, _ = self._analysis_cache[cache_key]
            logger.debug(f"Возвращен кэшированный анализ для {context.user_id}")
            return cached_analysis
        
        try:
            analysis_request = self._prepare_analysis_request(context)
            

            chain = self.prompt_template | self.llm
            
            
            start_time = time.time()
            analysis_result = await chain.ainvoke({"analysis_request": analysis_request})
            analysis_time = time.time() - start_time
            
            logger.info(f"Анализ завершенности для {context.user_id}: {analysis_result.status} "
                       f"(уверенность: {analysis_result.confidence:.2f}, время: {analysis_time:.2f}s)")
            
            
            self._analysis_cache[cache_key] = (analysis_result, time.time())
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Ошибка при анализе завершенности сообщения: {e}")
        
            return MessageAnalysis(
                status=CompletenessStatus.COMPLETE,
                confidence=0.5,
                reasoning=f"Ошибка анализа: {str(e)}. При ошибке предпочитаем отвечать.",
                suggested_wait_time=0.0,
                indicators=["analysis_error", "fallback_complete"]
            )
    
    def _prepare_analysis_request(self, context: MessageContext) -> str:
        """Подготавливает запрос для анализа"""
        request = f"""Проанализируй сообщение: "{context.current_message}"

Длина: {len(context.current_message)} символов
Слов: {len(context.current_message.split())}"""

        if context.previous_messages:
            history_text = "\n".join([
                msg for msg in context.previous_messages[-5:]  
            ])
            request += f"""

КОНТЕКСТ ДИАЛОГА (последние сообщения):
{history_text}

Учитывай полный контекст беседы при анализе завершенности текущего сообщения пользователя."""
        
        request += "\n\nВерни структурированный анализ завершенности."
        
        return request

    def quick_heuristic_check(self, message: str) -> Optional[CompletenessStatus]:
        """
        Минимальная проверка только для очевидных случаев
        Возвращает None, если нужен полный анализ LLM
        """
        message = message.strip()
        
      
        if not message or len(message.strip()) == 0:
            return CompletenessStatus.INCOMPLETE
            
        if not any(c.isalnum() for c in message):
            return CompletenessStatus.INCOMPLETE
            
        
        # Проверяем распространённые слова-подтверждения (acknowledgements)
        normalized = re.sub(r"[^\w\s]", "", message.lower()).strip()
        ack_words = {
            "да", "ок", "okay", "окей", "супер", "хорошо",
            "понятно", "yep", "yeah", "sure", "got it", "roger"
        }
        if normalized in ack_words:
            return CompletenessStatus.INCOMPLETE

        if len(message) <= 2:
            return CompletenessStatus.INCOMPLETE
            
        
        return None 

_analyzer_instance: Optional[MessageCompletenessAnalyzer] = None


_message_accumulator: Dict[str, Dict[str, any]] = {}
ACCUMULATOR_TTL = 300  

def get_analyzer() -> MessageCompletenessAnalyzer:
    """Возвращает глобальный экземпляр анализатора"""
    global _analyzer_instance
    if _analyzer_instance is None:
        raise RuntimeError("Анализатор не инициализирован. Вызовите initialize_analyzer() первым.")
    return _analyzer_instance

def initialize_analyzer(openai_api_key: str, model_name: str = "o3-mini"):
    """Инициализирует глобальный экземпляр анализатора"""
    global _analyzer_instance
    _analyzer_instance = MessageCompletenessAnalyzer(openai_api_key, model_name)
    logger.info(f"Анализатор завершенности сообщений инициализирован с моделью {model_name}")

def _clean_accumulator():
    """Очищает устаревшие записи накопителя"""
    current_time = time.time()
    expired_keys = [
        user_id for user_id, data in _message_accumulator.items()
        if current_time - data['last_update'] > ACCUMULATOR_TTL
    ]
    for key in expired_keys:
        del _message_accumulator[key]
        logger.debug(f"Удален устаревший накопитель для пользователя {key}")

def add_message_to_accumulator(user_id: str, message: str) -> str:
    """
    Добавляет сообщение в накопитель и возвращает склеенное сообщение
    
    Args:
        user_id: ID пользователя
        message: Новое сообщение для добавления
        
    Returns:
        str: Склеенное сообщение из всех частей
    """
    _clean_accumulator()
    current_time = time.time()
    
    if user_id not in _message_accumulator:
    
        _message_accumulator[user_id] = {
            'accumulated_message': message,
            'last_update': current_time,
            'parts': [message],
            'part_count': 1
        }
        logger.info(f"Создан новый накопитель для {user_id}: '{message[:50]}...'")
    else:
        accumulator = _message_accumulator[user_id]
        
        separator = _get_smart_separator(accumulator['accumulated_message'], message)
        
        accumulator['accumulated_message'] += separator + message
        accumulator['last_update'] = current_time
        accumulator['parts'].append(message)
        accumulator['part_count'] += 1
        
        logger.info(f"Добавлено к накопителю {user_id} (часть {accumulator['part_count']}): '{message[:30]}...'")
        logger.debug(f"Склеенное сообщение для {user_id}: '{accumulator['accumulated_message'][:100]}...'")
    
    return _message_accumulator[user_id]['accumulated_message']

def get_accumulated_message(user_id: str) -> Optional[str]:
    """Возвращает накопленное сообщение для пользователя, если есть"""
    _clean_accumulator()
    if user_id in _message_accumulator:
        return _message_accumulator[user_id]['accumulated_message']
    return None

def clear_accumulator(user_id: str):
    """Очищает накопитель для пользователя"""
    if user_id in _message_accumulator:
        parts_count = _message_accumulator[user_id]['part_count']
        del _message_accumulator[user_id]
        logger.info(f"Очищен накопитель для {user_id} ({parts_count} частей)")

def _get_smart_separator(existing: str, new: str) -> str:
    """
    Определяет умный разделитель между частями сообщения
    
    Args:
        existing: Существующая часть сообщения  
        new: Новая часть сообщения
        
    Returns:
        str: Подходящий разделитель
    """
  
    if existing.rstrip().endswith(('.', '!', '?', ':', ';')):
        return ' '
    
    
    if existing.rstrip().endswith(','):
        return ' '
    
    
    if new and new[0].islower():
        return ' '
    
    conjunctions = ['и', 'а', 'но', 'или', 'что', 'как', 'когда', 'где', 'потому что', 'так как']
    for conj in conjunctions:
        if existing.rstrip().lower().endswith(conj):
            return ' '
    
    return ' '

async def analyze_message(
    message: str, 
    user_id: str, 
    previous_messages: Optional[List[str]] = None,
    time_since_last: Optional[float] = None
) -> MessageAnalysis:
    """
    Удобная функция для анализа сообщения
    
    Args:
        message: Текст сообщения для анализа
        user_id: ID пользователя
        previous_messages: Предыдущие сообщения (опционально)
        time_since_last: Время с последнего сообщения в секундах
        
    Returns:
        MessageAnalysis: Результат анализа
    """
    analyzer = get_analyzer()
    
    quick_result = analyzer.quick_heuristic_check(message)
    if quick_result is not None:
        logger.debug(f"Быстрый анализ для {user_id}: {quick_result}")
        
        confidence_map = {
            CompletenessStatus.INCOMPLETE: 0.85,
            CompletenessStatus.COMPLETE: 0.9
        }
        
        wait_time_map = {
            CompletenessStatus.INCOMPLETE: 5.0,
            CompletenessStatus.COMPLETE: 0.0
        }
        
        return MessageAnalysis(
            status=quick_result,
            confidence=confidence_map[quick_result],
            reasoning="Эвристический анализ",
            suggested_wait_time=wait_time_map[quick_result],
            indicators=["heuristic_analysis"]
        )

    context = MessageContext(
        current_message=message,
        previous_messages=previous_messages or [],
        user_id=user_id,
        time_since_last_message=time_since_last
    )
    
    return await analyzer.analyze_message_completeness(context)

def get_history_via_langchain(tenant_id: str, user_id: str, limit: int = 15) -> List[str]:
    """
    Получает историю чата используя тот же подход, что и основной агент.
    Использует LangChain TenantAwareRedisChatMessageHistory для правильной обработки структуры data.content
    
    Args:
        tenant_id: ID тенанта
        user_id: ID пользователя  
        limit: Количество сообщений
        
    Returns:
        List[str]: Список отформатированных сообщений для контекста
    """
    try:
        # Импортируем готовый класс из redis_history
        from redis_history import TenantAwareRedisChatMessageHistory
        
        # Создаем экземпляр истории (тот же подход что у основного агента)
        chat_history = TenantAwareRedisChatMessageHistory(tenant_id=tenant_id, session_id=user_id)
        
        # Получаем сообщения через LangChain (автоматически обрабатывает data.content)
        messages = chat_history.messages
        logger.info(f"[LangChain История] Получено {len(messages)} BaseMessage объектов")
        
        if not messages:
            return []
        
        # Берем последние сообщения согласно лимиту
        recent_messages = messages[-limit:] if len(messages) > limit else messages
        
        # Форматируем сообщения для контекста
        formatted_messages = []
        for msg in recent_messages:
            # BaseMessage имеет атрибуты type и content
            msg_type = getattr(msg, 'type', 'unknown')
            msg_content = getattr(msg, 'content', '').strip()
            
            if msg_content and len(msg_content) > 2:
                # Форматируем с указанием автора
                author = "👤 Пользователь" if msg_type == 'human' else "🤖 Ассистент"
                formatted_message = f"{author}: {msg_content}"
                formatted_messages.append(formatted_message)
                logger.debug(f"[LangChain История] ✅ Добавлено сообщение: '{formatted_message[:50]}...'")
            else:
                logger.debug(f"[LangChain История] ❌ Пропущено: type='{msg_type}', len={len(msg_content)}")
        
        logger.info(f"[LangChain История] Возвращаем {len(formatted_messages)} отформатированных сообщений")
        return formatted_messages
        
    except Exception as e:
        logger.error(f"Ошибка при получении истории через LangChain для {tenant_id}:{user_id}: {e}", exc_info=True)
        return []

async def analyze_message_with_accumulation(
    message: str, 
    user_id: str, 
    previous_messages: Optional[List[str]] = None,
    time_since_last: Optional[float] = None
) -> Tuple[MessageAnalysis, str]:
    """
    Анализирует сообщение с простым накоплением частей
    
    Args:
        message: Текст нового сообщения
        user_id: ID пользователя  
        previous_messages: Предыдущие сообщения (опционально)
        time_since_last: Время с последнего сообщения в секундах
        
    Returns:
        Tuple[MessageAnalysis, str]: Результат анализа и накопленное сообщение
    """
    accumulated_message = add_message_to_accumulator(user_id, message)
    
    # Анализируем весь накопленный текст
    analysis = await analyze_message(accumulated_message, user_id, previous_messages, time_since_last)
    
    # Если накопленное сообщение завершено - очищаем накопитель и возвращаем для обработки
    if analysis.status == CompletenessStatus.COMPLETE:
        final_message = accumulated_message
        clear_accumulator(user_id)
        logger.info(f"Сообщение для {user_id} завершено после накопления: '{final_message[:100]}...'")
        return analysis, final_message
    
    # Если неполное - оставляем в накопителе и возвращаем для ожидания
    else:
        logger.info(f"Сообщение для {user_id} неполное, продолжаем накопление: '{accumulated_message[:100]}...'")
        return analysis, accumulated_message

def should_wait_for_completion(analysis: MessageAnalysis) -> bool:
    """
    Определяет, нужно ли ждать завершения сообщения
    
    Args:
        analysis: Результат анализа сообщения
        
    Returns:
        bool: True если нужно ждать, False если можно отвечать
    """
    if analysis.status == CompletenessStatus.COMPLETE:
        return False
        
    if analysis.status == CompletenessStatus.INCOMPLETE:
        return analysis.confidence > 0.7
        
    return False
        