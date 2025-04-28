document.addEventListener('DOMContentLoaded', () => {
    const messagesContainer = document.getElementById('messages-container');
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const resetButton = document.getElementById('reset-button');
    const typingIndicator = document.querySelector('.typing-indicator');
    const assistantStatus = document.getElementById('assistant-status');
    const activeSessions = document.getElementById('active-sessions');
    const lastUpdateTime = document.getElementById('last-update-time');
    const promptButtons = document.querySelectorAll('.prompt-btn');
    const promptTextElement = document.getElementById('prompt-text');
    const currentPromptTitle = document.getElementById('current-prompt-title');
    const editPromptBtn = document.getElementById('edit-prompt-btn');
    const promptEditor = document.getElementById('prompt-editor');
    const savePromptBtn = document.getElementById('save-prompt-btn');
    const cancelEditBtn = document.getElementById('cancel-edit-btn');
    const promptPreview = document.querySelector('.prompt-preview');
    const promptEditorContainer = document.querySelector('.prompt-editor');
    
    // User ID элементы
    const currentUserIdElement = document.getElementById('current-user-id');
    const editUserIdBtn = document.getElementById('edit-user-id-btn');
    const userIdEditor = document.getElementById('user-id-editor');
    const userIdInput = document.getElementById('user-id-input');
    const saveUserIdBtn = document.getElementById('save-user-id-btn');
    const cancelUserIdBtn = document.getElementById('cancel-user-id-btn');
    
    // Переменные для хранения элементов управления логами
    let logsPanel = null;
    let logsList = null;
    let refreshLogsBtn = null;
    let logsTabs = null;
    let logsCount = null;
    let clearLogsBtn = null;
    
    // Новые элементы для ID пользователя
    const userIdField = document.getElementById('user-id-field');
    const logsContainer = document.getElementById('logs-container');
    
    // Добавить div для логов в интерфейс, если его еще нет
    initializeLogsPanel();
    
    // Функция инициализации панели логов
    function initializeLogsPanel() {
        // Проверяем, существует ли уже панель логов
        if (document.querySelector('.logs-panel')) {
            return;
        }
        
        // Создаем панель логов и добавляем в system-info-panel
        const systemInfoPanel = document.querySelector('.system-info-panel');
        if (!systemInfoPanel) {
            console.error('Элемент .system-info-panel не найден!');
            return;
        }
        
        const logsCard = document.createElement('div');
        logsCard.className = 'system-info-card logs-panel';
        logsCard.innerHTML = `
            <div class="logs-header">
                <h3>Системные логи</h3>
                <button class="refresh-logs" title="Обновить логи">&#x21bb;</button>
            </div>
            <div class="logs-container">
                <div class="logs-tabs">
                    <button class="log-tab active" data-type="all">Все</button>
                    <button class="log-tab" data-type="info">Инфо</button>
                    <button class="log-tab" data-type="warning">Предупр.</button>
                    <button class="log-tab" data-type="error">Ошибки</button>
                </div>
                <div class="logs-content">
                    <div id="logs-list" class="logs-list">
                        <div class="log-entry log-info">
                            <span class="log-time">00:00:00</span>
                            <span class="log-level">INFO</span>
                            <span class="log-message">Загрузка логов...</span>
                        </div>
                    </div>
                </div>
                <div class="logs-actions">
                    <div class="logs-status">
                        Показано: <span id="logs-count">0</span> записей
                    </div>
                    <button id="clear-logs-btn" class="btn-outline-small">Очистить</button>
                </div>
            </div>
        `;
        
        systemInfoPanel.appendChild(logsCard);
        
        // Получаем ссылки на элементы
        logsPanel = document.querySelector('.logs-panel');
        logsList = document.getElementById('logs-list');
        refreshLogsBtn = document.querySelector('.refresh-logs');
        logsTabs = document.querySelectorAll('.log-tab');
        clearLogsBtn = document.getElementById('clear-logs-btn');
        logsCount = document.getElementById('logs-count');
        
        // Добавляем обработчики событий
        refreshLogsBtn.addEventListener('click', fetchLogs);
        
        // Обработчик для переключения вкладок
        logsTabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // Удалить класс active у всех вкладок
                logsTabs.forEach(t => t.classList.remove('active'));
                // Добавить класс active на выбранную вкладку
                tab.classList.add('active');
                
                // Фильтрация логов по выбранному типу
                const logType = tab.dataset.type;
                filterLogs(logType);
            });
        });
        
        // Обработчик для кнопки очистки логов
        clearLogsBtn.addEventListener('click', () => {
            logsList.innerHTML = '';
            logsCount.textContent = '0';
        });
        
        // Загрузить логи при инициализации
        fetchLogs();
    }
    
    // Функция для получения логов с сервера
    function fetchLogs() {
        // Отображаем сообщение о загрузке
        logsList.innerHTML = '<div class="log-entry log-info"><span class="log-time">--:--:--</span><span class="log-level">INFO</span><span class="log-message">Загрузка логов...</span></div>';
        
        // Делаем запрос на сервер
        fetch('/logs')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Ошибка при получении логов');
                }
                return response.json();
            })
            .then(data => {
                displayLogs(data.logs || []);
            })
            .catch(error => {
                logsList.innerHTML = `<div class="log-entry log-error">
                    <span class="log-time">--:--:--</span>
                    <span class="log-level">ERROR</span>
                    <span class="log-message">Ошибка загрузки логов: ${error.message}</span>
                </div>`;
            });
    }
    
    // Функция для отображения логов
    function displayLogs(logs) {
        if (!Array.isArray(logs) || logs.length === 0) {
            logsList.innerHTML = '<div class="log-entry log-info"><span class="log-time">--:--:--</span><span class="log-level">INFO</span><span class="log-message">Логи отсутствуют</span></div>';
            logsCount.textContent = '0';
            return;
        }
        
        // Очищаем текущие логи
        logsList.innerHTML = '';
        
        // Добавляем каждую запись лога
        logs.forEach(log => {
            const logEntry = document.createElement('div');
            
            // Определяем класс в зависимости от уровня лога
            let logClass = 'log-info';
            if (log.level.toLowerCase().includes('error') || log.level.toLowerCase().includes('ошибка')) {
                logClass = 'log-error';
            } else if (log.level.toLowerCase().includes('warn') || log.level.toLowerCase().includes('предупр')) {
                logClass = 'log-warning';
            }
            
            logEntry.className = `log-entry ${logClass}`;
            
            // Форматируем время, если оно есть
            let timeString = '--:--:--';
            if (log.time) {
                timeString = log.time;
            }
            
            // Создаем содержимое записи
            logEntry.innerHTML = `
                <span class="log-time">${timeString}</span>
                <span class="log-level">${log.level}</span>
                <span class="log-message">${log.message}</span>
            `;
            
            logsList.appendChild(logEntry);
        });
        
        // Обновляем счетчик
        logsCount.textContent = logs.length;
        
        // Применяем текущий фильтр
        const activeTab = document.querySelector('.log-tab.active');
        if (activeTab) {
            filterLogs(activeTab.dataset.type);
        }
    }
    
    // Функция для фильтрации логов
    function filterLogs(logType) {
        const entries = document.querySelectorAll('.log-entry');
        let visibleCount = 0;
        
        entries.forEach(entry => {
            if (logType === 'all') {
                entry.style.display = 'flex';
                visibleCount++;
            } else if (logType === 'info' && entry.classList.contains('log-info')) {
                entry.style.display = 'flex';
                visibleCount++;
            } else if (logType === 'warning' && entry.classList.contains('log-warning')) {
                entry.style.display = 'flex';
                visibleCount++;
            } else if (logType === 'error' && entry.classList.contains('log-error')) {
                entry.style.display = 'flex';
                visibleCount++;
            } else {
                entry.style.display = 'none';
            }
        });
        
        // Обновляем счетчик видимых записей
        logsCount.textContent = visibleCount;
    }
    
    // Отладка для проверки элементов редактирования промта
    console.log('Загрузка DOM...');
    console.log('promptTextElement:', promptTextElement);
    console.log('currentPromptTitle:', currentPromptTitle);
    console.log('editPromptBtn:', editPromptBtn);
    console.log('promptEditor:', promptEditor);
    console.log('savePromptBtn:', savePromptBtn);
    console.log('cancelEditBtn:', cancelEditBtn);
    console.log('promptPreview:', promptPreview);
    console.log('promptEditorContainer:', promptEditorContainer);
    
    let userId = localStorage.getItem('clinicUserId') || null;
    let selectedPrompt = localStorage.getItem('selectedPrompt') || 'default';
    
    // Словарь с промтами для разных типов
    const promptTemplates = {
        default: `Ты - вежливый, **ОЧЕНЬ ВНИМАТЕЛЬНЫЙ** и информативный ассистент медицинской клиники "Med YU Med".
Твоя главная задача - помогать пользователям, отвечая на их вопросы об услугах, ценах, специалистах и филиалах клиники, используя предоставленные инструменты и историю диалога.

**ВАЖНО! ВСЕГДА В ПЕРВУЮ ОЧЕРЕДЬ ИСПОЛЬЗУЙ ПОИСК ПО ИНФОРМАЦИИ ИЗ БАЗЫ ЗНАНИЙ:**
Когда пользователь спрашивает о деталях услуг, описаниях процедур, или задает общие вопросы о специалистах и их квалификации, 
**ОБЯЗАТЕЛЬНО** используй предоставленный RAG индекс для поиска релевантной информации. 
**НИКОГДА НЕ ВЫДУМЫВАЙ ОТВЕТЫ** о специалистах и услугах клиники. Опирайся только на найденную информацию.
**ЕСЛИ ТЫ НЕ НАШЁЛ ИНФОРМАЦИЮ В РЕЗУЛЬТАТАХ ПОИСКА**, так и скажи, но не придумывай факты.

**КЛЮЧЕВЫЕ ПРАВИЛА ВЫБОРА ИНСТРУМЕНТА И РАБОТЫ С КОНТЕКСТОМ:**

1.  **СНАЧАЛА ОПРЕДЕЛИ ТИП ЗАПРОСА:**
    *   **ОБЩИЙ ВОПРОС / ЗАПРОС ОПИСАНИЯ (Что? Как? Зачем? Посоветуй... Расскажи о...):** Если пользователь спрашивает **общее описание** услуги (что это, как работает, какой эффект), просит **подробности** об опыте/специализации врача, или задает **открытый вопрос** ("что для омоложения?", "какие пилинги бывают?", "посоветуй от морщин"), **ПОИСК ДОП. ИНФОРМАЦИИ:** Ты **должен** использовать предоставленные тебе текстовые материалы (описания услуг и врачей) для поиска релевантной информации. **СИНТЕЗИРУЙ** свой ответ на основе найденных описаний. НЕ пытайся сразу вызывать функции для таких общих вопросов.
    *   **КОНКРЕТНЫЙ ЗАПРОС (Сколько? Где? Кто? Сравни...):** Если пользователь спрашивает **конкретную цену**, **наличие в филиале**, **список врачей/услуг по критерию**, **сравнение цен**, **ИСПОЛЬЗУЙ Function Calling**.`,
        
        custom: localStorage.getItem('customPrompt') || `[Введите свой собственный промт для ассистента. Опишите его роль, задачи, ограничения и особенности поведения. 
Убедитесь, что промт содержит четкие инструкции по работе с базой знаний клиники и Function Calling.]`
    };
    
    // Загружаем сохраненные редакции промтов из localStorage, если они есть
    Object.keys(promptTemplates).forEach(key => {
        const savedPrompt = localStorage.getItem(`prompt_${key}`);
        if (savedPrompt) {
            promptTemplates[key] = savedPrompt;
        }
    });
    
    let lastUpdated = '-';
    
    // Инициализация кнопок редактирования промта
    console.log('Настройка кнопок редактирования...');
    if (editPromptBtn) {
        // Удаляем прежний обработчик (включая inline)
        editPromptBtn.onclick = null;
        editPromptBtn.removeAttribute('onclick');
        
        // Добавляем новый обработчик
        editPromptBtn.addEventListener('click', function() {
            console.log('Кнопка редактирования нажата');
            
            // Проверяем элементы
            if (!promptPreview) {
                console.error('promptPreview не найден');
                return;
            }
            if (!promptEditorContainer) {
                console.error('promptEditorContainer не найден');
                return;
            }
            
            // Меняем отображение
            promptPreview.style.display = 'none';
            promptEditorContainer.style.display = 'block';
            
            // Загружаем текущий промт в редактор
            promptEditor.value = promptTemplates[selectedPrompt];
            promptEditor.focus();
        });
    } else {
        console.error('ОШИБКА: Элемент editPromptBtn не найден!');
    }
    
    if (savePromptBtn) {
        savePromptBtn.onclick = null;
        savePromptBtn.removeAttribute('onclick');
        
        savePromptBtn.addEventListener('click', function() {
            console.log('Кнопка сохранения нажата');
            
            // Сохраняем отредактированный промт
            promptTemplates[selectedPrompt] = promptEditor.value;
            
            // Сохраняем в localStorage
            localStorage.setItem(`prompt_${selectedPrompt}`, promptEditor.value);
            
            // Для индивидуального промта сохраняем отдельно
            if (selectedPrompt === 'custom') {
                localStorage.setItem('customPrompt', promptEditor.value);
            }
            
            // Обновляем отображение текста промта
            updatePromptDisplay();
            
            // Сбрасываем чат с обновленным промтом
            resetChat(true);
            
            // Возвращаемся к режиму просмотра
            promptPreview.style.display = 'block';
            promptEditorContainer.style.display = 'none';
        });
    }
    
    if (cancelEditBtn) {
        cancelEditBtn.onclick = null;
        cancelEditBtn.removeAttribute('onclick');
        
        cancelEditBtn.addEventListener('click', function() {
            console.log('Кнопка отмены нажата');
            promptPreview.style.display = 'block';
            promptEditorContainer.style.display = 'none';
        });
    }
    
    // Отображаем текущий промт при загрузке
    updatePromptDisplay();
    
    // Активируем нужную кнопку промта при загрузке
    promptButtons.forEach(button => {
        if (button.dataset.prompt === selectedPrompt) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
    
    // Добавляем обработчик выбора промта
    promptButtons.forEach(button => {
        button.addEventListener('click', () => {
            // Убираем класс active у всех кнопок
            promptButtons.forEach(btn => btn.classList.remove('active'));
            // Добавляем класс active выбранной кнопке
            button.classList.add('active');
            // Сохраняем выбранный промт
            selectedPrompt = button.dataset.prompt;
            localStorage.setItem('selectedPrompt', selectedPrompt);
            
            // Обновляем отображение текста промта
            updatePromptDisplay();
            
            // Сбрасываем чат с новым промтом
            resetChat(true);
        });
    });
    
    // Функция обновления отображения текста промта
    function updatePromptDisplay() {
        // Обновляем заголовок
        let promptTitle = selectedPrompt;
        
        switch(selectedPrompt) {
            case 'default': promptTitle = 'Стандартный'; break;
            case 'custom': promptTitle = 'Индивидуальный'; break;
        }
        
        currentPromptTitle.textContent = promptTitle;
        
        // Отображаем текст промта
        const currentPromptText = promptTemplates[selectedPrompt] || '';
        promptTextElement.textContent = currentPromptText;
    }
    
    // Загружаем статус системы при загрузке страницы
    updateSystemStatus();
    
    // Обновляем статус каждые 30 секунд
    setInterval(updateSystemStatus, 30000);
    
    if (!localStorage.getItem('chatInitialized')) {
        showWelcomeMessage();
    } else {
        loadChatHistory();
    }
    
    // Функция для получения текста приветственного сообщения в зависимости от выбранного промта
    function getWelcomeMessage() {
        switch (selectedPrompt) {
            case 'custom':
                return `Здравствуйте! Я виртуальный ассистент клиники Med YU Med с индивидуальными настройками.

✅ Я настроен согласно индивидуальному промту, который вы создали.
🔍 Буду отвечать на ваши вопросы в соответствии с заданными параметрами.

Чем могу помочь?`;
                
            default:
                return `Здравствуйте! Я виртуальный ассистент клиники Med YU Med.

✅ Я могу ответить на ваши вопросы о:
• Услугах клиники и их описаниях
• Ценах на процедуры в разных филиалах
• Квалификации врачей и их специализации
• Наличии услуг в конкретных филиалах

🔍 Как я работаю:
• Для поиска описаний услуг и врачей использую технологию векторного поиска (RAG)
• Для точных данных о ценах и расписании применяю функциональные вызовы к базе данных
• Запоминаю контекст разговора и не переспрашиваю одну и ту же информацию

Чем я могу вам помочь сегодня?`;
        }
    }
    
    function showWelcomeMessage() {
        const welcomeMessage = getWelcomeMessage();
        addMessage(welcomeMessage, 'assistant', true);
        localStorage.setItem('chatInitialized', 'true');
    }
    
    sendButton.addEventListener('click', sendMessage);
    
    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    resetButton.addEventListener('click', () => resetChat(false));
    
    async function sendMessage() {
        const messageText = messageInput.value.trim();
        if (!messageText) return;
        
        messageInput.value = '';
        
        addMessage(messageText, 'user');
        
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        typingIndicator.classList.add('active');
        
        try {
            const requestData = {
                message: messageText,
                user_id: userId,
                reset_session: false,
                prompt_type: selectedPrompt,
                prompt_text: selectedPrompt === 'custom' ? promptTemplates[selectedPrompt] : null
            };
            
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`Ошибка: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.user_id) {
                userId = data.user_id;
                localStorage.setItem('clinicUserId', userId);
            }
            
            typingIndicator.classList.remove('active');
            
            addMessage(data.response, 'assistant');
            
            // Обновляем статус системы после каждого сообщения
            updateSystemStatus();
            
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            
            saveChatHistory();
            
        } catch (error) {
            console.error('Ошибка при отправке сообщения:', error);
            
            typingIndicator.classList.remove('active');
            
            addMessage('Произошла ошибка при обработке запроса. Пожалуйста, попробуйте еще раз.', 'assistant');
            
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
    }
    
    function addMessage(text, sender, isFirst = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `message-${sender}`);
        if (isFirst) {
            messageElement.classList.add('message-first');
        }
        
        // Преобразуем переносы строк в <br> для корректного отображения
        const formattedText = text.replace(/\n/g, '<br>');
        messageElement.innerHTML = formattedText;
        
        // Добавляем элементы оценки только для сообщений ассистента (кроме первого)
        if (sender === 'assistant' && !isFirst) {
            const ratingElement = document.createElement('div');
            ratingElement.classList.add('message-rating');
            
            const messageId = Date.now().toString();
            
            // Кнопка "Нравится"
            const likeButton = document.createElement('button');
            likeButton.classList.add('rating-btn', 'like');
            likeButton.innerHTML = `<svg viewBox="0 0 24 24"><path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-1.91l-.01-.01L23 10z"></path></svg>`;
            likeButton.setAttribute('data-message-id', messageId);
            likeButton.addEventListener('click', () => rateMessage(messageId, messageElement, 'like'));
            
            // Кнопка "Не нравится"
            const dislikeButton = document.createElement('button');
            dislikeButton.classList.add('rating-btn', 'dislike');
            dislikeButton.innerHTML = `<svg viewBox="0 0 24 24"><path d="M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v1.91l.01.01L1 14c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.59-6.59c.36-.36.58-.86.58-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z"></path></svg>`;
            dislikeButton.setAttribute('data-message-id', messageId);
            dislikeButton.addEventListener('click', () => rateMessage(messageId, messageElement, 'dislike'));
            
            // Текст благодарности
            const thanksText = document.createElement('span');
            thanksText.classList.add('rating-thanks');
            thanksText.textContent = 'Спасибо за оценку!';
            
            ratingElement.appendChild(likeButton);
            ratingElement.appendChild(dislikeButton);
            ratingElement.appendChild(thanksText);
            
            messageElement.appendChild(ratingElement);
        }
        
        messagesContainer.appendChild(messageElement);
        
        const chatHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');
        chatHistory.push({ 
            text, 
            sender,
            isFirst: isFirst
        });
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
    }
    
    function rateMessage(messageId, messageElement, rating) {
        // Находим элементы оценки
        const ratingElement = messageElement.querySelector('.message-rating');
        const likeButton = ratingElement.querySelector('.like');
        const dislikeButton = ratingElement.querySelector('.dislike');
        const thanksText = ratingElement.querySelector('.rating-thanks');
        
        // Устанавливаем активную кнопку
        if (rating === 'like') {
            likeButton.classList.add('active');
            dislikeButton.classList.remove('active');
        } else {
            dislikeButton.classList.add('active');
            likeButton.classList.remove('active');
        }
        
        // Показываем текст благодарности
        thanksText.classList.add('visible');
        
        // Скрываем текст через 3 секунды
        setTimeout(() => {
            thanksText.classList.remove('visible');
        }, 3000);
        
        // Здесь можно добавить отправку оценки на сервер
        console.log(`Сообщение ${messageId} оценено: ${rating}`);
        
        // В реальном приложении здесь будет отправка данных на сервер
        /*
        fetch('/rate-message', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message_id: messageId,
                rating: rating,
                user_id: userId,
                prompt_type: selectedPrompt
            })
        });
        */
    }
    
    async function resetChat(switchingPrompt = false) {
        if (!switchingPrompt && !confirm('Вы уверены, что хотите сбросить историю чата?')) {
            return;
        }
        
        messagesContainer.innerHTML = '';
        
        localStorage.removeItem('chatHistory');
        localStorage.removeItem('chatInitialized');
        
        if (userId) {
            try {
                await fetch('/reset_session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 
                        user_id: userId,
                        prompt_type: selectedPrompt
                    })
                });
                
                if (!switchingPrompt) {
                    localStorage.removeItem('clinicUserId');
                    userId = null;
                }
                
            } catch (error) {
                console.error('Ошибка при сбросе сессии:', error);
            }
        }
        
        showWelcomeMessage();
        
        // Обновляем статус после сброса
        updateSystemStatus();
    }
    
    function loadChatHistory() {
        const chatHistory = JSON.parse(localStorage.getItem('chatHistory') || '[]');
        
        // Если история чата была создана с другим промтом, сбрасываем её
        const storedPromptType = localStorage.getItem('chatHistoryPromptType');
        if (storedPromptType && storedPromptType !== selectedPrompt) {
            // Очищаем историю и показываем новое приветственное сообщение с выбранным промтом
            localStorage.removeItem('chatHistory');
            localStorage.removeItem('chatHistoryPromptType');
            showWelcomeMessage();
            return;
        }
        
        chatHistory.forEach(message => {
            const messageElement = document.createElement('div');
            messageElement.classList.add('message', `message-${message.sender}`);
            if (message.isFirst) {
                messageElement.classList.add('message-first');
            }
            
            // Преобразуем переносы строк в <br> для корректного отображения
            const formattedText = message.text.replace(/\n/g, '<br>');
            messageElement.innerHTML = formattedText;
            
            // Добавляем элементы оценки только для сообщений ассистента (кроме первого)
            if (message.sender === 'assistant' && !message.isFirst) {
                const ratingElement = document.createElement('div');
                ratingElement.classList.add('message-rating');
                
                const messageId = Date.now().toString() + Math.random().toString(36).substr(2, 5);
                
                // Кнопка "Нравится"
                const likeButton = document.createElement('button');
                likeButton.classList.add('rating-btn', 'like');
                likeButton.innerHTML = `<svg viewBox="0 0 24 24"><path d="M1 21h4V9H1v12zm22-11c0-1.1-.9-2-2-2h-6.31l.95-4.57.03-.32c0-.41-.17-.79-.44-1.06L14.17 1 7.59 7.59C7.22 7.95 7 8.45 7 9v10c0 1.1.9 2 2 2h9c.83 0 1.54-.5 1.84-1.22l3.02-7.05c.09-.23.14-.47.14-.73v-1.91l-.01-.01L23 10z"></path></svg>`;
                likeButton.setAttribute('data-message-id', messageId);
                likeButton.addEventListener('click', () => rateMessage(messageId, messageElement, 'like'));
                
                // Кнопка "Не нравится"
                const dislikeButton = document.createElement('button');
                dislikeButton.classList.add('rating-btn', 'dislike');
                dislikeButton.innerHTML = `<svg viewBox="0 0 24 24"><path d="M15 3H6c-.83 0-1.54.5-1.84 1.22l-3.02 7.05c-.09.23-.14.47-.14.73v1.91l.01.01L1 14c0 1.1.9 2 2 2h6.31l-.95 4.57-.03.32c0 .41.17.79.44 1.06L9.83 23l6.59-6.59c.36-.36.58-.86.58-1.41V5c0-1.1-.9-2-2-2zm4 0v12h4V3h-4z"></path></svg>`;
                dislikeButton.setAttribute('data-message-id', messageId);
                dislikeButton.addEventListener('click', () => rateMessage(messageId, messageElement, 'dislike'));
                
                // Текст благодарности
                const thanksText = document.createElement('span');
                thanksText.classList.add('rating-thanks');
                thanksText.textContent = 'Спасибо за оценку!';
                
                ratingElement.appendChild(likeButton);
                ratingElement.appendChild(dislikeButton);
                ratingElement.appendChild(thanksText);
                
                messageElement.appendChild(ratingElement);
            }
            
            messagesContainer.appendChild(messageElement);
        });
        
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    function saveChatHistory() {
        const messages = messagesContainer.querySelectorAll('.message');
        const chatHistory = [];
        
        messages.forEach(message => {
            const sender = message.classList.contains('message-user') ? 'user' : 'assistant';
            const isFirst = message.classList.contains('message-first');
            
            // Удаляем блок рейтинга перед сохранением текста
            const messageTextContent = message.innerHTML;
            const textWithoutRating = messageTextContent.split('<div class="message-rating">')[0];
            
            chatHistory.push({
                text: textWithoutRating.replace(/<br>/g, '\n'),
                sender,
                isFirst
            });
        });
        
        localStorage.setItem('chatHistory', JSON.stringify(chatHistory));
        // Сохраняем тип промта, с которым была создана история чата
        localStorage.setItem('chatHistoryPromptType', selectedPrompt);
    }
    
    /**
     * Обновляет информацию о статусе системы
     */
    function updateSystemStatus() {
        fetch('/health')
            .then(response => {
                console.log('Ответ от сервера /health:', response.status);
                return response.json();
            })
            .then(data => {
                console.log('Данные статуса:', data);
                if (data.status === 'ok') {
                    assistantStatus.innerHTML = `
                        <span class="status-indicator online"></span>
                        Онлайн
                    `;
                } else {
                    assistantStatus.innerHTML = `
                        <span class="status-indicator offline"></span>
                        Офлайн
                    `;
                }
                
                activeSessions.textContent = data.active_sessions;
                lastUpdated = new Date().toLocaleTimeString();
                lastUpdateTime.textContent = lastUpdated;
                
                // Обновляем логи при обновлении статуса
                fetchLogs();
            })
            .catch(error => {
                console.error('Ошибка при получении статуса системы:', error);
                
                // Если эндпоинт /health недоступен, попробуем проверить, работает ли API через
                // запрос к корневому URL или предположим, что сервер онлайн
                assistantStatus.innerHTML = `
                    <span class="status-indicator warning"></span>
                    Статус недоступен
                `;
                
                // Все равно пытаемся обновить логи
                if (typeof fetchLogs === 'function') {
                    fetchLogs();
                }
            });
    }
    
    function updateUserIdDisplay() {
        if (userId) {
            currentUserIdElement.textContent = userId;
        } else {
            currentUserIdElement.textContent = 'Не установлен';
        }
    }
    
    if (editUserIdBtn) {
        editUserIdBtn.addEventListener('click', function() {
            userIdEditor.style.display = 'flex';
            userIdInput.value = userId || '';
            userIdInput.focus();
        });
    }
    
    if (saveUserIdBtn) {
        saveUserIdBtn.addEventListener('click', function() {
            const newUserId = userIdInput.value.trim();
            if (newUserId) {
                userId = newUserId;
                localStorage.setItem('clinicUserId', userId);
                updateUserIdDisplay();
                userIdEditor.style.display = 'none';
                
            
                resetChat(true);
            }
        });
    }
    
    if (cancelUserIdBtn) {
        cancelUserIdBtn.addEventListener('click', function() {
            userIdEditor.style.display = 'none';
        });
    }
    
    updateUserIdDisplay();

    // Функция для загрузки логов с сервера
    function loadLogs() {
        logsContainer.innerHTML = '<p class="loading-logs">Загрузка логов...</p>';
        
        fetch('/logs')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Ошибка HTTP: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                logsContainer.innerHTML = '';
                
                if (data.logs && data.logs.length > 0) {
                    data.logs.forEach(log => {
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry';
                        
                        // Определить класс в зависимости от уровня лога
                        if (log.toLowerCase().includes('error') || log.toLowerCase().includes('ошибка')) {
                            logEntry.classList.add('log-error');
                        } else if (log.toLowerCase().includes('warning') || log.toLowerCase().includes('предупреждение')) {
                            logEntry.classList.add('log-warning');
                        } else {
                            logEntry.classList.add('log-info');
                        }
                        
                        logEntry.textContent = log;
                        logsContainer.appendChild(logEntry);
                    });
                } else {
                    logsContainer.innerHTML = '<p class="no-logs">Логи отсутствуют</p>';
                }
            })
            .catch(error => {
                console.error('Ошибка при загрузке логов:', error);
                logsContainer.innerHTML = `<p class="log-error">Ошибка при загрузке логов: ${error.message}</p>`;
            });
    }
    
    // Обновление поля user_id (единый ключ 'clinicUserId')
    function updateUserIdField() {
        // Получаем или создаем user_id
        let id = localStorage.getItem('clinicUserId') || generateUserId();
        localStorage.setItem('clinicUserId', id);
        userId = id;
        userIdField.value = id;
        updateUserIdDisplay();
    }
    
    // Сохранение user_id (единый ключ 'clinicUserId')
    function saveUserId() {
        const newUserId = userIdField.value.trim();
        if (!newUserId) {
            showNotification('ID пользователя не может быть пустым', 'error');
            return;
        }
        const oldUserId = localStorage.getItem('clinicUserId');
        localStorage.setItem('clinicUserId', newUserId);
        userId = newUserId;
        updateUserIdDisplay();
        // Сбросить сессию при изменении ID
        if (oldUserId !== newUserId) {
            fetch('/reset_session', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ user_id: newUserId })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error('Ошибка при сбросе сессии');
                }
                showNotification('ID пользователя сохранен и сессия сброшена', 'success');
            })
            .catch(error => {
                console.error('Ошибка:', error);
                showNotification('Произошла ошибка: ' + error.message, 'error');
            });
        } else {
            showNotification('ID пользователя сохранен', 'success');
        }
    }
    
    // Сброс user_id (единый ключ 'clinicUserId')
    function resetUserId() {
        const newUserId = generateUserId();
        localStorage.setItem('clinicUserId', newUserId);
        userId = newUserId;
        updateUserIdField();
        updateUserIdDisplay();
        // Сбросить сессию
        fetch('/reset_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ user_id: newUserId })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Ошибка при сбросе сессии');
            }
            showNotification('Создан новый ID пользователя и сессия сброшена', 'success');
        })
        .catch(error => {
            console.error('Ошибка:', error);
            showNotification('Произошла ошибка: ' + error.message, 'error');
        });
    }
    
    // Показать уведомление
    function showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Удалить предыдущие уведомления
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(n => n.remove());
        
        document.body.appendChild(notification);
        
        // Удалить уведомление через 3 секунды
        setTimeout(() => {
            notification.classList.add('fadeOut');
            setTimeout(() => {
                notification.remove();
            }, 500);
        }, 3000);
    }
    
    // Инициализация обработчиков событий
    document.addEventListener('DOMContentLoaded', function() {
        updateUserIdField();
        loadLogs();
        
        saveUserIdBtn.addEventListener('click', saveUserId);
        resetUserIdBtn.addEventListener('click', resetUserId);
        refreshLogsBtn.addEventListener('click', loadLogs);
    });

    // Генерация уникального ID пользователя
    function generateUserId() {
        return 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }
}); 