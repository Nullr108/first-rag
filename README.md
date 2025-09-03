# First RAG Project

Простой проект на базе RAG (Retrieval-Augmented Generation) для работы с текстовыми данными.

## Функциональность
- Загрузка и обработка веб-страниц
- Разделение текста на чанки
- Векторизация текста с использованием Sentence Transformers
- Поиск по семантически близким документам
- Генерация ответов с помощью языковой модели

## Установка
1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/Nullr108/first-rag.git
   ```
2. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```
3. Создать файл `.env` с переменными окружения (при необходимости):
   ```
   MODEL=your_model_name
   BASE_URL=your_api_url
   API_KEY=your_api_key
   ```

## Использование
Запустите `rag_data.py` чтобы спарсить страницу и создать векторное хранилище:
   ```bash
   python rag_data.py
   ```

Запустите `query_rag.py` для выполнения запросов к данным:
```bash
python query_rag.py
```

## Зависимости
Список зависимостей находится в файле [requirements.txt](requirements.txt).
