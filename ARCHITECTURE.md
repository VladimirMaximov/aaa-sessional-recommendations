# Архитектура

Этот документ описывает, как устроен текущий MVP SessionRec, как движутся данные и где расширять систему.

## Обзор

Сервис — это API сессионных рекомендаций с простым Tinder-like UI.
Состав:

- API (FastAPI) для ленты и взаимодействий
- In-memory кэш каталога (Polars), загружается при старте
- Redis для сырых логов и истории сессий
- Dataset hub с одним датасетом на сегодня: popularity index
- ML-ядро со стратегиями (candgen + ranker), которые подменяются через конфиг
- Статический UI для демо

## Модель данных

Логи должны быть в строго заданной схеме:

```
| timestamp | eid | user_id | item_id | session_id |
|    i64    | u32 |   u32   |   u32   |    i32     |
```

- `timestamp`: в миллисекундах
- `eid`: идентификатор типа события (например, like/skip/click)
- `user_id`: идентификатор пользователя
- `item_id`: идентификатор объявления
- `session_id`: идентификатор сессии внутри пользователя

## Хранилища

### 1) Сырые логи (RedisLogStore)

- Redis list ключ: `events:all`
- Назначение: append-only поток всех событий
- Запись при каждом `POST /api/v1/interact`
- Общий источник для offline-данных и бутстрапа датасетов

Пример события в Redis list:

```json
{
  "timestamp": 1773630581000,
  "eid": 7,
  "user_id": 123,
  "item_id": 456,
  "session_id": "1"
}
```

### 2) Индекс истории сессии (RedisSessionStore)

- Redis list ключ: `session:{user_id}:{session_id}`
- Назначение: быстрый доступ к истории текущей сессии
- TTL применяется на ключ сессии
- Используется в `GET /api/v1/feed` для фильтрации уже просмотренных `item_id`

Это именно индекс сессии, а не отдельная система. Он оптимизирован под быстрые онлайн-чтения.

## Dataset слой (для моделей)

Dataset слой строит форматы данных, удобные для моделей.
Сейчас реализован один датасет:

### PopularityIndex

- Redis ZSET:
  - `popularity:all` (глобальная популярность)
  - `popularity:eid:{eid}` (популярность по типу события)
- Строится из логов при старте
- Обновляется онлайн при каждом взаимодействии
- Используется `TopPopularCandGen`

### DatasetHub

- `DatasetHub.bootstrap(log_df)`: бутстрап (первичное построение) датасетов из исторических логов
- `DatasetHub.publish(event)`: онлайн-обновление датасетов при новых событиях

## ML-ядро

### Интерфейсы

- `BaseCandGen.get_candidates(user_history)`
- `BaseRanker.rank(user_history, candidates)`

### Реализации

- `RandomCandGen`: случайная выборка из каталога
- `TopPopularCandGen`: кандидаты из `PopularityIndex`
- `RandomRanker`: случайное перемешивание

### StrategyRegistry

- Собирает зависимости и инстанцирует стратегии по имени или по пути класса
- Позволяет менять подходы через конфиг без правок API

## Потоки API

### Старт приложения

1. Загружается каталог (parquet или mock)
2. Загружаются логи в `LogCache` из `LOGS_PATH` (если задано)
3. Строится `PopularityIndex` через `DatasetHub.bootstrap(...)`
4. Готовится `StrategyRegistry`

### GET /api/v1/feed

1. Читается история сессии из `RedisSessionStore` по паре `(user_id, session_id)`
2. Выбираются candgen/ranker из конфига или query
3. Генерируются кандидаты и ранжируются
4. Отфильтровываются уже просмотренные `item_id`
5. Данные обогащаются метаданными каталога и возвращаются

### POST /api/v1/interact

1. Запись в `RedisSessionStore`
2. Запись в `RedisLogStore`
3. Публикация события в `DatasetHub` для онлайн-обновления индексов

## Конфигурация

Переменные окружения (см. [app/core/config.py](app/core/config.py)):

```
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=86400
CATALOG_PATH=/path/to/items.parquet
CATALOG_SIZE=500
EVENTS_MAX_LENGTH=100000
LOGS_PATH=/path/to/logs.parquet
LOGS_BOOTSTRAP_RESET=1
CANDGEN_STRATEGY=random|popular|path.to.Class
CANDGEN_FALLBACK=random
RANKER_STRATEGY=random|path.to.Class
POPULAR_EID=7
```

## Точки расширения

### Добавить новый датасет (для новой модели)

1. Создать класс датасета с методами:
   - `bootstrap(df, reset=True)` для исторических логов
   - `on_event(event)` для онлайн-обновления
2. Зарегистрировать его в `DatasetHub` в [app/main.py](app/main.py)
3. Передать его в стратегию через `StrategyRegistry`, если нужно

### Добавить новую стратегию

1. Реализовать новый `BaseCandGen` или `BaseRanker`
2. Сделать конструктор, принимающий нужные зависимости
3. Указать `CANDGEN_STRATEGY` или `RANKER_STRATEGY` как путь до класса

## Текущие ограничения

- Redis — единственное хранилище состояния (нет долговременного хранения логов)
- Нет offline-пайплайна обучения (онлайн MVP)
- UI использует фиксированный тестовый `user_id`

## Ключевые файлы

- API и жизненный цикл: [app/main.py](app/main.py), [app/api/v1.py](app/api/v1.py)
- Лог-хранилище и валидация схемы логов: [app/services/log_store.py](app/services/log_store.py)
- Индекс сессий: [app/services/session_store.py](app/services/session_store.py)
- Dataset слой: [app/services/datasets.py](app/services/datasets.py)
- Стратегии и реестр: [app/ml/strategies.py](app/ml/strategies.py), [app/ml/factory.py](app/ml/factory.py)
- Demo UI: [frontend/index.html](frontend/index.html)
- Конфиг: [app/core/config.py](app/core/config.py)
- Docker: [docker-compose.yml](docker-compose.yml), [Dockerfile](Dockerfile)
