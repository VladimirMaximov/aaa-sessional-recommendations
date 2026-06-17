# Tinder-like сессионные рекомендации

**Команда «Пузырь, но с вайбом»**  
Максим Кочнов (капитан), Владимир Максимов, Александр Фридман

---

## Идея

Мы рассматриваем сервис не как новую рекомендательную систему, а как обёртку поверх неё.

Выдача уже сформирована базовой RS Авито — мы предполагаем, что она хорошо знает долгосрочные интересы пользователя. Поверх этой выдачи мы ловим сильный явный сигнал (лайк / скип) и динамически адаптируем оставшуюся часть выдачи под текущий интерес сессии.

> Можно ли с помощью нескольких лайков пользователя поднять наиболее релевантные объявления выше в выдаче?

### Почему это полезно

- **Cold start / onboarding** — Tinder-like интерфейс быстро собирает несколько сильных сигналов.
- **Уточнение текущего интереса** — пользователь обычно смотрит автомобили, но сейчас ищет коляску. Лайки скорректируют выдачу под текущую задачу без изменения долгосрочного профиля.
- **Выход из filter bubble** — итоговая выдача сочетает исходный RS-порядок, сессионный сигнал и exploration-компонент через ANN.

---

## Результаты экспериментов

Оффлайн-оценка на тестовой части датасета. Для каждой сессии каждый лайк на позиции k — момент реранкинга: кандидаты k+1..N переставляются, задача — поднять следующие лайки выше.

| Метод | NDCG@10 | mean rank lift | pct\_lifted |
| --- | --- | --- | --- |
| Исходный порядок (baseline) | 0.1985 | — | — |
| Cosine similarity по эмбеддингу | 0.3036 | +5.8 | 71% |
| CatBoost YetiRank | **0.3444** | **+7.1** | **78%** |

CatBoost обучен на парах (лайкнутый айтем, кандидат) с YetiRank-лоссом. Фичи: косинусное сходство с лайкнутым айтемом, 80 числовых признаков кандидата, совпадение по трём уровням категорий. NDCG@10 вырос с 0.20 до 0.34 — почти в 1.7 раза относительно исходного порядка.

---

## Архитектура приложения

```text
Базовая RS (Авито)
        ↓
Готовая выдача кандидатов (test.parquet)
        ↓
Tinder-like интерфейс
        ↓
Лайк / Скип пользователя
        ↓
EMA-реранкинг (CatBoost YetiRank + cosine fallback)
        ↓
ANN retrieval (FAISS OPQ+IVFPQ — новые кандидаты вне исходной выдачи)
        ↓
Обновлённый порядок показа
```

**Стек:**

- **FastAPI** — REST API, `uvicorn`
- **Redis** — хранение истории сессии с TTL
- **CatBoost** — реранкер, загружается из S3 при старте
- **FAISS OPQ+IVFPQ** — ANN-индекс над 121 505 мультимодальными эмбеддингами (768d, изображение + текст), загружается из S3
- **Polars** — обработка данных
- **S3 (Beget)** — хранение всех артефактов (данные, эмбеддинги, модели, индекс)
- **Docker Compose** — оркестрация Redis + API

**EMA-реранкинг.** При каждом запросе ленты история сессии реплеируется: на каждом шаге скоры кандидатов обновляются по формуле `scores = (1 − w) × scores + w × new_scores`.  `new_scores` — нормированные скоры CatBoost (или cosine, если модель недоступна).

**ANN.** После каждых N лайков FAISS-индекс возвращает K ближайших соседей к усреднённому эмбеддингу лайкнутых айтемов. Новые кандидаты примешиваются в пул и реранжируются вместе с исходной выдачей.

---

## Запуск локально

Приложение: **<http://localhost:8000/ui>**

### Требования

- Docker + Docker Compose
- (Опционально, для ноутбуков) Python 3.13+, [uv](https://github.com/astral-sh/uv)

---

### Вариант 1 — с S3 (prod-конфиг)

Все артефакты (данные, эмбеддинги, модели) хранятся в S3 и загружаются при старте.

```bash
git clone <repo-url>
cd SessionRec
cp .env.example .env  # заполнить S3_BUCKET, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY
docker compose up --build
```

---

### Вариант 2 — без S3, локальные файлы

Приложение умеет работать без S3: если переменные S3 не заданы, оно ищет файлы в директории `LOCAL_DATA_DIR` (по умолчанию `./data`).

#### Шаг 1 — Подготовить данные

Исходные данные на Kaggle:

- [table-data-train-val-test](https://www.kaggle.com/datasets/michosi/table-data-train-val-test) — табличные фичи (train/val/test parquet)
- [items-part-1](https://www.kaggle.com/datasets/michosi/items-part-1) и [items-part-2](https://www.kaggle.com/datasets/michosi/items-part-2) — изображения и тексты айтемов

#### Шаг 2 — Разложить файлы

Минимальная структура для запуска (только лента + dummy-эмбеддинги):

```text
data/
└── feed.parquet          # переименованный test/test.parquet с Kaggle
```

Полная структура (лента + реранкер + ANN):

```text
data/
├── feed.parquet              # test/test.parquet
├── item_embeddings.parquet   # из notebooks/extract_embeddings.ipynb
├── item_catalog.parquet      # из notebooks/extract_embeddings.ipynb
└── models/
    ├── reranker_catboost.cbm # из notebooks/reranker.ipynb
    ├── faiss_ivfpq.index     # из notebooks/build_faiss_ann.ipynb
    └── faiss_item_ids.npy    # из notebooks/build_faiss_ann.ipynb
```

Если какой-то файл отсутствует — приложение деградирует:

| Отсутствует | Поведение |
| --- | --- |
| `item_embeddings.parquet` | Dummy-эмбеддинги (случайные векторы, реранкинг только по CatBoost) |
| `reranker_catboost.cbm` | Реранкинг по cosine similarity без CatBoost |
| FAISS-файлы | `ANN_ENABLED=false` автоматически, только исходная выдача + реранкинг |

#### Шаг 3 — Запустить

```bash
# Minimal .env (без S3)
cat > .env <<'EOF'
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=86400
FEED_MAX_GROUPS=2000
EMB_DIM=768
ANN_ENABLED=false
IMAGE_BASE_URL=   # изображения не загрузятся без S3/CDN
EOF

docker compose up --build
```

Или без Docker:

```bash
uv sync

docker run -d -p 6379:6379 redis:7-alpine

uv run uvicorn app.main:app --reload --port 8000
```

---

### Переменные окружения

```env
REDIS_URL=redis://localhost:6379/0
SESSION_TTL_SECONDS=86400
FEED_MAX_GROUPS=2000
EMB_DIM=768

# Локальная директория с данными (используется если S3 не настроен)
LOCAL_DATA_DIR=./data

# S3 — если заданы, имеют приоритет над локальными файлами
S3_ENDPOINT_URL=https://s3.ru1.storage.beget.cloud
S3_BUCKET=<bucket>
S3_ACCESS_KEY_ID=<key-id>
S3_SECRET_ACCESS_KEY=<secret>
S3_REGION=ru-1
S3_PRESIGN_TTL_SECONDS=3600

# S3 ключи объектов
FEED_S3_KEY=val/val.parquet
EMB_S3_KEY=embeddings/item_embeddings.parquet
RERANKER_S3_KEY=models/reranker_catboost.cbm
FAISS_INDEX_S3_KEY=models/faiss_ivfpq.index
FAISS_IDS_S3_KEY=models/faiss_item_ids.npy
CATALOG_S3_KEY=item_catalog/item_catalog.parquet

# Публичный префикс для изображений
IMAGE_BASE_URL=https://s3.ru1.storage.beget.cloud/<bucket>

# ANN
ANN_ENABLED=true
ANN_NPROBE=32
ANN_POOL_N=400
ANN_EXPLOIT_K=6
ANN_EVERY=3
ANN_RECENT_LIKES=5
```

---

### Ноутбуки

Исследовательские ноутбуки в `notebooks/`. Для запуска нужны dev-зависимости:

```bash
uv sync --group dev
uv run jupyter lab
```

Основной ноутбук — `notebooks/reranker.ipynb`: обучение CatBoost YetiRank.
