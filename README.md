# Engine checker — AI Business SPB Hackathon 2026

Система акустической диагностики двигателей на основе AI. Принимает WAV-записи работы двигателя и возвращает `anomaly_score` — числовой скор аномальности чанка.

- **Swagger UI:** [http://178.154.233.146:8000/docs](http://178.154.233.146:8000/docs)
- **Метрики Prometheus:** [http://178.154.233.146:8000/metrics](http://178.154.233.146:8000/metrics)

---

## Как это работает

```
WAV-файл (5 сек, 48kHz, stereo, 24-bit)
        │
        ▼
┌───────────────────────────────┐
│        FastAPI Backend        │
│                               │
│  POST audio/classify          │
│  POST audio/classify-batch    │
│                               │
│  ┌─────────────────────────┐  │
│  │     AudioService        │  │
│  │  валидация + батчинг    │  │
│  └────────────┬────────────┘  │
│               │               │
│  ┌────────────▼────────────┐  │
│  │   AudioClassifier       │  │
│  │   PredictionModel       │  │
│  │                         │  │
│  │  mono = stereo.mean()   │  │
│  │  score = percentile(    │  │
│  │    |mono|, 99)          │  │
│  └────────────┬────────────┘  │
│               │               │
│  ┌────────────▼────────────┐  │
│  │  Prometheus Metrics     │  │
│  │  anomaly_score_hist     │  │
│  │  anomaly_detected_total │  │
│  │  files_processed_total  │  │
│  └─────────────────────────┘  │
└───────────────────────────────┘
        │
        ▼
{
  "anomaly_score": 0.36,
  "label": "normal",
  "rpm_estimate": null,
  "model_version": "1.0.0"
}
```

### Модель: 99-й перцентиль амплитуды

Подход — **unsupervised, без обучения**. Обучающие данные содержат только нормальные записи, поэтому классический supervised-подход неприменим.

Ключевое наблюдение: аномальные двигатели (стук подшипника, задевание, нестабильная работа цилиндра) генерируют **пиковые импульсы** поверх нормального сигнала. 99-й перцентиль абсолютной амплитуды улавливает эти выбросы лучше, чем RMS (среднеквадратичное), которое сглаживает кратковременные события.

```python
mono = audio.mean(axis=1)           # stereo → mono
score = np.percentile(|mono|, 99)   # топ 1% амплитудных пиков
```

**Результат на тест-сете:**
```
ROC-AUC:        0.7311
Recall@FPR5%:   0.2286
Combined score: 0.5301
```

---

## Карта репозитория

```
.
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── v1/
│   │       └── audio.py          # роутеры: /classify, /classify-batch
│   ├── core/
│   │   └── metrics.py            # Prometheus метрики
│   ├── schemas/
│   │   └── audio.py              # Pydantic схемы запросов/ответов
│   ├── services/
│   │   ├── audio_service.py      # бизнес-логика, батчинг, работа с ZIP
│   │   └── classifier.py        # обёртка над PredictionModel
│   └── main.py                   # FastAPI app, CORS, Prometheus
│
├── solution.py                   # ML-модель (PredictionModel)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## API

### `POST /api/v1/audio/classify`

Классификация одного WAV-файла.

**Параметры запроса** (`multipart/form-data`):

| Поле | Тип | Обязательный | Описание |
|---|---|---|---|
| `file` | WAV | ✅ | Запись двигателя, 48kHz 24-bit stereo |
| `vehicle_id` | string | ❌ | Идентификатор машины, например `BUS-042` |
| `segment_type` | enum | ❌ | `idle` / `high_hold` / `background` |
| `duration_sec` | float | ❌ | Длительность сегмента в секундах |

**Response:**
```json
{
  "anomaly_score": 0.36,
  "label": "normal",
  "rpm_estimate": null,
  "model_version": "1.0.0"
}
```

**curl-пример:**
```bash
curl -X POST http://178.154.233.146:8000/api/v1/audio/classify \
     -F "file=@engine.wav" \
     -F "vehicle_id=BUS-042" \
     -F "segment_type=idle" \
     -F "duration_sec=5.0"
```

---

### `POST /api/v1/audio/classify-batch`

Классификация пачки файлов через ZIP-архив. Максимум 10 000 файлов, до 5 GB.

**Параметры запроса** (`multipart/form-data`):

| Поле | Тип | Обязательный | Описание |
|---|---|---|---|
| `file` | ZIP | ✅ | Архив с WAV-файлами |
| `vehicle_id` | string | ❌ | Идентификатор машины |
| `segment_type` | enum | ❌ | `idle` / `high_hold` / `background` |

**Response:**
```json
{
  "items": [
    {
      "filename": "chunk_001.wav",
      "anomaly_score": 0.67,
      "label": "anomaly",
      "rpm_estimate": null,
      "model_version": "1.0.0",
      "error": null
    },
    {
      "filename": "chunk_002.wav",
      "anomaly_score": 0.28,
      "label": "normal",
      "rpm_estimate": null,
      "model_version": "1.0.0",
      "error": null
    }
  ],
  "total": 2,
  "successful": 2,
  "failed": 0
}
```

**curl-пример:**
```bash
curl -X POST http://178.154.233.146:8000/api/v1/audio/classify-batch \
     -F "file=@chunks.zip" \
     -F "vehicle_id=BUS-042" \
     -F "segment_type=idle"
```

---

### `GET /ping`

Healthcheck.
```json
{"message": "pong"}
```

---

### `GET /metrics`

Prometheus-метрики для мониторинга:

| Метрика | Тип | Описание |
|---|---|---|
| `audio_anomaly_score` | Histogram | Распределение anomaly_score |
| `audio_anomaly_detected_total` | Counter | Всего обнаружено аномалий |
| `audio_files_processed_total` | Counter | Обработано файлов (по статусу) |
| `http_request_duration_seconds` | Histogram | Латентность по роутам |

---

## Зависимости

| Пакет | Назначение |
|---|---|
| `fastapi` | Web-фреймворк |
| `uvicorn` | ASGI-сервер |
| `soundfile` | Чтение WAV-файлов |
| `numpy` | Вычисление перцентиля |
| `prometheus-fastapi-instrumentator` | Метрики |
| `pydantic` | Валидация схем |
| `python-multipart` | Загрузка файлов |

---

## Деплой

### Через Docker Compose (рекомендуется)

Приложение упаковано в Docker-образ и запускается одной командой. Все зависимости устанавливаются внутри контейнера через `uv`. Образ собирается в два слоя для эффективного кэширования — зависимости пересобираются только при изменении `pyproject.toml`, а не при каждом изменении кода.

```bash
# 1. Клонируй репозиторий
git clone https://github.com/Wertiba/AI-Business-SPB-Hackathon-Backend
cd AI-Business-SPB-Hackathon-Backend

# 2. Создай .env файл
cp docs/.env.example .env

# 3. Собери и запусти
docker compose up -d --build

# 4. Проверь что работает
curl http://localhost:8000/ping
```

Swagger будет доступен по адресу: [http://localhost:8000/docs](http://localhost:8000/docs)

### Обновление на сервере

```bash
git pull origin main
docker compose down
docker compose up -d --build
```

### Локально без Docker

```bash
# Установи uv если нет
pip install uv

# Установи зависимости
uv sync

# Запусти сервер
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Healthcheck

Docker Compose автоматически проверяет состояние сервиса каждые 30 секунд:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/ping"]
  interval: 30s
  timeout: 10s
  retries: 3
```

Если сервис падает — контейнер помечается как `unhealthy`. Для автоматического перезапуска добавь в `docker-compose.yml`:

```yaml
restart: unless-stopped
```
