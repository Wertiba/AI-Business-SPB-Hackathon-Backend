# NeuroICE — AI Business SPB Hackathon 2026

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
│  POST /audio/classify         │
│  POST /audio/classify-batch   │
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
  "anomaly_score": 0.287,
  "result": false,
  "message": "Normal"
}
```

### Модель: 99-й перцентиль амплитуды

Подход — **unsupervised, без обучения**. Обучающие данные содержат только нормальные записи, поэтому классический supervised-подход неприменим.

Ключевое наблюдение: аномальные двигатели (стук подшипника, задевание, нестабильная работа цилиндра) генерируют **пиковые импульсы** поверх нормального сигнала. 99-й перцентиль абсолютной амплитуды улавливает эти выбросы лучше, чем RMS (среднеквадратичное), которое сглаживает кратковременные события.

```python
mono = audio.mean(axis=1)          # stereo → mono
score = np.percentile(|mono|, 99)  # топ 1% амплитудных пиков
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

### `POST /audio/classify`
Классификация одного WAV-файла.

**Request:** `multipart/form-data`, поле `file` — WAV 48kHz

**Response:**
```json
{
  "anomaly_score": 0.287,
  "result": false,
  "message": "Normal"
}
```

**curl-пример:**
```bash
curl -X POST http://178.154.233.146:8000/api/v1/audio/classify \
     -F "file=simple_file.wav"
```

---

### `POST /audio/classify-batch`
Классификация пачки файлов через ZIP-архив.

**Request:** `multipart/form-data`, поле `file` — ZIP с WAV-файлами (до 10 000 файлов, до 5 GB)

**Response:**
```json
{
  "items": [
    {
      "filename": "chunk_001.wav",
      "anomaly_score": 0.312,
      "result": true,
      "message": "Anomaly detected"
    }
  ],
  "total": 10,
  "successful": 10,
  "failed": 0
}
```

**curl-пример:**
```bash
curl -X POST http://178.154.233.146:8000/api/v1/audio/classify-batch  \
     -F "file=chunks.zip"
```

---

### `GET /ping`
Healthcheck.
```json
{"message": "pong"}
```

---

### `GET /metrics`
Prometheus-метрики. Полезные для мониторинга:

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
