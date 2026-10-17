# Verity AI Router

Локальный AI-роутер на FastAPI, который объединяет языковые модели и синтез речи
под одним HTTP API. Провайдеры подключаются как отдельные модули и
обнаруживаются автоматически. Это нужно для запуска Verity JE, локально или подключая другие API.

## Возможности

- Единый OpenAI-совместимый endpoint для чата.
- Выбор провайдера по имени модели: DeepSeek или Ollama.
- Стриминг ответов.
- Два варианта синтеза речи: локальный Piper и удалённый TTS-сервер по HTTP API.
- Модульная архитектура: новые языковые и голосовые провайдеры добавляются
  отдельными пакетами.
- Настройка через `.env`.
- Автоматическая документация Swagger: `/docs`.

## Требования

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Ollama — для локальных языковых моделей (необязательно)
- API-ключ DeepSeek — для облачных моделей (необязательно)
- Голосовые модели Piper в `src/verity_ai/data/tts`

Каждая голосовая модель состоит из пары файлов:

```text
src/verity_ai/data/tts/irina.onnx
src/verity_ai/data/tts/irina.json
```

Имя файла без расширения используется как имя голоса (`irina`, `ruslan`).

## Установка

```bash
uv sync
```

Создание `.env`:

```bash
uv run python scripts/create_env.py
```

Скрипт создаёт файл с настройками по умолчанию. В интерактивном режиме он
запрашивает значения, уже существующий `.env` перезаписывается только с флагом
`--force`. Шаблон также доступен в `.env.example`.

## Тесты

```bash
uv run pytest
```

Тесты используют моки для HTTP-запросов и не требуют запущенных Ollama,
DeepSeek, Piper или TTS-сервера.

## Настройка

| Переменная | По умолчанию | Назначение |
| --- | --- | --- |
| `HOST` | `127.0.0.1` | Адрес сервера |
| `PORT` | `8880` | Порт сервера |
| `DEEPSEEK_API_KEY` | — | API-ключ DeepSeek |
| `DEEPSEEK_URL` | `https://api.deepseek.com/chat/completions` | Endpoint DeepSeek |
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/chat` | Endpoint Ollama |
| `REQUEST_TIMEOUT` | `300` | Таймаут запросов к провайдерам, секунды |
| `DEFAULT_MODEL` | `deepseek-flash` | Модель по умолчанию |
| `DEFAULT_TTS_VOICE` | `irina` | Голос по умолчанию |
| `TTS_MODEL_DIR` | `src/verity_ai/data/tts` | Каталог голосовых моделей |
| `LLM_MODEL_DIR` | `src/verity_ai/data/llm` | Каталог локальных языковых моделей |
| `TTS_API_URL` | — | URL удалённого TTS endpoint; пустое значение отключает провайдер |
| `TTS_API_KEY` | — | Bearer-токен для удалённого TTS |
| `TTS_API_TIMEOUT` | `30` | Таймаут запросов к удалённому TTS, секунды |
| `TTS_API_VOICE` | `default` | Голос удалённого TTS по умолчанию |
| `TTS_API_VOICES` | — | Список голосов удалённого TTS через запятую |
| `TTS_API_TRUST_ENV` | `false` | Использовать системные proxy-переменные для удалённого TTS |

`.env` добавлен в `.gitignore` и не должен попадать в репозиторий.

## Запуск

```bash
uv run main.py
```

После запуска:

- API: `http://127.0.0.1:8880`
- Swagger: `http://127.0.0.1:8880/docs`
- Список модулей: `http://127.0.0.1:8880/api/ai/modules`

## API

### Чат

```bash
curl http://127.0.0.1:8880/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-flash",
    "messages": [{"role": "user", "content": "Привет"}]
  }'
```

Поддерживаемые пути:

- `POST /api/chat`
- `POST /v1/chat/completions`
- `POST /chat/completions`

Для стриминга передайте `"stream": true`. DeepSeek отвечает в формате SSE,
Ollama — в формате NDJSON.

Псевдонимы моделей:

| Провайдер | Модели |
| --- | --- |
| DeepSeek | `deepseek`, `deepseek-chat`, `deepseek-coder`, `deepseek-flash` |
| Ollama | `qwen1` → `qwen2.5:3b`, `qwen2` → `qwen2.5:7b`, `qwen2.5:3b`, `llama` / `llama3.2` → `llama3.2` |

Для Ollama также поддерживается префикс `ollama-`, например `ollama-qwen1`.

### Модели

```bash
curl http://127.0.0.1:8880/v1/models
curl http://127.0.0.1:8880/api/tags
```

### Синтез речи

```bash
curl http://127.0.0.1:8880/v1/audio/voices
```

```bash
curl -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Привет, это Verity AI", "voice": "irina"}' \
  --output speech.wav
```

Ответ всегда отдаётся в формате WAV.

### Удалённый TTS

Второй вариант использования TTS — внешний сервер, который принимает
OpenAI-совместимый запрос. Провайдер включается, когда задан `TTS_API_URL`:

```dotenv
TTS_API_URL=http://127.0.0.1:8080/v1/audio/speech
TTS_API_KEY=secret
TTS_API_VOICES=alice,bob
TTS_API_VOICE=alice
```

Сервер должен принимать `POST` с JSON:

```json
{
  "input": "Текст для синтеза",
  "voice": "alice",
  "response_format": "wav"
}
```

В ответ ожидаются байты аудио. Если задан `TTS_API_KEY`, он передаётся в заголовке
`Authorization: Bearer <key>`. Голоса из `TTS_API_VOICES` добавляются в общий
список `/v1/audio/voices`, после чего их можно использовать в
`/v1/audio/speech`.

По умолчанию удалённый провайдер не учитывает системные proxy-переменные.
Включить их можно через `TTS_API_TRUST_ENV=true`.

## Структура проекта

```text
.
├── main.py                         # точка входа
├── pyproject.toml
├── .env.example
├── scripts/
│   └── create_env.py               # генератор .env
└── src/
    └── verity_ai/
        ├── app.py                  # FastAPI-приложение
        ├── config.py               # настройки из .env
        ├── registry.py             # автообнаружение модулей
        ├── core/
        │   ├── llm/
        │   │   ├── provider.py     # интерфейс языкового провайдера
        │   │   └── providers.py    # поиск провайдеров
        │   └── tts/
        │       ├── api.py          # база для удалённых TTS-провайдеров
        │       ├── voice.py        # интерфейс голосового провайдера
        │       └── voices.py       # поиск голосовых провайдеров
        ├── modules/
        │   ├── llm_modules/
        │   │   ├── deepseek/       # провайдер DeepSeek
        │   │   ├── ollama/         # провайдер Ollama
        │   │   └── llm/            # общий роутер чата
        │   └── tts_modules/
        │       ├── piper/          # локальный провайдер Piper
        │       ├── remote/         # удалённый TTS по HTTP API
        │       └── tts/            # общий роутер синтеза речи
        └── data/
            ├── llm/                # локальные языковые модели
            └── tts/                # голосовые модели Piper
```

Каталоги `data/` (голосовые модели Piper и локальные веса LLM) исключены из Git
через `.gitignore` и не попадают в коммиты.

`registry.py` обходит пакеты `llm_modules` и `tts_modules`, импортирует модули и
включает их роутеры в общее приложение. Провайдеры регистрируются через
атрибуты `provider` или `voice_provider`.

## Добавление провайдера

### Языковая модель

1. Создайте пакет `src/verity_ai/modules/llm_modules/myprovider/`.
2. Реализуйте класс-наследник `verity_ai.core.llm.AIProvider` в `provider.py`.
3. В `__init__.py` экспортируйте экземпляр и метаданные:

```python
from verity_ai.modules.llm_modules.myprovider.provider import MyProvider

provider = MyProvider()

NAME = "myprovider"
TITLE = "My Provider"
DESCRIPTION = "Описание провайдера"
KIND = "llm"
```

Провайдер должен реализовать методы `supports`, `resolve_model`, `chat`,
`aliases`, `models` и `tags`. Общий роутер `llm` сам направит запрос в нужный
провайдер по имени модели.

### Голосовая модель

1. Создайте пакет `src/verity_ai/modules/tts_modules/myvoice/`.
2. Реализуйте класс-наследник `verity_ai.core.tts.VoiceProvider` в `provider.py`.
3. В `__init__.py` экспортируйте экземпляр:

```python
from verity_ai.modules.tts_modules.myvoice.provider import MyVoiceProvider

voice_provider = MyVoiceProvider()

NAME = "myvoice"
TITLE = "My Voice"
DESCRIPTION = "Описание голосового провайдера"
KIND = "tts"
```

Провайдер должен реализовать методы `voices` и `synthesize`. Общий роутер `tts`
сам добавит голоса в список и будет использовать нужный провайдер.

Для провайдеров, работающих через HTTP API, используйте базу
`verity_ai.core.tts.APIVoiceProvider`: она инкапсулирует заголовки
авторизации, таймауты, обработку ошибок и методы `_post_json` / `_get_json`.

## Проверка Ollama

Для локальных моделей Ollama должна быть запущена:

```bash
ollama serve
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull llama3.2
```

Если Ollama не запущена, запросы к её моделям вернут ошибку `502`, остальные
возможности продолжат работать.
