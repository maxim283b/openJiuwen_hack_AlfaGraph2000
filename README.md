# AlfaGraph2000 — openJiuwen Challenge 2026

Агент на openJiuwen. Выбирает задачи и прогнозирует оставшийся расход токенов.
Использует разрешённые модели через API организаторов.

## Запуск

```bash
docker build -t alfagraph2000 .
export LLM_API_KEY='ваш-ключ'
docker run --rm --read-only --tmpfs /tmp:rw,size=2g \
  --cpus 4 --memory 16g \
  -e LLM_API_KEY -e BUDGET=100000 \
  -v /absolute/path/to/tasks:/data/tasks:rw \
  alfagraph2000 --tasks-dir /data/tasks
```

Вход: каталоги задач с `task.json` и `workspace/`.
Ответ: `answer.txt` рядом с `task.json`; исправления coding — в `workspace/`.
Для невыполнимых задач файл ответа не создаётся.
Отчёт и траектории: `TASKS/.harness/`.

Параметры:

- `LLM_API_KEY` — ключ API; передаётся только при запуске.
- `LLM_BASE_URL` — по умолчанию `https://api.neuraldeep.ru/v1`.
- `BUDGET` — общий бюджет; 10 входных токенов равны одному выходному.
- `AGENT_CONCURRENCY` — по умолчанию 3, максимум 5.

Зависимости закреплены в `requirements.lock`. Локальные веса LLM не нужны.
`--offline` запускает только локальные решатели.
