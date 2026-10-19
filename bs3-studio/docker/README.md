# BS Profiler 3.1 в Docker

Готовый к запуску BS Profiler 3.1: страница анализа видео и командная строка. Всё работает на одном компьютере с видеокартой NVIDIA и без интернета: все модели уже внутри образов. Страница открывается только на этом компьютере, по адресу `http://localhost:7880`.

## Коротко

```bash
docker load -i bs-profiler-3.1-images.tar
docker compose up -d
```

Затем открыть в браузере `http://localhost:7880`. Остановить: `docker compose down`.

## Что нужно

- **Видеокарта NVIDIA с 24 ГБ памяти** — RTX 3090, 4090 или 5090; проверено на RTX 5090 Laptop. Во время анализа продукт занимает около 15 ГБ видеопамяти.
- **Драйвер NVIDIA 580 или новее**: внутри PyTorch для CUDA 13.0.
- **Docker с поддержкой видеокарты:**
  - Linux — Docker Engine 24 или новее с `docker compose` и NVIDIA Container Toolkit;
  - Windows 11 — Docker Desktop с движком WSL2: поддержка видеокарты в нём уже есть.
- **Оперативная память** — 32 ГБ и больше; продукт занимает около 12 ГБ.
- **Диск** — около 70 ГБ свободного места: архив 19 ГБ, образы после загрузки 26–45 ГБ (больше — у Docker Desktop с хранилищем containerd) и задания (0,35–0,8 ГБ на ролик). После загрузки образов архив можно удалить.
- **Интернет не нужен** ни для установки, ни для работы.

## Что в комплекте

| Файл | Что это |
| --- | --- |
| `bs-profiler-3.1-images.tar` | два образа: `bs-profiler:3.1` (продукт) и `bs-profiler-ollama:3.1` (Ollama с моделью `qwen2.5vl:7b`) |
| `docker-compose.yml` | как их запускать вместе |
| `README.md` | эта инструкция |
| `SHA256SUMS` | контрольные суммы файлов комплекта |

## Установка (один раз)

1. Проверить, что архив скачался целиком:

   ```bash
   sha256sum -c SHA256SUMS
   ```

   В Windows (PowerShell): `Get-FileHash bs-profiler-3.1-images.tar` — сумма должна совпасть со строкой в `SHA256SUMS`.

2. Загрузить образы в Docker (около 20 минут, в конце появятся две строки `Loaded image`):

   ```bash
   docker load -i bs-profiler-3.1-images.tar
   ```

3. Проверить, что Docker видит видеокарту:

   ```bash
   docker run --rm --gpus all bs-profiler:3.1 nvidia-smi
   ```

   Должна появиться таблица с названием видеокарты. Если ошибка — раздел «Частые проблемы».

## Запуск и остановка

Из папки, где лежит `docker-compose.yml`:

```bash
docker compose up -d
```

Первый запуск занимает около минуты. Страница: `http://localhost:7880`. Первый анализ после запуска идёт дольше обычного: модели загружаются в видеопамять. Обработка ролика занимает примерно столько же времени, сколько он длится. Видеокарта обрабатывает один ролик за раз: если анализ уже идёт, страница покажет «В очереди».

```bash
docker compose ps              # что запущено
docker compose logs -f bs3     # журнал работы
docker compose down            # остановить
```

После `docker compose down` задания и журнал сохраняются. Контейнеры перезапускаются сами после перезагрузки компьютера, пока их не остановили командой `down`.

## Где лежат результаты

Задания хранятся в томе Docker `bs-profiler_bs3_data`:

- `web_jobs/` — по папке на каждый анализ: копия ролика, `result.json`, PDF-отчёт;
- `logs/journal.txt` — журнал: кто открыл страницу, какой файл анализировали, итог.

Скопировать их в текущую папку:

```bash
docker compose cp bs3:/home/bs/bs3_data/web_jobs ./web_jobs
docker compose cp bs3:/home/bs/bs3_data/logs/journal.txt .
```

Удалить все задания и журнал вместе с контейнерами: `docker compose down -v`.

## Командная строка

Анализ без страницы. Ролик кладётся в том, результат появляется рядом:

```bash
docker compose cp clip.mp4 bs3:/home/bs/bs3_data/clip.mp4
docker compose exec bs3 bs3 infer /home/bs/bs3_data/clip.mp4 \
  --backend mm --out /home/bs/bs3_data/clip.json
docker compose cp bs3:/home/bs/bs3_data/clip.json .
```

`--backend mm` — модель AMLAI 1.0, `--backend oceanai` — OCEAN-AI. Все параметры: `docker compose exec bs3 bs3 infer --help`.

## Частые проблемы

| Что видно | Что сделать |
| --- | --- |
| `could not select device driver "nvidia"` | Linux: установить NVIDIA Container Toolkit и перезапустить Docker. Windows: обновить Docker Desktop, включить движок WSL2 |
| ошибка CUDA при запуске или анализе | обновить драйвер NVIDIA до 580 или новее |
| нехватка памяти видеокарты (`out of memory`) | закрыть другие программы, которые заняли видеокарту; продукту нужно около 15 ГБ |
| страница не открывается | `docker compose ps`: оба контейнера должны быть `healthy`; подробности — `docker compose logs bs3` |
| порт 7880 занят | в `docker-compose.yml` заменить левую часть `127.0.0.1:7880:7880`, например на `127.0.0.1:7890:7880` |

**Открыть страницу другим компьютерам сети** можно, заменив в `docker-compose.yml` строку `127.0.0.1:7880:7880` на `7880:7880`. Входа по паролю на странице нет: любой в сети увидит загруженные ролики и результаты.

## Что внутри

| Часть | Версия |
| --- | --- |
| BS Profiler | 3.1 (пакет 3.1.0a1), код — коммит `13a0260` ветки `v3` |
| Python и PyTorch | Python 3.12, PyTorch 2.11 для CUDA 13.0 |
| Ollama | 0.40.6 с моделью `qwen2.5vl:7b` |
| Веса AMLAI 1.0 | пять файлов, те же, что в релизе `v1.0.0` |
| Веса OCEAN-AI | для русской речи (MuPTA) и английской (FIV2) |

Модели Hugging Face внутри образа: `openai/whisper-large-v3-turbo` (распознавание речи), `openai/clip-vit-base-patch32`, `laion/clap-htsat-fused`, `michellejieli/emotion_text_classifier` (признаки AMLAI 1.0 и эмоции текста), `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (голос), `trpakov/vit-face-expression` (мимика), `Helsinki-NLP/opus-mt-ru-en` и `opus-mt-en-ru` (перевод), `google-bert/bert-base-multilingual-cased` (OCEAN-AI).

Контрольные суммы весов AMLAI 1.0:

```
796ad7f1441cbb7157ebaccb17a95dd351f430164673aa6e80cd75a7d29f8791  seed1/best.pt
fcf65d3d0077eabdd29429f912bbc7e80d34448b5c9d597a47bce665d2d1e1d3  seed2/best.pt
3697cd53066290ab482bdd03995c3f4da64f08fe57b8ea2d00b0b235855f6183  seed3/best.pt
00a5c108b6b51fd470d48ba1f25a82b44e64a6f4a18115a38e4b5a8bdd704b36  seed4/best.pt
ba8531bea61ed7318d02c86ab99fdb69c3fd4fd927965de1fc4df14f81eb9432  seed42/best.pt
```

## Лицензии

Оценки Big Five получены весами, обученными на некоммерческих данных: AMLAI 1.0 — на First Impressions V2 (CC BY-NC 4.0), веса OCEAN-AI для русской речи — на корпусе MuPTA (только для исследований). Модель голоса audeering — CC BY-NC-SA 4.0, у `emotion_text_classifier` явной лицензии нет. Поэтому комплект — только для исследовательского и некоммерческого использования. Остальное: CLIP и Whisper — MIT; CLAP, `qwen2.5vl:7b`, `vit-face-expression`, BERT — Apache-2.0; перевод opus-mt — CC BY 4.0; Ollama — MIT.

## Как собран комплект

Сборка — из папки `bs3-studio/docker/` репозитория: `stage.sh` готовит контексты сборки из git (только отслеживаемые файлы, без заданий и роликов) и из локальных кэшей моделей машины разработки, затем `docker build` для `Dockerfile` и `ollama/Dockerfile`.
