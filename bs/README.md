# bs — Big Five (OCEAN) по видео

Этап 1: обёртка над готовыми весами **OCEAN-AI** (aimclub/OCEANAI, BSD-3): аудио + видео + текст → пять оценок
кажущихся черт личности в диапазоне 0…1 (открытость, добросовестность, экстраверсия, доброжелательность,
эмоциональная стабильность = non-neuroticism). Два набора весов: `fi` (First Impressions V2, английская речь)
и `mupta` (MuPTA, русская речь).

Работает в WSL2 Ubuntu 24.04 на GPU (torch cu130). Исходники лежат в этой папке (Windows), окружение, веса и
кэш — на ext4 в `~/bs/`.

## Установка (один раз, из WSL)

```bash
bash /mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs/scripts/setup_wsl.sh
~/bs/venv/bin/pip install -e /mnt/c/Users/nrsmh/OneDrive/Documents/AMLAI/cloude_projects/code/BS/bs
~/bs/venv/bin/bs setup-weights --lang all      # скачивает веса OCEAN-AI (~250 МБ на корпус) в ~/bs/models
```

## Использование

```bash
# одно видео, речь на английском, транскрипт через Whisper large-v3-turbo
~/bs/venv/bin/bs infer video.mp4 --out result.json

# русская речь: веса MuPTA + перевод транскрипта
~/bs/venv/bin/bs infer video.mp4 --lang ru --out result.json

# готовый транскрипт вместо ASR
~/bs/venv/bin/bs infer video.mp4 --transcript video.txt --out result.json

# папка с видео -> CSV
~/bs/venv/bin/bs infer-dir ./videos --out results.csv

# точность на клипах FIV2 (папка с <stem>.mp4, <stem>.txt, labels.csv; см. scripts/make_eval_set.py)
~/bs/venv/bin/bs eval-fiv2 --dir ~/bs/eval/fi_test200 --out ~/bs/eval/fi_test200_eval.json
~/bs/venv/bin/bs eval-fiv2 --dir ~/bs/eval/fi_test200 --out ... --asr          # Whisper вместо .txt
~/bs/venv/bin/bs eval-fiv2 --backend sslmepr --dir ~/bs/eval/fi_test200 --out ...  # бенчмарк SSL-MEPR
```

Второй бэкенд `--backend sslmepr` — предшественник SSL-MEPR на трёх публичных модальностях (сцена, аудио, текст;
веса лица и тела не опубликованы). Его MCDM-фьюжн без двух модальностей схлопывается, поэтому основной оценкой
служит среднее трёх унимодальных предсказаний; выход MCDM и каждая модальность сохраняются как варианты
(`variant_scores` в JSON, колонки `fusion:*`, `scene:*`, `audio:*`, `text:*` в CSV).

Готовые скрипты для WSL лежат в `scripts/` (`stage1_smoke.sh`, `smoke_asr.sh`, `eval_fi200.sh <backend> <dir> [--asr]`).
Результаты замеров — в `results/`, сводка — в `../STAGE1_REPORT.md`.

Формат `result.json` описан в `../TZ_BigFive_Video_Pipeline.md`, раздел 6. Перцентили считаются относительно
разметки train-сплита FIV2 (`bs_bigfive/fiv2_norms.json`, строится скриптом `scripts/build_norms.py`).

## Что внутри OCEAN-AI

- аудио: openSMILE-признаки + мел-спектрограммы, окна 2 с; видео: MediaPipe FaceMesh (468 точек) и ResNet50
  EmoAffectNet по лицу 224×224 при 5 fps, окна 10 с; текст: LIWC-словарь + BERT multilingual; фьюжн `avt`.
- ASR: transformers Whisper (по умолчанию в библиотеке `whisper-base`, здесь заменён на `large-v3-turbo`).
- Заявленная точность на test FIV2 (ноутбуки документации OCEAN-AI): mACC 0.9245 (три модальности).

## Точность (test FIV2, 1997 клипов, транскрипты Rev, описания из статьи)

| Бэкенд | mACC | mCCC |
| --- | ---: | ---: |
| `oceanai` | 0.9255 | 0.703 |
| `mm` (своя модель, 4 модальности + «собеседование», среднее 5 seed) | 0.9202 | 0.715 |
| `sslmepr` (сцена) | 0.9135 | 0.670 |
| `ensemble --ensemble-members oceanai,mm` (по умолчанию) | **0.9274** | **0.733** |
| `ensemble --ensemble-members oceanai,scene` | 0.9256 | 0.720 |

С Whisper вместо готовых транскриптов mACC ниже примерно на 0.3 п.п.; с описаниями от локальной Ollama вместо
описаний статьи — ещё на 0.1–0.2 п.п. Подробности — `../STAGE1_REPORT.md` и `../STAGE2_REPORT.md`.

## Этап 2: собственная модель (`--backend mm`)

Пакет `bs_bigfive/mm/`: рецепт MM-PSYCHE только для personality. Признаки: 30 кадров → MediaPipe → CLIP (лицо),
CLAP (аудио), EmoRoBERTa (транскрипт и описание поведения), mean‖std; фьюжн MCDM (порт `MultiModalFusionModel_v1`).
Шестой выход — метка ChaLearn «пригласить на собеседование» (`--targets big5+interview`).

```bash
bash scripts/mm_extract_all.sh 6              # признаки для train/dev/test в ~/data/fiv2/features (лица в 6 потоков)
bash scripts/mm_train_all.sh                  # 7 вариантов модели -> ~/bs/mm_runs/<config>/{best.pt,result.json}
~/bs/venv/bin/bs infer video.mp4 --backend mm --mm-ckpt ~/bs/mm_runs/all4_interview/best.pt --out result.json
~/bs/venv/bin/bs explain video.mp4 --mm-ckpt ~/bs/mm_runs/all4_interview/best.pt --out ./explain_out
```

Описание поведения на инференсе генерирует локальная Ollama (`--ollama-model qwen3-vl:30b`, 16 кадров, промпт из
статьи); из WSL сервер доступен по адресу шлюза, бэкенд находит его сам. `bs explain` дополнительно считает вклад
модальностей (Input×Gradient и leave-one-out), ключевые кадры (сохраняет JPEG с рамкой лица) и слова транскрипта
и описания, повлиявшие на каждую оценку.

## Веб-интерфейс

```bash
bash scripts/run_web.sh 7860          # запускает `bs web` в фоне (лог ~/bs/logs/web.log); остановить: pkill -f "bs web"
```

Откройте http://localhost:7860 в браузере Windows. Загрузите видео, выберите язык (en: OCEAN-AI на весах FIV2 +
своя модель; ru: OCEAN-AI на весах MuPTA + своя модель, транскрипт переводится на английский моделью Helsinki
ru→en) и нажмите «Анализировать». Ролики длиннее 30 с режутся на сегменты по 20 с (`bs_bigfive/longvideo.py`):
каждый сегмент анализируется целиком с собственным куском транскрипта (Whisper с временными метками) и своим
описанием поведения, итог — среднее по сегментам с весом по длительности, на странице показывается таймлайн и
разброс; объяснения строятся по сегменту, ближайшему к среднему профилю. В CLI то же: `bs infer ... --segment 20`
(0 отключает).

Кнопка **«Экспорт в PDF»** вверху страницы (активна после анализа) собирает полный отчёт
(`bs_bigfive/pdf_report.py`): метаданные файла (размер, длительность, разрешение, кодеки, дата, SHA-256), параметры
анализа, оценки с перцентилями, таймлайн по сегментам, ключевые кадры картинками, вклад модальностей, слова,
описание поведения, транскрипт с переводом. PDF сохраняется в папке запроса `~/bs/web_jobs/<время>/` и
скачивается браузером. На странице: полоски пяти черт с
перцентилями, «собеседование», описание поведения, транскрипт, ключевые кадры, вклад модальностей, слова с
наибольшим вкладом, оценки участников ансамбля и полный JSON. Каждый запрос сохраняется в
`~/bs/web_jobs/<время>/result.json` (+ `explain/`). Первый запрос загружает модели (около минуты), далее
15–40 с на ролик.

## Пакетная обработка папки русских видео

```bash
bash scripts/run_batch_ru.sh            # копирует BS/video_ru в ~/bs/video_ru, считает всё в ~/bs/ru_runs, копирует результаты в results/video_ru
bash scripts/batch_status.sh            # что готово, что считается, какие процессы и сколько памяти GPU занято
~/bs/venv/bin/python scripts/make_batch_pdfs.py ~/bs/ru_runs ~/bs/video_ru --lang ru   # PDF-отчёт на каждое видео (как «Экспорт в PDF»)
bash scripts/after_batch.sh             # ждёт конца батча, затем отладочный прогон сегментов, PDF, копирование
```

`batch_ru.py` пропускает дубликаты (одинаковые файлы под разными именами), не пересчитывает готовые видео
(`--force` пересчитывает, старый результат остаётся как `result_prev.json`, описания поведения переиспользуются из
`segments/timeline.json`; `--summary-only` только пересобирает сводку), а при «липкой» ошибке CUDA завершает
процесс — `run_batch_ru.sh` перезапускает его до пяти раз. Итог: `summary.csv`,
`summary.md` (оценки, перцентили, согласие двух систем, разброс по сегментам) и `<имя>/result.json` с таймлайном.
`debug_segments.py RUN_DIR 18 19 20` повторяет указанные сегменты в одном процессе по участникам ансамбля
(с `CUDA_LAUNCH_BLOCKING=1` — чтобы найти сегмент и модель, где срабатывает device-side assert).
`diag_ru_population.py ~/bs/ru_runs` пересчитывает те же сегменты через OCEAN-AI на весах FIV2 и свою модель без
аудио-узла (`MMConfig(drop_modalities=("audio",))`), чтобы отделить эффект опорной популяции от эффекта русской речи.

Режим `ru` (вариант A): OCEAN-AI (MuPTA) и своя модель (FIV2) обучены на разных популяциях и дают оценки на разных
шкалах (подробности — `../STAGE2_REPORT.md`, раздел 7), поэтому для русской речи **основная оценка — OCEAN-AI на
весах MuPTA**, своя модель показывается как второе мнение и даёт объяснения. Перцентили в этом режиме считаются не
по FIV2, а как ранг среди пула уже обработанных русских роликов (`bs_bigfive/pool.py`, файлы
`~/bs/pool/ru/<отпечаток>.json`, каждый ролик учитывается один раз; меньше трёх роликов — перцентиль не выводится).
В JSON: `traits.<черта>.percentile` и `percentile_ref` (что взято за опору), `model.primary`, `model.scale`;
`percentile_vs_fiv2` остаётся только для шкалы FIV2. В CLI режим задаётся `--primary auto|mean|oceanai|mm`
(auto = OCEAN-AI при `--lang ru`). Переводчик OCEAN-AI ограничен 510 токенами, иначе цикл повторов на транскрипте
без пунктуации выводил из строя CUDA-контекст.

## Аудио-ветвь: кандидаты вместо CLAP

`scripts/audio_candidates.sh` извлекает признаки альтернативных речевых энкодеров (`bs_bigfive/mm/extractors_audio.py`:
энкодер Whisper large-v3, XLS-R 300m, wav2vec2-emotion audeering, eGeMAPS, emotion2vec+ через отдельное окружение
`~/bs/venv_e2v`) и обучает по 5 seed «вместо CLAP» и «рядом с CLAP»; таблица — `results/mm_runs_audio/audio_candidates.md`.
Итог на test FIV2: wav2vec2-emotion 0.9217 / 0.724 и энкодер Whisper 0.9211 / 0.722 против CLAP 0.9202 / 0.715;
пятый узел не помогает. Бэкенд `mm` строит нужный энкодер по списку модальностей чекпойнта, так что любой из этих
наборов весов подставляется через `--mm-ckpt "~/bs/mm_runs_audio/audio_whisper_replace/seed*/best.pt"`.
Проверка на русских роликах — `scripts/audio_ru_check.py` (`results/video_ru/audio_ru_check.md`).

## LLM-оценщик (эксперимент, не используется в системе)

`scripts/run_rater_experiment.sh qwen3-vl:30b 16` — видеоязыковая модель через Ollama оценивает Big Five напрямую
по кадрам и транскрипту (`llm_rater.py`, кэш по клипу), `rater_eval.py` считает mACC/CCC сырых и калиброванных
оценок и ансамблей. Результат на test200 FIV2: mACC 0.853 (0.885 после калибровки), CCC 0.15 — ансамбль с ним
хуже текущего; подробности в `../STAGE2_REPORT.md`, раздел 8. Числа кадров/разрешение: `--frames`, `--max-side`
(кэш для нестандартных настроек хранится отдельно), итоги в `results/rater/`.

## Ограничения этапа 1

- Видео без речи (пустой транскрипт) OCEAN-AI пропускает: `bs infer` завершится ошибкой «no predictions».
  Запасной аудио-видео фьюжн без текста ещё не подключён.

- Веса 2022–2023 годов, библиотека пинит `transformers==4.45.1`; ставим её без зависимостей и подбираем
  окружение сами (`scripts/setup_wsl.sh`).
- Модель оценивает *кажущуюся* личность по первому впечатлению, не психодиагностика. Disclaimer есть в каждом JSON.
