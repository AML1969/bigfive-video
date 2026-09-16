# ТЗ. Оценка Big Five (OCEAN) по видео — пайплайн «BS»

Версия 0.1 · 2026-09-14 · статус: черновик для обсуждения

> **Статус выполнения.** Этап 1 (инференс на готовых весах) реализован: пакет `bs/`, бэкенды `oceanai`,
> `sslmepr`, `ensemble`, замеры на test FIV2. Итоги и цифры — в [STAGE1_REPORT.md](STAGE1_REPORT.md).
> Веса лица и тела SSL-MEPR не опубликованы, поэтому этап 1 построен на OCEAN-AI (раздел 11 ниже устарел в этой части).
> Этап 2 (собственная модель по рецепту MM-PSYCHE, бэкенд `mm`, шестой выход «собеседование», описание поведения
> как объяснение, команда `bs explain`) — в [STAGE2_REPORT.md](STAGE2_REPORT.md). Лучшая система: ансамбль
> OCEAN-AI + своя модель (5 seed), mACC 0.927 / CCC 0.733 на полном test. Веб-интерфейс (раздел 9, этап Э6)
> сделан: `bs web`, http://localhost:7860, см. `bs/README.md`.

---

## 1. Назначение и границы

**Что делаем.** Локальная программа, которая принимает видеофайл с говорящим в кадре человеком и возвращает пять оценок кажущихся черт личности по модели Big Five (OCEAN): открытость, добросовестность, экстраверсия, доброжелательность, эмоциональная стабильность (в терминах датасета — *non-neuroticism*). Оценки в диапазоне 0…1, как в First Impressions V2.

**Основа.** Воспроизводим ветку «apparent personality» из MM-PSYCHE (LEYA-HSE, IEEE Access 2026): те же замороженные энкодеры, тот же препроцессинг, та же fusion-модель MCDM, но только с одной задачей — personality. Эмоции и амбивалентность в v1 не делаем; архитектура это позволяет добавить позже флагом `active_tasks`.

**Этапность.**
1. v1 — CLI: `видео → JSON/CSV с пятью оценками` (этот документ).
2. v2 — веб-интерфейс: загрузил видео → получил результат (отдельное ТЗ позже).

**Что это не является.** Не клинический и не психодиагностический инструмент. FIV2 размечен краудсорсингом (Amazon Mechanical Turk) по первому впечатлению за 15 секунд, поэтому модель предсказывает, *как человека воспринимают наблюдатели*, а не его личность по опроснику. Это обязательно указывать в выводе программы и интерфейсе.

---

## 2. Что даёт исследование MM-PSYCHE

Репозиторий изучен полностью (30 файлов кода, ~7 400 строк, MIT). Склонирован в `BS/MM-PSYCHE/`.

| Есть в репозитории | Нет в репозитории |
| --- | --- |
| Полный код обучения и оценки (PyTorch), три варианта MCDM | Обученные веса MM-PSYCHE (ни в репо, ни в релизах, ни на HF) |
| Экстракторы признаков: CLIP ViT-B/32 (лицо), CLAP (аудио), EmoRoBERTa (текст и описание поведения) | Видео и аудио корпусов |
| Препроцессор видео: 30 равномерных кадров, MediaPipe FaceDetection / YOLOv8-face (`yolov8n-face.pt` лежит в репо) | Скрипт инференса «одно видео → результат» |
| CSV с разметкой FIV2 на все 10 000 клипов: метки OCEAN, профессиональные транскрипты, описания поведения от Qwen3-VL-4B | Whisper-этап (в схеме статьи есть, в коде — нет) |
| Кэш признаков, тренер с early stopping, метрики mACC и CCC | Веб-интерфейс |
| Промпт для Qwen3-VL (полный и короткий) | Поддержка CUDA 12.8+/Blackwell (requirements — CUDA 11.8, torch 2.6) |

**Результаты статьи на FIV2 (test), которые служат ориентиром:**

| Конфигурация | mACC | CCC |
| --- | ---: | ---: |
| Single-task SL — только FIV2, только personality (**наш v1**) | 91.8 | 74.0 |
| Single-task SSL — personality + псевдометки на MOSEI и BAH | 92.6 | 77.2 |
| Multitask SSL — три задачи | 91.9 | 72.0 |
| Лучшие внешние работы 2025–2026 | 92.0–92.1 | — |

mACC = 1 − MAE, усреднённый по пяти чертам. CCC — коэффициент конкордантной корреляции. Реалистичная цель v1: **mACC ≥ 0.915, CCC ≥ 0.72** на официальном test-сплите.

**Ключевое из кода, что берём как есть** (файлы в `MM-PSYCHE/src/`):
- `data_loading/video_preprocessor.py` — выбор кадров, детектор лиц, усреднение нескольких лиц, fallback на полный кадр.
- `data_loading/pretrained_extractors.py` — классы CLIP/CLAP/EmoRoBERTa, режим `seq` → агрегация mean‖std.
- `models/models.py` — `MultiModalFusionModel_v1/v2`: проекторы модальностей → graph-слой → task-projectors → cross-attention → head(sigmoid) + GuideBank.
- `utils/measures.py` — `acc_func`, `ccc`. `utils/losses.py` — MAE/CCC для personality.
- `utils/feature_store.py` — кэш признаков с ключом по экстрактору и версии препроцессинга.

---

## 3. Источники данных и весов

| Ресурс | Где | Объём | Лицензия | Статус |
| --- | --- | --- | --- | --- |
| Код MM-PSYCHE | github.com/LEYA-HSE/MM-PSYCHE | 50 МБ | MIT | склонирован |
| Разметка FIV2: OCEAN + транскрипты + описания Qwen3-VL | `MM-PSYCHE/data/fiv2/*.csv` | 6000 / 2000 / 2000 строк | как у корпуса | есть |
| Видео FIV2 (10 000 mp4, 15 с, YouTube, англ.) | HF `yeray142/first-impressions-v2` | ≈3.1 МБ/клип → **≈31 ГБ** | CC BY-NC 4.0 | публичный, без gating |
| Официальный источник FIV2 | chalearnlap.cvc.uab.cat/dataset/24 | то же | CC BY-NC 4.0 | требует регистрации |
| Веса SSL-MEPR (предшественник, 5 модальностей) | ветка `app` репо SSL-MEPR + Google Drive + Яндекс.Диск | fusion 8.5 МБ + унимодальные до 340 МБ | MIT (код) | склонирован в `BS/SSL-MEPR-app/` |
| Замороженные энкодеры | HF: `openai/clip-vit-base-patch32`, `laion/clap-htsat-fused`, `michellejieli/emotion_text_classifier`, `Qwen/Qwen3-VL-4B-Instruct`, Whisper large-v3-turbo | ≈ 0.6 + 0.8 + 0.3 + 9 + 1.6 ГБ | открытые | Whisper уже в кэше WSL |

Проверено скриптом: все 10 000 имён `video_name` из CSV присутствуют в HF-зеркале (сплит `dev` в CSV = `validation` в зеркале). Пустых транскриптов: 8 в train, 3 в test, 0 в dev. Пустых описаний поведения нет.

Статистика меток (train): средние 0.48–0.57, стандартные отклонения 0.13–0.16 по чертам, диапазон 0…1. Эти значения используем для перевода оценок в перцентили.

**Лицензионное ограничение.** FIV2 — CC BY-NC: модель, обученная на нём, годится для исследовательского и некоммерческого использования. Для коммерческого продукта потребуется другой корпус или согласование с ChaLearn.

---

## 4. Целевая архитектура

```
видео.mp4
  │
  ├─ ffmpeg ─────────────► wav mono 48 kHz (CLAP) и 16 kHz (Whisper)
  │
  ├─ 30 равномерных кадров ─► MediaPipe FaceDetection (fallback: YOLOv8-face; нет лица → полный кадр)
  │                            └► кропы лиц ─► CLIP ViT-B/32 image features [30×512] ─► mean‖std ─► 1024
  │
  ├─ wav 48k ──────────────► CLAP audio encoder, last_hidden_state [T×768] ─► mean‖std ─► 1536
  │
  ├─ wav 16k ──► Whisper large-v3-turbo ─► транскрипт ─► EmoRoBERTa [L×768] ─► mean‖std ─► 1536
  │
  └─ (опция) видео ──► Qwen3-VL-4B-Instruct, промпт из статьи (≤75 токенов) ─► описание поведения
                        ─► EmoRoBERTa ─► mean‖std ─► 1536
  │
  ▼
MCDM (MultiModalFusionModel_v2), active_tasks = ["personality"]
  проекторы модальностей (→512) → graph-слой по модальностям → task-projector (512→5→512)
  → cross-attention (8 голов, запросы = предсказания, ключи = модальности) → среднее
  → head: Linear(512→512)+ReLU → Linear(512→5) → Sigmoid
  → усреднение с cos-сходством к GuideBank (5×512, sigmoid)
  ▼
5 оценок OCEAN ∈ [0,1]
```

**Размерности входов fusion-модели:** face 1024, audio 1536, text 1536, behavior 1536. Гиперпараметры из `config.toml` статьи: hidden 512, 8 голов, dropout 0.15, out_features 512.

**Режимы модальностей.** Обучаем и сохраняем несколько чекпоинтов: `face`, `face+audio`, `face+audio+text`, `face+audio+text+behavior`. Инференс выбирает чекпоинт по доступным модальностям (нет речи → без text; Qwen выключен → без behavior). Это дешевле, чем учить дропаут модальностей, и совпадает с абляциями статьи.

**Особенности, которые надо повторить точно** (иначе признаки не совпадут с обученной моделью):
- порог относительной площади лица 0.3 и усреднение нескольких лиц (`average_multi_face = true`);
- повтор последнего валидного кропа, если лицо не найдено; отсечение ведущих полных кадров;
- нормализация wav по максимуму амплитуды, ресемплинг в частоту модели;
- агрегация `mean‖std` с несмещённой дисперсией `unbiased=False`;
- `emb_normalize = false`.

---

## 5. Обучение

| Параметр | Значение (из статьи) |
| --- | --- |
| Данные | FIV2 train 6000, отбор по dev 2000, отчёт по test 2000 |
| Признаки | предварительно извлечены и закэшированы (`features/`), обучение идёт по кэшу |
| Loss | MAE по пяти чертам (в коде доступны CCC, MSE, RMSE-варианты) |
| Оптимизатор | Adam, lr 1e-4, weight decay 1e-5 |
| Планировщик | ReduceLROnPlateau по метрике dev |
| Batch / эпохи / patience | 32 / до 100 / 15 |
| Метрика выбора | mean_pkl = (mACC + CCC)/2 на dev |
| Seed | 42, детерминированный cudnn |

**Ожидаемое время на RTX 5090 Laptop** (оценки, уточним на этапе Э2):

| Шаг | Оценка |
| --- | --- |
| Скачивание зеркала FIV2 (31 ГБ) | 30–90 мин, зависит от канала |
| ffmpeg: 10 000 wav | 15–25 мин при 8 параллельных процессах |
| Признаки лица (декодирование + MediaPipe + CLIP) | ~1 с/клип → 1.5–3 ч; с параллельным декодированием < 1 ч |
| Признаки CLAP | ~0.1 с/клип → 20 мин |
| Признаки текста и описаний (EmoRoBERTa) | минуты |
| Обучение fusion-модели, 100 эпох | 15–40 мин (модель маленькая, всё из кэша) |
| Опционально: Whisper-транскрипты для всех 10 000 клипов | 1.5–3 ч |

**Опциональный шаг согласования train/inference.** В обучении используются профессиональные транскрипты (Rev), в инференсе — Whisper. Чтобы убрать сдвиг, прогоняем Whisper по всему FIV2 и обучаем вариант модели на Whisper-транскриптах; сравниваем на dev. GPU это позволяет.

**Что не входит в v1, но заложено:** cross-domain SSL с псевдометками на CMU-MOSEI и BAH (даёт +0.8 mACC, +3.2 CCC в статье). Оба корпуса требуют запроса у владельцев.

---

## 6. Инференс: интерфейс и форматы

**CLI (v1).**

```bash
bs infer input.mp4 --out result.json [--no-behavior] [--lang auto|en|ru] [--segment 15] [--device cuda]
bs infer-dir ./videos --out results.csv
bs eval --split test          # воспроизводит mACC/CCC на FIV2
bs extract-features --split all
bs train --modalities face,audio,text,behavior
```

**Вход.** mp4/mov/mkv/webm/avi (всё, что читает ffmpeg). Один основной человек в кадре, речь на английском (v1). Длительность любая: видео режется на окна по 15 с (длина клипов FIV2), оценки по окнам усредняются, разброс между окнами отдаётся как мера неопределённости.

**Выход — JSON:**

```json
{
  "input": "input.mp4",
  "duration_sec": 47.3,
  "segments": 3,
  "traits": {
    "openness":            {"score": 0.61, "std_across_segments": 0.04, "percentile_vs_fiv2": 62},
    "conscientiousness":   {"score": 0.55, "std_across_segments": 0.05, "percentile_vs_fiv2": 57},
    "extraversion":        {"score": 0.44, "std_across_segments": 0.07, "percentile_vs_fiv2": 41},
    "agreeableness":       {"score": 0.58, "std_across_segments": 0.03, "percentile_vs_fiv2": 60},
    "emotional_stability": {"score": 0.52, "std_across_segments": 0.06, "percentile_vs_fiv2": 50}
  },
  "transcript": "...",
  "behavior_description": "...",
  "modalities_used": ["face", "audio", "text", "behavior"],
  "warnings": ["face not detected in 2 of 90 frames"],
  "model": {"name": "bs-bigfive", "version": "1.0.0", "checkpoint": "face_audio_text_behavior_ep37.pt", "trained_on": "FIV2 train"},
  "timings_sec": {"ffmpeg": 0.8, "face": 2.1, "audio": 0.4, "asr": 1.9, "behavior": 12.5, "fusion": 0.02},
  "disclaimer": "Apparent personality as perceived by observers; not a psychological assessment."
}
```

`emotional_stability` = `non-neuroticism` из FIV2 (чем выше, тем спокойнее). Оба имени пишем в выводе, чтобы не путать со «нейротизмом».

**Краевые случаи.**
- Лицо не найдено ни в одном кадре → полный кадр в CLIP, предупреждение в `warnings`, флаг низкой надёжности.
- Нет речи или пустой транскрипт → text-модальность подаётся как пустая строка (так же 8 клипов в train), либо берётся чекпоинт без text; выбор фиксируем на Э4 по dev.
- Несколько лиц → по умолчанию усреднение кропов, как в статье; опция `--face largest`.
- Речь не на английском → Whisper с задачей translate → английский транскрипт (v1); мультиязычный энкодер XLM-R — отдельный эксперимент v1.5.

**Производительность инференса** (цель): 15-секундный клип без Qwen — до 5 с после прогрева; с Qwen — до 20 с. Все модели загружаются один раз и держатся в памяти (в веб-версии — фоновый воркер).

---

## 7. Окружение и железо

**Факты о машине** (сняты 2026-09-14):

| Компонент | Значение |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 **Laptop** GPU, 24 463 МиБ VRAM, лимит 100 Вт, Blackwell sm_120 |
| Драйвер / CUDA | 591.74 / CUDA 13.1 (WDDM) |
| CPU / RAM | Intel Core Ultra 9 275HX, 24 потока / 96 ГБ |
| Диски | C: 1.99 ТБ свободно из 3.8; D: 1.02 ТБ из 4.66; WSL ext4: 749 ГБ свободно |
| ОС | Windows 11 Pro 26200 + WSL2 Ubuntu 24.04 (запущена), Docker 29.7.2 |
| Python | Windows: 3.11.9 (torch 2.8.0 **CPU-only** — не годится). WSL: 3.12.3 |
| Проверенный CUDA-стек | WSL `~/echophone/venv`: torch 2.13.0+cu130, torchvision 0.28.0+cu130, transformers 5.14.1 — `cuda.is_available() = True`, capability (12, 0) |
| ASR уже есть | WSL `~/asr` venv: faster-whisper 1.2.1, ctranslate2 4.8.2; сервер `~/asr-server/server.py` (OpenAI-совместимый `/v1/audio/transcriptions`, модель `mobiuslabsgmbh/faster-whisper-large-v3-turbo`); модели в HF-кэше (6.8 ГБ) |
| ffmpeg | Windows 9.0.1, WSL 6.1.1 |

Важно: карта — мобильная версия с **24 ГБ**, а не настольные 32 ГБ. Это влияет только на Qwen3-VL: 4B в bf16 (~9 ГБ) плюс видео-токены влезают, но грузить его следует лениво и не одновременно с обучением.

**Решение по платформе: WSL2 Ubuntu 24.04.** Причины: GPU-стек на этой машине уже проверен именно там; в статье и коде — POSIX-пути; на нативном Windows часть пакетов (mediapipe, PyG, faster-whisper CUDA-библиотеки) даёт больше проблем. Данные и кэш признаков держим на ext4 внутри WSL (не на `/mnt/c`), исходники — в этой папке проекта.

**Стек версий для v1** (проверено наличие колёс на 2026-09-14):

| Пакет | Версия | Примечание |
| --- | --- | --- |
| Python | 3.12 (WSL system) | mediapipe 0.10.21 имеет cp312-колесо |
| torch / torchvision | 2.13.0+cu130 / 0.28.0+cu130 | уже работает на этой машине; альтернатива 2.14.0/0.29.0 |
| torchaudio | **не используем** | последняя версия 2.11.0, с torch 2.13 не ставится; `torchaudio.load` заменяем на `soundfile` + `soxr` |
| transformers | 4.57.x или 5.x | Qwen3-VL требует ≥ 4.57; проверить CLAP/CLIP на 5.x, иначе пин 4.57.6 |
| mediapipe | 0.10.21 | legacy-API `mp.solutions.face_detection`, как в статье; 1.0.x — только Tasks API |
| ultralytics | 8.3.177 (пин статьи) | резервный детектор YOLOv8-face |
| torch-geometric 2.8 + pyg_lib 0.9 | индекс data.pyg.org для torch-2.13.0+cu130 (колёса `cp310-abi3`, подходят для 3.12) | нужны только для `GraphAttentionLayer_V2`. Колёс **torch_sparse / torch_scatter** для torch 2.13 нет: в `layers.py` заменяем `SparseTensor` на пару `edge_index + edge_weight` (PyG сам это поддерживает, правка ~10 строк) или используем `MultiModalFusionModel_v1` (чистый PyTorch) и сравниваем оба варианта на dev |
| faster-whisper | 1.2.1 | уже в WSL, CUDA 12 библиотеки через pip |
| qwen-vl-utils | 0.0.14 | только для behavior-модальности |
| opencv-python, soundfile, soxr, librosa, pandas, scikit-learn, toml, tqdm | актуальные | |

**Бюджет VRAM при инференсе:** CLIP 0.6 + CLAP 0.8 + EmoRoBERTa 0.3 + Whisper turbo fp16 1.6 + fusion < 0.1 ≈ 3.5 ГБ постоянно; Qwen3-VL-4B ещё ~12–14 ГБ с активациями — загружается по запросу. Обучение fusion-модели: < 2 ГБ.

---

## 8. Структура проекта (предложение)

```
BS/
├─ TZ_BigFive_Video_Pipeline.md      ← этот документ
├─ MM-PSYCHE/                        ← оригинал, не трогаем (reference)
├─ SSL-MEPR-app/                     ← reference, путь А
└─ bs/                               ← наш пакет
   ├─ pyproject.toml
   ├─ config/default.toml            ← пути, модальности, гиперпараметры
   ├─ bs/
   │  ├─ media.py                    ← ffmpeg, окна по 15 с
   │  ├─ faces.py                    ← кадры, MediaPipe/YOLO, кропы (перенос из video_preprocessor)
   │  ├─ extractors.py               ← CLIP / CLAP / EmoRoBERTa (перенос из pretrained_extractors)
   │  ├─ asr.py                      ← faster-whisper (локально или через ваш asr-server)
   │  ├─ behavior.py                 ← Qwen3-VL описание, промпт из статьи
   │  ├─ fusion/model.py             ← MCDM, только personality (перенос models.py)
   │  ├─ data/fiv2.py                ← загрузка зеркала, раскладка, CSV → meta
   │  ├─ features/store.py           ← кэш признаков (перенос feature_store)
   │  ├─ train.py · evaluate.py · infer.py · cli.py
   ├─ scripts/download_fiv2.py · extract_audio.sh · setup_wsl.sh
   ├─ tests/                         ← smoke-тест на 3 клипах из test-сплита
   └─ checkpoints/                   ← результат обучения (+ карточка модели с метриками)
```

Данные в WSL: `~/data/fiv2/{video,audio}/{train,dev,test}/`, кэш признаков `~/data/fiv2/features/`.

---

## 9. План работ

| Этап | Содержание | Результат | Оценка |
| --- | --- | --- | --- |
| Э0 Окружение | venv в WSL, torch cu130, все пакеты, `setup_wsl.sh`, smoke-тест GPU | воспроизводимая установка одной командой | 0.5 дня |
| Э1 Данные | скачать зеркало FIV2, разложить по сплитам, ffmpeg wav, сверка с CSV | 10 000 mp4 + wav, отчёт о целостности | 0.5 дня (+ время скачивания) |
| Э2 Признаки | извлечение face/audio/text/behavior в кэш; замер времени на клип | кэш признаков для всех сплитов | 0.5–1 день |
| Э3 Обучение | 4 варианта наборов модальностей, выбор по dev, отчёт на test (mACC, CCC, per-trait MAE) | чекпоинты + таблица результатов | 0.5 дня |
| Э4 Инференс | `bs infer` от видео до JSON, окна по 15 с, краевые случаи, батч по папке | рабочий CLI | 1 день |
| Э5 Валидация | `bs eval` воспроизводит метрики из кэша и «с нуля» на 100 test-клипах через полный пайплайн (проверка совпадения признаков) | отчёт, карточка модели | 0.5 дня |
| Э6 Веб | отдельное ТЗ: FastAPI + очередь + простой фронт или Gradio | — | позже |

Итого v1: около 4–5 рабочих дней плюс машинное время.

---

## 10. Риски и ограничения

| Риск | Влияние | Митигация |
| --- | --- | --- |
| Лицензия FIV2 CC BY-NC | нельзя коммерциализировать модель | зафиксировать в README; для коммерции — отдельный корпус |
| Домен FIV2: YouTube-влоги, английский, 15 с, один человек, HD | падение качества на других данных (собеседования, русский язык, несколько людей) | окна по 15 с, translate в Whisper, XLM-R как эксперимент, честная маркировка надёжности |
| Совместимость библиотек с Blackwell/cu130 | сборка окружения | стек уже проверен для torch; для PyG нет `torch_sparse` — правим `layers.py` или берём `v1`-модель; torchaudio исключён |
| MediaPipe legacy-API устареет | детектор лиц | пин 0.10.21; резерв YOLOv8-face из репо |
| Транскрипты Rev vs Whisper | небольшой сдвиг признаков текста | вариант обучения на Whisper-транскриптах (раздел 5) |
| Qwen3-VL медленный и тяжёлый | латентность, VRAM | модальность опциональна; отдельный чекпоинт без behavior |
| Воспроизведение цифр статьи | mACC может отличаться на ±0.3 | сравниваем с 91.8/74.0, а не с SSL-вариантом |
| Этика | неверное использование оценок | disclaimer в каждом выводе, нет «диагнозов» и категорий |

---

## 11. Альтернативный путь А: готовые веса SSL-MEPR

У той же лаборатории есть предшественник SSL-MEPR (ветка `app`): Gradio-приложение, fusion-чекпоинт `ssl_mepr_ckpt.pt` в репо, унимодальные модели на Google Drive и Яндекс.Диске (для scene — 337 МБ + 33 МБ + 2 МБ, для text — 5 файлов `*bge-small*.pt`, для face/body — папка на Drive). Заявленный результат на FIV2 — mACC 92.88.

| Плюсы | Минусы |
| --- | --- |
| Инференс без обучения, уже есть Whisper и UI | 5 модальностей × 2–3 чекпоинта, часть весов надо выкачивать вручную с Drive |
| Чистый PyTorch (Mamba реализована вручную, без `mamba_ssm`) | пины torch 2.6 / transformers 4.52 — переезд на cu130 не проверен |
| Есть attribution (какая модальность повлияла) | считает и эмоции — лишнее для нашей цели, нельзя переобучить под свои данные без всей трёхстадийной процедуры |

**Рекомендация:** основной путь — B (своё обучение по рецепту MM-PSYCHE, разделы 4–5): полностью контролируемый, воспроизводимый, только Big Five, легко расширяется. Путь А — опционально как внешний бенчмарк на этапе Э5, если захотим сравнить оценки двух систем на своих видео.

---

## 12. Решения, которые нужно принять до старта

| Вопрос | Предлагаемый ответ по умолчанию |
| --- | --- |
| Язык видео в v1 | английский; русский через Whisper translate, XLM-R — эксперимент |
| Нужна ли behavior-модальность (Qwen3-VL) в v1 | да, но как опция `--no-behavior`; отдельный чекпоинт без неё |
| Платформа | WSL2 Ubuntu 24.04, данные на ext4 |
| Версия torch | 2.13.0+cu130 (проверена на машине) |
| Где хранить 31 ГБ видео и кэш | `~/data/fiv2` в WSL (749 ГБ свободно) |
| ASR | faster-whisper large-v3-turbo локально в процессе (не через отдельный сервер), чтобы CLI был самодостаточным |
| Максимальная длительность входного видео | без лимита, окна по 15 с; для веб-версии ограничим 10 мин |
| Хотим ли позже эмоции/амбивалентность | архитектуру не урезаем: `active_tasks` остаётся параметром |

---

## Приложение А. Карта кода MM-PSYCHE

| Файл | Назначение | Берём |
| --- | --- | --- |
| `main.py` | оркестрация: конфиг → экстракторы → датасеты → обучение/поиск | заменяем на `bs/cli.py` |
| `config.toml` | пути, гиперпараметры, абляции | переносим значения |
| `src/data_loading/video_preprocessor.py` | кадры, детекторы, кропы | да |
| `src/data_loading/pretrained_extractors.py` | CLIP/CLAP/EmoRoBERTa/XLM-R/Wav2Vec2 | да (без torchaudio) |
| `src/data_loading/dataset_multimodal.py` | CSV → meta → признаки → кэш | да, упрощаем до одного датасета |
| `src/data_loading/dataset_builder.py` | collate, конкатенация mean‖std | да |
| `src/models/models.py` | MCDM v1/v2/v3, головы, GuideBank | да |
| `src/models/layers.py` | `GraphAttentionLayer_V2` на torch-geometric (импортирует `torch_sparse`) | да, с правкой: убрать зависимость от `torch_sparse`; иначе v1 |
| `src/models/attention/*` | CrossMPT и др. для v3 | нет |
| `src/train.py` | цикл обучения, метрики, чекпоинты | да, только personality-ветка |
| `src/utils/losses.py` | MAE/CCC/…, GradNorm (SSL) | только MAE/CCC |
| `src/utils/measures.py` | `acc_func`, `ccc` | да |
| `src/utils/feature_store.py` | кэш признаков | да |
| `src/utils/search_utils.py`, `schedulers.py`, `logger_setup.py` | поиск гиперпараметров, планировщик, логи | частично |
| `pytorch_Qwen3-VL.py` | генерация описаний поведения | да, для инференса |
| `yolov8n-face.pt` | резервный детектор лиц | да |
| `data/fiv2/*.csv` | метки, транскрипты, описания | да |

## Приложение Б. Проверочные команды

```bash
# GPU в WSL
wsl -d Ubuntu-24.04 -- nvidia-smi --query-gpu=name,memory.total --format=csv
```

```bash
# проверенный CUDA-стек
wsl -d Ubuntu-24.04 -- ~/echophone/venv/bin/python -c "import torch;print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_capability(0))"
```

```bash
# размер зеркала FIV2 по первой странице train
curl -s "https://huggingface.co/api/datasets/yeray142/first-impressions-v2/tree/main/train" | python -c "import sys,json;d=json.load(sys.stdin);s=[x['size'] for x in d if x['type']=='file'];print(len(s),round(sum(s)/len(s)/1e6,2),'MB/clip')"
```

Ссылки: MM-PSYCHE https://github.com/LEYA-HSE/MM-PSYCHE · SSL-MEPR https://github.com/LEYA-HSE/SSL-MEPR · FIV2 https://chalearnlap.cvc.uab.cat/dataset/24/description/ · зеркало https://huggingface.co/datasets/yeray142/first-impressions-v2 · статья https://doi.org/10.1109/ACCESS.2026.3723020
