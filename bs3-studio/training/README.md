# training/ — research code, not part of the app

This folder holds the code that built and measured AMLAI 1.0, the own Big Five model (MM-PSYCHE recipe): FIV2
feature extraction, training, and the accuracy of a model on FIV2 clips. The web page and the `bs3` command never
import it. It imports the runtime package `bs3` (encoders, fusion model, trait keys, the model options of `bs3 infer`),
so run it from `bs3-studio` with the project venv:

    cd bs3-studio
    ~/bs/venv/bin/python -m training.mm_train --help

| File | What it does |
|---|---|
| `mm_data.py` | FIV2 index from the MM-PSYCHE csv files (`BS/MM-PSYCHE/data/fiv2`), audio extraction, per-modality feature cache under `~/data/fiv2` |
| `mm_extract.py` | extracts and caches the face, audio, text and behaviour features of one split |
| `mm_train.py` | trains the fusion model on the cached features; writes `best.pt`, `result.json` and `test_pred.csv` |
| `evaluate.py` | MAE, ACC = 1 - MAE, CCC and Pearson per trait of predictions against FIV2 labels |
| `eval_fiv2.py` | scores a folder of FIV2 clips with AMLAI 1.0 or OCEAN-AI and evaluates them (formerly `bs3 eval-fiv2`) |
| `interview_labels.csv` | ChaLearn job-interview label per FIV2 clip (HF mirror), for `mm_train --targets big5+interview` |
| `ru_labels.py` | the own Russian set: labels.csv and the check tables from the three tables of the organizer memo (docs/dataset/5) |
| `ru_extract.py` | the own set: segment features exactly as the product computes them, plus the AMLAI 1.0 scores (docs/dataset/4) |
| `ru_train.py` | the own set: trains AMLAI 2.0 on the segment features, one fold per run, from scratch or from an AMLAI 1.0 checkpoint |
| `ru_eval.py` | the own set: per-video scores, CCC / Pearson / ACC with bootstrap intervals, rivals raw and calibrated |

## Typical runs

    python -m training.mm_extract --split train --modalities audio,text,behavior
    # one of 6 parallel shards:
    python -m training.mm_extract --split train --modalities face --shard 0/6
    python -m training.mm_extract --split train --modalities face --merge
    python -m training.mm_train --modalities face,audio,text,behavior \
      --out ~/bs/mm_runs/all
    python -m training.eval_fiv2 --dir DIR --out eval.json --backend mm --lang en

The app loads its checkpoints from `~/bs/mm_runs_seeds/seed*/best.pt` (5 seeds, averaged). A training run writes to
its `--out` folder and does not replace them.

## The own Russian set (AMLAI 2.0)

The four `ru_*` modules follow docs/dataset/4_Как_обучить_AMLAI_2.0.md and 5_Как_разметить_набор.md (in Russian):

    python -m training.ru_labels --set /mnt/d/ru_set_v1
    python -m training.ru_extract --videos /mnt/d/ru_set_v1/work \
      --labels /mnt/d/ru_set_v1/labels.csv --out ~/data/ru_v1
    python -m training.ru_train --data ~/data/ru_v1 --labels /mnt/d/ru_set_v1/labels.csv \
      --fold 1 --seed 1 --lr 1e-5 --init ~/bs/mm_runs_seeds/seed1/best.pt \
      --out ~/bs/amlai2_runs/A_lr1e-5/fold1
    python -m training.ru_eval --mode test --data ~/data/ru_v1 \
      --labels /mnt/d/ru_set_v1/labels.csv --ckpt "~/bs/amlai2/v1/fold*/best.pt" \
      --out ~/data/ru_v1/eval_test.json

Feature store of `ru_extract`: `features/<video>.pt` with `x` = {modality: FloatTensor[segments, D]}, the segment
times and texts and the AMLAI 1.0 scores; `amlai1_scores.csv` is rebuilt from these files after every pass. The
scripts are checked on fakes and synthetic features only (`tests/test_ru_labels.py`, `tests/test_ru_training.py`):
they have not yet run on real videos.

`eval_fiv2` expects in `DIR`: `<stem>.mp4`, `<stem>.txt` (the transcript; `--asr` uses Whisper instead) and
`labels.csv` (`video_name` and the five FIV2 label columns). AMLAI 1.0 also reads `<stem>.behavior.txt` when present;
without it the behaviour description comes from Ollama. FIV2 speech is English, so pass `--lang en`: the default is
Russian, and OCEAN-AI then uses its MuPTA weights instead of the FIV2 ones.
