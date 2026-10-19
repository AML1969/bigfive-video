# syntax=docker/dockerfile:1
# BS Profiler 3.1 — the web page and the command line, with every model inside the image (works without internet).
# Built from the context made by docker/stage.sh (code from git, model weights from the build machine's caches):
#   docker build -t bs-profiler:3.1 <context>
# GPU: NVIDIA, driver 580 or newer (PyTorch 2.11 for CUDA 13.0), started with --gpus all / the compose file.
FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv ca-certificates curl \
        ffmpeg poppler-utils fonts-dejavu-core \
        libgl1 libegl1 libgles2 libglib2.0-0t64 libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# the ubuntu image has user "ubuntu" with uid 1000; the app runs as its own user "bs"
RUN useradd -m -u 1001 -s /bin/bash bs

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python packages: the pinned top level of requirements-wsl.txt, every other version pinned by constraints.txt
# (pip freeze of the tested environment); OCEAN-AI without its own dependencies, as in the install guide.
# The pip download cache lives in a BuildKit cache mount: a rebuild does not download PyTorch again.
COPY req/ /tmp/req/
RUN --mount=type=cache,target=/root/.cache/pip \
    python3.12 -m venv /opt/venv \
    && env -u PIP_NO_CACHE_DIR pip install -r /tmp/req/requirements-wsl.txt -c /tmp/req/constraints.txt \
    && env -u PIP_NO_CACHE_DIR pip install --no-deps oceanai==1.0.0a48 \
    && rm -rf /tmp/req

# Model weights (large, rarely change: separate layers before the code).
#   Hugging Face: Whisper, CLIP, CLAP, emotion models, translation, BERT for OCEAN-AI
#   OCEAN-AI: its weights for Russian (MuPTA) and English
#   AMLAI 1.0: five checkpoints, averaged
COPY --chown=bs:bs models/hf /home/bs/.cache/huggingface
COPY --chown=bs:bs models/oceanai /home/bs/bs/models
COPY --chown=bs:bs models/mm_runs_seeds /home/bs/bs/mm_runs_seeds

# The code of BS Profiler 3.1 (git archive of bs3-studio at the commit written in VERSION).
COPY app/bs3-studio /opt/bs3-studio
RUN pip install --no-deps -e /opt/bs3-studio \
    && python -c "import bs3; print(bs3.PRODUCT, bs3.__version__)"

ENV HF_HOME=/home/bs/.cache/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_TEMP_DIR=/home/bs/bs3_data/gradio_tmp \
    BS3_OLLAMA_URL=http://ollama:11434 \
    PYTHONWARNINGS=ignore \
    no_proxy=localhost,127.0.0.1,0.0.0.0,ollama \
    NO_PROXY=localhost,127.0.0.1,0.0.0.0,ollama

USER bs
WORKDIR /home/bs
RUN mkdir -p /home/bs/bs3_data/gradio_tmp /home/bs/bs3_data/logs /home/bs/bs3_data/web_jobs
# jobs, journal and Gradio uploads live in a volume: they survive a new image and stay on the customer's machine
VOLUME /home/bs/bs3_data

EXPOSE 7880
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS -o /dev/null http://localhost:7880/ || exit 1
CMD ["bs3", "web", "--port", "7880", "--host", "0.0.0.0"]
