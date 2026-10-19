# Ollama with the vision model qwen2.5vl:7b inside (behaviour descriptions and frame phrases of AMLAI 1.0, and the
# translation of the descriptions). Same Ollama version as on the build machine; the model files are copied from its
# local store by docker/stage.sh, so the image holds exactly the model the product was tested with.
FROM ollama/ollama:0.40.6

COPY models /root/.ollama/models

ENV OLLAMA_HOST=0.0.0.0:11434 \
    OLLAMA_KEEP_ALIVE=30m

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=30 \
    CMD ollama list >/dev/null 2>&1 || exit 1
