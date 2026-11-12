# Imagen única para api, frontend e ingesta (mismo código, mismo encoder, mismo contrato).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    HF_HOME=/cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /work

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# torch va en su PROPIA capa: si un paso posterior falla o se cancela, no se reinstala.
# --no-compile evita precompilar miles de .py de torch (muy lento en discos de WSL2/Windows).
RUN pip install --no-compile torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-compile -r requirements-dev.txt

COPY app/ app/
COPY scripts/ scripts/
COPY configs/ configs/
COPY tests/ tests/
COPY pytest.ini ./

EXPOSE 8000 8501
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
