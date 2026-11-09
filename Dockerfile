FROM python:3.12.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/tmp/huggingface \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/cache \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

WORKDIR /app
COPY requirements.lock /app/requirements.lock
RUN pip install --no-cache-dir --require-hashes -r /app/requirements.lock
ENV AGENT_PROFILE=/app/profiles/public-20261108.json
ENV AGENT_BANDIT=/app/profiles/bandit-20261108.json
ENV AGENT_FORECAST_PROFILE=/app/profiles/forecast-20261108.json
ENV AGENT_FORECAST_POLICY=density
ENV AGENT_VERIFICATION=1
ENV AGENT_REVIEW_STRATEGY=search_only
ENV AGENT_CONCURRENCY=2
ENV AGENT_RPM=45
ENV AGENT_REVIEW_POLICY=/app/profiles/review-expanded-20261108.json
COPY alfagraph /app/alfagraph
COPY profiles/public-20261108.json profiles/bandit-20261108.json profiles/forecast-20261108.json profiles/review-expanded-20261108.json profiles/review-balanced-20261109.json /app/profiles/
COPY run_agent.py /app/run_agent.py
ENTRYPOINT ["python", "/app/run_agent.py"]
