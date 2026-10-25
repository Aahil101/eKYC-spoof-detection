# Python 3.12 (not the originally planned 3.11): the pinned, tested stack (numpy 2.5, scipy 1.18)
# requires >=3.12 and the laptop/test environment is 3.12. See docs/ARCHITECTURE.md section 17.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# ffmpeg libs are bundled in the PyAV wheel; libgl not needed with opencv-headless.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements/ requirements/
RUN pip install -r requirements/base.txt "psycopg[binary]==3.2.10"

COPY pyproject.toml ./
COPY backend/ backend/
COPY ml/ ml/
COPY monitoring/ monitoring/
COPY config/ config/

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data /app/models/bundles && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
