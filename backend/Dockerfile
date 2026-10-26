# Python 3.12 (not the originally planned 3.11): the pinned, tested stack (numpy 2.5, scipy 1.18)
# requires >=3.12 and the laptop/test environment is 3.12. See docs/ARCHITECTURE.md section 17.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# ffmpeg libs are bundled in the PyAV wheel; libgl not needed with opencv-headless.
# libegl1/libgles2: MediaPipe tasks load EGL even for CPU inference.
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libegl1 libgles2 && rm -rf /var/lib/apt/lists/*

COPY requirements/ requirements/
RUN pip install -r requirements/base.txt "psycopg[binary]==3.2.10" \
 && pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements/ml.txt \
 && pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless \
 && pip install opencv-contrib-python-headless==5.0.0.93
# ^ mediapipe depends on the GUI OpenCV build (needs X11 libs); the headless contrib build provides the same cv2.
ENV HF_HOME=/app/models/store/hf CODMAV_WEIGHTS_DIR=/app/models/store

COPY pyproject.toml ./
COPY backend/ backend/
COPY ml/ ml/
COPY monitoring/ monitoring/
COPY config/ config/

RUN useradd --create-home --uid 10001 app && mkdir -p /app/data /app/models/bundles && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:get_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
