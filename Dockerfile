FROM python:3.10-slim

# System deps (optional: for librosa, soundfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install base requirements
COPY wav_reverse_engineer/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY wav_reverse_engineer /app/wav_reverse_engineer

ENV PYTHONPATH=/app/wav_reverse_engineer

# Default command prints version
CMD ["python", "wav_reverse_engineer/cli.py", "version"]
