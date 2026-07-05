FROM python:3.11-slim

WORKDIR /app

# Install system dependencies yang dibutuhkan sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies dulu (layer ini di-cache Docker)
# Kalau requirements.txt tidak berubah, layer ini tidak di-rebuild saat redeploy
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download dan cache model BAAI/bge-small-en-v1.5 saat BUILD TIME
# Ini kunci supaya Railway tidak perlu download 120MB tiap cold start
# Model tersimpan di /root/.cache/huggingface/ di dalam image
RUN python -c "
from sentence_transformers import SentenceTransformer
print('Downloading BAAI/bge-small-en-v1.5...')
model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print('Model cached successfully.')
"

# Copy source code
# chroma_db/ tidak di-copy (ada di .dockerignore) karena
# akan di-mount sebagai Railway persistent volume di /app/chroma_db
COPY . .

EXPOSE 8001

CMD ["uvicorn", "rag_service:app", "--host", "0.0.0.0", "--port", "8001"]