FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache embedding model at build time — avoids downloading at runtime
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Copy all source files
COPY . .

# Make startup script executable
RUN chmod +x start.sh

EXPOSE 8001

# start.sh checks for ChromaDB, builds if missing, then starts uvicorn
CMD ["./start.sh"]