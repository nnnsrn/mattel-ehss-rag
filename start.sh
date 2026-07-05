#!/bin/bash
set -e

echo "=== EHSS RAG Service Startup ==="

echo "Checking ChromaDB..."
if [ ! -d "/app/chroma_db" ] || [ -z "$(ls -A /app/chroma_db 2>/dev/null)" ]; then
    echo "ChromaDB not found — building from OSHA documents..."
    echo "Step 1: Fetching OSHA documents from eCFR API..."
    python ehss_pipeline.py
    echo "Step 2: Building ChromaDB vector store..."
    python build_knowledge_base.py
    echo "ChromaDB build complete."
else
    echo "ChromaDB found — skipping build."
fi

echo "Starting RAG service on port 8001..."
exec uvicorn rag_service:app --host 0.0.0.0 --port 8001