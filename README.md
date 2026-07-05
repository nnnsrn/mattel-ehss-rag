# EHSS RAG Microservice

RAG service for Mattel EHSS Safety Vision project.
Exposes endpoints for hazard corrective action generation and EHSS knowledge chatbot.

## Stack
- Embedding: BAAI/bge-small-en-v1.5 (local, no API key needed)
- LLM: Gemini 3.5 Flash (API key required)
- Vector store: ChromaDB (local)
- Framework: FastAPI + LangChain

## Setup

### 1. Install dependencies
pip install -r requirements.txt

### 2. Create .env file
Create a `.env` file in this folder:
GOOGLE_API_KEY=your_gemini_api_key_here

### 3. Build knowledge base (first time only)
python ehss_pipeline.py        # fetch + preprocess OSHA documents
python build_knowledge_base.py # chunk + embed + index into ChromaDB

### 4. Run the service
uvicorn rag_service:app --host 0.0.0.0 --port 8001 --reload

### 5. Test
Open http://localhost:8001/docs for interactive API docs

## Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /rag/generate-corrective-actions | Generate corrective actions from YOLO hazard detections |
| POST | /rag/chat | EHSS AI Assistant — answers safety questions |
| POST | /rag/index | (stub) Index new EHSS document from admin upload |
| GET  | /health | Service health check |

## Notes for RF
- chroma_db/ is NOT in the repo — you must build it locally (step 3)
- .env is NOT in the repo — create your own with your API key
- Port: 8001 (RF's FastAPI should call http://localhost:8001)