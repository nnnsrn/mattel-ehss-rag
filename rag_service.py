"""
RAG Service — Separate HTTP Microservice (Opsi A)
=================================================
Exposes endpoints for RF's FastAPI to call:

  POST /rag/generate-corrective-actions
      Input : list of YOLO-detected hazards (label, confidence, ocr_text)
      Output: list of corrective actions (hazard_label + action_description only)
              Priority and due_date are intentionally excluded — RF handles
              those via severity rule table in FastAPI (Opsi 2, agreed 2026-07-05)

  POST /rag/chat
      Input : free-text safety question from inspector
      Output: conversational answer grounded in OSHA documents, with citations

  POST /rag/index   (stub — implement after main endpoints are tested)
      Input : new EHSS document URL from admin upload
      Output: indexing confirmation

Run this service:
    uvicorn rag_service:app --host 0.0.0.0 --port 8001 --reload

Requirements:
    pip install fastapi uvicorn langchain langchain-google-genai langchain-chroma
                chromadb python-dotenv langchain-huggingface sentence-transformers

Embedding : BAAI/bge-small-en-v1.5 (local, no API key, no quota)
Generation: Gemini 3.5 Flash (requires GOOGLE_API_KEY in .env)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PERSIST_DIR    = "./chroma_db"
GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
RETRIEVAL_K    = 3

# Labels that are NOT hazards — no corrective action generated for these.
# Maintained here (not in RF's FastAPI) because this is CV/detection domain
# logic: "person" is a YOLO detection anchor class, not a safety violation.
# Priority and due_date rules live in RF's FastAPI (single source of truth).
NON_HAZARD_LABELS = {"person"}

# Enriched retrieval queries per hazard label.
# Improves retrieval precision over using the raw label string as a query —
# e.g. "blocked_walkway" alone matched ladder sections instead of egress/housekeeping.
RETRIEVAL_QUERY_MAP = {
    "no_helmet":       "head protection helmet PPE hard hat",
    "helmet":          "head protection helmet PPE hard hat",
    "no_vest":         "high visibility vest PPE protective clothing",
    "safety_vest":     "high visibility vest PPE protective clothing",
    "wet_floor":       "wet floor slip housekeeping walking surface",
    "blocked_walkway": "walkway egress path clear obstruction housekeeping",
    "exposed_cable":   "electrical cable cord damage protection wiring",
    "chemical_spill":  "chemical spill hazardous material emergency response",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App + startup
# ---------------------------------------------------------------------------

app = FastAPI(title="EHSS RAG Service", version="1.0.0")

vectorstore = None
llm         = None


@app.on_event("startup")
async def startup():
    global vectorstore, llm

    # Local embedding model — no API key, no quota, no rate limit.
    # Must match the model used in build_knowledge_base.py exactly.
    # normalize_embeddings=True is required for bge models.
    logger.info("Loading local embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Loading ChromaDB...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
    logger.info(f"ChromaDB loaded — {vectorstore._collection.count()} vectors")

    # Gemini Flash for generation only — embedding is fully local.
    logger.info("Loading Gemini Flash LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )
    logger.info("RAG service ready.")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def retrieve_context(query: str, k: int = RETRIEVAL_K) -> list[dict]:
    """Embed query locally and retrieve top-k chunks from ChromaDB."""
    results = vectorstore.similarity_search(query, k=k)
    return [
        {
            "content":  r.page_content,
            "section":  r.metadata.get("section", "unknown"),
            "category": r.metadata.get("category", "unknown"),
        }
        for r in results
    ]


def format_context_for_prompt(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered block for the LLM prompt."""
    return "\n\n".join(
        f"[Source {i+1} — OSHA § {c['section']}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )


def extract_text_from_response(content) -> str:
    """
    Normalize LLM response content to plain string.
    gemini-3.5-flash returns a list of content blocks instead of a plain
    string — this handles both formats defensively.
    """
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in content
        ).strip()
    return str(content).strip()


# ---------------------------------------------------------------------------
# Endpoint 1: /rag/generate-corrective-actions
# ---------------------------------------------------------------------------

class HazardInput(BaseModel):
    label:            str
    confidence_score: float
    ocr_text:         Optional[str] = ""

class CorrectiveAction(BaseModel):
    label:       str
    action_description: str
    # NOTE: priority and due_date intentionally excluded.
    # RF's FastAPI is single source of truth for these (Opsi 2).

class CorrectiveActionsRequest(BaseModel):
    hazards: list[HazardInput]

class CorrectiveActionsResponse(BaseModel):
    actions: list[CorrectiveAction]


@app.post("/rag/generate-corrective-actions", response_model=CorrectiveActionsResponse)
async def generate_corrective_actions(request: CorrectiveActionsRequest):
    """
    Takes YOLO-detected hazards, retrieves relevant OSHA regulations,
    generates corrective action descriptions via Gemini in one batched call.

    Design decisions:
    - NON_HAZARD_LABELS (e.g. 'person') are filtered out — not a safety violation
    - All hazards batched into ONE Gemini call to minimize API usage
    - RETRIEVAL_QUERY_MAP enriches queries for better retrieval precision
    - OCR text further enriches the retrieval query when present
    - priority and due_date NOT returned — RF handles these (Opsi 2)
    """
    actionable = [h for h in request.hazards if h.label not in NON_HAZARD_LABELS]
    if not actionable:
        return CorrectiveActionsResponse(actions=[])

    # Retrieve OSHA context for each hazard
    hazard_contexts = {}
    for hazard in actionable:
        base_query = RETRIEVAL_QUERY_MAP.get(
            hazard.label, hazard.label.replace("_", " ")
        )
        if hazard.ocr_text and hazard.ocr_text.strip():
            base_query += f" {hazard.ocr_text.strip()}"
        hazard_contexts[hazard.label] = retrieve_context(base_query, k=RETRIEVAL_K)

    # Build ONE batch prompt for all hazards
    hazard_blocks = []
    for hazard in actionable:
        context  = format_context_for_prompt(hazard_contexts[hazard.label])
        ocr_note = (
            f"\nOCR text from image: \"{hazard.ocr_text}\""
            if hazard.ocr_text and hazard.ocr_text.strip() else ""
        )
        hazard_blocks.append(
            f"HAZARD: {hazard.label} (confidence: {hazard.confidence_score:.0%}){ocr_note}\n"
            f"Relevant OSHA regulations:\n{context}"
        )

    prompt = f"""You are a workplace safety expert. For each detected hazard below,
generate a specific corrective action grounded in the provided OSHA regulations.

{chr(10).join(f'---{chr(10)}{block}' for block in hazard_blocks)}

---
Respond ONLY with a valid JSON array. One object per hazard. No preamble, no markdown.
Format:
[
  {{
    "hazard_label": "<exact label from input>",
    "action_description": "<specific corrective action, cite the OSHA section number>"
  }}
]"""

    try:
        response = llm.invoke(prompt)
        raw      = extract_text_from_response(response.content)
        raw      = re.sub(r"^```(?:json)?\s*", "", raw)
        raw      = re.sub(r"\s*```$", "", raw)
        parsed   = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON: {e}\nRaw: {raw}")
        raise HTTPException(status_code=500, detail="LLM returned malformed JSON")
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    description_map = {item["hazard_label"]: item["action_description"] for item in parsed}

    return CorrectiveActionsResponse(actions=[
        CorrectiveAction(
            label       = hazard.label,
            action_description = description_map.get(
                hazard.label, "Follow OSHA general safety standards."
            ),
        )
        for hazard in actionable
    ])


# ---------------------------------------------------------------------------
# Endpoint 2: /rag/chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str

class ChatSource(BaseModel):
    section:  str
    category: str
    excerpt:  str

class ChatResponse(BaseModel):
    answer:  str
    sources: list[ChatSource]


@app.post("/rag/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Answers a free-text safety question grounded in OSHA documents.
    Backend for the 'EHSS AI Assistant' panel in RF's UI.
    Returns answer text + source citations for display alongside it.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chunks = retrieve_context(request.question, k=RETRIEVAL_K)

    if not chunks:
        return ChatResponse(
            answer  = "I could not find relevant OSHA regulations for your question. "
                      "Please rephrase or consult your safety officer.",
            sources = [],
        )

    context = format_context_for_prompt(chunks)

    prompt = f"""You are an EHSS (Environmental Health, Safety, and Sustainability)
expert assistant for Mattel manufacturing facilities. Answer the inspector's question
using ONLY the provided OSHA regulation excerpts below.

If the answer is not covered by the excerpts, say so clearly rather than guessing.
Always cite the specific OSHA section number (e.g. § 1910.132) when making a claim.
Keep the answer concise and practical — inspectors need actionable information.

OSHA Regulation Excerpts:
{context}

Inspector's question: {request.question}

Answer:"""

    try:
        response = llm.invoke(prompt)
        answer   = extract_text_from_response(response.content)
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    return ChatResponse(
        answer  = answer,
        sources = [
            ChatSource(
                section  = c["section"],
                category = c["category"],
                excerpt  = c["content"][:200] + "..." if len(c["content"]) > 200 else c["content"],
            )
            for c in chunks
        ],
    )


# ---------------------------------------------------------------------------
# Endpoint 3: /rag/index (stub)
# ---------------------------------------------------------------------------

class IndexRequest(BaseModel):
    doc_id:   str
    file_url: str
    title:    str
    category: str

class IndexResponse(BaseModel):
    status:         str
    doc_id:         str
    chunks_indexed: int


@app.post("/rag/index", response_model=IndexResponse)
async def index_document(request: IndexRequest):
    """
    Stub — implement after /generate-corrective-actions and /chat are tested.
    Returns 501 until implemented. RF can wire this up without getting a 404.
    """
    raise HTTPException(
        status_code=501,
        detail="Not implemented yet. Will be built after core endpoints are complete."
    )


# ---------------------------------------------------------------------------
# Health check — RF calls this to verify service is up before sending requests
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    count = vectorstore._collection.count() if vectorstore else 0
    return {
        "status":          "ok",
        "vectors_stored":  count,
        "embedding_model": "BAAI/bge-small-en-v1.5 (local)",
        "llm_model":       "gemini-3.5-flash",
    }