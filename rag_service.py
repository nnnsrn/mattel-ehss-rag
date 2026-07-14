"""
RAG Service — Separate HTTP Microservice (Opsi A)
=================================================
Vector store: Supabase pgvector
Embedding:    BAAI/bge-small-en-v1.5 (local, no API key)
Generation:   Gemini 3.5 Flash

YOLO classes (6 actionable, 1 filtered):
  person        → filtered (not a hazard)
  helmet        → high priority
  safety_vest   → medium priority
  wet_floor     → medium priority (covers both wet surface AND chemical-related
                   floor hazards — chemical_spill class removed from YOLO per
                   supervisor decision. OCR text enriches retrieval when
                   chemical vocabulary is present in image signage.)
  blocked_walkway → high priority
  exposed_cable   → high priority

Run locally:
    uvicorn rag_service:app --host 0.0.0.0 --port 8001 --reload

Deploy: Railway (Dockerfile + railway.json in repo root)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPABASE_URL              = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
GOOGLE_API_KEY            = os.environ["GOOGLE_API_KEY"]
RETRIEVAL_K               = 3

# Only 'person' is filtered — it is a YOLO detection anchor, not a hazard.
# chemical_spill was removed from YOLO by Johana (supervisor approved) —
# wet_floor now covers both wet surface and chemical floor hazards.
NON_HAZARD_LABELS = {"person"}

# Enriched retrieval queries per hazard label.
# wet_floor intentionally includes chemical/spill vocabulary so that
# OCR text like "CHEMICAL" or "SPILL" on signage pulls hazcom content
# in addition to walking surfaces content — broader coverage for the
# merged class.
RETRIEVAL_QUERY_MAP = {
    "no_helmet":        "head protection helmet PPE hard hat",
    "helmet":           "head protection helmet PPE hard hat",
    "no_vest":          "high visibility vest PPE protective clothing",
    "safety_vest":      "high visibility vest PPE protective clothing",
    "wet_floor":        "wet floor slip housekeeping walking surface spill liquid chemical hazard",
    "blocked_walkway":  "walkway egress path clear obstruction housekeeping",
    "exposed_cable":    "electrical cable cord damage protection wiring",
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

    logger.info("Loading local embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Connecting to Supabase pgvector...")
    supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    vectorstore = SupabaseVectorStore(
        embedding  = embeddings,
        client     = supabase_client,
        table_name = "documents",
        query_name = "match_documents",
    )
    logger.info("Supabase pgvector connected.")

    logger.info("Loading Gemini Flash LLM...")
    llm = ChatGoogleGenerativeAI(
        model          = "gemini-1.5-flash",
        google_api_key = GOOGLE_API_KEY,
        temperature    = 0.2,
    )
    logger.info("RAG service ready.")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def retrieve_context(query: str, k: int = RETRIEVAL_K) -> list[dict]:
    """Embed query locally, retrieve top-k chunks from Supabase pgvector."""
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
    return "\n\n".join(
        f"[Source {i+1} — OSHA § {c['section']}]\n{c['content']}"
        for i, c in enumerate(chunks)
    )


def extract_text_from_response(content) -> str:
    """Handle both str and list response formats from Gemini."""
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
    label:              str
    action_description: str
    # priority and due_date excluded — RF's FastAPI is single source of truth

class CorrectiveActionsRequest(BaseModel):
    hazards: list[HazardInput]

class CorrectiveActionsResponse(BaseModel):
    actions: list[CorrectiveAction]


@app.post("/rag/generate-corrective-actions", response_model=CorrectiveActionsResponse)
async def generate_corrective_actions(request: CorrectiveActionsRequest):
    """
    Takes YOLO-detected hazards, retrieves OSHA regulations, generates
    corrective action descriptions in one batched Gemini call.

    wet_floor query is enriched with chemical vocabulary so that OCR text
    containing chemical-related terms pulls hazcom content in addition to
    walking surfaces — covering the merged wet_floor/chemical_spill class.
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
        # OCR text enrichment — if image signage mentions "CHEMICAL", "ACID",
        # "FLAMMABLE" etc., this pulls chemical-relevant chunks for wet_floor
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
    "label": "<exact label from input>",
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

    description_map = {item["label"]: item["action_description"] for item in parsed}

    return CorrectiveActionsResponse(actions=[
        CorrectiveAction(
            label              = hazard.label,
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
    Answers free-text safety questions grounded in OSHA documents.
    Backend for the EHSS AI Assistant panel in RF's frontend UI.
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
    raise HTTPException(status_code=501, detail="Not implemented yet.")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    try:
        test      = vectorstore.similarity_search("PPE helmet", k=1)
        connected = len(test) > 0
    except Exception:
        connected = False
    return {
        "status":           "ok" if connected else "degraded",
        "supabase":         "connected" if connected else "error",
        "embedding_model":  "BAAI/bge-small-en-v1.5 (local)",
        "llm_model":        "gemini-3.5-flash",
    }