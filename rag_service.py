"""
RAG Service — Separate HTTP Microservice (Opsi A)
=================================================
Vector store: Supabase pgvector
Embedding:    BAAI/bge-small-en-v1.5 (local, no API key)
Generation:   Groq llama-3.1-8b-instant (free, 14,400 req/day)

UPDATED (post company visit):
- Added area context to corrective action prompt
- Added 7 new area-specific YOLO classes to RETRIEVAL_QUERY_MAP
- HazardInput now accepts optional 'area' field
- NON_HAZARD_LABELS updated

API CONTRACT CHANGE — notify RF:
  /rag/generate-corrective-actions now accepts optional 'area' field per hazard:
  {
    "hazards": [
      {
        "label": "no_glasses",
        "confidence_score": 0.91,
        "ocr_text": "",
        "area": "Spray/Decoration Area"   ← NEW optional field
      }
    ]
  }
  RF should pass area from the inspection session metadata.
  If area is omitted, corrective actions are still generated (area-generic).

Run locally:  uvicorn rag_service:app --host 0.0.0.0 --port 8001 --reload
Deploy:       Railway (auto-deploy from GitHub, CMD uses $PORT)

Requirements:
    pip install fastapi uvicorn langchain langchain-community langchain-huggingface
                sentence-transformers supabase vecs python-dotenv langchain-groq
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
from langchain_groq import ChatGroq


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUPABASE_URL              = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
GROQ_API_KEY              = os.environ.get("GROQ_API_KEY")

if not SUPABASE_URL:
    raise ValueError("SUPABASE_URL is not set. Add it to .env or Railway Variables.")
if not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("SUPABASE_SERVICE_ROLE_KEY is not set.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Get a free key at console.groq.com.")

RETRIEVAL_K = 3

# Labels that are NOT hazards — no corrective action generated.
# 'person' is a YOLO detection anchor class, not a safety violation.
NON_HAZARD_LABELS = {"person"}

# Enriched retrieval queries per YOLO class.
# Each query is designed to pull the most relevant OSHA section for that hazard.
# Updated with 7 new area-specific classes from company visit.
RETRIEVAL_QUERY_MAP = {
    # ── Original classes ────────────────────────────────────────────────────
    "no_helmet":          "head protection helmet PPE hard hat § 1910.135",
    "helmet":             "head protection helmet PPE hard hat § 1910.135",
    "no_vest":            "high visibility vest PPE protective clothing § 1910.132",
    "safety_vest":        "high visibility vest PPE protective clothing § 1910.132",
    "wet_floor":          "wet floor slip housekeeping walking surface spill liquid § 1910.22",
    "blocked_walkway":    "walkway egress path clear obstruction housekeeping § 1910.22 § 1910.37",
    "exposed_cable":      "electrical cable cord damage protection wiring § 1910.305",

    # ── NEW: Spray / Decoration Area ────────────────────────────────────────
    "no_glasses":         "eye face protection safety glasses goggles chemical spray § 1910.133",
    "no_gloves":          "hand protection gloves chemical spray hazardous materials § 1910.138",
    "no_apron":           "protective clothing apron body protection spray chemical operations § 1910.132",

    # ── NEW: Central Staging Area ───────────────────────────────────────────
    "no_safety_shoes":    "foot protection safety shoes footwear heavy load § 1910.136",

    # ── NEW: Assembly Area ──────────────────────────────────────────────────
    "trolley_out_of_lane": "powered industrial truck forklift aisle pedestrian lane marking § 1910.178 § 1910.22",
    "person_out_of_lane":  "pedestrian walkway aisle marking safe path egress § 1910.22 § 1910.37",

    # ── NEW: General Hallways ───────────────────────────────────────────────
    "phone_while_walking": "distracted walking mobile phone hallway walking surface attention § 1910.22",
}

# Area descriptions for prompt context — maps area names to safety focus
AREA_CONTEXT = {
    "Spray/Decoration Area":  "a chemical spray and decoration zone requiring eye, hand, and body protection",
    "Central Staging Area":   "a heavy materials staging zone requiring head and foot protection",
    "Assembly Area":          "an assembly zone with designated trolley lanes and pedestrian walkways",
    "General":                "a general hallway and common area with pedestrian safety rules",
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App + startup
# ---------------------------------------------------------------------------

app = FastAPI(title="EHSS RAG Service", version="2.0.0")

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

    logger.info("Loading Groq LLM (llama-3.1-8b-instant)...")
    llm = ChatGroq(
        model       = "llama-3.1-8b-instant",
        api_key     = GROQ_API_KEY,
        temperature = 0.2,
    )
    logger.info("RAG service v2.0 ready.")


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
    """Handle both str and list response formats from LLMs."""
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
    area:             Optional[str] = ""   # NEW: e.g. "Spray/Decoration Area"
                                           # RF passes this from inspection session

class CorrectiveAction(BaseModel):
    label:              str
    action_description: str
    # priority and due_date excluded — RF handles (Opsi 2, single source of truth)

class CorrectiveActionsRequest(BaseModel):
    hazards: list[HazardInput]

class CorrectiveActionsResponse(BaseModel):
    actions: list[CorrectiveAction]


@app.post("/rag/generate-corrective-actions", response_model=CorrectiveActionsResponse)
async def generate_corrective_actions(request: CorrectiveActionsRequest):
    """
    Receives YOLO-detected hazards, retrieves OSHA regulations,
    generates area-specific corrective actions via Groq in one batch call.

    New in v2.0:
    - Accepts optional 'area' field per hazard for area-specific context
    - 7 new hazard classes supported (glasses, gloves, apron, safety shoes,
      trolley lane, person lane, phone while walking)
    - Area context injected into prompt for more specific recommendations
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
        # OCR text enriches retrieval — e.g. "CHEMICAL STORAGE" on a sign
        # helps pull hazcom content for wet_floor detections near chemicals
        if hazard.ocr_text and hazard.ocr_text.strip():
            base_query += f" {hazard.ocr_text.strip()}"
        hazard_contexts[hazard.label] = retrieve_context(base_query, k=RETRIEVAL_K)

    # Build ONE batch prompt for all hazards
    hazard_blocks = []
    for hazard in actionable:
        context = format_context_for_prompt(hazard_contexts[hazard.label])

        # Area context injection — makes corrective actions specific to the zone
        area_note = ""
        if hazard.area and hazard.area.strip():
            area_desc = AREA_CONTEXT.get(hazard.area.strip(), hazard.area.strip())
            area_note = f"\nInspection area: {hazard.area} ({area_desc})"

        ocr_note = (
            f"\nOCR text from image signage: \"{hazard.ocr_text}\""
            if hazard.ocr_text and hazard.ocr_text.strip() else ""
        )

        hazard_blocks.append(
            f"HAZARD: {hazard.label} (confidence: {hazard.confidence_score:.0%})"
            f"{area_note}{ocr_note}\n"
            f"Relevant OSHA regulations:\n{context}"
        )

    prompt = f"""You are a workplace safety expert for a Mattel manufacturing facility.
For each detected hazard below, generate a specific corrective action grounded in
the provided OSHA regulations. Tailor the action to the inspection area when specified.

{chr(10).join(f'---{chr(10)}{block}' for block in hazard_blocks)}

---
Respond ONLY with a valid JSON array. One object per hazard. No preamble, no markdown.
Format:
[
  {{
    "label": "<exact label from input>",
    "action_description": "<specific corrective action citing the OSHA section number, mention the area if provided>"
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
                hazard.label,
                "Follow applicable OSHA general safety standards for this hazard type."
            ),
        )
        for hazard in actionable
    ])


# ---------------------------------------------------------------------------
# Endpoint 2: /rag/chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    area:     Optional[str] = ""   # optional: filter context to specific area

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
    Backend for the EHSS AI Assistant panel in RF's frontend.

    New in v2.0: optional 'area' field to provide area context in the answer.
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

    area_context = ""
    if request.area and request.area.strip():
        area_desc    = AREA_CONTEXT.get(request.area.strip(), request.area.strip())
        area_context = f"\nThe inspector is asking about the {request.area} ({area_desc})."

    prompt = f"""You are an EHSS (Environmental Health, Safety, and Sustainability)
expert assistant for Mattel manufacturing facilities.{area_context}
Answer the inspector's question using ONLY the provided OSHA regulation excerpts below.

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
                excerpt  = c["content"][:400] + "..." if len(c["content"]) > 400 else c["content"],
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
        "status":          "ok" if connected else "degraded",
        "supabase":        "connected" if connected else "error",
        "embedding_model": "BAAI/bge-small-en-v1.5 (local)",
        "llm_model":       "groq/llama-3.1-8b-instant",
        "version":         "2.0.0",
        "new_classes":     [
            "no_glasses", "no_gloves", "no_apron",
            "no_safety_shoes", "trolley_out_of_lane",
            "person_out_of_lane", "phone_while_walking"
        ],
    }