"""
EHSS RAG Ingestion Pipeline — Step 2 of 2
==========================================
Reads per-category OSHA text files, chunks them, generates local embeddings,
and stores everything in Supabase pgvector (replaces ChromaDB).

Run ONCE from your local machine — vectors are stored permanently in Supabase.
Railway service does NOT need to run this — it just connects to Supabase.

Prerequisites:
1. Run supabase_setup.sql in Supabase SQL Editor first
2. ehss_docs/ folder must exist (run ehss_pipeline.py first)
3. SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env

Requirements:
    pip install supabase vecs langchain-community langchain-huggingface sentence-transformers
"""

from dotenv import load_dotenv
load_dotenv()

import os
import re
import glob
from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EHSS_DIR          = "./ehss_docs"
TARGET_CHUNK_SIZE = 900

SUPABASE_URL              = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

SECTION_HEADING_RE = re.compile(r"^§ 1910\.(\d+) ", re.MULTILINE)


# ---------------------------------------------------------------------------
# Parsing + chunking (same logic as before)
# ---------------------------------------------------------------------------

def parse_sections(text: str) -> list[tuple[str, str]]:
    raw_sections = text.split("\n\n---\n\n")
    sections = []
    for raw in raw_sections:
        raw = raw.strip()
        if not raw:
            continue
        parts   = raw.split("\n\n", 1)
        heading = parts[0].strip()
        body    = parts[1] if len(parts) > 1 else ""
        sections.append((heading, body))
    return sections


def merge_paragraphs_into_chunks(heading: str, body: str, target_size: int) -> list[str]:
    paragraphs  = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks      = []
    current     = []
    current_len = len(heading)
    for para in paragraphs:
        if current and current_len + len(para) > target_size:
            chunks.append(heading + "\n\n" + "\n\n".join(current))
            current     = []
            current_len = len(heading)
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append(heading + "\n\n" + "\n\n".join(current))
    return chunks


def build_documents() -> list[Document]:
    documents = []
    txt_files = glob.glob(os.path.join(EHSS_DIR, "*.txt"))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files in {EHSS_DIR}. Run ehss_pipeline.py first."
        )
    for filepath in txt_files:
        category = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        for heading, body in parse_sections(text):
            section_match = SECTION_HEADING_RE.match(heading)
            section_id    = f"1910.{section_match.group(1)}" if section_match else "unknown"
            for i, chunk_text in enumerate(merge_paragraphs_into_chunks(heading, body, TARGET_CHUNK_SIZE)):
                documents.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "category":    category,
                        "section":     section_id,
                        "chunk_index": i,
                        "source_file": os.path.basename(filepath),
                    },
                ))
    return documents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Step 1: Parsing and chunking OSHA category files...")
    documents = build_documents()
    sizes     = [len(d.page_content) for d in documents]
    print(f"  Total chunks : {len(documents)}")
    print(f"  Chunk sizes  : min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)} chars")

    print("\nStep 2: Loading local embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("\nStep 3: Connecting to Supabase...")
    supabase_client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    print("  Connected.")

    # Store vectors in Supabase pgvector
    # This replaces Chroma.from_documents() — same interface, different backend
    # Vectors are stored permanently in Supabase, no local disk needed
    print("\nStep 4: Embedding and storing in Supabase pgvector...")
    print("  This runs locally with no API quota (local embeddings).")
    print("  Estimated time: ~3-5 minutes for 1109 chunks on CPU.")

    vectorstore = SupabaseVectorStore.from_documents(
        documents       = documents,
        embedding       = embeddings,
        client          = supabase_client,
        table_name      = "documents",
        query_name      = "match_documents",
    )

    print(f"\nDone. {len(documents)} vectors stored in Supabase.")
    print("Railway service can now connect to Supabase — no local ChromaDB needed.")


if __name__ == "__main__":
    main()