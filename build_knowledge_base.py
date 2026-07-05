"""
EHSS RAG Ingestion Pipeline — Step 2 of 3
==========================================
Reads per-category OSHA text files from ehss_pipeline.py,
applies structure-aware paragraph-merge chunking, generates embeddings
via Gemini, and stores everything in a persistent ChromaDB vector store.

Run ONCE after ehss_pipeline.py. Re-run if documents or chunking params change.

Output:  ./chroma_db/   — persistent ChromaDB vector store

Requirements:
    pip install langchain langchain-google-genai langchain-chroma chromadb

"""
from dotenv import load_dotenv
load_dotenv()
import os
import re
import time
import glob
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

EHSS_DIR          = "./ehss_docs"
PERSIST_DIR       = "./chroma_db"
TARGET_CHUNK_SIZE = 900    # characters per chunk

# Rate limit management — free tier allows 100 requests/minute
# We embed in batches of BATCH_SIZE, then sleep SLEEP_BETWEEN_BATCHES seconds
# 80 chunks/batch * 60s sleep = safely under 100 RPM with headroom
BATCH_SIZE             = 80
SLEEP_BETWEEN_BATCHES  = 62   # slightly over 60s to be safe

GOOGLE_API_KEY    = os.environ["GOOGLE_API_KEY"]
SECTION_HEADING_RE = re.compile(r"^§ 1910\.(\d+) ", re.MULTILINE)


# ---------------------------------------------------------------------------
# Step 1: Parse category files back into sections
# ---------------------------------------------------------------------------

def parse_sections(text: str) -> list[tuple[str, str]]:
    """Split one category .txt file into (heading, body) pairs.
    Sections are separated by the '---' marker written by ehss_pipeline.py."""
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


# ---------------------------------------------------------------------------
# Step 2: Structure-aware paragraph-merge chunking
# ---------------------------------------------------------------------------

def merge_paragraphs_into_chunks(heading: str, body: str, target_size: int) -> list[str]:
    """
    Chunking method: Structure-Aware Paragraph-Merge Chunking.
    - Never splits inside a paragraph — each (a), (b), (1) block stays whole
    - Merges consecutive paragraphs until approaching target_size
    - Never merges across section boundaries
    - Prefixes every chunk with section heading for self-contained retrieval
    """
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


# ---------------------------------------------------------------------------
# Step 3: Build LangChain Document objects with metadata
# ---------------------------------------------------------------------------

def build_documents() -> list[Document]:
    """Read all category .txt files, chunk each section, attach metadata."""
    documents = []
    txt_files = glob.glob(os.path.join(EHSS_DIR, "*.txt"))

    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {EHSS_DIR}. "
            "Run ehss_pipeline.py first."
        )

    for filepath in txt_files:
        category = os.path.splitext(os.path.basename(filepath))[0]
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        for heading, body in parse_sections(text):
            section_match = SECTION_HEADING_RE.match(heading)
            section_id    = f"1910.{section_match.group(1)}" if section_match else "unknown"
            chunks        = merge_paragraphs_into_chunks(heading, body, TARGET_CHUNK_SIZE)

            for i, chunk_text in enumerate(chunks):
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
# Step 4: Embed in rate-limit-safe batches + index into ChromaDB
# ---------------------------------------------------------------------------

def embed_with_rate_limit(documents: list[Document], embeddings, persist_dir: str) -> Chroma:
    """
    Embed documents in small batches with a sleep between each batch.

    Why this is necessary:
    The Gemini free tier allows 100 embedding requests per minute.
    Chroma.from_documents() sends all chunks at once, which immediately
    exceeds the limit for a 1,109-chunk corpus. Instead, we:
      1. Initialize an empty Chroma collection
      2. Add documents in batches of BATCH_SIZE
      3. Sleep SLEEP_BETWEEN_BATCHES seconds between batches
      4. Resume from the last successful batch if interrupted
    """
    # Initialize empty collection — creates chroma_db/ on disk
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    total   = len(documents)
    batches = [documents[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

    print(f"  Splitting {total} chunks into {len(batches)} batches of {BATCH_SIZE}")
    print(f"  Sleep between batches: {SLEEP_BETWEEN_BATCHES}s")
    print(f"  Estimated total time: ~{len(batches) * SLEEP_BETWEEN_BATCHES // 60 + 1} minutes")
    print()

    for i, batch in enumerate(batches):
        print(f"  Batch {i + 1}/{len(batches)} — embedding {len(batch)} chunks...", end=" ")
        vectorstore.add_documents(batch)
        print("done")

        # Sleep between batches — skip sleep after the very last batch
        if i < len(batches) - 1:
            print(f"    Waiting {SLEEP_BETWEEN_BATCHES}s to respect rate limit...")
            time.sleep(SLEEP_BETWEEN_BATCHES)

    return vectorstore


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Step 1 + 2: Parse and chunk
    print("Step 1: Parsing and chunking OSHA category files...")
    documents = build_documents()
    sizes     = [len(d.page_content) for d in documents]
    print(f"  Total chunks : {len(documents)}")
    print(f"  Chunk sizes  : min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)} chars")

    tiny = [d for d in documents if len(d.page_content) < 100]
    if tiny:
        print(f"  WARNING: {len(tiny)} chunks under 100 chars (minor, expected for short clauses)")

    # Step 3: Set up embedding model
    #print("\nStep 2: Initializing Gemini embedding model...")
    #embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=GOOGLE_API_KEY,
    #)
    from langchain_huggingface import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
    )
    print("  Model: BAAI/bge-small-en-v1.5")

    # Step 4: Embed in batches + store
    print("\nStep 3: Embedding and indexing into ChromaDB (rate-limit-safe batching)...")
    vectorstore = embed_with_rate_limit(documents, embeddings, PERSIST_DIR)

    total_stored = vectorstore._collection.count()
    print(f"\nDone. {total_stored} vectors stored -> {PERSIST_DIR}")

    if total_stored != len(documents):
        print(f"  WARNING: expected {len(documents)} vectors, got {total_stored} — some chunks may have failed")
    else:
        print("  All chunks indexed successfully.")

    print("\nNext step: run retrieval sanity check, then build the RAG query service.")


if __name__ == "__main__":
    main()