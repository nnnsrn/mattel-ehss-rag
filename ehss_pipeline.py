"""
EHSS RAG Ingestion Pipeline — Step 1 of 3
==========================================
Fetches OSHA 29 CFR Part 1910 from the eCFR public API,
preprocesses the raw XML into clean plain text, then splits
into per-category files aligned to your 5 YOLO hazard categories.

Output
------
./ehss_docs_raw/1910_raw.xml   — raw XML backup (for debugging)
./ehss_docs_raw/1910_full.txt  — full preprocessed plain text
./ehss_docs/ppe.txt            — PPE / Subpart I
./ehss_docs/walking_surfaces.txt
./ehss_docs/egress.txt
./ehss_docs/electrical.txt
./ehss_docs/hazcom.txt

Requirements
------------
pip install requests lxml
"""

import re
import os
import requests
from lxml import etree


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL   = "https://www.ecfr.gov"
RAW_DIR    = "./ehss_docs_raw"
OUTPUT_DIR = "./ehss_docs"

# Maps each hazard category to the section number range inside OSHA Part 1910.
# These align directly with the YOLO label set your CV engineer is using.
CATEGORY_RANGES = {
    "ppe":              (132, 140),   # Subpart I  — helmet, vest, gloves, foot
    "walking_surfaces": (21,  30),    # Subpart D  — wet floor, housekeeping
    "egress":           (33,  39),    # Subpart E  — blocked walkway, emergency exits
    "electrical":       (301, 399),   # Subpart S  — exposed cable, wiring
    "hazcom":           (1200, 1200), # Subpart Z  — chemical spill, SDS, labeling
}


# ---------------------------------------------------------------------------
# Step 1: Acquisition — fetch raw XML from eCFR API
# ---------------------------------------------------------------------------

def get_latest_date(title: str = "29") -> str:
    """
    Query the eCFR metadata endpoint to get the most recent valid snapshot
    date for the given title.

    Why this is needed: the versioner API URL requires a specific date
    (/full/{date}/title-29.xml). Using today's date can return a 404 because
    eCFR typically lags 1-2 business days behind the Federal Register.
    Fetching the metadata first guarantees we always use a valid date.
    """
    resp = requests.get(f"{BASE_URL}/api/versioner/v1/titles.json")
    resp.raise_for_status()
    for t in resp.json()["titles"]:
        if str(t["number"]) == title:
            return t["up_to_date_as_of"]
    raise ValueError(f"Title {title} not found in eCFR titles list")


def fetch_part_xml(date: str, title: str, part: str) -> bytes:
    """
    Fetch a single CFR part as XML from the eCFR versioner API.

    Uses the ?part= query parameter to scope the response to one part only
    (e.g. Part 1910) rather than the full Title 29 (~gigabytes). Without
    this filter the request would time out.

    Returns raw bytes (not decoded string) because lxml.etree.fromstring()
    expects bytes for correct XML encoding handling.
    """
    url  = f"{BASE_URL}/api/versioner/v1/full/{date}/title-{title}.xml"
    resp = requests.get(url, params={"part": part}, timeout=120)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Step 2: Preprocessing — convert XML to clean plain text
# ---------------------------------------------------------------------------

def xml_to_text(xml_bytes: bytes) -> str:
    """
    Parse eCFR XML and extract clean plain text.

    Preprocessing operations performed here:
    - Tag stripping: only <P> (paragraph) and <HEAD> (section heading) tags
      are kept; all structural/metadata tags (DIV, CITA, NOTE, etc.) are
      silently dropped via the XPath selector.
    - HTML entity decoding: lxml automatically converts &#xA7; -> §,
      &#x2014; -> —, &quot; -> ", etc.
    - Empty node removal: paragraphs that are empty after stripping whitespace
      are filtered out.
    - Nested tag flattening: itertext() traverses the element AND all
      descendant tags (e.g. <I> for italics, <E> for emphasis, cross-reference
      links), yielding all text nodes in document order. This is critical —
      using p.text instead would silently drop everything after the first
      nested tag, causing empty paragraphs for sections like 1910.132(a).

    What is NOT preprocessed here (known limitations for your report):
    - [Reserved] placeholder paragraphs are kept
    - Federal Register citation lines like [39 FR 23502, ...] are kept
    - Non-mandatory appendices are not separated from the main section text
    """
    root       = etree.fromstring(xml_bytes)
    paragraphs = root.xpath("//P | //HEAD")
    text_parts = []
    for p in paragraphs:
        full_text = "".join(p.itertext()).strip()
        if full_text:
            text_parts.append(full_text)
    # Double newline as paragraph separator — this is what the splitting
    # step below relies on to identify paragraph boundaries.
    return "\n\n".join(text_parts)


# ---------------------------------------------------------------------------
# Step 3: Domain filtering — split full text into per-category files
# ---------------------------------------------------------------------------

def split_into_sections(text: str) -> dict[int, str]:
    """
    Split the full Part 1910 text into a dict of {section_number: section_text}.

    Regex design rationale:
    The pattern anchors to start-of-line (^) and requires a capital letter
    after the section number followed by the line ending in a period.
    This distinguishes true section headings:
        § 1910.132 General requirements.
    from inline cross-references that appear mid-paragraph:
        ...IBR approved for § 1910.1200.
    Without this anchor, cross-references cause false splits — this was the
    root cause of the hazcom.txt truncation bug (267 chars instead of 271KB).
    """
    pattern = re.compile(r"^§ 1910\.(\d+) [A-Z][^\n]*\.\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = {}
    for i, m in enumerate(matches):
        sec_num = int(m.group(1))
        start   = m.start()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[sec_num] = text[start:end].strip()
    return sections


def save_category_files(sections: dict[int, str]) -> None:
    """
    For each hazard category, collect the sections in its number range
    and save them to a single .txt file, with sections separated by
    '\n\n---\n\n'. The --- separator is what build_knowledge_base.py
    uses to split sections back apart during chunking.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for category, (lo, hi) in CATEGORY_RANGES.items():
        matched = [sections[n] for n in sorted(sections) if lo <= n <= hi]
        if not matched:
            print(f"  WARNING: no sections found for {category} (range {lo}-{hi})")
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{category}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(matched))
        print(f"  {category}: {len(matched)} sections saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    # Step 1: Acquire
    print("Step 1: Fetching OSHA Part 1910 from eCFR API...")
    date      = get_latest_date("29")
    print(f"  Using snapshot date: {date}")
    xml_bytes = fetch_part_xml(date, "29", "1910")

    # Save raw XML as backup — lets you re-run preprocessing locally
    # without making another API call if bugs are found later
    xml_path = os.path.join(RAW_DIR, "1910_raw.xml")
    with open(xml_path, "wb") as f:
        f.write(xml_bytes)
    print(f"  Raw XML saved -> {xml_path} ({len(xml_bytes):,} bytes)")

    # Step 2: Preprocess
    print("\nStep 2: Preprocessing XML -> plain text...")
    text     = xml_to_text(xml_bytes)
    txt_path = os.path.join(RAW_DIR, "1910_full.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Clean text saved -> {txt_path} ({len(text):,} chars)")

    # Step 3: Domain filtering — split into per-category files
    print("\nStep 3: Splitting into per-category files...")
    sections = split_into_sections(text)
    print(f"  Total sections found in Part 1910: {len(sections)}")
    save_category_files(sections)

    print("\nDone. Next step: run build_knowledge_base.py to chunk, embed, and index.")


if __name__ == "__main__":
    main()