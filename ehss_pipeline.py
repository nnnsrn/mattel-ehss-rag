"""
EHSS RAG Ingestion Pipeline — Step 1 of 2
==========================================
Fetches OSHA 29 CFR Part 1910 from the eCFR public API,
preprocesses the raw XML into clean plain text, then splits
into per-category files aligned to YOLO hazard classes.

UPDATED: Added materials_handling category (§ 1910.176-181)
for Assembly Area trolley/forklift detection class.

Run this ONCE (or when documents need refreshing).

Output
------
./ehss_docs_raw/1910_raw.xml
./ehss_docs_raw/1910_full.txt
./ehss_docs/ppe.txt
./ehss_docs/walking_surfaces.txt
./ehss_docs/egress.txt
./ehss_docs/electrical.txt
./ehss_docs/hazcom.txt
./ehss_docs/materials_handling.txt   ← NEW: covers trolley/forklift hazards

Requirements: pip install requests lxml
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

# Maps each hazard category to the OSHA Part 1910 section number range.
# Updated to include materials_handling for Assembly Area trolley class.
CATEGORY_RANGES = {
    "ppe":                (132, 140),   # Subpart I  — helmet, vest, glasses, gloves, apron, foot
    "walking_surfaces":   (21,  30),    # Subpart D  — wet floor, housekeeping, aisle markings
    "egress":             (33,  39),    # Subpart E  — blocked walkway, emergency exits
    "electrical":         (301, 399),   # Subpart S  — exposed cable, wiring
    "hazcom":             (1200, 1200), # Subpart Z  — chemical labeling, SDS
    "materials_handling": (176, 181),   # Subpart N  — NEW: powered industrial trucks, forklifts,
                                        #              trolleys, aisle/pedestrian separation
                                        #              Covers: § 1910.176 (materials storage)
                                        #                      § 1910.178 (powered industrial trucks)
                                        #                      § 1910.179 (overhead cranes)
}


# ---------------------------------------------------------------------------
# Step 1: Acquisition
# ---------------------------------------------------------------------------

def get_latest_date(title: str = "29") -> str:
    """Get the most recent valid eCFR snapshot date for Title 29.
    Avoids 404 errors from using today's date when eCFR lags 1-2 days."""
    resp = requests.get(f"{BASE_URL}/api/versioner/v1/titles.json")
    resp.raise_for_status()
    for t in resp.json()["titles"]:
        if str(t["number"]) == title:
            return t["up_to_date_as_of"]
    raise ValueError(f"Title {title} not found")


def fetch_part_xml(date: str, title: str, part: str) -> bytes:
    """Fetch one CFR part as XML. Uses ?part= to scope response to avoid timeout."""
    url  = f"{BASE_URL}/api/versioner/v1/full/{date}/title-{title}.xml"
    resp = requests.get(url, params={"part": part}, timeout=120)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Step 2: Preprocessing — XML → clean plain text
# ---------------------------------------------------------------------------

def xml_to_text(xml_bytes: bytes) -> str:
    """Parse eCFR XML into clean plain text.

    Preprocessing performed:
    - Tag stripping: only <P> and <HEAD> tags kept; all structural tags dropped
    - HTML entity decoding: &#xA7; → §, &#x2014; → —, etc. (lxml handles automatically)
    - Nested tag flattening: itertext() captures text inside <I>, <E>, cross-references
      (using p.text instead would silently drop text after the first nested tag)
    - Empty paragraph removal
    """
    root       = etree.fromstring(xml_bytes)
    paragraphs = root.xpath("//P | //HEAD")
    text_parts = []
    for p in paragraphs:
        full_text = "".join(p.itertext()).strip()
        if full_text:
            text_parts.append(full_text)
    return "\n\n".join(text_parts)


# ---------------------------------------------------------------------------
# Step 3: Domain filtering — split full text into per-category files
# ---------------------------------------------------------------------------

def split_into_sections(text: str) -> dict[int, str]:
    """Split Part 1910 text into {section_number: section_text}.

    Regex anchored to start-of-line with capital letter after section number
    to distinguish true section headings from inline cross-references.
    Example heading:  '§ 1910.178 Powered industrial trucks.'
    Example cross-ref: '...as required by § 1910.178.' (NOT a heading)
    """
    pattern = re.compile(r"^§ 1910\.(\d+) [A-Z][^\n]*\.\s*$", re.MULTILINE)
    matches  = list(pattern.finditer(text))
    sections = {}
    for i, m in enumerate(matches):
        sec_num = int(m.group(1))
        start   = m.start()
        end     = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[sec_num] = text[start:end].strip()
    return sections


def save_category_files(sections: dict[int, str]) -> None:
    """Save one .txt file per category, sections separated by '---'."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for category, (lo, hi) in CATEGORY_RANGES.items():
        matched = [sections[n] for n in sorted(sections) if lo <= n <= hi]
        if not matched:
            print(f"  WARNING: no sections found for {category} (range {lo}-{hi})")
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{category}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n\n---\n\n".join(matched))
        print(f"  {category}: {len(matched)} sections → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Step 1: Fetching OSHA Part 1910 from eCFR API...")
    date      = get_latest_date("29")
    print(f"  Snapshot date: {date}")
    xml_bytes = fetch_part_xml(date, "29", "1910")
    with open(os.path.join(RAW_DIR, "1910_raw.xml"), "wb") as f:
        f.write(xml_bytes)
    print(f"  Raw XML saved ({len(xml_bytes):,} bytes)")

    print("\nStep 2: Preprocessing XML → plain text...")
    text = xml_to_text(xml_bytes)
    with open(os.path.join(RAW_DIR, "1910_full.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Clean text saved ({len(text):,} chars)")

    print("\nStep 3: Splitting into per-category files...")
    sections = split_into_sections(text)
    print(f"  Total sections found: {len(sections)}")
    save_category_files(sections)

    print("\nDone. Next: run build_knowledge_base.py to re-embed and re-index.")
    print("NOTE: Supabase documents table should be cleared before re-indexing")
    print("to avoid duplicate chunks from the old corpus.")


if __name__ == "__main__":
    main()