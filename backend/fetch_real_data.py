"""
fetch_real_data.py — MedTrace Real Medical Data Fetcher
=======================================================
Fetches real medical Q&A data from 3 sources:
  1. MedQuAD   — NIH-sourced 47K+ Q&A pairs (via GitHub raw)
  2. OpenFDA   — FDA drug labeling & adverse event data
  3. MedlinePlus — NIH health topics API

Run:  python fetch_real_data.py
Output: knowledge_base/data/medical_qa.json (replaces old file)
"""

import json
import time
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import urllib.request
import urllib.parse
import urllib.error

OUTPUT_FILE = Path(__file__).parent / "knowledge_base" / "data" / "medical_qa.json"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def http_get(url, timeout=20):
    """Simple HTTP GET with retry logic (no external deps needed)."""
    headers = {"User-Agent": "MedTrace-DataFetcher/1.0 (educational use)"}
    req = urllib.request.Request(url, headers=headers)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            print(f"   ⚠️  Attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
    return None

def clean_text(text):
    """Remove extra whitespace and HTML-like tags."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─── Source 1: MedQuAD (NIH Q&A via GitHub) ──────────────────────────────────

MEDQUAD_URLS = [
    # These are the raw JSON exports of MedQuAD categories on GitHub (abachaa/MedQuAD)
    # We use MedlinePlus Health Topics category as primary
    ("https://raw.githubusercontent.com/abachaa/MedQuAD/master/README.md", None),  # just a check
]

# MedQuAD doesn't have a simple JSON API — we'll use their XML files via known structure
# Instead we use a curated JSONL version hosted on HuggingFace datasets (public, no auth)
MEDQUAD_HF_URL = "https://datasets-server.huggingface.co/rows?dataset=lavita%2FMedQuAD&config=default&split=train&offset={offset}&length=100"

def fetch_medquad(max_records=500):
    """Fetch MedQuAD Q&A pairs from HuggingFace dataset API."""
    print("\n📥 [1/3] Fetching MedQuAD from HuggingFace...")
    qa_pairs = []
    offset = 0
    batch = 100

    while len(qa_pairs) < max_records:
        url = MEDQUAD_HF_URL.format(offset=offset)
        print(f"   Fetching rows {offset}–{offset+batch}...")
        raw = http_get(url)
        if not raw:
            print("   ❌ Failed to fetch MedQuAD batch")
            break
        try:
            data = json.loads(raw)
            rows = data.get("rows", [])
            if not rows:
                break
            for row in rows:
                r = row.get("row", {})
                q = clean_text(r.get("question", ""))
                a = clean_text(r.get("answer", ""))
                topic = clean_text(r.get("focus_area", r.get("source", "General Medicine")))
                source_doc = r.get("source", "MedQuAD-NIH")
                if q and a and len(a) > 50:
                    qa_pairs.append({
                        "question": q,
                        "answer": a[:2000],  # cap at 2000 chars
                        "topic": topic,
                        "source": f"MedQuAD-NIH/{source_doc}",
                        "category": categorize(q)
                    })
            offset += batch
            time.sleep(0.3)  # be polite
        except Exception as e:
            print(f"   ❌ Parse error: {e}")
            break

    print(f"   ✅ MedQuAD: {len(qa_pairs)} Q&A pairs fetched")
    return qa_pairs

def categorize(question):
    """Simple heuristic to assign category."""
    q = question.lower()
    if any(w in q for w in ["symptom", "sign", "feel", "pain", "hurt"]):
        return "symptom_diagnosis"
    if any(w in q for w in ["treat", "therapy", "cure", "manage", "surgery"]):
        return "treatment_protocol"
    if any(w in q for w in ["dose", "dosage", "how much", "mg", "tablet"]):
        return "dosage_info"
    if any(w in q for w in ["interact", "contraindic", "side effect", "avoid"]):
        return "drug_interaction"
    if any(w in q for w in ["emergency", "urgent", "911", "overdose", "poison"]):
        return "emergency"
    if any(w in q for w in ["diagnos", "test", "criteria", "screen"]):
        return "symptom_diagnosis"
    return "general_medicine"

# ─── Source 2: OpenFDA Drug Labels ───────────────────────────────────────────

OPENFDA_URL = "https://api.fda.gov/drug/label.json?search=_exists_:warnings_and_cautions+AND+_exists_:dosage_and_administration&limit=100&skip={skip}"

# Drug interaction search
OPENFDA_INTERACTION_URL = "https://api.fda.gov/drug/label.json?search=_exists_:drug_interactions&limit=100&skip={skip}"

DRUGS_TO_FETCH = [
    "metformin", "lisinopril", "atorvastatin", "amoxicillin", "ibuprofen",
    "aspirin", "metoprolol", "amlodipine", "omeprazole", "sertraline",
    "levothyroxine", "warfarin", "insulin", "prednisone", "furosemide",
    "gabapentin", "hydrochlorothiazide", "albuterol", "losartan", "pantoprazole"
]

def fetch_openfda(max_drugs=20):
    """Fetch real FDA drug label data and convert to Q&A format."""
    print("\n📥 [2/3] Fetching OpenFDA drug label data...")
    qa_pairs = []

    for drug in DRUGS_TO_FETCH[:max_drugs]:
        url = f"https://api.fda.gov/drug/label.json?search=openfda.generic_name:{urllib.parse.quote(drug)}&limit=1"
        print(f"   Fetching: {drug}...")
        raw = http_get(url)
        if not raw:
            continue
        try:
            data = json.loads(raw)
            results = data.get("results", [])
            if not results:
                continue
            r = results[0]
            brand = r.get("openfda", {}).get("brand_name", [drug.title()])[0]
            generic = r.get("openfda", {}).get("generic_name", [drug])[0].title()

            # Q1: Indications
            indications = r.get("indications_and_usage", [""])[0]
            if indications and len(indications) > 80:
                qa_pairs.append({
                    "question": f"What is {generic} ({brand}) used for?",
                    "answer": clean_text(indications)[:2000],
                    "topic": "Pharmacology",
                    "source": f"OpenFDA-Label/{brand}",
                    "category": "treatment_protocol"
                })

            # Q2: Dosage
            dosage = r.get("dosage_and_administration", [""])[0]
            if dosage and len(dosage) > 80:
                qa_pairs.append({
                    "question": f"What is the recommended dosage for {generic}?",
                    "answer": clean_text(dosage)[:2000],
                    "topic": "Medication Dosage",
                    "source": f"OpenFDA-Label/{brand}",
                    "category": "dosage_info"
                })

            # Q3: Warnings
            warnings = r.get("warnings_and_cautions", r.get("warnings", [""]))[0]
            if warnings and len(warnings) > 80:
                qa_pairs.append({
                    "question": f"What are the warnings and precautions for {generic}?",
                    "answer": clean_text(warnings)[:2000],
                    "topic": "Drug Safety",
                    "source": f"OpenFDA-Label/{brand}",
                    "category": "drug_interaction"
                })

            # Q4: Contraindications
            contraindications = r.get("contraindications", [""])[0]
            if contraindications and len(contraindications) > 80:
                qa_pairs.append({
                    "question": f"What are the contraindications of {generic}?",
                    "answer": clean_text(contraindications)[:2000],
                    "topic": "Drug Interactions",
                    "source": f"OpenFDA-Label/{brand}",
                    "category": "drug_interaction"
                })

            # Q5: Drug Interactions
            interactions = r.get("drug_interactions", [""])[0]
            if interactions and len(interactions) > 80:
                qa_pairs.append({
                    "question": f"What drugs interact with {generic}?",
                    "answer": clean_text(interactions)[:2000],
                    "topic": "Drug Interactions",
                    "source": f"OpenFDA-Label/{brand}",
                    "category": "drug_interaction"
                })

            time.sleep(0.2)  # FDA rate limit
        except Exception as e:
            print(f"   ⚠️  Error for {drug}: {e}")

    print(f"   ✅ OpenFDA: {len(qa_pairs)} Q&A pairs fetched")
    return qa_pairs

# ─── Source 3: MedlinePlus Health Topics API ──────────────────────────────────

MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term={term}&retmax=20"

HEALTH_TOPICS = [
    "diabetes", "hypertension", "heart attack", "stroke", "asthma",
    "pneumonia", "sepsis", "appendicitis", "kidney disease", "liver disease",
    "anemia", "thyroid", "depression", "anxiety", "COPD",
    "heart failure", "atrial fibrillation", "deep vein thrombosis",
    "pulmonary embolism", "urinary tract infection"
]

def fetch_medlineplus():
    """Fetch NIH MedlinePlus health topic summaries and convert to Q&A."""
    print("\n📥 [3/3] Fetching MedlinePlus (NIH) health topics...")
    qa_pairs = []

    for topic in HEALTH_TOPICS:
        url = MEDLINEPLUS_URL.format(term=urllib.parse.quote(topic))
        print(f"   Fetching: {topic}...")
        raw = http_get(url)
        if not raw:
            continue
        try:
            root = ET.fromstring(raw)
            # MedlinePlus returns XML with <document> elements
            for doc in root.findall(".//document"):
                title_el = doc.find(".//content[@name='title']")
                summary_el = doc.find(".//content[@name='FullSummary']")
                if title_el is None or summary_el is None:
                    continue
                title = clean_text(title_el.text or "")
                summary = clean_text(summary_el.text or "")
                if not title or not summary or len(summary) < 100:
                    continue

                qa_pairs.append({
                    "question": f"What is {title}? Explain the symptoms, causes, and treatment.",
                    "answer": summary[:2000],
                    "topic": title,
                    "source": "MedlinePlus-NIH",
                    "category": categorize(title)
                })
                break  # 1 best result per topic
            time.sleep(0.3)
        except Exception as e:
            print(f"   ⚠️  Error for {topic}: {e}")

    print(f"   ✅ MedlinePlus: {len(qa_pairs)} Q&A pairs fetched")
    return qa_pairs

# ─── Deduplication ────────────────────────────────────────────────────────────

def deduplicate(qa_pairs):
    """Remove near-duplicate questions based on first 80 chars."""
    seen = set()
    unique = []
    for pair in qa_pairs:
        key = pair["question"][:80].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(pair)
    return unique

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("🏥 MedTrace Real Data Fetcher")
    print("=" * 60)

    all_qa = []

    # 1. MedQuAD
    try:
        medquad_data = fetch_medquad(max_records=500)
        all_qa.extend(medquad_data)
    except Exception as e:
        print(f"❌ MedQuAD failed: {e}")

    # 2. OpenFDA
    try:
        fda_data = fetch_openfda(max_drugs=20)
        all_qa.extend(fda_data)
    except Exception as e:
        print(f"❌ OpenFDA failed: {e}")

    # 3. MedlinePlus
    try:
        medline_data = fetch_medlineplus()
        all_qa.extend(medline_data)
    except Exception as e:
        print(f"❌ MedlinePlus failed: {e}")

    # Deduplicate
    all_qa = deduplicate(all_qa)

    # Save
    print(f"\n💾 Saving {len(all_qa)} unique Q&A pairs to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Backup old file
    old_file = OUTPUT_FILE.with_suffix(".old.json")
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.rename(old_file)
        print(f"   📦 Old file backed up as: {old_file.name}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ Done! Total Q&A pairs saved: {len(all_qa)}")
    print(f"   📁 File: {OUTPUT_FILE}")
    print("\n📊 Breakdown by source:")
    source_counts = {}
    for pair in all_qa:
        src = pair["source"].split("/")[0]
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in source_counts.items():
        print(f"   {src}: {count}")
    print("\n📊 Breakdown by category:")
    cat_counts = {}
    for pair in all_qa:
        cat = pair.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"   {cat}: {count}")
    print("=" * 60)
    print("\n🚀 Next step: Reload the knowledge base with:")
    print("   python -c \"from knowledge_base.loader import initialize_knowledge_base; initialize_knowledge_base(force_reload=True)\"")

if __name__ == "__main__":
    main()
