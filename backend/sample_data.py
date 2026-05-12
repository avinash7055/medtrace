"""
sample_data.py — Populate MedTrace knowledge base with 50 medical Q&A examples.
Run: python sample_data.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

EXTRA_QA = [
  {"question":"What is the normal range for blood pressure?","answer":"Normal BP: <120/80 mmHg. Elevated: 120-129/<80. Stage 1 HTN: 130-139/80-89. Stage 2 HTN: ≥140/90. Hypertensive crisis: >180/120 — seek immediate care. Measure after 5 min rest, no caffeine/exercise for 30 min. Two readings 1-2 min apart, average them. [Source: ACC/AHA 2017]","topic":"Diagnosis","source":"MedTraceKB-Cardiology","category":"symptom_diagnosis"},
  {"question":"What is the antidote for benzodiazepine overdose?","answer":"Flumazenil 0.2mg IV over 30 seconds, repeat 0.2mg every 60 seconds up to 1mg total. CAUTION: may precipitate seizures in benzodiazepine-dependent patients or those with TCA co-ingestion. Short duration (1-2h) — resedation can occur. Primarily supportive care: airway management, assisted ventilation. Activated charcoal if recent ingestion and no aspiration risk. [Source: Poisindex, UpToDate]","topic":"Emergency Medicine","source":"MedTraceKB-Emergency","category":"emergency"},
  {"question":"What are the signs of anemia?","answer":"Anemia symptoms: fatigue, pallor (conjunctival, palmar), dyspnea on exertion, palpitations, dizziness, headache, cold extremities. Severe: angina, syncope. Signs: tachycardia, flow murmur, splenomegaly (hemolytic). Classification: microcytic (IDA, thalassemia), normocytic (chronic disease, aplastic), macrocytic (B12/folate deficiency, hypothyroid). Labs: CBC, reticulocyte count, iron studies, B12, folate, peripheral smear. Treat underlying cause. Iron deficiency: ferrous sulfate 325mg TID between meals. B12 deficiency: IM cyanocobalamin or high-dose oral 1000mcg daily. [Source: ASH Guidelines]","topic":"Diagnosis","source":"MedTraceKB-Hematology","category":"symptom_diagnosis"},
  {"question":"How is asthma classified and treated?","answer":"Asthma severity: Intermittent (≤2 days/week) → SABA PRN (albuterol inhaler). Mild persistent (>2 days/week) → add low-dose ICS (fluticasone 88mcg BD). Moderate persistent (daily symptoms) → medium-dose ICS + LABA (salmeterol). Severe persistent (continuous) → high-dose ICS + LABA + consider OCS. Acute exacerbation: SABA every 20 min × 3, ipratropium, systemic steroids (prednisone 40-60mg), magnesium sulfate 2g IV for severe. Intubate if respiratory failure. Monitor peak flow and SpO2. Action plan for all patients. [Source: NAEPP/GINA Guidelines]","topic":"Treatment Protocols","source":"MedTraceKB-Pulmonology","category":"treatment_protocol"},
  {"question":"What is the interaction between fluoroquinolones and antacids?","answer":"Fluoroquinolones (ciprofloxacin, levofloxacin) + antacids containing aluminum, magnesium, calcium, iron, or zinc: MODERATE interaction — chelation reduces fluoroquinolone absorption by 50-90%. Management: Take fluoroquinolone 2 hours BEFORE or 6 hours AFTER antacids, dairy products, or supplements containing divalent/trivalent cations. Sucralfate also reduces absorption — same timing. IV fluoroquinolones unaffected. This interaction can lead to treatment failure for serious infections. [Source: Micromedex Drug Interactions]","topic":"Drug Interactions","source":"MedTraceKB-Pharmacology","category":"drug_interaction"},
  {"question":"What is the typical dose of prednisone for acute asthma?","answer":"Prednisone for acute asthma exacerbation: Adults — 40-60mg/day for 5-7 days (no taper needed for short courses). Children — 1-2mg/kg/day (max 60mg) for 3-5 days. IV methylprednisolone 1-2mg/kg/day if unable to tolerate oral. Continue until PEFR >70% predicted or symptoms resolve. No superiority of higher doses or longer courses shown for typical exacerbation. For moderate-severe persistent asthma maintenance: lowest effective dose, prefer inhaled corticosteroids to minimize systemic effects. [Source: NAEPP EPR-3, GINA 2023]","topic":"Medication Dosage","source":"MedTraceKB-Pulmonology","category":"dosage_info"},
  {"question":"What are the symptoms of hypothyroidism?","answer":"Hypothyroidism symptoms: fatigue, weight gain, cold intolerance, constipation, dry skin, hair loss, bradycardia, menstrual irregularities, depression, cognitive slowing ('brain fog'), hoarse voice, myxedema (puffiness). Severe (myxedema coma): hypothermia, altered consciousness — emergency. Labs: elevated TSH (most sensitive), low free T4. Treatment: levothyroxine 1.6mcg/kg/day (start 25-50mcg in elderly/cardiac disease). Recheck TSH in 6-8 weeks, adjust by 12.5-25mcg. Take on empty stomach 30-60 min before breakfast. Drug interactions: calcium, iron, PPIs reduce absorption. [Source: ATA Hypothyroidism Guidelines 2014]","topic":"Diagnosis","source":"MedTraceKB-Endocrinology","category":"symptom_diagnosis"},
  {"question":"What is the first-line antibiotic for urinary tract infections?","answer":"Uncomplicated UTI (cystitis) in women: First-line — nitrofurantoin 100mg ER twice daily × 5 days (avoid if eGFR <30) OR trimethoprim-sulfamethoxazole DS twice daily × 3 days (check local resistance <20%). Fosfomycin 3g single dose alternative. NOT first-line: fluoroquinolones (reserve for complicated UTI). Complicated UTI/pyelonephritis: fluoroquinolone (ciprofloxacin 500mg BD × 7 days) or TMP-SMX × 14 days based on culture. Culture-guided therapy preferred. Asymptomatic bacteriuria: treat ONLY in pregnancy (amoxicillin/nitrofurantoin × 5-7 days) or pre-urologic procedure. [Source: IDSA UTI Guidelines 2011, updated 2019]","topic":"Treatment Protocols","source":"MedTraceKB-Infectious","category":"treatment_protocol"},
  {"question":"What is the mechanism of warfarin and how is it monitored?","answer":"Warfarin inhibits vitamin K epoxide reductase (VKORC1), blocking regeneration of vitamin K, which is required for activation of clotting factors II, VII, IX, X and proteins C and S. Narrow therapeutic index — requires INR monitoring. Standard target INR: 2.0-3.0 (most indications including AF, DVT/PE). Mechanical mitral valve: 2.5-3.5. Monitor: INR baseline then after initiation every 3-5 days until stable, then monthly. Interactions: hundreds of drugs, foods. Vitamin K antagonist — consistent dietary K+ intake recommended. Reversal: for major bleeding — 4-factor PCC + vitamin K 10mg IV. Elective: oral vitamin K. [Source: CHEST Antithrombotic Guidelines]","topic":"Pharmacology","source":"MedTraceKB-Cardiology","category":"treatment_protocol"},
  {"question":"When should statins be started after a myocardial infarction?","answer":"Post-MI statin therapy: Start HIGH-INTENSITY statin as soon as possible after STEMI/NSTEMI — ideally within 24 hours. High-intensity: atorvastatin 40-80mg or rosuvastatin 20-40mg (reduces LDL ≥50%). Target LDL: <70 mg/dL (Class I recommendation); consider <55 mg/dL for very high-risk. If LDL still elevated at max statin: add ezetimibe 10mg (IMPROVE-IT trial +6.4% RRR). Add PCSK9 inhibitor (evolocumab/alirocumab) if LDL ≥70 despite statin + ezetimibe. Statins are lifelong — do NOT stop after 'completing a course'. [Source: ACC/AHA 2019 Cholesterol Guidelines, 2023 ACS Update]","topic":"Treatment Protocols","source":"MedTraceKB-Cardiology","category":"treatment_protocol"}
]

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "knowledge_base", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Load existing data
    existing_file = os.path.join(data_dir, "medical_qa.json")
    existing = []
    if os.path.exists(existing_file):
        with open(existing_file) as f:
            existing = json.load(f)
    print(f"Existing Q&A pairs: {len(existing)}")

    # Save extra data
    extra_file = os.path.join(data_dir, "medical_qa_extra.json")
    with open(extra_file, "w") as f:
        json.dump(EXTRA_QA, f, indent=2)
    print(f"Saved {len(EXTRA_QA)} extra Q&A pairs to {extra_file}")

    total = len(existing) + len(EXTRA_QA)
    print(f"Total Q&A pairs available: {total}")

    # Now index into ChromaDB
    print("\nIndexing into ChromaDB...")
    from knowledge_base.loader import initialize_knowledge_base
    count = initialize_knowledge_base(force_reload=True)
    print(f"✅ ChromaDB indexed: {count} document chunks")
    print("\nSample data loading complete!")

if __name__ == "__main__":
    main()
