"""
test_agent.py — Run 10 test queries through the MedTrace agent and display scores.
Run: python test_agent.py
"""
import asyncio, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from dotenv import load_dotenv
load_dotenv()

TEST_QUERIES = [
    "What are the symptoms of a heart attack?",
    "What is the interaction between warfarin and aspirin?",
    "What is the correct dose of metformin for diabetes?",
    "How do you treat anaphylaxis?",
    "What are the signs of diabetic ketoacidosis?",
    "What drugs are contraindicated in pregnancy?",
    "How is sepsis diagnosed and treated?",
    "What is the first-line treatment for hypertension?",
    "What are the side effects of ACE inhibitors?",
    "How should opioid overdose be managed?",
]

async def run_tests():
    from agent.main_agent import run_medtrace_agent

    print("=" * 70)
    print("MedTrace Agent — Test Suite")
    print("=" * 70)

    all_scores = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/10] Query: {query[:60]}...")
        try:
            result = await run_medtrace_agent(query)
            avg = result.get("avg_score", 0)
            scores = result.get("eval_scores", {})
            version = result.get("prompt_version", "v1")
            time_ms = result.get("processing_time_ms", 0)
            all_scores.append(avg)

            color = "\033[92m" if avg >= 7.5 else "\033[93m" if avg >= 6.0 else "\033[91m"
            reset = "\033[0m"
            print(f"  Avg Score: {color}{avg:.1f}/10{reset} | Version: {version} | Time: {time_ms:.0f}ms")
            print(f"  Accuracy:{scores.get('medical_accuracy',0):.1f} | Safety:{scores.get('safety',0):.1f} | Clarity:{scores.get('clarity',0):.1f}")
            print(f"  Answer: {result.get('answer','')[:120]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_scores.append(0)

    print("\n" + "=" * 70)
    if all_scores:
        overall = sum(all_scores) / len(all_scores)
        print(f"Overall Average Score: {overall:.2f}/10")
        print(f"Queries Scored ≥7.5: {sum(1 for s in all_scores if s >= 7.5)}/10")
        print(f"Queries Scored <6.5: {sum(1 for s in all_scores if s < 6.5)}/10")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_tests())
