from app.services.evidence_service import retrieve_evidence
from app.services.evidence_analysis_service import analyze_evidence


caption = (
    "Donald Trump meets Ukrainian President "
    "Volodymyr Zelenskyy"
)

print("\nRetrieving evidence...")

evidence = retrieve_evidence(
    caption,
    max_results=5,
)

print(f"Retrieved {len(evidence)} evidence items.")

print("\nAnalyzing evidence with Qwen...")

results = analyze_evidence(
    caption=caption,
    evidence_items=evidence,
)

print("\n==============================")
print("ANALYZED EVIDENCE")
print("==============================")

for index, result in enumerate(results, start=1):
    print(f"\n--- Evidence {index} ---")
    print("Source:", result["source"])
    print("Title:", result["title"])
    print("Relation:", result["relation"])
    print("Relevance:", result["relevance"])
    print("Reason:", result.get("analysis_reason", ""))