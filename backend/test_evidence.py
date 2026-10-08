from app.services.evidence_service import retrieve_evidence


caption = (
    "Donald Trump meets Ukrainian President "
    "Volodymyr Zelenskyy"
)

results = retrieve_evidence(
    caption,
    max_results=5,
)

print("\n==============================")
print("EVIDENCE RESULTS")
print("==============================")

for index, result in enumerate(results, start=1):

    print(f"\n--- Result {index} ---")
    print("Source:", result["source"])
    print("Title:", result["title"])
    print("Date:", result["publishedDate"])
    print("URL:", result["url"])
    print("Excerpt:", result["excerpt"])