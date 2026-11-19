import csv
import mimetypes
from pathlib import Path

import requests

API_URL = "https://ooc-verify.onrender.com/api/v1/verify"
CASES_FILE = Path(__file__).resolve().parent / "evaluation_cases.csv"
RESULTS_FILE = Path(__file__).resolve().parent / "evaluation_results.csv"

VALID_LABELS = {"Genuine", "Misleading", "Inconclusive"}


def resolve_image_path(raw_path: str) -> Path:
    """
    Resolve raw image paths portably:
    - If absolute and existing, returns directly.
    - Otherwise probes candidate locations relative to backend root,
      repo root, script directory, test_images, and current working directory.
    """
    path_obj = Path(raw_path)
    if path_obj.is_absolute() and path_obj.exists():
        return path_obj

    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    script_dir = Path(__file__).resolve().parent

    candidates = [
        backend_root / path_obj,
        repo_root / path_obj,
        script_dir / path_obj,
        backend_root / "test_images" / path_obj.name,
        Path.cwd() / path_obj,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return (backend_root / path_obj).resolve()


def main():
    results = []

    if not CASES_FILE.exists():
        print(f"Cases file not found: {CASES_FILE}")
        return

    with CASES_FILE.open("r", newline="", encoding="utf-8") as file:
        cases = list(csv.DictReader(file))

    for index, case in enumerate(cases, start=1):
        raw_path = case.get("image_path", "").strip()
        caption = case.get("caption", "").strip()
        expected = case.get("expected_label", "").strip().capitalize()

        image_path = resolve_image_path(raw_path)

        if not image_path.exists():
            print(f"[{index}] Image not found: {raw_path} (resolved as: {image_path})")
            continue

        content_type = (
            mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
        )

        try:
            with image_path.open("rb") as image:
                response = requests.post(
                    API_URL,
                    files={
                        "image": (
                            image_path.name,
                            image,
                            content_type,
                        )
                    },
                    data={"caption": caption},
                    timeout=150,
                )

            response.raise_for_status()
            output = response.json()
            predicted = output.get("prediction", "").strip().capitalize()

            # Handle Inconclusive as a first-class valid verdict alongside Genuine and Misleading
            if predicted not in VALID_LABELS:
                print(f"[{index}] Unexpected prediction label: '{predicted}' (expected one of {VALID_LABELS})")
                continue

            if expected not in VALID_LABELS:
                print(f"[{index}] Invalid expected ground truth label: '{expected}' (expected one of {VALID_LABELS})")
                continue

            correct = predicted == expected

            results.append({
                "image_path": raw_path,
                "caption": caption,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "reason": output.get("reason", ""),
            })

            print(
                f"[{index}/{len(cases)}] "
                f"Expected: {expected} | Predicted: {predicted} | "
                f"{'PASS' if correct else 'FAIL'}"
            )

        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            print(f"[{index}] Request failed: {error}")

    if not results:
        print("No successful predictions. Check your API and CSV.")
        return

    with RESULTS_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # -------------------------------------------------------------------------
    # Evaluation Metrics & Reporting
    # -------------------------------------------------------------------------
    # Inconclusive is a recognized safety verdict indicating that visual signals
    # or evidence are insufficient/ambiguous to make a definitive verification.
    # To avoid presenting binary-only metrics as if all predictions were binary:
    # 1. Overall exact-match accuracy is computed across all evaluated cases.
    # 2. Prediction distribution is reported (Genuine, Misleading, Inconclusive/Abstention rate).
    # 3. Binary decision performance (Precision, Recall, F1 for Misleading vs Genuine)
    #    is reported explicitly for the resolved binary subset (coverage).
    # 4. Inconclusive breakdown details how ground-truth Misleading vs Genuine claims were handled.
    # -------------------------------------------------------------------------
    total_evaluated = len(results)
    total_correct = sum(1 for r in results if r["correct"])
    overall_accuracy = total_correct / total_evaluated if total_evaluated else 0.0

    inconclusive_preds = sum(1 for r in results if r["predicted"] == "Inconclusive")
    genuine_preds = sum(1 for r in results if r["predicted"] == "Genuine")
    misleading_preds = sum(1 for r in results if r["predicted"] == "Misleading")

    abstention_rate = inconclusive_preds / total_evaluated if total_evaluated else 0.0

    # Definitive binary subset: cases where both ground truth and prediction were binary
    binary_cases = [
        r for r in results
        if r["expected"] in {"Genuine", "Misleading"} and r["predicted"] in {"Genuine", "Misleading"}
    ]

    tp = sum(r["expected"] == "Misleading" and r["predicted"] == "Misleading" for r in binary_cases)
    fp = sum(r["expected"] == "Genuine" and r["predicted"] == "Misleading" for r in binary_cases)
    fn = sum(r["expected"] == "Misleading" and r["predicted"] == "Genuine" for r in binary_cases)
    tn = sum(r["expected"] == "Genuine" and r["predicted"] == "Genuine" for r in binary_cases)

    binary_accuracy = (tp + tn) / len(binary_cases) if binary_cases else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    misleading_inconclusive = sum(1 for r in results if r["expected"] == "Misleading" and r["predicted"] == "Inconclusive")
    genuine_inconclusive = sum(1 for r in results if r["expected"] == "Genuine" and r["predicted"] == "Inconclusive")

    print("\n" + "=" * 60)
    print("           Remote Verification Evaluation Summary")
    print("=" * 60)
    print(f"Total Evaluated Cases:  {total_evaluated}")
    print(f"Overall Exact Match:    {total_correct}/{total_evaluated} ({overall_accuracy:.2%})")
    print("-" * 60)
    print("Prediction Breakdown:")
    print(f"  - Genuine:            {genuine_preds} ({genuine_preds / total_evaluated:.1%})")
    print(f"  - Misleading:         {misleading_preds} ({misleading_preds / total_evaluated:.1%})")
    print(f"  - Inconclusive:       {inconclusive_preds} ({abstention_rate:.1%}) [Abstention Rate]")
    print("-" * 60)
    coverage_pct = len(binary_cases) / total_evaluated if total_evaluated else 0.0
    print(f"Definitive Binary Subset (Coverage: {len(binary_cases)}/{total_evaluated} = {coverage_pct:.1%}):")
    print("  Note: Metrics below evaluate discrimination solely on resolved binary cases.")
    print(f"  - Binary Accuracy:    {binary_accuracy:.2%}")
    print(f"  - Precision (OOC):    {precision:.2%}")
    print(f"  - Recall (OOC):       {recall:.2%}")
    print(f"  - F1-Score:           {f1:.2%}")
    print(f"  - Confusion (TP/FP/FN/TN): {tp}/{fp}/{fn}/{tn}")
    print("-" * 60)
    print("Abstention (Inconclusive) Analysis:")
    print(f"  - Ground-truth Misleading claims returned Inconclusive: {misleading_inconclusive}")
    print(f"  - Ground-truth Genuine claims returned Inconclusive:    {genuine_inconclusive}")
    print("=" * 60)
    print(f"Results saved to: {RESULTS_FILE.resolve()}\n")


if __name__ == "__main__":
    main()