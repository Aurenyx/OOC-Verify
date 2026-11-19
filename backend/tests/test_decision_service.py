import pytest
from app.services.decision_service import (
    extract_qwen_prediction,
    make_final_decision,
)


def test_qwen_genuine_alignment_misleading_returns_genuine():
    """
    Required Case 1:
    Qwen Genuine + alignment Misleading -> final Genuine.
    Alignment disagreement must not override a valid Qwen prediction.
    """
    result = make_final_decision(
        alignment_prediction="Misleading",
        alignment_confidence=0.6470,
        qwen_response=(
            "Prediction: Genuine\n"
            "Reason: The image directly depicts the scene described in the caption."
        ),
        evidence_items=[],
        caption="Residents stand beside a road flooded after heavy rain.",
    )

    assert result["prediction"] == "Genuine"
    assert result["qwen_prediction"] == "Genuine"
    assert result["confidence_score"] == 0.60
    assert "visual reasoning confirmed genuine consistency" in result["reason"]


def test_qwen_misleading_alignment_genuine_returns_misleading():
    """
    Required Case 2:
    Qwen Misleading + alignment Genuine -> final Misleading.
    Alignment disagreement must not override a valid Qwen prediction.
    """
    result = make_final_decision(
        alignment_prediction="Genuine",
        alignment_confidence=0.7800,
        qwen_response=(
            "Prediction: Misleading\n"
            "Reason: The image shows a sunny park, which contradicts the caption's blizzard claim."
        ),
        evidence_items=[],
        caption="Heavy blizzard shuts down roads in downtown Chicago.",
    )

    assert result["prediction"] == "Misleading"
    assert result["qwen_prediction"] == "Misleading"
    assert result["confidence_score"] == 0.60
    assert "visual reasoning determined the image does not support the caption" in result["reason"]


def test_qwen_genuine_alignment_genuine_returns_genuine():
    """
    Required Case 3:
    Qwen Genuine + alignment Genuine -> final Genuine.
    When both models agree, confidence receives the alignment corroboration boost.
    """
    result = make_final_decision(
        alignment_prediction="Genuine",
        alignment_confidence=0.8500,
        qwen_response=(
            "Prediction: Genuine\n"
            "Reason: Two golden retrievers are playing with a ball in a park."
        ),
        evidence_items=[],
        caption="Two golden retrievers play with a tennis ball on green grass.",
    )

    assert result["prediction"] == "Genuine"
    assert result["qwen_prediction"] == "Genuine"
    # confidence = min(0.85, 0.65 + 0.20 * 0.85) = 0.82
    assert result["confidence_score"] == 0.82
    assert "supported by the image-caption alignment signal" in result["reason"]


def test_qwen_misleading_alignment_misleading_returns_misleading():
    """
    Qwen Misleading + alignment Misleading -> final Misleading with agreement boost.
    """
    result = make_final_decision(
        alignment_prediction="Misleading",
        alignment_confidence=0.9000,
        qwen_response=(
            "Prediction: Misleading\n"
            "Reason: The image shows an office interior, not an outdoor parade."
        ),
        evidence_items=[],
        caption="Thousands attend annual parade in downtown Madrid.",
    )

    assert result["prediction"] == "Misleading"
    assert result["qwen_prediction"] == "Misleading"
    # confidence = min(0.85, 0.65 + 0.20 * 0.90) = 0.83
    assert result["confidence_score"] == 0.83
    assert "supported by the image-caption alignment signal" in result["reason"]


def test_qwen_failure_fallback_behavior():
    """
    Required Case 4a:
    Qwen failure (e.g. MLLM error / CUDA OOM) -> fallback Inconclusive (confidence 0.50).
    Never silently converts an invalid or failed response into Genuine.
    """
    failed_response = (
        "Prediction: Unknown\n"
        "Reason: MLLM analysis failed: CUDA out of memory\n"
        "Visual Evidence: Unavailable"
    )
    result = make_final_decision(
        alignment_prediction="Genuine",
        alignment_confidence=0.8500,
        qwen_response=failed_response,
        evidence_items=[],
        caption="A public event in progress.",
    )

    assert result["prediction"] == "Inconclusive"
    assert result["confidence_score"] == 0.50
    assert result["qwen_prediction"] == "Pending"
    assert "Insufficient evidence or unavailable multimodal signals" in result["reason"]


def test_qwen_invalid_prediction_fallback_behavior():
    """
    Required Case 4b:
    Qwen invalid prediction / unparseable output -> fallback Inconclusive (confidence 0.50).
    """
    unparseable_response = "Here is my reasoning without any formatted prediction."
    result = make_final_decision(
        alignment_prediction="Misleading",
        alignment_confidence=0.7500,
        qwen_response=unparseable_response,
        evidence_items=[],
        caption="Breaking news report.",
    )

    assert result["prediction"] == "Inconclusive"
    assert result["confidence_score"] == 0.50
    assert result["qwen_prediction"] == "Pending"


def test_qwen_inconclusive_prediction_fallback_behavior():
    """
    Required Case 4c:
    When Qwen itself explicitly outputs Inconclusive, final prediction is Inconclusive.
    """
    inconclusive_response = (
        "Prediction: Inconclusive\n"
        "Reason: The image is too blurry to verify the claimed individual."
    )
    result = make_final_decision(
        alignment_prediction="Genuine",
        alignment_confidence=0.8000,
        qwen_response=inconclusive_response,
        evidence_items=[],
        caption="Mayor speaks at press conference.",
    )

    assert result["prediction"] == "Inconclusive"
    assert result["confidence_score"] == 0.50
    assert result["qwen_prediction"] == "Inconclusive"


def test_qwen_genuine_not_overridden_by_contradictory_evidence():
    """
    Evidence contradiction does not override a valid Qwen Genuine prediction.
    """
    contradicting_evidence = [
        {"relation": "contradicts", "relevance": 0.88, "source": "News", "title": "Fact check"}
    ]
    result = make_final_decision(
        alignment_prediction="Genuine",
        alignment_confidence=0.75,
        qwen_response="Prediction: Genuine\nReason: Visual scene matches description.",
        evidence_items=contradicting_evidence,
        caption="Event caption.",
    )

    assert result["prediction"] == "Genuine"
    assert result["contradiction_score"] >= 0.70


def test_qwen_misleading_not_overridden_by_supporting_evidence_or_contextual_claim():
    """
    Contextual claim and supporting evidence do not override a valid Qwen Misleading prediction.
    """
    supporting_evidence = [
        {"relation": "supports", "relevance": 0.90, "source": "Reuters", "title": "Summit verified"}
    ]
    result = make_final_decision(
        alignment_prediction="Genuine",
        alignment_confidence=0.70,
        qwen_response="Prediction: Misleading\nReason: Image shows a basketball court, not a diplomatic summit.",
        evidence_items=supporting_evidence,
        caption="President addresses G7 summit in Berlin in 2026.",
    )

    assert result["prediction"] == "Misleading"
    assert result["qwen_prediction"] == "Misleading"


def test_make_final_decision_preserves_substantive_mllm_explanation():
    """
    Ensure the substantive MLLM reasoning from Qwen is preserved as the explanation,
    rather than being replaced with a generic stage-scores summary.
    """
    mllm_reasoning = (
        "Prediction: Genuine\n"
        "Reason: The image directly depicts the flood scene described in the caption.\n"
        "Visual Evidence: People standing near road waterlogged with rainwater.\n"
        "Claim Assessment: direct_visual claim supported by pixels."
    )
    result = make_final_decision(
        alignment_prediction="Genuine",
        alignment_confidence=0.82,
        qwen_response=mllm_reasoning,
        evidence_items=[],
        caption="Flood waters submerge residential road.",
    )

    assert result["explanation"] == mllm_reasoning.strip()
    assert result["prediction"] == "Genuine"


def test_make_final_decision_fallback_explanation_when_mllm_unavailable_or_failed():
    """
    Ensure that when Qwen fails or is unavailable, the explanation cleanly
    falls back to 'MLLM explanation unavailable' without inventing text.
    """
    failed_qwen = "Prediction: Unknown\nReason: MLLM analysis failed: CUDA out of memory"
    result = make_final_decision(
        alignment_prediction="Genuine",
        alignment_confidence=0.80,
        qwen_response=failed_qwen,
        evidence_items=[],
        caption="Some caption.",
    )

    assert result["explanation"] == "MLLM explanation unavailable"
    assert result["prediction"] == "Inconclusive"

    empty_result = make_final_decision(
        alignment_prediction="Genuine",
        alignment_confidence=0.80,
        qwen_response="",
        evidence_items=[],
        caption="Some caption.",
    )

    assert empty_result["explanation"] == "MLLM explanation unavailable"
    assert empty_result["prediction"] == "Inconclusive"
