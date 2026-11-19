import pytest
from app.schemas.verification import VerificationResult
from app.services.decision_service import (
    extract_qwen_prediction,
    make_final_decision,
)
from app.pipeline.orchestrator import _parse_qwen_prediction


def test_extract_qwen_prediction_successful():
    """Verify that Genuine, Misleading, and Inconclusive are correctly parsed."""
    sample_genuine = (
        "Prediction: Genuine\n"
        "Reason: The image clearly shows a dog running in the park.\n"
        "Visual Evidence: Dog running in grass"
    )
    sample_misleading = (
        "Prediction: Misleading\n"
        "Reason: The image depicts an apple, not a rocket.\n"
        "Visual Evidence: Red apple on a wooden table"
    )
    sample_inconclusive = (
        "Prediction: Inconclusive\n"
        "Reason: Cannot verify identity or location from pixels alone.\n"
        "Visual Evidence: Unidentifiable crowd at dusk"
    )

    assert extract_qwen_prediction(sample_genuine) == "Genuine"
    assert extract_qwen_prediction(sample_misleading) == "Misleading"
    assert extract_qwen_prediction(sample_inconclusive) == "Inconclusive"

    # Case insensitivity and whitespace tolerance
    assert extract_qwen_prediction("prediction:  misleading\nReason:...") == "Misleading"
    assert extract_qwen_prediction("PREDICTION: GENUINE") == "Genuine"
    assert extract_qwen_prediction("Prediction: Inconclusive.") == "Inconclusive"


def test_extract_qwen_prediction_unparseable_and_failed():
    """Verify that unparseable or error outputs return None."""
    assert extract_qwen_prediction(None) is None
    assert extract_qwen_prediction("") is None
    assert extract_qwen_prediction("Unparseable gibberish output without prediction label") is None
    assert extract_qwen_prediction("Prediction: Unknown\nReason: Not sure") is None
    assert extract_qwen_prediction("Prediction: Maybe") is None
    assert extract_qwen_prediction(
        "Prediction: Unknown\n"
        "Reason: MLLM analysis failed: CUDA out of memory\n"
        "Visual Evidence: Unavailable"
    ) is None


def test_orchestrator_parse_qwen_prediction_compatibility():
    """Verify _parse_qwen_prediction uses shared logic and defaults to Pending."""
    assert _parse_qwen_prediction("Prediction: Misleading\nReason: ...") == "Misleading"
    assert _parse_qwen_prediction("Prediction: Genuine\nReason: ...") == "Genuine"
    assert _parse_qwen_prediction("Prediction: Inconclusive\nReason: ...") == "Inconclusive"
    assert _parse_qwen_prediction("Garbage output") == "Pending"
    assert _parse_qwen_prediction("Reason: MLLM analysis failed: timeout") == "Pending"


def test_qwen_misleading_preserves_qwen_prediction_and_final_prediction():
    """
    Core explainability test:
    When Qwen detects Misleading, the final prediction is Misleading and
    the structured qwen_prediction is also preserved as 'Misleading'.
    """
    qwen_response = (
        "Prediction: Misleading\n"
        "Reason: The image shows ordinary street traffic, not an explosion.\n"
        "Visual Evidence: Vehicles on an asphalt road in daylight"
    )

    # Contextual claim ("protest in Delhi 2025") with alignment classifier and weak external evidence
    decision = make_final_decision(
        alignment_prediction="Misleading",
        alignment_confidence=0.62,
        qwen_response=qwen_response,
        evidence_items=[{"relation": "insufficient", "relevance": 0.30}],
        caption="Protest in Delhi turns violent in July 2025 according to police reports",
    )

    # 1. Final prediction preserves Qwen's Misleading verdict
    assert decision["prediction"] == "Misleading"

    # 2. Structured Qwen prediction is preserved as Misleading
    assert decision["qwen_prediction"] == "Misleading"
    assert extract_qwen_prediction(qwen_response) == "Misleading"


def test_verification_result_schema_qwen_prediction_field():
    """Verify VerificationResult model accepts and serializes qwen_prediction."""
    result = VerificationResult(
        id="test-uuid",
        prediction="Inconclusive",
        confidence_score=0.50,
        confidenceScore=0.50,
        clip_score=0.45,
        clipScore=0.45,
        alignment_score=0.45,
        alignmentScore=0.45,
        qwen_prediction="Misleading",
        qwenPrediction="Misleading",
        reason="Contextual claim requires further evidence",
        evidence=[],
        explanation="Full explanation",
        inconsistency_type="Evidence-aware multimodal decision",
        inconsistencyType="Evidence-aware multimodal decision",
        dataset="Custom Pair",
        pipeline_stages=[],
        is_stub=False,
    )

    data = result.model_dump()
    assert data["prediction"] == "Inconclusive"
    assert data["qwen_prediction"] == "Misleading"
    assert data["qwenPrediction"] == "Misleading"

    # Backward compatibility: default is None
    default_result = VerificationResult(
        id="default-uuid",
        prediction="Genuine",
        confidence_score=0.85,
        reason="Visual match",
        explanation="Visual explanation",
        inconsistency_type="None",
    )
    default_data = default_result.model_dump()
    assert default_data["qwen_prediction"] is None
    assert default_data["qwenPrediction"] is None
