import logging
import os
import tempfile
import uuid
from typing import Optional

from app.schemas.verification import (
    PipelineStageResult,
    VerificationResult,
)
from app.services.clip_service import clip_service
from app.services.qwen_service import analyze_image
from app.services.alignment_classifier_service import (
    predict as predict_alignment,
)
from app.services.evidence_service import retrieve_evidence
from app.services.evidence_analysis_service import analyze_evidence
from app.services.decision_service import make_final_decision


logger = logging.getLogger(__name__)


def _run_qwen(
    image_bytes: bytes,
    filename: Optional[str],
    caption: str,
) -> str:
    """Run Qwen on a temporary image file."""

    suffix = os.path.splitext(filename or ".jpg")[1].lower()

    if suffix not in [".jpg", ".jpeg", ".png", ".webp"]:
        suffix = ".jpg"

    temp_image_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(image_bytes)
            temp_image_path = temp_file.name

        return analyze_image(
            image_path=temp_image_path,
            caption=caption,
        )

    except Exception as exc:
        logger.exception("Qwen analysis failed.")

        return (
            "Prediction: Unknown\n"
            f"Reason: MLLM analysis failed: {exc}\n"
            "Visual Evidence: Unavailable"
        )

    finally:
        if (
            temp_image_path
            and os.path.exists(temp_image_path)
        ):
            os.remove(temp_image_path)


def _parse_qwen_prediction(qwen_response: str) -> str:
    """Extract Genuine/Misleading from Qwen response."""

    response_lower = qwen_response.lower()

    if "prediction: misleading" in response_lower:
        return "Misleading"

    if "prediction: genuine" in response_lower:
        return "Genuine"

    return "Pending"


def _calculate_simple_confidence(
    prediction: str,
    qwen_response: str,
) -> float:
    """
    Confidence used for the MLLM-only configuration.

    This is a heuristic confidence, not a calibrated probability.
    """

    if prediction == "Pending":
        return 0.50

    response_lower = qwen_response.lower()

    if "prediction:" in response_lower:
        return 0.80

    return 0.70


def run_verification(
    image_bytes: bytes,
    filename: Optional[str] = None,
    caption: str = "",
    dataset: Optional[str] = None,
    configuration: Optional[str] = None,
) -> VerificationResult:

    configuration = (
        configuration.strip().lower()
        if configuration
        else "proposed"
    )

    if configuration not in {
        "alignment_only",
        "mllm_only",
        "proposed",
    }:
        raise ValueError(
            f"Unsupported configuration: {configuration}"
        )

    stages = []

    # =========================================================
    # 1. Input validation
    # =========================================================

    if not image_bytes:
        raise ValueError("Image data is empty.")

    if not caption.strip():
        raise ValueError("Caption is required.")

    stages.append(
        PipelineStageResult(
            stage="input_validation",
            status="completed",
            detail="Image and caption received successfully.",
        )
    )

    # =========================================================
    # ALIGNMENT-ONLY CONFIGURATION
    #
    # Image + Caption
    #       ↓
    #      CLIP
    #       ↓
    # Alignment Classifier
    #       ↓
    #     Decision
    # =========================================================

    if configuration == "alignment_only":

        # -----------------------------------------------------
        # 2. CLIP alignment
        # -----------------------------------------------------

        clip_score = clip_service.compute_alignment_score(
            image_input=image_bytes,
            caption=caption,
        )

        stages.append(
            PipelineStageResult(
                stage="image_text_analysis",
                status="completed",
                detail=(
                    "Image and caption prepared for "
                    "CLIP alignment analysis."
                ),
            )
        )

        stages.append(
            PipelineStageResult(
                stage="cross_modal_alignment",
                status="completed",
                detail=(
                    "CLIP image-caption alignment calculated."
                ),
                score=float(clip_score),
            )
        )

        # -----------------------------------------------------
        # 3. Alignment classifier
        # -----------------------------------------------------

        alignment_result = predict_alignment(
            alignment_score=clip_score
        )

        alignment_prediction = alignment_result["prediction"]
        alignment_confidence = alignment_result["confidence"]

        stages.append(
            PipelineStageResult(
                stage="alignment_classification",
                status="completed",
                detail=(
                    "Alignment classifier prediction: "
                    f"{alignment_prediction}"
                ),
                score=float(alignment_confidence),
            )
        )

        # -----------------------------------------------------
        # 4. Skip MLLM
        # -----------------------------------------------------

        qwen_response = (
            "Prediction: Unknown\n"
            "Reason: MLLM reasoning was not executed "
            "in alignment-only configuration.\n"
            "Visual Evidence: Not evaluated by MLLM."
        )

        stages.append(
            PipelineStageResult(
                stage="mllm_reasoning",
                status="skipped",
                detail=(
                    "MLLM reasoning skipped because "
                    "configuration is alignment_only."
                ),
            )
        )

        # -----------------------------------------------------
        # 5. Skip evidence retrieval
        # -----------------------------------------------------

        evidence_items = []

        stages.append(
            PipelineStageResult(
                stage="evidence_retrieval",
                status="skipped",
                detail=(
                    "External evidence retrieval skipped "
                    "because configuration is alignment_only."
                ),
            )
        )

        # -----------------------------------------------------
        # 6. Skip evidence analysis
        # -----------------------------------------------------

        stages.append(
            PipelineStageResult(
                stage="evidence_analysis",
                status="skipped",
                detail=(
                    "Evidence analysis skipped because "
                    "configuration is alignment_only."
                ),
            )
        )

        # -----------------------------------------------------
        # 7. Alignment-only decision
        # -----------------------------------------------------

        final_prediction = alignment_prediction

        final_confidence = float(alignment_confidence)

        if final_prediction == "Genuine":
            reason = (
                "The alignment classifier found strong "
                "image-caption semantic alignment."
            )
        else:
            reason = (
                "The alignment classifier found insufficient "
                "image-caption semantic alignment."
            )

        explanation = (
            "Alignment-only configuration used the CLIP "
            "image-caption alignment score followed by the "
            "trained alignment classifier. MLLM reasoning and "
            "external evidence were intentionally excluded."
        )

        stages.append(
            PipelineStageResult(
                stage="final_decision",
                status="completed",
                detail=(
                    "Final prediction: "
                    f"{final_prediction}"
                ),
                score=final_confidence,
            )
        )

        inconsistency_type = (
            "CLIP alignment classification"
        )

    # =========================================================
    # MLLM-ONLY CONFIGURATION
    #
    # Image + Caption
    #       ↓
    #      Qwen
    #       ↓
    #     Decision
    # =========================================================

    elif configuration == "mllm_only":

        # -----------------------------------------------------
        # 2. Image preparation
        # -----------------------------------------------------

        stages.append(
            PipelineStageResult(
                stage="image_text_analysis",
                status="completed",
                detail=(
                    "Image and caption prepared for "
                    "MLLM reasoning."
                ),
            )
        )

        # -----------------------------------------------------
        # 3. Skip CLIP
        # -----------------------------------------------------

        clip_score = None

        stages.append(
            PipelineStageResult(
                stage="cross_modal_alignment",
                status="skipped",
                detail=(
                    "CLIP alignment skipped because "
                    "configuration is mllm_only."
                ),
            )
        )

        # -----------------------------------------------------
        # 4. Skip alignment classifier
        # -----------------------------------------------------

        alignment_prediction = "Pending"
        alignment_confidence = 0.0

        stages.append(
            PipelineStageResult(
                stage="alignment_classification",
                status="skipped",
                detail=(
                    "Alignment classifier skipped because "
                    "configuration is mllm_only."
                ),
            )
        )

        # -----------------------------------------------------
        # 5. Qwen reasoning
        # -----------------------------------------------------

        qwen_response = _run_qwen(
            image_bytes=image_bytes,
            filename=filename,
            caption=caption,
        )

        qwen_prediction = _parse_qwen_prediction(
            qwen_response
        )

        qwen_confidence = _calculate_simple_confidence(
            qwen_prediction,
            qwen_response,
        )

        stages.append(
            PipelineStageResult(
                stage="mllm_reasoning",
                status="completed",
                detail=qwen_response[:1000],
                score=qwen_confidence,
            )
        )

        # -----------------------------------------------------
        # 6. Skip evidence retrieval
        # -----------------------------------------------------

        evidence_items = []

        stages.append(
            PipelineStageResult(
                stage="evidence_retrieval",
                status="skipped",
                detail=(
                    "External evidence retrieval skipped "
                    "because configuration is mllm_only."
                ),
            )
        )

        # -----------------------------------------------------
        # 7. Skip evidence analysis
        # -----------------------------------------------------

        stages.append(
            PipelineStageResult(
                stage="evidence_analysis",
                status="skipped",
                detail=(
                    "Evidence analysis skipped because "
                    "configuration is mllm_only."
                ),
            )
        )

        # -----------------------------------------------------
        # 8. MLLM-only decision
        # -----------------------------------------------------

        final_prediction = qwen_prediction
        final_confidence = qwen_confidence

        if final_prediction == "Genuine":
            reason = (
                "The multimodal language model determined that "
                "the image provides visual support for the caption."
            )
        elif final_prediction == "Misleading":
            reason = (
                "The multimodal language model determined that "
                "the image does not sufficiently support the caption."
            )
        else:
            reason = (
                "The multimodal language model did not return "
                "a valid Genuine or Misleading prediction."
            )

        explanation = (
            "MLLM-only configuration used Qwen multimodal "
            "reasoning on the image-caption pair. CLIP alignment, "
            "alignment classification, and external evidence "
            "retrieval were intentionally excluded."
        )

        stages.append(
            PipelineStageResult(
                stage="final_decision",
                status="completed",
                detail=(
                    "Final prediction: "
                    f"{final_prediction}"
                ),
                score=final_confidence,
            )
        )

        inconsistency_type = (
            "MLLM multimodal reasoning"
        )

    # =========================================================
    # PROPOSED CONFIGURATION
    #
    # Image + Caption
    #       ↓
    #      CLIP
    #       ↓
    # Alignment Classifier
    #       ↓
    #      Qwen
    #       ↓
    # Evidence Retrieval
    #       ↓
    # Evidence Analysis
    #       ↓
    # Evidence-Aware Fusion
    # =========================================================

    else:

        # -----------------------------------------------------
        # 2. CLIP alignment
        # -----------------------------------------------------

        clip_score = clip_service.compute_alignment_score(
            image_input=image_bytes,
            caption=caption,
        )

        stages.append(
            PipelineStageResult(
                stage="image_text_analysis",
                status="completed",
                detail=(
                    "Image and caption prepared for "
                    "multimodal analysis."
                ),
            )
        )

        stages.append(
            PipelineStageResult(
                stage="cross_modal_alignment",
                status="completed",
                detail=(
                    "CLIP image-caption alignment calculated."
                ),
                score=float(clip_score),
            )
        )

        # -----------------------------------------------------
        # 3. Alignment classifier
        # -----------------------------------------------------

        alignment_result = predict_alignment(
            alignment_score=clip_score
        )

        alignment_prediction = alignment_result["prediction"]
        alignment_confidence = alignment_result["confidence"]

        stages.append(
            PipelineStageResult(
                stage="alignment_classification",
                status="completed",
                detail=(
                    "Alignment classifier prediction: "
                    f"{alignment_prediction}"
                ),
                score=float(alignment_confidence),
            )
        )

        # -----------------------------------------------------
        # 4. Qwen reasoning
        # -----------------------------------------------------

        qwen_response = _run_qwen(
            image_bytes=image_bytes,
            filename=filename,
            caption=caption,
        )

        stages.append(
            PipelineStageResult(
                stage="mllm_reasoning",
                status="completed",
                detail=qwen_response[:1000],
            )
        )

        # -----------------------------------------------------
        # 5. Evidence retrieval
        # -----------------------------------------------------

        evidence_items = retrieve_evidence(
            caption=caption,
            max_results=5,
        )

        evidence_not_required = (
            len(evidence_items) == 1
            and evidence_items[0].get("title")
            == "No external evidence required"
        )

        if evidence_not_required:
            evidence_items = []

            evidence_retrieval_detail = (
                "External evidence retrieval was not required "
                "for this primarily visual caption."
            )

            evidence_retrieval_score = 0.0

        else:
            evidence_retrieval_detail = (
                f"Retrieved {len(evidence_items)} "
                "external evidence items."
            )

            evidence_retrieval_score = float(
                len(evidence_items)
            )

        stages.append(
            PipelineStageResult(
                stage="evidence_retrieval",
                status="completed",
                detail=evidence_retrieval_detail,
                score=evidence_retrieval_score,
            )
        )

        # -----------------------------------------------------
        # 6. Evidence analysis
        # -----------------------------------------------------

        evidence_items = analyze_evidence(
            caption=caption,
            evidence_items=evidence_items,
        )

        if evidence_not_required:
            evidence_analysis_detail = (
                "No external evidence required; "
                "visual consistency was evaluated using "
                "the image and multimodal reasoning."
            )
        else:
            evidence_analysis_detail = (
                "Retrieved evidence analyzed against "
                "the caption claim."
            )

        stages.append(
            PipelineStageResult(
                stage="evidence_analysis",
                status="completed",
                detail=evidence_analysis_detail,
            )
        )

        # -----------------------------------------------------
        # 7. Evidence-aware fusion
        # -----------------------------------------------------

        final_decision = make_final_decision(
            alignment_prediction=alignment_prediction,
            alignment_confidence=alignment_confidence,
            qwen_response=qwen_response,
            evidence_items=evidence_items,
            caption=caption,
        )

        final_prediction = final_decision["prediction"]
        final_confidence = float(
            final_decision["confidence_score"]
        )

        stages.append(
            PipelineStageResult(
                stage="final_decision",
                status="completed",
                detail=(
                    "Final prediction: "
                    f"{final_prediction}"
                ),
                score=final_confidence,
            )
        )

        reason = final_decision["reason"]
        explanation = final_decision["explanation"]

        inconsistency_type = (
            "Evidence-aware multimodal decision"
        )

    # =========================================================
    # FINAL RESULT
    # =========================================================

    unique_id = str(uuid.uuid4())

    resolved_dataset = (
        dataset.strip()
        if dataset
        else "Custom Pair"
    )

    return VerificationResult(
        id=unique_id,

        prediction=final_prediction,

        confidence_score=final_confidence,
        confidenceScore=final_confidence,

        clip_score=clip_score,
        clipScore=clip_score,

        alignment_score=clip_score,
        alignmentScore=clip_score,

        reason=reason,

        evidence=evidence_items,

        explanation=explanation,

        inconsistency_type=inconsistency_type,
        inconsistencyType=inconsistency_type,

        dataset=resolved_dataset,

        ground_truth_label=None,
        groundTruthLabel=None,

        pipeline_stages=stages,

        is_stub=False,
    )