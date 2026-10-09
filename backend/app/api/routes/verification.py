import mimetypes
import uuid

from app.schemas.verification import (
    PipelineStageResult,
    VerificationResult,
)
from app.services.openrouter_service import analyze_remote_image

import os
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status


from app.config import settings

router = APIRouter()


@router.post(
    "/verify",
    response_model=VerificationResult,
    status_code=status.HTTP_200_OK,
    summary="Execute OOC-Verify pipeline on an image-caption pair",
)
async def verify_claim(
    image: UploadFile = File(..., description="Uploaded image file (JPEG, PNG, WEBP)"),
    caption: str = Form(..., description="Caption or text claim associated with the image"),
    dataset: Optional[str] = Form(None, description="Optional benchmark dataset identifier"),
    configuration: Optional[str] = Form(
        None,
        description="Optional pipeline configuration: alignment_only | mllm_only | proposed",
    ),
):
    """
    Verification endpoint for out-of-context multimodal analysis.

    Validates inputs and executes the pipeline orchestrator:
    - Runs real CLIP cross-modal alignment scoring (openai/clip-vit-base-patch32) with GPU/CUDA acceleration when available.
    - Maintains remaining stages (MLLM reasoning, retrieval, classifier) as stubs.
    """
    # 1. Validate caption
    trimmed_caption = caption.strip() if caption else ""
    if not trimmed_caption:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Caption is required and cannot be empty.",
        )

    # 2. Validate configuration option if provided
    if configuration and configuration.strip():
        config_clean = configuration.strip().lower()
        if config_clean not in settings.SUPPORTED_CONFIGURATIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Invalid configuration '{configuration}'. "
                    f"Supported options: {', '.join(sorted(settings.SUPPORTED_CONFIGURATIONS))}"
                ),
            )
        configuration = config_clean
    else:
        configuration = "proposed"

    # 3. Validate image presence and content type
    if not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image file must have a valid filename.",
        )

    ext = os.path.splitext(image.filename)[1].lower()
    content_type = (image.content_type or "").lower()

    is_valid_type = (
        content_type in settings.ALLOWED_IMAGE_TYPES
        or ext in settings.ALLOWED_EXTENSIONS
    )

    if not is_valid_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported image format: {content_type or ext or 'unknown'}. "
                "Supported formats: JPEG, PNG, WEBP."
            ),
        )

    # 4. Read image contents and enforce size limit
    try:
        contents = await image.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded image file: {str(exc)}",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image file is empty.",
        )

    if len(contents) > settings.MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Image file size ({len(contents) / (1024 * 1024):.2f} MB) "
                f"exceeds the maximum limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
            ),
        )

    # Remote demo mode: use OpenRouter without local CLIP/Qwen.
    if os.getenv("OOC_VERIFY_MODE", "local").strip().lower() == "remote":
        try:
            content_type = (
                image.content_type
                or mimetypes.guess_type(image.filename)[0]
                or "image/jpeg"
            )

            remote_result = analyze_remote_image(
                image_bytes=contents,
                content_type=content_type,
                caption=trimmed_caption,
            )
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid verification input: {exc}",
            ) from None
        except RuntimeError as exc:
            err_msg = str(exc)
            if "OPENROUTER_API_KEY" in err_msg:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Remote verification service is misconfigured (missing OPENROUTER_API_KEY).",
                ) from None
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Remote verification failed: {err_msg}",
            ) from None
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Remote verification failed: {exc}",
            ) from None

        required_keys = {"prediction", "reason", "visual_evidence", "model"}
        if not isinstance(remote_result, dict) or not required_keys.issubset(remote_result.keys()):
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Remote verification failed. The model service returned an invalid result.",
            )

        prediction = remote_result["prediction"]
        reason = remote_result["reason"]
        visual_evidence = remote_result["visual_evidence"]

        stages = [
            PipelineStageResult(
                stage="input_validation",
                status="completed",
                detail="Image and caption validated.",
            ),
            PipelineStageResult(
                stage="remote_multimodal_analysis",
                status="completed",
                detail=(
                    "Image-caption consistency analyzed through "
                    f"remote model: {remote_result['model']}."
                ),
            ),
            PipelineStageResult(
                stage="evidence_retrieval",
                status="skipped",
                detail=(
                    "External evidence retrieval was not performed. "
                    "This result assesses visual consistency only."
                ),
            ),
            PipelineStageResult(
                stage="final_decision",
                status="completed",
                detail=f"Remote model prediction: {prediction}.",
            ),
        ]

        return VerificationResult(
            id=str(uuid.uuid4()),
            prediction=prediction,
            confidence_score=0.0,
            confidenceScore=0.0,
            clip_score=None,
            clipScore=None,
            alignment_score=None,
            alignmentScore=None,
            reason=reason,
            evidence=[],
            explanation=(
                f"Remote multimodal model: {remote_result['model']}. "
                f"Visual evidence reported by the model: {visual_evidence} "
                "External evidence retrieval and the frozen research "
                "fusion pipeline were not executed. The model's prediction "
                "is not a calibrated confidence estimate."
            ),
            inconsistency_type="Remote visual-caption consistency",
            inconsistencyType="Remote visual-caption consistency",
            dataset=dataset.strip() if dataset else "Custom Pair",
            ground_truth_label=None,
            groundTruthLabel=None,
            pipeline_stages=stages,
            is_stub=False,
        )

    # 5. Execute pipeline orchestrator with CLIP alignment
    from app.pipeline.orchestrator import run_verification

    try:
        result = run_verification(
            image_bytes=contents,
            filename=image.filename,
            caption=trimmed_caption,
            dataset=dataset,
            configuration=configuration,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Verification pipeline failed: {str(exc)}",
        )

    return result
