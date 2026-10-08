import os
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import settings
from app.pipeline.orchestrator import run_verification
from app.schemas.verification import VerificationResult

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
    - Runs real CLIP cross-modal alignment scoring (openai/clip-vit-base-patch32)
      with GPU/CUDA acceleration when available.
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

    # 5. Execute pipeline orchestrator with CLIP alignment
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
