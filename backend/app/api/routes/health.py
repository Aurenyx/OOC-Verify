import os

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint for backend status and service identification."""

    mode = os.getenv("OOC_VERIFY_MODE", "local").strip().lower()

    response = {
        "status": "ok",
        "service": "ooc-verify-backend",
        "mode": mode,
    }

    # Do not import CLIP or PyTorch in remote mode.
    if mode == "remote":
        response["clip"] = {
            "loaded": False,
            "model": "not loaded in remote mode",
            "device": "remote",
            "device_name": "OpenRouter",
        }
    else:
        from app.services.clip_service import clip_service

        response["clip"] = {
            "loaded": clip_service.is_loaded,
            "model": clip_service.model_name,
            "device": clip_service.device,
            "device_name": clip_service.device_name,
        }

    return response

