from fastapi import APIRouter
from app.services.clip_service import clip_service

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint for backend status and service identification.
    """
    return {
        "status": "ok",
        "service": "ooc-verify-backend",
        "clip": {
            "loaded": clip_service.is_loaded,
            "model": clip_service.model_name,
            "device": clip_service.device,
            "device_name": clip_service.device_name,
        },
    }
