import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.verification import router as verification_router
from app.config import settings

logger = logging.getLogger("uvicorn.default")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mode = os.getenv("OOC_VERIFY_MODE", "local").strip().lower()

    logger.info(
        "Initializing OOC-Verify backend services in %s mode...",
        mode,
    )

    if mode == "remote":
        logger.info(
            "Remote mode enabled: skipping local CLIP and Qwen model loading."
        )
    else:
        from app.services.clip_service import clip_service
        from app.services.qwen_service import load_model as load_qwen_model

        clip_service.load_model()
        load_qwen_model()

    yield

    logger.info("Shutting down OOC-Verify backend services...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multimodal Out-Of-Context Misinformation Verification Research API",
    lifespan=lifespan,
)

# Configure CORS for local React frontend development
# Allows requests from localhost:5173, etc.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if settings.CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes under /api/v1
app.include_router(health_router, prefix=settings.API_V1_PREFIX, tags=["Health"])
app.include_router(
    verification_router, prefix=settings.API_V1_PREFIX, tags=["Verification"]
)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "OOC-Verify Backend API",
        "version": settings.VERSION,
        "docs_url": "/docs",
        "health_url": f"{settings.API_V1_PREFIX}/health",
    }
