from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.verification import router as verification_router
from app.config import settings
from app.services.clip_service import clip_service
from app.services.qwen_service import load_model as load_qwen_model

logger = logging.getLogger("uvicorn.default")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads CLIP and Qwen models once into memory at startup.
    """
    logger.info("Initializing OOC-Verify backend services...")

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
